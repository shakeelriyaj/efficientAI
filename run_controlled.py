"""Controlled comparisons on matched A100 allocations: batches, overhead and KD."""
import argparse
import functools
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from torch import nn
from transformers import AutoTokenizer

from controlled_routes import RoutedBatch, fixed_batch
from data import loader, logits_for, macro_f1, save_json, seed_everything
from exits import new_head
from models import fingerprint, load_model
from run import environment, load_config
from track_controlled import Tracker
from train import train_student


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_assets(path):
    manifest = load_config(path / "manifest.json")
    for name, expected in manifest["sha256"].items():
        if sha256(path / name) != expected:
            raise ValueError(f"Changed asset: {path / name}")
    # No test labels are read while fitting or selecting checkpoints.
    splits = {name: pd.read_json(path / "splits" / f"{name}.jsonl", lines=True)
              for name in ("train", "validation", "calibration")}
    return manifest, splits


def exit_assets(teacher, assets, exit_run, device):
    if exit_run is None:
        heads = nn.ModuleDict({"8": new_head(teacher)})
        heads["8"].load_state_dict(torch.load(assets / "head8.pt", weights_only=True, map_location="cpu"))
        temperatures, policy = {8: 1.2914971148258343}, ((8, .90),)
    else:
        heads = nn.ModuleDict({str(k): new_head(teacher) for k in (2, 4, 8)})
        heads.load_state_dict(torch.load(exit_run / "exit_heads.pt", weights_only=True, map_location="cpu"))
        temperatures = {int(k): v for k, v in load_config(exit_run / "temperatures.json").items()}
        policy = ((4, .92), (8, .92))
    heads.to(device).requires_grad_(False).eval()
    return heads, temperatures, policy


def encoded(frame, size, tokenizer, device, cfg):
    batches = [frame.iloc[i:i + size] for i in range(0, len(frame), size)]
    def encode(batch):
        return dict(tokenizer(batch.text.tolist(), padding=True, truncation=True,
                              max_length=cfg["max_length"], return_tensors="pt").to(device))
    return batches, encode


def evaluate_routes(routes, frame, size, tokenizer, device, cfg):
    batches, encode = encoded(frame, size, tokenizer, device, cfg)
    tokens = [encode(batch) for batch in batches]
    return {name: tuple(np.concatenate(values) for values in zip(*(route(t) for t in tokens)))
            for name, route in routes.items()}


def time_routes(routes, frame, size, tokenizer, device, cfg):
    """Paired wall-clock measurements; W&B is inactive throughout this function."""
    chosen = frame.sample(n=min(cfg["latency_samples"], len(frame)), random_state=cfg["timing_seed"])
    batches, encode = encoded(chosen, size, tokenizer, device, cfg)
    tokens = [encode(batch) for batch in batches]
    expected = {name: [route(t) for t in tokens] for name, route in routes.items()}
    records = []
    for session in range(cfg["timing_sessions"]):
        rng = np.random.default_rng(cfg["timing_seed"] + session)
        for route in routes.values():
            for i in range(cfg["warmup"]):
                route(tokens[i % len(tokens)])
        for mode in ("model", "e2e"):
            for repeat in range(cfg["timing_repeats"]):
                for index in rng.permutation(len(batches)):
                    for name in rng.permutation(list(routes)):
                        torch.cuda.synchronize(device)
                        started = time.perf_counter_ns()
                        inputs = tokens[index] if mode == "model" else encode(batches[index])
                        labels, depths = routes[name](inputs)
                        torch.cuda.synchronize(device)
                        elapsed = (time.perf_counter_ns() - started) / 1e6
                        ref_labels, ref_depths = expected[name][index]
                        if not (np.array_equal(labels, ref_labels) and np.array_equal(depths, ref_depths)):
                            raise RuntimeError(f"Timed prediction/depth mismatch: {name}")
                        records.append({"name": name, "batch_size": size, "batch": int(index),
                                        "session": session, "mode": mode, "repeat": repeat,
                                        "headlines": len(labels), "ms": elapsed,
                                        "row_ids": json.dumps(batches[index].row_id.astype(int).tolist()),
                                        "predictions": json.dumps(labels.tolist()),
                                        "exit_layers": json.dumps(depths.tolist())})
    return pd.DataFrame(records)


def summarize_timing(raw):
    rows = []
    for (size, mode, session, name), part in raw.groupby(["batch_size", "mode", "session", "name"]):
        rows.append({"batch_size": size, "mode": mode, "session": session, "name": name,
                     "mean_ms": part.ms.mean(), "median_ms": part.ms.median(),
                     "p95_ms": part.ms.quantile(.95),
                     "headlines_per_second": 1000 * part.headlines.sum() / part.ms.sum()})
    return pd.DataFrame(rows)


def fit_students(teacher, tokenizer, splits, cfg, out, group, student_source=None):
    full_f1 = macro_f1(splits["validation"].label,
                       logits_for(teacher, loader(splits["validation"], tokenizer, cfg), "cuda").argmax(-1))
    selected = []
    for seed in cfg["seeds"]:
        for objective, weight in (("ce", 1.), ("kd", .75)):
            name = f"{objective}-seed-{seed}"
            directory = (student_source or out) / name
            recipe = {**cfg, "seed": seed, "gold_weight": weight}
            if student_source is not None and not (directory / "selection.json").exists():
                raise FileNotFoundError(f"Missing locked checkpoint: {directory}")
            if (directory / "selection.json").exists():
                if load_config(directory / "config.json") != recipe:
                    raise ValueError(f"Cannot resume a different configuration: {directory}")
            else:
                if directory.exists():
                    raise RuntimeError(f"Incomplete training preserved at {directory}; inspect before resuming")
                directory.mkdir()
                save_json(directory / "config.json", recipe)
                tracker = Tracker(directory, f"{out.name}-{name}", recipe, group, "student-training")
                student = train_student(teacher, tokenizer, splits, recipe, directory,
                                        torch.device("cuda"), on_epoch=tracker.epoch)
                del student
                tracker.finish(directory)
            selection = load_config(directory / "selection.json")
            drop = 100 * (full_f1 - selection["validation_macro_f1"])
            selected.append({"name": name, "objective": objective, "seed": seed,
                             "directory": str(directory), **selection,
                             "validation_f1_drop_pp": drop,
                             "validation_eligible": drop <= cfg["validation_f1_margin_pp"]})
    save_json(out / "locked_checkpoints.json", selected)
    return selected


def run_dataset(name, assets, exit_run, cfg, root, group, student_source=None):
    out = root / name
    out.mkdir(exist_ok=True)
    for filename in ("environment.json", "gpu.csv", "source_sha256.json"):
        shutil.copy2(root / filename, out / filename)
    manifest, splits = read_assets(assets)
    save_json(out / "asset_manifest.json", manifest)
    tokenizer = AutoTokenizer.from_pretrained(assets / "teacher", local_files_only=True)
    teacher = load_model(assets / "teacher", torch.device("cuda"))
    before = fingerprint(teacher)
    cfg = {**cfg, "dataset": name, "teacher_fingerprint": before}
    heads, temperatures, policy = exit_assets(teacher, assets, exit_run, torch.device("cuda"))
    head_before = fingerprint(heads)
    save_json(out / "exit_policy.json", {"thresholds": policy, "temperatures": temperatures,
                                        "head_fingerprint": head_before})
    source = student_source / name if student_source is not None else None
    selected = fit_students(teacher, tokenizer, splits, cfg, out, f"{group}-{name}", source)
    if source is not None:
        save_json(out / "reused_students.json", {"source": str(source), "retrained": False,
                  "checkpoint_sha256": {row["name"]: sha256(Path(row["directory"]) / "student/model.safetensors")
                                        for row in selected}})
    primary = {row["objective"]: row for row in selected if row["seed"] == cfg["primary_seed"]}
    students = {objective: load_model(Path(row["directory"]) / "student", torch.device("cuda"))
                for objective, row in primary.items()}
    routes = {"full12": functools.partial(fixed_batch, teacher),
              "dynamic": RoutedBatch(teacher, heads, temperatures, policy),
              **{objective + "8": functools.partial(fixed_batch, model)
                 for objective, model in students.items()}}
    validation = evaluate_routes(routes, splits["validation"], 16, tokenizer, "cuda", cfg)
    full_validation = macro_f1(splits["validation"].label, validation["full12"][0])
    eligible = {key: 100 * (full_validation - macro_f1(splits["validation"].label, values[0]))
                <= cfg["validation_f1_margin_pp"] for key, values in validation.items()}
    save_json(out / "frozen_validation.json", {"eligible": eligible, "test_used": False})

    test = pd.read_json(assets / "splits/test.jsonl", lines=True)
    full_test = logits_for(teacher, loader(test, tokenizer, cfg), "cuda").argmax(-1).numpy()
    full_f1 = macro_f1(test.label, full_test)
    seed_quality, seed_predictions = [], []
    for row in selected:
        student = load_model(Path(row["directory"]) / "student", torch.device("cuda"))
        predictions = logits_for(student, loader(test, tokenizer, cfg), "cuda").argmax(-1).numpy()
        f1 = macro_f1(test.label, predictions)
        drop = 100 * (full_f1 - f1)
        seed_quality.append({**row, "full_macro_f1": full_f1, "macro_f1": f1,
                             "f1_drop_pp": drop, "within_budget": drop <= cfg["test_f1_budget_pp"]})
        seed_predictions.append(test[["row_id", "label", "group_id"]].assign(
            name=row["name"], prediction=predictions))
        del student
    pd.DataFrame(seed_quality).to_csv(out / "seed_quality.csv", index=False)
    pd.concat(seed_predictions).to_csv(out / "seed_predictions.csv", index=False)

    quality, prediction_frames, timings = [], [], []
    for size in cfg["batch_sizes"]:
        outputs = evaluate_routes(routes, test, size, tokenizer, "cuda", cfg)
        for key, (labels, depths) in outputs.items():
            f1 = macro_f1(test.label, labels)
            drop = 100 * (macro_f1(test.label, outputs["full12"][0]) - f1)
            quality.append({"name": key, "batch_size": size, "macro_f1": f1,
                            "f1_drop_pp": drop, "within_budget": drop <= cfg["test_f1_budget_pp"],
                            "validation_eligible": eligible[key], "mean_layers": depths.mean()})
            prediction_frames.append(test[["row_id", "label", "group_id"]].assign(
                name=key, batch_size=size, prediction=labels, exit_layer=depths))
        print(f"{name}: paired batch-{size} timing", flush=True)
        raw = time_routes(routes, test, size, tokenizer, torch.device("cuda"), cfg)
        raw.to_csv(out / f"timings-batch-{size}.csv", index=False)
        timings.append(raw)
    pd.DataFrame(quality).to_csv(out / "quality.csv", index=False)
    pd.concat(prediction_frames).to_csv(out / "predictions.csv", index=False)
    summarize_timing(pd.concat(timings)).to_csv(out / "timing_summary.csv", index=False)

    overhead = {"full12": routes["full12"], **{
        mode: RoutedBatch(teacher, heads, temperatures, policy, mode)
        for mode in ("loop", "head", "check")}}
    for key, (labels, depths) in evaluate_routes(overhead, test, 1, tokenizer, "cuda", cfg).items():
        if not (np.array_equal(labels, full_test) and np.all(depths == 12)):
            raise AssertionError(f"Overhead ablation changed predictions: {key}")
    raw = time_routes(overhead, test, 1, tokenizer, torch.device("cuda"), cfg)
    raw.to_csv(out / "overhead_timings.csv", index=False)
    summarize_timing(raw).to_csv(out / "overhead_summary.csv", index=False)
    if fingerprint(teacher) != before:
        raise AssertionError("Teacher weights changed")
    if fingerprint(heads) != head_before:
        raise AssertionError("Exit head weights changed")
    save_json(out / "verification.json", {"teacher_unchanged": True,
              "exit_heads_unchanged": True,
              "quality_examples": len(test), "test_previously_used_in_development": True,
              "test_used_for_checkpoint_or_policy_selection": False,
              "timed_predictions_and_depths_checked": True,
              "all_overhead_routes_match_full_predictions": True})
    # Start tracking only after all measured calls have finished.
    tracker = Tracker(out, f"{name}-evaluation", cfg, f"{group}-{name}", "paired-evaluation")
    tracker.run.log({"seed_quality": wandb.Table(dataframe=pd.DataFrame(seed_quality)),
                     "quality": wandb.Table(dataframe=pd.DataFrame(quality)),
                     "timing": wandb.Table(dataframe=summarize_timing(pd.concat(timings)))})
    tracker.finish(out, {"full_macro_f1": full_f1, "teacher_unchanged": True})
    print(f"{name}: complete", flush=True)
    del routes, overhead, students, teacher, heads
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentfin-assets", required=True, type=Path)
    parser.add_argument("--phrasebank-assets", required=True, type=Path)
    parser.add_argument("--phrasebank-exits", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", choices=["sentfin", "phrasebank"], default=["sentfin", "phrasebank"])
    parser.add_argument("--student-source", type=Path, help="Reuse locked checkpoints after an interrupted timing job")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("controlled.json"))
    args = parser.parse_args()
    cfg = {**load_config(Path(__file__).with_name("experiment.json")), **load_config(args.config)}
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != cfg["expected_gpu"]:
        raise RuntimeError(f"This protocol requires exactly {cfg['expected_gpu']}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed_everything(cfg["primary_seed"])
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "config.json").exists() and load_config(args.out / "config.json") != cfg:
        raise ValueError("Output folder belongs to another protocol")
    save_json(args.out / "config.json", cfg)
    save_json(args.out / "environment.json", environment(torch.device("cuda")))
    hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total",
                                        "--format=csv"], text=True)
    (args.out / "gpu.csv").write_text(hardware)
    source = Path(__file__).parent
    save_json(args.out / "source_sha256.json", {p.name: sha256(p) for p in source.glob("*.py")})
    for name, assets, exits in [("sentfin", args.sentfin_assets, None),
                                ("phrasebank", args.phrasebank_assets, args.phrasebank_exits)]:
        if name in args.datasets:
            run_dataset(name, assets, exits, cfg, args.out, args.out.name, args.student_source)
    save_json(args.out / "complete.json", {"datasets": args.datasets,
                                           "same_gpu_allocation": True, "protocol": cfg})


if __name__ == "__main__":
    main()
