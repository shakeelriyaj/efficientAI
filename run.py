"""Two commands: replay saved models, or retrain the selected compact student."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer

from benchmark import benchmark, encode
from data import load_splits, loader, logits_for, macro_f1, save_json, seed_everything
from models import DynamicExit, fingerprint, load_model, predict_fixed
from train import train_student


def evaluate(teacher, student, dynamic, tokenizer, frame, cfg, device):
    result = frame[["row_id", "label", "group_id"]].copy()
    for name, model in (("full12", teacher), ("compact8_student", student)):
        result[name + "_prediction"] = logits_for(model, loader(frame, tokenizer, cfg), device).argmax(-1).numpy()
        result[name + "_exit_layer"] = len(model.bert.encoder.layer)
    dynamic_outputs = [dynamic.predict(encode(text, tokenizer, device, cfg["max_length"])) for text in frame.text]
    result["dynamic8_quality_prediction"] = [label for label, _ in dynamic_outputs]
    result["dynamic8_quality_exit_layer"] = [depth for _, depth in dynamic_outputs]
    rows = [{"name": name, "macro_f1": macro_f1(frame.label, result[name + "_prediction"]),
             "mean_layers": float(result[name + "_exit_layer"].mean())}
            for name in ("full12", "dynamic8_quality", "compact8_student")]
    quality = pd.DataFrame(rows)
    quality["f1_drop_pp"] = 100 * (rows[0]["macro_f1"] - quality.macro_f1)
    quality["within_budget"] = quality.f1_drop_pp <= 100 * cfg["f1_drop_budget"]
    return quality, result


def environment(device):
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "cuda": torch.version.cuda, "threads": torch.get_num_threads(), "tf32": False, "inference": "FP32 eager",
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}


def load_config(path, seed=None):
    """Read experiment settings; an explicit seed replaces only the training seed."""
    cfg = json.loads(Path(path).read_text())
    if seed is not None:
        cfg["seed"] = seed
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "train"])
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment.json"))
    parser.add_argument("--seed", type=int, help="Override the training seed from the config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-timing", action="store_true", help="Quality-only debugging; not a speed result")
    args = parser.parse_args()
    cfg = load_config(args.config, args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    save_json(args.out / "config.json", cfg)
    seed_everything(cfg["seed"])
    device = torch.device(args.device)
    splits = load_splits(args.assets)
    tokenizer = AutoTokenizer.from_pretrained(args.assets / "teacher", local_files_only=True)
    teacher = load_model(args.assets / "teacher", device)
    dynamic = DynamicExit(teacher, args.assets / "head8.pt", cfg)
    teacher_before = fingerprint(teacher)
    student = (train_student(teacher, tokenizer, splits, cfg, args.out, device)
               if args.command == "train" else load_model(args.assets / "student", device))
    quality, predictions = evaluate(teacher, student, dynamic, tokenizer, splits["test"], cfg, device)
    predictions.to_csv(args.out / "predictions.csv", index=False)
    reference = pd.read_csv(args.assets / "colab_predictions.csv").set_index("row_id")
    aligned = predictions.set_index("row_id").loc[reference.index]
    mismatches = {name: {"labels": int((aligned[name + "_prediction"] != reference[name + "_prediction"]).sum()),
                         "depths": int((aligned[name + "_exit_layer"] != reference[name + "_exit_layer"]).sum())}
                  for name in ("full12", "dynamic8_quality", "compact8_student")}
    save_json(args.out / "reference_comparison.json", mismatches)
    if not args.skip_timing:
        routes = {"full12": lambda tokens: predict_fixed(teacher, tokens),
                  "dynamic8_quality": dynamic.predict,
                  "compact8_student": lambda tokens: predict_fixed(student, tokens)}
        timing, raw = benchmark(routes, tokenizer, splits["test"], cfg, device,
                                predictions.set_index("row_id"))
        raw.to_csv(args.out / "timings.csv", index=False)
        quality = quality.merge(timing, on="name", validate="one_to_one")
    quality.to_csv(args.out / "metrics.csv", index=False)
    teacher_after = fingerprint(teacher)
    assert teacher_after == teacher_before
    unchanged_routes = mismatches if args.command == "verify" else {key: mismatches[key] for key in ("full12", "dynamic8_quality")}
    exact = all(not counts["labels"] and not counts["depths"] for counts in unchanged_routes.values())
    save_json(args.out / "verification.json", {"command": args.command, "saved_routes_match_colab": exact,
              "all_teacher_weights_unchanged": True, "student_backbone_frozen_during_training": False,
              "test_used_for_selection": False, "test_set_reused": True,
              "quality_examples": len(predictions), "timed_predictions_checked": 0 if args.skip_timing else len(raw),
              "student_quality_budget_met": bool(quality.loc[quality.name == "compact8_student", "within_budget"].iloc[0])})
    save_json(args.out / "environment.json", environment(device))
    print(quality.to_string(index=False), flush=True)
    print("Differences from Colab:", mismatches, flush=True)
    if not exact:
        raise RuntimeError("Saved checkpoint routes differ from Colab; inspect reference_comparison.json")


if __name__ == "__main__":
    main()
