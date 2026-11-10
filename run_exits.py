"""Multi-exit study: train heads at layers 2/4/8, choose exit policies on validation, test once.

    python run_exits.py --assets assets --out results/exits
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from benchmark import benchmark, encode
from data import load_splits, macro_f1, save_json, seed_everything
from exits import (MultiExit, StaticExit, calibrate, exit_table, finalists, policy_grid, score,
                   simulate, sweep, train_heads)
from models import fingerprint, load_model, predict_fixed
from run import environment, load_config


def make_route(policy, teacher, heads, temperatures):
    if policy["kind"] == "full":
        return lambda tokens: predict_fixed(teacher, tokens)
    if policy["kind"] == "static":
        return StaticExit(teacher, heads[str(policy["layer"])], policy["layer"]).predict
    return MultiExit(teacher, heads, temperatures, policy["exit_layers"], policy["rule"], policy["threshold"]).predict


def expected_outputs(table, policies, depth):
    """Simulated labels/depths in the column format the benchmark checks against."""
    columns = {}
    for policy in policies:
        columns[policy["name"] + "_prediction"], columns[policy["name"] + "_exit_layer"] = simulate(table, policy, depth)
    return pd.DataFrame(columns, index=table.index)


def select(validation_sweep, validation_table, candidates, teacher, heads, temperatures, tokenizer, splits, cfg,
           device, timing, out):
    """One policy per rule, using validation data only."""
    full = {"name": "full12", "kind": "full"}
    depth = len(teacher.bert.encoder.layer)
    ranked = validation_sweep[validation_sweep.name.isin([policy["name"] for policy in candidates])]
    key = "mean_layers"
    if timing and candidates:
        routes = {policy["name"]: make_route(policy, teacher, heads, temperatures) for policy in [full, *candidates]}
        timing_cfg = {**cfg, "latency_samples": cfg["validation_latency_samples"],
                      "timing_repeats": cfg["validation_timing_repeats"]}
        latency, _ = benchmark(routes, tokenizer, splits["validation"], timing_cfg, device,
                               expected_outputs(validation_table, [full, *candidates], depth))
        ranked = validation_sweep.merge(latency, on="name", validate="one_to_one")
        ranked.to_csv(out / "validation_timing.csv", index=False)
        ranked = ranked[ranked.kind == "dynamic"]
        key = "e2e_p50_ms"
    by_name = {policy["name"]: policy for policy in candidates}
    chosen = {}
    for rule in cfg["rules"]:
        rows = ranked[ranked.rule == rule].sort_values([key, "macro_f1"], ascending=[True, False])
        if len(rows):
            chosen[rule] = by_name[rows.name.iloc[0]]
    save_json(out / "selected_policies.json", {
        "selected_on": "validation", "ranked_by": key,
        "validation_f1_drop_limit_pp": 100 * cfg["validation_f1_drop_limit"],
        "finalists": [policy["name"] for policy in candidates], "selected": chosen,
        "rules_without_an_eligible_policy": [rule for rule in cfg["rules"] if rule not in chosen]})
    return chosen


def evaluate_test(policies, teacher, heads, temperatures, tokenizer, frame, cfg, device):
    """Real batch-one inference for every route, compared with the exact simulation."""
    depth = len(teacher.bert.encoder.layer)
    layers = [*cfg["exit_layers"], depth]
    table = exit_table(teacher, heads, temperatures, frame, tokenizer, cfg, device)
    predictions = frame[["row_id", "label", "group_id", "text"]].copy()
    mismatches = {}
    tokens = [encode(text, tokenizer, device, cfg["max_length"]) for text in frame.text]
    for policy in policies:
        route = make_route(policy, teacher, heads, temperatures)
        labels, depths = map(np.array, zip(*(route(item) for item in tokens)))
        expected_labels, expected_depths = simulate(table, policy, depth)
        mismatches[policy["name"]] = {"labels": int((labels != expected_labels).sum()),
                                      "depths": int((depths != expected_depths).sum())}
        predictions[policy["name"] + "_prediction"] = labels
        predictions[policy["name"] + "_exit_layer"] = depths
    full_f1 = macro_f1(frame.label, predictions["full12_prediction"])
    quality = pd.DataFrame([score(policy, frame.label.to_numpy(), predictions[policy["name"] + "_prediction"].to_numpy(),
                                  predictions[policy["name"] + "_exit_layer"].to_numpy(), full_f1, layers)
                            for policy in policies])
    quality["within_budget"] = quality.f1_drop_pp <= 100 * cfg["f1_drop_budget"]
    return quality, predictions, table, mismatches


def study(teacher, tokenizer, splits, cfg, out, device, timing=True):
    seed_everything(cfg["seed"])
    teacher_before = fingerprint(teacher)
    depth = len(teacher.bert.encoder.layer)
    layers = [*cfg["exit_layers"], depth]

    heads, history, epochs = train_heads(teacher, tokenizer, splits, cfg, device)
    torch.save(heads.state_dict(), out / "exit_heads.pt")
    save_json(out / "head_history.json", history)
    save_json(out / "head_selection.json", {"best_epoch_by_layer": {str(k): v for k, v in epochs.items()},
                                            "test_used_for_selection": False})
    temperatures, calibration = calibrate(heads, teacher, tokenizer, splits["calibration"], cfg, device)
    save_json(out / "temperatures.json", {str(layer): value for layer, value in temperatures.items()})
    calibration.to_csv(out / "calibration.csv", index=False)

    grid = policy_grid(cfg)
    save_json(out / "policy_grid.json", grid)
    validation_table = exit_table(teacher, heads, temperatures, splits["validation"], tokenizer, cfg, device)
    validation_sweep = sweep(validation_table, splits["validation"].label.to_numpy(), grid, layers)
    validation_sweep.to_csv(out / "validation_sweep.csv", index=False)
    candidates = finalists(validation_sweep, grid, cfg["validation_f1_drop_limit"])
    chosen = select(validation_sweep, validation_table, candidates, teacher, heads, temperatures, tokenizer, splits, cfg,
                    device, timing, out)

    test_policies = [grid[0], *(policy for policy in grid if policy["kind"] == "static"),
                     *({**policy, "name": f"dynamic_{rule}"} for rule, policy in chosen.items())]
    quality, predictions, table, mismatches = evaluate_test(test_policies, teacher, heads, temperatures,
                                                            tokenizer, splits["test"], cfg, device)
    predictions.to_csv(out / "predictions.csv", index=False)
    table.to_csv(out / "test_exit_scores.csv", index_label="row_id")
    timed = 0
    if timing:
        routes = {policy["name"]: make_route(policy, teacher, heads, temperatures) for policy in test_policies}
        latency, raw = benchmark(routes, tokenizer, splits["test"], cfg, device, predictions.set_index("row_id"))
        raw.to_csv(out / "timings.csv", index=False)
        quality = quality.merge(latency, on="name", validate="one_to_one")
        full_p50 = quality.loc[quality.name == "full12", "e2e_p50_ms"].iloc[0]
        quality["e2e_p50_speedup"] = full_p50 / quality.e2e_p50_ms
        timed = len(raw)
    quality.to_csv(out / "metrics.csv", index=False)
    assert fingerprint(teacher) == teacher_before, "The frozen teacher changed"
    exact = all(not counts["labels"] and not counts["depths"] for counts in mismatches.values())
    save_json(out / "verification.json", {
        "simulation_matches_real_inference": exact, "simulation_mismatches": mismatches,
        "all_teacher_weights_unchanged": True, "test_used_for_selection": False, "test_set_reused": True,
        "quality_examples": len(predictions), "timed_predictions_checked": timed})
    if not exact:
        raise RuntimeError("Simulated policies differ from real inference; inspect verification.json")
    return quality


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("exits.json"))
    parser.add_argument("--seed", type=int, help="Override the head-training seed from the config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-timing", action="store_true",
                        help="Quality-only debugging: policies are picked by mean layers, not latency")
    args = parser.parse_args()
    cfg = load_config(args.config, args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    save_json(args.out / "config.json", cfg)
    seed_everything(cfg["seed"])
    device = torch.device(args.device)
    splits = load_splits(args.assets)
    tokenizer = AutoTokenizer.from_pretrained(args.assets / "teacher", local_files_only=True)
    teacher = load_model(args.assets / "teacher", device)
    quality = study(teacher, tokenizer, splits, cfg, args.out, device, timing=not args.skip_timing)
    save_json(args.out / "environment.json", environment(device))
    columns = [c for c in ("name", "macro_f1", "f1_drop_pp", "mean_layers", "early_exit_fraction",
                           "e2e_p50_ms", "e2e_p50_speedup", "within_budget") if c in quality]
    print(quality[columns].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
