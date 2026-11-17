"""Select a validation-only PhraseBank policy or replay it on another GPU."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer

from benchmark import benchmark
from data import macro_f1, save_json, seed_everything
from exits import exit_table, new_head, score, simulate
from models import load_model
from run import environment, load_config
from run_exits import evaluate_test, make_route
from tools.build_assets import sha256

# Fixed before inspecting the new validation sweep. Keep the previous exit locations.
THRESHOLDS = (0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 0.999)
EXIT_LAYERS = [4, 8]
POINT_MARGIN_PP, UPPER_MARGIN_PP = 0.25, 0.75
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 5000, 6111626
FULL = {"name": "full12", "kind": "full"}


def split(assets, name, checksums):
    path = assets / "splits" / f"{name}.jsonl"
    if sha256(path) != checksums[f"splits/{name}.jsonl"]:
        raise ValueError(f"Changed split: {name}")
    return pd.read_json(path, lines=True)


def bootstrap_f1(gold, prediction, group_index, weights):
    counts = np.zeros((weights.shape[1], 3, 3))
    np.add.at(counts, (group_index, gold, prediction), 1)
    confusion = (weights @ counts.reshape(-1, 9)).reshape(-1, 3, 3)
    totals = confusion.sum(axis=1) + confusion.sum(axis=2)
    return np.divide(2 * np.diagonal(confusion, axis1=1, axis2=2), totals,
                     out=np.zeros_like(totals), where=totals > 0).mean(axis=1)


def select_policy(teacher, heads, temperatures, validation, tokenizer, cfg, device, out):
    table = exit_table(teacher, heads, temperatures, validation, tokenizer, cfg, device)
    depth = len(teacher.bert.encoder.layer)
    gold = validation.label.to_numpy()
    full, _ = simulate(table, FULL, depth)
    full_f1 = macro_f1(gold, full)
    groups, group_index = np.unique(validation.group_id, return_inverse=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=BOOTSTRAP_DRAWS)
    full_bootstrap = bootstrap_f1(gold, full, group_index, weights)
    records, eligible = [], []
    for threshold in THRESHOLDS:
        policy = {"name": "dynamic_conservative", "kind": "dynamic", "rule": "max_prob",
                  "exit_layers": EXIT_LAYERS, "threshold": threshold}
        pred, depths = simulate(table, policy, depth)
        boot = bootstrap_f1(gold, pred, group_index, weights)
        upper = float(np.quantile(100 * (full_bootstrap - boot), 0.95))
        metrics = score(policy, gold, pred, depths, full_f1, [*EXIT_LAYERS, depth])
        passes = (metrics["f1_drop_pp"] <= POINT_MARGIN_PP and upper <= UPPER_MARGIN_PP
                  and metrics["early_exit_fraction"] > 0)
        records.append({**metrics, "upper_95_one_sided_f1_drop_pp": upper, "eligible": passes})
        if passes:
            eligible.append((metrics["mean_layers"], -threshold, policy, records[-1]))
    pd.DataFrame(records).to_csv(out / "validation_thresholds.csv", index=False)
    best = min(eligible, key=lambda item: item[:2]) if eligible else None
    return {"selected_on": "validation", "criteria": {"threshold_candidates": THRESHOLDS,
             "maximum_observed_f1_drop_pp": POINT_MARGIN_PP, "maximum_bootstrap_upper_pp": UPPER_MARGIN_PP,
             "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
             "ranking": "fewest validation mean layers, then higher threshold"},
            "validation_examples": len(validation), "validation_groups": len(groups),
            "validation_full_macro_f1": full_f1, "policy": best[2] if best else None,
            "validation_result": best[3] if best else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--exit-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frozen-policy", type=Path, help="Replay a policy with matching input hashes")
    parser.add_argument("--skip-timing", action="store_true", help="Run prediction checks only")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.exit_run / "config.json")
    seed_everything(cfg["seed"])
    device = torch.device(args.device)
    manifest = load_config(args.assets / "manifest.json")
    for name, expected in manifest["sha256"].items():
        if name.startswith("teacher/") and sha256(args.assets / name) != expected:
            raise ValueError(f"Changed teacher asset: {name}")
    teacher = load_model(args.assets / "teacher", device)
    tokenizer = AutoTokenizer.from_pretrained(args.assets / "teacher", local_files_only=True)
    heads = nn.ModuleDict({str(layer): new_head(teacher) for layer in cfg["exit_layers"]})
    head_path = args.exit_run / "exit_heads.pt"
    heads.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
    heads.to(device).requires_grad_(False).eval()
    temperatures = {int(k): v for k, v in load_config(args.exit_run / "temperatures.json").items()}
    input_hashes = {"exit_heads_sha256": sha256(head_path),
                    "asset_manifest_sha256": sha256(args.assets / "manifest.json"),
                    "exit_config_sha256": sha256(args.exit_run / "config.json"),
                    "temperatures_sha256": sha256(args.exit_run / "temperatures.json")}
    decision_path = args.out / "frozen_policy.json"
    if args.frozen_policy:
        decision = load_config(args.frozen_policy)
        if decision.get("input_hashes") != input_hashes or decision.get("selected_on") != "validation":
            raise ValueError("Frozen policy was not selected on validation for these exact inputs")
        decision_path.write_bytes(args.frozen_policy.read_bytes())
    else:
        validation = split(args.assets, "validation", manifest["sha256"])
        decision = select_policy(teacher, heads, temperatures, validation, tokenizer, cfg, device, args.out)
        decision["input_hashes"] = input_hashes
        save_json(decision_path, decision)
    decision_hash = sha256(decision_path)
    print("Policy locked before test:", decision["policy"], "SHA256", decision_hash, flush=True)

    # Open the test split only after persisting the policy and all validation evidence.
    frozen = load_config(decision_path)
    policies = [FULL, frozen["policy"]] if frozen["policy"] else [FULL]
    test = split(args.assets, "test", manifest["sha256"])
    quality, predictions, _, mismatches = evaluate_test(policies, teacher, heads, temperatures,
                                                         tokenizer, test, cfg, device)
    if any(any(counts.values()) for counts in mismatches.values()):
        raise RuntimeError(f"Simulated versus actual prediction mismatch: {mismatches}")
    predictions.drop(columns="text").to_csv(args.out / "predictions.csv", index=False)
    print("Test macro-F1 before timing:", quality[["name", "macro_f1", "f1_drop_pp"]].to_dict("records"), flush=True)
    timed = 0
    if not args.skip_timing:
        routes = {policy["name"]: make_route(policy, teacher, heads, temperatures) for policy in policies}
        timing, raw = benchmark(routes, tokenizer, test, cfg, device, predictions.set_index("row_id"))
        raw.to_csv(args.out / "timings.csv", index=False)
        timed = len(raw)
        quality = quality.merge(timing, on="name", validate="one_to_one")
    quality.to_csv(args.out / "metrics.csv", index=False)
    save_json(args.out / "environment.json", environment(device))
    assert sha256(decision_path) == decision_hash
    save_json(args.out / "verification.json", {"simulation_mismatches": mismatches,
              "frozen_policy_sha256": decision_hash, "test_used_for_selection": False,
              "test_previously_evaluated_in_initial_study": True, "quality_examples": len(test),
              "timed_predictions_checked": timed, "checkpoint_retrained": False,
              "replayed_frozen_policy": args.frozen_policy is not None})
    print((args.out / "metrics.csv").read_text(), flush=True)


if __name__ == "__main__":
    main()
