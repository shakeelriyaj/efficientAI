"""Benchmark fixed and batch-size-dependent FinBERT exit policies."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer

from data import load_splits, macro_f1, save_json, seed_everything
from exits import decision, final_logits, new_head, score
from models import load_model
from run import environment, load_config


BATCH_SIZES = (1, 4, 16, 32)
FULL = ()


@torch.inference_mode()
def predict_batch(teacher, heads, temperatures, tokens, policy):
    """Return CPU labels and depths in the original order."""
    depth = len(teacher.bert.encoder.layer)
    if not policy:
        labels = teacher(**tokens).logits.argmax(-1).cpu().numpy()
        return labels, np.full(len(labels), depth)

    bert = teacher.bert
    input_ids = tokens["input_ids"]
    mask = bert.get_extended_attention_mask(tokens["attention_mask"], input_ids.shape)
    hidden = bert.embeddings(input_ids=input_ids, token_type_ids=tokens.get("token_type_ids"))
    active = np.arange(len(input_ids))
    outputs = np.empty((len(active), 2), dtype=np.int64)
    thresholds = dict(policy)

    for layer, block in enumerate(bert.encoder.layer, 1):
        hidden = block(hidden, attention_mask=mask)[0]
        if layer not in thresholds:
            continue
        decisions = decision(heads[str(layer)](hidden), temperatures[layer]).cpu().numpy()
        exiting = decisions[:, 0] >= thresholds[layer]
        outputs[active[exiting], 0] = decisions[exiting, 2]
        outputs[active[exiting], 1] = layer
        active = active[~exiting]
        if not len(active):
            return outputs[:, 0], outputs[:, 1]
        if exiting.any():
            keep = torch.as_tensor(np.flatnonzero(~exiting), device=hidden.device)
            hidden = hidden.index_select(0, keep)
            mask = mask.index_select(0, keep)

    outputs[active, 0] = final_logits(teacher, hidden).argmax(-1).cpu().numpy()
    outputs[active, 1] = depth
    return outputs[:, 0], outputs[:, 1]


def batches_for(frame, size):
    return [frame.iloc[start:start + size] for start in range(0, len(frame), size)]


def encode_batch(batch, tokenizer, device, cfg):
    return dict(tokenizer(batch.text.tolist(), padding=True, truncation=True,
                          max_length=cfg["max_length"], return_tensors="pt").to(device))


def evaluate(route, tokens):
    labels, depths = zip(*(route(batch) for batch in tokens))
    return np.concatenate(labels), np.concatenate(depths)


def time_routes(routes, batches, encode, expected, device, cfg, repeats, samples):
    """Time identical full batches, verifying each timed prediction and depth."""
    rng = np.random.default_rng(cfg["timing_seed"])
    full_batches = [i for i, batch in enumerate(batches) if len(batch) == len(batches[0])]
    count = min(len(full_batches), max(1, samples // len(batches[0])))
    selected = rng.choice(full_batches, count, replace=False)
    for route in routes.values():
        for index in np.resize(selected, cfg["warmup"]):
            route(encode(batches[index]))
    offsets = np.cumsum([0] + [len(batch) for batch in batches])
    records = []
    names = list(routes)
    for repeat in range(repeats):
        for index in rng.permutation(selected):
            batch = batches[index]
            rows = slice(offsets[index], offsets[index + 1])
            for name in rng.permutation(names):
                torch.cuda.synchronize(device)
                started = time.perf_counter_ns()
                labels, depths = routes[name](encode(batch))
                torch.cuda.synchronize(device)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                expected_labels, expected_depths = expected[name]
                if not (np.array_equal(labels, expected_labels[rows]) and
                        np.array_equal(depths, expected_depths[rows])):
                    raise RuntimeError(f"Timed prediction mismatch for {name}, batch {index}")
                records.append({"name": name, "batch": int(index), "repeat": repeat,
                                "headlines": len(batch), "ms": elapsed})
    return pd.DataFrame(records)


def candidate_policies(cfg):
    # A small shared grid, with the adaptive search also allowing a stricter
    # threshold at earlier heads. Exit sets already encode skipped heads.
    for layers in cfg["exit_sets"]:
        for threshold in (0.75, 0.85, 0.90, 0.95, 0.99):
            yield tuple((layer, threshold) for layer in layers)
            if len(layers) > 1 and threshold < 0.99:
                yield tuple((layer, min(0.99, threshold + 0.05) if layer != layers[-1] else threshold)
                            for layer in layers)


def load_scalar_reference(exit_run):
    selection = json.loads((exit_run / "selected_policies.json").read_text())
    policy = selection["selected"].get("max_prob")
    if policy is None:
        return None
    return tuple((layer, policy["threshold"]) for layer in sorted(policy["exit_layers"]))


def load_frozen_policies(path, reference, exit_layers):
    """Read the schedules written by select_policies, including full inference."""
    saved = json.loads(path.read_text())

    def schedule(entries):
        policy = tuple((layer, threshold) for layer, threshold in entries)
        layers = [layer for layer, _ in policy]
        if layers != sorted(set(layers)) or any(
            layer not in exit_layers or not 0 <= threshold <= 1
            for layer, threshold in policy
        ):
            raise ValueError(f"Invalid frozen exit schedule: {policy}")
        return policy

    if saved["scalar_reference"] != (
        None if reference is None else [list(entry) for entry in reference]
    ):
        raise ValueError("Frozen policies and --exit-run use different reference policies")
    if set(saved["adaptive"]) != {str(size) for size in BATCH_SIZES}:
        raise ValueError(f"Frozen policies must cover batch sizes {BATCH_SIZES}")
    fixed = None if saved["fixed"] is None else schedule(saved["fixed"])
    shared = schedule(saved["shared"])
    adaptive = {size: schedule(saved["adaptive"][str(size)]) for size in BATCH_SIZES}
    return fixed, shared, adaptive


def select_policies(make_route, reference, frame, encode, device, cfg, out):
    policies = list(dict.fromkeys([FULL, *candidate_policies(cfg)]))
    if reference is not None and reference not in policies:
        policies.append(reference)
    routes = {str(i): make_route(policy) for i, policy in enumerate(policies)}
    rows, timings = [], []
    for size in BATCH_SIZES:
        batches = batches_for(frame, size)
        tokens = [encode(batch) for batch in batches]
        expected = {name: evaluate(route, tokens) for name, route in routes.items()}
        baseline = macro_f1(frame.label, expected["0"][0])
        drops = {name: baseline - macro_f1(frame.label, labels)
                 for name, (labels, _) in expected.items()}
        eligible = {name: routes[name] for name, drop in drops.items()
                    if drop <= cfg["validation_f1_drop_limit"]}
        raw = time_routes(eligible, batches, encode, expected, device, cfg,
                          cfg["timing_repeats"], cfg["latency_samples"])
        timings.append(raw.assign(batch_size=size))
        medians = raw.groupby("name").ms.median()
        for name, drop in drops.items():
            median = medians.get(name, np.nan)
            rows.append({"batch_size": size, "candidate": name, "f1_drop_pp": 100 * drop,
                         "p50_ms": median, "relative_p50": median / medians["0"],
                         "fixed_threshold": len({t for _, t in policies[int(name)]}) == 1})
    results = pd.DataFrame(rows)
    results.to_csv(out / "validation_policies.csv", index=False)
    pd.concat(timings).to_csv(out / "validation_timings.csv", index=False)
    # A shared policy must meet the quality limit at every batch size.
    eligible_shared = results.groupby("candidate").relative_p50.agg(["count", "mean"])
    eligible_shared = eligible_shared[eligible_shared["count"] == len(BATCH_SIZES)]["mean"]
    shared = policies[int(eligible_shared.idxmin())]
    uniform_candidates = results.loc[results.fixed_threshold, "candidate"].unique()
    eligible_fixed = eligible_shared[eligible_shared.index.isin(uniform_candidates)]
    fixed = policies[int(eligible_fixed.idxmin())] if len(eligible_fixed) else None
    adaptive = results.loc[results.groupby("batch_size").p50_ms.idxmin()].set_index("batch_size").candidate
    adaptive = {size: policies[int(adaptive[size])] for size in BATCH_SIZES}
    save_json(out / "selected_policies.json", {
        "candidates": policies,
        "scalar_reference": reference,
        "fixed": fixed,
        "shared": shared,
        "adaptive": adaptive,
    })
    return fixed, shared, adaptive


def test_policies(make_route, reference, fixed, shared, adaptive, frame, encode, device, cfg, out, depth):
    metrics, timings, predictions = [], [], []
    layers = [*cfg["exit_layers"], depth]
    for size in BATCH_SIZES:
        routes = {"full": make_route(FULL), "shared": make_route(shared),
                  "adaptive": make_route(adaptive[size])}
        if fixed is not None:
            routes["fixed"] = make_route(fixed)
        if reference is not None:
            routes["scalar_reference"] = make_route(reference)
        batches = batches_for(frame, size)
        tokens = [encode(batch) for batch in batches]
        expected = {name: evaluate(route, tokens) for name, route in routes.items()}
        raw = time_routes(routes, batches, encode, expected, device, cfg,
                          cfg["timing_repeats"], cfg["latency_samples"])
        timings.append(raw.assign(batch_size=size))
        full_f1 = macro_f1(frame.label, expected["full"][0])
        for name, (labels, depths) in expected.items():
            selected = raw[raw.name == name]
            row = score({"name": name, "kind": name}, frame.label, labels, depths, full_f1, layers)
            row.update(batch_size=size, p50_ms=float(selected.ms.median()),
                       p95_ms=float(selected.ms.quantile(0.95)),
                       headlines_per_second=1000 * selected.headlines.sum() / selected.ms.sum(),
                       within_budget=row["f1_drop_pp"] <= 100 * cfg["f1_drop_budget"])
            metrics.append(row)
            predictions.append(frame[["row_id", "label"]].assign(
                batch_size=size, name=name, prediction=labels, exit_layer=depths))
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.concat(timings).to_csv(out / "timings.csv", index=False)
    pd.concat(predictions).to_csv(out / "predictions.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--exit-run", type=Path, required=True,
                        help="Existing run_exits.py output containing trained heads and temperatures")
    parser.add_argument("--frozen-policies", type=Path,
                        help="Replay selected_policies.json from a prior batch benchmark")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("The latency study requires CUDA.")
    cfg = load_config(args.exit_run / "config.json")
    reference = load_scalar_reference(args.exit_run)
    frozen = (load_frozen_policies(args.frozen_policies, reference, cfg["exit_layers"])
              if args.frozen_policies is not None else None)
    seed_everything(cfg["seed"])
    device = torch.device("cuda")
    splits = load_splits(args.assets)
    tokenizer = AutoTokenizer.from_pretrained(args.assets / "teacher", local_files_only=True)
    teacher = load_model(args.assets / "teacher", device)
    heads = nn.ModuleDict({str(layer): new_head(teacher) for layer in cfg["exit_layers"]})
    heads.load_state_dict(torch.load(args.exit_run / "exit_heads.pt", map_location="cpu", weights_only=True))
    heads.to(device).requires_grad_(False).eval()
    temperatures = {int(layer): value for layer, value in
                    json.loads((args.exit_run / "temperatures.json").read_text()).items()}

    def make_route(policy):
        return lambda tokens: predict_batch(teacher, heads, temperatures, tokens, policy)

    def encode(batch):
        return encode_batch(batch, tokenizer, device, cfg)

    args.out.mkdir(parents=True, exist_ok=False)
    save_json(args.out / "environment.json", environment(device))
    save_json(args.out / "config.json", {"exit_config": cfg, "batch_sizes": BATCH_SIZES,
                                        "exit_run": str(args.exit_run.resolve())})
    if frozen is None:
        fixed, shared, adaptive = select_policies(
            make_route, reference, splits["validation"], encode, device, cfg, args.out)
    else:
        fixed, shared, adaptive = frozen
        save_json(args.out / "selected_policies.json", {
            "replayed_from": str(args.frozen_policies.resolve()),
            "scalar_reference": reference,
            "fixed": fixed,
            "shared": shared,
            "adaptive": adaptive,
        })
    test_policies(make_route, reference, fixed, shared, adaptive, splits["test"], encode,
                  device, cfg, args.out, len(teacher.bert.encoder.layer))


if __name__ == "__main__":
    main()
