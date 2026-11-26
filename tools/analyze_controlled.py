"""Independently recompute quality and summarize the saved paired measurements."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def f1(gold, predicted):
    return f1_score(gold, predicted, labels=[0, 1, 2], average="macro", zero_division=0)


def verify_timed(raw, predictions, overhead=False):
    for row in raw.drop_duplicates(["batch_size", "batch", "name"]).itertuples():
        name = "full12" if overhead else row.name
        reference = predictions.query("batch_size == @row.batch_size and name == @name").set_index("row_id")
        expected = reference.loc[json.loads(row.row_ids)]
        np.testing.assert_array_equal(json.loads(row.predictions), expected.prediction)
        np.testing.assert_array_equal(json.loads(row.exit_layers), expected.exit_layer)


def analyze(folder):
    quality = pd.read_csv(folder / "quality.csv")
    predictions = pd.read_csv(folder / "predictions.csv")
    for row in quality.itertuples():
        part = predictions.query("name == @row.name and batch_size == @row.batch_size")
        assert not part.row_id.duplicated().any()
        assert abs(f1(part.label, part.prediction) - row.macro_f1) < 1e-12
        assert abs(part.exit_layer.mean() - row.mean_layers) < 1e-12
    seed_scores = pd.read_csv(folder / "seed_quality.csv")
    seed_predictions = pd.read_csv(folder / "seed_predictions.csv")
    for row in seed_scores.itertuples():
        part = seed_predictions.query("name == @row.name")
        assert abs(f1(part.label, part.prediction) - row.macro_f1) < 1e-12

    raw = pd.concat(pd.read_csv(path) for path in sorted(folder.glob("timings-batch-*.csv")))
    keys = ["batch_size", "mode", "session", "repeat", "batch", "name"]
    assert not raw.duplicated(keys).any()
    repeated = raw.groupby(["batch_size", "batch", "name"])
    assert repeated.size().eq(30).all()  # three sessions, five repeats, two timing modes
    assert repeated[["row_ids", "predictions", "exit_layers"]].nunique().eq(1).all().all()
    assert raw.groupby(["batch_size", "batch"]).row_ids.nunique().eq(1).all()
    verify_timed(raw, predictions)
    sessions = []
    for (size, mode, session), part in raw.groupby(["batch_size", "mode", "session"]):
        stats = part.groupby("name").ms.agg(mean_ms="mean", median_ms="median", p95_ms=lambda x: x.quantile(.95))
        for name, values in stats.iterrows():
            sessions.append({"batch_size": int(size), "mode": mode, "session": int(session), "name": name,
                             **values.to_dict(),
                             "headlines_per_second": 1000 * part[part.name == name].headlines.sum() / part[part.name == name].ms.sum(),
                             "mean_change_pct": 100 * (values.mean_ms / stats.loc["full12", "mean_ms"] - 1),
                             "median_change_pct": 100 * (values.median_ms / stats.loc["full12", "median_ms"] - 1),
                             "p95_change_pct": 100 * (values.p95_ms / stats.loc["full12", "p95_ms"] - 1)})
    sessions = pd.DataFrame(sessions)
    sessions.to_csv(folder / "verified_timing_sessions.csv", index=False)
    summary = sessions.groupby(["batch_size", "mode", "name"]).agg(
        median_ms=("median_ms", "median"), median_change_pct=("median_change_pct", "median"),
        median_change_min=("median_change_pct", "min"), median_change_max=("median_change_pct", "max"),
        mean_ms=("mean_ms", "median"), p95_ms=("p95_ms", "median"),
        mean_change_pct=("mean_change_pct", "median"),
        p95_change_pct=("p95_change_pct", "median"),
        headlines_per_second=("headlines_per_second", "median")).reset_index()
    summary.to_csv(folder / "verified_timing_summary.csv", index=False)

    overhead = pd.read_csv(folder / "overhead_timings.csv")
    verify_timed(overhead, predictions, overhead=True)
    paired = overhead.pivot(index=["session", "mode", "repeat", "batch"], columns="name", values="ms")
    deltas = pd.DataFrame({"manual_loop": paired.loop-paired.full12,
                           "head_computation": paired["head"]-paired.loop,
                           "host_check": paired.check-paired["head"],
                           "total_check": paired.check-paired.full12})
    overhead_summary = deltas.groupby("mode").median().to_dict("index")
    seed_summary = seed_scores.groupby("objective").agg(
        mean_f1=("macro_f1", "mean"), std_f1=("macro_f1", "std"),
        mean_drop_pp=("f1_drop_pp", "mean"), worst_drop_pp=("f1_drop_pp", "max"),
        budget_passes=("within_budget", "sum"), validation_passes=("validation_eligible", "sum")).to_dict("index")
    # Paired bootstrap resamples duplicate groups together, using the primary batch-one predictions.
    primary = predictions.query("batch_size == 1").pivot(index=["row_id", "label", "group_id"], columns="name", values="prediction")
    gold = primary.index.get_level_values("label").to_numpy()
    groups = primary.index.get_level_values("group_id").to_numpy()
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(10143)
    drops = {name: [] for name in primary.columns if name != "full12"}
    for _ in range(2000):
        take = np.concatenate([members[i] for i in rng.integers(len(unique), size=len(unique))])
        full = f1(gold[take], primary.full12.to_numpy()[take])
        for name in drops:
            drops[name].append(100 * (full - f1(gold[take], primary[name].to_numpy()[take])))
    intervals = {name: np.quantile(values, [.025, .975]).tolist() for name, values in drops.items()}
    single = raw.query("batch_size == 1")
    dynamic = single.query("name == 'dynamic'").drop_duplicates("batch")
    fallback_batches = dynamic[dynamic.exit_layers.map(lambda s: json.loads(s)[0] == 12)].batch
    paired_dynamic = single.pivot(index=["session", "mode", "repeat", "batch"], columns="name", values="ms")
    fallback = paired_dynamic[paired_dynamic.index.get_level_values("batch").isin(fallback_batches)]
    dynamic_summary = {
        "test_early_fraction": float(predictions.query("batch_size == 1 and name == 'dynamic'").exit_layer.lt(12).mean()),
        "timing_fallback_fraction": len(fallback_batches) / len(dynamic),
        "fallback_paired_median_ms": (fallback.dynamic - fallback.full12).groupby("mode").median().to_dict()}
    result = {"quality_recomputed": True, "timed_calls": len(raw), "overhead_calls": len(overhead),
              "training_seeds": seed_summary, "paired_overhead_median_ms": overhead_summary,
              "f1_drop_group_bootstrap_95_pp": intervals, "dynamic_routing": dynamic_summary}
    (folder / "independent_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(folder.name, json.dumps(result, indent=2))
    print(quality.query("batch_size == 1").to_string(index=False))
    print(summary.query("mode == 'e2e'").to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    assert (args.root / "complete.json").exists(), "Wait for both datasets to finish."
    for dataset in ("sentfin", "phrasebank"):
        analyze(args.root / dataset)
