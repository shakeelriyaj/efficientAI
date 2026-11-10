"""Combine several `run.py train --seed N` result folders into one table.

Usage: python tools/summarize_seeds.py results/seeds-<array job id>
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def read_run(folder):
    """One row per training run: its seed, selection, quality, and speed."""
    config = json.loads((folder / "config.json").read_text())
    selection = json.loads((folder / "selection.json").read_text())
    verification = json.loads((folder / "verification.json").read_text())
    metrics = pd.read_csv(folder / "metrics.csv").set_index("name")
    full, student = metrics.loc["full12"], metrics.loc["compact8_student"]
    row = {"seed": config["seed"], "folder": folder.name,
           "best_epoch": selection["best_epoch"],
           "validation_macro_f1": selection["validation_macro_f1"],
           "full12_macro_f1": full.macro_f1,
           "student_macro_f1": student.macro_f1,
           "student_f1_drop_pp": student.f1_drop_pp,
           "within_budget": bool(student.within_budget),
           "saved_routes_match_colab": verification["saved_routes_match_colab"],
           "timed_predictions_checked": verification["timed_predictions_checked"]}
    if "e2e_p50_ms" in metrics:
        row["student_e2e_p50_ms"] = student.e2e_p50_ms
        row["full12_e2e_p50_ms"] = full.e2e_p50_ms
        row["median_latency_reduction_pct"] = 100 * (1 - student.e2e_p50_ms / full.e2e_p50_ms)
    return row


def summarize(root):
    folders = sorted(path.parent for path in Path(root).glob("*/selection.json"))
    if not folders:
        raise SystemExit(f"No training result folders found under {root}")
    runs = pd.DataFrame([read_run(folder) for folder in folders]).sort_values("seed")
    if runs.seed.duplicated().any():
        raise SystemExit(f"Duplicate seeds: {sorted(runs.seed[runs.seed.duplicated()])}")
    if not runs.saved_routes_match_colab.all():
        raise SystemExit("A run's teacher or dynamic route differs from Colab; inspect its folder")
    if runs.full12_macro_f1.nunique() != 1:
        raise SystemExit("The frozen full model scored differently across runs")
    return runs


def report(runs):
    def spread(column, digits=4):
        values = runs[column]
        sd = values.std(ddof=1) if len(values) > 1 else float("nan")
        return (f"mean {values.mean():.{digits}f}, sd {sd:.{digits}f}, "
                f"min {values.min():.{digits}f}, max {values.max():.{digits}f}")

    lines = [f"Seeds ({len(runs)}): {', '.join(map(str, runs.seed))}",
             f"Full 12-layer macro-F1 (frozen): {runs.full12_macro_f1.iloc[0]:.6f}",
             f"Student test macro-F1:       {spread('student_macro_f1')}",
             f"Student F1 drop (pp):        {spread('student_f1_drop_pp', 3)}",
             f"Worst-seed F1 drop (pp):     {runs.student_f1_drop_pp.max():.3f}",
             f"Runs within 1 pp budget:     {runs.within_budget.sum()} of {len(runs)}",
             f"Selected epochs:             {', '.join(map(str, runs.best_epoch))}"]
    if "median_latency_reduction_pct" in runs and runs.median_latency_reduction_pct.notna().all():
        lines.append(f"Median latency reduction (%): {spread('median_latency_reduction_pct', 1)}")
    lines.append("Seed spread is training variation only; the test set was reused in development.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="Folder containing one subfolder per seed")
    args = parser.parse_args()
    runs = summarize(args.root)
    runs.to_csv(args.root / "seed_summary.csv", index=False)
    text = report(runs)
    (args.root / "seed_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
