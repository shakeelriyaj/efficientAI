import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from summarize_seeds import report, summarize  # noqa: E402

RECORDED_RUN = ROOT / "reports" / "nyu_hpc" / "retrain-22013353"


def copy_run(root, seed):
    folder = root / f"seed-{seed}"
    shutil.copytree(RECORDED_RUN, folder)
    config = json.loads((folder / "config.json").read_text())
    config["seed"] = seed
    (folder / "config.json").write_text(json.dumps(config))
    return folder


def test_summary_reads_recorded_hpc_runs(tmp_path):
    for seed in (4142, 7):
        copy_run(tmp_path, seed)
    runs = summarize(tmp_path)
    assert list(runs.seed) == [7, 4142]
    assert runs.best_epoch.tolist() == [4, 4]
    assert runs.median_latency_reduction_pct.round(1).tolist() == [26.3, 26.3]
    text = report(runs)
    assert "Runs within 1 pp budget:     2 of 2" in text


def test_summary_rejects_duplicate_seeds(tmp_path):
    copy_run(tmp_path, 1)
    shutil.copytree(tmp_path / "seed-1", tmp_path / "again")
    with pytest.raises(SystemExit, match="Duplicate"):
        summarize(tmp_path)
