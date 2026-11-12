import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from exits import MultiExit
from run_batch_exits import FULL, load_frozen_policies, predict_batch
from test_exits import tiny_teacher, tokenizer, varied_heads


def test_batch_size_one_matches_scalar_route():
    teacher = tiny_teacher()
    heads = varied_heads(teacher, [1, 2])
    temperatures = {1: 1.0, 2: 1.0}
    tokens = {"input_ids": torch.tensor([[2, 5, 3]]),
              "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    scalar = MultiExit(teacher, heads, temperatures, [1, 2], "max_prob", 0.75)
    labels, depths = predict_batch(teacher, heads, temperatures, tokens, ((1, 0.75), (2, 0.75)))
    assert (labels.item(), depths.item()) == scalar.predict(tokens)


def test_mixed_batch_preserves_order_and_compacts(monkeypatch):
    teacher = tiny_teacher()
    heads = varied_heads(teacher, [1])
    tokens = {"input_ids": torch.tensor([[2, 5, 3, 0], [2, 6, 7, 3], [2, 8, 3, 0]]),
              "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0]])}
    full_labels, _ = predict_batch(teacher, heads, {1: 1.0}, tokens, FULL)
    monkeypatch.setattr("run_batch_exits.decision", lambda *_: torch.tensor([
        [0.4, 0.0, 0], [0.99, 0.0, 2], [0.4, 0.0, 0],
    ]))
    sizes = []
    with_hook = teacher.bert.encoder.layer[1].register_forward_pre_hook(
        lambda _module, args: sizes.append(len(args[0])))
    try:
        labels, depths = predict_batch(teacher, heads, {1: 1.0}, tokens, ((1, 0.75),))
    finally:
        with_hook.remove()
    np.testing.assert_array_equal(labels, [full_labels[0], 2, full_labels[2]])
    np.testing.assert_array_equal(depths, [4, 1, 4])
    assert sizes == [2]


def test_all_rows_exit_before_next_block():
    teacher = tiny_teacher()
    heads = varied_heads(teacher, [1])
    tokens = {"input_ids": torch.tensor([[2, 5, 3], [2, 6, 3]]),
              "attention_mask": torch.ones(2, 3, dtype=torch.long)}
    sizes = []
    hook = teacher.bert.encoder.layer[1].register_forward_pre_hook(
        lambda _module, args: sizes.append(len(args[0])))
    try:
        _, depths = predict_batch(teacher, heads, {1: 1.0}, tokens, ((1, 0.0),))
    finally:
        hook.remove()
    np.testing.assert_array_equal(depths, [1, 1])
    assert sizes == []


def test_validation_selection_and_report_on_cpu(monkeypatch, tmp_path, tokenizer):
    from run_batch_exits import encode_batch, load_scalar_reference, select_policies, test_policies as report
    from test_exits import headlines

    teacher = tiny_teacher()
    heads = varied_heads(teacher, [1])
    temperatures = {1: 1.0}
    cfg = {"max_length": 8, "exit_sets": [[1]], "exit_layers": [1],
           "validation_f1_drop_limit": 1.0, "f1_drop_budget": 0.01,
           "timing_seed": 0, "warmup": 1, "timing_repeats": 1, "latency_samples": 8}
    device = torch.device("cpu")
    monkeypatch.setattr("run_batch_exits.BATCH_SIZES", (1, 4))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)

    def route(policy):
        return lambda tokens: predict_batch(teacher, heads, temperatures, tokens, policy)

    def encode(batch):
        return encode_batch(batch, tokenizer, device, cfg)

    exit_run = tmp_path / "scalar"
    exit_run.mkdir()
    (exit_run / "selected_policies.json").write_text(json.dumps({
        "selected": {"max_prob": {"exit_layers": [1], "threshold": 0.76}},
    }))
    reference = load_scalar_reference(exit_run)
    assert reference == ((1, 0.76),)
    fixed, shared, adaptive = select_policies(
        route, reference, headlines(9, 1), encode, device, cfg, tmp_path)
    assert fixed and set(adaptive) == {1, 4}
    report(route, reference, fixed, shared, adaptive, headlines(9, 2), encode, device, cfg, tmp_path, 4)
    selected = json.loads((tmp_path / "selected_policies.json").read_text())
    assert selected["scalar_reference"] == [[1, 0.76]]
    assert [[1, 0.76]] in selected["candidates"]
    assert selected["shared"] == [list(exit_rule) for exit_rule in shared]
    metrics = pd.read_csv(tmp_path / "metrics.csv")
    assert set(metrics.name) == {"full", "fixed", "shared", "adaptive", "scalar_reference"}
    assert metrics.groupby("name").batch_size.apply(list).to_dict() == {
        name: [1, 4] for name in metrics.name.unique()
    }
    assert len(pd.read_csv(tmp_path / "predictions.csv")) == 90


def test_load_frozen_policies_replays_the_a100_run(tmp_path):
    source = Path(__file__).resolve().parents[1] / "reports/nyu_hpc/batch-exits-29406/selected_policies.json"
    reference = ((8, 0.76),)
    fixed, shared, adaptive = load_frozen_policies(source, reference, [2, 4, 8])

    assert fixed == shared == reference
    assert adaptive == {1: reference, 4: FULL, 16: FULL, 32: reference}
    with pytest.raises(ValueError, match="different reference policies"):
        load_frozen_policies(source, ((8, 0.9),), [2, 4, 8])

    saved = json.loads(source.read_text())
    del saved["adaptive"]["32"]
    incomplete = tmp_path / "selected_policies.json"
    incomplete.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="must cover batch sizes"):
        load_frozen_policies(incomplete, reference, [2, 4, 8])
