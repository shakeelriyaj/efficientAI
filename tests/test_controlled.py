import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from controlled_routes import RoutedBatch, fixed_batch
from exits import new_head
from test_core import tiny_teacher


def tokens():
    return {"input_ids": torch.tensor([[2, 5, 3], [2, 6, 3], [2, 7, 3]]),
            "attention_mask": torch.ones(3, 3, dtype=torch.long)}


def test_forced_controls_preserve_full_predictions_and_depth():
    teacher = tiny_teacher()
    heads = nn.ModuleDict({"1": new_head(teacher)})
    with torch.no_grad():
        heads["1"][-1].weight.zero_()
        heads["1"][-1].bias.copy_(torch.tensor([20., 0., 0.]))
    full, _ = fixed_batch(teacher, tokens())
    for mode in ["loop", "head", "check"]:
        calls = []
        hook = teacher.bert.encoder.layer[-1].register_forward_hook(lambda *_: calls.append(1))
        labels, depths = RoutedBatch(teacher, heads, {1: 1.}, [(1, .9)], mode)(tokens())
        hook.remove()
        np.testing.assert_array_equal(labels, full)
        np.testing.assert_array_equal(depths, [3, 3, 3])
        assert calls == [1]


def test_dynamic_preserves_mixed_batch_order_and_compacts():
    class MixedHead(nn.Module):
        def forward(self, hidden):
            return torch.tensor([[0., 0., 0.], [0., 20., 0.], [0., 0., 0.]], device=hidden.device)
    teacher = tiny_teacher()
    full, _ = fixed_batch(teacher, tokens())
    sizes = []
    hook = teacher.bert.encoder.layer[1].register_forward_pre_hook(lambda _, args: sizes.append(len(args[0])))
    labels, depths = RoutedBatch(teacher, {"1": MixedHead()}, {1: 1.}, [(1, .9)])(tokens())
    hook.remove()
    np.testing.assert_array_equal(labels, [full[0], 1, full[2]])
    np.testing.assert_array_equal(depths, [3, 1, 3])
    assert sizes == [2]


def test_timing_rejects_prediction_changes(monkeypatch):
    from run_controlled import time_routes
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    class Tokenizer:
        def __call__(self, text, **_):
            class Encoding(dict):
                def to(self, _):
                    return self
            return Encoding(input_ids=torch.ones(len(text), 2, dtype=torch.long))
    calls = []
    def unstable(_):
        calls.append(1)
        return np.array([int(len(calls) > 1)]), np.array([3])
    frame = pd.DataFrame({"text": ["headline"], "row_id": [1]})
    cfg = {"timing_seed": 42, "latency_samples": 1, "max_length": 8,
           "timing_sessions": 1, "timing_repeats": 1, "warmup": 0}
    with pytest.raises(RuntimeError, match="Timed prediction/depth mismatch"):
        time_routes({"unstable": unstable}, frame, 1, Tokenizer(), "cpu", cfg)


def test_ce_control_uses_only_gold_labels():
    from train import distillation_loss
    student = torch.tensor([[1., 0., -1.]], requires_grad=True)
    teacher = torch.tensor([[4., 0., 0.]])
    labels = torch.tensor([0])
    loss = distillation_loss(student, teacher, labels,
                             {"gold_weight": 1., "distillation_temperature": 2.})
    torch.testing.assert_close(loss, nn.functional.cross_entropy(student, labels))
