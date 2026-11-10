import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from benchmark import encode
from exits import MultiExit, decision, exit_table, new_head, policy_grid, simulate, train_heads
from models import fingerprint
from run_exits import study

WORDS = ["good", "bad", "flat", "up", "down", "profit", "loss", "beats", "misses", "steady"]


def tiny_teacher(layers=4):
    torch.manual_seed(0)
    config = BertConfig(vocab_size=5 + len(WORDS), hidden_size=12, num_hidden_layers=layers,
                        num_attention_heads=3, intermediate_size=24, num_labels=3)
    config._attn_implementation = "eager"
    return BertForSequenceClassification(config).requires_grad_(False).eval()


@pytest.fixture
def tokenizer(tmp_path):
    vocabulary = tmp_path / "vocab.txt"
    vocabulary.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]))
    return BertTokenizerFast(vocab_file=str(vocabulary))


def headlines(count, seed):
    rng = np.random.default_rng(seed)
    texts = [" ".join(rng.choice(WORDS, size=rng.integers(1, 5))) for _ in range(count)]
    return pd.DataFrame({"row_id": np.arange(count) + 1000 * seed, "text": texts,
                         "label": rng.integers(0, 3, count), "group_id": [f"g{i}" for i in range(count)]})


def varied_heads(teacher, layers):
    heads = torch.nn.ModuleDict({str(layer): new_head(teacher) for layer in layers})
    torch.manual_seed(1)
    for head in heads.values():
        torch.nn.init.normal_(head[-1].weight, std=2.0)
    return heads.eval()


def test_decision_values_for_both_rules():
    uniform, confident = decision(torch.zeros(1, 3), 1.0)[0], decision(torch.tensor([[9., 0., 0.]]), 1.0)[0]
    torch.testing.assert_close(uniform[:2], torch.tensor([1 / 3, 1.0]))
    assert confident[0] > 0.99 and confident[1] < 0.01 and confident[2] == 0
    # A higher temperature flattens probabilities: lower confidence, higher entropy.
    warm = decision(torch.tensor([[9., 0., 0.]]), 4.0)[0]
    assert warm[0] < confident[0] and warm[1] > confident[1]


def test_simulation_exactly_matches_real_inference(tokenizer):
    teacher = tiny_teacher()
    heads = varied_heads(teacher, [1, 2, 3])
    temperatures = {1: 1.3, 2: 0.8, 3: 1.0}
    frame = headlines(40, seed=1)
    cfg = {"max_length": 8}
    table = exit_table(teacher, heads, temperatures, frame, tokenizer, cfg, torch.device("cpu"))
    tokens = [encode(text, tokenizer, torch.device("cpu"), 8) for text in frame.text]
    exits_seen = set()
    for rule, values in (("max_prob", [0.4, 0.6, 0.8, 0.95]), ("entropy", [0.2, 0.5, 0.8, 0.95])):
        for exit_set in ([3], [1, 3], [1, 2, 3]):
            for threshold in values:
                policy = {"kind": "dynamic", "rule": rule, "exit_layers": exit_set, "threshold": threshold}
                expected = simulate(table, policy, 4)
                route = MultiExit(teacher, heads, temperatures, exit_set, rule, threshold)
                labels, depths = map(np.array, zip(*(route.predict(item) for item in tokens)))
                np.testing.assert_array_equal(labels, expected[0])
                np.testing.assert_array_equal(depths, expected[1])
                exits_seen |= set(depths)
    assert exits_seen == {1, 2, 3, 4}, "The test should exercise every exit and the full depth"


def test_confident_exit_skips_every_later_layer():
    teacher = tiny_teacher()
    heads = varied_heads(teacher, [2])
    calls = []
    handles = [teacher.bert.encoder.layer[index].register_forward_hook(lambda *args: calls.append(1))
               for index in (2, 3)]
    tokens = {"input_ids": torch.tensor([[2, 5, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    assert MultiExit(teacher, heads, {2: 1.0}, [2], "max_prob", 0.0).predict(tokens)[1] == 2
    assert calls == []
    assert MultiExit(teacher, heads, {2: 1.0}, [2], "entropy", -1.0).predict(tokens)[1] == 4
    assert calls == [1, 1]
    for handle in handles:
        handle.remove()


def test_head_training_keeps_teacher_and_ignores_test(tokenizer):
    teacher = tiny_teacher()
    before = fingerprint(teacher)
    cfg = {"seed": 3, "max_length": 8, "batch_size": 8, "exit_layers": [1, 2], "head_epochs": 3,
           "head_learning_rate": 0.01, "head_batch_size": 8, "head_weight_decay": 0.01,
           "head_warmup_fraction": 0.1, "gold_weight": 0.75, "distillation_temperature": 2.0}
    splits = {"train": headlines(24, seed=2), "validation": headlines(12, seed=3)}  # no test split given
    heads, history, epochs = train_heads(teacher, tokenizer, splits, cfg, torch.device("cpu"))
    assert fingerprint(teacher) == before
    assert set(heads) == {"1", "2"} and set(epochs) == {1, 2}
    assert all(not parameter.requires_grad for parameter in heads.parameters())
    assert [row["epoch"] for row in history if row["layer"] == 1] == [0, 1, 2, 3]


def test_policy_grid_covers_both_rules_and_static_exits():
    cfg = {"exit_layers": [2, 4, 8], "rules": ["max_prob", "entropy"], "exit_sets": [[8], [2, 4, 8]],
           "max_prob_thresholds": [0.5, 0.9, 0.1], "entropy_thresholds": [0.1, 0.3, 0.1]}
    grid = policy_grid(cfg)
    assert [policy["name"] for policy in grid[:4]] == ["full12", "static2", "static4", "static8"]
    dynamic = [policy for policy in grid if policy["kind"] == "dynamic"]
    assert len(dynamic) == 2 * 5 + 2 * 3
    assert math.isclose(max(policy["threshold"] for policy in dynamic if policy["rule"] == "max_prob"), 0.9)


def test_study_runs_end_to_end_without_timing(tmp_path, tokenizer):
    teacher = tiny_teacher()
    cfg = {"seed": 3, "max_length": 8, "batch_size": 8, "exit_layers": [1, 2], "head_epochs": 2,
           "head_learning_rate": 0.01, "head_batch_size": 8, "head_weight_decay": 0.01,
           "head_warmup_fraction": 0.1, "gold_weight": 0.75, "distillation_temperature": 2.0,
           "rules": ["max_prob", "entropy"], "exit_sets": [[2], [1, 2]],
           "max_prob_thresholds": [0.34, 0.99, 0.05], "entropy_thresholds": [0.01, 0.99, 0.05],
           "validation_f1_drop_limit": 1.0, "f1_drop_budget": 1.0}
    splits = {"train": headlines(24, 4), "validation": headlines(16, 5),
              "calibration": headlines(16, 6), "test": headlines(20, 7)}
    out = tmp_path / "study"
    out.mkdir()
    quality = study(teacher, tokenizer, splits, cfg, out, torch.device("cpu"), timing=False)
    assert list(quality.name) == ["full12", "static1", "static2", "dynamic_max_prob", "dynamic_entropy"]
    verification = json.loads((out / "verification.json").read_text())
    assert verification["simulation_matches_real_inference"]
    assert not verification["test_used_for_selection"]
    selected = json.loads((out / "selected_policies.json").read_text())
    assert selected["ranked_by"] == "mean_layers" and set(selected["selected"]) == {"max_prob", "entropy"}
    for name in ("exit_heads.pt", "temperatures.json", "calibration.csv", "validation_sweep.csv",
                 "predictions.csv", "test_exit_scores.csv", "metrics.csv"):
        assert (out / name).exists(), name
