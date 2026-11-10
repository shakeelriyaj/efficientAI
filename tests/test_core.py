import copy
import json
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
import pandas as pd

from data import loader, logits_for, macro_f1
from models import DynamicExit, fingerprint, make_student
from run import load_config
from train import distillation_loss, train_student


def tiny_teacher():
    config = BertConfig(vocab_size=12, hidden_size=12, num_hidden_layers=3,
                        num_attention_heads=3, intermediate_size=24, num_labels=3)
    config._attn_implementation = "eager"
    return BertForSequenceClassification(config).requires_grad_(False).eval()


def test_wrong_teacher_does_not_teach_its_mistake():
    student = torch.tensor([[1., 0., -1.]], requires_grad=True)
    teacher = torch.tensor([[0., 4., 0.]], requires_grad=True)
    labels = torch.tensor([0])
    cfg = {"gold_weight": .75, "distillation_temperature": 2.}
    loss = distillation_loss(student, teacher, labels, cfg)
    torch.testing.assert_close(loss, F.cross_entropy(student, labels))
    loss.backward()
    assert student.grad is not None and teacher.grad is None


def test_training_adapts_student_without_changing_teacher_or_reading_test(tmp_path):
    teacher = tiny_teacher()
    before = fingerprint(teacher)
    vocabulary = tmp_path / "vocab.txt"
    vocabulary.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "good", "bad", "flat"]))
    tokenizer = BertTokenizerFast(vocab_file=str(vocabulary))
    frame = pd.DataFrame({"text": ["good", "bad", "flat"] * 2, "label": [0,1,2] * 2})
    cfg = {"seed":4142,"max_length":8,"batch_size":2,"epochs":1,"kept_layers":[1,3],
           "learning_rate":.01,"weight_decay":.01,"warmup_fraction":.1,
           "gold_weight":.75,"distillation_temperature":2.}
    # Deliberately provide no test or calibration split to the training function.
    trained = train_student(teacher, tokenizer, {"train":frame,"validation":frame}, cfg, tmp_path, torch.device("cpu"))
    assert fingerprint(teacher) == before
    assert len(trained.bert.encoder.layer) == 2
    history = json.loads((tmp_path/"history.json").read_text())
    score = macro_f1(frame.label, logits_for(trained,loader(frame,tokenizer,cfg),torch.device("cpu")).argmax(-1))
    assert score == max(row["validation_macro_f1"] for row in history)
    assert not json.loads((tmp_path/"selection.json").read_text())["test_used_for_selection"]


def test_dynamic_exit_really_skips_later_blocks(tmp_path):
    teacher = tiny_teacher()
    head = nn.Sequential(copy.deepcopy(teacher.bert.pooler),copy.deepcopy(teacher.dropout),copy.deepcopy(teacher.classifier))
    with torch.no_grad():
        head[-1].weight.zero_()
        head[-1].bias.copy_(torch.tensor([20.,0.,0.]))
    path = tmp_path/"head.pt"
    torch.save(head.state_dict(),path)
    route = DynamicExit(teacher,path,{"exit_layer":2,"exit_temperature":1.,"exit_threshold":.9})
    calls = []
    handle = teacher.bert.encoder.layer[2].register_forward_hook(lambda *args:calls.append(1))
    tokens = {"input_ids":torch.tensor([[2,5,3]]),"attention_mask":torch.ones(1,3,dtype=torch.long)}
    assert route.predict(tokens) == (0,2)
    assert calls == []
    with torch.no_grad():route.head[-1].bias.zero_()
    assert route.predict(tokens)[1] == 3
    assert calls == [1]
    handle.remove()


def test_seed_override_changes_only_the_seed(tmp_path):
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps({"seed": 4142, "epochs": 5}))
    assert load_config(path) == {"seed": 4142, "epochs": 5}
    assert load_config(path, seed=7) == {"seed": 7, "epochs": 5}
