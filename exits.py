"""Exit heads at several layers, two exit rules, and exact policy simulation.

The full teacher stays frozen. Each exit head is a copy of the teacher's
pooler/dropout/classifier, trained on that layer's [CLS] vector. An exit fires
when the calibrated head is confident enough:

- max_prob: the highest class probability is at least the threshold.
- entropy:  normalized entropy (0 = certain, 1 = uniform) is at most the threshold.

Policies are scored from per-headline decision values recorded at batch size
one with the same operations used during real inference, so simulated and
real predictions can be compared exactly.
"""
import copy
import math

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from torch import nn
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

from benchmark import encode
from data import loader, macro_f1
from train import distillation_loss

RULES = ("max_prob", "entropy")


def new_head(teacher):
    return nn.Sequential(copy.deepcopy(teacher.bert.pooler), copy.deepcopy(teacher.dropout),
                         copy.deepcopy(teacher.classifier))


def decision(logits, temperature):
    """Return [max probability, normalized entropy, label] for each row."""
    probabilities = (logits / temperature).softmax(-1)
    max_prob, label = probabilities.max(-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1) / math.log(logits.shape[-1])
    return torch.stack((max_prob, entropy, label.to(probabilities.dtype)), -1)


def passes(score, rule, threshold):
    if rule == "max_prob":
        return score >= threshold
    if rule == "entropy":
        return score <= threshold
    raise ValueError(f"Unknown exit rule: {rule}")


def start(bert, tokens):
    if tokens["input_ids"].shape[0] != 1:
        raise ValueError("Exit routing supports one headline at a time.")
    mask = bert.get_extended_attention_mask(tokens["attention_mask"], tokens["input_ids"].shape)
    hidden = bert.embeddings(input_ids=tokens["input_ids"], token_type_ids=tokens.get("token_type_ids"))
    return hidden, mask


def final_logits(teacher, hidden):
    return teacher.classifier(teacher.dropout(teacher.bert.pooler(hidden)))


class MultiExit:
    """Check exit heads in order; return at the first confident one."""
    def __init__(self, teacher, heads, temperatures, exit_layers, rule, threshold):
        if rule not in RULES:
            raise ValueError(f"Unknown exit rule: {rule}")
        self.teacher, self.heads, self.temperatures = teacher, heads, temperatures
        self.exit_layers, self.rule, self.threshold = set(exit_layers), rule, threshold

    @torch.inference_mode()
    def predict(self, tokens):
        bert = self.teacher.bert
        hidden, mask = start(bert, tokens)
        for number, layer in enumerate(bert.encoder.layer, 1):
            hidden = layer(hidden, attention_mask=mask)[0]
            if number in self.exit_layers:
                # One GPU-to-CPU copy per check carries both the score and the label.
                max_prob, entropy, label = decision(self.heads[str(number)](hidden),
                                                    self.temperatures[number]).cpu().tolist()[0]
                score = max_prob if self.rule == "max_prob" else entropy
                if passes(score, self.rule, self.threshold):
                    return int(label), number
        return int(final_logits(self.teacher, hidden).argmax(-1).item()), len(bert.encoder.layer)


class StaticExit:
    """Always run a fixed number of layers and use that layer's head; no checks."""
    def __init__(self, teacher, head, layer):
        self.teacher, self.head, self.layer = teacher, head, layer
        self.prefix = tuple(teacher.bert.encoder.layer)[:layer]

    @torch.inference_mode()
    def predict(self, tokens):
        hidden, mask = start(self.teacher.bert, tokens)
        for layer in self.prefix:
            hidden = layer(hidden, attention_mask=mask)[0]
        return int(self.head(hidden).argmax(-1).item()), self.layer


@torch.inference_mode()
def cache_features(teacher, frame, tokenizer, cfg, device):
    """Batched [CLS] vectors at each exit layer plus teacher logits (not timed)."""
    teacher.eval()
    cls = {layer: [] for layer in cfg["exit_layers"]}
    logits, labels = [], []
    for batch in loader(frame, tokenizer, cfg):
        labels.append(batch.pop("labels"))
        output = teacher(**{key: value.to(device) for key, value in batch.items()},
                         output_hidden_states=True, return_dict=True)
        # hidden_states[0] is the embedding output; hidden_states[k] follows layer k.
        for layer in cls:
            cls[layer].append(output.hidden_states[layer][:, 0].float().cpu())
        logits.append(output.logits.float().cpu())
    return {"cls": {layer: torch.cat(values) for layer, values in cls.items()},
            "teacher": torch.cat(logits), "labels": torch.cat(labels)}


@torch.no_grad()
def head_logits(head, cls, device, batch_size=256):
    head.eval()
    return torch.cat([head(part.to(device).unsqueeze(1)).float().cpu() for part in cls.split(batch_size)])


def train_head(teacher, layer, train, validation, cfg, device):
    """Train one head on training features; choose its epoch on validation F1."""
    torch.manual_seed(cfg["seed"] + layer)
    order = torch.Generator().manual_seed(cfg["seed"] + layer)
    head = new_head(teacher).to(device).requires_grad_(True)
    features = train["cls"][layer].to(device)
    targets, labels = train["teacher"].to(device), train["labels"].to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["head_learning_rate"],
                                  weight_decay=cfg["head_weight_decay"])
    steps = cfg["head_epochs"] * math.ceil(len(labels) / cfg["head_batch_size"])
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(steps * cfg["head_warmup_fraction"])), steps)
    best = macro_f1(validation["labels"], head_logits(head, validation["cls"][layer], device).argmax(-1))
    best_state, best_epoch = copy.deepcopy(head.state_dict()), 0
    history = [{"layer": layer, "epoch": 0, "loss": None, "validation_macro_f1": best}]
    for epoch in range(1, cfg["head_epochs"] + 1):
        head.train()
        total = 0.0
        for indices in torch.randperm(len(labels), generator=order).split(cfg["head_batch_size"]):
            indices = indices.to(device)
            loss = distillation_loss(head(features[indices].unsqueeze(1)), targets[indices], labels[indices], cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss for the layer-{layer} head")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * len(indices)
        score = macro_f1(validation["labels"], head_logits(head, validation["cls"][layer], device).argmax(-1))
        history.append({"layer": layer, "epoch": epoch, "loss": total / len(labels), "validation_macro_f1": score})
        if score > best:
            best, best_state, best_epoch = score, copy.deepcopy(head.state_dict()), epoch
    head.load_state_dict(best_state)
    return head.requires_grad_(False).eval(), history, best_epoch


def train_heads(teacher, tokenizer, splits, cfg, device):
    """Only training labels enter backward(); validation picks epochs."""
    train = cache_features(teacher, splits["train"], tokenizer, cfg, device)
    validation = cache_features(teacher, splits["validation"], tokenizer, cfg, device)
    heads, history, epochs = nn.ModuleDict(), [], {}
    for layer in cfg["exit_layers"]:
        heads[str(layer)], rows, epochs[layer] = train_head(teacher, layer, train, validation, cfg, device)
        history += rows
    return heads, history, epochs


def fit_temperature(logits, labels):
    """One positive scalar per head, fitted only on calibration labels."""
    logits, labels = logits.double(), labels.long()

    def nll(log_temperature):
        return F.cross_entropy(logits / math.exp(log_temperature), labels).item()

    solution = minimize_scalar(nll, bounds=(math.log(0.05), math.log(20.0)), method="bounded")
    return math.exp(solution.x) if solution.success and nll(solution.x) <= nll(0.0) else 1.0


def calibration_errors(logits, labels, bins=15):
    probabilities = logits.softmax(-1)
    confidence, predicted = probabilities.max(-1)
    correct = predicted.eq(labels).double()
    edges = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        members = (confidence > low) & (confidence <= high)
        if members.any():
            gap = correct[members].mean() - confidence[members].double().mean()
            ece += float(members.double().mean() * gap.abs())
    return {"nll": F.cross_entropy(logits.double(), labels).item(), "ece": ece}


def calibrate(heads, teacher, tokenizer, frame, cfg, device):
    cache = cache_features(teacher, frame, tokenizer, cfg, device)
    temperatures, rows = {}, []
    for layer in cfg["exit_layers"]:
        logits = head_logits(heads[str(layer)], cache["cls"][layer], device)
        temperatures[layer] = fit_temperature(logits, cache["labels"])
        before = calibration_errors(logits, cache["labels"])
        after = calibration_errors(logits / temperatures[layer], cache["labels"])
        rows.append({"layer": layer, "temperature": temperatures[layer],
                     **{f"before_{key}": value for key, value in before.items()},
                     **{f"after_{key}": value for key, value in after.items()}})
    return temperatures, pd.DataFrame(rows)


@torch.inference_mode()
def exit_table(teacher, heads, temperatures, frame, tokenizer, cfg, device):
    """Per headline at batch size one: every exit's decision values and the final label."""
    teacher.eval()
    bert, rows = teacher.bert, []
    for text in frame.text:
        hidden, mask = start(bert, encode(text, tokenizer, device, cfg["max_length"]))
        row = {}
        for number, layer in enumerate(bert.encoder.layer, 1):
            hidden = layer(hidden, attention_mask=mask)[0]
            if str(number) in heads:
                max_prob, entropy, label = decision(heads[str(number)](hidden), temperatures[number]).cpu().tolist()[0]
                row.update({f"max_prob_{number}": max_prob, f"entropy_{number}": entropy, f"label_{number}": int(label)})
        row[f"label_{len(bert.encoder.layer)}"] = int(final_logits(teacher, hidden).argmax(-1).item())
        rows.append(row)
    return pd.DataFrame(rows, index=frame.row_id.to_numpy())


def simulate(table, policy, depth):
    """Labels and executed layers a policy would produce, from recorded decisions."""
    if policy["kind"] == "full":
        return table[f"label_{depth}"].to_numpy().copy(), np.full(len(table), depth)
    if policy["kind"] == "static":
        layer = policy["layer"]
        return table[f"label_{layer}"].to_numpy().copy(), np.full(len(table), layer)
    labels, depths = table[f"label_{depth}"].to_numpy().copy(), np.full(len(table), depth)
    decided = np.zeros(len(table), dtype=bool)
    for layer in sorted(policy["exit_layers"]):
        stop = ~decided & passes(table[f"{policy['rule']}_{layer}"].to_numpy(), policy["rule"], policy["threshold"])
        labels[stop], depths[stop] = table[f"label_{layer}"].to_numpy()[stop], layer
        decided |= stop
    return labels, depths


def thresholds(cfg, rule):
    first, last, step = cfg[f"{rule}_thresholds"]
    return [round(value, 4) for value in np.arange(first, last + step / 2, step)]


def policy_grid(cfg):
    grid = [{"name": "full12", "kind": "full"}]
    grid += [{"name": f"static{layer}", "kind": "static", "layer": layer} for layer in cfg["exit_layers"]]
    for rule in cfg["rules"]:
        for exit_set in cfg["exit_sets"]:
            for threshold in thresholds(cfg, rule):
                grid.append({"name": f"{rule}_exits{'-'.join(map(str, exit_set))}_t{threshold:g}",
                             "kind": "dynamic", "rule": rule, "exit_layers": list(exit_set),
                             "threshold": threshold})
    return grid


def score(policy, labels, predictions, depths, full_f1, layers):
    f1 = macro_f1(labels, predictions)
    row = {"name": policy["name"], "kind": policy["kind"], "rule": policy.get("rule"),
           "exit_set": "-".join(map(str, policy.get("exit_layers", []))) or None,
           "threshold": policy.get("threshold"), "macro_f1": f1,
           "accuracy": float(np.mean(np.asarray(labels) == predictions)),
           "f1_drop_pp": 100 * (full_f1 - f1), "mean_layers": float(depths.mean()),
           "early_exit_fraction": float((depths < max(layers)).mean())}
    row.update({f"exit_at_{layer}": float((depths == layer).mean()) for layer in layers})
    return row


def sweep(table, labels, grid, layers):
    depth = max(layers)
    full_f1 = macro_f1(labels, table[f"label_{depth}"])
    rows = []
    for policy in grid:
        predictions, depths = simulate(table, policy, depth)
        rows.append(score(policy, labels, predictions, depths, full_f1, layers))
    return pd.DataFrame(rows)


def finalists(validation_sweep, grid, limit):
    """Per rule and exit set: the in-budget policy with the fewest mean layers."""
    eligible = validation_sweep[(validation_sweep.kind == "dynamic") & (validation_sweep.f1_drop_pp <= 100 * limit)]
    best = eligible.sort_values(["mean_layers", "macro_f1"], ascending=[True, False]).groupby(
        ["rule", "exit_set"], sort=False).head(1)
    by_name = {policy["name"]: policy for policy in grid}
    return [by_name[name] for name in best.name]
