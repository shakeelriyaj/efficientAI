"""Three routes: full teacher, one dynamic exit, and a fixed eight-layer student."""
import copy
import hashlib
import torch
from torch import nn
from transformers import BertForSequenceClassification


def fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_model(path, device):
    return BertForSequenceClassification.from_pretrained(
        path, attn_implementation="eager", local_files_only=True
    ).to(device).requires_grad_(False).eval()


def make_student(teacher, kept_layers):
    """Copy selected blocks in their original order; never modify the teacher."""
    if kept_layers != sorted(set(kept_layers)) or not all(
            1 <= layer <= len(teacher.bert.encoder.layer) for layer in kept_layers):
        raise ValueError("Choose unique, increasing one-based layer numbers.")
    student = copy.deepcopy(teacher)
    blocks = list(student.bert.encoder.layer)
    student.bert.encoder.layer = nn.ModuleList([blocks[layer - 1] for layer in kept_layers])
    student.config.num_hidden_layers = len(kept_layers)
    student.bert.config.num_hidden_layers = len(kept_layers)
    return student


class DynamicExit:
    """Check confidence once at layer 8; continue only for uncertain inputs."""
    def __init__(self, teacher, head_path, cfg):
        self.teacher, self.cfg = teacher, cfg
        self.head = nn.Sequential(copy.deepcopy(teacher.bert.pooler),
                                  copy.deepcopy(teacher.dropout), copy.deepcopy(teacher.classifier))
        self.head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
        self.head.to(next(teacher.parameters()).device).requires_grad_(False).eval()
        layers = tuple(teacher.bert.encoder.layer)
        self.prefix, self.remaining = layers[:cfg["exit_layer"]], layers[cfg["exit_layer"]:]

    @torch.inference_mode()
    def predict(self, tokens):
        if tokens["input_ids"].shape[0] != 1:
            raise ValueError("Dynamic routing supports one headline at a time.")
        bert = self.teacher.bert
        mask = bert.get_extended_attention_mask(tokens["attention_mask"], tokens["input_ids"].shape)
        hidden = bert.embeddings(input_ids=tokens["input_ids"], token_type_ids=tokens.get("token_type_ids"))
        for layer in self.prefix:
            hidden = layer(hidden, attention_mask=mask)[0]
        scores = self.head(hidden) / self.cfg["exit_temperature"]
        confidence, label = scores.softmax(-1).max(-1)
        # One transfer carries both values; no second transfer when exiting.
        confidence, label = torch.stack((confidence, label.to(confidence.dtype)), -1).cpu().tolist()[0]
        if confidence >= self.cfg["exit_threshold"]:
            return int(label), self.cfg["exit_layer"]
        for layer in self.remaining:
            hidden = layer(hidden, attention_mask=mask)[0]
        scores = self.teacher.classifier(self.teacher.dropout(bert.pooler(hidden)))
        return int(scores.argmax(-1).item()), len(bert.encoder.layer)


@torch.inference_mode()
def predict_fixed(model, tokens):
    return int(model(**tokens).logits.argmax(-1).item()), len(model.bert.encoder.layer)
