"""Use the exact frozen Colab splits; do not split or clean the data again."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_splits(assets):
    manifest = json.loads((assets / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        with (assets / name).open("rb") as file:
            actual = hashlib.file_digest(file, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Input checksum mismatch: {name}")
    splits = {name: pd.read_json(assets / "splits" / f"{name}.jsonl", lines=True)
              for name in ("train", "validation", "calibration", "test")}
    return splits


class Headlines(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        self.tokens = tokenizer(frame.text.tolist(), truncation=True, max_length=max_length)
        self.labels = frame.label.astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {**{key: values[index] for key, values in self.tokens.items()}, "labels": self.labels[index]}


def loader(frame, tokenizer, cfg):
    return DataLoader(Headlines(frame, tokenizer, cfg["max_length"]),
                      batch_size=cfg["batch_size"], shuffle=False,
                      generator=torch.Generator().manual_seed(42),
                      collate_fn=DataCollatorWithPadding(tokenizer))


@torch.inference_mode()
def logits_for(model, batches, device):
    model.eval()
    return torch.cat([model(**{key: value.to(device) for key, value in batch.items()
                              if key != "labels"}).logits.float().cpu() for batch in batches])


def macro_f1(labels, predictions):
    return float(f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0))
