"""Rebuild assets/ from public sources when the Colab archives are unavailable.

Step 1 needs internet but little compute (a Torch login node or a laptop):
    python tools/build_assets.py download --out assets
Step 2 needs a GPU (see hpc/build_assets.sbatch):
    python tools/build_assets.py build --out assets

Step 2 repeats the Colab notebook's data pipeline exactly (pinned SEntFiN commit,
single-entity filtering, duplicate grouping, grouped stratified splits) and
checks the splits against the Colab test rows and file checksums. It then
fine-tunes ProsusAI/finbert with the notebook's teacher recipe. The new teacher
is a fresh training run: close to, but not bitwise identical to, Colab's.
"""
import argparse
import ast
import hashlib
import json
import math
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_COMMIT = "eba43610fe3cb57a3e5773cd6c037da1f8992ad5"
DATA_URL = f"https://raw.githubusercontent.com/pyRis/SEntFiN/{DATA_COMMIT}/SEntFiN.csv"
DATA_SHA256 = "570af7ba58f724f3fcc414dd5ba80da5b1a3b342134d76c6ebdf53a32c54a92f"
MODEL_ID = "ProsusAI/finbert"
MODEL_REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
MODEL_FILES = ["config.json", "model.safetensors", "vocab.txt", "tokenizer_config.json",
               "special_tokens_map.json"]
LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}
# The Colab notebook's settings for data preparation and teacher fine-tuning.
SPLIT_SEED, NEAR_DUPLICATE_SIMILARITY = 20261023, 0.90
TEACHER = {"seed": 42, "epochs": 3, "learning_rate": 2e-5, "weight_decay": 0.01,
           "warmup_fraction": 0.1, "batch_size": 16, "max_length": 96}


def sha256(path):
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def download(out):
    source = out / "source"
    source.mkdir(parents=True, exist_ok=True)
    csv = source / "SEntFiN.csv"
    if not csv.exists():
        urllib.request.urlretrieve(DATA_URL, csv)
    if sha256(csv) != DATA_SHA256:
        raise ValueError("SEntFiN checksum mismatch; delete the file and download again.")
    from huggingface_hub import hf_hub_download
    for name in MODEL_FILES:
        hf_hub_download(MODEL_ID, name, revision=MODEL_REVISION, local_dir=source / "finbert")
    print(f"Downloaded SEntFiN ({DATA_COMMIT[:7]}) and {MODEL_ID} ({MODEL_REVISION[:7]}) to {source}")


# ---- Data preparation: copied from the Colab notebook's sentfin_pipeline.py ----

def parse_decisions(value, *, with_audit=False):
    """No eval. Handle the release's unescaped apostrophes with a strict fallback."""
    repaired = False
    try:
        decisions = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        repaired = True
        if not isinstance(value, str) or not value.strip().startswith("{") or not value.strip().endswith("}"):
            raise ValueError("Malformed Decisions field.") from None
        body = value.strip()[1:-1]
        pattern = re.compile(r'''\s*(['"])(.*?)\1\s*:\s*(['"])(positive|negative|neutral)\3\s*(?:,|$)''')
        decisions, position = {}, 0
        while position < len(body):
            match = pattern.match(body, position)
            if match is None:
                raise ValueError("Unrecognized Decisions field; no partial parsing allowed.")
            decisions[match.group(2)] = match.group(4)
            position = match.end()
    if not isinstance(decisions, dict) or not decisions:
        raise ValueError("Expected a nonempty entity-to-label dictionary.")
    if any(not isinstance(k, str) or not k.strip() or v not in LABEL2ID for k, v in decisions.items()):
        raise ValueError("Invalid entity or label in Decisions.")
    return (decisions, repaired) if with_audit else decisions


def normalize_headline(text, *, mask_numbers=False):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    if mask_numbers:
        text = re.sub(r"\d+(?:[.,]\d+)*", " number ", text)
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def prepare_sentfin(csv_path):
    raw = pd.read_csv(csv_path)
    records, rejected, repaired_ids, multi_entity = [], [], [], 0
    for index, row in raw.iterrows():
        try:
            decisions, repaired = parse_decisions(row["Decisions"], with_audit=True)
            if repaired:
                repaired_ids.append(int(index))
            if len(decisions) != 1:
                multi_entity += 1
                continue
            entity, label = next(iter(decisions.items()))
            if not isinstance(row["Title"], str) or not row["Title"].strip():
                raise ValueError("Empty headline.")
            records.append({"row_id": int(index), "text": row["Title"].strip(), "entity": entity,
                            "label": LABEL2ID[label], "normalized": normalize_headline(row["Title"])})
        except (ValueError, TypeError):
            rejected.append(int(index))
    frame = pd.DataFrame(records)
    # Drop all conflicting exact-text labels instead of choosing a convenient label.
    conflicts = frame.groupby("normalized")["label"].nunique()
    conflict_keys = set(conflicts[conflicts > 1].index)
    conflicting_count = int(frame["normalized"].isin(conflict_keys).sum())
    frame = frame[~frame["normalized"].isin(conflict_keys)].copy()
    before = len(frame)
    frame = frame.drop_duplicates("normalized").reset_index(drop=True)
    audit = {"raw_headlines": len(raw), "single_entity_before_dedup": len(records),
             "excluded_multi_entity": multi_entity, "rejected_row_ids": rejected,
             "safely_repaired_decisions_row_ids": repaired_ids,
             "conflicting_exact_text_rows_removed": conflicting_count,
             "exact_duplicate_rows_removed": before - len(frame), "retained": len(frame)}
    return frame, audit


def assign_duplicate_groups(frame, similarity):
    """Group numeric variants plus character-ngram near duplicates, transitively."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    frame = frame.copy().reset_index(drop=True)
    parent = np.arange(len(frame))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = root(i), root(j)
        parent[max(a, b)] = min(a, b)

    templates = {}
    for i, text in enumerate(frame.text):
        key = normalize_headline(text, mask_numbers=True)
        if key in templates:
            union(i, templates[key])
        else:
            templates[key] = i
    features = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), dtype=np.float32).fit_transform(frame.normalized)
    neighbors = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=1).fit(features)
    for start in range(0, len(frame), 128):
        matches = neighbors.radius_neighbors(features[start:start + 128], radius=1 - similarity + 1e-7,
                                             return_distance=False)
        for offset, indices in enumerate(matches):
            for j in indices:
                union(start + offset, int(j))
    frame["group_id"] = [f"group_{root(i):05d}" for i in range(len(frame))]
    return frame


def make_splits(frame, directory, seed):
    """Ten grouped stratified folds -> approximately 60/10/10/20 percent."""
    from sklearn.model_selection import StratifiedGroupKFold
    frame = frame.copy()
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    frame["split"] = ""
    for fold, (_, heldout) in enumerate(splitter.split(frame, frame.label, frame.group_id)):
        name = "train" if fold < 6 else "validation" if fold == 6 else "calibration" if fold == 7 else "test"
        frame.loc[frame.index[heldout], "split"] = name
    if frame.groupby("group_id").split.nunique().max() != 1:
        raise AssertionError("A duplicate group crossed split boundaries.")
    splits = {name: part.reset_index(drop=True) for name, part in frame.groupby("split")}
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "split_manifest.csv", index=False)
    for name, part in splits.items():
        part.to_json(directory / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
    summary = pd.crosstab(frame.split, frame.label).rename(columns=ID2LABEL)
    summary["total"] = summary.sum(axis=1)
    summary["groups"] = frame.groupby("split").group_id.nunique()
    summary.to_csv(directory / "split_counts.csv")
    return splits


def check_splits(splits, out):
    """Compare with the frozen Colab test rows and split-file checksums."""
    reference = pd.read_csv(ROOT / "reference" / "colab_predictions.csv")
    test = splits["test"]
    same_test = (len(test) == len(reference) and
                 test[["row_id", "label", "group_id"]].reset_index(drop=True)
                 .equals(reference[["row_id", "label", "group_id"]]))
    frozen = json.loads((ROOT / "reference" / "assets_manifest.json").read_text())["sha256"]
    files = {name: sha256(out / name) == expected for name, expected in frozen.items()
             if name.startswith("splits/")}
    return {"test_rows_labels_groups_match_colab": bool(same_test),
            "split_files_identical_to_colab": files,
            "sizes": {name: len(part) for name, part in splits.items()}}


# ---- Teacher fine-tuning: the notebook's recipe ----

def fine_tune_teacher(splits, out, device):
    import torch
    from torch.utils.data import DataLoader
    from transformers import (AutoTokenizer, BertForSequenceClassification, DataCollatorWithPadding,
                              get_linear_schedule_with_warmup)
    from data import Headlines, loader, logits_for, macro_f1, seed_everything

    cfg = TEACHER
    pretrained = out / "source" / "finbert"
    tokenizer = AutoTokenizer.from_pretrained(pretrained, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(pretrained, attn_implementation="eager",
                                                          use_safetensors=True, local_files_only=True)
    if model.config.label2id != LABEL2ID:
        raise ValueError("FinBERT's label order differs from the SEntFiN mapping.")
    model.to(device)
    seed_everything(cfg["seed"])
    batches = DataLoader(Headlines(splits["train"], tokenizer, cfg["max_length"]), batch_size=cfg["batch_size"],
                         shuffle=True, generator=torch.Generator().manual_seed(cfg["seed"]),
                         collate_fn=DataCollatorWithPadding(tokenizer))
    validation = loader(splits["validation"], tokenizer, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * len(batches)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * cfg["warmup_fraction"]), steps)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, history = -math.inf, []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total = 0.0
        for batch in batches:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite teacher training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous:
                scheduler.step()
            total += float(loss.detach()) * len(batch["labels"])
        score = macro_f1(splits["validation"].label, logits_for(model, validation, device).argmax(-1))
        history.append({"epoch": epoch, "train_loss": total / len(splits["train"]), "validation_macro_f1": score})
        print(history[-1], flush=True)
        if score > best:
            best = score
            model.save_pretrained(out / "teacher", safe_serialization=True)
            tokenizer.save_pretrained(out / "teacher")
    return history


def build(out, device_name):
    import torch
    csv = out / "source" / "SEntFiN.csv"
    if not csv.exists() or sha256(csv) != DATA_SHA256:
        raise SystemExit("Run the download step first.")
    if (out / "teacher").exists():
        raise SystemExit(f"{out / 'teacher'} already exists; use a new --out folder.")
    frame, audit = prepare_sentfin(csv)
    frame = assign_duplicate_groups(frame, NEAR_DUPLICATE_SIMILARITY)
    splits = make_splits(frame, out / "splits", SPLIT_SEED)
    check = check_splits(splits, out)
    print("Split check:", json.dumps(check, indent=2), flush=True)
    history = fine_tune_teacher(splits, out, torch.device(device_name))
    files = sorted(str(path.relative_to(out)) for folder in ("splits", "teacher")
                   for path in (out / folder).iterdir() if path.is_file())
    (out / "manifest.json").write_text(json.dumps({
        "sha256": {name: sha256(out / name) for name in files},
        "rebuilt_from_public_sources": True, "dataset_commit": DATA_COMMIT, "split_seed": SPLIT_SEED,
        "pretrained_model_revision": MODEL_REVISION, "teacher_recipe": TEACHER}, indent=2) + "\n")
    (out / "build_record.json").write_text(json.dumps({
        "data_audit": audit, "split_check": check, "teacher_history": history}, indent=2) + "\n")
    print(f"Built {out}: splits, fine-tuned teacher, and manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["download", "build"])
    parser.add_argument("--out", type=Path, default=Path("assets"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.step == "download":
        download(args.out)
    else:
        build(args.out, args.device)


if __name__ == "__main__":
    main()
