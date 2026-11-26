"""Prepare grouped PhraseBank splits and fine-tune a finance language model."""
import argparse
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import save_json, seed_everything
from tools.build_assets import (
    ID2LABEL, LABEL2ID, SPLIT_SEED, TEACHER, assign_duplicate_groups,
    fine_tune_teacher, make_splits, normalize_headline, sha256,
)

DATA_REVISION = "8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964"
DATA_SHA256 = "0e1a06c4900fdae46091d031068601e3773ba067c7cecb5b0da1dcba5ce989a6"
MODEL_ID = "yiyanghkust/finbert-pretrain"
MODEL_REVISION = "88ab954a39ea6d3ce2b62cff086dd5ad1172c664"


def phrasebank_data(path):
    if sha256(path) != DATA_SHA256:
        raise ValueError("Financial PhraseBank archive checksum mismatch.")
    with ZipFile(path) as source:
        lines = source.read("FinancialPhraseBank-v1.0/Sentences_50Agree.txt").decode("latin-1").splitlines()
    records = []
    for row_id, line in enumerate(lines):
        if line.strip():
            text, label = line.rsplit("@", 1)
            records.append((row_id, text.strip(), LABEL2ID[label.strip()], normalize_headline(text)))
    frame = pd.DataFrame(records, columns=["row_id", "text", "label", "normalized"])
    labels_by_text = frame.groupby("normalized").label.nunique()
    conflicts = labels_by_text[labels_by_text > 1].index
    clean = frame[~frame.normalized.isin(conflicts)]
    unique = clean.drop_duplicates("normalized")
    return unique, {"raw_rows": len(frame), "conflicting_rows_removed": len(frame) - len(clean),
                    "exact_duplicates_removed": len(clean) - len(unique), "retained": len(unique)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    frame, audit = phrasebank_data(args.downloads / "FinancialPhraseBank-v1.0.zip")
    splits = make_splits(assign_duplicate_groups(frame, 0.90), args.out / "splits", SPLIT_SEED)
    audit["split_sizes"] = {name: len(part) for name, part in splits.items()}
    audit["split_seed"] = SPLIT_SEED
    save_json(args.out / "split_audit.json", audit)
    print("Grouped PhraseBank splits:", audit, flush=True)

    base = args.downloads / "checkpoint"
    seed_everything(TEACHER["seed"])
    model, info = BertForSequenceClassification.from_pretrained(
        base, attn_implementation="eager", num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
        use_safetensors=False, local_files_only=True, output_loading_info=True,
    )
    if any(name.startswith("bert.") for name in info["missing_keys"]):
        raise ValueError(f"Missing BERT weights: {info['missing_keys']}")
    initial = args.out / "source" / "finbert"
    model.save_pretrained(initial, safe_serialization=True)
    BertTokenizerFast(vocab_file=str(base / "vocab.txt"), do_lower_case=True).save_pretrained(initial)
    del model

    history = fine_tune_teacher(splits, args.out, torch.device(args.device))
    files = sorted(str(path.relative_to(args.out)) for folder in ("splits", "teacher")
                   for path in (args.out / folder).iterdir() if path.is_file())
    provenance = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                  "model_weights_sha256": sha256(base / "pytorch_model.bin"),
                  "dataset_revision": DATA_REVISION, "dataset_sha256": DATA_SHA256,
                  "teacher_recipe": TEACHER, "teacher_device": args.device,
                  "teacher_history": history, "test_used_for_training_or_selection": False}
    save_json(args.out / "manifest.json", {**provenance,
              "sha256": {name: sha256(args.out / name) for name in files}})
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
