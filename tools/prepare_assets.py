"""Assemble the exact inputs from the three previously delivered Colab archives."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--teacher", type=Path, required=True, help="Full experiment results ZIP")
parser.add_argument("--head", type=Path, required=True, help="Layer-8 training results ZIP")
parser.add_argument("--student", type=Path, required=True, help="Compact student checkpoint ZIP")
parser.add_argument("--out", type=Path, default=Path("assets"))
args = parser.parse_args()
reference = Path(__file__).resolve().parents[1] / "reference"


def copy_member(bundle, name, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with bundle.open(name) as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination)


with zipfile.ZipFile(args.teacher) as bundle:
    for name in bundle.namelist():
        if name.startswith("deployment/backbone/"):
            copy_member(bundle, name, args.out / "teacher" / Path(name).name)
        elif name.startswith("deployment/") and Path(name).name in {
                "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"}:
            copy_member(bundle, name, args.out / "teacher" / Path(name).name)
        elif name.startswith("splits/") and not name.endswith("/"):
            copy_member(bundle, name, args.out / "splits" / Path(name).name)
with zipfile.ZipFile(args.head) as bundle:
    copy_member(bundle, "selected_head.pt", args.out / "head8.pt")
with zipfile.ZipFile(args.student) as bundle:
    for name in bundle.namelist():
        if name.startswith("best_student/") and not name.endswith("/"):
            copy_member(bundle, name, args.out / "student" / Path(name).name)
for name in ("colab_predictions.csv", "colab_metrics.csv", "colab_environment.json"):
    shutil.copyfile(reference / name, args.out / name)
manifest = json.loads((reference / "assets_manifest.json").read_text())
for name, expected in manifest["sha256"].items():
    with (args.out / name).open("rb") as file:
        assert hashlib.file_digest(file, "sha256").hexdigest() == expected, name
shutil.copyfile(reference / "assets_manifest.json", args.out / "manifest.json")
print("All checkpoint, split, and reference checksums match the Colab inputs.")
