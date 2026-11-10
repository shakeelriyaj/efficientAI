#!/bin/bash
#SBATCH --job-name=finbert-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/setup-%j.log
# Prepare software on a CPU compute node, not the login node.
# Optional: FINBERT_TORCH_CUDA=cu128 (or cu126) installs PyTorch built for an
# older NVIDIA driver; the default PyPI build needs a CUDA 13 driver (580+).
source "$SLURM_SUBMIT_DIR/hpc/env.sh"
export UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.python"
UV=$(find .bootstrap -type f -name uv 2>/dev/null | head -1 || true)
if [[ -z "$UV" ]]; then
  mkdir -p .bootstrap
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PWD/.bootstrap" UV_NO_MODIFY_PATH=1 sh
  UV=$(find .bootstrap -type f -name uv | head -1)
fi
export UV TORCH_INDEX=${FINBERT_TORCH_CUDA:+https://download.pytorch.org/whl/$FINBERT_TORCH_CUDA}
in_env '
"$UV" venv --python 3.13.16 .venv
if [[ -n "$TORCH_INDEX" ]]; then
  "$UV" pip install --python .venv/bin/python torch==2.11.0 --index-url "$TORCH_INDEX"
fi
"$UV" pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -c "import torch, transformers; print(\"torch\", torch.__version__, \"cuda build\", torch.version.cuda, \"transformers\", transformers.__version__)"
'
