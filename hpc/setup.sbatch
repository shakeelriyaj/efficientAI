#!/bin/bash
#SBATCH --job-name=finbert-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=logs/setup-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Prepare software on a CPU compute node, not the login node.
apptainer exec --overlay /share/apps/images/anaconda3-2024.10-1.sqf:ro \
  /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif /bin/bash -c '
set -euo pipefail
export PATH=/ext3/anaconda3/2024.10-1/bin:$PATH
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
python -m pip install --quiet --target .bootstrap uv
export PYTHONPATH="$PWD/.bootstrap"
python -m uv venv --python 3.13.16 .venv
python -m uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pip --version 2>/dev/null || true
'
