#!/bin/bash
#SBATCH --job-name=finbert-build-assets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda
#SBATCH --time=00:45:00
#SBATCH --output=logs/build-assets-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false
# Requires: hpc/setup.sbatch finished, and `tools/build_assets.py download` run on the login node.
apptainer exec --nv /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
  /bin/bash -c '
set -euo pipefail
unset PYTHONPATH
.venv/bin/python tools/build_assets.py build --out assets
'
