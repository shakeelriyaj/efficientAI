#!/bin/bash
#SBATCH --job-name=finbert-build-assets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=logs/build-assets-%j.log
# Requires: hpc/setup.sbatch finished and `tools/build_assets.py download` already run.
source "$SLURM_SUBMIT_DIR/hpc/env.sh"
show_gpu
in_env '.venv/bin/python tools/build_assets.py build --out assets'
