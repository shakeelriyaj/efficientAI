#!/bin/bash
#SBATCH --job-name=finbert-reproduce
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --time=00:30:00
#SBATCH --output=logs/reproduce-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false
# One allocated GPU for both baselines and student; never benchmark on login.
apptainer exec --nv /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
  /bin/bash -c '
set -euo pipefail
unset PYTHONPATH
.venv/bin/python run.py verify --assets assets --out "results/replay-$SLURM_JOB_ID"
.venv/bin/python run.py train --assets assets --out "results/retrain-$SLURM_JOB_ID"
'
