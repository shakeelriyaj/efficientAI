#!/bin/bash
#SBATCH --job-name=finbert-reproduce
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --time=00:30:00
#SBATCH --output=logs/reproduce-%j.log
# Requires the original Colab assets. One allocated GPU for all routes.
source "$SLURM_SUBMIT_DIR/hpc/env.sh"
show_gpu
in_env '
.venv/bin/python run.py verify --assets assets --out "results/replay-$SLURM_JOB_ID"
.venv/bin/python run.py train --assets assets --out "results/retrain-$SLURM_JOB_ID"
'
