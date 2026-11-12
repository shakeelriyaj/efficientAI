#!/bin/bash
#SBATCH --job-name=finbert-batch-exits
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/batch-exits-%j.log

source "$SLURM_SUBMIT_DIR/hpc/env.sh"
show_gpu
export EXIT_RUN="${1:?Pass the completed multi-exit run directory}"
export OUT="results/batch-exits-$SLURM_JOB_ID"
in_env '.venv/bin/python run_batch_exits.py --assets assets --exit-run "$EXIT_RUN" --out "$OUT"'
