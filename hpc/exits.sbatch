#!/bin/bash
#SBATCH --job-name=finbert-exits
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --time=00:45:00
#SBATCH --output=logs/exits-%j.log
# Train exit heads at layers 2/4/8, select policies on validation, test once.
# Pass a seed as the first argument to repeat with another head-training seed.
source "$SLURM_SUBMIT_DIR/hpc/env.sh"
show_gpu
export SEED_ARG=${1:+--seed $1}
export OUT="results/exits-${1:-default}-$SLURM_JOB_ID"
# A requeued job restarts cleanly instead of failing on its partial folder.
rm -rf "$OUT"
in_env '.venv/bin/python run_exits.py --assets assets --out "$OUT" $SEED_ARG'
