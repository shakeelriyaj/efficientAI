#!/bin/bash
#SBATCH --job-name=finbert-seeds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --time=00:30:00
#SBATCH --array=0-9
#SBATCH --output=logs/seeds-%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false
# Fixed before any results are seen. Report every seed; never drop a bad one.
# 4142 repeats the original run as a consistency check.
SEEDS=(4142 1 2 3 4 5 6 7 8 9)
if (( SLURM_ARRAY_TASK_ID >= ${#SEEDS[@]} )); then
  echo "Array index $SLURM_ARRAY_TASK_ID has no seed; adjust --array" >&2
  exit 1
fi
export SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
export OUT="results/seeds-$SLURM_ARRAY_JOB_ID/seed-$SEED"
# Each task trains on its own GPU and times the student against the full model
# on that same GPU, so latency ratios stay valid across different nodes.
# A requeued task restarts cleanly instead of failing on its partial folder.
rm -rf "$OUT"
apptainer exec --nv /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
  /bin/bash -c '
set -euo pipefail
unset PYTHONPATH
.venv/bin/python run.py train --seed "$SEED" --assets assets --out "$OUT"
'
