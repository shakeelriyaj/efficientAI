#!/bin/bash
#SBATCH --job-name=phrasebank-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/phrasebank-%j.log
source "$SLURM_SUBMIT_DIR/hpc/env.sh"
show_gpu
export PHRASEBANK_ASSETS=${1:?pass the PhraseBank assets directory}
export PHRASEBANK_EXIT_RUN=${2:?pass the original exit-run directory}
in_env '
.venv/bin/python run_phrasebank.py --assets "$PHRASEBANK_ASSETS" --exit-run "$PHRASEBANK_EXIT_RUN" \
    --frozen-policy reference/phrasebank_frozen_policy.json \
    --out "results/phrasebank-$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}" --device cuda
'
