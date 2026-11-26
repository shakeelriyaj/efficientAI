#!/bin/bash
#SBATCH --job-name=finbert-controlled
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/controlled-%j.log
source "$SLURM_SUBMIT_DIR/hpc/env.sh"
show_gpu
export SENTFIN_ASSETS=${1:?pass original SEntFiN assets}
export PHRASEBANK_ASSETS=${2:?pass PhraseBank assets}
export PHRASEBANK_EXITS=${3:?pass PhraseBank trained exits}
export CONTROLLED_OUT="results/controlled-$SLURM_JOB_ID"
export WANDB_MODE=$(cat wandb_mode.txt)
export CONTROLLED_DATASETS=${4:-"sentfin phrasebank"}
export STUDENT_SOURCE=${5:-}
in_env 'replay_args=(); if [[ -n "$STUDENT_SOURCE" ]]; then replay_args=(--student-source "$STUDENT_SOURCE"); fi; .venv/bin/python run_controlled.py --sentfin-assets "$SENTFIN_ASSETS" \
  --phrasebank-assets "$PHRASEBANK_ASSETS" --phrasebank-exits "$PHRASEBANK_EXITS" \
  --out "$CONTROLLED_OUT" --datasets $CONTROLLED_DATASETS "${replay_args[@]}"'
