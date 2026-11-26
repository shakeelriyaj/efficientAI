#!/bin/bash
#SBATCH --job-name=finbert-controlled-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=logs/controlled-setup-%j.log
source "$SLURM_SUBMIT_DIR/hpc/env.sh"
export UV=$(find /scratch/wk2376/efficient-ai-repro-20261105/.bootstrap -type f -name uv | head -1)
export UV_CACHE_DIR=/scratch/wk2376/efficient-ai-repro-20261105/.cache/uv
export BASEPY=/scratch/wk2376/efficient-ai-repro-20261105/.venv/bin/python
in_env '
"$UV" venv --python "$BASEPY" .venv
"$UV" pip install --python .venv/bin/python -r requirements-controlled.txt pytest==8.4.2
.venv/bin/python -m pytest tests/test_controlled.py tests/test_core.py -q
.venv/bin/python preflight_controlled.py
'
