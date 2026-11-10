# Shared by the Slurm scripts; works on NYU Torch and NYU Cloud Bursting.
# On Torch, commands run inside NYU's CUDA container. Where that image or
# apptainer is missing (Cloud Bursting), they run directly on the node; the
# pip-installed PyTorch brings its own CUDA libraries and needs only the driver.
# Account, partition, and GPU type are chosen on the sbatch command line.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false
FINBERT_IMAGE=${FINBERT_IMAGE:-/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif}

in_env() {
  if command -v apptainer >/dev/null 2>&1 && [[ -f "$FINBERT_IMAGE" ]]; then
    apptainer exec --nv "$FINBERT_IMAGE" /bin/bash -c "set -euo pipefail; unset PYTHONPATH; $1"
  else
    /bin/bash -c "set -euo pipefail; unset PYTHONPATH; $1"
  fi
}

show_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
}
