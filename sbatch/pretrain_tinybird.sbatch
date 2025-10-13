#!/bin/bash
#SBATCH -J tinybird
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-80gb,no-mig
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH -o logs/tinybird-pretrain-%j.out
#SBATCH -e logs/tinybird-pretrain-%j.err

set -euo pipefail

module load miniconda3/20240410
eval "$(conda shell.bash hook)"
conda activate ~/.conda/envs/tinybird

# CPU / CUDA runtime knobs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "$SLURM_SUBMIT_DIR"

echo "Node: $(hostname)"
nvidia-smi || true

# Quick sanity check before training
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("GPU 0:", torch.cuda.get_device_name(0))
PY

PIRG=$(id -nG | tr ' ' '\n' | sed -n 's/^is\.racs\.pirg\.//p' | head -n1)
MY_SCRATCH="/scratch/${PIRG}/${USER}"
SPEC_DIR="${MY_SCRATCH}/specs"
mkdir -p "${SPEC_DIR}" logs

RUN_NAME=tinybird_pretrain_$(date +%Y%m%d_%H%M%S)

python -u src/pretrain.py \
  --train_dir "$SPEC_DIR" \
  --val_dir   "$SPEC_DIR" \
  --run_name  "$RUN_NAME" \
  --steps 5000000 \
  --num_workers 12 \
  --amp
