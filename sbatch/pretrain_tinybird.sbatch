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

MY_SCRATCH="/scratch/gardnerlab/georgev"
SPEC_DIR="${MY_SCRATCH}/specs_XCM"
mkdir -p "${SPEC_DIR}" logs

RUN_NAME=tinybird_pretrain_$(date +%Y%m%d_%H%M%S)

python -u src/pretrain.py \
  --train_dir "$SPEC_DIR" \
  --val_dir   "$SPEC_DIR" \
  --run_name  "$RUN_NAME" \
  --steps 5000000 \
  --num_workers 12 \
  --amp
