#!/bin/bash
#SBATCH -J tinybird
#SBATCH -p gpulong
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-80gb,no-mig
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH -o logs/tinybird-pretrain-%j.out
#SBATCH -e logs/tinybird-pretrain-%j.err

set -euo pipefail

module load miniconda3/20240410
eval "$(conda shell.bash hook)"
conda activate ~/.conda/envs/tinybird
pip install -r "$HOME/TinyBird/requirements.txt"

# CPU / CUDA runtime knobs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "$SLURM_SUBMIT_DIR"

MY_SCRATCH="/scratch/gardnerlab/georgev"
SPEC_DIR="${MY_SCRATCH}/XCM_Bambird"
TRAIN_DIR="${MY_SCRATCH}/XCM_Train"
VAL_DIR="${MY_SCRATCH}/XCM_Test"
mkdir -p "${SPEC_DIR}" logs

RUN_NAME="xcm_voronoi_mask_no_normalize_128h_1w"

python -u src/pretrain.py \
  --train_dir "$TRAIN_DIR" \
  --val_dir "$VAL_DIR" \
  --run_name "$RUN_NAME" \
  --steps 500000 \
  --batch_size 256 \
  --num_workers 48 \
  --lr 0.0003 \
  --dropout 0.1 \
  --mask_p 0.75 \
  --mask_c 0.1 \
  --mask_type voronoi \
  --eval_every 500 \
  --warmup_steps 1000 \
  --min_lr 0.00001 \
  --weight_decay 0.1 \
  --patch_height 128 \
  --patch_width 1 \
  --num_timebins 1024 \
  --enc_hidden_d 384 \
  --enc_n_head 6 \
  --enc_n_layer 6 \
  --enc_dim_ff 1536 \
  --dec_hidden_d 192 \
  --dec_n_head 6 \
  --dec_n_layer 2 \
  --dec_dim_ff 768 \
  --amp \
  --no_normalize_patches \
  --wandb
