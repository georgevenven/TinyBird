#!/bin/bash

# Exit on any error
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Navigate to project root (one level up from shell/)
cd "$SCRIPT_DIR/.."

echo "Starting training chain..."

python src/pretrain.py \
    --train_dir "" \
    --val_dir "" \
    --amp \
    --run_name "Noise_1_No_Norm" \
    --batch_size 128 \
    --patch_height 32 \
    --patch_width 1 \
    --steps 100_000 \
    --num_timebins 1000 \
    --lr 1e-4 \
    --mask_p 0.5 \
    --mask_c 0.1

echo "Train 1 completed."

python src/pretrain.py \
    --train_dir "" \
    --val_dir "" \
    --amp \
    --run_name "Noise_0_No_Norm" \
    --batch_size 128 \
    --patch_height 32 \
    --patch_width 1 \
    --steps 100_000 \
    --num_timebins 1000 \
    --lr 1e-4 \
    --mask_p 0.5 \
    --mask_c 0.1

echo "Train 2 completed."

# Canary Training 1
python src/pretrain.py \
    --train_dir "" \
    --val_dir "" \
    --amp \
    --run_name "Noise_1_Norm" \
    --batch_size 128 \
    --patch_height 32 \
    --patch_width 1 \
    --steps 100_000 \
    --num_timebins 1000 \
    --lr 1e-4 \
    --mask_p 0.5 \
    --mask_c 0.1

echo "Train 3 completed."

python src/pretrain.py \
    --train_dir "" \
    --val_dir "" \
    --amp \
    --run_name "Noise_0_Norm" \
    --batch_size 128 \
    --patch_height 32 \
    --patch_width 1 \
    --steps 100_000 \
    --num_timebins 1000 \
    --lr 1e-4 \
    --mask_p 0.5 \
    --mask_c 0.1

echo "Train 4 completed."

echo "All training runs completed!"
