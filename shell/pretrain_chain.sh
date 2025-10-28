#!/bin/bash

# Exit on any error
set -e

echo "Starting training chain..."

# Training command 1
python src/pretrain.py \
    --train_dir "/media/george-vengrovski/disk2/specs/zf_specs_64hop" \
    --val_dir "/media/george-vengrovski/disk2/specs/zf_specs_64hop" \
    --amp \
    --run_name "ZF_32_1" \
    --batch_size 32 \
    --patch_height 32 \
    --patch_width 1 \
    --steps 100_000 \
    --num_timebins 1000 \
    --lr 1e-4

echo "Training 1 completed. Starting training 2..."

python src/pretrain.py \
    --train_dir "/media/george-vengrovski/disk2/specs/zf_specs_64hop" \
    --val_dir "/media/george-vengrovski/disk2/specs/zf_specs_64hop" \
    --amp \
    --run_name "ZF_32_2" \
    --batch_size 32 \
    --patch_height 32 \
    --patch_width 1 \
    --steps 100_000 \
    --num_timebins 1000 \
    --lr 1e-4

python src/pretrain.py \
    --train_dir "/media/george-vengrovski/disk2/specs/zf_specs_64hop" \
    --val_dir "/media/george-vengrovski/disk2/specs/zf_specs_64hop" \
    --amp \
    --run_name "ZF_32_3" \
    --batch_size 32 \
    --patch_height 32 \
    --patch_width 1 \
    --steps 100_000 \
    --num_timebins 1000 \
    --lr 1e-4

echo "All training runs completed!"

