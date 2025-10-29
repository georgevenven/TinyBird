#!/bin/bash

# Exit on any error
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Navigate to project root (one level up from shell/)
cd "$SCRIPT_DIR/.."

echo "Starting training chain..."

# # ZF Training 1
# python src/pretrain.py \
#     --train_dir "/media/george-vengrovski/disk2/specs/zf_64hop_32khz_train" \
#     --val_dir "/media/george-vengrovski/disk2/specs/zf_64hop_32khz_val" \
#     --amp \
#     --run_name "ZF_128_1" \
#     --batch_size 42 \
#     --patch_height 128 \
#     --patch_width 1 \
#     --steps 50_000 \
#     --num_timebins 1000 \
#     --lr 1e-4 \
#     --mask_p 0.25 \
#     --mask_c 0.01

# echo "ZF Training 1 completed. Starting ZF training 2..."

# python src/pretrain.py \
#     --train_dir "/media/george-vengrovski/disk2/specs/zf_64hop_32khz_train" \
#     --val_dir "/media/george-vengrovski/disk2/specs/zf_64hop_32khz_val" \
#     --amp \
#     --run_name "ZF_128_10" \
#     --batch_size 42 \
#     --patch_height 128 \
#     --patch_width 10 \
#     --steps 50_000 \
#     --num_timebins 1000 \
#     --lr 1e-4 \
#     --mask_p 0.25 \
#     --mask_c 0.01

echo "ZF Training 2 completed. Starting ZF training 3..."

python src/pretrain.py \
    --train_dir "/media/george-vengrovski/disk2/specs/zf_64hop_32khz_train" \
    --val_dir "/media/george-vengrovski/disk2/specs/zf_64hop_32khz_val" \
    --amp \
    --run_name "ZF_32_1" \
    --batch_size 42 \
    --patch_height 32 \
    --patch_width 2 \
    --steps 50_000 \
    --num_timebins 1000 \
    --lr 1e-4 \
    --mask_p 0.25 \
    --mask_c 0.01

echo "ZF Training 3 completed. Starting ZF training 4..."

python src/pretrain.py \
    --train_dir "/media/george-vengrovski/disk2/specs/zf_64hop_32khz_train" \
    --val_dir "/media/george-vengrovski/disk2/specs/zf_64hop_32khz_val" \
    --amp \
    --run_name "ZF_32_10" \
    --batch_size 42 \
    --patch_height 32 \
    --patch_width 10 \
    --steps 50_000 \
    --num_timebins 1000 \
    --lr 1e-4 \
    --mask_p 0.25 \
    --mask_c 0.01

echo "All ZF training runs completed! Starting Canary training..."

# Canary Training 1
python src/pretrain.py \
    --train_dir "/media/george-vengrovski/disk2/specs/canary_64hop_32khz_train" \
    --val_dir "/media/george-vengrovski/disk2/specs/canary_64hop_32khz_test" \
    --amp \
    --run_name "Canary_128_1" \
    --batch_size 42 \
    --patch_height 128 \
    --patch_width 1 \
    --steps 50_000 \
    --num_timebins 1000 \
    --lr 1e-4 \
    --mask_p 0.25 \
    --mask_c 0.01

echo "Canary Training 1 completed. Starting Canary training 2..."

python src/pretrain.py \
    --train_dir "/media/george-vengrovski/disk2/specs/canary_64hop_32khz_train" \
    --val_dir "/media/george-vengrovski/disk2/specs/canary_64hop_32khz_test" \
    --amp \
    --run_name "Canary_128_10" \
    --batch_size 42 \
    --patch_height 128 \
    --patch_width 10 \
    --steps 50_000 \
    --num_timebins 1000 \
    --lr 1e-4 \
    --mask_p 0.25 \
    --mask_c 0.01

echo "Canary Training 2 completed. Starting Canary training 3..."

python src/pretrain.py \
    --train_dir "/media/george-vengrovski/disk2/specs/canary_64hop_32khz_train" \
    --val_dir "/media/george-vengrovski/disk2/specs/canary_64hop_32khz_test" \
    --amp \
    --run_name "Canary_32_1" \
    --batch_size 42 \
    --patch_height 32 \
    --patch_width 2 \
    --steps 50_000 \
    --num_timebins 1000 \
    --lr 1e-4 \
    --mask_p 0.25 \
    --mask_c 0.01

echo "Canary Training 3 completed. Starting Canary training 4..."

python src/pretrain.py \
    --train_dir "/media/george-vengrovski/disk2/specs/canary_64hop_32khz_train" \
    --val_dir "/media/george-vengrovski/disk2/specs/canary_64hop_32khz_test" \
    --amp \
    --run_name "Canary_32_10" \
    --batch_size 42 \
    --patch_height 32 \
    --patch_width 10 \
    --steps 50_000 \
    --num_timebins 1000 \
    --lr 1e-4 \
    --mask_p 0.25 \
    --mask_c 0.01

echo "All training runs completed!"

