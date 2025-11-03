#!/usr/bin/env bash
#
# Run per-channel clustering, sync into the global registry, and reclassify noise blocks.
# Can be invoked from any directory; assumes the repo lives at ~/code/BirdCallAuth/TinyBird.

set -euo pipefail

readonly REPO_DIR="$HOME/code/BirdCallAuth/TinyBird"
readonly TRAIN_AUDIO="/mnt/birdconv/tb_conv_data"
readonly VAL_AUDIO="/mnt/birdconv/tb_conv_data_val"
readonly TRAIN_OUT="/mnt/birdconv/tb_conv_cluster"
readonly VAL_OUT="/mnt/birdconv/tb_conv_cluster_val"
readonly REGISTRY_DB="$HOME/code/BirdCallAuth/global_clusters.sqlite"
readonly MATCH_THRESHOLD="0.35"

echo "==> Switching to repository: ${REPO_DIR}"
cd "${REPO_DIR}"

echo "==> Step 1: per-channel clustering (training set)"
# uv run python src/audio2cluster.py \
#   --src_dir "${TRAIN_AUDIO}" \
#   --dst_dir "${TRAIN_OUT}"

echo "==> Step 1: per-channel clustering (validation set)"
# uv run python src/audio2cluster.py \
#   --src_dir "${VAL_AUDIO}" \
#   --dst_dir "${VAL_OUT}"

echo "==> Step 2: sync training clusters into global registry"
uv run python src/cluster_regsitry.y sync \
  --registry "${REGISTRY_DB}" \
  --clusters_dir "${TRAIN_OUT}/cluster" \
  --split train \
  --match-threshold "${MATCH_THRESHOLD}"

echo "==> Step 2: sync validation clusters into global registry"
uv run python src/cluster_regsitry.py sync \
  --registry "${REGISTRY_DB}" \
  --clusters_dir "${VAL_OUT}/cluster" \
  --split validation \
  --match-threshold "${MATCH_THRESHOLD}"

echo "==> Step 3: reclassify noise blocks (training set)"
uv run python src/cluster_regsitry.py reclassify \
  --registry "${REGISTRY_DB}" \
  --clusters_dir "${TRAIN_OUT}/cluster" \
  --match-threshold "${MATCH_THRESHOLD}"

echo "==> Step 3: reclassify noise blocks (validation set)"
uv run python src/cluster_regsitry.py reclassify \
  --registry "${REGISTRY_DB}" \
  --clusters_dir "${VAL_OUT}/cluster" \
  --match-threshold "${MATCH_THRESHOLD}"

echo "==> Registry summary"
uv run python src/cluster_regsitry.py info \
  --registry "${REGISTRY_DB}"

echo "==> Pipeline complete."

