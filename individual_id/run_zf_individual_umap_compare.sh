#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

SPEC_ROOT="/media/george-vengrovski/disk2/specs"
SPEC_DIR="$SPEC_ROOT/zf_64hop_32khz"
ANNOTATION_JSON="$PROJECT_ROOT/files/zf_annotations.json"
OUT_DIR="$PROJECT_ROOT/results/individual_id_umap/zebra_finch"

SONGMAE_RUN=""
CHECKPOINT=""
SONGS_PER_BIRD="5"
MAX_BIRDS="0"
SEED="42"
POOL_WINDOWS="1,8,32,128"
POOL_MODE="mean"
POOL_HOP_RATIO="0.5"
SPEC_WINDOW_BINS="32"
SPEC_WINDOW_HOP_BINS="16"
MAX_POINTS_PER_BIRD="1500"
NUM_TIMEBINS="5000000"
UMAP_NEIGHBORS="100"
UMAP_MIN_DIST="0.1"
UMAP_METRIC="cosine"
DETERMINISTIC=0
FORCE_REEXTRACT=0
SPEC_BASELINE=1
COPY_MODE="symlink"

usage() {
    cat <<EOF_USAGE
Usage: $0 --songmae_run RUN [options]

Build Zebra Finch individual-ID UMAP comparisons from 5 recordings per bird:
  1) SongMAE embeddings with mean pooling at multiple window sizes
  2) Raw spectrogram-window baseline

Required:
  --songmae_run PATH_OR_RUN_NAME  SongMAE run directory/path/name

Optional:
  --spec_dir PATH                 Default: ${SPEC_DIR}
  --annotation_json PATH          Default: ${ANNOTATION_JSON}
  --out_dir PATH                  Default: ${OUT_DIR}
  --checkpoint FILE               Optional checkpoint file
  --songs_per_bird N              Default: ${SONGS_PER_BIRD}
  --max_birds N                   Default: ${MAX_BIRDS} (0=all birds)
  --seed N                        Default: ${SEED}
  --pool_windows CSV              Default: ${POOL_WINDOWS}
  --pool_mode mean|max|sum        Default: ${POOL_MODE}
  --pool_hop_ratio FLOAT          Default: ${POOL_HOP_RATIO}
  --spec_window_bins N            Default: ${SPEC_WINDOW_BINS}
  --spec_window_hop_bins N        Default: ${SPEC_WINDOW_HOP_BINS}
  --max_points_per_bird N         Default: ${MAX_POINTS_PER_BIRD}
  --num_timebins N                Default: ${NUM_TIMEBINS}
  --umap_neighbors N              Default: ${UMAP_NEIGHBORS}
  --umap_min_dist FLOAT           Default: ${UMAP_MIN_DIST}
  --umap_metric NAME              Default: ${UMAP_METRIC}
  --copy_mode symlink|copy        Default: ${COPY_MODE}
  --no_spec_baseline              Skip spectrogram-window baseline UMAP
  --deterministic                 Deterministic UMAP
  --force_reextract               Recompute per-bird embeddings even if NPZ exists
  -h, --help                      Show this help
EOF_USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --songmae_run)
            SONGMAE_RUN="$2"
            shift 2
            ;;
        --spec_dir)
            SPEC_DIR="$2"
            shift 2
            ;;
        --annotation_json)
            ANNOTATION_JSON="$2"
            shift 2
            ;;
        --out_dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --songs_per_bird)
            SONGS_PER_BIRD="$2"
            shift 2
            ;;
        --max_birds)
            MAX_BIRDS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --pool_windows)
            POOL_WINDOWS="$2"
            shift 2
            ;;
        --pool_mode)
            POOL_MODE="$2"
            shift 2
            ;;
        --pool_hop_ratio)
            POOL_HOP_RATIO="$2"
            shift 2
            ;;
        --spec_window_bins)
            SPEC_WINDOW_BINS="$2"
            shift 2
            ;;
        --spec_window_hop_bins)
            SPEC_WINDOW_HOP_BINS="$2"
            shift 2
            ;;
        --max_points_per_bird)
            MAX_POINTS_PER_BIRD="$2"
            shift 2
            ;;
        --num_timebins)
            NUM_TIMEBINS="$2"
            shift 2
            ;;
        --umap_neighbors)
            UMAP_NEIGHBORS="$2"
            shift 2
            ;;
        --umap_min_dist)
            UMAP_MIN_DIST="$2"
            shift 2
            ;;
        --umap_metric)
            UMAP_METRIC="$2"
            shift 2
            ;;
        --copy_mode)
            COPY_MODE="$2"
            shift 2
            ;;
        --no_spec_baseline)
            SPEC_BASELINE=0
            shift
            ;;
        --deterministic)
            DETERMINISTIC=1
            shift
            ;;
        --force_reextract)
            FORCE_REEXTRACT=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" 1>&2
            usage
            exit 1
            ;;
    esac
done

if [ -z "$SONGMAE_RUN" ]; then
    echo "Missing required --songmae_run" 1>&2
    usage
    exit 1
fi

mkdir -p "$OUT_DIR"

CMD=(
    python "$PROJECT_ROOT/scripts/eval/individual_umap_compare.py"
    --species "Zebra_Finch"
    --annotation_json "$ANNOTATION_JSON"
    --spec_dir "$SPEC_DIR"
    --run_dir "$SONGMAE_RUN"
    --out_dir "$OUT_DIR"
    --songs_per_bird "$SONGS_PER_BIRD"
    --max_birds "$MAX_BIRDS"
    --seed "$SEED"
    --num_timebins "$NUM_TIMEBINS"
    --pool_windows "$POOL_WINDOWS"
    --pool_mode "$POOL_MODE"
    --pool_hop_ratio "$POOL_HOP_RATIO"
    --spec_window_bins "$SPEC_WINDOW_BINS"
    --spec_window_hop_bins "$SPEC_WINDOW_HOP_BINS"
    --max_points_per_bird "$MAX_POINTS_PER_BIRD"
    --umap_neighbors "$UMAP_NEIGHBORS"
    --umap_min_dist "$UMAP_MIN_DIST"
    --umap_metric "$UMAP_METRIC"
    --copy_mode "$COPY_MODE"
)

if [ -n "$CHECKPOINT" ]; then
    CMD+=(--checkpoint "$CHECKPOINT")
fi
if [ "$DETERMINISTIC" -eq 1 ]; then
    CMD+=(--deterministic)
fi
if [ "$FORCE_REEXTRACT" -eq 1 ]; then
    CMD+=(--force_reextract)
fi
if [ "$SPEC_BASELINE" -eq 0 ]; then
    CMD+=(--no_spec_baseline)
fi

"${CMD[@]}"

echo "Finished. Outputs in: $OUT_DIR"
