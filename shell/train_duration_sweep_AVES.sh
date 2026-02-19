#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

SPEC_ROOT="/media/george-vengrovski/disk2/specs"
WAV_ROOT="/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae"
MODE="classify"
PROBE_MODE="finetune"
LR="5e-5"
STEPS="1000"
BIRD_LIST_JSON="files/SFT_experiment_birds.json"
TEMP_ROOT="temp"
RUN_TAG_PREFIX="aves_duration_sweep"
RESULTS_DIR="results/aves_duration_sweep"

usage() {
    echo "Usage: $0 [Bengalese_Finch|Zebra_Finch|Canary] [--unit_detection]" 1>&2
}

TARGET_SPECIES=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --unit_detection)
            MODE="unit_detect"
            RUN_TAG_PREFIX="aves_duration_sweep_unit_detect"
            RESULTS_DIR="results/aves_duration_sweep_unit_detect"
            shift
            ;;
        Bengalese_Finch|bengalese|bf)
            TARGET_SPECIES="Bengalese_Finch"
            shift
            ;;
        Zebra_Finch|zebra|zf)
            TARGET_SPECIES="Zebra_Finch"
            shift
            ;;
        Canary|canary)
            TARGET_SPECIES="Canary"
            shift
            ;;
        *)
            echo "Unknown arg: $1" 1>&2
            usage
            exit 1
            ;;
    esac
done

if [ ! -f "$BIRD_LIST_JSON" ]; then
    echo "Bird list JSON not found: $BIRD_LIST_JSON" 1>&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"

LR_TAG="${LR//./p}"
BIRD_COUNT=0
RUN_ATTEMPTS=0

while IFS=: read -r SPECIES BIRD_ID; do
    if [ -n "$TARGET_SPECIES" ] && [ "$SPECIES" != "$TARGET_SPECIES" ]; then
        continue
    fi

    PREP_OUT_DIR="$TEMP_ROOT/aves_pool/$SPECIES/$BIRD_ID"

    if [ "$MODE" == "unit_detect" ]; then
        TRAIN_SECONDS_LIST=("8" "16" "32" "64" "128" "256" "MAX")
    elif [ "$SPECIES" == "Canary" ]; then
        TRAIN_SECONDS_LIST=("32" "64" "128" "256" "512" "MAX")
    elif [ "$SPECIES" == "Bengalese_Finch" ] || [ "$SPECIES" == "Zebra_Finch" ]; then
        TRAIN_SECONDS_LIST=("16" "32" "64" "128" "256" "MAX")
    else
        continue
    fi

    BIRD_COUNT=$((BIRD_COUNT + 1))

    for TRAIN_SECONDS in "${TRAIN_SECONDS_LIST[@]}"; do
        TRAIN_SECONDS_TAG="${TRAIN_SECONDS//./p}"
        RUN_TAG="${RUN_TAG_PREFIX}/${SPECIES}_${BIRD_ID}_t${TRAIN_SECONDS_TAG}_lr${LR_TAG}"

        rm -rf "$PREP_OUT_DIR"

        if ! bash shell/train_AVES.sh \
            --spec_root "$SPEC_ROOT" \
            --wav_root "$WAV_ROOT" \
            --temp_root "$TEMP_ROOT" \
            --species "$SPECIES" \
            --bird_id "$BIRD_ID" \
            --train_seconds "$TRAIN_SECONDS" \
            --mode "$MODE" \
            --prep_only; then
            echo "prep failed/infeasible: ${SPECIES} ${BIRD_ID} train_seconds=${TRAIN_SECONDS} (skipping)"
            rm -rf "$PREP_OUT_DIR"
            continue
        fi

        # Require per-bird filtered annotations so class counts stay individual-specific.
        if [ ! -f "$PREP_OUT_DIR/pool/annotations_filtered.json" ]; then
            echo "prep missing filtered annotations: ${SPECIES} ${BIRD_ID} train_seconds=${TRAIN_SECONDS} (skipping)"
            rm -rf "$PREP_OUT_DIR"
            continue
        fi

        RUN_ATTEMPTS=$((RUN_ATTEMPTS + 1))
        if ! bash shell/train_AVES.sh \
            --spec_root "$SPEC_ROOT" \
            --wav_root "$WAV_ROOT" \
            --temp_root "$TEMP_ROOT" \
            --species "$SPECIES" \
            --bird_id "$BIRD_ID" \
            --train_seconds "$TRAIN_SECONDS" \
            --mode "$MODE" \
            --probe_mode "$PROBE_MODE" \
            --lr "$LR" \
            --steps "$STEPS" \
            --run_tag "$RUN_TAG" \
            --no-save_intermediate_checkpoints \
            --use_prepared; then
            echo "run failed: ${SPECIES} ${BIRD_ID} train_seconds=${TRAIN_SECONDS}"
        else
            python "$PROJECT_ROOT/scripts/eval/eval_val_outputs_f1.py" \
                --runs_root "$PROJECT_ROOT/runs" \
                --run_names "$RUN_TAG" \
                --out_csv "$RESULTS_DIR/eval_f1.csv" \
                --append \
                --no_summary
        fi

        rm -rf "$PREP_OUT_DIR"
    done

    rm -rf "$PREP_OUT_DIR"
done < <(python - <<PY
import json
from pathlib import Path

data = json.loads(Path("$BIRD_LIST_JSON").read_text(encoding="utf-8"))
for entry in data:
    species = entry.get("species", "")
    bird_id = entry.get("bird_id", "")
    if species and bird_id:
        print(f"{species}:{bird_id}")
PY
)

if [ "$BIRD_COUNT" -eq 0 ]; then
    if [ -n "$TARGET_SPECIES" ]; then
        echo "No birds found for species: $TARGET_SPECIES" 1>&2
    else
        echo "No birds found in $BIRD_LIST_JSON" 1>&2
    fi
    exit 1
fi

if [ "$RUN_ATTEMPTS" -eq 0 ]; then
    echo "No feasible AVES duration runs were launched." 1>&2
    exit 1
fi
