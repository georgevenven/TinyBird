#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

SPEC_ROOT="/media/george/George-SSD/specs"
MODE="classify"
PROBE_MODE="lora"
LORA_RANK="32"
LR="1e-3"
STEPS="1000"
BIRD_LIST_JSON="files/SFT_experiment_birds.json"
TEMP_ROOT="temp"
RUN_TAG_PREFIX="duration_sweep"

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 /path/to/pretrained_run [Bengalese_Finch|Zebra_Finch|Canary] [--unit_detection]" 1>&2
    exit 1
fi
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 /path/to/pretrained_run [Bengalese_Finch|Zebra_Finch|Canary] [--unit_detection]"
    exit 0
fi

PRETRAINED_RUN=""
TARGET_SPECIES=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --unit_detection)
            MODE="unit_detect"
            RUN_TAG_PREFIX="duration_sweep_unit_detect"
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
            if [ -z "$PRETRAINED_RUN" ]; then
                PRETRAINED_RUN="$1"
                shift
            else
                echo "Unknown arg: $1" 1>&2
                exit 1
            fi
            ;;
    esac
done

if [ -z "$PRETRAINED_RUN" ]; then
    echo "Missing /path/to/pretrained_run" 1>&2
    exit 1
fi

if [ -d "runs/$PRETRAINED_RUN" ]; then
    PRETRAINED_RUN="runs/$PRETRAINED_RUN"
fi
if [ ! -d "$PRETRAINED_RUN" ]; then
    echo "Pretrained run not found: $PRETRAINED_RUN" 1>&2
    exit 1
fi
PRE_NAME="$(basename "$PRETRAINED_RUN")"

if [ ! -f "$BIRD_LIST_JSON" ]; then
    echo "Bird list JSON not found: $BIRD_LIST_JSON" 1>&2
    exit 1
fi

LR_TAG="${LR//./p}"
BIRD_COUNT=0

while IFS=: read -r SPECIES BIRD_ID; do
    if [ -n "$TARGET_SPECIES" ] && [ "$SPECIES" != "$TARGET_SPECIES" ]; then
        continue
    fi
    PREP_OUT_DIR="$TEMP_ROOT/tinybird_pool/$SPECIES/$BIRD_ID"

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
        RUN_TAG="${RUN_TAG_PREFIX}/${PRE_NAME}_${SPECIES}_${BIRD_ID}_t${TRAIN_SECONDS_TAG}_r${LORA_RANK}_lr${LR_TAG}"
        rm -rf "$PREP_OUT_DIR"

        if ! bash shell/classify_detect_bench.sh \
            --spec_root "$SPEC_ROOT" \
            --species "$SPECIES" \
            --bird_id "$BIRD_ID" \
            --train_seconds "$TRAIN_SECONDS" \
            --mode "$MODE" \
            --probe_mode "$PROBE_MODE" \
            --lora_rank "$LORA_RANK" \
            --lr "$LR" \
            --steps "$STEPS" \
            --run_tag "$RUN_TAG" \
            --pretrained_run "$PRETRAINED_RUN"; then
            echo "run failed: ${PRE_NAME} ${SPECIES} ${BIRD_ID} train_seconds=${TRAIN_SECONDS}"
        fi
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
