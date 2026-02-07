#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

SPEC_ROOT="/media/george/George-SSD/specs"
MODE="classify"
PROBE_MODE="lora"
STEPS="1000"
BIRD_LIST_JSON="files/SFT_experiment_birds.json"
TEMP_ROOT="temp"
AUDIO_PARAMS_SOURCE="spec"
TARGET_SPECIES=""
TARGET_BIRD_ID=""

LORA_RANK_LIST=("2" "4" "8" "16" "32" "64")
LR_LIST=("5e-3" "1e-3" "5e-4" "1e-4")

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 /path/to/pretrained_run [--audio_params_source pretrain|spec] [--species SPECIES] [--bird_id ID] [--bird BIRD_OR_SPECIES]" 1>&2
    exit 1
fi

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 /path/to/pretrained_run [--audio_params_source pretrain|spec] [--species SPECIES] [--bird_id ID] [--bird BIRD_OR_SPECIES]"
    exit 0
fi

PRETRAINED_RUN=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --audio_params_source)
            AUDIO_PARAMS_SOURCE="$2"
            shift 2
            ;;
        --species)
            TARGET_SPECIES="$2"
            shift 2
            ;;
        --bird_id)
            TARGET_BIRD_ID="$2"
            shift 2
            ;;
        --bird)
            case "$2" in
                Bengalese_Finch|bengalese|bf)
                    TARGET_SPECIES="Bengalese_Finch"
                    ;;
                Zebra_Finch|zebra|zf)
                    TARGET_SPECIES="Zebra_Finch"
                    ;;
                Canary|canary)
                    TARGET_SPECIES="Canary"
                    ;;
                *)
                    TARGET_BIRD_ID="$2"
                    ;;
            esac
            shift 2
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
PRE_NAME="$(basename "$PRETRAINED_RUN")"

BIRD_COUNT=0
while IFS=: read -r SPECIES BIRD_ID; do
    if [ -n "$TARGET_SPECIES" ] && [ "$SPECIES" != "$TARGET_SPECIES" ]; then
        continue
    fi
    if [ -n "$TARGET_BIRD_ID" ] && [ "$BIRD_ID" != "$TARGET_BIRD_ID" ]; then
        continue
    fi
    BIRD_COUNT=$((BIRD_COUNT + 1))
    PREP_OUT_DIR="$TEMP_ROOT/tinybird_pool/$SPECIES/$BIRD_ID"

    TRAIN_SECONDS="32"
    if [ "$SPECIES" == "Canary" ]; then
        TRAIN_SECONDS="64"
    fi

    if ! bash shell/classify_detect_bench.sh \
        --spec_root "$SPEC_ROOT" \
        --species "$SPECIES" \
        --bird_id "$BIRD_ID" \
        --train_seconds "$TRAIN_SECONDS" \
        --mode "$MODE" \
        --audio_params_source "$AUDIO_PARAMS_SOURCE" \
        --prep_only; then
        echo "prep failed/infeasible: ${SPECIES} ${BIRD_ID} (skipping)"
        rm -rf "$PREP_OUT_DIR"
        continue
    fi

    for LORA_RANK in "${LORA_RANK_LIST[@]}"; do
        for LR in "${LR_LIST[@]}"; do
            LR_TAG="${LR//./p}"
            RUN_TAG="grid_search/${PRE_NAME}_${SPECIES}_${BIRD_ID}_r${LORA_RANK}_lr${LR_TAG}"
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
                --pretrained_run "$PRETRAINED_RUN" \
                --audio_params_source "$AUDIO_PARAMS_SOURCE" \
                --use_prepared; then
                echo "run failed: ${PRE_NAME} ${SPECIES} ${BIRD_ID} rank=${LORA_RANK} lr=${LR}"
            fi
        done
    done

    rm -rf "$PREP_OUT_DIR"
done < <(python - <<PY
import json
from pathlib import Path

path = Path("$BIRD_LIST_JSON")
data = json.loads(path.read_text(encoding="utf-8"))
for entry in data:
    species = entry.get("species", "")
    bird_id = entry.get("bird_id", "")
    if not species or not bird_id:
        continue
    print(f"{species}:{bird_id}")
PY
)

if [ "$BIRD_COUNT" -eq 0 ]; then
    if [ -n "$TARGET_SPECIES" ] || [ -n "$TARGET_BIRD_ID" ]; then
        echo "No birds matched filters: species='${TARGET_SPECIES:-*}' bird_id='${TARGET_BIRD_ID:-*}'" 1>&2
    else
        echo "No birds found in $BIRD_LIST_JSON" 1>&2
    fi
    exit 1
fi
