#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

SPEC_ROOT="/media/george/George-SSD/specs"
TRAIN_SECONDS="100000"
MODE="classify"
PROBE_MODE="lora"
STEPS="1000"
BIRD_LIST_JSON="files/SFT_experiment_birds.json"
TEMP_ROOT="temp"

LORA_RANK_LIST=("1" "2" "4" "8" "16" "32" "64")
LR_LIST=("5e-3" "1e-3" "5e-4" "1e-4")

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/pretrained_run" 1>&2
    exit 1
fi

PRETRAINED_RUN="$1"
PRE_NAME="$(basename "$PRETRAINED_RUN")"

while IFS=: read -r SPECIES BIRD_ID; do
    PREP_OUT_DIR="$TEMP_ROOT/tinybird_pool/$SPECIES/$BIRD_ID"

    bash shell/classify_detect_bench.sh \
        --spec_root "$SPEC_ROOT" \
        --species "$SPECIES" \
        --bird_id "$BIRD_ID" \
        --train_seconds "$TRAIN_SECONDS" \
        --mode "$MODE" \
        --prep_only

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
