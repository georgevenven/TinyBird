#!/bin/bash

set -euo pipefail

SPEC_ROOT="/media/george/George-SSD/specs"
TRAIN_SECONDS="MAX"
MODE="classify"
PROBE_MODE="linear"
LR="1e-2"
BIRD_LIST_JSON="files/SFT_experiment_birds.json"
TEMP_ROOT="temp"

PRETRAINED_RUNS=("$@")
if [ "${#PRETRAINED_RUNS[@]}" -eq 0 ]; then
    exit 1
fi

while IFS=: read -r SPECIES BIRD_ID; do
    PREP_OUT_DIR="$TEMP_ROOT/tinybird_pool/$SPECIES/$BIRD_ID"
    bash shell/classify_detect_bench.sh \
        --spec_root "$SPEC_ROOT" \
        --species "$SPECIES" \
        --bird_id "$BIRD_ID" \
        --train_seconds "$TRAIN_SECONDS" \
        --mode "$MODE" \
        --prep_only

    for PRETRAINED_RUN in "${PRETRAINED_RUNS[@]}"; do
        PRE_NAME="$(basename "$PRETRAINED_RUN")"
        RUN_TAG="linear_probes/linear_probe_${PRE_NAME}_${SPECIES}_${BIRD_ID}"
        if ! bash shell/classify_detect_bench.sh \
            --spec_root "$SPEC_ROOT" \
            --species "$SPECIES" \
            --bird_id "$BIRD_ID" \
            --train_seconds "$TRAIN_SECONDS" \
            --mode "$MODE" \
            --probe_mode "$PROBE_MODE" \
            --lr "$LR" \
            --run_tag "$RUN_TAG" \
            --pretrained_run "$PRETRAINED_RUN" \
            --use_prepared; then
            echo "run failed: ${PRETRAINED_RUN} ${SPECIES} ${BIRD_ID}"
        fi
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
