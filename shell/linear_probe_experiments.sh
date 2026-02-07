#!/bin/bash

set -euo pipefail

SPEC_ROOT="/media/george/George-SSD/specs"
TRAIN_SECONDS="MAX"
MODE="classify"
PROBE_MODE="linear"
LR="1e-2"
BIRD_LIST_JSON="files/SFT_experiment_birds.json"
TEMP_ROOT="temp"
AUDIO_PARAMS_SOURCE="spec"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 /path/to/pretrained_run [/path/to/pretrained_run ...] [--audio_params_source pretrain|spec]" 1>&2
    exit 1
fi
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 /path/to/pretrained_run [/path/to/pretrained_run ...] [--audio_params_source pretrain|spec]"
    exit 0
fi

PRETRAINED_RUNS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --audio_params_source)
            AUDIO_PARAMS_SOURCE="$2"
            shift 2
            ;;
        *)
            PRETRAINED_RUNS+=("$1")
            shift
            ;;
    esac
done

if [ "${#PRETRAINED_RUNS[@]}" -eq 0 ]; then
    echo "Missing /path/to/pretrained_run" 1>&2
    exit 1
fi

while IFS=: read -r SPECIES BIRD_ID; do
    PREP_OUT_DIR="$TEMP_ROOT/tinybird_pool/$SPECIES/$BIRD_ID"
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
            --audio_params_source "$AUDIO_PARAMS_SOURCE" \
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
