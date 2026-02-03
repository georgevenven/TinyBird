#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

SPEC_ROOT="/media/george-vengrovski/disk2/specs"
WAV_ROOT="/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae"
TRAIN_SECONDS="MAX"
MODE="classify"
PROBE_MODE="linear"
LR="1e-2"
STEPS="1000"
BIRD_LIST_JSON="files/SFT_experiment_birds.json"
TEMP_ROOT="temp"
RESULTS_DIR="results/linear_aves"

mkdir -p "$RESULTS_DIR"

while IFS=: read -r SPECIES BIRD_ID; do
    PREP_OUT_DIR="$TEMP_ROOT/aves_pool/$SPECIES/$BIRD_ID"

    bash shell/train_AVES.sh \
        --spec_root "$SPEC_ROOT" \
        --wav_root "$WAV_ROOT" \
        --species "$SPECIES" \
        --bird_id "$BIRD_ID" \
        --train_seconds "$TRAIN_SECONDS" \
        --mode "$MODE" \
        --prep_only

    RUN_TAG="linear_aves/${SPECIES}_${BIRD_ID}"
    if ! bash shell/train_AVES.sh \
        --spec_root "$SPEC_ROOT" \
        --wav_root "$WAV_ROOT" \
        --species "$SPECIES" \
        --bird_id "$BIRD_ID" \
        --train_seconds "$TRAIN_SECONDS" \
        --mode "$MODE" \
        --probe_mode "$PROBE_MODE" \
        --lr "$LR" \
        --steps "$STEPS" \
        --run_tag "$RUN_TAG" \
        --use_prepared; then
        echo "run failed: ${SPECIES} ${BIRD_ID}"
    else
        python "$PROJECT_ROOT/scripts/eval/eval_val_outputs_f1.py" \
            --runs_root "$PROJECT_ROOT/runs" \
            --run_names "$RUN_TAG" \
            --out_csv "$RESULTS_DIR/eval_f1.csv" \
            --append \
            --no_summary
    fi

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
