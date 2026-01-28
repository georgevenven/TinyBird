#!/bin/bash

# classify_detect_linear_probe_sweep.sh
# Sweep linear probes across multiple pretrained runs for unit_detect and classify.

set -e

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

# ================= CONFIGURATION =================
SPEC_ROOT="specs"
ANNOTATION_ROOT="files"
RESULTS_DIR_DEFAULT="results/linear_probe"
RESULTS_DIR="$RESULTS_DIR_DEFAULT"
RESULTS_DIR_SET=0
RESULTS_NAME=""
RESULTS_PREFIX="linear_probe_sweep"
PRETRAINED_RUNS=()

# Experiment Settings
STEPS=1000
BATCH_SIZE=24
NUM_WORKERS=8
EVAL_EVERY=100
EARLY_STOP_PATIENCE=4
EARLY_STOP_EMA_ALPHA=0.75
EARLY_STOP_MIN_DELTA=0.0
MAX_BIRDS=3
POOL_SEED=42
TRAIN_PERCENT=80
LR="1e-2"
SPECIES_FILTER=""
RUN_TAG=""
RUNS_SUBDIR="linear_probes"
TASK_MODE="unit_detect,classify"

# ================= ARGUMENT PARSING =================
ORIGINAL_ARGS=("$@")
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --spec_root)
        SPEC_ROOT="$2"
        shift 2
        ;;
        --annotation_root)
        ANNOTATION_ROOT="$2"
        shift 2
        ;;
        --results_dir)
        RESULTS_DIR="$2"
        RESULTS_DIR_SET=1
        shift 2
        ;;
        --results_name)
        RESULTS_NAME="$2"
        shift 2
        ;;
        --pretrained_runs)
        IFS=',' read -r -a PRETRAINED_RUNS <<< "$2"
        shift 2
        ;;
        --pretrained_run)
        PRETRAINED_RUNS+=("$2")
        shift 2
        ;;
        --task_mode)
        TASK_MODE="$2"
        shift 2
        ;;
        --steps)
        STEPS="$2"
        shift 2
        ;;
        --batch_size)
        BATCH_SIZE="$2"
        shift 2
        ;;
        --lr)
        LR="$2"
        shift 2
        ;;
        --num_workers)
        NUM_WORKERS="$2"
        shift 2
        ;;
        --eval_every)
        EVAL_EVERY="$2"
        shift 2
        ;;
        --early_stop_patience)
        EARLY_STOP_PATIENCE="$2"
        shift 2
        ;;
        --early_stop_ema_alpha)
        EARLY_STOP_EMA_ALPHA="$2"
        shift 2
        ;;
        --early_stop_min_delta)
        EARLY_STOP_MIN_DELTA="$2"
        shift 2
        ;;
        --max_birds)
        MAX_BIRDS="$2"
        shift 2
        ;;
        --train_percent)
        TRAIN_PERCENT="$2"
        shift 2
        ;;
        --pool_seed)
        POOL_SEED="$2"
        shift 2
        ;;
        --species)
        SPECIES_FILTER="$2"
        shift 2
        ;;
        --run_tag)
        RUN_TAG="$2"
        shift 2
        ;;
        --runs_subdir)
        RUNS_SUBDIR="$2"
        shift 2
        ;;
        *)
        echo "Unknown argument: $1"
        shift 1
        ;;
    esac
done

if [ "${#PRETRAINED_RUNS[@]}" -eq 0 ]; then
    echo "Error: provide --pretrained_runs or --pretrained_run"
    exit 1
fi

RUN_TAG_PREFIX=""
if [ -n "$RUN_TAG" ]; then
    RUN_TAG_PREFIX="${RUN_TAG}_"
fi
if [ -n "$RESULTS_NAME" ]; then
    if [[ "$RESULTS_NAME" = /* ]]; then
        RESULTS_DIR="$RESULTS_NAME"
    else
        RESULTS_DIR="results/$RESULTS_NAME"
    fi
elif [ "$RESULTS_DIR_SET" -eq 0 ] && [ -n "$RUN_TAG" ]; then
    RESULTS_DIR="results/${RESULTS_PREFIX}_${RUN_TAG}"
fi
RUNS_SUBDIR="${RUNS_SUBDIR%/}"

# Ensure results directory exists before logging
mkdir -p "$RESULTS_DIR"

# Log resolved parameters to results directory
PARAMS_JSON="$RESULTS_DIR/run_params_classify_detect_linear_probe.json"
printf '{\n' > "$PARAMS_JSON"
printf '  "command": "%s",\n' "$0 ${ORIGINAL_ARGS[*]}" >> "$PARAMS_JSON"
printf '  "spec_root": "%s",\n' "$SPEC_ROOT" >> "$PARAMS_JSON"
printf '  "annotation_root": "%s",\n' "$ANNOTATION_ROOT" >> "$PARAMS_JSON"
printf '  "results_dir": "%s",\n' "$RESULTS_DIR" >> "$PARAMS_JSON"
printf '  "results_name": "%s",\n' "$RESULTS_NAME" >> "$PARAMS_JSON"
printf '  "pretrained_runs": [%s],\n' "$(printf '"%s",' "${PRETRAINED_RUNS[@]}" | sed 's/,$//')" >> "$PARAMS_JSON"
printf '  "task_mode": "%s",\n' "$TASK_MODE" >> "$PARAMS_JSON"
printf '  "steps": %s,\n' "$STEPS" >> "$PARAMS_JSON"
printf '  "batch_size": %s,\n' "$BATCH_SIZE" >> "$PARAMS_JSON"
printf '  "num_workers": %s,\n' "$NUM_WORKERS" >> "$PARAMS_JSON"
printf '  "eval_every": %s,\n' "$EVAL_EVERY" >> "$PARAMS_JSON"
printf '  "early_stop_patience": %s,\n' "$EARLY_STOP_PATIENCE" >> "$PARAMS_JSON"
printf '  "early_stop_ema_alpha": %s,\n' "$EARLY_STOP_EMA_ALPHA" >> "$PARAMS_JSON"
printf '  "early_stop_min_delta": %s,\n' "$EARLY_STOP_MIN_DELTA" >> "$PARAMS_JSON"
printf '  "max_birds": %s,\n' "$MAX_BIRDS" >> "$PARAMS_JSON"
printf '  "train_percent": %s,\n' "$TRAIN_PERCENT" >> "$PARAMS_JSON"
printf '  "pool_seed": %s,\n' "$POOL_SEED" >> "$PARAMS_JSON"
printf '  "lr": "%s",\n' "$LR" >> "$PARAMS_JSON"
printf '  "run_tag": "%s",\n' "$RUN_TAG" >> "$PARAMS_JSON"
printf '  "runs_subdir": "%s",\n' "$RUNS_SUBDIR" >> "$PARAMS_JSON"
printf '  "species_filter": "%s"\n' "$SPECIES_FILTER" >> "$PARAMS_JSON"
printf '}\n' >> "$PARAMS_JSON"

echo " Configuration:"
echo "   SPEC_ROOT: $SPEC_ROOT"
echo "   RESULTS_DIR: $RESULTS_DIR"
echo "   RESULTS_NAME: $RESULTS_NAME"
echo "   TASK_MODE: $TASK_MODE"
echo "   PRETRAINED_RUNS: ${PRETRAINED_RUNS[*]}"
echo "   STEPS: $STEPS"
echo "   BATCH_SIZE: $BATCH_SIZE"
echo "   NUM_WORKERS: $NUM_WORKERS"
echo "   EVAL_EVERY: $EVAL_EVERY"
echo "   EARLY_STOP_PATIENCE: $EARLY_STOP_PATIENCE"
echo "   EARLY_STOP_EMA_ALPHA: $EARLY_STOP_EMA_ALPHA"
echo "   EARLY_STOP_MIN_DELTA: $EARLY_STOP_MIN_DELTA"
echo "   MAX_BIRDS: $MAX_BIRDS"
echo "   TRAIN_PERCENT: $TRAIN_PERCENT"
echo "   POOL_SEED: $POOL_SEED"
echo "   LR: $LR"
echo "   RUN_TAG: $RUN_TAG"
echo "   RUNS_SUBDIR: $RUNS_SUBDIR"
echo "   SPECIES_FILTER: $SPECIES_FILTER"

WORK_ROOT="$PROJECT_ROOT/temp"
EVAL_DIR="$RESULTS_DIR/eval"
mkdir -p "$WORK_ROOT"
mkdir -p "$EVAL_DIR"

# Species Map: "SpeciesName:AnnotationFile:SpecSubDir"
SPECIES_LIST=(
    "Bengalese_Finch:bf_annotations.json:bf_64hop_32khz"
    "Canary:canary_annotations.json:canary_64hop_32khz"
    "Zebra_Finch:zf_annotations.json:zf_64hop_32khz"
)

SELECTED_SPECIES_LIST=()
if [ -n "$SPECIES_FILTER" ]; then
    IFS=',' read -r -a FILTERS <<< "$SPECIES_FILTER"
    for ENTRY in "${SPECIES_LIST[@]}"; do
        IFS=":" read -r SPECIES _ <<< "$ENTRY"
        for F in "${FILTERS[@]}"; do
            F="${F// /}"
            if [ "$SPECIES" == "$F" ]; then
                SELECTED_SPECIES_LIST+=("$ENTRY")
                break
            fi
        done
    done
    if [ "${#SELECTED_SPECIES_LIST[@]}" -eq 0 ]; then
        echo "Error: --species filter did not match any species in SPECIES_LIST"
        exit 1
    fi
else
    SELECTED_SPECIES_LIST=("${SPECIES_LIST[@]}")
fi

task_enabled() {
    if [ "$TASK_MODE" == "all" ]; then
        return 0
    fi
    [[ ",$TASK_MODE," == *",$1,"* ]]
}

has_npy() {
    local dir="$1"
    if ls "$dir"/*.npy >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

write_filtered_annotations() {
    local annot_path="$1"
    local bird_id="$2"
    local out_path="$3"
    python - <<PY
import json
from pathlib import Path

annot_path = Path("$annot_path")
out_path = Path("$out_path")
bird_id = "$bird_id"

data = json.loads(annot_path.read_text(encoding="utf-8"))
recordings = []
for rec in data.get("recordings", []):
    if rec.get("recording", {}).get("bird_id") == bird_id:
        recordings.append(rec)

unit_ids = set()
for rec in recordings:
    for event in rec.get("detected_events", []):
        for unit in event.get("units", []):
            unit_id = unit.get("id")
            if unit_id is None:
                continue
            unit_ids.add(int(unit_id))

id_map = {old: new for new, old in enumerate(sorted(unit_ids))}
if id_map:
    for rec in recordings:
        for event in rec.get("detected_events", []):
            for unit in event.get("units", []):
                unit_id = unit.get("id")
                if unit_id is None:
                    continue
                unit["id"] = id_map[int(unit_id)]

filtered = dict(data)
filtered["recordings"] = recordings
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
if id_map:
    print(f"Remapped {len(id_map)} unit IDs for bird {bird_id}")
print(f"Wrote filtered annotations: {out_path}")
PY
}

split_recordings_with_units() {
    local annot_path="$1"
    local out_dir="$2"
    local seed="$3"
    local train_percent="$4"
    python - <<PY
import json
import random
from pathlib import Path

annot_path = Path("$annot_path")
out_dir = Path("$out_dir")
out_dir.mkdir(parents=True, exist_ok=True)
seed = int($seed)
train_percent = float($train_percent)

data = json.loads(annot_path.read_text(encoding="utf-8"))
recordings = data.get("recordings", [])

records = []
all_units = set()
total_ms = 0.0

for rec in recordings:
    filename = rec.get("recording", {}).get("filename", "")
    stem = Path(filename).stem
    ms = 0.0
    units = set()
    for event in rec.get("detected_events", []):
        onset = event.get("onset_ms", 0)
        offset = event.get("offset_ms", 0)
        if offset > onset:
            ms += (offset - onset)
        for unit in event.get("units", []):
            unit_id = unit.get("id")
            if unit_id is None:
                continue
            units.add(int(unit_id))
    total_ms += ms
    all_units.update(units)
    records.append({"stem": stem, "ms": ms, "units": units, "rec": rec})

rng = random.Random(seed)
rng.shuffle(records)
records_sorted = sorted(records, key=lambda r: (-len(r["units"]), -r["ms"]))

train = []
covered = set()
train_ms = 0.0

for rec in records_sorted:
    if covered >= all_units:
        break
    new_units = rec["units"] - covered
    if not new_units:
        continue
    train.append(rec)
    covered.update(rec["units"])
    train_ms += rec["ms"]

target_ms = total_ms * (train_percent / 100.0)
remaining = [rec for rec in records if rec not in train]
rng.shuffle(remaining)

for rec in remaining:
    if train_ms >= target_ms:
        break
    train.append(rec)
    train_ms += rec["ms"]

train_stems = [r["stem"] for r in train]
test_stems = [r["stem"] for r in records if r not in train]
test_ms = total_ms - train_ms

missing_units = sorted(all_units - covered)
if missing_units:
    print(f"Warning: missing units in train split: {missing_units}")

def write_subset(stems, out_path):
    subset = dict(data)
    subset["recordings"] = [
        rec for rec in recordings
        if Path(rec.get("recording", {}).get("filename", "")).stem in stems
    ]
    out_path.write_text(json.dumps(subset, indent=2), encoding="utf-8")

train_txt = out_dir / "train_stems.txt"
test_txt = out_dir / "test_stems.txt"
train_txt.write_text("\\n".join(train_stems) + ("\\n" if train_stems else ""), encoding="utf-8")
test_txt.write_text("\\n".join(test_stems) + ("\\n" if test_stems else ""), encoding="utf-8")

write_subset(train_stems, out_dir / "annotations_train.json")
write_subset(test_stems, out_dir / "annotations_test.json")

(out_dir / "train_seconds.txt").write_text(f"{train_ms/1000.0:.6f}", encoding="utf-8")
(out_dir / "test_seconds.txt").write_text(f"{test_ms/1000.0:.6f}", encoding="utf-8")

print(f"Split recordings: train={len(train_stems)}, test={len(test_stems)}")
print(f"Train seconds: {train_ms/1000.0:.3f}, Test seconds: {test_ms/1000.0:.3f}")
PY
}

build_event_chunks() {
    local spec_dir="$1"
    local out_dir="$2"
    local seconds="$3"
    local seed="$4"
    local annot_json="$5"
    local mode="$6"
    if [ -d "$out_dir" ] && has_npy "$out_dir"; then
        return
    fi
    PYTHONWARNINGS=ignore python scripts/sample_by_seconds.py \
        --spec_dir "$spec_dir" \
        --out_dir "$out_dir" \
        --seconds "$seconds" \
        --seed "$seed" \
        --annotation_json "$annot_json" \
        --event_chunks \
        --mode "$mode"
}

eval_val_outputs_f1() {
    local run_name="$1"
    local pretrained_run="$2"
    local dest_csv="$EVAL_DIR/eval_f1.csv"

    if [ -z "$run_name" ]; then
        return
    fi

    python scripts/eval/eval_val_outputs_f1.py \
        --runs_root "$PROJECT_ROOT/runs" \
        --run_names "$run_name" \
        --out_csv "$dest_csv" \
        --append \
        --no_summary \
        --pretrained_run "$pretrained_run" 1>&2

    python - <<PY
import csv
from pathlib import Path

dest = Path("$dest_csv")
run_name = "$run_name"
base_name = run_name.split("/")[-1]
f1 = "0"
fer = "0"
if dest.exists():
    with dest.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("run_name") in (run_name, base_name):
                f1 = row.get("f1", "0") or "0"
                fer = row.get("fer", "0") or "0"
                break

print(f"{f1} {fer}")
PY
}

# Linear probe args (frozen encoder, no class weighting)
PROBE_ARGS="--linear_probe --freeze_encoder --lr $LR --no-class_weighting"

RUN_NAME_PREFIX=""
if [ -n "$RUNS_SUBDIR" ]; then
    RUN_NAME_PREFIX="${RUNS_SUBDIR}/"
fi

for ENTRY in "${SELECTED_SPECIES_LIST[@]}"; do
    IFS=":" read -r SPECIES ANNOT_FILE SPEC_SUBDIR <<< "$ENTRY"

    ANNOT_PATH="$ANNOTATION_ROOT/$ANNOT_FILE"
    SPEC_DIR="$SPEC_ROOT/$SPEC_SUBDIR"
    SPECIES_WORK_DIR="$WORK_ROOT/$SPECIES"

    echo "Processing $SPECIES..."
    echo "  Annotation: $ANNOT_PATH"
    echo "  Spectrograms: $SPEC_DIR"

    if [ ! -f "$ANNOT_PATH" ]; then
        echo "Error: Annotation file not found at $ANNOT_PATH"
        continue
    fi
    if [ ! -d "$SPEC_DIR" ]; then
        echo "Error: Spectrogram directory not found at $SPEC_DIR"
        continue
    fi

    BIRDS=$(python -c "import json; d=json.load(open('$ANNOT_PATH')); print(' '.join(sorted(list(set(r['recording']['bird_id'] for r in d['recordings'])))))")
    echo "  Found individuals: $BIRDS"

    BIRD_COUNT=0
    for BIRD in $BIRDS; do
        if [ "$BIRD_COUNT" -ge "$MAX_BIRDS" ]; then
            echo "  Reached max birds ($MAX_BIRDS). Skipping remaining individuals."
            break
        fi
        ((BIRD_COUNT+=1))

        echo "  Processing Individual: $BIRD"
        BIRD_WORK_DIR="$SPECIES_WORK_DIR/$BIRD"
        mkdir -p "$BIRD_WORK_DIR"

        BIRD_ANNOT="$BIRD_WORK_DIR/annotations_filtered.json"
        if [ ! -f "$BIRD_ANNOT" ]; then
            write_filtered_annotations "$ANNOT_PATH" "$BIRD" "$BIRD_ANNOT"
        fi

        SPLIT_DIR="$BIRD_WORK_DIR/split"
        if [ ! -f "$SPLIT_DIR/train_stems.txt" ]; then
            split_recordings_with_units "$BIRD_ANNOT" "$SPLIT_DIR" "$POOL_SEED" "$TRAIN_PERCENT"
        fi

        TRAIN_ANNOT="$SPLIT_DIR/annotations_train.json"
        TEST_ANNOT="$SPLIT_DIR/annotations_test.json"
        TRAIN_SECONDS=$(cat "$SPLIT_DIR/train_seconds.txt")
        TEST_SECONDS=$(cat "$SPLIT_DIR/test_seconds.txt")

        TRAIN_DIR="$BIRD_WORK_DIR/train"
        TEST_DIR="$BIRD_WORK_DIR/test"

        build_event_chunks "$SPEC_DIR" "$TRAIN_DIR" "$TRAIN_SECONDS" "$POOL_SEED" "$TRAIN_ANNOT" "unit_detect"
        build_event_chunks "$SPEC_DIR" "$TEST_DIR" "$TEST_SECONDS" "$POOL_SEED" "$TEST_ANNOT" "unit_detect"

        if ! has_npy "$TRAIN_DIR" || ! has_npy "$TEST_DIR"; then
            echo "  Skipping $SPECIES / $BIRD (empty train/test after split)"
            rm -rf "$BIRD_WORK_DIR"
            continue
        fi

        for PRETRAINED_RUN in "${PRETRAINED_RUNS[@]}"; do
            PRETRAINED_RUN="${PRETRAINED_RUN%/}"
            PRETRAIN_TAG="$(basename "$PRETRAINED_RUN")"
            PRETRAIN_TAG="${PRETRAIN_TAG// /_}"
            RUN_BASE="${RUN_NAME_PREFIX}${RUN_TAG_PREFIX}linear_${PRETRAIN_TAG}_${SPECIES}_${BIRD}"

            if task_enabled "unit_detect"; then
                RUN_NAME="${RUN_BASE}_unit_detect"
                LOG_FILE="runs/$RUN_NAME/loss_log.txt"
                if [ ! -f "$LOG_FILE" ]; then
                    PYTHONWARNINGS=ignore python src/supervised_train.py \
                        --train_dir "$TRAIN_DIR" \
                        --val_dir "$TEST_DIR" \
                        --run_name "$RUN_NAME" \
                        --pretrained_run "$PRETRAINED_RUN" \
                        --annotation_file "$BIRD_ANNOT" \
                        --mode unit_detect \
                        --steps "$STEPS" \
                        --batch_size "$BATCH_SIZE" \
                        --val_batch_size "$BATCH_SIZE" \
                        --num_workers "$NUM_WORKERS" \
                        --eval_every "$EVAL_EVERY" \
                        --early_stop_patience "$EARLY_STOP_PATIENCE" \
                        --early_stop_ema_alpha "$EARLY_STOP_EMA_ALPHA" \
                        --early_stop_min_delta "$EARLY_STOP_MIN_DELTA" \
                        --amp \
                        --no-save_intermediate_checkpoints \
                        $PROBE_ARGS
                else
                    echo "    Skipping training (run exists): $RUN_NAME"
                fi

                VAL_F1="0"
                if [ -d "runs/$RUN_NAME/val_outputs" ]; then
                    read -r VAL_F1 _ < <(eval_val_outputs_f1 "$RUN_NAME" "$PRETRAINED_RUN")
                fi
                if [ -z "$VAL_F1" ]; then VAL_F1="0"; fi
            fi

            if task_enabled "classify"; then
                RUN_NAME="${RUN_BASE}_classify"
                LOG_FILE="runs/$RUN_NAME/loss_log.txt"
                if [ ! -f "$LOG_FILE" ]; then
                    PYTHONWARNINGS=ignore python src/supervised_train.py \
                        --train_dir "$TRAIN_DIR" \
                        --val_dir "$TEST_DIR" \
                        --run_name "$RUN_NAME" \
                        --pretrained_run "$PRETRAINED_RUN" \
                        --annotation_file "$BIRD_ANNOT" \
                        --mode classify \
                        --steps "$STEPS" \
                        --batch_size "$BATCH_SIZE" \
                        --val_batch_size "$BATCH_SIZE" \
                        --num_workers "$NUM_WORKERS" \
                        --eval_every "$EVAL_EVERY" \
                        --early_stop_patience "$EARLY_STOP_PATIENCE" \
                        --early_stop_ema_alpha "$EARLY_STOP_EMA_ALPHA" \
                        --early_stop_min_delta "$EARLY_STOP_MIN_DELTA" \
                        --amp \
                        --no-save_intermediate_checkpoints \
                        $PROBE_ARGS
                else
                    echo "    Skipping training (run exists): $RUN_NAME"
                fi

                VAL_F1="0"
                VAL_FER="0"
                if [ -d "runs/$RUN_NAME/val_outputs" ]; then
                    read -r VAL_F1 VAL_FER < <(eval_val_outputs_f1 "$RUN_NAME" "$PRETRAINED_RUN")
                fi
                if [ -z "$VAL_F1" ]; then VAL_F1="0"; fi
                if [ -z "$VAL_FER" ]; then VAL_FER="0"; fi
            fi
        done

        echo "  Cleaning up temp data for $SPECIES / $BIRD..."
        rm -rf "$BIRD_WORK_DIR"
    done
done

echo "Cleaning up temporary files..."
rm -rf "$WORK_ROOT"

echo "Linear probe sweep completed! Results: $EVAL_DIR/eval_f1.csv"
