#!/bin/bash

# train_AVES.sh
# Benchmark script for AVES detection, unit detection, and classification
# Mirrors classify_detect_bench.sh but trains AVES on wav files.

set -e

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

# ================= CONFIGURATION =================
SPEC_ROOT="/media/george-vengrovski/disk2/specs"
WAV_ROOT="/media/george-vengrovski/disk2/wavs"
ANNOTATION_ROOT="/home/george-vengrovski/Documents/projects/TinyBird/files"
RESULTS_DIR_DEFAULT="results/aves_benchmark"
RESULTS_DIR="$RESULTS_DIR_DEFAULT"
RESULTS_DIR_SET=0
RESULTS_NAME=""
RESULTS_PREFIX="aves_benchmark"

# AVES model files
AVES_MODEL="/home/george-vengrovski/Documents/projects/TinyBird/files/aves-base-bio.torchaudio.pt"
AVES_CONFIG="/home/george-vengrovski/Documents/projects/TinyBird/files/aves-base-bio.torchaudio.model_config.json"
AVES_SR=16000
EMBEDDING_DIM=768

# Experiment Settings
SAMPLE_SECONDS=(100)
TEST_PERCENT=20
STEPS=100
BATCH_SIZE=12
NUM_WORKERS=4
MAX_BIRDS=3
POOL_SIZE=0
MAX_TRAIN=0
POOL_SEED=42
RUN_TAG=""
CLASS_WEIGHTING=0
SPECIES_FILTER=""
LR="1e-4"
LR_SET=0
RUNS_SUBDIR="aves_bench"
CONTEXT_TIMEBINS=0
PRETRAINED_RUN=""

# Probe Type: "linear", "finetune", or comma-separated list
PROBE_MODE="linear"

# Task Selection: "all" or a comma-separated list
TASK_MODE="classify"

# ================= ARGUMENT PARSING =================
ORIGINAL_ARGS=("$@")
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --spec_root)
        SPEC_ROOT="$2"
        shift 2
        ;;
        --wav_root)
        WAV_ROOT="$2"
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
        --aves_model)
        AVES_MODEL="$2"
        shift 2
        ;;
        --aves_config)
        AVES_CONFIG="$2"
        shift 2
        ;;
        --aves_sr)
        AVES_SR="$2"
        shift 2
        ;;
        --embedding_dim)
        EMBEDDING_DIM="$2"
        shift 2
        ;;
        --probe_mode)
        PROBE_MODE="$2"
        shift 2
        ;;
        --task_mode)
        TASK_MODE="$2"
        shift 2
        ;;
        --lr)
        LR="$2"
        LR_SET=1
        shift 2
        ;;
        --runs_subdir)
        RUNS_SUBDIR="$2"
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
        --num_workers)
        NUM_WORKERS="$2"
        shift 2
        ;;
        --max_birds)
        MAX_BIRDS="$2"
        shift 2
        ;;
        --sample_seconds)
        IFS=',' read -r -a SAMPLE_SECONDS <<< "$2"
        shift 2
        ;;
        --sample_sizes)
        IFS=',' read -r -a SAMPLE_SECONDS <<< "$2"
        shift 2
        ;;
        --class_weighting)
        CLASS_WEIGHTING=1
        shift 1
        ;;
        --no_class_weighting)
        CLASS_WEIGHTING=0
        shift 1
        ;;
        --species)
        SPECIES_FILTER="$2"
        shift 2
        ;;
        --pool_size)
        POOL_SIZE="$2"
        shift 2
        ;;
        --max_train)
        MAX_TRAIN="$2"
        shift 2
        ;;
        --pool_seed)
        POOL_SEED="$2"
        shift 2
        ;;
        --run_tag)
        RUN_TAG="$2"
        shift 2
        ;;
        --context_timebins)
        CONTEXT_TIMEBINS="$2"
        shift 2
        ;;
        *)
        echo "Unknown argument: $1"
        shift 1
        ;;
    esac
done

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
PRETRAINED_RUN="$AVES_MODEL"

# Ensure results directory exists before logging
mkdir -p "$RESULTS_DIR"
EVAL_DIR="$RESULTS_DIR/eval"
mkdir -p "$EVAL_DIR"

# Log resolved parameters
PARAMS_JSON="$RESULTS_DIR/run_params_train_AVES.json"
printf '{\n' > "$PARAMS_JSON"
printf '  "command": "%s",\n' "$0 ${ORIGINAL_ARGS[*]}" >> "$PARAMS_JSON"
printf '  "spec_root": "%s",\n' "$SPEC_ROOT" >> "$PARAMS_JSON"
printf '  "wav_root": "%s",\n' "$WAV_ROOT" >> "$PARAMS_JSON"
printf '  "annotation_root": "%s",\n' "$ANNOTATION_ROOT" >> "$PARAMS_JSON"
printf '  "results_dir": "%s",\n' "$RESULTS_DIR" >> "$PARAMS_JSON"
printf '  "results_name": "%s",\n' "$RESULTS_NAME" >> "$PARAMS_JSON"
printf '  "aves_model": "%s",\n' "$AVES_MODEL" >> "$PARAMS_JSON"
printf '  "aves_config": "%s",\n' "$AVES_CONFIG" >> "$PARAMS_JSON"
printf '  "probe_mode": "%s",\n' "$PROBE_MODE" >> "$PARAMS_JSON"
printf '  "task_mode": "%s",\n' "$TASK_MODE" >> "$PARAMS_JSON"
printf '  "steps": %s,\n' "$STEPS" >> "$PARAMS_JSON"
printf '  "batch_size": %s,\n' "$BATCH_SIZE" >> "$PARAMS_JSON"
printf '  "num_workers": %s,\n' "$NUM_WORKERS" >> "$PARAMS_JSON"
printf '  "max_birds": %s,\n' "$MAX_BIRDS" >> "$PARAMS_JSON"
printf '  "pool_size": %s,\n' "$POOL_SIZE" >> "$PARAMS_JSON"
printf '  "max_train": %s,\n' "$MAX_TRAIN" >> "$PARAMS_JSON"
printf '  "pool_seed": %s,\n' "$POOL_SEED" >> "$PARAMS_JSON"
printf '  "run_tag": "%s",\n' "$RUN_TAG" >> "$PARAMS_JSON"
printf '  "runs_subdir": "%s",\n' "$RUNS_SUBDIR" >> "$PARAMS_JSON"
printf '  "class_weighting": %s,\n' "$CLASS_WEIGHTING" >> "$PARAMS_JSON"
printf '  "lr": %s,\n' "$LR" >> "$PARAMS_JSON"
printf '  "context_timebins": %s,\n' "$CONTEXT_TIMEBINS" >> "$PARAMS_JSON"
printf '  "sample_seconds": [' >> "$PARAMS_JSON"
for i in "${!SAMPLE_SECONDS[@]}"; do
    if [ "$i" -gt 0 ]; then printf ', ' >> "$PARAMS_JSON"; fi
    printf '%s' "${SAMPLE_SECONDS[$i]}" >> "$PARAMS_JSON"
done
printf ']\n' >> "$PARAMS_JSON"
printf '}\n' >> "$PARAMS_JSON"

echo " Configuration:"
echo "   SPEC_ROOT: $SPEC_ROOT"
echo "   WAV_ROOT: $WAV_ROOT"
echo "   RESULTS_DIR: $RESULTS_DIR"
echo "   RESULTS_NAME: $RESULTS_NAME"
echo "   AVES_MODEL: $AVES_MODEL"
echo "   AVES_CONFIG: $AVES_CONFIG"
echo "   TASK_MODE: $TASK_MODE"
echo "   PROBE_MODE: $PROBE_MODE"
echo "   MAX_BIRDS: $MAX_BIRDS"
echo "   SAMPLE_SECONDS: ${SAMPLE_SECONDS[*]}"
echo "   POOL_SIZE: $POOL_SIZE"
echo "   MAX_TRAIN: $MAX_TRAIN"
echo "   POOL_SEED: $POOL_SEED"
echo "   RUN_TAG: $RUN_TAG"
echo "   RUNS_SUBDIR: $RUNS_SUBDIR"
echo "   CLASS_WEIGHTING: $CLASS_WEIGHTING"
echo "   LR: $LR"

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

copy_audio_params() {
    local target_dir="$1"
    if [ -z "$AUDIO_PARAMS_SRC" ]; then
        return
    fi
    if [ ! -f "$AUDIO_PARAMS_SRC" ]; then
        return
    fi
    if [ -d "$target_dir" ]; then
        cp -f "$AUDIO_PARAMS_SRC" "$target_dir/audio_params.json"
    fi
}

sample_by_seconds() {
    local src_dir="$1"
    local out_dir="$2"
    local seconds="$3"
    local seed="$4"
    local order_file="$5"
    local truncate_last="$6"
    local move="$7"
    local annot_json="$8"
    local bird_id="$9"
    local extra_args=()
    if [ "$#" -gt 9 ]; then
        extra_args=("${@:10}")
    fi

    if [ ! -d "$out_dir" ] || ! has_npy "$out_dir"; then
        python scripts/sample_by_seconds.py \
            --spec_dir "$src_dir" \
            --out_dir "$out_dir" \
            --seconds "$seconds" \
            --seed "$seed" \
            ${order_file:+--order_file "$order_file"} \
            $([ "$truncate_last" -eq 1 ] && echo "" || echo "--no_truncate_last") \
            $([ "$move" -eq 1 ] && echo "--move" || echo "") \
            ${annot_json:+--annotation_json "$annot_json"} \
            ${bird_id:+--bird_id "$bird_id"} \
            "${extra_args[@]}"
    fi
    copy_audio_params "$out_dir"
}

make_train_order() {
    local pool_dir="$1"
    local seed="$2"
    local order_file="$pool_dir/train_order.txt"
    if [ ! -f "$order_file" ]; then
        python - <<PY
import random
from pathlib import Path
pool = Path("$pool_dir")
files = sorted(pool.glob("*.npy"))
random.seed(int($seed))
random.shuffle(files)
with open("$order_file", "w") as f:
    for p in files:
        f.write(p.name + "\\n")
PY
    fi
}

make_fixed_pool_seconds() {
    local src_dir="$1"
    local pool_dir="$2"
    local test_dir="$3"
    local pool_seconds="$4"
    local max_train_seconds="$5"
    local seed="$6"
    local annot_json="$7"
    local bird_id="$8"
    local min_timebins="${9:-0}"

    local extra_args=()
    if [ -n "$annot_json" ] && [ -n "$bird_id" ]; then
        extra_args=(--ensure_units --mode classify --min_timebins "$min_timebins" --random_crop --event_chunks)
    fi
    sample_by_seconds "$src_dir" "$pool_dir" "$pool_seconds" "$seed" "" 1 0 "$annot_json" "$bird_id" "${extra_args[@]}"

    local test_seconds
    test_seconds=$(python - <<PY
pool = float("$pool_seconds")
max_train = float("$max_train_seconds")
print(max(pool - max_train, 0.0))
PY
)

    if [ "$(python - <<PY
print(int(float("$test_seconds") > 0))
PY
)" -eq 1 ]; then
        sample_by_seconds "$pool_dir" "$test_dir" "$test_seconds" "$seed" "" 0 1
    fi

    make_train_order "$pool_dir" "$seed"
}

copy_train_subset_seconds() {
    local pool_dir="$1"
    local train_dir="$2"
    local seconds="$3"
    local seed="$4"
    local annot_json="$5"
    local bird_id="$6"
    local ensure_units="${7:-0}"
    local min_timebins="${8:-0}"

    make_train_order "$pool_dir" "$seed"
    local extra_args=()
    if [ "$ensure_units" -eq 1 ]; then
        extra_args=(--ensure_units --mode classify --min_timebins "$min_timebins" --random_crop --event_chunks)
    fi
    sample_by_seconds \
        "$pool_dir" \
        "$train_dir" \
        "$seconds" \
        "$seed" \
        "$pool_dir/train_order.txt" \
        1 \
        0 \
        "$annot_json" \
        "$bird_id" \
        "${extra_args[@]}"
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

eval_val_outputs_f1() {
    local run_name="$1"
    local annot_json="$2"
    local mode="$3"
    local audio_params="$4"
    local dest_csv="$EVAL_DIR/eval_f1.csv"

    if [ -z "$run_name" ]; then
        return
    fi
    if [ -z "$annot_json" ] || [ -z "$mode" ] || [ -z "$audio_params" ]; then
        echo "Missing annot_json/mode/audio_params for eval: $run_name" 1>&2
        return
    fi

    python scripts/eval/eval_val_outputs_f1.py \
        --runs_root "$PROJECT_ROOT/runs" \
        --run_names "$run_name" \
        --out_csv "$dest_csv" \
        --append \
        --no_summary \
        --pretrained_run "$PRETRAINED_RUN" 1>&2

    ms_f1=$(python - <<PY
import json
import math
import subprocess
from pathlib import Path

audio_path = Path("$audio_params")
if not audio_path.exists():
    raise SystemExit(f"audio_params.json not found: {audio_path}")
audio = json.loads(audio_path.read_text(encoding="utf-8"))
sr = audio.get("sr")
hop = audio.get("hop_size")
if sr is None or hop is None:
    raise SystemExit(f"audio_params missing sr/hop_size: {audio_path}")
ms_per_timebin = float(hop) / float(sr) * 1000.0
ms_round = int(round(ms_per_timebin))
if ms_round <= 0 or abs(ms_per_timebin - ms_round) > 1e-6:
    raise SystemExit(f"Non-integer ms_per_timebin={ms_per_timebin:.6f} from {audio_path}")

val_outputs = Path("$PROJECT_ROOT") / "runs" / "$run_name" / "val_outputs"
if not val_outputs.exists():
    raise SystemExit(f"val_outputs not found: {val_outputs}")

cmd = [
    "python",
    "scripts/eval/eval_ms_f1.py",
    "--val_outputs_dir",
    str(val_outputs),
    "--annotation_json",
    "$annot_json",
    "--mode",
    "$mode",
    "--ms_per_timebin",
    str(ms_round),
]
out = subprocess.check_output(cmd, text=True).strip().splitlines()
if not out:
    raise SystemExit("eval_ms_f1 produced no output")
last = out[-1]
# Expect: "MS-F1 (macro): 96.12" or "MS-F1 (binary): 88.90"
parts = last.split(":")
if len(parts) < 2:
    raise SystemExit(f"Unexpected eval_ms_f1 output: {last}")
print(parts[-1].strip())
PY
)

    python - <<PY
import csv
from pathlib import Path

dest = Path("$dest_csv")
run_name = "$run_name"
base_name = run_name.split("/")[-1]
ms_f1 = "$ms_f1"
if not dest.exists():
    raise SystemExit(0)

rows = []
with dest.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames or []
    for row in reader:
        if row.get("run_name") in (run_name, base_name):
            row["f1"] = ms_f1
        rows.append(row)

with dest.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
PY

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

CLASS_WEIGHTING_FLAG="--no-class_weighting"
if [ "$CLASS_WEIGHTING" -eq 1 ]; then
    CLASS_WEIGHTING_FLAG="--class_weighting"
fi

# Resolve probe modes list
if [ "$PROBE_MODE" == "all" ]; then
    PROBE_MODES=("linear" "finetune")
else
    IFS=',' read -r -a PROBE_MODES <<< "$PROBE_MODE"
fi

RUN_NAME_PREFIX=""
if [ -n "$RUNS_SUBDIR" ]; then
    RUN_NAME_PREFIX="${RUNS_SUBDIR}/"
fi

for PROBE in "${PROBE_MODES[@]}"; do
    PROBE="${PROBE// /}"
    PROBE_LR="$LR"
    if [ "$LR_SET" -eq 0 ]; then
        if [ "$PROBE" == "linear" ]; then
            PROBE_LR="1e-2"
        else
            PROBE_LR="1e-4"
        fi
    fi
    PROBE_LABEL="linear"
    PROBE_ARGS=("--linear_probe" "--lr" "$PROBE_LR")
    RUN_PREFIX="linear_"
    if [ "$PROBE" == "finetune" ]; then
        PROBE_LABEL="finetune"
        PROBE_ARGS=("--finetune" "--lr" "$PROBE_LR")
        RUN_PREFIX="finetune_"
    fi

    for ENTRY in "${SELECTED_SPECIES_LIST[@]}"; do
        IFS=":" read -r SPECIES ANNOT_FILE SPEC_SUBDIR <<< "$ENTRY"

        ANNOT_PATH="$ANNOTATION_ROOT/$ANNOT_FILE"
        SPEC_DIR="$SPEC_ROOT/$SPEC_SUBDIR"
        SPECIES_WORK_DIR="$RESULTS_DIR/work/$SPECIES/$PROBE_LABEL"
        WAV_SPEC_DIR="$WAV_ROOT/$SPEC_SUBDIR"
        WAV_DIR="$WAV_ROOT"
        if [ -d "$WAV_SPEC_DIR" ]; then
            WAV_DIR="$WAV_SPEC_DIR"
        fi
        AUDIO_PARAMS_SRC="$SPEC_DIR/audio_params.json"

        echo "Processing $SPECIES ($PROBE_LABEL)..."
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

        # =========================================
        # TASK 1: DETECTION
        # =========================================
        if task_enabled "detect"; then
            echo "--- Starting Detection Benchmark for $SPECIES ---"
            DET_WORK_DIR="$SPECIES_WORK_DIR/detect"
            mkdir -p "$DET_WORK_DIR"

            POOL_DIR="$DET_WORK_DIR/pool"
            TEST_DIR="$DET_WORK_DIR/test"

            if [ "$(python - <<PY
print(int(float("$POOL_SIZE") > 0 and float("$MAX_TRAIN") > 0))
PY
)" -eq 1 ]; then
                echo "  Creating Fixed Pool/Test Split (pool=${POOL_SIZE}s, max_train=${MAX_TRAIN}s)..."
                make_fixed_pool_seconds "$SPEC_DIR" "$POOL_DIR" "$TEST_DIR" "$POOL_SIZE" "$MAX_TRAIN" "$POOL_SEED"
            else
                if [ ! -d "$TEST_DIR" ] || ! has_npy "$TEST_DIR"; then
                    echo "  Creating Test Set..."
                    python scripts/split_train_test.py \
                        --spec_dir "$SPEC_DIR" \
                        --train_dir "$POOL_DIR" \
                        --test_dir "$TEST_DIR" \
                        --annotation_json "$ANNOT_PATH" \
                        --train_percent $((100 - TEST_PERCENT)) \
                        --ignore_bird_id \
                        --mode split
                    make_train_order "$POOL_DIR" "$POOL_SEED"
                else
                    echo "  Test Set already exists."
                    make_train_order "$POOL_DIR" "$POOL_SEED"
                fi
            fi

            for SECONDS in "${SAMPLE_SECONDS[@]}"; do
                SECONDS_TAG="${SECONDS//./p}"
                echo "  Running Detection with ${SECONDS}s..."
                RUN_NAME="${RUN_NAME_PREFIX}${RUN_PREFIX}${SPECIES}_detect_${SECONDS_TAG}s"
                TRAIN_DIR="$DET_WORK_DIR/train_${SECONDS_TAG}s"

                copy_train_subset_seconds "$POOL_DIR" "$TRAIN_DIR" "$SECONDS" "$POOL_SEED"

                LOG_FILE="runs/$RUN_NAME/loss_log.txt"
                if [ ! -f "$LOG_FILE" ]; then
                    PYTHONWARNINGS=ignore python src/aves.py \
                        --train_dir "$TRAIN_DIR" \
                        --val_dir "$TEST_DIR" \
                        --run_name "$RUN_NAME" \
                        --annotation_file "$ANNOT_PATH" \
                        --mode detect \
                        --wav_root "$WAV_DIR" \
                        --aves_model_path "$AVES_MODEL" \
                        --aves_config_path "$AVES_CONFIG" \
                        --audio_sr "$AVES_SR" \
                        --embedding_dim "$EMBEDDING_DIM" \
                        $([ "$CONTEXT_TIMEBINS" -gt 0 ] && echo "--num_timebins $CONTEXT_TIMEBINS") \
                        --steps "$STEPS" \
                        --batch_size "$BATCH_SIZE" \
                        --val_batch_size "$BATCH_SIZE" \
                        --num_workers "$NUM_WORKERS" \
                        --amp \
                        "${PROBE_ARGS[@]}"
                else
                    echo "    Skipping training (run exists)"
                fi

                VAL_F1="0"
                if [ -d "runs/$RUN_NAME/val_outputs" ]; then
                    read -r VAL_F1 _ < <(eval_val_outputs_f1 "$RUN_NAME" "$ANNOT_PATH" "detect" "$TEST_DIR/audio_params.json")
                fi
                if [ -z "$VAL_F1" ]; then VAL_F1="0"; fi
                ERROR=$(python - <<PY
print(100 - float("$VAL_F1"))
PY
)
                echo "    Result: F1 = $VAL_F1 %, F1 Error = $ERROR %"
            done

            echo "  Cleaning up copied Detection files for $SPECIES..."
            rm -rf "$DET_WORK_DIR"
        else
            echo "Skipping Detection Task (TASK_MODE=$TASK_MODE)"
        fi

        # =========================================
        # TASK 1b: UNIT DETECTION (Individual Level)
        # =========================================
        if task_enabled "unit_detect"; then
            echo "--- Starting Unit Detection Benchmark for $SPECIES ---"
            UNIT_WORK_DIR="$SPECIES_WORK_DIR/unit_detect"
            mkdir -p "$UNIT_WORK_DIR"

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
                UNIT_BIRD_DIR="$UNIT_WORK_DIR/$BIRD"
                mkdir -p "$UNIT_BIRD_DIR"

                BIRD_TRAIN_POOL="$UNIT_BIRD_DIR/pool_train"
                BIRD_TEST="$UNIT_BIRD_DIR/test"
                BIRD_POOL_FIXED="$UNIT_BIRD_DIR/pool_fixed"
                FILTERED_ANNOT="$UNIT_BIRD_DIR/annotations_filtered.json"
                if [ ! -f "$FILTERED_ANNOT" ]; then
                    write_filtered_annotations "$ANNOT_PATH" "$BIRD" "$FILTERED_ANNOT"
                fi
                BIRD_ANNOT="$FILTERED_ANNOT"

                if [ "$(python - <<PY
print(int(float("$POOL_SIZE") > 0 and float("$MAX_TRAIN") > 0))
PY
)" -eq 1 ]; then
                    make_fixed_pool_seconds "$SPEC_DIR" "$BIRD_POOL_FIXED" "$BIRD_TEST" "$POOL_SIZE" "$MAX_TRAIN" "$POOL_SEED" "$BIRD_ANNOT" "$BIRD" "$CONTEXT_TIMEBINS"
                    TRAIN_POOL_DIR="$BIRD_POOL_FIXED"
                else
                    if [ ! -d "$BIRD_TEST" ] || ! has_npy "$BIRD_TEST"; then
                        python scripts/split_train_test.py \
                            --mode filter_bird \
                            --spec_dir "$SPEC_DIR" \
                            --train_dir "$BIRD_TRAIN_POOL" \
                            --annotation_json "$ANNOT_PATH" \
                            --bird_id "$BIRD"
                        python scripts/split_train_test.py \
                            --mode split \
                            --spec_dir "$BIRD_TRAIN_POOL" \
                            --train_dir "$BIRD_TRAIN_POOL" \
                            --test_dir "$BIRD_TEST" \
                            --annotation_json "$BIRD_ANNOT" \
                            --train_percent $((100 - TEST_PERCENT)) \
                            --ignore_bird_id
                    fi
                    TRAIN_POOL_DIR="$BIRD_TRAIN_POOL"
                fi

                for SECONDS in "${SAMPLE_SECONDS[@]}"; do
                    SECONDS_TAG="${SECONDS//./p}"
                    RUN_NAME="${RUN_NAME_PREFIX}${RUN_PREFIX}${SPECIES}_unit_${BIRD}_${SECONDS_TAG}s"
                    TRAIN_DIR="$UNIT_BIRD_DIR/train_${SECONDS_TAG}s"

                    copy_train_subset_seconds "$TRAIN_POOL_DIR" "$TRAIN_DIR" "$SECONDS" "$POOL_SEED" "$BIRD_ANNOT" "$BIRD" 1 "$CONTEXT_TIMEBINS"

                    LOG_FILE="runs/$RUN_NAME/loss_log.txt"
                    if [ ! -f "$LOG_FILE" ]; then
                        PYTHONWARNINGS=ignore python src/aves.py \
                            --train_dir "$TRAIN_DIR" \
                            --val_dir "$BIRD_TEST" \
                            --run_name "$RUN_NAME" \
                            --annotation_file "$BIRD_ANNOT" \
                            --mode unit_detect \
                            --wav_root "$WAV_DIR" \
                            --aves_model_path "$AVES_MODEL" \
                            --aves_config_path "$AVES_CONFIG" \
                            --audio_sr "$AVES_SR" \
                            --embedding_dim "$EMBEDDING_DIM" \
                            $([ "$CONTEXT_TIMEBINS" -gt 0 ] && echo "--num_timebins $CONTEXT_TIMEBINS") \
                            --steps "$STEPS" \
                            --batch_size "$BATCH_SIZE" \
                            --val_batch_size "$BATCH_SIZE" \
                            --num_workers "$NUM_WORKERS" \
                            --amp \
                            "${PROBE_ARGS[@]}"
                    else
                        echo "    Skipping training (run exists)"
                    fi

                    VAL_F1="0"
                    if [ -d "runs/$RUN_NAME/val_outputs" ]; then
                        read -r VAL_F1 _ < <(eval_val_outputs_f1 "$RUN_NAME" "$BIRD_ANNOT" "unit_detect" "$BIRD_TEST/audio_params.json")
                    fi
                    if [ -z "$VAL_F1" ]; then VAL_F1="0"; fi
                    ERROR=$(python - <<PY
print(100 - float("$VAL_F1"))
PY
)
                    echo "    Result: F1 = $VAL_F1 %, F1 Error = $ERROR %"
                done
            done

            echo "  Cleaning up copied Unit Detection files for $SPECIES..."
            rm -rf "$UNIT_WORK_DIR"
        else
            echo "Skipping Unit Detection Task (TASK_MODE=$TASK_MODE)"
        fi

        # =========================================
        # TASK 2: CLASSIFICATION (Individual Level)
        # =========================================
        if task_enabled "classify"; then
            echo "--- Starting Classification Benchmark for $SPECIES ---"

            CLS_WORK_DIR="$SPECIES_WORK_DIR/classify"
            mkdir -p "$CLS_WORK_DIR"

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
                CLS_BIRD_DIR="$CLS_WORK_DIR/$BIRD"
                mkdir -p "$CLS_BIRD_DIR"

                BIRD_TRAIN_POOL="$CLS_BIRD_DIR/pool_train"
                BIRD_TEST="$CLS_BIRD_DIR/test"
                BIRD_POOL_FIXED="$CLS_BIRD_DIR/pool_fixed"
                FILTERED_ANNOT="$CLS_BIRD_DIR/annotations_filtered.json"
                if [ ! -f "$FILTERED_ANNOT" ]; then
                    write_filtered_annotations "$ANNOT_PATH" "$BIRD" "$FILTERED_ANNOT"
                fi
                BIRD_ANNOT="$FILTERED_ANNOT"

                if [ "$(python - <<PY
print(int(float("$POOL_SIZE") > 0 and float("$MAX_TRAIN") > 0))
PY
)" -eq 1 ]; then
                    make_fixed_pool_seconds "$SPEC_DIR" "$BIRD_POOL_FIXED" "$BIRD_TEST" "$POOL_SIZE" "$MAX_TRAIN" "$POOL_SEED" "$BIRD_ANNOT" "$BIRD"
                    TRAIN_POOL_DIR="$BIRD_POOL_FIXED"
                else
                    if [ ! -d "$BIRD_TEST" ] || ! has_npy "$BIRD_TEST"; then
                        python scripts/split_train_test.py \
                            --mode filter_bird \
                            --spec_dir "$SPEC_DIR" \
                            --train_dir "$BIRD_TRAIN_POOL" \
                            --annotation_json "$ANNOT_PATH" \
                            --bird_id "$BIRD"
                        python scripts/split_train_test.py \
                            --mode split \
                            --spec_dir "$BIRD_TRAIN_POOL" \
                            --train_dir "$BIRD_TRAIN_POOL" \
                            --test_dir "$BIRD_TEST" \
                            --annotation_json "$BIRD_ANNOT" \
                            --train_percent $((100 - TEST_PERCENT)) \
                            --ignore_bird_id
                    fi
                    TRAIN_POOL_DIR="$BIRD_TRAIN_POOL"
                fi

                for SECONDS in "${SAMPLE_SECONDS[@]}"; do
                    SECONDS_TAG="${SECONDS//./p}"
                    RUN_NAME="${RUN_NAME_PREFIX}${RUN_PREFIX}${SPECIES}_classify_${BIRD}_${SECONDS_TAG}s"
                    TRAIN_DIR="$CLS_BIRD_DIR/train_${SECONDS_TAG}s"

                    copy_train_subset_seconds "$TRAIN_POOL_DIR" "$TRAIN_DIR" "$SECONDS" "$POOL_SEED" "$BIRD_ANNOT" "$BIRD" 1 "$CONTEXT_TIMEBINS"

                    LOG_FILE="runs/$RUN_NAME/loss_log.txt"
                    if [ ! -f "$LOG_FILE" ]; then
                        PYTHONWARNINGS=ignore python src/aves.py \
                            --train_dir "$TRAIN_DIR" \
                            --val_dir "$BIRD_TEST" \
                            --run_name "$RUN_NAME" \
                            --annotation_file "$BIRD_ANNOT" \
                            --mode classify \
                            --wav_root "$WAV_DIR" \
                            --aves_model_path "$AVES_MODEL" \
                            --aves_config_path "$AVES_CONFIG" \
                            --audio_sr "$AVES_SR" \
                            --embedding_dim "$EMBEDDING_DIM" \
                            $([ "$CONTEXT_TIMEBINS" -gt 0 ] && echo "--num_timebins $CONTEXT_TIMEBINS") \
                            --steps "$STEPS" \
                            --batch_size "$BATCH_SIZE" \
                            --val_batch_size "$BATCH_SIZE" \
                            --num_workers "$NUM_WORKERS" \
                            --amp \
                            --log_f1 \
                            $CLASS_WEIGHTING_FLAG \
                            "${PROBE_ARGS[@]}"
                    else
                        echo "    Skipping training (run exists)"
                    fi

                    VAL_F1="0"
                    if [ -d "runs/$RUN_NAME/val_outputs" ]; then
                        read -r VAL_F1 VAL_FER < <(eval_val_outputs_f1 "$RUN_NAME" "$BIRD_ANNOT" "classify" "$BIRD_TEST/audio_params.json")
                    fi
                    if [ -z "$VAL_F1" ]; then VAL_F1="0"; fi
                    ERROR=$(python - <<PY
print(100 - float("$VAL_F1"))
PY
)
                    echo "    Result: F1 = $VAL_F1 %, F1 Error = $ERROR %"
                done
            done

            echo "  Cleaning up copied Classification files for $SPECIES..."
            rm -rf "$CLS_WORK_DIR"
        else
            echo "Skipping Classification Task (TASK_MODE=$TASK_MODE)"
        fi
    done
done
