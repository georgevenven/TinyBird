#!/bin/bash

# classify_detect_bench.sh
# Benchmark script for TinyBird detection, unit detection, and classification
# Generates Error vs Training Seconds plots

set -e

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

# ================= CONFIGURATION =================
# PATHS - PLEASE UPDATE THESE IF NEEDED
# Assuming spectrograms are in a 'spectrograms' folder or similar.
# Since I couldn't automatically locate them, please set SPEC_ROOT.
SPEC_ROOT="/media/george-vengrovski/disk2/specs"
ANNOTATION_ROOT="/home/george-vengrovski/Documents/projects/TinyBird/files"
RESULTS_DIR="results/benchmark"
PRETRAINED_RUN="/home/george-vengrovski/Documents/projects/TinyBird/runs/tinybird_pretrain_20251122_091539" # the pretrained model 

# Experiment Settings
# SAMPLE_SECONDS, POOL_SIZE, and MAX_TRAIN are interpreted as seconds.
SAMPLE_SECONDS=(100)
TEST_PERCENT=20
STEPS=100
BATCH_SIZE=24
NUM_WORKERS=4
MAX_BIRDS=3
POOL_SIZE=0
MAX_TRAIN=0
POOL_SEED=42
RUN_TAG=""
CLASS_WEIGHTING=0
SPECIES_FILTER=""

# Probe Type: "linear", "finetune", or comma-separated list (e.g. "linear,finetune")
PROBE_MODE="finetune"

# Task Selection: "all" or a comma-separated list (e.g. "unit_detect,classify")
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
        --annotation_root)
        ANNOTATION_ROOT="$2"
        shift 2
        ;;
        --results_dir)
        RESULTS_DIR="$2"
        shift 2
        ;;
        --pretrained_run)
        PRETRAINED_RUN="$2"
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
        *)
        echo "Unknown argument: $1"
        shift 1
        ;;
    esac
done

# Determine context window (num_timebins) from pretrain config for safe cropping.
CONTEXT_TIMEBINS=$(python - <<PY
import json
import os
from pathlib import Path

path = "$PRETRAINED_RUN"
project_root = "$PROJECT_ROOT"
if not os.path.isabs(path):
    candidate = Path(project_root) / path
    if candidate.exists():
        path = candidate
    else:
        path = Path(project_root) / "runs" / path
cfg_path = Path(path) / "config.json"
if cfg_path.exists():
    cfg = json.loads(cfg_path.read_text())
    print(int(cfg.get("num_timebins", 0)))
else:
    print(0)
PY
)

# Validate pooled sampling config
if [ "$(python - <<PY
print(int(float("$POOL_SIZE") > 0 or float("$MAX_TRAIN") > 0))
PY
)" -eq 1 ]; then
    if [ "$(python - <<PY
print(int(float("$POOL_SIZE") <= 0 or float("$MAX_TRAIN") <= 0))
PY
)" -eq 1 ]; then
        echo "Error: --pool_size and --max_train must both be > 0"
        exit 1
    fi
    if [ -z "$RUN_TAG" ]; then
        RUN_TAG="pool${POOL_SIZE}s_train${MAX_TRAIN}s"
    fi
fi

MAX_SAMPLE=$(python - <<PY
vals = [float(v) for v in "${SAMPLE_SECONDS[*]}".split()] if "${SAMPLE_SECONDS[*]}".strip() else []
print(max(vals) if vals else 0)
PY
)
if [ "$(python - <<PY
print(int(float("$MAX_TRAIN") > 0 and float("$MAX_SAMPLE") > float("$MAX_TRAIN")))
PY
)" -eq 1 ]; then
    echo "Error: max sample seconds ($MAX_SAMPLE) exceeds --max_train ($MAX_TRAIN)"
    exit 1
fi

RUN_TAG_PREFIX=""
if [ -n "$RUN_TAG" ]; then
    RUN_TAG_PREFIX="${RUN_TAG}_"
    RESULTS_DIR="${RESULTS_DIR}/results_${RUN_TAG}"
fi

# Ensure results directory exists before logging
mkdir -p "$RESULTS_DIR"

# Log resolved parameters (including defaults) to results directory
PARAMS_JSON="$RESULTS_DIR/run_params_classify_detect_bench.json"
printf '{\n' > "$PARAMS_JSON"
printf '  "command": "%s",\n' "$0 ${ORIGINAL_ARGS[*]}" >> "$PARAMS_JSON"
printf '  "spec_root": "%s",\n' "$SPEC_ROOT" >> "$PARAMS_JSON"
printf '  "annotation_root": "%s",\n' "$ANNOTATION_ROOT" >> "$PARAMS_JSON"
printf '  "results_dir": "%s",\n' "$RESULTS_DIR" >> "$PARAMS_JSON"
printf '  "pretrained_run": "%s",\n' "$PRETRAINED_RUN" >> "$PARAMS_JSON"
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
printf '  "class_weighting": %s,\n' "$CLASS_WEIGHTING" >> "$PARAMS_JSON"
printf '  "species_filter": "%s",\n' "$SPECIES_FILTER" >> "$PARAMS_JSON"
printf '  "sample_seconds": [' >> "$PARAMS_JSON"
for i in "${!SAMPLE_SECONDS[@]}"; do
    if [ "$i" -gt 0 ]; then printf ', ' >> "$PARAMS_JSON"; fi
    printf '%s' "${SAMPLE_SECONDS[$i]}" >> "$PARAMS_JSON"
done
printf ']\n' >> "$PARAMS_JSON"
printf '}\n' >> "$PARAMS_JSON"

echo " Configuration:"
echo "   SPEC_ROOT: $SPEC_ROOT"
echo "   RESULTS_DIR: $RESULTS_DIR"
echo "   PRETRAINED_RUN: $PRETRAINED_RUN"
echo "   TASK_MODE: $TASK_MODE"
echo "   PROBE_MODE: $PROBE_MODE"
echo "   MAX_BIRDS: $MAX_BIRDS"
echo "   SAMPLE_SECONDS: ${SAMPLE_SECONDS[*]}"
echo "   POOL_SIZE: $POOL_SIZE"
echo "   MAX_TRAIN: $MAX_TRAIN"
echo "   POOL_SEED: $POOL_SEED"
echo "   RUN_TAG: $RUN_TAG"
echo "   CLASS_WEIGHTING: $CLASS_WEIGHTING"
echo "   CONTEXT_TIMEBINS: $CONTEXT_TIMEBINS"
echo "   SPECIES_FILTER: $SPECIES_FILTER"

# Species Map: "SpeciesName:AnnotationFile:SpecSubDir"
# SpecSubDir is the folder name inside SPEC_ROOT containing .npy files
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

# =================================================

RESULTS_CSV="$RESULTS_DIR/results.csv"
# Initialize CSV if not exists
if [ ! -f "$RESULTS_CSV" ]; then
    echo "task,species,individual,samples,run_name,metric_name,metric_value" > "$RESULTS_CSV"
fi

# Resolve probe modes list
if [ "$PROBE_MODE" == "all" ]; then
    PROBE_MODES=("linear" "finetune")
else
    IFS=',' read -r -a PROBE_MODES <<< "$PROBE_MODE"
fi

echo "Results will be saved to $RESULTS_DIR"
WORK_ROOT="$PROJECT_ROOT/temp"
EVAL_DIR="$RESULTS_DIR/eval"
RUNS_LIST_FILE="$EVAL_DIR/run_names.txt"
mkdir -p "$EVAL_DIR"
: > "$RUNS_LIST_FILE"
mkdir -p "$WORK_ROOT"

# Task helper
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

PRETRAIN_AUDIO_PARAMS="${PRETRAINED_RUN%/}/audio_params.json"
PRETRAIN_AUDIO_PARAMS_WARNED=0

copy_pretrain_audio_params() {
    local target_dir="$1"
    if [ -z "$PRETRAIN_AUDIO_PARAMS" ]; then
        return
    fi
    if [ ! -f "$PRETRAIN_AUDIO_PARAMS" ]; then
        if [ "$PRETRAIN_AUDIO_PARAMS_WARNED" -eq 0 ]; then
            echo "Warning: audio_params.json not found at $PRETRAIN_AUDIO_PARAMS"
            PRETRAIN_AUDIO_PARAMS_WARNED=1
        fi
        return
    fi
    if [ -d "$target_dir" ]; then
        cp -f "$PRETRAIN_AUDIO_PARAMS" "$target_dir/audio_params.json"
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
    copy_pretrain_audio_params "$out_dir"
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

CLASS_WEIGHTING_FLAG="--no-class_weighting"
if [ "$CLASS_WEIGHTING" -eq 1 ]; then
    CLASS_WEIGHTING_FLAG="--class_weighting"
fi

record_run_name() {
    local name="$1"
    if [ -z "$name" ]; then
        return
    fi
    if ! grep -Fxq "$name" "$RUNS_LIST_FILE" 2>/dev/null; then
        echo "$name" >> "$RUNS_LIST_FILE"
    fi
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

filtered = dict(data)
filtered["recordings"] = recordings
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
print(f"Wrote filtered annotations: {out_path}")
PY
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


# Loop over Probe Modes
for PROBE in "${PROBE_MODES[@]}"; do
    PROBE="${PROBE// /}"
    if [ "$PROBE" == "linear" ]; then
        PROBE_ARGS="--linear_probe --freeze_encoder --lr 1e-4 --warmup_steps 50 --min_lr 0.0"
        RUN_PREFIX="linear_"
        echo "Probe Mode: Linear Probe (Frozen Encoder)"
    elif [ "$PROBE" == "finetune" ]; then
        # Finetune defaults: constant lr (no warmup/decay)
        PROBE_ARGS="--lr 1e-4 --warmup_steps 50 --min_lr 0.0"
        RUN_PREFIX="finetune_"
        echo "Probe Mode: Finetune (Unfrozen Encoder)"
    else
        echo "Unknown probe mode: $PROBE (skipping)"
        continue
    fi

    RUN_PREFIX="${RUN_PREFIX}${RUN_TAG_PREFIX}"

    # Loop over Species
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
        # Try to find it?
        continue
    fi

    USE_FIXED_POOL=0
    if [ "$(python - <<PY
print(int(float("$POOL_SIZE") > 0 and float("$MAX_TRAIN") > 0))
PY
)" -eq 1 ]; then
        USE_FIXED_POOL=1
        echo "  Pool seconds: $POOL_SIZE"
        echo "  Max train seconds: $MAX_TRAIN"
    fi

    # =========================================
    # TASK 1: DETECTION (Species Level)
    # =========================================
    if task_enabled "detect"; then
        echo "--- Starting Detection Benchmark for $SPECIES ---"
        
        # 1. Prepare Fixed Test Set (Pool vs Test)
        DET_WORK_DIR="$SPECIES_WORK_DIR/detect"
        mkdir -p "$DET_WORK_DIR"
        
        POOL_DIR="$DET_WORK_DIR/pool"
        TEST_DIR="$DET_WORK_DIR/test"
        
        if [ "$USE_FIXED_POOL" -eq 1 ]; then
            echo "  Creating Fixed Pool/Test Split (pool=${POOL_SIZE}s, max_train=${MAX_TRAIN}s)..."
            make_fixed_pool_seconds "$SPEC_DIR" "$POOL_DIR" "$TEST_DIR" "$POOL_SIZE" "$MAX_TRAIN" "$POOL_SEED"
        else
            # Check if already prepared
            if [ ! -d "$TEST_DIR" ] || ! has_npy "$TEST_DIR"; then
                echo "  Creating Test Set..."
                # We use split_train_test.py to split ALL species data into Pool (Train) and Test
                # We ignore bird_id for detection split to mix all individuals? 
                # Or do we keep bird separation? 
                # "there we only need to keep seperate species" -> Combine all birds.
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

        copy_pretrain_audio_params "$POOL_DIR"
        copy_pretrain_audio_params "$TEST_DIR"
        
        # 2. Train with varying sample sizes
        for SECONDS in "${SAMPLE_SECONDS[@]}"; do
            SECONDS_TAG="${SECONDS//./p}"
            echo "  Running Detection with ${SECONDS}s..."
            RUN_NAME="${RUN_PREFIX}${SPECIES}_detect_${SECONDS_TAG}s"
            TRAIN_DIR="$DET_WORK_DIR/train_${SECONDS_TAG}s"
            record_run_name "$RUN_NAME"
            
            # Prepare Train Set
            copy_train_subset_seconds "$POOL_DIR" "$TRAIN_DIR" "$SECONDS" "$POOL_SEED"
            
            # Run Training
            # Check if already run (log exists)
            LOG_FILE="runs/$RUN_NAME/loss_log.txt"
            if [ ! -f "$LOG_FILE" ]; then
                python src/supervised_train.py \
                    --train_dir "$TRAIN_DIR" \
                    --val_dir "$TEST_DIR" \
                    --run_name "$RUN_NAME" \
                    --pretrained_run "$PRETRAINED_RUN" \
                    --annotation_file "$ANNOT_PATH" \
                    --mode detect \
                    --steps "$STEPS" \
                    --batch_size "$BATCH_SIZE" \
                    --val_batch_size 1 \
                    --num_workers "$NUM_WORKERS" \
                    --amp \
                    --no-save_intermediate_checkpoints \
                    $PROBE_ARGS \
                    $CLASS_WEIGHTING_FLAG
            else
                echo "    Skipping training (run exists)"
            fi
            
            # Extract Metrics
            # Col 7 is val_f1
            VAL_F1=$(tail -n +2 "$LOG_FILE" | cut -d',' -f7 | sort -n | tail -n 1)
            # Handle empty/nan
            if [ -z "$VAL_F1" ]; then VAL_F1="0"; fi
            
            # Error = 100 - F1
            ERROR=$(python -c "print(100 - float('$VAL_F1'))")
            
            echo "detect,$SPECIES,all,$SECONDS,$RUN_NAME,F1_Error,$ERROR" >> "$RESULTS_CSV"
            echo "    Result: F1 Error = $ERROR %"
        done

        # Cleanup copied spec files for this task to avoid filling disk
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

        # Discover Individuals
        BIRDS=$(python -c "import json; d=json.load(open('$ANNOT_PATH')); print(' '.join(sorted(list(set(r['recording']['bird_id'] for r in d['recordings'])))))")
        echo "  Found individuals: $BIRDS"

        BIRD_COUNT=0
        for BIRD in $BIRDS; do
            if [ "$BIRD_COUNT" -ge "$MAX_BIRDS" ]; then
                echo "  Reached max birds ($MAX_BIRDS). Skipping remaining individuals."
                break
            fi
            # With `set -e`, `((var++))` can exit the script on the first iteration (returns status 1 when var was 0).
            ((BIRD_COUNT+=1))

            echo "  Processing Individual: $BIRD"
            UNIT_BIRD_DIR="$UNIT_WORK_DIR/$BIRD"
            mkdir -p "$UNIT_BIRD_DIR"

            BIRD_TRAIN_POOL="$UNIT_BIRD_DIR/pool_train"
            BIRD_TEST="$UNIT_BIRD_DIR/test"
            BIRD_POOL_FIXED="$UNIT_BIRD_DIR/pool_fixed"

            # 2. Split into Train Pool and Test (random within bird)
            if [ "$USE_FIXED_POOL" -eq 1 ]; then
                make_fixed_pool_seconds "$SPEC_DIR" "$BIRD_POOL_FIXED" "$BIRD_TEST" "$POOL_SIZE" "$MAX_TRAIN" "$POOL_SEED" "$ANNOT_PATH" "$BIRD" "$CONTEXT_TIMEBINS"
                TRAIN_POOL_DIR="$BIRD_POOL_FIXED"
            else
                if [ ! -d "$BIRD_TEST" ] || ! has_npy "$BIRD_TEST"; then
                    FILTERED_ANNOT="$UNIT_BIRD_DIR/annotations_filtered.json"
                    write_filtered_annotations "$ANNOT_PATH" "$BIRD" "$FILTERED_ANNOT"
                    python scripts/split_train_test.py \
                        --mode split \
                        --spec_dir "$SPEC_DIR" \
                        --train_dir "$BIRD_TRAIN_POOL" \
                        --test_dir "$BIRD_TEST" \
                        --annotation_json "$FILTERED_ANNOT" \
                        --train_percent $((100 - TEST_PERCENT)) \
                        --ignore_bird_id
                fi
                TRAIN_POOL_DIR="$BIRD_TRAIN_POOL"
                make_train_order "$TRAIN_POOL_DIR" "$POOL_SEED"
            fi

            copy_pretrain_audio_params "$TRAIN_POOL_DIR"
            copy_pretrain_audio_params "$BIRD_TEST"

            # 3. Train with varying sample sizes
            for SECONDS in "${SAMPLE_SECONDS[@]}"; do
                SECONDS_TAG="${SECONDS//./p}"
                echo "    Running Unit Detection with ${SECONDS}s..."
                RUN_NAME="${RUN_PREFIX}${SPECIES}_${BIRD}_unit_detect_${SECONDS_TAG}s"
                TRAIN_DIR="$UNIT_BIRD_DIR/train_${SECONDS_TAG}s"
                record_run_name "$RUN_NAME"

                # Prepare Train Set
                copy_train_subset_seconds "$TRAIN_POOL_DIR" "$TRAIN_DIR" "$SECONDS" "$POOL_SEED" "$ANNOT_PATH" "$BIRD" 1 "$CONTEXT_TIMEBINS"

                # Run Training
                LOG_FILE="runs/$RUN_NAME/loss_log.txt"
                if [ ! -f "$LOG_FILE" ]; then
                    python src/supervised_train.py \
                        --train_dir "$TRAIN_DIR" \
                        --val_dir "$BIRD_TEST" \
                        --run_name "$RUN_NAME" \
                        --pretrained_run "$PRETRAINED_RUN" \
                        --annotation_file "$ANNOT_PATH" \
                        --mode unit_detect \
                        --steps "$STEPS" \
                        --batch_size "$BATCH_SIZE" \
                    --val_batch_size 1 \
                        --num_workers "$NUM_WORKERS" \
                        --amp \
                        --no-save_intermediate_checkpoints \
                        $PROBE_ARGS \
                        $CLASS_WEIGHTING_FLAG
                else
                    echo "    Skipping training (run exists)"
                fi

                # Extract Metrics
                # Col 7 is val_f1 (same logging format as detect)
                VAL_F1=$(tail -n +2 "$LOG_FILE" | cut -d',' -f7 | sort -n | tail -n 1)
                if [ -z "$VAL_F1" ]; then VAL_F1="0"; fi

                # Error = 100 - F1
                ERROR=$(python -c "print(100 - float('$VAL_F1'))")

                echo "unit_detect,$SPECIES,$BIRD,$SECONDS,$RUN_NAME,F1_Error,$ERROR" >> "$RESULTS_CSV"
                echo "      Result: F1 Error = $ERROR %"
            done

            # Cleanup copied spec files for this bird to avoid filling disk
            echo "  Cleaning up copied Unit Detection files for $SPECIES / $BIRD..."
            rm -rf "$UNIT_BIRD_DIR"
        done
    else
        echo "Skipping Unit Detection Task (TASK_MODE=$TASK_MODE)"
    fi

    # =========================================
    # TASK 2: CLASSIFICATION (Individual Level)
    # =========================================
    if task_enabled "classify"; then
        echo "--- Starting Classification Benchmark for $SPECIES ---"
        
        # Discover Individuals
        # Use python to parse json and list bird_ids
        BIRDS=$(python -c "import json; d=json.load(open('$ANNOT_PATH')); print(' '.join(sorted(list(set(r['recording']['bird_id'] for r in d['recordings'])))))")
        
        echo "  Found individuals: $BIRDS"
        
        BIRD_COUNT=0
        for BIRD in $BIRDS; do
            if [ "$BIRD_COUNT" -ge "$MAX_BIRDS" ]; then
                echo "  Reached max birds ($MAX_BIRDS). Skipping remaining individuals."
                break
            fi
            # With `set -e`, `((var++))` can exit the script on the first iteration (returns status 1 when var was 0).
            ((BIRD_COUNT+=1))

            echo "  Processing Individual: $BIRD"
            CLS_WORK_DIR="$SPECIES_WORK_DIR/classify/$BIRD"
            mkdir -p "$CLS_WORK_DIR"
            
            BIRD_TRAIN_POOL="$CLS_WORK_DIR/pool_train"
            BIRD_TEST="$CLS_WORK_DIR/test"
            BIRD_POOL_FIXED="$CLS_WORK_DIR/pool_fixed"
            
            # 2. Split into Train Pool and Test
            if [ "$USE_FIXED_POOL" -eq 1 ]; then
                make_fixed_pool_seconds "$SPEC_DIR" "$BIRD_POOL_FIXED" "$BIRD_TEST" "$POOL_SIZE" "$MAX_TRAIN" "$POOL_SEED" "$ANNOT_PATH" "$BIRD"
                TRAIN_POOL_DIR="$BIRD_POOL_FIXED"
            else
                if [ ! -d "$BIRD_TEST" ] || ! has_npy "$BIRD_TEST"; then
                    FILTERED_ANNOT="$CLS_WORK_DIR/annotations_filtered.json"
                    write_filtered_annotations "$ANNOT_PATH" "$BIRD" "$FILTERED_ANNOT"
                    python scripts/split_train_test.py \
                        --mode split \
                        --spec_dir "$SPEC_DIR" \
                        --train_dir "$BIRD_TRAIN_POOL" \
                        --test_dir "$BIRD_TEST" \
                        --annotation_json "$FILTERED_ANNOT" \
                        --train_percent $((100 - TEST_PERCENT)) \
                        --ignore_bird_id # Already filtered by bird, so random split is fine
                fi
                TRAIN_POOL_DIR="$BIRD_TRAIN_POOL"
                make_train_order "$TRAIN_POOL_DIR" "$POOL_SEED"
            fi

            copy_pretrain_audio_params "$TRAIN_POOL_DIR"
            copy_pretrain_audio_params "$BIRD_TEST"
            
            # 3. Train with varying sample sizes
            for SECONDS in "${SAMPLE_SECONDS[@]}"; do
                SECONDS_TAG="${SECONDS//./p}"
                echo "    Running Classification with ${SECONDS}s..."
                RUN_NAME="${RUN_PREFIX}${SPECIES}_${BIRD}_classify_${SECONDS_TAG}s"
                TRAIN_DIR="$CLS_WORK_DIR/train_${SECONDS_TAG}s"
                record_run_name "$RUN_NAME"
                
                # Prepare Train Set
                copy_train_subset_seconds "$TRAIN_POOL_DIR" "$TRAIN_DIR" "$SECONDS" "$POOL_SEED"
                
                # Run Training
                LOG_FILE="runs/$RUN_NAME/loss_log.txt"
                if [ ! -f "$LOG_FILE" ]; then
                    python src/supervised_train.py \
                        --train_dir "$TRAIN_DIR" \
                        --val_dir "$BIRD_TEST" \
                        --run_name "$RUN_NAME" \
                        --pretrained_run "$PRETRAINED_RUN" \
                        --annotation_file "$ANNOT_PATH" \
                        --mode classify \
                        --steps "$STEPS" \
                        --batch_size "$BATCH_SIZE" \
                    --val_batch_size 1 \
                        --num_workers "$NUM_WORKERS" \
                        --amp \
                        --no-save_intermediate_checkpoints \
                        $PROBE_ARGS \
                        $CLASS_WEIGHTING_FLAG
                else
                    echo "      Skipping training (run exists)"
                fi
                
                # Extract Metrics
                # Col 5 is val_acc
                VAL_ACC=$(tail -n +2 "$LOG_FILE" | cut -d',' -f5 | sort -n | tail -n 1)
                if [ -z "$VAL_ACC" ]; then VAL_ACC="0"; fi
                
                # FER = 100 - Accuracy
                FER=$(python -c "print(100 - float('$VAL_ACC'))")
                
                echo "classify,$SPECIES,$BIRD,$SECONDS,$RUN_NAME,FER,$FER" >> "$RESULTS_CSV"
                echo "      Result: FER = $FER %"
            done

            # Cleanup copied spec files for this bird to avoid filling disk
            echo "  Cleaning up copied Classification files for $SPECIES / $BIRD..."
            rm -rf "$CLS_WORK_DIR"
        done
    else
        echo "Skipping Classification Task (TASK_MODE=$TASK_MODE)"
    fi
    done
done

# Generate Plots
echo "Generating Plots..."
python -c "from src.plotting_utils import plot_benchmark_results; plot_benchmark_results('$RESULTS_CSV', '$RESULTS_DIR')"

echo "Evaluating val_outputs F1..."
python scripts/eval/eval_val_outputs_f1.py \
    --runs_root "$PROJECT_ROOT/runs" \
    --run_names_file "$RUNS_LIST_FILE" \
    --out_csv "$EVAL_DIR/eval_f1.csv" \
    --summary_csv "$EVAL_DIR/eval_f1_summary.csv"

python - <<PY
import csv
from pathlib import Path

eval_csv = Path("$EVAL_DIR") / "eval_f1.csv"
run_list_path = Path("$RUNS_LIST_FILE")
if not eval_csv.exists() or not run_list_path.exists():
    raise SystemExit(0)

run_names = {line.strip() for line in run_list_path.read_text(encoding="utf-8").splitlines() if line.strip()}
if not run_names:
    raise SystemExit(0)

rows = []
with eval_csv.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("run_name") in run_names:
            rows.append(row)

if not rows:
    raise SystemExit(0)

print("F1 results (val_outputs):")
print("run_name\tmode\tspecies\tf1")
for row in rows:
    print(f"{row.get('run_name','')}\t{row.get('mode','')}\t{row.get('species','')}\t{row.get('f1','')}")
PY

# Cleanup temporary work directories
echo "Cleaning up temporary files..."
rm -rf "$WORK_ROOT"

echo "Benchmark Completed! See $RESULTS_DIR"
