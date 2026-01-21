#!/bin/bash

# classify_detect_bench.sh
# Benchmark script for TinyBird detection, unit detection, and classification
# Generates Error vs Training Samples plots

set -e

# Navigate to project root
cd "$(dirname "$0")/.."

# ================= CONFIGURATION =================
# PATHS - PLEASE UPDATE THESE IF NEEDED
# Assuming spectrograms are in a 'spectrograms' folder or similar.
# Since I couldn't automatically locate them, please set SPEC_ROOT.
SPEC_ROOT="/media/george-vengrovski/disk2/specs"
ANNOTATION_ROOT="/home/george-vengrovski/Documents/projects/TinyBird/files"
RESULTS_DIR="results/benchmark"
PRETRAINED_RUN="/home/george-vengrovski/Documents/projects/TinyBird/runs/tinybird_pretrain_20251122_091539" # the pretrained model 

# Experiment Settings
SAMPLE_SIZES=(100)
TEST_PERCENT=20
STEPS=100
BATCH_SIZE=24
NUM_WORKERS=4
MAX_BIRDS=3
POOL_SIZE=0
MAX_TRAIN=0
POOL_SEED=42
RUN_TAG=""

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
        --sample_sizes)
        IFS=',' read -r -a SAMPLE_SIZES <<< "$2"
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

# Validate pooled sampling config
if [ "$POOL_SIZE" -gt 0 ] || [ "$MAX_TRAIN" -gt 0 ]; then
    if [ "$POOL_SIZE" -le 0 ] || [ "$MAX_TRAIN" -le 0 ]; then
        echo "Error: --pool_size and --max_train must both be > 0"
        exit 1
    fi
    if [ -z "$RUN_TAG" ]; then
        RUN_TAG="pool${POOL_SIZE}_train${MAX_TRAIN}"
    fi
fi

MAX_SAMPLE=0
for N in "${SAMPLE_SIZES[@]}"; do
    if [ "$N" -gt "$MAX_SAMPLE" ]; then
        MAX_SAMPLE="$N"
    fi
done
if [ "$MAX_TRAIN" -gt 0 ] && [ "$MAX_SAMPLE" -gt "$MAX_TRAIN" ]; then
    echo "Error: max sample size ($MAX_SAMPLE) exceeds --max_train ($MAX_TRAIN)"
    exit 1
fi

RUN_TAG_PREFIX=""
if [ -n "$RUN_TAG" ]; then
    RUN_TAG_PREFIX="${RUN_TAG}_"
    RESULTS_DIR="${RESULTS_DIR}_${RUN_TAG}"
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
printf '  "sample_sizes": [' >> "$PARAMS_JSON"
for i in "${!SAMPLE_SIZES[@]}"; do
    if [ "$i" -gt 0 ]; then printf ', ' >> "$PARAMS_JSON"; fi
    printf '%s' "${SAMPLE_SIZES[$i]}" >> "$PARAMS_JSON"
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
echo "   SAMPLE_SIZES: ${SAMPLE_SIZES[*]}"
echo "   POOL_SIZE: $POOL_SIZE"
echo "   MAX_TRAIN: $MAX_TRAIN"
echo "   POOL_SEED: $POOL_SEED"
echo "   RUN_TAG: $RUN_TAG"


# Species Map: "SpeciesName:AnnotationFile:SpecSubDir"
# SpecSubDir is the folder name inside SPEC_ROOT containing .npy files
SPECIES_LIST=(
    "Bengalese_Finch:bf_annotations.json:bf_64hop_32khz"
    "Canary:canary_annotations.json:canary_64hop_32khz"
    "Zebra_Finch:zf_annotations.json:zf_64hop_32khz"
)

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

make_fixed_pool() {
    local src_dir="$1"
    local pool_dir="$2"
    local test_dir="$3"
    local pool_size="$4"
    local max_train="$5"
    local seed="$6"

    if [ ! -d "$pool_dir" ] || ! has_npy "$pool_dir"; then
        python scripts/split_train_test.py \
            --mode sample \
            --spec_dir "$src_dir" \
            --train_dir "$pool_dir" \
            --n_samples "$pool_size" \
            --seed "$seed"
    fi

    local actual_count
    actual_count=$(python - <<PY
from pathlib import Path
pool = Path("$pool_dir")
print(len(list(pool.glob("*.npy"))))
PY
)

    local test_count=$((actual_count - max_train))
    if [ "$test_count" -lt 0 ]; then
        test_count=0
    fi

    if [ "$test_count" -gt 0 ] && { [ ! -d "$test_dir" ] || ! has_npy "$test_dir"; }; then
        python scripts/split_train_test.py \
            --mode sample \
            --spec_dir "$pool_dir" \
            --train_dir "$test_dir" \
            --n_samples "$test_count" \
            --seed "$seed" \
            --move
    fi

    make_train_order "$pool_dir" "$seed"
}

copy_train_subset() {
    local pool_dir="$1"
    local train_dir="$2"
    local n_samples="$3"
    if [ ! -d "$train_dir" ] || ! has_npy "$train_dir"; then
        mkdir -p "$train_dir"
        python - <<PY
from pathlib import Path
import shutil

pool = Path("$pool_dir")
train = Path("$train_dir")
order = pool / "train_order.txt"
names = [line.strip() for line in order.read_text().splitlines() if line.strip()]
limit = int($n_samples)
for name in names[:limit]:
    src = pool / name
    if src.exists():
        shutil.copy2(src, train / src.name)

audio_params = pool / "audio_params.json"
if audio_params.exists():
    shutil.copy2(audio_params, train / audio_params.name)
PY
    fi
}

# Loop over Probe Modes
for PROBE in "${PROBE_MODES[@]}"; do
    PROBE="${PROBE// /}"
    if [ "$PROBE" == "linear" ]; then
        PROBE_ARGS="--linear_probe --freeze_encoder --lr 1e-2"
        RUN_PREFIX="linear_"
        echo "Probe Mode: Linear Probe (Frozen Encoder)"
    elif [ "$PROBE" == "finetune" ]; then
        # Finetune defaults: constant lr (no warmup/decay)
        PROBE_ARGS="--lr 5e-5"
        RUN_PREFIX="finetune_"
        echo "Probe Mode: Finetune (Unfrozen Encoder)"
    else
        echo "Unknown probe mode: $PROBE (skipping)"
        continue
    fi

    RUN_PREFIX="${RUN_PREFIX}${RUN_TAG_PREFIX}"

    # Loop over Species
    for ENTRY in "${SPECIES_LIST[@]}"; do
        IFS=":" read -r SPECIES ANNOT_FILE SPEC_SUBDIR <<< "$ENTRY"
    
    ANNOT_PATH="$ANNOTATION_ROOT/$ANNOT_FILE"
    SPEC_DIR="$SPEC_ROOT/$SPEC_SUBDIR"
    SPECIES_WORK_DIR="$RESULTS_DIR/work/$SPECIES"
    
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
        
        if [ "$POOL_SIZE" -gt 0 ] && [ "$MAX_TRAIN" -gt 0 ]; then
            echo "  Creating Fixed Pool/Test Split (pool=$POOL_SIZE, max_train=$MAX_TRAIN)..."
            make_fixed_pool "$SPEC_DIR" "$POOL_DIR" "$TEST_DIR" "$POOL_SIZE" "$MAX_TRAIN" "$POOL_SEED"
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
            else
                echo "  Test Set already exists."
            fi
        fi
        
        # 2. Train with varying sample sizes
        for N in "${SAMPLE_SIZES[@]}"; do
            echo "  Running Detection with N=$N samples..."
            RUN_NAME="${RUN_PREFIX}${SPECIES}_detect_${N}"
            TRAIN_DIR="$DET_WORK_DIR/train_${N}"
            
            # Prepare Train Set
            if [ "$POOL_SIZE" -gt 0 ] && [ "$MAX_TRAIN" -gt 0 ]; then
                copy_train_subset "$POOL_DIR" "$TRAIN_DIR" "$N"
            else
                if [ ! -d "$TRAIN_DIR" ]; then
                    python scripts/split_train_test.py \
                        --mode sample \
                        --spec_dir "$POOL_DIR" \
                        --train_dir "$TRAIN_DIR" \
                        --n_samples "$N"
                fi
            fi
            
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
                    --num_workers "$NUM_WORKERS" \
                    --amp \
                    --no-save_intermediate_checkpoints \
                    $PROBE_ARGS
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
            
            echo "detect,$SPECIES,all,$N,$RUN_NAME,F1_Error,$ERROR" >> "$RESULTS_CSV"
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

            BIRD_POOL="$UNIT_BIRD_DIR/pool_source"
            BIRD_TRAIN_POOL="$UNIT_BIRD_DIR/pool_train"
            BIRD_TEST="$UNIT_BIRD_DIR/test"
            BIRD_POOL_FIXED="$UNIT_BIRD_DIR/pool_fixed"

            # 1. Extract Bird Files
            if [ ! -d "$BIRD_POOL" ]; then
                python scripts/split_train_test.py \
                    --mode filter_bird \
                    --spec_dir "$SPEC_DIR" \
                    --train_dir "$BIRD_POOL" \
                    --annotation_json "$ANNOT_PATH" \
                    --bird_id "$BIRD"
            fi

            # 2. Split into Train Pool and Test (random within bird)
            if [ "$POOL_SIZE" -gt 0 ] && [ "$MAX_TRAIN" -gt 0 ]; then
                make_fixed_pool "$BIRD_POOL" "$BIRD_POOL_FIXED" "$BIRD_TEST" "$POOL_SIZE" "$MAX_TRAIN" "$POOL_SEED"
                TRAIN_POOL_DIR="$BIRD_POOL_FIXED"
            else
                if [ ! -d "$BIRD_TEST" ] || ! has_npy "$BIRD_TEST"; then
                    FILTERED_ANNOT="$BIRD_POOL/annotations_filtered.json"
                    if [ ! -f "$FILTERED_ANNOT" ]; then
                        FILTERED_ANNOT="$ANNOT_PATH"
                    fi
                    python scripts/split_train_test.py \
                        --mode split \
                        --spec_dir "$BIRD_POOL" \
                        --train_dir "$BIRD_TRAIN_POOL" \
                        --test_dir "$BIRD_TEST" \
                        --annotation_json "$FILTERED_ANNOT" \
                        --train_percent $((100 - TEST_PERCENT)) \
                        --ignore_bird_id
                fi
                TRAIN_POOL_DIR="$BIRD_TRAIN_POOL"
            fi

            # 3. Train with varying sample sizes
            for N in "${SAMPLE_SIZES[@]}"; do
                echo "    Running Unit Detection with N=$N samples..."
                RUN_NAME="${RUN_PREFIX}${SPECIES}_${BIRD}_unit_detect_${N}"
                TRAIN_DIR="$UNIT_BIRD_DIR/train_${N}"

                # Prepare Train Set
                if [ "$POOL_SIZE" -gt 0 ] && [ "$MAX_TRAIN" -gt 0 ]; then
                    copy_train_subset "$TRAIN_POOL_DIR" "$TRAIN_DIR" "$N"
                else
                    if [ ! -d "$TRAIN_DIR" ]; then
                        python scripts/split_train_test.py \
                            --mode sample \
                            --spec_dir "$TRAIN_POOL_DIR" \
                            --train_dir "$TRAIN_DIR" \
                            --n_samples "$N"
                    fi
                fi

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
                        --num_workers "$NUM_WORKERS" \
                        --amp \
                        --no-save_intermediate_checkpoints \
                        $PROBE_ARGS
                else
                    echo "    Skipping training (run exists)"
                fi

                # Extract Metrics
                # Col 7 is val_f1 (same logging format as detect)
                VAL_F1=$(tail -n +2 "$LOG_FILE" | cut -d',' -f7 | sort -n | tail -n 1)
                if [ -z "$VAL_F1" ]; then VAL_F1="0"; fi

                # Error = 100 - F1
                ERROR=$(python -c "print(100 - float('$VAL_F1'))")

                echo "unit_detect,$SPECIES,$BIRD,$N,$RUN_NAME,F1_Error,$ERROR" >> "$RESULTS_CSV"
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
            
            BIRD_POOL="$CLS_WORK_DIR/pool_source"
            BIRD_TRAIN_POOL="$CLS_WORK_DIR/pool_train"
            BIRD_TEST="$CLS_WORK_DIR/test"
            BIRD_POOL_FIXED="$CLS_WORK_DIR/pool_fixed"
            
            # 1. Extract Bird Files
            if [ ! -d "$BIRD_POOL" ]; then
                python scripts/split_train_test.py \
                    --mode filter_bird \
                    --spec_dir "$SPEC_DIR" \
                    --train_dir "$BIRD_POOL" \
                    --annotation_json "$ANNOT_PATH" \
                    --bird_id "$BIRD"
            fi
            
            # 2. Split into Train Pool and Test
            if [ "$POOL_SIZE" -gt 0 ] && [ "$MAX_TRAIN" -gt 0 ]; then
                make_fixed_pool "$BIRD_POOL" "$BIRD_POOL_FIXED" "$BIRD_TEST" "$POOL_SIZE" "$MAX_TRAIN" "$POOL_SEED"
                TRAIN_POOL_DIR="$BIRD_POOL_FIXED"
            else
                if [ ! -d "$BIRD_TEST" ] || ! has_npy "$BIRD_TEST"; then
                    # Use bird-filtered annotation JSON if present (more efficient than scanning full species annotations)
                    FILTERED_ANNOT="$BIRD_POOL/annotations_filtered.json"
                    if [ ! -f "$FILTERED_ANNOT" ]; then
                        FILTERED_ANNOT="$ANNOT_PATH"
                    fi
                    python scripts/split_train_test.py \
                        --mode split \
                        --spec_dir "$BIRD_POOL" \
                        --train_dir "$BIRD_TRAIN_POOL" \
                        --test_dir "$BIRD_TEST" \
                        --annotation_json "$FILTERED_ANNOT" \
                        --train_percent $((100 - TEST_PERCENT)) \
                        --ignore_bird_id # Already filtered by bird, so random split is fine
                fi
                TRAIN_POOL_DIR="$BIRD_TRAIN_POOL"
            fi
            
            # 3. Train with varying sample sizes
            for N in "${SAMPLE_SIZES[@]}"; do
                echo "    Running Classification with N=$N samples..."
                RUN_NAME="${RUN_PREFIX}${SPECIES}_${BIRD}_classify_${N}"
                TRAIN_DIR="$CLS_WORK_DIR/train_${N}"
                
                # Prepare Train Set
                if [ "$POOL_SIZE" -gt 0 ] && [ "$MAX_TRAIN" -gt 0 ]; then
                    copy_train_subset "$TRAIN_POOL_DIR" "$TRAIN_DIR" "$N"
                else
                    if [ ! -d "$TRAIN_DIR" ]; then
                        python scripts/split_train_test.py \
                            --mode sample \
                            --spec_dir "$TRAIN_POOL_DIR" \
                            --train_dir "$TRAIN_DIR" \
                            --n_samples "$N"
                    fi
                fi
                
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
                        --num_workers "$NUM_WORKERS" \
                        --amp \
                        --no-save_intermediate_checkpoints \
                        $PROBE_ARGS
                else
                    echo "      Skipping training (run exists)"
                fi
                
                # Extract Metrics
                # Col 5 is val_acc
                VAL_ACC=$(tail -n +2 "$LOG_FILE" | cut -d',' -f5 | sort -n | tail -n 1)
                if [ -z "$VAL_ACC" ]; then VAL_ACC="0"; fi
                
                # FER = 100 - Accuracy
                FER=$(python -c "print(100 - float('$VAL_ACC'))")
                
                echo "classify,$SPECIES,$BIRD,$N,$RUN_NAME,FER,$FER" >> "$RESULTS_CSV"
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

# Cleanup temporary work directories
echo "Cleaning up temporary files..."
rm -rf "$RESULTS_DIR/work"

echo "Benchmark Completed! See $RESULTS_DIR"
