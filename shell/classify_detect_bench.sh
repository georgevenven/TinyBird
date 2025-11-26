#!/bin/bash

# classify_detect_bench.sh
# Benchmark script for TinyBird detection and classification
# Generates Error vs Training Samples plots

set -e

# Navigate to project root
cd "$(dirname "$0")/.."

# ================= CONFIGURATION =================
# PATHS - PLEASE UPDATE THESE IF NEEDED
# Assuming spectrograms are in a 'spectrograms' folder or similar.
# Since I couldn't automatically locate them, please set SPEC_ROOT.
SPEC_ROOT="/Users/georgev/Documents/data/SongMAE_Bench_Data" 
ANNOTATION_ROOT="/Users/georgev/Documents/data/SongMAE_Bench_Data"
RESULTS_DIR="results/benchmark"
PRETRAINED_RUN="/Users/georgev/Documents/codebases/TinyBird/runs/tinybird_pretrain_20251122_091539" # the pretrained model 

# Experiment Settings
SAMPLE_SIZES=(1 2 4 8 16)
TEST_PERCENT=20
STEPS=250
BATCH_SIZE=24
NUM_WORKERS=4
MAX_BIRDS=3

# Probe Type: "linear" (Freeze Encoder) or "finetune" (MLP + Unfreeze)
PROBE_MODE="finetune"

# Task Selection: "both", "detect", or "classify"
TASK_MODE="classify"

# ================= ARGUMENT PARSING =================
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
        *)
        echo "Unknown argument: $1"
        shift 1
        ;;
    esac
done

echo " Configuration:"
echo "   SPEC_ROOT: $SPEC_ROOT"
echo "   RESULTS_DIR: $RESULTS_DIR"
echo "   PRETRAINED_RUN: $PRETRAINED_RUN"
echo "   TASK_MODE: $TASK_MODE"
echo "   PROBE_MODE: $PROBE_MODE"
echo "   MAX_BIRDS: $MAX_BIRDS"


# Species Map: "SpeciesName:AnnotationFile:SpecSubDir"
# SpecSubDir is the folder name inside SPEC_ROOT containing .npy files
SPECIES_LIST=(
    "BengaleseFinch:bf_annotations.json:bengalese_finch"
    "Canary:canary_annotations.json:canary"
    "ZebraFinch:zf_annotations.json:zebra_finch"
)

# =================================================

# Setup
mkdir -p "$RESULTS_DIR"
RESULTS_CSV="$RESULTS_DIR/results.csv"
# Initialize CSV if not exists
if [ ! -f "$RESULTS_CSV" ]; then
    echo "task,species,individual,samples,metric_name,metric_value" > "$RESULTS_CSV"
fi

# Set Probe Arguments
if [ "$PROBE_MODE" == "linear" ]; then
    PROBE_ARGS="--linear_probe --freeze_encoder --lr 1e-2"
    echo "Mode: Linear Probe (Frozen Encoder)"
else
    PROBE_ARGS="--lr 1e-4" # Lower LR for fine-tuning
    echo "Mode: Fine-tuning (Unfrozen Encoder + MLP)"
fi

echo "Results will be saved to $RESULTS_DIR"

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
    if [ "$TASK_MODE" == "detect" ] || [ "$TASK_MODE" == "both" ]; then
        echo "--- Starting Detection Benchmark for $SPECIES ---"
        
        # 1. Prepare Fixed Test Set (Pool vs Test)
        DET_WORK_DIR="$SPECIES_WORK_DIR/detect"
        mkdir -p "$DET_WORK_DIR"
        
        POOL_DIR="$DET_WORK_DIR/pool"
        TEST_DIR="$DET_WORK_DIR/test"
        
        # Check if already prepared
        if [ ! -d "$TEST_DIR" ] || [ -z "$(ls -A $TEST_DIR)" ]; then
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
        
        # 2. Train with varying sample sizes
        for N in "${SAMPLE_SIZES[@]}"; do
            echo "  Running Detection with N=$N samples..."
            RUN_NAME="${SPECIES}_detect_${N}"
            TRAIN_DIR="$DET_WORK_DIR/train_${N}"
            
            # Prepare Train Set
            if [ ! -d "$TRAIN_DIR" ]; then
                python scripts/split_train_test.py \
                    --mode sample \
                    --spec_dir "$POOL_DIR" \
                    --train_dir "$TRAIN_DIR" \
                    --n_samples "$N"
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
            
            echo "detect,$SPECIES,all,$N,F1_Error,$ERROR" >> "$RESULTS_CSV"
            echo "    Result: F1 Error = $ERROR %"
        done
    else
        echo "Skipping Detection Task (TASK_MODE=$TASK_MODE)"
    fi

    # =========================================
    # TASK 2: CLASSIFICATION (Individual Level)
    # =========================================
    if [ "$TASK_MODE" == "classify" ] || [ "$TASK_MODE" == "both" ]; then
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
            ((BIRD_COUNT++))

            echo "  Processing Individual: $BIRD"
            CLS_WORK_DIR="$SPECIES_WORK_DIR/classify/$BIRD"
            mkdir -p "$CLS_WORK_DIR"
            
            BIRD_POOL="$CLS_WORK_DIR/pool_source"
            BIRD_TRAIN_POOL="$CLS_WORK_DIR/pool_train"
            BIRD_TEST="$CLS_WORK_DIR/test"
            
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
            if [ ! -d "$BIRD_TEST" ]; then
                python scripts/split_train_test.py \
                    --mode split \
                    --spec_dir "$BIRD_POOL" \
                    --train_dir "$BIRD_TRAIN_POOL" \
                    --test_dir "$BIRD_TEST" \
                    --annotation_json "$ANNOT_PATH" \
                    --train_percent $((100 - TEST_PERCENT)) \
                    --ignore_bird_id # Already filtered by bird, so random split is fine
            fi
            
            # 3. Train with varying sample sizes
            for N in "${SAMPLE_SIZES[@]}"; do
                echo "    Running Classification with N=$N samples..."
                RUN_NAME="${SPECIES}_${BIRD}_classify_${N}"
                TRAIN_DIR="$CLS_WORK_DIR/train_${N}"
                
                # Prepare Train Set
                if [ ! -d "$TRAIN_DIR" ]; then
                    python scripts/split_train_test.py \
                        --mode sample \
                        --spec_dir "$BIRD_TRAIN_POOL" \
                        --train_dir "$TRAIN_DIR" \
                        --n_samples "$N"
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
                
                echo "classify,$SPECIES,$BIRD,$N,FER,$FER" >> "$RESULTS_CSV"
                echo "      Result: FER = $FER %"
            done
        done
    else
        echo "Skipping Classification Task (TASK_MODE=$TASK_MODE)"
    fi
done

# Generate Plots
echo "Generating Plots..."
python -c "from src.plotting_utils import plot_benchmark_results; plot_benchmark_results('$RESULTS_CSV', '$RESULTS_DIR')"

# Cleanup temporary work directories
echo "Cleaning up temporary files..."
rm -rf "$RESULTS_DIR/work"

echo "Benchmark Completed! See $RESULTS_DIR"

