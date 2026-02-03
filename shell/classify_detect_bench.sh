#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

# ====== Basic config (override via args) ======
SPEC_ROOT=""
ANNOTATION_ROOT="files"
TEMP_ROOT="$PROJECT_ROOT/temp"
PRETRAINED_RUN=""

# One bird at a time
SPECIES=""
BIRD_ID=""
SPEC_DIR=""
ANNOTATION_FILE=""
OUT_DIR=""
POOL_DIR=""
TEST_DIR=""
SEED=42
MODE="classify"
TRAIN_SECONDS=""
TRAIN_DIR=""
TEST_SAMPLE_DIR=""
RUN_TAG=""
PROBE_MODE="linear"
LORA_RANK=4
LORA_ALPHA=16
LORA_DROPOUT=0.0
STEPS=1000
LR=1e-3
BATCH_SIZE=24
NUM_WORKERS=8
WEIGHT_DECAY=0.1
EVAL_EVERY=100
RUN_NAME=""
PREP_ONLY=0
USE_PREPARED=0

# Species map: "Species:AnnotationFile:SpecSubDir"
SPECIES_LIST=(
    "Bengalese_Finch:bf_annotations.json:bf_64hop_32khz"
    "Canary:canary_annotations.json:canary_64hop_32khz"
    "Zebra_Finch:zf_annotations.json:zf_64hop_32khz"
)

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
        --temp_root)
        TEMP_ROOT="$2"
        shift 2
        ;;
        --species)
        SPECIES="$2"
        shift 2
        ;;
        --bird_id)
        BIRD_ID="$2"
        shift 2
        ;;
        --spec_dir)
        SPEC_DIR="$2"
        shift 2
        ;;
        --annotation_file)
        ANNOTATION_FILE="$2"
        shift 2
        ;;
        --out_dir)
        OUT_DIR="$2"
        shift 2
        ;;
        --seed)
        SEED="$2"
        shift 2
        ;;
        --mode)
        MODE="$2"
        shift 2
        ;;
        --train_seconds)
        TRAIN_SECONDS="$2"
        shift 2
        ;;
        --run_tag)
        RUN_TAG="$2"
        shift 2
        ;;
        --run_name)
        RUN_NAME="$2"
        shift 2
        ;;
        --probe_mode)
        PROBE_MODE="$2"
        shift 2
        ;;
        --lora_rank)
        LORA_RANK="$2"
        shift 2
        ;;
        --lora_alpha)
        LORA_ALPHA="$2"
        shift 2
        ;;
        --lora_dropout)
        LORA_DROPOUT="$2"
        shift 2
        ;;
        --steps)
        STEPS="$2"
        shift 2
        ;;
        --lr)
        LR="$2"
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
        --weight_decay)
        WEIGHT_DECAY="$2"
        shift 2
        ;;
        --eval_every)
        EVAL_EVERY="$2"
        shift 2
        ;;
        --pretrained_run)
        PRETRAINED_RUN="$2"
        shift 2
        ;;
        --prep_only)
        PREP_ONLY=1
        shift 1
        ;;
        --use_prepared)
        USE_PREPARED=1
        shift 1
        ;;
        *)
        shift 1
        ;;
    esac
done

if [ -z "$SPEC_DIR" ] || [ -z "$ANNOTATION_FILE" ]; then
    if [ -z "$SPECIES" ]; then
        exit 1
    fi
    if [ -z "$SPEC_ROOT" ]; then
        exit 1
    fi
    for ENTRY in "${SPECIES_LIST[@]}"; do
        IFS=":" read -r NAME ANNOT_FILE SPEC_SUBDIR <<< "$ENTRY"
        if [ "$NAME" == "$SPECIES" ]; then
            ANNOTATION_FILE="$ANNOTATION_ROOT/$ANNOT_FILE"
            SPEC_DIR="$SPEC_ROOT/$SPEC_SUBDIR"
            break
        fi
    done
fi

if [ -z "$SPEC_DIR" ] || [ -z "$ANNOTATION_FILE" ]; then
    exit 1
fi

if [ -z "$BIRD_ID" ]; then
    exit 1
fi

if [ -z "$OUT_DIR" ]; then
    if [ -n "$SPECIES" ]; then
        OUT_DIR="$TEMP_ROOT/tinybird_pool/$SPECIES/$BIRD_ID"
    else
        OUT_DIR="$TEMP_ROOT/tinybird_pool/$BIRD_ID"
    fi
fi
POOL_DIR="$OUT_DIR/pool"
TRAIN_DIR="$OUT_DIR/train"
TEST_SAMPLE_DIR="$OUT_DIR/test"

echo "Copying pool for bird:"
echo "  SPEC_DIR: $SPEC_DIR"
echo "  ANNOTATION_FILE: $ANNOTATION_FILE"
echo "  BIRD_ID: $BIRD_ID"
echo "  POOL_DIR: $POOL_DIR"
echo "  TEST_DIR: $TEST_SAMPLE_DIR"

if [ "$USE_PREPARED" -eq 0 ]; then
    python "$PROJECT_ROOT/src/bench_utils/copy_bird_pool.py" \
        --annotation_file "$ANNOTATION_FILE" \
        --spec_dir "$SPEC_DIR" \
        --out_dir "$POOL_DIR" \
        --bird_id "$BIRD_ID"

    if [ -z "$TRAIN_SECONDS" ]; then
        exit 1
    fi
    TRAIN_SECONDS_TAG="${TRAIN_SECONDS//./p}"
    TRAIN_DIR="$OUT_DIR/$MODE/$BIRD_ID/train_${TRAIN_SECONDS_TAG}s"

    python "$PROJECT_ROOT/src/bench_utils/solver_split_by_seconds.py" \
        --pool_dir "$POOL_DIR" \
        --annotation_json "$POOL_DIR/annotations_filtered.json" \
        --test_dir "$TEST_SAMPLE_DIR" \
        --train_dir "$TRAIN_DIR" \
        --train_seconds "$TRAIN_SECONDS" \
        --test_ratio "0.2" \
        --seed "$SEED"
else
    if [ -z "$TRAIN_SECONDS" ]; then
        exit 1
    fi
    TRAIN_SECONDS_TAG="${TRAIN_SECONDS//./p}"
    TRAIN_DIR="$OUT_DIR/$MODE/$BIRD_ID/train_${TRAIN_SECONDS_TAG}s"
fi

if [ "$PREP_ONLY" -eq 1 ]; then
    exit 0
fi

if [ -z "$RUN_TAG" ]; then
    exit 1
fi
if [ -z "$PRETRAINED_RUN" ]; then
    exit 1
fi

if [ -z "$RUN_NAME" ]; then
    RUN_NAME="$RUN_TAG"
fi

SUP_ARGS=()
if [ "$PROBE_MODE" == "linear" ]; then
    SUP_ARGS+=(--linear_probe --freeze_encoder)
elif [ "$PROBE_MODE" == "lora" ]; then
    SUP_ARGS+=(--lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT")
fi

if [ -d "$PROJECT_ROOT/runs/$RUN_TAG" ]; then
    :
else
    PYTHONWARNINGS=ignore python "$PROJECT_ROOT/src/supervised_train.py" \
        --train_dir "$TRAIN_DIR" \
        --val_dir "$TEST_SAMPLE_DIR" \
        --run_name "$RUN_TAG" \
        --pretrained_run "$PRETRAINED_RUN" \
        --annotation_file "$POOL_DIR/annotations_filtered.json" \
        --mode "$MODE" \
    --steps "$STEPS" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --weight_decay "$WEIGHT_DECAY" \
    --eval_every "$EVAL_EVERY" \
    --viz_last_only \
    "${SUP_ARGS[@]}"
fi

RESULTS_CSV="$PROJECT_ROOT/results/$RUN_TAG/eval_f1.csv"
if [[ "$RUN_TAG" == linear_probes/* ]]; then
    RESULTS_CSV="$PROJECT_ROOT/results/linear_probes/eval_f1.csv"
fi

python "$PROJECT_ROOT/scripts/eval/eval_val_outputs_f1.py" \
    --runs_root "$PROJECT_ROOT/runs" \
    --run_names "$RUN_TAG" \
    --out_csv "$RESULTS_CSV" \
    --append \
    --no_summary
