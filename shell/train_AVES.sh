#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

# ====== Basic config (override via args) ======
SPEC_ROOT=""
WAV_ROOT=""
ANNOTATION_ROOT="files"
TEMP_ROOT="$PROJECT_ROOT/temp"

AVES_MODEL="$PROJECT_ROOT/files/aves-base-bio.torchaudio.pt"
AVES_CONFIG="$PROJECT_ROOT/files/aves-base-bio.torchaudio.model_config.json"

SPECIES=""
BIRD_ID=""
SPEC_DIR=""
ANNOTATION_FILE=""
OUT_DIR=""
SEED=42
MODE="classify"
TRAIN_SECONDS=""
RUN_TAG=""
RUN_NAME=""
PROBE_MODE="linear"
PREP_ONLY=0
USE_PREPARED=0
SAVE_INTERMEDIATE_CHECKPOINTS=0

STEPS=1000
LR=1e-3
BATCH_SIZE=24
VAL_BATCH_SIZE=0
NUM_WORKERS=8
WEIGHT_DECAY=0.1
EVAL_EVERY=100
CLIP_SECONDS=2
AUDIO_SR=16000
EMBEDDING_DIM=768
EARLY_STOP_PATIENCE=4
EARLY_STOP_EMA_ALPHA=0.75
EARLY_STOP_MIN_DELTA=0.0
CLASS_WEIGHTING=0
MS_PER_TIMEBIN=""
NUM_TIMEBINS=0
WAV_EXTS=".wav,.flac,.ogg,.mp3"

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
        --wav_root)
        WAV_ROOT="$2"
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
        --aves_model)
        AVES_MODEL="$2"
        shift 2
        ;;
        --aves_config)
        AVES_CONFIG="$2"
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
        --val_batch_size)
        VAL_BATCH_SIZE="$2"
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
        --clip_seconds)
        CLIP_SECONDS="$2"
        shift 2
        ;;
        --audio_sr)
        AUDIO_SR="$2"
        shift 2
        ;;
        --embedding_dim)
        EMBEDDING_DIM="$2"
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
        --class_weighting)
        CLASS_WEIGHTING=1
        shift 1
        ;;
        --no_class_weighting)
        CLASS_WEIGHTING=0
        shift 1
        ;;
        --ms_per_timebin)
        MS_PER_TIMEBIN="$2"
        shift 2
        ;;
        --num_timebins)
        NUM_TIMEBINS="$2"
        shift 2
        ;;
        --wav_exts)
        WAV_EXTS="$2"
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
        --save_intermediate_checkpoints)
        SAVE_INTERMEDIATE_CHECKPOINTS=1
        shift 1
        ;;
        --no-save_intermediate_checkpoints)
        SAVE_INTERMEDIATE_CHECKPOINTS=0
        shift 1
        ;;
        *)
        shift 1
        ;;
    esac
done

if [ -z "$SPEC_DIR" ] || [ -z "$ANNOTATION_FILE" ]; then
    if [ -z "$SPECIES" ]; then
        echo "Missing --species or --spec_dir/--annotation_file" 1>&2
        exit 1
    fi
    if [ -z "$SPEC_ROOT" ]; then
        echo "Missing --spec_root" 1>&2
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
    echo "Unable to resolve spec_dir/annotation_file" 1>&2
    exit 1
fi

if [ -z "$WAV_ROOT" ]; then
    echo "Missing --wav_root" 1>&2
    exit 1
fi

if [ -z "$BIRD_ID" ]; then
    echo "Missing --bird_id" 1>&2
    exit 1
fi

if [ -z "$OUT_DIR" ]; then
    if [ -n "$SPECIES" ]; then
        OUT_DIR="$TEMP_ROOT/aves_pool/$SPECIES/$BIRD_ID"
    else
        OUT_DIR="$TEMP_ROOT/aves_pool/$BIRD_ID"
    fi
fi

POOL_DIR="$OUT_DIR/pool"
TEST_SAMPLE_DIR="$OUT_DIR/test"

if [ "$USE_PREPARED" -eq 0 ]; then
    rm -rf "$OUT_DIR"

    python "$PROJECT_ROOT/src/bench_utils/copy_bird_pool.py" \
        --annotation_file "$ANNOTATION_FILE" \
        --spec_dir "$SPEC_DIR" \
        --out_dir "$POOL_DIR" \
        --bird_id "$BIRD_ID"

    if [ -z "$TRAIN_SECONDS" ]; then
        echo "Missing --train_seconds" 1>&2
        exit 1
    fi
    TRAIN_SECONDS_TAG="${TRAIN_SECONDS//./p}"
    TRAIN_DIR="$OUT_DIR/$MODE/$BIRD_ID/train_${TRAIN_SECONDS_TAG}s"
    TEST_SAMPLE_DIR="$OUT_DIR/$MODE/$BIRD_ID/test"

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
        echo "Missing --train_seconds" 1>&2
        exit 1
    fi
    TRAIN_SECONDS_TAG="${TRAIN_SECONDS//./p}"
    TRAIN_DIR="$OUT_DIR/$MODE/$BIRD_ID/train_${TRAIN_SECONDS_TAG}s"
    TEST_SAMPLE_DIR="$OUT_DIR/$MODE/$BIRD_ID/test"
fi

if [ "$PREP_ONLY" -eq 1 ]; then
    exit 0
fi

if [ -z "$RUN_NAME" ]; then
    if [ -z "$RUN_TAG" ]; then
        echo "Missing --run_tag or --run_name" 1>&2
        exit 1
    fi
    RUN_NAME="$RUN_TAG"
fi

ANNOTATION_TRAIN="$POOL_DIR/annotations_filtered.json"
if [ ! -f "$ANNOTATION_TRAIN" ]; then
    ANNOTATION_TRAIN="$ANNOTATION_FILE"
fi

WAV_MANIFEST="$OUT_DIR/$MODE/$BIRD_ID/wav_manifest.json"
python "$PROJECT_ROOT/src/bench_utils/build_wav_manifest.py" \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$TEST_SAMPLE_DIR" \
    --wav_root "$WAV_ROOT" \
    --out_path "$WAV_MANIFEST" \
    --wav_exts "$WAV_EXTS"

PROBE_ARGS=()
if [ "$PROBE_MODE" == "linear" ]; then
    PROBE_ARGS+=(--linear_probe)
elif [ "$PROBE_MODE" == "finetune" ]; then
    PROBE_ARGS+=(--finetune)
else
    echo "Unknown --probe_mode: $PROBE_MODE" 1>&2
    exit 1
fi

CLASS_WEIGHTING_FLAG="--no-class_weighting"
if [ "$CLASS_WEIGHTING" -eq 1 ]; then
    CLASS_WEIGHTING_FLAG="--class_weighting"
fi

if [ -z "$MS_PER_TIMEBIN" ] && [ -f "$SPEC_DIR/audio_params.json" ]; then
    MS_PER_TIMEBIN=$(python "$PROJECT_ROOT/src/bench_utils/get_ms_per_timebin.py" \
        --spec_dir "$SPEC_DIR")
fi

EXTRA_ARGS=()
if [ -n "$MS_PER_TIMEBIN" ]; then
    EXTRA_ARGS+=(--ms_per_timebin "$MS_PER_TIMEBIN")
fi
if [ "$NUM_TIMEBINS" -gt 0 ]; then
    EXTRA_ARGS+=(--num_timebins "$NUM_TIMEBINS")
fi
if [ "$VAL_BATCH_SIZE" -gt 0 ]; then
    EXTRA_ARGS+=(--val_batch_size "$VAL_BATCH_SIZE")
fi
if [ "$SAVE_INTERMEDIATE_CHECKPOINTS" -eq 0 ]; then
    EXTRA_ARGS+=(--no-save_intermediate_checkpoints)
fi

PYTHONWARNINGS=ignore python "$PROJECT_ROOT/src/aves.py" \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$TEST_SAMPLE_DIR" \
    --run_name "$RUN_NAME" \
    --annotation_file "$ANNOTATION_TRAIN" \
    --mode "$MODE" \
    --wav_root "$WAV_ROOT" \
    --wav_manifest "$WAV_MANIFEST" \
    --wav_exts "$WAV_EXTS" \
    --aves_model_path "$AVES_MODEL" \
    --aves_config_path "$AVES_CONFIG" \
    --audio_sr "$AUDIO_SR" \
    --embedding_dim "$EMBEDDING_DIM" \
    --clip_seconds "$CLIP_SECONDS" \
    --steps "$STEPS" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --weight_decay "$WEIGHT_DECAY" \
    --eval_every "$EVAL_EVERY" \
    --log_f1 \
    $CLASS_WEIGHTING_FLAG \
    "${PROBE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
