#!/bin/bash
#
# classify_layerwise_linear_probe.sh
# Runs a classification *linear probe* against each encoder layer and logs per-layer macro-F1 error (100 - best val_f1).
#
# Notes:
# - Uses `src/supervised_train.py --encoder_layer_idx <L> --linear_probe --freeze_encoder --log_f1`
# - For classify mode, F1 is macro-F1 over all classes (including silence=0), computed per time patch.
#

set -e

# Navigate to project root
cd "$(dirname "$0")/.."

# ================= CONFIGURATION =================
SPEC_ROOT="/media/george-vengrovski/disk2/specs"
ANNOTATION_ROOT="/home/george-vengrovski/Documents/projects/TinyBird/files"
RESULTS_DIR="results/layerwise_linear_probe"
PRETRAINED_RUN="/home/george-vengrovski/Documents/projects/TinyBird/runs/tinybird_pretrain_20251122_091539"
RUNS_SUBDIR="layerwise_linear"

# Experiment Settings
SAMPLE_SIZES=(100)
TEST_PERCENT=20
STEPS=1000
BATCH_SIZE=24
NUM_WORKERS=4
MAX_BIRDS=3

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
        *)
        echo "Unknown argument: $1"
        shift 1
        ;;
    esac
done

# Ensure results directory exists before logging
mkdir -p "$RESULTS_DIR"
RUNS_SUBDIR="${RUNS_SUBDIR%/}"

# Log resolved parameters (including defaults) to results directory
PARAMS_JSON="$RESULTS_DIR/run_params_classify_layerwise_linear_probe.json"
printf '{\n' > "$PARAMS_JSON"
printf '  "command": "%s",\n' "$0 ${ORIGINAL_ARGS[*]}" >> "$PARAMS_JSON"
printf '  "spec_root": "%s",\n' "$SPEC_ROOT" >> "$PARAMS_JSON"
printf '  "annotation_root": "%s",\n' "$ANNOTATION_ROOT" >> "$PARAMS_JSON"
printf '  "results_dir": "%s",\n' "$RESULTS_DIR" >> "$PARAMS_JSON"
printf '  "pretrained_run": "%s",\n' "$PRETRAINED_RUN" >> "$PARAMS_JSON"
printf '  "runs_subdir": "%s",\n' "$RUNS_SUBDIR" >> "$PARAMS_JSON"
printf '  "steps": %s,\n' "$STEPS" >> "$PARAMS_JSON"
printf '  "batch_size": %s,\n' "$BATCH_SIZE" >> "$PARAMS_JSON"
printf '  "num_workers": %s,\n' "$NUM_WORKERS" >> "$PARAMS_JSON"
printf '  "max_birds": %s,\n' "$MAX_BIRDS" >> "$PARAMS_JSON"
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
echo "   RUNS_SUBDIR: $RUNS_SUBDIR"
echo "   STEPS: $STEPS"
echo "   BATCH_SIZE: $BATCH_SIZE"
echo "   MAX_BIRDS: $MAX_BIRDS"

# Species Map: "SpeciesName:AnnotationFile:SpecSubDir"
SPECIES_LIST=(
    "Bengalese_Finch:bf_annotations.json:bf_64hop_32khz"
    "Canary:canary_annotations.json:canary_64hop_32khz"
    "Zebra_Finch:zf_annotations.json:zf_64hop_32khz"
)

# =================================================

RESULTS_CSV="$RESULTS_DIR/results.csv"
if [ ! -f "$RESULTS_CSV" ]; then
    echo "task,species,individual,layer,samples,run_name,metric_name,metric_value" > "$RESULTS_CSV"
fi

# Determine number of encoder layers from pretrained config.json
ENC_N_LAYER=$(python -c "import json, os; cfg=json.load(open(os.path.join('$PRETRAINED_RUN','config.json'))); print(int(cfg['enc_n_layer']))")
echo "Encoder layers detected: $ENC_N_LAYER"

# Linear probe args (frozen encoder)
PROBE_ARGS="--linear_probe --freeze_encoder --lr 1e-2 --log_f1"

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
        continue
    fi

    echo "--- Starting Layerwise Linear-Probe Classification for $SPECIES ---"

    # Discover Individuals
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
                --ignore_bird_id
        fi

        # 3. Layerwise probe
        for LAYER in $(seq 0 $((ENC_N_LAYER - 1))); do
            echo "    Probing encoder layer: $LAYER"

            for N in "${SAMPLE_SIZES[@]}"; do
                echo "      Running Classification with N=$N samples..."
                RUN_NAME_PREFIX=""
                if [ -n "$RUNS_SUBDIR" ]; then
                    RUN_NAME_PREFIX="${RUNS_SUBDIR}/"
                fi
                RUN_NAME="${RUN_NAME_PREFIX}${SPECIES}_${BIRD}_classify_${N}_layer${LAYER}"
                TRAIN_DIR="$CLS_WORK_DIR/train_${N}"

                if [ ! -d "$TRAIN_DIR" ]; then
                    python scripts/split_train_test.py \
                        --mode sample \
                        --spec_dir "$BIRD_TRAIN_POOL" \
                        --train_dir "$TRAIN_DIR" \
                        --n_samples "$N"
                fi

                LOG_FILE="runs/$RUN_NAME/loss_log.txt"
                if [ ! -f "$LOG_FILE" ]; then
                    python src/supervised_train.py \
                        --train_dir "$TRAIN_DIR" \
                        --val_dir "$BIRD_TEST" \
                        --run_name "$RUN_NAME" \
                        --pretrained_run "$PRETRAINED_RUN" \
                        --annotation_file "$ANNOT_PATH" \
                        --mode classify \
                        --encoder_layer_idx "$LAYER" \
                        --steps "$STEPS" \
                        --batch_size "$BATCH_SIZE" \
                        --num_workers "$NUM_WORKERS" \
                        --amp \
                        --no-save_intermediate_checkpoints \
                        $PROBE_ARGS
                else
                    echo "        Skipping training (run exists)"
                fi

                # Extract best validation macro-F1 from the log (column 7 is val_f1 with --log_f1)
                VAL_F1=$(tail -n +2 "$LOG_FILE" | cut -d',' -f7 | sort -n | tail -n 1)
                if [ -z "$VAL_F1" ]; then VAL_F1="0"; fi

                # Error = 100 - F1
                F1_ERROR=$(python -c "print(100 - float('$VAL_F1'))")

                echo "classify,$SPECIES,$BIRD,$LAYER,$N,$RUN_NAME,F1_Error,$F1_ERROR" >> "$RESULTS_CSV"
                echo "        Result: F1 Error = $F1_ERROR % (Val F1 = $VAL_F1 %)"
            done
        done

        # Cleanup copied spec files for this bird to avoid filling disk
        echo "  Cleaning up copied Classification files for $SPECIES / $BIRD..."
        rm -rf "$CLS_WORK_DIR"
    done
done

echo "Cleaning up temporary files..."
rm -rf "$RESULTS_DIR/work"

echo "Layerwise probe completed! Results: $RESULTS_CSV"

