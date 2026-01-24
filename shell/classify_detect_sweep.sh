#!/bin/bash

# classify_detect_sweep.sh
# Wrapper around classify_detect_bench.sh to sweep LoRA and finetune configs.

set -e

# Navigate to project root
cd "$(dirname "$0")/.."

BASE_ARGS=()
RUN_TAG_PREFIX=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_tag)
        RUN_TAG_PREFIX="$2"
        shift 2
        ;;
        --help|-h)
        echo "Usage: $0 [classify_detect_bench.sh args...]"
        echo "  Optional: --run_tag <prefix> (used as a prefix for sweep run tags)"
        exit 0
        ;;
        *)
        BASE_ARGS+=("$1")
        shift 1
        ;;
    esac
done

if [ -n "$RUN_TAG_PREFIX" ]; then
    RUN_TAG_PREFIX="${RUN_TAG_PREFIX}_"
fi

LORA_RANKS=(1 2 4 16)
FINETUNE_LABELS=("finetune_full" "finetune_freeze3" "finetune_last1")
FINETUNE_FREEZE_UP_TO=("" "2" "-2")
LRS=("1e-3" "1e-4" "5e-5")
CW_FLAGS=("--class_weighting" "--no_class_weighting")

for LR in "${LRS[@]}"; do
    for CW_FLAG in "${CW_FLAGS[@]}"; do
        if [ "$CW_FLAG" == "--class_weighting" ]; then
            CW_LABEL="cw"
        else
            CW_LABEL="nocw"
        fi

        for RANK in "${LORA_RANKS[@]}"; do
            RUN_TAG="${RUN_TAG_PREFIX}lora_r${RANK}_lr${LR}_${CW_LABEL}"
            echo "=== Sweep: lora rank=${RANK}, lr=${LR}, class_weighting=${CW_LABEL} ==="
            bash shell/classify_detect_bench.sh \
                "${BASE_ARGS[@]}" \
                --probe_mode lora \
                --lora_rank "$RANK" \
                --lr "$LR" \
                "$CW_FLAG" \
                --run_tag "$RUN_TAG"
        done

        for idx in "${!FINETUNE_LABELS[@]}"; do
            LABEL="${FINETUNE_LABELS[$idx]}"
            FREEZE_UP_TO="${FINETUNE_FREEZE_UP_TO[$idx]}"
            RUN_TAG="${RUN_TAG_PREFIX}${LABEL}_lr${LR}_${CW_LABEL}"
            echo "=== Sweep: ${LABEL}, lr=${LR}, class_weighting=${CW_LABEL} ==="
            CMD=(
                bash shell/classify_detect_bench.sh
                "${BASE_ARGS[@]}"
                --probe_mode finetune
                --lr "$LR"
                "$CW_FLAG"
                --run_tag "$RUN_TAG"
            )
            if [ -n "$FREEZE_UP_TO" ]; then
                CMD+=(--finetune_freeze_up_to "$FREEZE_UP_TO")
            fi
            "${CMD[@]}"
        done
    done
done
