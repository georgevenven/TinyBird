#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

SPEC_ROOT="/media/george-vengrovski/disk2/specs"
WAV_ROOT="/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae"
ANNOTATION_ROOT="files"
BIRD_LIST_JSON=""

TEMP_ROOT="$PROJECT_ROOT/temp/individual_id"
RESULTS_DIR="$PROJECT_ROOT/results/individual_id_linear_probes"
RUN_TAG_PREFIX="individual_id_linear_probes"

MODE="classify"
TRAIN_SECONDS="MAX"
SEED="42"
STEPS="1000"
AUDIO_PARAMS_SOURCE="spec"

SONGMAE_LR="1e-2"
AVES_LR="1e-2"
MAX_BIRDS=""

TARGET_SPECIES=""
TARGET_BIRD_ID=""
KEEP_TEMP=0

SONGMAE_RUN=""
AVES_CLIP_SECONDS=""

usage() {
    cat <<EOF_USAGE
Usage: $0 --songmae_run RUN [options]

Train per-individual linear probes on a shared train/test split for:
  1) SongMAE (TinyBird supervised linear probe)
  2) AVES (linear probe)

Defaults assume:
  SPEC_ROOT=/media/george-vengrovski/disk2/specs
  with subdirs canary_64hop_32khz, bf_64hop_32khz, zf_64hop_32khz

Required:
  --songmae_run PATH_OR_RUN_NAME      Pretrained SongMAE run (absolute path,
                                       project-relative path, or run name under runs/)

Optional:
  --spec_root PATH                     Default: ${SPEC_ROOT}
  --wav_root PATH                      Default: ${WAV_ROOT}
  --annotation_root PATH               Default: ${ANNOTATION_ROOT}
  --bird_list_json PATH                Optional list with [{"species","bird_id"}, ...]
  --temp_root PATH                     Default: ${TEMP_ROOT}
  --results_dir PATH                   Default: ${RESULTS_DIR}
  --run_tag_prefix STR                 Default: ${RUN_TAG_PREFIX}

  --train_seconds FLOAT|MAX            Default: ${TRAIN_SECONDS}
  --steps N                            Default: ${STEPS}
  --seed N                             Default: ${SEED}
  --songmae_lr LR                      Default: ${SONGMAE_LR}
  --aves_lr LR                         Default: ${AVES_LR}
  --audio_params_source pretrain|spec  Default: ${AUDIO_PARAMS_SOURCE}

  --species NAME                       Filter species (Canary|Zebra_Finch|Bengalese_Finch)
  --bird_id ID                         Filter one bird id (optionally with --species)
  --max_birds N                        Process at most N birds after filtering
  --aves_clip_seconds FLOAT            Override AVES clip length.
                                       If omitted, it is derived from SongMAE num_timebins
                                       and pool audio_params so both probes use same context length.
  --keep_temp                          Keep per-bird temp splits (default cleans them)
  -h, --help                           Show this help
EOF_USAGE
}

normalize_species() {
    local raw="$1"
    case "$raw" in
        Bengalese_Finch|BengaleseFinch|bengalese|bf)
            echo "Bengalese_Finch"
            ;;
        Zebra_Finch|ZebraFinch|zebra|zf)
            echo "Zebra_Finch"
            ;;
        Canary|canary)
            echo "Canary"
            ;;
        *)
            echo "$raw"
            ;;
    esac
}

species_mapping() {
    local species="$1"
    case "$species" in
        Bengalese_Finch)
            echo "bf_annotations.json:bf_64hop_32khz"
            ;;
        Canary)
            echo "canary_annotations.json:canary_64hop_32khz"
            ;;
        Zebra_Finch)
            echo "zf_annotations.json:zf_64hop_32khz"
            ;;
        *)
            return 1
            ;;
    esac
}

resolve_songmae_run() {
    local run_arg="$1"

    if [ -d "$run_arg" ]; then
        python - "$run_arg" <<'PY'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
        return 0
    fi

    if [ -d "$PROJECT_ROOT/$run_arg" ]; then
        python - "$PROJECT_ROOT/$run_arg" <<'PY'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
        return 0
    fi

    if [ -d "$PROJECT_ROOT/runs/$run_arg" ]; then
        python - "$PROJECT_ROOT/runs/$run_arg" <<'PY'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
        return 0
    fi

    return 1
}

derive_aves_clip_seconds() {
    local songmae_run_dir="$1"
    local pool_audio_params="$2"

    python - "$songmae_run_dir" "$pool_audio_params" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
audio_params_path = Path(sys.argv[2])
config_path = run_dir / "config.json"

if not config_path.exists():
    raise SystemExit(f"Missing SongMAE config: {config_path}")
if not audio_params_path.exists():
    raise SystemExit(f"Missing pool audio_params.json: {audio_params_path}")

config = json.loads(config_path.read_text(encoding="utf-8"))
audio = json.loads(audio_params_path.read_text(encoding="utf-8"))

num_timebins = config.get("num_timebins")
sr = audio.get("sr")
hop = audio.get("hop_size")

if num_timebins is None:
    raise SystemExit(f"num_timebins missing in {config_path}")
if sr is None or hop is None:
    raise SystemExit(f"sr/hop_size missing in {audio_params_path}")

num_timebins = float(num_timebins)
sr = float(sr)
hop = float(hop)
if sr <= 0 or hop <= 0:
    raise SystemExit(f"invalid sr/hop_size in {audio_params_path}")

clip_seconds = num_timebins * hop / sr
print(f"{clip_seconds:.6f}")
PY
}

build_bird_stream() {
    python - "$BIRD_LIST_JSON" "$ANNOTATION_ROOT" "$TARGET_SPECIES" "$TARGET_BIRD_ID" "$MAX_BIRDS" <<'PY'
import json
import sys
from pathlib import Path

bird_list_json = sys.argv[1].strip()
annotation_root = Path(sys.argv[2])
target_species = sys.argv[3].strip()
target_bird_id = sys.argv[4].strip()
max_birds_token = sys.argv[5].strip()

sources = {
    "Bengalese_Finch": "bf_annotations.json",
    "Canary": "canary_annotations.json",
    "Zebra_Finch": "zf_annotations.json",
}

def normalize_species(species: str) -> str:
    m = {
        "bengalese": "Bengalese_Finch",
        "bf": "Bengalese_Finch",
        "zebra": "Zebra_Finch",
        "zf": "Zebra_Finch",
        "canary": "Canary",
        "bengalese_finch": "Bengalese_Finch",
        "zebra_finch": "Zebra_Finch",
    }
    s = species.strip()
    if not s:
        return ""
    if s in sources:
        return s
    return m.get(s.lower(), s)

pairs = []

if bird_list_json:
    path = Path(bird_list_json)
    if not path.exists():
        raise SystemExit(f"--bird_list_json not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    for entry in data:
        species = normalize_species(str(entry.get("species", "")))
        bird_id = str(entry.get("bird_id", "")).strip()
        if species and bird_id:
            pairs.append((species, bird_id))
else:
    for species, ann_name in sources.items():
        ann_path = annotation_root / ann_name
        if not ann_path.exists():
            raise SystemExit(f"Annotation file not found: {ann_path}")
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        bird_ids = {
            str(rec.get("recording", {}).get("bird_id", "")).strip()
            for rec in data.get("recordings", [])
        }
        for bird_id in sorted(b for b in bird_ids if b):
            pairs.append((species, bird_id))

pairs = sorted(set(pairs), key=lambda x: (x[0], x[1]))

if target_species:
    target_species = normalize_species(target_species)
    pairs = [p for p in pairs if p[0] == target_species]
if target_bird_id:
    pairs = [p for p in pairs if p[1] == target_bird_id]

if max_birds_token:
    max_birds = int(max_birds_token)
    if max_birds > 0:
        pairs = pairs[:max_birds]

for species, bird_id in pairs:
    print(f"{species}:{bird_id}")
PY
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --songmae_run)
            SONGMAE_RUN="$2"
            shift 2
            ;;
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
        --bird_list_json)
            BIRD_LIST_JSON="$2"
            shift 2
            ;;
        --temp_root)
            TEMP_ROOT="$2"
            shift 2
            ;;
        --results_dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --run_tag_prefix)
            RUN_TAG_PREFIX="$2"
            shift 2
            ;;
        --train_seconds)
            TRAIN_SECONDS="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --songmae_lr)
            SONGMAE_LR="$2"
            shift 2
            ;;
        --aves_lr)
            AVES_LR="$2"
            shift 2
            ;;
        --audio_params_source)
            AUDIO_PARAMS_SOURCE="$2"
            shift 2
            ;;
        --species)
            TARGET_SPECIES="$(normalize_species "$2")"
            shift 2
            ;;
        --bird_id)
            TARGET_BIRD_ID="$2"
            shift 2
            ;;
        --max_birds)
            MAX_BIRDS="$2"
            shift 2
            ;;
        --aves_clip_seconds)
            AVES_CLIP_SECONDS="$2"
            shift 2
            ;;
        --keep_temp)
            KEEP_TEMP=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" 1>&2
            usage
            exit 1
            ;;
    esac
done

if [ -z "$SONGMAE_RUN" ]; then
    echo "Missing required --songmae_run" 1>&2
    usage
    exit 1
fi

if ! SONGMAE_RUN_RESOLVED="$(resolve_songmae_run "$SONGMAE_RUN")"; then
    echo "Unable to resolve --songmae_run: $SONGMAE_RUN" 1>&2
    exit 1
fi

if [ ! -f "$SONGMAE_RUN_RESOLVED/config.json" ]; then
    echo "SongMAE run missing config.json: $SONGMAE_RUN_RESOLVED" 1>&2
    exit 1
fi

if [ ! -d "$SPEC_ROOT" ]; then
    echo "spec_root not found: $SPEC_ROOT" 1>&2
    exit 1
fi
if [ ! -d "$WAV_ROOT" ]; then
    echo "wav_root not found: $WAV_ROOT" 1>&2
    exit 1
fi
if [ ! -d "$ANNOTATION_ROOT" ]; then
    echo "annotation_root not found: $ANNOTATION_ROOT" 1>&2
    exit 1
fi

mkdir -p "$TEMP_ROOT"
mkdir -p "$RESULTS_DIR"

SONGMAE_RESULTS_CSV="$RESULTS_DIR/songmae_eval_f1.csv"
AVES_RESULTS_CSV="$RESULTS_DIR/aves_eval_f1.csv"

PRE_NAME="$(basename "$SONGMAE_RUN_RESOLVED")"
BIRD_COUNT=0
SUCCESS_SONGMAE=0
SUCCESS_AVES=0

while IFS=: read -r SPECIES BIRD_ID; do
    if [ -z "$SPECIES" ] || [ -z "$BIRD_ID" ]; then
        continue
    fi

    if ! MAP="$(species_mapping "$SPECIES")"; then
        echo "Skipping unknown species mapping: $SPECIES" 1>&2
        continue
    fi

    IFS=':' read -r ANNOT_FILE SPEC_SUBDIR <<< "$MAP"

    SPEC_DIR="$SPEC_ROOT/$SPEC_SUBDIR"
    ANNOTATION_FILE="$ANNOTATION_ROOT/$ANNOT_FILE"

    if [ ! -d "$SPEC_DIR" ]; then
        echo "Skipping ${SPECIES}:${BIRD_ID} (missing spec dir: $SPEC_DIR)" 1>&2
        continue
    fi
    if [ ! -f "$ANNOTATION_FILE" ]; then
        echo "Skipping ${SPECIES}:${BIRD_ID} (missing annotation file: $ANNOTATION_FILE)" 1>&2
        continue
    fi

    BIRD_COUNT=$((BIRD_COUNT + 1))

    PREP_OUT_DIR="$TEMP_ROOT/$SPECIES/$BIRD_ID"
    rm -rf "$PREP_OUT_DIR"

    echo
    echo "=== ${SPECIES}:${BIRD_ID} ==="
    echo "Preparing shared split in: $PREP_OUT_DIR"

    if ! bash "$PROJECT_ROOT/shell/classify_detect_bench.sh" \
        --spec_root "$SPEC_ROOT" \
        --annotation_root "$ANNOTATION_ROOT" \
        --species "$SPECIES" \
        --bird_id "$BIRD_ID" \
        --train_seconds "$TRAIN_SECONDS" \
        --mode "$MODE" \
        --seed "$SEED" \
        --audio_params_source "$AUDIO_PARAMS_SOURCE" \
        --out_dir "$PREP_OUT_DIR" \
        --prep_only; then
        echo "prep failed/infeasible: ${SPECIES}:${BIRD_ID} (skipping)" 1>&2
        rm -rf "$PREP_OUT_DIR"
        continue
    fi

    POOL_AUDIO_PARAMS="$PREP_OUT_DIR/pool/audio_params.json"
    if [ -z "$AVES_CLIP_SECONDS" ]; then
        if ! DERIVED_CLIP_SECONDS="$(derive_aves_clip_seconds "$SONGMAE_RUN_RESOLVED" "$POOL_AUDIO_PARAMS")"; then
            echo "Failed to derive AVES clip_seconds for ${SPECIES}:${BIRD_ID} (skipping)" 1>&2
            rm -rf "$PREP_OUT_DIR"
            continue
        fi
    else
        DERIVED_CLIP_SECONDS="$AVES_CLIP_SECONDS"
    fi

    SONGMAE_RUN_TAG="$RUN_TAG_PREFIX/songmae_${PRE_NAME}_${SPECIES}_${BIRD_ID}"
    AVES_RUN_TAG="$RUN_TAG_PREFIX/aves_${SPECIES}_${BIRD_ID}"

    echo "SongMAE run_tag: $SONGMAE_RUN_TAG"
    if bash "$PROJECT_ROOT/shell/classify_detect_bench.sh" \
        --spec_root "$SPEC_ROOT" \
        --annotation_root "$ANNOTATION_ROOT" \
        --species "$SPECIES" \
        --bird_id "$BIRD_ID" \
        --train_seconds "$TRAIN_SECONDS" \
        --mode "$MODE" \
        --seed "$SEED" \
        --audio_params_source "$AUDIO_PARAMS_SOURCE" \
        --out_dir "$PREP_OUT_DIR" \
        --probe_mode "linear" \
        --lr "$SONGMAE_LR" \
        --steps "$STEPS" \
        --run_tag "$SONGMAE_RUN_TAG" \
        --pretrained_run "$SONGMAE_RUN_RESOLVED" \
        --use_prepared; then
        SUCCESS_SONGMAE=$((SUCCESS_SONGMAE + 1))
        python "$PROJECT_ROOT/scripts/eval/eval_val_outputs_f1.py" \
            --runs_root "$PROJECT_ROOT/runs" \
            --run_names "$SONGMAE_RUN_TAG" \
            --out_csv "$SONGMAE_RESULTS_CSV" \
            --append \
            --no_summary
    else
        echo "SongMAE run failed: ${SPECIES}:${BIRD_ID}" 1>&2
    fi

    echo "AVES run_tag: $AVES_RUN_TAG (clip_seconds=${DERIVED_CLIP_SECONDS})"
    if bash "$PROJECT_ROOT/shell/train_AVES.sh" \
        --spec_root "$SPEC_ROOT" \
        --wav_root "$WAV_ROOT" \
        --annotation_root "$ANNOTATION_ROOT" \
        --species "$SPECIES" \
        --bird_id "$BIRD_ID" \
        --train_seconds "$TRAIN_SECONDS" \
        --mode "$MODE" \
        --seed "$SEED" \
        --out_dir "$PREP_OUT_DIR" \
        --probe_mode "linear" \
        --lr "$AVES_LR" \
        --steps "$STEPS" \
        --clip_seconds "$DERIVED_CLIP_SECONDS" \
        --run_tag "$AVES_RUN_TAG" \
        --use_prepared; then
        SUCCESS_AVES=$((SUCCESS_AVES + 1))
        python "$PROJECT_ROOT/scripts/eval/eval_val_outputs_f1.py" \
            --runs_root "$PROJECT_ROOT/runs" \
            --run_names "$AVES_RUN_TAG" \
            --out_csv "$AVES_RESULTS_CSV" \
            --append \
            --no_summary
    else
        echo "AVES run failed: ${SPECIES}:${BIRD_ID}" 1>&2
    fi

    if [ "$KEEP_TEMP" -eq 0 ]; then
        rm -rf "$PREP_OUT_DIR"
    fi
done < <(build_bird_stream)

if [ "$BIRD_COUNT" -eq 0 ]; then
    echo "No birds matched filters or annotation inputs." 1>&2
    exit 1
fi

echo
echo "Completed birds: ${BIRD_COUNT}"
echo "Successful SongMAE runs: ${SUCCESS_SONGMAE}"
echo "Successful AVES runs: ${SUCCESS_AVES}"
echo "SongMAE results CSV: ${SONGMAE_RESULTS_CSV}"
echo "AVES results CSV: ${AVES_RESULTS_CSV}"
