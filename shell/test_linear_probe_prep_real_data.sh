#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

SPEC_ROOT="/media/george-vengrovski/disk2/specs"
TRAIN_SECONDS="16"
MODE="classify"
SEED="42"
BIRD_LIST_JSON="files/SFT_experiment_birds.json"
TEMP_ROOT="temp"
MAX_BIRDS=""
PREVIEW_FILES_PER_SPLIT="2"
ONLY_SPECIES=""
ONLY_BIRD_ID=""
REQUIRED_TOTAL_FILES_PER_UNIT="2"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
SUMMARY_DIR="${TEMP_ROOT}/linear_probe_prep_test/${RUN_TAG}"
SUMMARY_CSV="${SUMMARY_DIR}/split_summary.csv"

usage() {
    cat <<EOF
Usage: $0 [options]

Runs only the prep/split stage used by shell/linear_probe_experiments.sh
using real data, then reports unique syllables in pool/test/train and writes
spectrogram preview images.

Options:
  --spec_root PATH                Default: ${SPEC_ROOT}
  --bird_list_json PATH           Default: ${BIRD_LIST_JSON}
  --train_seconds FLOAT           Default: ${TRAIN_SECONDS}
  --temp_root PATH                Default: ${TEMP_ROOT}
  --max_birds N                   Optional cap on number of birds processed
  --preview_files_per_split N     Pool-only cap (train/test always render all)
  --species NAME                  Process a single species (requires --bird_id)
  --bird_id ID                    Process a single bird id (requires --species)
  -h, --help                      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    key="$1"
    case "$key" in
        --spec_root)
            SPEC_ROOT="$2"
            shift 2
            ;;
        --bird_list_json)
            BIRD_LIST_JSON="$2"
            shift 2
            ;;
        --train_seconds)
            TRAIN_SECONDS="$2"
            shift 2
            ;;
        --temp_root)
            TEMP_ROOT="$2"
            shift 2
            ;;
        --max_birds)
            MAX_BIRDS="$2"
            shift 2
            ;;
        --preview_files_per_split)
            PREVIEW_FILES_PER_SPLIT="$2"
            shift 2
            ;;
        --species)
            ONLY_SPECIES="$2"
            shift 2
            ;;
        --bird_id)
            ONLY_BIRD_ID="$2"
            shift 2
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

if [[ -n "$ONLY_SPECIES" || -n "$ONLY_BIRD_ID" ]]; then
    if [[ -z "$ONLY_SPECIES" || -z "$ONLY_BIRD_ID" ]]; then
        echo "--species and --bird_id must be provided together" 1>&2
        exit 1
    fi
fi

mkdir -p "$SUMMARY_DIR"
cat > "$SUMMARY_CSV" <<EOF
species,bird_id,status,skip_reason,pool_unique_units,test_unique_units,train_unique_units,union_unique_units,missing_from_train,missing_from_test,missing_from_union,pool_units,test_units,train_units,union_units
EOF

process_one_bird() {
    local species="$1"
    local bird_id="$2"
    local spec_subdir=""
    local annotation_file=""
    local spec_dir=""
    local prep_out_dir="${TEMP_ROOT}/tinybird_pool/${species}/${bird_id}"
    local pool_dir="${prep_out_dir}/pool"
    local test_dir="${prep_out_dir}/test"
    local train_seconds_tag="${TRAIN_SECONDS//./p}"
    local train_dir="${prep_out_dir}/${MODE}/${bird_id}/train_${train_seconds_tag}s"
    local filtered_annotation="${pool_dir}/annotations_filtered.json"
    local preview_root="${prep_out_dir}/spectrogram_previews"
    local feasibility_json="${prep_out_dir}/feasibility.json"

    case "$species" in
        Bengalese_Finch)
            spec_subdir="bf_64hop_32khz"
            annotation_file="files/bf_annotations.json"
            ;;
        Canary)
            spec_subdir="canary_64hop_32khz"
            annotation_file="files/canary_annotations.json"
            ;;
        Zebra_Finch)
            spec_subdir="zf_64hop_32khz"
            annotation_file="files/zf_annotations.json"
            ;;
        *)
            echo "Unknown species in mapping: ${species}" 1>&2
            return 1
            ;;
    esac
    spec_dir="${SPEC_ROOT}/${spec_subdir}"

    echo
    echo "=== ${species}:${bird_id} ==="
    rm -rf "$prep_out_dir"
    echo "Preparing pool (copy only)..."
    python "$PROJECT_ROOT/src/bench_utils/copy_bird_pool.py" \
        --annotation_file "$annotation_file" \
        --spec_dir "$spec_dir" \
        --out_dir "$pool_dir" \
        --bird_id "$bird_id"

    echo "Feasibility check: unit support per distinct file in pool..."
    python - "$pool_dir" "$filtered_annotation" "$REQUIRED_TOTAL_FILES_PER_UNIT" "$feasibility_json" <<'PY'
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

pool_dir = Path(sys.argv[1])
annotation_file = Path(sys.argv[2])
need_total = int(sys.argv[3])
out_json = Path(sys.argv[4])

chunk_re = re.compile(r"^(?P<base>.+)__ms_(?P<start>\d+)_(?P<end>\d+)$")

def parse_chunk(stem):
    base = stem
    start = None
    end = None
    while True:
        m = chunk_re.match(base)
        if not m:
            break
        base = m.group("base")
        start = int(m.group("start"))
        end = int(m.group("end"))
    return base, start, end

ann = json.loads(annotation_file.read_text(encoding="utf-8"))
units_by_base = defaultdict(list)
all_units = set()
for rec in ann.get("recordings", []):
    base = Path(rec["recording"]["filename"]).stem
    for event in rec.get("detected_events", []):
        for unit in event.get("units", []):
            uid = int(unit["id"])
            on = float(unit["onset_ms"])
            off = float(unit["offset_ms"])
            units_by_base[base].append((uid, on, off))
            all_units.add(uid)

files_by_unit = defaultdict(set)
for path in sorted(pool_dir.glob("*.npy")):
    base, start, end = parse_chunk(path.stem)
    chunk_start = float("-inf") if start is None else float(start)
    chunk_end = float("inf") if end is None else float(end)
    for uid, onset, offset in units_by_base.get(base, []):
        if offset <= chunk_start or onset >= chunk_end:
            continue
        files_by_unit[uid].add(path.name)

counts = {int(uid): len(files_by_unit.get(uid, set())) for uid in sorted(all_units)}
infeasible = [int(uid) for uid, cnt in counts.items() if cnt < need_total]
result = {
    "feasible": len(infeasible) == 0,
    "required_total_files_per_unit": need_total,
    "unit_file_counts": counts,
    "infeasible_units": infeasible,
}
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

print(f"required distinct files per unit: {need_total}")
print(f"units in annotations: {len(all_units)}")
print(f"infeasible units: {infeasible}")
PY

    if ! python - "$feasibility_json" <<'PY'
import json
import sys
obj = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())
raise SystemExit(0 if obj.get("feasible", False) else 1)
PY
    then
        echo "Skipping ${species}:${bird_id} (not enough distinct files per unit for disjoint split)."
        python - "$species" "$bird_id" "$SUMMARY_CSV" "$feasibility_json" <<'PY'
import csv
import json
import sys
species, bird_id, summary_csv, feasibility_json = sys.argv[1:5]
info = json.loads(open(feasibility_json, "r", encoding="utf-8").read())
reason = (
    f"infeasible_units={info.get('infeasible_units', [])}; "
    f"required_total={info.get('required_total_files_per_unit')}"
)
with open(summary_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([species, bird_id, "skipped", reason, "", "", "", "", "", "", "", "", "", "", ""])
PY
        return 0
    fi

    echo "Sampling test/train with solver split..."
    if ! python "$PROJECT_ROOT/src/bench_utils/solver_split_by_seconds.py" \
        --pool_dir "$pool_dir" \
        --annotation_json "$filtered_annotation" \
        --test_dir "$test_dir" \
        --train_dir "$train_dir" \
        --train_seconds "$TRAIN_SECONDS" \
        --test_ratio "0.2" \
        --feasibility_json "$feasibility_json"
    then
        echo "Skipping ${species}:${bird_id} (solver could not find feasible split)."
        python - "$species" "$bird_id" "$SUMMARY_CSV" "$feasibility_json" <<'PY'
import csv
import json
import sys
species, bird_id, summary_csv, feasibility_json = sys.argv[1:5]
info = {}
if feasibility_json and feasibility_json != "None":
    try:
        info = json.loads(open(feasibility_json, "r", encoding="utf-8").read())
    except Exception:
        info = {}
reason = (
    f"solver_failed status={info.get('solver_status', 'unknown')} "
    f"msg={info.get('solver_message', '')}"
)
with open(summary_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([species, bird_id, "skipped", reason, "", "", "", "", "", "", "", "", "", "", ""])
PY
        return 0
    fi

    python - "$species" "$bird_id" "$pool_dir" "$test_dir" "$train_dir" "$filtered_annotation" "$preview_root" "$PREVIEW_FILES_PER_SPLIT" "$SUMMARY_CSV" <<'PY'
import csv
import json
import re
import shutil
import sys
from pathlib import Path

species = sys.argv[1]
bird_id = sys.argv[2]
pool_dir = Path(sys.argv[3])
test_dir = Path(sys.argv[4])
train_dir = Path(sys.argv[5])
annotation_file = Path(sys.argv[6])
preview_root = Path(sys.argv[7])
preview_n = int(sys.argv[8])
summary_csv = Path(sys.argv[9])

chunk_re = re.compile(r"^(?P<base>.+)__ms_(?P<start>\d+)_(?P<end>\d+)$")

if not annotation_file.exists():
    raise SystemExit(f"Missing filtered annotations: {annotation_file}")

annotations = json.loads(annotation_file.read_text(encoding="utf-8"))

units_by_base = {}
all_units = set()
for rec in annotations.get("recordings", []):
    base = Path(rec["recording"]["filename"]).stem
    unit_events = []
    for event in rec.get("detected_events", []):
        for unit in event.get("units", []):
            onset = float(unit["onset_ms"])
            offset = float(unit["offset_ms"])
            uid = int(unit["id"])
            unit_events.append((onset, offset, uid))
            all_units.add(uid)
    units_by_base[base] = unit_events

def parse_chunk(stem):
    base = stem
    start = None
    end = None
    while True:
        m = chunk_re.match(base)
        if not m:
            break
        base = m.group("base")
        start = int(m.group("start"))
        end = int(m.group("end"))
    return base, start, end

def unique_units(split_dir):
    out = set()
    if not split_dir.exists():
        return out
    for path in sorted(split_dir.glob("*.npy")):
        base, start, end = parse_chunk(path.stem)
        start = float("-inf") if start is None else float(start)
        end = float("inf") if end is None else float(end)
        for onset, offset, uid in units_by_base.get(base, []):
            if offset <= start or onset >= end:
                continue
            out.add(uid)
    return out

pool_units = unique_units(pool_dir)
test_units = unique_units(test_dir)
train_units = unique_units(train_dir)
union_units = pool_units | test_units | train_units

missing_from_train = sorted(all_units - train_units)
missing_from_test = sorted(all_units - test_units)
missing_from_union = sorted(all_units - union_units)

print(f"all_units ({len(all_units)}): {sorted(all_units)}")
print(f"pool_units ({len(pool_units)}): {sorted(pool_units)}")
print(f"test_units ({len(test_units)}): {sorted(test_units)}")
print(f"train_units ({len(train_units)}): {sorted(train_units)}")
print(f"union_units ({len(union_units)}): {sorted(union_units)}")
print(f"missing_from_train ({len(missing_from_train)}): {missing_from_train}")
print(f"missing_from_test ({len(missing_from_test)}): {missing_from_test}")
print(f"missing_from_union ({len(missing_from_union)}): {missing_from_union}")

def list_str(values):
    return ";".join(str(v) for v in sorted(values))

with summary_csv.open("a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            species,
            bird_id,
            "processed",
            "",
            len(pool_units),
            len(test_units),
            len(train_units),
            len(union_units),
            len(missing_from_train),
            len(missing_from_test),
            len(missing_from_union),
            list_str(pool_units),
            list_str(test_units),
            list_str(train_units),
            list_str(union_units),
        ]
    )

try:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path.cwd() / "src"))
    from plotting_utils import save_supervised_prediction_plot
except Exception as exc:
    print(f"Skipping previews: matplotlib/numpy unavailable ({exc})")
    raise SystemExit(0)

sr = None
hop_size = None
audio_params_path = pool_dir / "audio_params.json"
if audio_params_path.exists():
    audio_params = json.loads(audio_params_path.read_text(encoding="utf-8"))
    sr = int(audio_params.get("sr", 0))
    hop_size = int(audio_params.get("hop_size", 0))

def ms_to_bin(ms):
    if not sr or not hop_size:
        return None
    return int((float(ms) / 1000.0) * sr / hop_size)

num_classes = (max(all_units) + 2) if all_units else 2

def labels_for_file(path):
    arr = np.load(path, mmap_mode="r")
    timebins = int(arr.shape[1])
    labels = np.zeros((timebins,), dtype=np.int64)
    base, start, end = parse_chunk(path.stem)
    offset_ms = 0.0 if start is None else float(start)
    chunk_start = float("-inf") if start is None else float(start)
    chunk_end = float("inf") if end is None else float(end)

    for onset, offset, uid in units_by_base.get(base, []):
        if offset <= chunk_start or onset >= chunk_end:
            continue
        onset_clip = max(onset, chunk_start)
        offset_clip = min(offset, chunk_end)
        if offset_clip <= onset_clip:
            continue
        if ms_to_bin(0.0) is None:
            continue
        start_bin = ms_to_bin(onset_clip - offset_ms)
        end_bin = ms_to_bin(offset_clip - offset_ms)
        if start_bin is None or end_bin is None:
            continue
        start_bin = max(0, min(start_bin, timebins))
        end_bin = max(0, min(end_bin, timebins))
        if end_bin > start_bin:
            labels[start_bin:end_bin] = int(uid) + 1
    return labels

def save_split_previews(split_name, split_dir, *, force_all=False):
    if not split_dir.exists():
        return 0
    out_dir = preview_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(split_dir.glob("*.npy"))
    if force_all:
        selected = npy_files
    else:
        if preview_n <= 0:
            return 0
        selected = npy_files[:preview_n]

    if force_all and not selected:
        return 0

    count = 0
    for idx, path in enumerate(selected):
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2:
            raise SystemExit(f"Expected 2D spectrogram array in {path}, got shape {arr.shape}")
        labels = labels_for_file(path)
        max_tb = int(arr.shape[1])
        if max_tb <= 0:
            raise SystemExit(f"Empty spectrogram file cannot be previewed: {path}")
        spec = np.array(arr[:, :max_tb], dtype=np.float32)
        labels_ds = labels[:max_tb]
        preds_ds = labels_ds.copy()
        tmp_path = save_supervised_prediction_plot(
            spectrogram=spec,
            labels=labels_ds,
            predictions=preds_ds,
            probabilities=None,
            logits=None,
            filename=path.name,
            mode="classify",
            num_classes=num_classes,
            output_dir=str(out_dir),
            step_num=idx,
            split=split_name,
        )
        final_path = out_dir / f"{path.stem}.png"
        shutil.move(tmp_path, final_path)
        count += 1

    if force_all and count != len(npy_files):
        raise SystemExit(
            f"Preview/image count mismatch for {split_name}: generated={count}, files={len(npy_files)}"
        )
    return count

pool_prev = save_split_previews("pool", pool_dir, force_all=False)
test_prev = save_split_previews("test", test_dir, force_all=True)
train_prev = save_split_previews("train", train_dir, force_all=True)
print(
    f"saved previews -> pool:{pool_prev}, test:{test_prev}, train:{train_prev} "
    f"at {preview_root}"
)
PY
}

if [[ -n "$ONLY_SPECIES" ]]; then
    process_one_bird "$ONLY_SPECIES" "$ONLY_BIRD_ID"
else
    processed=0
    while IFS=: read -r species bird_id; do
        process_one_bird "$species" "$bird_id"
        processed=$((processed + 1))
        if [[ -n "$MAX_BIRDS" && "$processed" -ge "$MAX_BIRDS" ]]; then
            break
        fi
    done < <(python - <<PY
import json
from pathlib import Path

path = Path("$BIRD_LIST_JSON")
data = json.loads(path.read_text(encoding="utf-8"))
for entry in data:
    species = entry.get("species", "")
    bird_id = entry.get("bird_id", "")
    if not species or not bird_id:
        continue
    print(f"{species}:{bird_id}")
PY
)
fi

echo
echo "Done."
echo "Summary CSV: ${SUMMARY_CSV}"
