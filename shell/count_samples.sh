#!/bin/bash

# count_samples.sh
# Counts filtered dataset stats for each species and individual.
# Birds are filtered using files/SFT_experiment_birds.json.

set -e

# Navigate to project root
cd "$(dirname "$0")/.."

# ================= CONFIGURATION =================
# Using same config structure as benchmark script
ANNOTATION_ROOT="files"
BIRD_FILTER_JSON="files/SFT_experiment_birds.json"

SPECIES_LIST=(
    "BengaleseFinch:bf_annotations.json"
    "Canary:canary_annotations.json"
    "ZebraFinch:zf_annotations.json"
)
# =================================================

echo "=============================================================="
echo "              FILTERED DATASET DURATION COUNTS                "
echo "=============================================================="
printf "%-16s %-15s %-10s %-12s %-13s %-13s %-11s %s\n" \
    "Species" "Individual" "Seconds" "UniqueUnits" "AvgTrainSec" "StdTrainSec" "AvgUnits" "StdUnits"
echo "---------------------------------------------------------------------------------------------------------------------"

if [ ! -f "$BIRD_FILTER_JSON" ]; then
    echo "Error: $BIRD_FILTER_JSON not found"
    exit 1
fi

for ENTRY in "${SPECIES_LIST[@]}"; do
    IFS=":" read -r SPECIES ANNOT_FILE <<< "$ENTRY"
    ANNOT_PATH="$ANNOTATION_ROOT/$ANNOT_FILE"
    
    if [ ! -f "$ANNOT_PATH" ]; then
        echo "Error: $ANNOT_PATH not found"
        continue
    fi
    
    # Parse annotation JSON, filter birds, and aggregate totals/means/std.
    python -c "
import json
from collections import Counter, defaultdict

def mean_std(vals):
    if not vals:
        return 0.0, 0.0
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
    return mean, var ** 0.5

try:
    with open('$ANNOT_PATH', 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open('$BIRD_FILTER_JSON', 'r', encoding='utf-8') as f:
        bird_filter_rows = json.load(f)

    species_alias = {
        'BengaleseFinch': 'Bengalese_Finch',
        'ZebraFinch': 'Zebra_Finch',
        'Canary': 'Canary',
    }
    filter_species = species_alias.get('$SPECIES', '$SPECIES')
    allowed_birds = set()
    train_seconds_by_bird = {}
    for row in bird_filter_rows:
        if row.get('species') != filter_species:
            continue
        bird_id = row.get('bird_id')
        if not bird_id:
            continue
        allowed_birds.add(bird_id)
        secs = row.get('seconds')
        try:
            train_seconds_by_bird[bird_id] = float(secs)
        except (TypeError, ValueError):
            pass

    # Sum detected-event seconds and unique units per individual,
    # plus per-recording stats to compute mean/std.
    counts = Counter()
    units_by_bird = defaultdict(set)

    for r in data['recordings']:
        bird_id = r.get('recording', {}).get('bird_id')
        if not bird_id or bird_id not in allowed_birds:
            continue
        total_ms = 0
        recording_units = set()
        for event in r['detected_events']:
            on = event.get('onset_ms')
            off = event.get('offset_ms')
            if on is not None and off is not None and off >= on:
                total_ms += off - on
            for unit in event.get('units', []):
                if 'id' in unit:
                    uid = int(unit['id'])
                    units_by_bird[bird_id].add(uid)
                    recording_units.add(uid)

        counts[bird_id] += total_ms / 1000.0

    # Print total for species.
    total = sum(counts.values())
    # Unit IDs are local to each bird, so species-level uniqueness must key on
    # (bird_id, unit_id) rather than raw unit_id integers.
    species_units = {
        (bird_id, unit_id)
        for bird_id, ids in units_by_bird.items()
        for unit_id in ids
    }
    species_train_seconds = [
        train_seconds_by_bird[bird_id]
        for bird_id in sorted(counts.keys())
        if bird_id in train_seconds_by_bird
    ]
    bird_unit_counts = [
        len(units_by_bird[bird_id]) + 1
        for bird_id in sorted(counts.keys())
    ]
    avg_sec, std_sec = mean_std(species_train_seconds)
    avg_units, std_units = mean_std(bird_unit_counts)
    print(
        f'{total:.2f}\\t{len(species_units) + 1}\\t'
        f'{avg_sec:.2f}\\t{std_sec:.2f}\\t{avg_units:.2f}\\t{std_units:.2f}\\tSPECIES_TOTAL'
    )

    # Print per individual.
    for bird_id, count in sorted(counts.items()):
        b_avg_sec = train_seconds_by_bird.get(bird_id, count)
        b_std_sec = 0.0
        b_avg_units = float(len(units_by_bird[bird_id]) + 1)
        b_std_units = 0.0
        print(
            f'{count:.2f}\\t{len(units_by_bird[bird_id]) + 1}\\t'
            f'{b_avg_sec:.2f}\\t{b_std_sec:.2f}\\t{b_avg_units:.2f}\\t{b_std_units:.2f}\\t{bird_id}'
        )

except Exception as e:
    print(f'Error: {e}')
" | while IFS=$'\t' read -r COUNT UNIT_COUNT AVG_SEC STD_SEC AVG_UNITS STD_UNITS INDIVIDUAL; do
        if [ "$INDIVIDUAL" == "SPECIES_TOTAL" ]; then
            # Species header line
            printf "%-16s %-15s %-10s %-12s %-13s %-13s %-11s %s\n" \
                "$SPECIES" "ALL" "$COUNT" "$UNIT_COUNT" "$AVG_SEC" "$STD_SEC" "$AVG_UNITS" "$STD_UNITS"
        else
            # Individual line
            printf "%-16s %-15s %-10s %-12s %-13s %-13s %-11s %s\n" \
                "" "$INDIVIDUAL" "$COUNT" "$UNIT_COUNT" "$AVG_SEC" "$STD_SEC" "$AVG_UNITS" "$STD_UNITS"
        fi
    done
    echo "---------------------------------------------------------------------------------------------------------------------"
done
