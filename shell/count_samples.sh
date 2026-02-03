#!/bin/bash

# count_samples.sh
# Counts total detected-event seconds for each species and individual

set -e

# Navigate to project root
cd "$(dirname "$0")/.."

# ================= CONFIGURATION =================
# Using same config structure as benchmark script
ANNOTATION_ROOT="files"

SPECIES_LIST=(
    "BengaleseFinch:bf_annotations.json"
    "Canary:canary_annotations.json"
    "ZebraFinch:zf_annotations.json"
)
# =================================================

echo "=============================================================="
echo "                DATASET DURATION COUNTS                       "
echo "=============================================================="
printf "%-20s %-15s %-12s %s\n" "Species" "Individual" "Seconds" "UniqueUnits(+silence)"
echo "--------------------------------------------------------------"

for ENTRY in "${SPECIES_LIST[@]}"; do
    IFS=":" read -r SPECIES ANNOT_FILE <<< "$ENTRY"
    ANNOT_PATH="$ANNOTATION_ROOT/$ANNOT_FILE"
    
    if [ ! -f "$ANNOT_PATH" ]; then
        echo "Error: $ANNOT_PATH not found"
        continue
    fi
    
    # Python one-liner to parse JSON and aggregate seconds
    python -c "
import json
from collections import Counter, defaultdict

try:
    with open('$ANNOT_PATH', 'r') as f:
        data = json.load(f)
    
    # Sum detected-event seconds and unique units per individual
    counts = Counter()
    units_by_bird = defaultdict(set)
    for r in data['recordings']:
        bird_id = r['recording']['bird_id']
        total_ms = 0
        for event in r['detected_events']:
            total_ms += event['offset_ms'] - event['onset_ms']
            for unit in event.get('units', []):
                if 'id' in unit:
                    units_by_bird[bird_id].add(int(unit['id']))
        counts[bird_id] += total_ms / 1000.0
        
    # Print total for species
    total = sum(counts.values())
    species_units = set()
    for ids in units_by_bird.values():
        species_units.update(ids)
    print(f'{total:<10.2f} {len(species_units) + 1:<10d} SPECIES_TOTAL')
    
    # Print per individual
    for bird_id, count in sorted(counts.items()):
        print(f'{count:<10.2f} {len(units_by_bird[bird_id]) + 1:<10d} {bird_id}')
        
except Exception as e:
    print(f'Error: {e}')
" | while read -r COUNT UNIT_COUNT INDIVIDUAL; do
        if [ "$INDIVIDUAL" == "SPECIES_TOTAL" ]; then
            # Species header line
             printf "%-20s %-15s %-12s %s\n" "$SPECIES" "ALL" "$COUNT" "$UNIT_COUNT"
        else
            # Individual line
             printf "%-20s %-15s %-12s %s\n" "" "$INDIVIDUAL" "$COUNT" "$UNIT_COUNT"
        fi
    done
    echo "--------------------------------------------------------------"
done
