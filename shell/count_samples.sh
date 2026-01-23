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

echo "================================================="
echo "           DATASET DURATION COUNTS               "
echo "================================================="
printf "%-20s %-15s %s\n" "Species" "Individual" "Seconds"
echo "-------------------------------------------------"

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
from collections import Counter

try:
    with open('$ANNOT_PATH', 'r') as f:
        data = json.load(f)
    
    # Sum detected-event seconds per individual
    counts = Counter()
    for r in data['recordings']:
        total_ms = 0
        for event in r['detected_events']:
            total_ms += event['offset_ms'] - event['onset_ms']
        counts[r['recording']['bird_id']] += total_ms / 1000.0
        
    # Print total for species
    total = sum(counts.values())
    print(f'{total:<10.2f} SPECIES_TOTAL')
    
    # Print per individual
    for bird_id, count in sorted(counts.items()):
        print(f'{count:<10.2f} {bird_id}')
        
except Exception as e:
    print(f'Error: {e}')
" | while read -r COUNT INDIVIDUAL; do
        if [ "$INDIVIDUAL" == "SPECIES_TOTAL" ]; then
            # Species header line
             printf "%-20s %-15s %s\n" "$SPECIES" "ALL" "$COUNT"
        else
            # Individual line
             printf "%-20s %-15s %s\n" "" "$INDIVIDUAL" "$COUNT"
        fi
    done
    echo "-------------------------------------------------"
done
