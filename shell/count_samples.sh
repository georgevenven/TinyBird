#!/bin/bash

# count_samples.sh
# Counts total recordings for each species and individual

set -e

# Navigate to project root
cd "$(dirname "$0")/.."

# ================= CONFIGURATION =================
# Using same config structure as benchmark script
ANNOTATION_ROOT="/Users/georgev/Documents/data/SongMAE_Bench_Data"

SPECIES_LIST=(
    "BengaleseFinch:bf_annotations.json"
    "Canary:canary_annotations.json"
    "ZebraFinch:zf_annotations.json"
)
# =================================================

echo "================================================="
echo "           DATASET SAMPLE COUNTS                 "
echo "================================================="
printf "%-20s %-15s %s\n" "Species" "Individual" "Count"
echo "-------------------------------------------------"

for ENTRY in "${SPECIES_LIST[@]}"; do
    IFS=":" read -r SPECIES ANNOT_FILE <<< "$ENTRY"
    ANNOT_PATH="$ANNOTATION_ROOT/$ANNOT_FILE"
    
    if [ ! -f "$ANNOT_PATH" ]; then
        echo "Error: $ANNOT_PATH not found"
        continue
    fi
    
    # Python one-liner to parse JSON and aggregate counts
    python -c "
import json
from collections import Counter

try:
    with open('$ANNOT_PATH', 'r') as f:
        data = json.load(f)
    
    # Count per individual
    counts = Counter()
    for r in data['recordings']:
        counts[r['recording']['bird_id']] += 1
        
    # Print total for species
    total = sum(counts.values())
    print(f'{total:<10} SPECIES_TOTAL')
    
    # Print per individual
    for bird_id, count in sorted(counts.items()):
        print(f'{count:<10} {bird_id}')
        
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

