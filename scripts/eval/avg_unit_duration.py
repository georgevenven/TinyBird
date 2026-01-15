#!/usr/bin/env python3
"""
Compute average unit duration (ms) from annotation JSONs.

Expected structure:
  annotations["recordings"][i]["detected_events"][j]["units"][k]["onset_ms"|"offset_ms"]
"""

import argparse
import json
import os
from typing import Dict, Tuple


def compute_unit_stats(path: str) -> Tuple[int, float, int, float]:
    with open(path, "r") as f:
        data = json.load(f)

    total_units = 0
    total_ms = 0.0
    gap_count = 0
    gap_total_ms = 0.0

    for rec in data.get("recordings", []):
        for event in rec.get("detected_events", []):
            units = event.get("units", [])
            for unit in units:
                onset = unit.get("onset_ms")
                offset = unit.get("offset_ms")
                if onset is None or offset is None:
                    continue
                total_ms += float(offset) - float(onset)
                total_units += 1
            # Inter-unit gaps within the same event
            units_sorted = [
                u for u in units if u.get("onset_ms") is not None and u.get("offset_ms") is not None
            ]
            units_sorted.sort(key=lambda u: float(u["onset_ms"]))
            for prev, nxt in zip(units_sorted, units_sorted[1:]):
                gap = float(nxt["onset_ms"]) - float(prev["offset_ms"])
                if gap >= 0:
                    gap_total_ms += gap
                    gap_count += 1

    avg = (total_ms / total_units) if total_units > 0 else 0.0
    avg_gap = (gap_total_ms / gap_count) if gap_count > 0 else 0.0
    return total_units, avg, gap_count, avg_gap


def main() -> None:
    default_root = os.path.join(os.path.dirname(__file__), "..", "..", "files")
    parser = argparse.ArgumentParser(description="Average unit duration from annotation JSONs.")
    parser.add_argument(
        "--jsons",
        nargs="+",
        default=[
            os.path.join(default_root, "bf_annotations.json"),
            os.path.join(default_root, "canary_annotations.json"),
            os.path.join(default_root, "zf_annotations.json"),
        ],
        help="List of annotation JSON files (default: bf/canary/zf).",
    )
    args = parser.parse_args()

    grand_units = 0
    grand_ms = 0.0
    grand_gap_count = 0
    grand_gap_ms = 0.0

    for path in args.jsons:
        if not os.path.isfile(path):
            print(f"Missing file: {path}")
            continue

        count, avg, gap_count, avg_gap = compute_unit_stats(path)
        print(
            f"{os.path.basename(path)}: units={count}, avg_ms={avg:.3f}, "
            f"inter_unit_gaps={gap_count}, avg_gap_ms={avg_gap:.3f}"
        )

        grand_units += count
        grand_ms += avg * count
        grand_gap_count += gap_count
        grand_gap_ms += avg_gap * gap_count

    overall_avg = (grand_ms / grand_units) if grand_units > 0 else 0.0
    overall_gap_avg = (grand_gap_ms / grand_gap_count) if grand_gap_count > 0 else 0.0
    print(
        f"Overall: units={grand_units}, avg_ms={overall_avg:.3f}, "
        f"inter_unit_gaps={grand_gap_count}, avg_gap_ms={overall_gap_avg:.3f}"
    )


if __name__ == "__main__":
    main()

