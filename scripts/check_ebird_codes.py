#!/usr/bin/env python3
"""
Check eBird codes in annotation JSON files and compare against expected lab species codes.

Conservative behavior:
- Only compares string codes unless a label map is provided.
- Does not infer codes from filenames or other metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


# Expected lab species codes from the user's reference list:
# - Zebra Finch (Taeniopygia guttata): zebfin2 (species), zebfin3 (domestic type)
# - Island Canary (Serinus canaria): comcan (species), islcan1 (domestic type)
# - White-rumped Munia (Lonchura striata): whrmun (species), whrmun8 (domestic type)
EXPECTED_CODES = {
    "domestic": {"zebfin3", "islcan1", "whrmun8"},
    "rolled_up": {"zebfin2", "comcan", "whrmun"},
}
EXPECTED_CODES["both"] = EXPECTED_CODES["domestic"] | EXPECTED_CODES["rolled_up"]


def load_annotations(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "recordings" in data:
        recs = data["recordings"]
        if isinstance(recs, list):
            return recs
        raise ValueError(f"'recordings' is not a list in {path}")

    if isinstance(data, list):
        return data

    raise ValueError(f"Unsupported JSON format in {path}")


def normalize_label_map(raw: object) -> Dict[int, str]:
    if isinstance(raw, dict):
        if "id2label" in raw:
            return normalize_label_map(raw["id2label"])
        if "label2id" in raw:
            return normalize_label_map(raw["label2id"])

    if isinstance(raw, list):
        return {idx: val for idx, val in enumerate(raw) if isinstance(val, str)}

    if isinstance(raw, dict):
        keys = list(raw.keys())
        vals = list(raw.values())

        if all(isinstance(v, str) for v in vals):
            mapped: Dict[int, str] = {}
            for k, v in raw.items():
                if isinstance(k, int):
                    mapped[k] = v
                elif isinstance(k, str) and k.isdigit():
                    mapped[int(k)] = v
            if mapped:
                return mapped

        if all(isinstance(k, str) for k in keys) and all(isinstance(v, int) for v in vals):
            # Invert code -> id mapping.
            return {v: k for k, v in raw.items()}

    raise ValueError("Unsupported label map format; expected JSON list or dict.")


def load_label_map(path: Path) -> Dict[int, str]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return normalize_label_map(raw)


def iter_recording_dicts(records: Iterable[dict]) -> Iterable[dict]:
    for rec in records:
        if isinstance(rec, dict) and "recording" in rec and isinstance(rec["recording"], dict):
            yield rec["recording"]
        elif isinstance(rec, dict):
            yield rec


def classify_code(
    code: object, str_counts: Counter, int_counts: Counter, other_counts: Counter
) -> None:
    if code is None:
        return
    if isinstance(code, str):
        str_counts[code] += 1
    elif isinstance(code, int):
        int_counts[code] += 1
    else:
        other_counts[type(code).__name__] += 1


def collect_codes(records: Iterable[dict]) -> Tuple[Counter, Counter, Counter, int]:
    str_counts: Counter = Counter()
    int_counts: Counter = Counter()
    other_counts: Counter = Counter()
    missing_fields = 0

    for rec in iter_recording_dicts(records):
        if not isinstance(rec, dict):
            continue

        has_field = False
        if "ebird_code" in rec:
            classify_code(rec.get("ebird_code"), str_counts, int_counts, other_counts)
            has_field = True
        if "ebird_code_multilabel" in rec:
            codes = rec.get("ebird_code_multilabel")
            if isinstance(codes, list):
                for c in codes:
                    classify_code(c, str_counts, int_counts, other_counts)
            else:
                classify_code(codes, str_counts, int_counts, other_counts)
            has_field = True

        if not has_field:
            missing_fields += 1

    return str_counts, int_counts, other_counts, missing_fields


def summarize_codes(
    file_path: Path,
    records: List[dict],
    expected_set: Set[str],
    label_map: Optional[Dict[int, str]],
) -> int:
    str_counts, int_counts, other_counts, missing_fields = collect_codes(records)
    unique_str = set(str_counts.keys())
    unique_int = set(int_counts.keys())

    mapped_codes: Set[str] = set()
    unknown_ints: Set[int] = set()
    if label_map and unique_int:
        for code in unique_int:
            mapped = label_map.get(code)
            if mapped is None:
                unknown_ints.add(code)
            else:
                mapped_codes.add(mapped)

    found_codes = unique_str | mapped_codes
    found_expected = sorted(expected_set & found_codes)
    missing_expected = sorted(expected_set - found_codes)

    print(f"File: {file_path}")
    print(f"  recordings: {len(records)}")
    if str_counts or int_counts or other_counts:
        print(
            "  ebird_code summary: "
            f"strings={len(unique_str)} unique, "
            f"ints={len(unique_int)} unique, "
            f"other_types={dict(other_counts) if other_counts else '{}'}"
        )
    else:
        print("  ebird_code summary: no ebird_code fields found")

    if missing_fields == len(records):
        print("  note: no ebird_code fields present; nothing to compare")
        print("")
        return 0

    if unique_int and not label_map:
        print("  note: numeric ebird_code values found; provide --label-map to compare")
    elif unique_int and label_map:
        print(f"  mapped numeric codes: {len(mapped_codes)} unique")
        if unknown_ints:
            print(f"  unmapped numeric codes: {len(unknown_ints)} unique")

    if expected_set:
        print(f"  expected codes found: {found_expected if found_expected else '[]'}")
        print(f"  expected codes missing: {missing_expected if missing_expected else '[]'}")
    print("")
    return 0


def default_annotation_paths() -> List[Path]:
    candidate = Path("files/XCM_train_annotations.json")
    if candidate.exists():
        return [candidate]
    # Fallback: pick any annotations in files/ if present.
    return sorted(Path("files").glob("*_annotations.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check eBird codes in annotation JSON files."
    )
    parser.add_argument(
        "--annotations",
        nargs="+",
        type=Path,
        help="One or more annotation JSON files.",
    )
    parser.add_argument(
        "--expected",
        choices=sorted(EXPECTED_CODES.keys()),
        default="both",
        help="Which expected code set to check (default: both).",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="Optional JSON label map for numeric ebird codes (id->code or code->id).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    paths = args.annotations or default_annotation_paths()
    if not paths:
        print("No annotation files found. Provide --annotations explicitly.", file=sys.stderr)
        return 2

    label_map = None
    if args.label_map:
        try:
            label_map = load_label_map(args.label_map)
        except Exception as exc:
            print(f"Failed to load label map: {exc}", file=sys.stderr)
            return 2

    expected_set = EXPECTED_CODES.get(args.expected, set())

    for path in paths:
        try:
            records = load_annotations(path)
        except Exception as exc:
            print(f"Failed to load {path}: {exc}", file=sys.stderr)
            continue
        summarize_codes(path, records, expected_set, label_map)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
