"""
Convert per-bird Canary CSV annotations into the TinyBird JSON schema.

The source directory should contain one CSV file per bird. Each row represents
an annotated unit with absolute onset/offset timestamps (in seconds) within an
audio file. Rows can optionally include a ``sequence`` column to group units
belonging to the same song/vocalization.

Output JSON:
{
  "metadata": {"units": "ms"},
  "recordings": [
    {
      "recording": {
        "filename": "clip.wav",
        "bird_id": "bird123",
        "detected_vocalizations": 5
      },
      "detected_events": [
        {
          "onset_ms": 12.3,
          "offset_ms": 456.7,
          "units": [
            {"onset_ms": 12.3, "offset_ms": 45.6, "id": 0},
            ...
          ]
        }
      ]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class UnitRow:
    bird_id: str
    audio_file: str
    onset_ms: float
    offset_ms: float
    label: str
    sequence: Optional[int]
    order: int  # Row order within the (bird_id, audio_file) grouping.


def detect_delimiter(header_line: str) -> str:
    """Infer the delimiter from the header line."""
    header_line = header_line.strip("\ufeff\r\n")
    if "\t" in header_line and "," in header_line:
        # Prefer tabs when both are present because tab-separated headers
        # often contain commas inside field names.
        return "\t"
    if "\t" in header_line:
        return "\t"
    return ","


def parse_sequence(raw_value: Optional[str]) -> Optional[int]:
    """Return integer sequence index when possible."""
    if raw_value is None:
        return None
    raw_value = raw_value.strip()
    if raw_value == "":
        return None
    try:
        return int(float(raw_value))
    except ValueError:
        return None


def load_units_from_csv(csv_path: Path) -> List[UnitRow]:
    """Load annotation rows from a single CSV file."""
    bird_id = csv_path.stem
    rows: List[UnitRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        header_line = f.readline()
        if not header_line:
            return rows
        delimiter = detect_delimiter(header_line)
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delimiter)

        per_audio_order: Dict[str, int] = {}
        for record in reader:
            label_raw = (record.get("label") or "").strip()
            audio_file = (record.get("audio_file") or "").strip()
            onset_raw = record.get("onset_ms") or record.get("onset_s") or record.get("start_s")
            offset_raw = record.get("offset_ms") or record.get("offset_s") or record.get("end_s")

            if not (label_raw and audio_file and onset_raw and offset_raw):
                continue

            try:
                onset = float(onset_raw)
                offset = float(offset_raw)
            except ValueError:
                continue

            # The CSV stores seconds; support optional millisecond columns.
            if "onset_ms" in record or "offset_ms" in record:
                onset_ms = onset
                offset_ms = offset
            else:
                onset_ms = onset * 1000.0
                offset_ms = offset * 1000.0

            if onset_ms > offset_ms:
                onset_ms, offset_ms = offset_ms, onset_ms

            seq_index = parse_sequence(record.get("sequence"))
            order = per_audio_order.setdefault(audio_file, 0)
            per_audio_order[audio_file] += 1

            rows.append(
                UnitRow(
                    bird_id=bird_id,
                    audio_file=audio_file,
                    onset_ms=onset_ms,
                    offset_ms=offset_ms,
                    label=label_raw,
                    sequence=seq_index,
                    order=order,
                )
            )
    return rows


def collect_all_units(src_dir: Path) -> List[UnitRow]:
    csv_paths = sorted(p for p in src_dir.glob("*.csv") if p.is_file())
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {src_dir}")

    all_rows: List[UnitRow] = []
    for csv_path in csv_paths:
        all_rows.extend(load_units_from_csv(csv_path))

    if not all_rows:
        raise ValueError(f"No annotations parsed from CSV files in {src_dir}")
    return all_rows


def build_label_map(rows: Sequence[UnitRow]) -> Dict[str, int]:
    labels = {row.label for row in rows}
    if not labels:
        raise ValueError("No labels discovered while building label map")
    return {label: idx for idx, label in enumerate(sorted(labels))}


def group_rows(rows: Sequence[UnitRow]) -> List[Dict[str, object]]:
    label_map = build_label_map(rows)
    recordings: Dict[Tuple[str, str], Dict[str, object]] = {}

    for row in rows:
        recording_key = (row.bird_id, row.audio_file)
        recording_entry = recordings.setdefault(
            recording_key,
            {
                "recording": {
                    "filename": row.audio_file,
                    "bird_id": row.bird_id,
                    "detected_vocalizations": 0,
                },
                "events": {},
            },
        )

        event_key = row.sequence if row.sequence is not None else f"row-{row.order}"
        events: Dict[object, Dict[str, object]] = recording_entry["events"]  # type: ignore[assignment]
        event = events.setdefault(
            event_key,
            {
                "onset_ms": row.onset_ms,
                "offset_ms": row.offset_ms,
                "units": [],
                "sort_onset": row.onset_ms,
            },
        )

        event["onset_ms"] = min(event["onset_ms"], row.onset_ms)
        event["offset_ms"] = max(event["offset_ms"], row.offset_ms)
        event["sort_onset"] = min(event["sort_onset"], row.onset_ms)
        event_units: List[Dict[str, float]] = event["units"]  # type: ignore[assignment]
        event_units.append(
            {
                "onset_ms": row.onset_ms,
                "offset_ms": row.offset_ms,
                "id": label_map[row.label],
            }
        )

        recording_entry["recording"]["detected_vocalizations"] += 1  # type: ignore[index]

    json_recordings: List[Dict[str, object]] = []
    for recording_key in sorted(recordings, key=lambda x: (x[0], x[1])):
        recording_entry = recordings[recording_key]
        events_dict: Dict[object, Dict[str, object]] = recording_entry.pop("events")  # type: ignore[assignment]
        detected_events = sorted(events_dict.values(), key=lambda e: e["sort_onset"])
        for event in detected_events:
            event.pop("sort_onset", None)
            event["units"].sort(key=lambda u: u["onset_ms"])
        json_recordings.append(
            {
                "recording": recording_entry["recording"],
                "detected_events": detected_events,
            }
        )
    return json_recordings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Canary CSV annotations into TinyBird JSON.")
    parser.add_argument("--src_dir", required=True, type=Path, help="Directory containing per-bird CSV files.")
    parser.add_argument("--dst_dir", required=True, type=Path, help="Directory where annotations.json will be written.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_dir: Path = args.src_dir
    dst_dir: Path = args.dst_dir

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    rows = collect_all_units(src_dir)
    recordings = group_rows(rows)
    payload = {"metadata": {"units": "ms"}, "recordings": recordings}

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / "annotations.json"
    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {dst_path}")


if __name__ == "__main__":
    main()
