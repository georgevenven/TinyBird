"""
Convert Tweety combined onset/offset detection JSON into the TinyBird schema.

The source file is expected to be a JSON list where each item describes a
recording, for example:

{
  "filename": "USA5177_45224.61961377_10_25_17_12_41.wav",
  "song_present": false,
  "segments": [
    {
      "onset_timebin": 1736,
      "offset_timebin": 3251,
      "onset_ms": 4684.444444444444,
      "offset_ms": 8772.539682539684
    }
  ],
  "spec_parameters": {...},
  "source_group": "USA5177_Oct2023"
}

This source only contains song-activity detections, not unit/syllable labels.
By default, each source segment becomes one TinyBird `detected_event` with an
empty `units` list. Use `--emit_placeholder_units` to add one unit per event if
you need compatibility with unit-level tooling.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


_BIRD_ID_PATTERN = re.compile(r"([A-Za-z]+\d+)")


def infer_bird_id(filename: str, source_group: str) -> str:
    stem = Path(filename).stem
    if stem:
        token = stem.split("_", 1)[0].strip()
        if token:
            return token

    source_token = source_group.split("_", 1)[0].strip()
    if source_token:
        return source_token

    for candidate in (filename, source_group):
        match = _BIRD_ID_PATTERN.search(candidate)
        if match:
            return match.group(1)

    raise ValueError(f"Could not infer bird_id from filename={filename!r}, source_group={source_group!r}")


def _normalize_event(
    raw_segment: Dict[str, object],
    emit_placeholder_units: bool,
    placeholder_unit_id: int,
) -> Optional[Dict[str, object]]:
    onset_raw = raw_segment.get("onset_ms")
    offset_raw = raw_segment.get("offset_ms")
    if onset_raw is None or offset_raw is None:
        return None

    onset_ms = float(onset_raw)
    offset_ms = float(offset_raw)
    if offset_ms < onset_ms:
        onset_ms, offset_ms = offset_ms, onset_ms

    if emit_placeholder_units:
        units = [
            {
                "onset_ms": onset_ms,
                "offset_ms": offset_ms,
                "id": placeholder_unit_id,
            }
        ]
    else:
        units = []

    return {
        "onset_ms": onset_ms,
        "offset_ms": offset_ms,
        "units": units,
    }


def load_source_json(src_json: Path) -> List[Dict[str, object]]:
    with src_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(f"Expected top-level list in {src_json}, found {type(payload).__name__}")

    normalized: List[Dict[str, object]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Expected object at index {idx}, found {type(item).__name__}")
        normalized.append(item)
    return normalized


def convert_recordings(
    source_items: Sequence[Dict[str, object]],
    emit_placeholder_units: bool,
    placeholder_unit_id: int,
) -> Tuple[List[Dict[str, object]], int]:
    recordings: Dict[Tuple[str, str], Dict[str, object]] = {}
    duplicate_recording_rows = 0

    for idx, item in enumerate(source_items):
        filename = str(item.get("filename") or "").strip()
        source_group = str(item.get("source_group") or "").strip()
        if not filename:
            raise ValueError(f"Missing filename at source index {idx}")

        bird_id = infer_bird_id(filename, source_group)
        raw_segments = item.get("segments") or []
        if not isinstance(raw_segments, list):
            raise ValueError(f"Expected list of segments for {filename}, found {type(raw_segments).__name__}")

        song_present = item.get("song_present")
        if song_present is True and not raw_segments:
            raise ValueError(f"song_present=true but no segments for {filename}")
        if song_present is False and raw_segments:
            raise ValueError(f"song_present=false but segments were provided for {filename}")

        recording_key = (bird_id, filename)
        already_exists = recording_key in recordings
        recording_entry = recordings.setdefault(
            recording_key,
            {
                "recording": {
                    "filename": filename,
                    "bird_id": bird_id,
                    "detected_vocalizations": 0,
                },
                "detected_events": [],
            },
        )
        if already_exists:
            duplicate_recording_rows += 1

        events: List[Dict[str, object]] = recording_entry["detected_events"]  # type: ignore[assignment]
        seen = {(event["onset_ms"], event["offset_ms"]) for event in events}
        for raw_segment in raw_segments:
            if not isinstance(raw_segment, dict):
                raise ValueError(
                    f"Expected segment objects for {filename}, found {type(raw_segment).__name__}"
                )
            event = _normalize_event(raw_segment, emit_placeholder_units, placeholder_unit_id)
            if event is None:
                continue
            event_key = (event["onset_ms"], event["offset_ms"])
            if event_key in seen:
                continue
            seen.add(event_key)
            events.append(event)

        events.sort(key=lambda event: (float(event["onset_ms"]), float(event["offset_ms"])))
        recording_entry["recording"]["detected_vocalizations"] = len(events)  # type: ignore[index]

    return (
        [recordings[key] for key in sorted(recordings, key=lambda key: (key[0], key[1]))],
        duplicate_recording_rows,
    )


def write_output(recordings: Sequence[Dict[str, object]], dst_dir: Path) -> Path:
    payload = {"metadata": {"units": "ms"}, "recordings": list(recordings)}
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / "annotations.json"
    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return dst_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Tweety combined onset/offset results into TinyBird JSON."
    )
    parser.add_argument("--src_json", required=True, type=Path, help="Path to combined_onset_offset_results.json")
    parser.add_argument("--dst_dir", required=True, type=Path, help="Directory where annotations.json will be written")
    parser.add_argument(
        "--emit_placeholder_units",
        action="store_true",
        help="Emit one unit per detected event using --placeholder_unit_id",
    )
    parser.add_argument(
        "--placeholder_unit_id",
        type=int,
        default=0,
        help="Unit id used with --emit_placeholder_units (default: 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.src_json.exists():
        raise FileNotFoundError(f"Source JSON not found: {args.src_json}")

    source_items = load_source_json(args.src_json)
    recordings, duplicate_recording_rows = convert_recordings(
        source_items,
        emit_placeholder_units=args.emit_placeholder_units,
        placeholder_unit_id=args.placeholder_unit_id,
    )
    dst_path = write_output(recordings, args.dst_dir)
    print(
        f"Wrote {dst_path} "
        f"({len(source_items)} source rows -> {len(recordings)} unique recordings; "
        f"merged {duplicate_recording_rows} duplicate filename rows)"
    )


if __name__ == "__main__":
    main()
