"""Convert .wav.not.mat annotations into aggregated JSON.

This script scans a directory of Raven-style `.wav.not.mat` files (MATLAB v5)
that contain syllable annotations (onsets, offsets, labels) and converts them
into the TinyBird JSON format used by the Bengalese finch tooling:

{
  "metadata": {"units": "ms"},
  "recordings": [
    {
      "recording": {
        "filename": "clip.wav",
        "bird_id": "B123",
        "detected_vocalizations": 42
      },
      "detected_events": [
        {
          "onset_ms": 123.4,
          "offset_ms": 567.8,
          "units": [
            {"onset_ms": 123.4, "offset_ms": 150.2, "id": 0},
            ...
          ]
        }
      ]
    }
  ]
}

Usage:
    python wavnotmat_annotation_to_json.py --src_dir /path/to/notmats --dst_dir ./out
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.io import loadmat


@dataclass
class Segment:
    label: str
    onset_ms: float
    offset_ms: float


@dataclass
class RecordingAnnotation:
    path: Path
    filename: str
    bird_id: str
    segments: List[Segment]


_LABEL_PATTERN = re.compile(r"([A-Za-z]+\d+)")
_BIRD_PATTERN = re.compile(r"(?:^|[^A-Za-z0-9])([Bb]\d{1,4})")


def find_not_mat_files(src_dir: Path) -> List[Path]:
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    files = sorted(p for p in src_dir.rglob("*.not.mat") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No .not.mat files found under {src_dir}")
    return files


def _ensure_string(value) -> str:
    """Convert MATLAB-loaded values into a plain Python string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.dtype.kind in {"U", "S"}:
            flat = value.astype(str).ravel(order="F")
            return "".join(flat.tolist())
        flat = value.ravel(order="F")
        return "".join(str(x) for x in flat.tolist())
    return str(value)


def _basename(value: str) -> str:
    """Return filename component handling Windows-style paths."""
    if not value:
        return ""
    return value.replace("\\", "/").split("/")[-1]


def _extract_labels(raw_labels) -> List[str]:
    text = _ensure_string(raw_labels)
    if not text:
        return []
    return [c for c in text if c not in {"", " ", "\n", "\r", "\t"}]


def _to_float_array(values, field: str, path: Path) -> np.ndarray:
    if values is None:
        raise ValueError(f"Missing '{field}' in {path}")
    arr = np.array(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Field '{field}' empty in {path}")
    return arr


def infer_bird_id(mat_path: Path, fname: str) -> str:
    candidates: List[str] = []
    for part in mat_path.parts[::-1]:
        match = _LABEL_PATTERN.search(part)
        if match:
            candidates.append(match.group(1))
        match = _BIRD_PATTERN.search(part)
        if match:
            candidates.append(match.group(1).upper())
    if fname:
        fname_norm = fname.replace("\\", "/")
        for token in fname_norm.split("/"):
            match = _LABEL_PATTERN.search(token)
            if match:
                candidates.append(match.group(1))
            match = _BIRD_PATTERN.search(token)
            if match:
                candidates.append(match.group(1).upper())
    if candidates:
        # Prefer explicit B### ids before generic label+number matches.
        for cand in candidates:
            if cand.upper().startswith("B") and cand[1:].isdigit():
                return cand.upper()
        return candidates[0]
    # Fallback to stem (without extensions like .wav/.not.mat)
    stem = mat_path.name
    if stem.lower().endswith(".not.mat"):
        stem = stem[:-8]
    return stem


def parse_not_mat(path: Path) -> RecordingAnnotation:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    fname = _ensure_string(data.get("fname", ""))
    labels = _extract_labels(data.get("labels"))
    onsets = _to_float_array(data.get("onsets"), "onsets", path)
    offsets = _to_float_array(data.get("offsets"), "offsets", path)

    n = min(len(labels), onsets.size, offsets.size)
    if n == 0:
        raise ValueError(f"No syllables found in {path}")

    segments: List[Segment] = []
    for label, onset, offset in zip(labels[:n], onsets[:n], offsets[:n]):
        if not label:
            continue
        if math.isnan(onset) or math.isnan(offset):
            continue
        if offset < onset:
            onset, offset = offset, onset
        segments.append(Segment(label=label, onset_ms=float(onset), offset_ms=float(offset)))

    if not segments:
        raise ValueError(f"All segments filtered out in {path}")

    recording_name = _basename(fname) if fname else path.with_suffix("").with_suffix("").name
    bird_id = infer_bird_id(path, fname)

    return RecordingAnnotation(
        path=path,
        filename=recording_name,
        bird_id=bird_id,
        segments=segments,
    )


def build_label_map(recordings: Sequence[RecordingAnnotation]) -> Dict[str, int]:
    labels = {seg.label for rec in recordings for seg in rec.segments}
    if not labels:
        raise ValueError("No labels collected from recordings")
    return {label: idx for idx, label in enumerate(sorted(labels))}


def to_json_dict(recordings: Sequence[RecordingAnnotation]) -> Dict[str, object]:
    label_map = build_label_map(recordings)
    json_recordings = []

    for rec in recordings:
        units = [
            {
                "onset_ms": seg.onset_ms,
                "offset_ms": seg.offset_ms,
                "id": label_map[seg.label],
            }
            for seg in rec.segments
        ]
        event_onset = min(seg.onset_ms for seg in rec.segments)
        event_offset = max(seg.offset_ms for seg in rec.segments)
        json_recordings.append(
            {
                "recording": {
                    "filename": rec.filename,
                    "bird_id": rec.bird_id,
                    "detected_vocalizations": len(units),
                },
                "detected_events": [
                    {
                        "onset_ms": event_onset,
                        "offset_ms": event_offset,
                        "units": units,
                    }
                ],
            }
        )

    return {"metadata": {"units": "ms"}, "recordings": json_recordings}


def write_json(output: Dict[str, object], dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / "annotations.json"
    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return dst_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate .wav.not.mat annotations into JSON")
    parser.add_argument("--src_dir", required=True, type=Path, help="Directory containing .wav.not.mat files")
    parser.add_argument("--dst_dir", required=True, type=Path, help="Directory where annotations.json will be written")
    args = parser.parse_args()

    not_mat_paths = find_not_mat_files(args.src_dir)
    recordings: List[RecordingAnnotation] = []
    for path in not_mat_paths:
        try:
            recordings.append(parse_not_mat(path))
        except Exception as exc:
            raise RuntimeError(f"Failed to parse {path}: {exc}") from exc

    json_payload = to_json_dict(recordings)
    dst_path = write_json(json_payload, args.dst_dir)
    print(f"Wrote {dst_path}")


if __name__ == "__main__":
    main()
