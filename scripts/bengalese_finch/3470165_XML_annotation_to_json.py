"""
Convert Bengalese finch XML annotations into the TinyBird JSON format.

Expected input layout:
  <src_dir>/
    <bird_id>/
      Annotation.xml
      Wave/

Each XML `<Sequence>` contains absolute onset/offset (in samples) for a song
segment and child `<Note>` elements with per-note offsets, lengths, and labels.
The resulting JSON mirrors the structure produced by wavnotmat_annotation_to_json.py.
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

SAMPLING_RATE = 32_000  # XML timing is expressed in 32 kHz samples.


def build_label_mapping(src_dir: Path) -> Dict[str, int]:
    """Scan all XML files to map source labels to integer IDs."""
    labels = set()
    for folder_path in sorted(p for p in src_dir.iterdir() if p.is_dir()):
        xml_path = folder_path / "Annotation.xml"
        if not xml_path.exists():
            continue
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for seq in root.findall("Sequence"):
            for note in seq.findall("Note"):
                label_elem = note.find("Label")
                label = (label_elem.text or "").strip() if label_elem is not None else ""
                if label:
                    labels.add(label)

    if not labels:
        raise ValueError(f"No labels found under {src_dir}")

    max_numeric = -1
    mapping: Dict[str, int] = {}
    non_numeric: List[str] = []
    for label in labels:
        try:
            value = int(label)
            mapping[label] = value
            if value > max_numeric:
                max_numeric = value
        except ValueError:
            non_numeric.append(label)

    next_id = max_numeric + 1
    for label in sorted(non_numeric):
        mapping[label] = next_id
        next_id += 1
    return mapping


def collect_recordings(src_dir: Path, label_map: Dict[str, int]) -> List[Dict[str, object]]:
    """Convert XML data into TinyBird-flavoured recording dictionaries."""
    recordings: Dict[Tuple[str, str], Dict[str, object]] = {}

    for folder_path in sorted(p for p in src_dir.iterdir() if p.is_dir()):
        xml_path = folder_path / "Annotation.xml"
        if not xml_path.exists():
            continue

        raw_bird_id = folder_path.name
        bird_id = re.sub(r"bird", "bird", raw_bird_id, flags=re.IGNORECASE)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for seq in root.findall("Sequence"):
            wav_elem = seq.find("WaveFileName")
            pos_elem = seq.find("Position")
            length_elem = seq.find("Length")

            if any(elem is None or elem.text is None for elem in (wav_elem, pos_elem, length_elem)):
                continue

            wav_name = wav_elem.text.strip()
            if not wav_name:
                continue

            wav_path = Path(wav_name)
            wav_stem = wav_path.stem
            wav_suffix = wav_path.suffix or ".wav"
            expected_suffix = f"_{bird_id}"
            expected_suffix_lower = expected_suffix.lower()
            stem_lower = wav_stem.lower()
            if stem_lower.endswith(expected_suffix_lower):
                base_stem = wav_stem[: -len(expected_suffix)]
            else:
                base_stem = wav_stem
            # Source XML omits or uppercases the bird identifier in the filename; normalize to match WAVs.
            wav_name = f"{base_stem}{expected_suffix}{wav_suffix}"

            seq_pos = int(pos_elem.text)
            seq_len = int(length_elem.text)

            units: List[Dict[str, float]] = []
            for note in seq.findall("Note"):
                pos_child = note.find("Position")
                len_child = note.find("Length")
                label_child = note.find("Label")
                if any(child is None or child.text is None for child in (pos_child, len_child, label_child)):
                    continue

                raw_label = label_child.text.strip()
                if raw_label == "":
                    continue

                note_pos = int(pos_child.text)
                note_len = int(len_child.text)
                unit_id = label_map[raw_label]

                unit_on = seq_pos + note_pos
                unit_off = unit_on + note_len
                units.append(
                    {
                        "onset_ms": (1000.0 * unit_on) / SAMPLING_RATE,
                        "offset_ms": (1000.0 * unit_off) / SAMPLING_RATE,
                        "id": unit_id,
                    }
                )

            if not units:
                continue

            event_onset_ms = (1000.0 * seq_pos) / SAMPLING_RATE
            event_offset_ms = (1000.0 * (seq_pos + seq_len)) / SAMPLING_RATE

            key = (bird_id, wav_name)
            if key not in recordings:
                recordings[key] = {
                    "recording": {
                        "filename": wav_name,
                        "bird_id": bird_id,
                        "detected_vocalizations": 0,
                    },
                    "detected_events": [],
                }

            recordings[key]["detected_events"].append(
                {
                    "onset_ms": event_onset_ms,
                    "offset_ms": event_offset_ms,
                    "units": units,
                }
            )
            recordings[key]["recording"]["detected_vocalizations"] += len(units)

    return [recordings[key] for key in sorted(recordings)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Bengalese finch XML annotations to TinyBird JSON.")
    parser.add_argument("--src_dir", required=True, type=Path, help="Directory containing per-bird XML folders.")
    parser.add_argument("--dst_dir", required=True, type=Path, help="Directory where annotations.json is written.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_dir: Path = args.src_dir
    dst_dir: Path = args.dst_dir

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    label_map = build_label_mapping(src_dir)
    recordings = collect_recordings(src_dir, label_map)
    output = {"metadata": {"units": "ms"}, "recordings": recordings}

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / "annotations.json"
    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Wrote {dst_path}")


if __name__ == "__main__":
    main()
