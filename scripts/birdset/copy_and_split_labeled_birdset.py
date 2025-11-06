#!/usr/bin/env python3
from __future__ import annotations

"""
Utility to copy locally stored recordings into train/val folders based on
filenames referenced in the BirdSet annotation JSON.

For every recording filename in the JSON, we look for files on disk whose
stem (filename without extension) matches. All matches are copied into the
requested split.
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split local files into train/val sets by filename stem.")
    parser.add_argument("--annotations", required=True, help="Path to xcm_annotations.json (or similar).")
    parser.add_argument("--source_dir", required=True, help="Directory containing the downloaded files.")
    parser.add_argument("--train_dir", required=True, help="Output directory for the training set.")
    parser.add_argument("--val_dir", required=True, help="Output directory for the validation set.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Fraction of stems assigned to train (default: 0.8).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    return parser.parse_args()


def normalize_stem(stem: str) -> str:
    """Return a canonical stem (drop prefixes before XC or XC-like ids)."""
    if "XC" in stem:
        idx = stem.find("XC")
        return stem[idx:]
    return stem


def load_annotation_stems(annotations_path: Path) -> list[str]:
    with annotations_path.open("r") as fp:
        data = json.load(fp)

    normalized = []
    for recording in data.get("recordings", []):
        filename = recording.get("recording", {}).get("filename")
        if not filename:
            continue
        stem = Path(filename).stem
        normalized.append(normalize_stem(stem))

    unique_normalized = sorted(set(normalized))
    print(f"Loaded {len(normalized)} recording entries -> {len(unique_normalized)} unique normalized stems from annotations.")
    return unique_normalized


def index_source_directory(source_dir: Path) -> dict[str, list[Path]]:
    mapping: dict[str, list[Path]] = defaultdict(list)
    for path in source_dir.rglob("*"):
        if path.is_file():
            key = normalize_stem(path.stem)
            mapping[key].append(path)
    total_files = sum(len(files) for files in mapping.values())
    print(f"Indexed {total_files} files under {source_dir}.")
    return mapping


def copy_group(paths: list[Path], destination: Path) -> int:
    copied = 0
    destination.mkdir(parents=True, exist_ok=True)
    for src in paths:
        stem = normalize_stem(src.stem)
        dst_name = f"{stem}{src.suffix}"
        dst = destination / dst_name
        counter = 1
        while dst.exists():
            dst = destination / f"{stem}_{counter}{src.suffix}"
            counter += 1
        shutil.copy2(src, dst)
        copied += 1
    return copied


def main() -> None:
    args = parse_args()

    annotations_path = Path(args.annotations).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()
    train_dir = Path(args.train_dir).expanduser().resolve()
    val_dir = Path(args.val_dir).expanduser().resolve()

    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotations_path}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    stems = load_annotation_stems(annotations_path)
    if not stems:
        print("No stems found in annotations; nothing to do.")
        return

    stem_to_paths = index_source_directory(source_dir)

    missing_stems = [stem for stem in stems if stem not in stem_to_paths]
    if missing_stems:
        print(f"WARNING: {len(missing_stems)} stems not found in source directory (showing up to 10): {missing_stems[:10]}")

    present_stems = [stem for stem in stems if stem in stem_to_paths]
    if not present_stems:
        print("No matching stems found in source directory; aborting.")
        return

    random.seed(args.seed)
    random.shuffle(present_stems)

    train_count = max(1, int(len(present_stems) * args.train_ratio))
    train_stems = set(present_stems[:train_count])
    val_stems = set(present_stems[train_count:])
    if not val_stems:
        # Ensure we always have at least one item in validation if possible
        moved_stem = present_stems[train_count - 1]
        train_stems.remove(moved_stem)
        val_stems.add(moved_stem)

    print(f"Splitting {len(present_stems)} stems -> train: {len(train_stems)}, val: {len(val_stems)}")

    train_copied = sum(copy_group(stem_to_paths[stem], train_dir) for stem in train_stems)
    val_copied = sum(copy_group(stem_to_paths[stem], val_dir) for stem in val_stems)

    print("\n=== Copy summary ===")
    print(f"Train files copied: {train_copied} (from {len(train_stems)} stems)")
    print(f"Val files copied:   {val_copied} (from {len(val_stems)} stems)")
    if missing_stems:
        print(f"Missing stems (not copied): {len(missing_stems)}")

    audio_params = source_dir / "audio_params.json"
    if audio_params.exists():
        shutil.copy2(audio_params, train_dir / "audio_params.json")
        shutil.copy2(audio_params, val_dir / "audio_params.json")
        print("Copied audio_params.json to both splits.")
    else:
        print("audio_params.json not found in source directory; skipped copying.")


if __name__ == "__main__":
    main()
