#!/usr/bin/env python3
"""
Compute F1 at millisecond resolution using posthoc logits export.

Assumes model outputs are aligned to timebins, where timebin i maps to:
  ms = (i + 1) * ms_per_timebin

Ground-truth onsets/offsets are rounded to nearest ms before labeling.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add `src/` to path because internal modules use absolute imports like `import utils`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from utils import parse_chunk_ms


def _round_ms(x: float) -> int:
    return int(round(float(x)))


def _load_annotations(path: str, mode: str) -> Dict[str, List[Tuple[int, int, int]]]:
    """
    Return mapping: filename -> list of (onset_ms, offset_ms, class_id).
    For detect: class_id is 1 for any event.
    For unit_detect/classify: class_id is unit["id"] + 1 (silence=0).
    """
    with open(path, "r") as f:
        data = json.load(f)

    out: Dict[str, List[Tuple[int, int, int]]] = {}
    for rec in data.get("recordings", []):
        filename = Path(rec["recording"]["filename"]).stem
        items: List[Tuple[int, int, int]] = []

        if mode == "detect":
            for event in rec.get("detected_events", []):
                onset = event.get("onset_ms")
                offset = event.get("offset_ms")
                if onset is None or offset is None:
                    continue
                items.append((_round_ms(onset), _round_ms(offset), 1))
        else:
            for event in rec.get("detected_events", []):
                for unit in event.get("units", []):
                    onset = unit.get("onset_ms")
                    offset = unit.get("offset_ms")
                    if onset is None or offset is None:
                        continue
                    class_id = int(unit.get("id", 0)) + 1
                    items.append((_round_ms(onset), _round_ms(offset), class_id))

        items.sort(key=lambda x: x[0])
        out[filename] = items

    return out


def _labels_from_units(
    units: List[Tuple[int, int, int]],
    total_timebins: int,
    ms_per_timebin: float,
    mode: str,
) -> np.ndarray:
    labels = np.zeros((total_timebins,), dtype=np.int64)
    if not units:
        return labels

    for onset_ms, offset_ms, class_id in units:
        if offset_ms <= onset_ms:
            continue

        # timebin i maps to ms = (i + 1) * ms_per_timebin
        start_i = max(0, int(math.ceil(onset_ms / ms_per_timebin)) - 1)
        end_i = int(math.floor((offset_ms - 1) / ms_per_timebin)) - 1
        if end_i < start_i:
            continue
        end_i = min(end_i, total_timebins - 1)

        if mode == "classify":
            labels[start_i : end_i + 1] = class_id
        else:
            labels[start_i : end_i + 1] = 1

    return labels


def _f1_binary(preds: np.ndarray, labels: np.ndarray) -> float:
    preds = preds.astype(np.int64)
    labels = labels.astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1 * 100.0


def _f1_macro(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) * 100.0 if f1s else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ms-resolution F1 from exported logits.")
    parser.add_argument("--val_outputs_dir", type=str, required=True, help="runs/<run_name>/val_outputs")
    parser.add_argument("--annotation_json", type=str, required=True, help="Annotation JSON used for labels")
    parser.add_argument("--mode", type=str, required=True, choices=["detect", "unit_detect", "classify"])
    parser.add_argument(
        "--audio_params",
        type=str,
        default=None,
        help="audio_params.json (must include sr and hop_size)",
    )
    parser.add_argument(
        "--ms_per_timebin",
        type=float,
        default=None,
        help="Override ms per timebin when audio_params is unavailable.",
    )
    parser.add_argument(
        "--ms_per_timebin_by_file",
        type=str,
        default=None,
        help="Optional JSON mapping filename -> ms_per_timebin.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for binary preds")
    args = parser.parse_args()

    out_dir = Path(args.val_outputs_dir)
    logits = np.load(out_dir / "logits.npy", mmap_mode="r")
    labels_patches = np.load(out_dir / "labels_patches.npy", mmap_mode="r")
    window_starts = np.load(out_dir / "window_starts.npy", mmap_mode="r")
    window_lengths = np.load(out_dir / "window_lengths.npy", mmap_mode="r")
    with open(out_dir / "filenames.json", "r") as f:
        filenames = json.load(f)

    with open(out_dir / "meta.json", "r") as f:
        meta = json.load(f)

    patch_width = int(meta["patch_width"])
    num_classes = int(meta["num_classes"])

    annotations = _load_annotations(args.annotation_json, args.mode)

    default_ms_per_timebin = None
    if args.ms_per_timebin is not None:
        default_ms_per_timebin = float(args.ms_per_timebin)
    elif args.audio_params:
        with open(args.audio_params, "r") as f:
            audio = json.load(f)
        if "sr" not in audio or "hop_size" not in audio:
            raise SystemExit(f"audio_params missing sr/hop_size: {args.audio_params}")
        default_ms_per_timebin = float(audio["hop_size"]) / float(audio["sr"]) * 1000.0

    ms_per_timebin_by_file: Dict[str, float] = {}
    if args.ms_per_timebin_by_file:
        mapping_path = Path(args.ms_per_timebin_by_file)
        if not mapping_path.exists():
            raise SystemExit(f"ms_per_timebin_by_file not found: {mapping_path}")
        with open(mapping_path, "r") as f:
            raw = json.load(f)
        ms_per_timebin_by_file = {str(k): float(v) for k, v in raw.items()}

    if default_ms_per_timebin is None and not ms_per_timebin_by_file:
        raise SystemExit("Provide --ms_per_timebin, --audio_params, or --ms_per_timebin_by_file.")

    def _ms_per_timebin_for(fn: str) -> float:
        if fn in ms_per_timebin_by_file:
            return float(ms_per_timebin_by_file[fn])
        if default_ms_per_timebin is None:
            raise SystemExit(f"No ms_per_timebin available for file: {fn}")
        return float(default_ms_per_timebin)

    # Precompute max timebins per file to build label arrays
    file_max_timebin: Dict[str, int] = {}
    ms_per_timebin_seen: Dict[str, float] = {}
    for fn, start, length in zip(filenames, window_starts, window_lengths):
        end = int(start) + int(length)
        file_max_timebin[fn] = max(file_max_timebin.get(fn, 0), end)
        if fn not in ms_per_timebin_seen:
            ms_per_timebin_seen[fn] = _ms_per_timebin_for(fn)

    file_max_ms: Dict[str, int] = {}
    for fn, max_t in file_max_timebin.items():
        ms_ptb = ms_per_timebin_seen.get(fn, _ms_per_timebin_for(fn))
        file_max_ms[fn] = int(math.ceil(float(max_t) * float(ms_ptb)))

    labels_ms_by_file: Dict[str, np.ndarray] = {}
    preds_ms_by_file: Dict[str, np.ndarray] = {}
    mask_ms_by_file: Dict[str, np.ndarray] = {}
    for fn, max_t in file_max_timebin.items():
        base_filename, chunk_start_ms, chunk_end_ms = parse_chunk_ms(fn)
        units = annotations.get(base_filename, [])
        ms_per_timebin = ms_per_timebin_seen.get(fn, _ms_per_timebin_for(fn))
        if chunk_start_ms is not None:
            clipped = []
            chunk_end = chunk_end_ms if chunk_end_ms is not None else float("inf")
            for onset_ms, offset_ms, class_id in units:
                if offset_ms <= chunk_start_ms or onset_ms >= chunk_end:
                    continue
                new_onset = max(onset_ms, chunk_start_ms) - chunk_start_ms
                new_offset = min(offset_ms, chunk_end) - chunk_start_ms
                clipped.append((new_onset, new_offset, class_id))
            units = clipped
        n_ms = int(file_max_ms.get(fn, 0) or 0)
        if n_ms <= 0:
            n_ms = int(math.ceil(float(max_t) * float(ms_per_timebin)))
        labels_ms = np.zeros((n_ms,), dtype=np.int64)
        preds_ms = np.zeros((n_ms,), dtype=np.int64)
        mask_ms = np.zeros((n_ms,), dtype=np.bool_)
        for onset_ms, offset_ms, class_id in units:
            onset_i = int(math.floor(float(onset_ms)))
            offset_i = int(math.ceil(float(offset_ms)))
            onset_i = max(0, min(onset_i, n_ms))
            offset_i = max(0, min(offset_i, n_ms))
            if offset_i <= onset_i:
                continue
            if args.mode == "classify":
                labels_ms[onset_i:offset_i] = int(class_id)
            else:
                labels_ms[onset_i:offset_i] = 1
        labels_ms_by_file[fn] = labels_ms
        preds_ms_by_file[fn] = preds_ms
        mask_ms_by_file[fn] = mask_ms

    for i in range(logits.shape[0]):
        fn = filenames[i]
        start = int(window_starts[i])
        length = int(window_lengths[i])
        if length <= 0:
            continue

        # Convert logits -> per-timebin preds
        if args.mode in ["detect", "unit_detect"]:
            probs = 1.0 / (1.0 + np.exp(-logits[i, :, 0]))
            pred_patches = (probs > args.threshold).astype(np.int64)
        else:
            pred_patches = np.argmax(logits[i, :, :], axis=-1).astype(np.int64)

        pred_time = np.repeat(pred_patches, patch_width)[: length]
        ms_ptb = ms_per_timebin_seen.get(fn, _ms_per_timebin_for(fn))
        file_len_tb = int(file_max_timebin.get(fn, start + length))
        file_len_ms = int(file_max_ms.get(fn, 0) or 0)
        if file_len_ms <= 0:
            file_len_ms = int(math.ceil(float(file_len_tb) * float(ms_ptb)))
        preds_ms = preds_ms_by_file.get(fn)
        labels_ms = labels_ms_by_file.get(fn)
        mask_ms = mask_ms_by_file.get(fn)
        if preds_ms is None or labels_ms is None or mask_ms is None:
            continue
        if preds_ms.shape[0] < file_len_ms:
            file_len_ms = preds_ms.shape[0]

        for j in range(length):
            tb_start = int(start + j)
            tb_end = tb_start + 1
            ms_start = int(math.floor(float(tb_start) * float(ms_ptb)))
            if tb_end >= file_len_tb:
                ms_end = file_len_ms
            else:
                ms_end = int(math.floor(float(tb_end) * float(ms_ptb)))
            ms_start = max(0, min(ms_start, file_len_ms))
            ms_end = max(ms_start + 1, min(ms_end, file_len_ms))
            preds_ms[ms_start:ms_end] = int(pred_time[j])
            mask_ms[ms_start:ms_end] = True

    all_preds_list = []
    all_labels_list = []
    for fn, preds_ms in preds_ms_by_file.items():
        labels_ms = labels_ms_by_file.get(fn)
        mask_ms = mask_ms_by_file.get(fn)
        if labels_ms is None or mask_ms is None:
            continue
        valid = mask_ms.astype(np.bool_)
        if not valid.any():
            continue
        all_preds_list.append(preds_ms[valid])
        all_labels_list.append(labels_ms[valid])

    all_preds = np.concatenate(all_preds_list) if all_preds_list else np.array([], dtype=np.int64)
    all_labels = np.concatenate(all_labels_list) if all_labels_list else np.array([], dtype=np.int64)

    if args.mode in ["detect", "unit_detect"]:
        f1 = _f1_binary(all_preds, all_labels)
        print(f"MS-F1 (binary): {f1:.2f}")
    else:
        f1 = _f1_macro(all_preds, all_labels, num_classes)
        print(f"MS-F1 (macro): {f1:.2f}")


if __name__ == "__main__":
    main()
