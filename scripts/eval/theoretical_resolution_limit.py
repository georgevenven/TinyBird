#!/usr/bin/env python3
"""
Compute theoretical max performance vs time resolution using annotation JSONs.

Approach:
1) Round ground-truth onsets/offsets to nearest ms.
2) Build 1ms labels per recording (within the max labeled offset).
3) Quantize labels into bins of size resolution_ms.
4) Expand quantized labels back to 1ms and compute metrics vs 1ms GT.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _round_ms(x: float) -> int:
    return int(round(float(x)))


def _load_recordings(path: str, mode: str) -> Tuple[List[Tuple[str, List[Tuple[int, int, int]]]], int]:
    """
    Returns:
      - list of (filename, units) where units are (onset_ms, offset_ms, class_id)
      - num_classes (including silence=0)
    """
    with open(path, "r") as f:
        data = json.load(f)

    recordings: List[Tuple[str, List[Tuple[int, int, int]]]] = []
    max_class_id = 1  # silence=0, so start at 1

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
            max_class_id = max(max_class_id, 1)
        else:
            for event in rec.get("detected_events", []):
                for unit in event.get("units", []):
                    onset = unit.get("onset_ms")
                    offset = unit.get("offset_ms")
                    if onset is None or offset is None:
                        continue
                    class_id = int(unit.get("id", 0)) + 1
                    items.append((_round_ms(onset), _round_ms(offset), class_id))
                    max_class_id = max(max_class_id, class_id)

        items.sort(key=lambda x: x[0])
        recordings.append((filename, items))

    num_classes = max_class_id + 1  # include silence=0
    return recordings, num_classes


def _build_ms_labels(items: List[Tuple[int, int, int]], mode: str) -> np.ndarray:
    if not items:
        return np.zeros((0,), dtype=np.int64)

    max_offset = max(offset for _, offset, _ in items)
    if max_offset <= 0:
        return np.zeros((0,), dtype=np.int64)

    labels = np.zeros((max_offset,), dtype=np.int64)
    for onset, offset, class_id in items:
        if offset <= onset:
            continue
        start = max(0, onset)
        end = max(start, offset)
        if mode == "classify":
            labels[start:end] = class_id
        else:
            labels[start:end] = 1
    return labels


def _quantize_labels(labels: np.ndarray, resolution_ms: int, num_classes: int, mode: str) -> np.ndarray:
    if labels.size == 0:
        return labels
    if resolution_ms <= 1:
        return labels.copy()

    total_ms = labels.shape[0]
    preds = np.zeros_like(labels)
    idx = 0
    while idx < total_ms:
        end = min(total_ms, idx + resolution_ms)
        window = labels[idx:end]

        if mode == "classify":
            counts = np.bincount(window, minlength=num_classes)
            # choose majority class; ties broken by lower class id
            pred_class = int(np.argmax(counts))
        else:
            # binary majority vote (ties -> 0)
            ones = int((window == 1).sum())
            zeros = int(window.size - ones)
            pred_class = 1 if ones > zeros else 0

        preds[idx:end] = pred_class
        idx = end

    return preds


def _update_binary_stats(stats: Dict[str, int], gt: np.ndarray, pred: np.ndarray) -> None:
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    correct = int((pred == gt).sum())
    total = int(gt.size)

    stats["tp"] += tp
    stats["fp"] += fp
    stats["fn"] += fn
    stats["correct"] += correct
    stats["total"] += total


def _update_multiclass_stats(stats: Dict[str, np.ndarray], gt: np.ndarray, pred: np.ndarray, num_classes: int) -> None:
    stats["correct"] += int((pred == gt).sum())
    stats["total"] += int(gt.size)
    for c in range(num_classes):
        tp = int(((pred == c) & (gt == c)).sum())
        fp = int(((pred == c) & (gt != c)).sum())
        fn = int(((pred != c) & (gt == c)).sum())
        stats["tp"][c] += tp
        stats["fp"][c] += fp
        stats["fn"][c] += fn


def _finalize_binary(stats: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
    return {
        "f1": f1 * 100.0,
        "accuracy": acc * 100.0,
        "error": (1.0 - acc) * 100.0,
    }


def _finalize_multiclass(stats: Dict[str, np.ndarray]) -> Dict[str, float]:
    tp = stats["tp"]
    fp = stats["fp"]
    fn = stats["fn"]
    f1s = []
    for c in range(tp.shape[0]):
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
    return {
        "macro_f1": macro_f1 * 100.0,
        "accuracy": acc * 100.0,
        "error": (1.0 - acc) * 100.0,
    }


def main() -> None:
    default_root = os.path.join(os.path.dirname(__file__), "..", "..", "files")
    parser = argparse.ArgumentParser(description="Theoretical max performance vs time resolution.")
    parser.add_argument(
        "--jsons",
        nargs="+",
        default=[
            os.path.join(default_root, "bf_annotations.json"),
            os.path.join(default_root, "canary_annotations.json"),
            os.path.join(default_root, "zf_annotations.json"),
        ],
        help="Annotation JSON files to include (default: bf/canary/zf).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="classify",
        choices=["detect", "unit_detect", "classify"],
        help="Task mode for labeling.",
    )
    parser.add_argument("--min_ms", type=int, default=1, help="Minimum resolution in ms.")
    parser.add_argument("--max_ms", type=int, default=250, help="Maximum resolution in ms.")
    parser.add_argument("--step_ms", type=int, default=1, help="Resolution step in ms.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=os.path.join(default_root, "..", "results", "theoretical_resolution_limit.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    resolutions = list(range(args.min_ms, args.max_ms + 1, args.step_ms))
    rows = []

    for path in args.jsons:
        if not os.path.isfile(path):
            print(f"Missing file: {path}")
            continue

        species = Path(path).stem.replace("_annotations", "")
        recordings, num_classes = _load_recordings(path, args.mode)

        for res in resolutions:
            if args.mode in ["detect", "unit_detect"]:
                stats = {"tp": 0, "fp": 0, "fn": 0, "correct": 0, "total": 0}
            else:
                stats = {
                    "tp": np.zeros((num_classes,), dtype=np.int64),
                    "fp": np.zeros((num_classes,), dtype=np.int64),
                    "fn": np.zeros((num_classes,), dtype=np.int64),
                    "correct": 0,
                    "total": 0,
                }

            for _, items in recordings:
                gt = _build_ms_labels(items, args.mode)
                if gt.size == 0:
                    continue
                pred = _quantize_labels(gt, res, num_classes, args.mode)
                if args.mode in ["detect", "unit_detect"]:
                    _update_binary_stats(stats, gt, pred)
                else:
                    _update_multiclass_stats(stats, gt, pred, num_classes)

            if args.mode in ["detect", "unit_detect"]:
                metrics = _finalize_binary(stats)
            else:
                metrics = _finalize_multiclass(stats)

            for metric_name, metric_value in metrics.items():
                rows.append(
                    {
                        "species": species,
                        "resolution_ms": res,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                    }
                )

    # Overall (pooled) across species
    if rows:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        with open(args.output_csv, "w") as f:
            f.write("species,resolution_ms,metric_name,metric_value\n")
            for row in rows:
                f.write(
                    f"{row['species']},{row['resolution_ms']},{row['metric_name']},{row['metric_value']:.6f}\n"
                )
        print(f"Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()

