#!/usr/bin/env python3
"""
Compute F1 from exported val_outputs for finetuned vs linear_probe runs.

Mirrors supervised_train.py compute_f1_score:
  - Binary (num_classes == 2): positive-class F1 with sigmoid threshold 0.5
  - Multi-class: macro-F1 over all classes (including silence=0)
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _infer_probe_mode(run_name: str) -> str:
    name = run_name.lower()
    if name.startswith("linear") or name.startswith("linear_") or name.startswith("linear/"):
        return "linear"
    if name.startswith("finetune") or name.startswith("finetune_") or name.startswith("finetune/"):
        return "finetune"
    return "unknown"


def _iter_run_dirs(runs_root: Path) -> Iterable[Tuple[str, Path]]:
    flat_runs: List[Path] = []
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "val_outputs").exists() or (child / "config.json").exists():
            flat_runs.append(child)

    if flat_runs:
        for run_dir in flat_runs:
            yield _infer_probe_mode(run_dir.name), run_dir
        return

    for probe_dir in sorted(runs_root.iterdir()):
        if not probe_dir.is_dir():
            continue
        probe_name = probe_dir.name
        for run_dir in sorted(probe_dir.iterdir()):
            if run_dir.is_dir():
                yield probe_name, run_dir


def _valid_patches(length: int, patch_width: int) -> int:
    if length <= 0:
        return 0
    return int(math.ceil(length / patch_width))


def _f1_binary(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1 * 100.0


def _f1_macro(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
    f1s: List[float] = []
    for c in range(len(tp)):
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) * 100.0 if f1s else 0.0


def _species_from_run(run_dir: Path) -> str:
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            annot = Path(cfg.get("annotation_file", "")).name.lower()
            if "canary" in annot:
                return "Canary"
            if annot.startswith("bf") or "bengalese" in annot:
                return "Bengalese_Finch"
            if annot.startswith("zf") or "zebra" in annot:
                return "Zebra_Finch"
        except Exception:
            pass

    for sp in ["Bengalese_Finch", "Canary", "Zebra_Finch"]:
        if run_dir.name.startswith(sp):
            return sp
    return "unknown"


def _eval_val_outputs(val_dir: Path, threshold: float) -> Dict[str, float]:
    logits = np.load(val_dir / "logits.npy", mmap_mode="r")
    labels_patches = np.load(val_dir / "labels_patches.npy", mmap_mode="r")
    window_lengths = np.load(val_dir / "window_lengths.npy", mmap_mode="r")

    with open(val_dir / "meta.json", "r") as f:
        meta = json.load(f)

    num_classes = int(meta["num_classes"])
    patch_width = int(meta["patch_width"])
    mode = meta.get("mode", "unknown")

    if num_classes == 2:
        tp = fp = fn = 0
        for i in range(logits.shape[0]):
            length = int(window_lengths[i])
            valid = _valid_patches(length, patch_width)
            if valid == 0:
                continue
            probs = 1.0 / (1.0 + np.exp(-logits[i, :valid, 0]))
            preds = (probs > threshold).astype(np.int64)
            labels = labels_patches[i, :valid].astype(np.int64)
            tp += int(((preds == 1) & (labels == 1)).sum())
            fp += int(((preds == 1) & (labels == 0)).sum())
            fn += int(((preds == 0) & (labels == 1)).sum())
        f1 = _f1_binary(tp, fp, fn)
    else:
        tp = np.zeros((num_classes,), dtype=np.int64)
        fp = np.zeros((num_classes,), dtype=np.int64)
        fn = np.zeros((num_classes,), dtype=np.int64)
        for i in range(logits.shape[0]):
            length = int(window_lengths[i])
            valid = _valid_patches(length, patch_width)
            if valid == 0:
                continue
            preds = np.argmax(logits[i, :valid, :], axis=-1).astype(np.int64)
            labels = labels_patches[i, :valid].astype(np.int64)
            for c in range(num_classes):
                tp[c] += int(((preds == c) & (labels == c)).sum())
                fp[c] += int(((preds == c) & (labels != c)).sum())
                fn[c] += int(((preds != c) & (labels == c)).sum())
        f1 = _f1_macro(tp, fp, fn)

    return {
        "f1": f1,
        "num_classes": float(num_classes),
        "patch_width": float(patch_width),
        "mode": mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute F1 from val_outputs across finetuned/linear_probe runs."
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="/media/george-vengrovski/Desk SSD/results",
        help="Root folder containing runs/finetuned and runs/linear_probe",
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        default=None,
        help="Optional direct path to runs directory (supports flat or probe-mode subdirs).",
    )
    parser.add_argument(
        "--run_names",
        type=str,
        default=None,
        help="Comma-separated run directory names to evaluate (optional filter).",
    )
    parser.add_argument(
        "--run_names_file",
        type=str,
        default=None,
        help="File containing run directory names to evaluate (one per line).",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Output CSV path (default: <results_root>/eval_f1.csv)",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Output summary CSV path (default: <results_root>/eval_f1_summary.csv)",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for binary F1")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    runs_root = Path(args.runs_root) if args.runs_root else (results_root / "runs")
    out_csv = Path(args.out_csv) if args.out_csv else results_root / "eval_f1.csv"
    summary_csv = (
        Path(args.summary_csv) if args.summary_csv else results_root / "eval_f1_summary.csv"
    )

    rows: List[str] = ["probe_mode,run_name,mode,species,num_classes,patch_width,f1"]
    records: List[Dict[str, object]] = []

    run_filter: set[str] = set()
    if args.run_names:
        run_filter.update({n.strip() for n in args.run_names.split(",") if n.strip()})
    if args.run_names_file:
        run_names_path = Path(args.run_names_file)
        if not run_names_path.exists():
            raise SystemExit(f"Run names file not found: {run_names_path}")
        for line in run_names_path.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name:
                run_filter.add(name)

    for probe_mode, run_dir in _iter_run_dirs(runs_root):
        if run_filter and run_dir.name not in run_filter:
            continue
        val_dir = run_dir / "val_outputs"
        if not val_dir.exists():
            continue
        stats = _eval_val_outputs(val_dir, args.threshold)
        species = _species_from_run(run_dir)
        rows.append(
            f"{probe_mode},{run_dir.name},{stats['mode']},{species},"
            f"{int(stats['num_classes'])},{int(stats['patch_width'])},{stats['f1']:.4f}"
        )
        records.append(
            {
                "probe_mode": probe_mode,
                "run_name": run_dir.name,
                "mode": stats["mode"],
                "species": species,
                "f1": float(stats["f1"]),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    # Summary: mean/std per (probe_mode, mode, species) and per (probe_mode, mode) across species.
    summary_rows: List[str] = ["probe_mode,mode,species,n,f1_mean,f1_std"]
    by_group: Dict[Tuple[str, str, str], List[float]] = {}
    for rec in records:
        key = (str(rec["probe_mode"]), str(rec["mode"]), str(rec["species"]))
        by_group.setdefault(key, []).append(float(rec["f1"]))

    for (probe_mode, mode, species), f1s in sorted(by_group.items()):
        arr = np.array(f1s, dtype=np.float64)
        mean = float(arr.mean()) if len(arr) else 0.0
        std = float(arr.std(ddof=0)) if len(arr) else 0.0
        summary_rows.append(f"{probe_mode},{mode},{species},{len(arr)},{mean:.4f},{std:.4f}")

    by_species_mean: Dict[Tuple[str, str], List[float]] = {}
    for (probe_mode, mode, _species), f1s in by_group.items():
        arr = np.array(f1s, dtype=np.float64)
        mean = float(arr.mean()) if len(arr) else 0.0
        by_species_mean.setdefault((probe_mode, mode), []).append(mean)

    for (probe_mode, mode), means in sorted(by_species_mean.items()):
        arr = np.array(means, dtype=np.float64)
        mean = float(arr.mean()) if len(arr) else 0.0
        std = float(arr.std(ddof=0)) if len(arr) else 0.0
        summary_rows.append(f"{probe_mode},{mode},ALL,{len(arr)},{mean:.4f},{std:.4f}")

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.write_text("\n".join(summary_rows) + "\n", encoding="utf-8")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {summary_csv}")


if __name__ == "__main__":
    main()
