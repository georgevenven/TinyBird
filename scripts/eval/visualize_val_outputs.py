#!/usr/bin/env python3
"""
Visualize exported val_outputs by overlaying spectrograms, predictions, and labels.

This is especially useful for AVES runs, which export logits/labels but do not
render the spectrogram-style plots during training like supervised_train.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np

# Use a headless backend before importing pyplot via plotting_utils.
import matplotlib

matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from plotting_utils import save_supervised_prediction_plot  # noqa: E402


def _resolve_run_dir(run_name: str, runs_root: Path) -> Path:
    run_path = Path(run_name)
    if run_path.is_absolute():
        return run_path
    return runs_root / run_path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _choose_indices(
    *,
    n: int,
    k: int,
    valid_mask: np.ndarray,
    seed: int,
) -> List[int]:
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size == 0:
        return []
    if k <= 0 or k >= valid_idx.size:
        return valid_idx.tolist()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(valid_idx, size=int(k), replace=False)
    chosen.sort()
    return chosen.tolist()


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    denom = np.sum(ex, axis=-1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return ex / denom


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _pad_spec_window(spec: np.ndarray, start: int, window_len: int) -> np.ndarray:
    """Slice a spectrogram window and right-pad to window_len if needed."""
    mel_bins = int(spec.shape[0])
    out = np.zeros((mel_bins, int(window_len)), dtype=spec.dtype)
    if start < 0:
        start = 0
    end = min(spec.shape[1], start + window_len)
    if end > start:
        out[:, : end - start] = spec[:, start:end]
    return out


def _sanitize_filename(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def visualize_val_outputs(
    *,
    run_dir: Path,
    spec_dir: Path,
    output_dir: Path,
    num_samples: int,
    seed: int,
    threshold: float,
    split: str,
    indices: Iterable[int] | None,
) -> None:
    val_dir = run_dir / "val_outputs"
    if not val_dir.exists():
        raise SystemExit(f"val_outputs not found: {val_dir}")

    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"config.json not found: {config_path}")
    config = _load_json(config_path)

    meta_path = val_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"meta.json not found: {meta_path}")
    meta = _load_json(meta_path)

    mode = str(config.get("mode", meta.get("mode", "detect")))
    num_classes = int(config.get("num_classes", meta.get("num_classes", 2)))
    window_len = int(meta.get("n_timebins") or config.get("num_timebins") or 0)
    if window_len <= 0:
        raise SystemExit(
            "Unable to determine window length. Expected meta['n_timebins'] or config['num_timebins']."
        )

    logits = np.load(val_dir / "logits.npy", mmap_mode="r")
    labels_patches = np.load(val_dir / "labels_patches.npy", mmap_mode="r")
    window_starts = np.load(val_dir / "window_starts.npy", mmap_mode="r")
    window_lengths = np.load(val_dir / "window_lengths.npy", mmap_mode="r")
    filenames = _load_json(val_dir / "filenames.json")

    n_windows = int(logits.shape[0])
    if len(filenames) != n_windows:
        raise SystemExit(
            f"filenames.json length mismatch: {len(filenames)} vs logits windows: {n_windows}"
        )

    valid_mask = np.asarray(window_lengths) > 0
    if indices:
        chosen_idx = [i for i in indices if 0 <= int(i) < n_windows]
    else:
        chosen_idx = _choose_indices(
            n=n_windows,
            k=num_samples,
            valid_mask=valid_mask,
            seed=seed,
        )

    if not chosen_idx:
        raise SystemExit("No windows selected to visualize.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_dir}")
    print(f"Spec directory: {spec_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Selected windows: {len(chosen_idx)} / {n_windows}")

    for step_num, idx in enumerate(chosen_idx, start=1):
        filename = str(filenames[idx])
        start = int(window_starts[idx])
        length = int(window_lengths[idx])

        spec_path = spec_dir / f"{filename}.npy"
        if not spec_path.exists():
            print(f"[skip] Missing spectrogram: {spec_path}")
            continue

        spec = np.load(spec_path)
        spec_win = _pad_spec_window(spec, start=start, window_len=window_len)

        logits_win = np.asarray(logits[idx])
        labels_win = np.asarray(labels_patches[idx]).astype(np.int64, copy=False)

        if logits_win.ndim == 2 and logits_win.shape[-1] == 1:
            logits_flat = logits_win.reshape(-1)
            vocal_prob = _sigmoid(logits_flat)
            probs = np.stack([1.0 - vocal_prob, vocal_prob], axis=-1)
            preds = (vocal_prob >= float(threshold)).astype(np.int64)
        else:
            probs = _softmax(logits_win)
            preds = np.argmax(probs, axis=-1).astype(np.int64)

        # Keep the filename descriptive so multiple windows do not overwrite.
        desc = f"{filename}__start_{start}__len_{length}"
        desc = _sanitize_filename(desc)

        temp_path = save_supervised_prediction_plot(
            spectrogram=spec_win,
            labels=labels_win,
            predictions=preds,
            probabilities=probs if mode in ("detect", "unit_detect") else None,
            logits=logits_win,
            filename=desc,
            mode=mode,
            num_classes=num_classes,
            output_dir=str(output_dir),
            step_num=step_num,
            split=split,
        )

        # plotting_utils uses a fixed filename; move it to a unique name.
        temp_path = Path(temp_path)
        final_name = f"prediction_{split}_idx_{idx:06d}_{desc}.png"
        final_path = output_dir / final_name
        os.replace(temp_path, final_path)
        print(f"[ok] {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render spectrogram + prediction + ground-truth plots from val_outputs."
    )
    parser.add_argument("--run_name", type=str, required=True, help="Run name or path under runs/")
    parser.add_argument(
        "--runs_root",
        type=Path,
        default=PROJECT_ROOT / "runs",
        help="Root directory that contains run folders (default: ./runs)",
    )
    parser.add_argument(
        "--spec_dir",
        type=Path,
        default=None,
        help="Override spectrogram directory (default: run config val_dir)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write images (default: results/val_viz/<run_name>)",
    )
    parser.add_argument("--num_samples", type=int, default=12, help="Number of windows to visualize")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for window sampling")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for binary outputs",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Split label to embed in filenames/titles (default: val)",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default="",
        help="Optional comma-separated list of explicit window indices to render",
    )

    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_name, args.runs_root)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    config = _load_json(run_dir / "config.json")
    spec_dir = args.spec_dir or Path(config.get("val_dir", ""))
    if not spec_dir or not spec_dir.exists():
        raise SystemExit(
            f"Could not resolve spec_dir. Provide --spec_dir. Attempted: {spec_dir}"
        )

    output_dir = args.output_dir
    if output_dir is None:
        safe_run = _sanitize_filename(str(args.run_name).replace(os.sep, "_"))
        output_dir = PROJECT_ROOT / "results" / "val_viz" / safe_run

    indices: List[int] | None = None
    if args.indices.strip():
        indices = [int(x) for x in args.indices.split(",") if x.strip()]

    visualize_val_outputs(
        run_dir=run_dir,
        spec_dir=spec_dir,
        output_dir=output_dir,
        num_samples=int(args.num_samples),
        seed=int(args.seed),
        threshold=float(args.threshold),
        split=str(args.split),
        indices=indices,
    )


if __name__ == "__main__":
    main()

