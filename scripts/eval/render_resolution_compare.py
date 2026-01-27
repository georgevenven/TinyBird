#!/usr/bin/env python3
"""
Render a compact AVES vs SongMAE comparison figure from the same audio window.

This script loads a WAV + matching spectrogram, runs both trained models on the
same time window, and renders a short (non-cropped) figure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torchaudio

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aves import AvesClassifier  # noqa: E402
from supervised_train import SupervisedTinyBird  # noqa: E402
from utils import load_model_from_checkpoint  # noqa: E402


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _latest_checkpoint(weights_dir: Path, fallback: str) -> Path:
    if (weights_dir / fallback).exists():
        return weights_dir / fallback
    candidates = sorted(weights_dir.glob("model_step_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {weights_dir}")
    return candidates[-1]


def _get_ms_per_timebin(spec_path: Path) -> float:
    params_path = spec_path.parent / "audio_params.json"
    if not params_path.exists():
        return 2.0
    params = _load_json(params_path)
    hop = float(params.get("hop_size", 64))
    sr = float(params.get("sr", 32000))
    if hop <= 0 or sr <= 0:
        return 2.0
    return hop / sr * 1000.0


def _parse_annotation_units(
    annotation_json: Path, wav_name: str
) -> List[Tuple[float, float, int]]:
    data = _load_json(annotation_json)
    units = []
    for rec in data.get("recordings", []):
        rec_info = rec.get("recording", {})
        if rec_info.get("filename") != wav_name:
            continue
        for event in rec.get("detected_events", []):
            for unit in event.get("units", []):
                units.append(
                    (
                        float(unit["onset_ms"]),
                        float(unit["offset_ms"]),
                        int(unit["id"]),
                    )
                )
        break
    return units


def _choose_dense_window(
    units: Iterable[Tuple[float, float, int]],
    window_ms: float,
    max_ms: float,
) -> float:
    onsets = sorted([u[0] for u in units])
    if not onsets:
        return 0.0
    best_start = 0.0
    best_count = -1
    for onset in onsets:
        start = max(0.0, min(onset, max_ms - window_ms))
        end = start + window_ms
        count = sum(1 for t in onsets if start <= t <= end)
        if count > best_count:
            best_count = count
            best_start = start
    return best_start


def _choose_random_window(window_ms: float, max_ms: float, rng: random.Random) -> float:
    if max_ms <= window_ms:
        return 0.0
    return rng.uniform(0.0, max_ms - window_ms)


def _labels_from_units(
    units: Iterable[Tuple[float, float, int]],
    start_ms: float,
    window_ms: float,
    ms_per_bin: float,
    num_classes: int,
) -> np.ndarray:
    bins = int(round(window_ms / ms_per_bin))
    labels = np.zeros((bins,), dtype=np.int64)
    for onset_ms, offset_ms, unit_id in units:
        if unit_id < 0 or unit_id >= num_classes:
            continue
        start = max(start_ms, onset_ms)
        end = min(start_ms + window_ms, offset_ms)
        if end <= start:
            continue
        b0 = int(math.floor((start - start_ms) / ms_per_bin))
        b1 = int(math.ceil((end - start_ms) / ms_per_bin))
        b0 = max(0, min(b0, bins))
        b1 = max(0, min(b1, bins))
        labels[b0:b1] = unit_id
    return labels


def _make_cmap(num_classes: int):
    base = plt.get_cmap("tab20", num_classes)
    colors = base(np.linspace(0, 1, num_classes))
    colors[0] = [0, 0, 0, 1]
    return matplotlib.colors.ListedColormap(colors)


def _plot_bar(ax, labels: np.ndarray, num_classes: int, title: str, window_ms: float):
    cmap = _make_cmap(num_classes)
    img = labels.reshape(1, -1)
    ax.imshow(
        img,
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=max(1, num_classes - 1),
        extent=[0, window_ms, 0, 1],
        origin="lower",
    )
    ax.set_yticks([])
    ax.set_xlim(0, window_ms)
    ax.set_title(title, fontsize=10)


def _plot_stacked_bars(
    ax,
    top_labels: np.ndarray,
    bottom_labels: np.ndarray,
    num_classes: int,
    top_title: str,
    bottom_title: str,
    window_ms: float,
):
    cmap = _make_cmap(num_classes)
    stacked = np.vstack([top_labels, bottom_labels])
    ax.imshow(
        stacked,
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=max(1, num_classes - 1),
        extent=[0, window_ms, 0, 2],
        origin="lower",
    )
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels([bottom_title, top_title], fontsize=9)
    ax.set_xlim(0, window_ms)
    ax.set_title("Predictions", fontsize=10)


def _resample_labels(labels: np.ndarray, target_len: int) -> np.ndarray:
    """Nearest-neighbor resample of categorical labels to target length."""
    src_len = int(labels.shape[0])
    if src_len == target_len:
        return labels
    if src_len <= 1:
        return np.full((target_len,), int(labels[0]) if src_len == 1 else 0, dtype=np.int64)
    # Map each target index to a source index.
    idx = np.linspace(0, src_len - 1, target_len)
    idx = np.rint(idx).astype(int)
    idx = np.clip(idx, 0, src_len - 1)
    return labels[idx]


def _index_wavs(wav_root: Path) -> Dict[str, Path]:
    wavs = {}
    for path in wav_root.rglob("*.wav"):
        stem = path.stem
        wavs.setdefault(stem, path)
    return wavs


def _render_one(
    *,
    wav_path: Path,
    spec_path: Path,
    aves_model: AvesClassifier,
    song_model: SupervisedTinyBird,
    aves_config: Dict,
    song_config: Dict,
    annotation_json: Optional[Path],
    window_ms: float,
    window_mode: str,
    rng: random.Random,
    out_path: Path,
    device: torch.device,
    start_ms_override: Optional[float] = None,
) -> None:
    spec = np.load(spec_path)
    ms_per_bin = _get_ms_per_timebin(spec_path)
    total_ms = spec.shape[1] * ms_per_bin

    if start_ms_override is not None:
        start_ms = max(0.0, min(start_ms_override, total_ms - window_ms))
    else:
        if annotation_json is not None and window_mode == "dense":
            units = _parse_annotation_units(annotation_json, wav_path.name)
            start_ms = _choose_dense_window(units, window_ms, total_ms)
        else:
            start_ms = _choose_random_window(window_ms, total_ms, rng)

    window_bins = int(round(window_ms / ms_per_bin))
    start_bin = int(round(start_ms / ms_per_bin))
    end_bin = start_bin + window_bins
    spec_win = spec[:, start_bin:end_bin]
    if spec_win.shape[1] < window_bins:
        pad = window_bins - spec_win.shape[1]
        spec_win = np.pad(spec_win, ((0, 0), (0, pad)), mode="constant")

    # AVES inference on WAV window (16 kHz)
    wav, sr = torchaudio.load(str(wav_path))
    wav = wav.mean(dim=0, keepdim=True)
    target_sr = int(aves_config.get("audio_sr", 16000))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    start_sample = int(round(start_ms / 1000.0 * sr))
    end_sample = int(round((start_ms + window_ms) / 1000.0 * sr))
    wav_win = wav[:, start_sample:end_sample]
    if wav_win.shape[1] == 0:
        raise SystemExit("WAV window is empty. Check start_ms/window_ms.")

    with torch.no_grad():
        aves_out = aves_model(wav_win.to(device))
        aves_out_len = None
        if isinstance(aves_out, tuple):
            aves_logits, aves_out_len = aves_out
        else:
            aves_logits = aves_out
        aves_logits = aves_logits.squeeze(0).cpu().numpy()
        if aves_logits.ndim == 1:
            aves_preds = (aves_logits >= 0).astype(np.int64)
        else:
            aves_preds = np.argmax(aves_logits, axis=-1).astype(np.int64)

        song_input = torch.from_numpy(spec_win).unsqueeze(0).unsqueeze(0).float().to(device)
        song_logits = song_model(song_input)
        song_logits = song_logits.squeeze(0).cpu().numpy()
        song_preds = np.argmax(song_logits, axis=-1).astype(np.int64)

    # Align AVES predictions to spectrogram bins for stacked display.
    target_len = int(spec_win.shape[1])
    aves_preds = _resample_labels(aves_preds, target_len)
    song_preds = _resample_labels(song_preds, target_len)

    gt_labels = None
    if annotation_json is not None:
        units = _parse_annotation_units(annotation_json, wav_path.name)
        gt_labels = _labels_from_units(
            units,
            start_ms=start_ms,
            window_ms=window_ms,
            ms_per_bin=ms_per_bin,
            num_classes=int(song_config.get("num_classes", 2)),
        )

    num_classes = int(song_config.get("num_classes", 2))

    fig = plt.figure(figsize=(10, 3.6))
    if gt_labels is None:
        gs = gridspec.GridSpec(2, 1, height_ratios=[2.4, 0.7])
    else:
        gs = gridspec.GridSpec(3, 1, height_ratios=[2.2, 0.7, 0.4])

    ax_spec = fig.add_subplot(gs[0, :])
    ax_spec.imshow(
        spec_win,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0, window_ms, 0, spec_win.shape[0]],
    )
    ax_spec.set_ylabel("Mel bins")
    ax_spec.set_title(f"{wav_path.stem}  |  {window_ms:.0f} ms @ {start_ms:.1f} ms")

    ax_pred = fig.add_subplot(gs[1, 0])
    _plot_stacked_bars(
        ax_pred,
        aves_preds,
        song_preds,
        num_classes,
        "AVES (20 ms)",
        "SongMAE (2 ms)",
        window_ms,
    )
    ax_pred.set_xlabel("Time (ms)")
    ax_pred.set_xticks([0, window_ms / 2, window_ms])

    if gt_labels is not None:
        ax_gt = fig.add_subplot(gs[2, 0])
        _plot_bar(ax_gt, gt_labels, num_classes, "Ground truth", window_ms)
        ax_gt.set_xlabel("Time (ms)")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render compact AVES vs SongMAE resolution comparison."
    )
    parser.add_argument("--wav", type=Path, default=None, help="Path to WAV file")
    parser.add_argument("--spec", type=Path, default=None, help="Path to spectrogram .npy")
    parser.add_argument("--spec_root", type=Path, default=None, help="Root directory of spec .npy files")
    parser.add_argument("--wav_root", type=Path, default=None, help="Root directory of wav files")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of random samples to render")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--aves_run", type=Path, required=True, help="AVES run dir")
    parser.add_argument("--song_run", type=Path, required=True, help="SongMAE run dir")
    parser.add_argument(
        "--annotation_json",
        type=Path,
        default=None,
        help="Optional annotations JSON for ground-truth labels",
    )
    parser.add_argument("--start_ms", type=float, default=None, help="Window start time (ms)")
    parser.add_argument("--window_ms", type=float, default=200.0, help="Window duration (ms)")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path (single)")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory (multi-sample)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--window_mode",
        type=str,
        default="random",
        choices=["random", "dense"],
        help="Random or annotation-dense window selection",
    )
    args = parser.parse_args()

    aves_config = _load_json(args.aves_run / "config.json")
    song_config = _load_json(args.song_run / "config.json")

    aves_weights = _latest_checkpoint(args.aves_run / "weights", "model_final.pth")
    song_weights = _latest_checkpoint(args.song_run / "weights", "model_final.pth")

    pretrained_run = Path(song_config["pretrained_run"])

    aves_model = AvesClassifier(
        config_path=Path(aves_config["aves_config_path"]),
        model_path=Path(aves_config["aves_model_path"]),
        num_classes=int(aves_config.get("num_classes", 2)),
        linear_probe=True,
        encoder_layer_idx=aves_config.get("encoder_layer_idx", None),
    )
    aves_state = torch.load(aves_weights, map_location="cpu")
    aves_model.load_state_dict(aves_state)
    aves_model.set_eval_mode()

    song_encoder, song_pre_cfg = load_model_from_checkpoint(
        str(pretrained_run), fallback_to_random=False
    )
    song_model = SupervisedTinyBird(
        pretrained_model=song_encoder,
        config=song_pre_cfg,
        num_classes=int(song_config.get("num_classes", 2)),
        freeze_encoder=bool(song_config.get("freeze_encoder", True)),
        freeze_encoder_up_to=song_config.get("freeze_encoder_up_to", None),
        mode=str(song_config.get("mode", "classify")),
        linear_probe=True,
        lora_rank=int(song_config.get("lora_rank", 0)),
        lora_alpha=float(song_config.get("lora_alpha", 1.0)),
        lora_dropout=float(song_config.get("lora_dropout", 0.0)),
    )
    song_state = torch.load(song_weights, map_location="cpu")
    song_model.load_state_dict(song_state)
    song_model.eval()

    device = torch.device(args.device)
    aves_model.to(device)
    song_model.to(device)

    rng = random.Random(int(args.seed))

    if args.spec_root is not None or args.wav_root is not None:
        if args.spec_root is None or args.wav_root is None:
            raise SystemExit("Provide both --spec_root and --wav_root for multi-sample mode.")
        spec_files = sorted(args.spec_root.rglob("*.npy"))
        if not spec_files:
            raise SystemExit(f"No .npy files found under {args.spec_root}")
        wav_index = _index_wavs(args.wav_root)
        chosen = rng.sample(spec_files, k=min(int(args.num_samples), len(spec_files)))
        out_dir = args.out_dir or (PROJECT_ROOT / "results" / "val_viz" / "resolution_samples")
        for spec_path in chosen:
            stem = spec_path.stem
            wav_path = wav_index.get(stem)
            if wav_path is None:
                print(f"[skip] No wav for {stem}")
                continue
            out_path = out_dir / f"{stem}__{args.window_mode}_{args.window_ms:.0f}ms.png"
            _render_one(
                wav_path=wav_path,
                spec_path=spec_path,
                aves_model=aves_model,
                song_model=song_model,
                aves_config=aves_config,
                song_config=song_config,
                annotation_json=args.annotation_json,
                window_ms=float(args.window_ms),
                window_mode=str(args.window_mode),
                rng=rng,
                out_path=out_path,
                device=device,
                start_ms_override=args.start_ms,
            )
        return

    if args.wav is None or args.spec is None or args.out is None:
        raise SystemExit("Provide --wav, --spec, and --out for single-sample mode.")

    _render_one(
        wav_path=args.wav,
        spec_path=args.spec,
        aves_model=aves_model,
        song_model=song_model,
        aves_config=aves_config,
        song_config=song_config,
        annotation_json=args.annotation_json,
        window_ms=float(args.window_ms),
        window_mode=str(args.window_mode),
        rng=rng,
        out_path=args.out,
        device=device,
        start_ms_override=args.start_ms,
    )


if __name__ == "__main__":
    main()
