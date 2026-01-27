#!/usr/bin/env python3
"""
Create a side-by-side figure that highlights temporal resolution differences.

This script scans existing prediction PNGs (generated elsewhere) and crops a
~200 ms window with dense syllable changes from each model, then combines them
into a single comparison figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageDraw, ImageFont


def _load_meta_timebin_ms(meta_path: Path) -> float:
    if not meta_path.exists():
        raise SystemExit(f"meta.json not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return float(meta.get("ms_per_timebin_default", 2.0))


def _load_meta_timebins(meta_path: Path) -> int:
    if not meta_path.exists():
        raise SystemExit(f"meta.json not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n_timebins = int(meta.get("n_timebins", 0))
    if n_timebins <= 0:
        raise SystemExit(f"Invalid n_timebins in {meta_path}")
    return n_timebins


def _row_transitions(pixels: Iterable[Tuple[int, int, int]]) -> int:
    it = iter(pixels)
    try:
        prev = next(it)
    except StopIteration:
        return 0
    count = 0
    for px in it:
        if px != prev:
            count += 1
            prev = px
    return count


def _find_bar_row(img: Image.Image) -> Tuple[int, int, int]:
    """
    Return (row_index, bar_left, bar_right) for the ground-truth bar region.
    """
    rgb = img.convert("RGB")
    w, h = rgb.size
    pix = rgb.load()
    # Assume white-ish background for axes. Sample top-left.
    bg = pix[5, 5]
    y_start = int(h * 0.8)
    y_end = int(h * 0.98)
    best = (y_start, 0, w - 1)
    best_score = -1
    for y in range(y_start, y_end):
        # Find foreground span by thresholding against background.
        xs = []
        for x in range(int(w * 0.05), int(w * 0.98)):
            px = pix[x, y]
            if sum(abs(px[i] - bg[i]) for i in range(3)) > 25:
                xs.append(x)
        if not xs:
            continue
        left, right = min(xs), max(xs)
        if right - left < w * 0.3:
            continue
        row_pixels = [pix[x, y] for x in range(left, right)]
        score = _row_transitions(row_pixels)
        if score > best_score:
            best_score = score
            best = (y, left, right)
    return best


def _best_window_for_image(
    img_path: Path,
    n_timebins: int,
    window_bins: int,
) -> Tuple[int, int, int, int, int]:
    """
    Returns (score, start_bin, bar_left, bar_right, bar_row).
    """
    img = Image.open(img_path).convert("RGB")
    bar_row, bar_left, bar_right = _find_bar_row(img)
    bar_width = max(1, bar_right - bar_left)
    # Extract a 3-pixel tall strip around the bar row to reduce noise.
    y0 = max(0, bar_row - 1)
    y1 = min(img.height, bar_row + 2)
    bar_strip = img.crop((bar_left, y0, bar_right, y1))
    # Collapse to 1px height and resize to timebins.
    bar_strip = bar_strip.resize((bar_width, 1), resample=Image.NEAREST)
    bar_strip = bar_strip.resize((n_timebins, 1), resample=Image.NEAREST)
    pixels = list(bar_strip.getdata())
    best_score = -1
    best_start = 0
    max_start = max(0, n_timebins - window_bins)
    for start in range(0, max_start + 1):
        window = pixels[start : start + window_bins]
        score = _row_transitions(window)
        if score > best_score:
            best_score = score
            best_start = start
    return best_score, best_start, bar_left, bar_right, bar_row


def _select_best_crop(
    image_paths: Iterable[Path],
    n_timebins: int,
    window_bins: int,
) -> Tuple[Path, Tuple[int, int]]:
    best = None
    for path in image_paths:
        score, start_bin, bar_left, bar_right, _ = _best_window_for_image(
            path, n_timebins, window_bins
        )
        if best is None or score > best[0]:
            best = (score, path, start_bin, bar_left, bar_right)
    if best is None:
        raise SystemExit("No images found to analyze.")
    _, path, start_bin, bar_left, bar_right = best
    bar_width = max(1, bar_right - bar_left)
    x0 = int(bar_left + (start_bin / n_timebins) * bar_width)
    x1 = int(bar_left + ((start_bin + window_bins) / n_timebins) * bar_width)
    return path, (x0, x1)


def _crop_window(img_path: Path, x_range: Tuple[int, int]) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    x0, x1 = x_range
    x0 = max(0, min(img.width - 1, x0))
    x1 = max(x0 + 1, min(img.width, x1))
    return img.crop((x0, 0, x1, img.height))


def _draw_label(img: Image.Image, text: str) -> Image.Image:
    width, height = img.size
    pad = 12
    label_height = 36
    out = Image.new("RGB", (width, height + label_height + pad), (255, 255, 255))
    out.paste(img, (0, label_height + pad))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    # Pillow 10+ removed textsize; use textbbox when available.
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        text_w, text_h = draw.textsize(text, font=font)
    draw.text(((width - text_w) // 2, (label_height - text_h) // 2), text, fill=(0, 0, 0), font=font)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create AVES vs SongMAE temporal resolution figure (200 ms window)."
    )
    parser.add_argument("--aves_dir", type=Path, required=True)
    parser.add_argument("--song_dir", type=Path, required=True)
    parser.add_argument("--aves_meta", type=Path, required=True)
    parser.add_argument("--song_meta", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window_ms", type=float, default=200.0)
    args = parser.parse_args()

    aves_ms = _load_meta_timebin_ms(args.aves_meta)
    song_ms = _load_meta_timebin_ms(args.song_meta)
    aves_bins = _load_meta_timebins(args.aves_meta)
    song_bins = _load_meta_timebins(args.song_meta)

    aves_window_bins = max(1, int(round(args.window_ms / aves_ms)))
    song_window_bins = max(1, int(round(args.window_ms / song_ms)))

    aves_images = sorted(args.aves_dir.glob("*.png"))
    song_images = sorted(args.song_dir.glob("*val*.png"))
    if not aves_images:
        raise SystemExit(f"No PNGs found in {args.aves_dir}")
    if not song_images:
        raise SystemExit(f"No PNGs found in {args.song_dir}")

    aves_img, aves_x = _select_best_crop(aves_images, aves_bins, aves_window_bins)
    song_img, song_x = _select_best_crop(song_images, song_bins, song_window_bins)

    aves_crop = _crop_window(aves_img, aves_x)
    song_crop = _crop_window(song_img, song_x)

    aves_crop = _draw_label(aves_crop, "AVES (20 ms)")
    song_crop = _draw_label(song_crop, "SongMAE (2 ms)")

    pad = 16
    combined_width = aves_crop.width + song_crop.width + pad
    combined_height = max(aves_crop.height, song_crop.height)
    combined = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
    combined.paste(aves_crop, (0, 0))
    combined.paste(song_crop, (aves_crop.width + pad, 0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.save(args.out)

    print(f"AVES source: {aves_img}")
    print(f"SongMAE source: {song_img}")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
