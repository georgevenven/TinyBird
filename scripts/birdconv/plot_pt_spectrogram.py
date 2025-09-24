#!/usr/bin/env python3

"""
Plot a spectrogram stored in a .pt file (as produced by audio2spec.py) into a single PNG,
stacked as 30-second windows vertically. Time resolution per frame is configurable.

Example:
  python scripts/birdconv/plot_pt_spectrogram.py \
      --pt /path/to/file.pt \
      --out /path/to/out.png \
      --frame_ms 10 \
      --window_s 30
"""

import argparse
from pathlib import Path
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def load_spectrogram(pt_path: Path) -> np.ndarray:
    """Load spectrogram array 's' from a .pt file and return as numpy (F, T)."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    print(data.keys())

    if 's' not in data:
        raise KeyError(f"Key 's' not found in {pt_path}")

    if 'chirp_intervals' not in data:
        raise KeyError(f"Key 'chirp_intervals' not found in {pt_path}")

    return data['s'].detach().cpu().numpy(), data['chirp_intervals'].detach().cpu().numpy()


def compute_windows(num_frames: int, frames_per_window: int) -> list[tuple[int, int]]:
    """Return list of (start, end) frame indices covering [0, num_frames) in chunks."""
    if frames_per_window <= 0:
        raise ValueError("frames_per_window must be positive")
    windows = []
    start = 0
    while start < num_frames:
        end = min(start + frames_per_window, num_frames)
        windows.append((start, end))
        start = end
    return windows

def plot_stacked_windows(spec_db: np.ndarray,\
                         chirp_intervals: np.ndarray,
                         frame_ms: float,
                         window_s: float,
                         out_path: Path,
                         cmap: str = 'viridis',
                         dpi: int = 200,
                         tight: bool = True) -> None:
    """
    Plot the spectrogram in stacked 30s (configurable) windows and save to PNG.
    spec_db: (F, T) in dB
    frame_ms: duration per time frame in milliseconds
    window_s: seconds per panel
    tv_lambda: TV denoising parameter (0 = no denoising)
    """
    F, T = spec_db.shape
    frames_per_window = max(1, int(round((window_s * 1000.0) / frame_ms)))
    windows = compute_windows(T, frames_per_window)

    print("Shape of spec_db: ", spec_db.shape)
    print("Shape of chirp_intervals: ", chirp_intervals.shape)
    print(f"Found {len(chirp_intervals)} ")
    print(f"Total frames: {T}")

    loudness = compute_loudness(spec_db)

    # Consistent color scaling across panels
    vmin = float(np.nanmin(spec_db))
    vmax = float(np.nanmax(spec_db))

    n_rows = len(windows)
    # Figure sizing: width proportional to 30s, height per row
    base_height_per_row = 2.5  # inches per 30s window
    fig_height = max(2.5, n_rows * base_height_per_row)
    fig_width = 12

    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()

    for i, (start, end) in enumerate(windows):
        ax = axes[i]
        seg = spec_db[:, start:end]
        # Time axis in seconds for this window
        local_frames = end - start
        t0 = (start * frame_ms) / 1000.0
        t1 = (end * frame_ms) / 1000.0
        extent = [t0, t1, 0, F]

        im = ax.imshow(seg,
                       origin='lower',
                       aspect='auto',
                       cmap=cmap,
                       vmin=vmin,
                       vmax=vmax,
                       extent=extent)
        
        # Overlay loudness curve in yellow on second y-axis
        window_loudness = loudness[start:end]
        if len(window_loudness) > 0:
            # Create second y-axis for loudness
            ax2 = ax.twinx()
            
            # Normalize loudness to [0, 1] range for plotting
            # This ensures the line visually spans the full height of the plot
            global_min_loudness = np.min(loudness)
            global_max_loudness = np.max(loudness)
            
            if global_max_loudness - global_min_loudness < 1e-10:
                # Handle constant loudness case: plot a flat line at 0.5 normalized height
                loudness_to_plot = np.full_like(window_loudness, 0.5)
            else:
                loudness_to_plot = (window_loudness - global_min_loudness) / (global_max_loudness - global_min_loudness)
            
            # Time axis for loudness
            time_axis = np.linspace(t0, t1, len(window_loudness))
            
            # Plot normalized loudness on second y-axis
            ax2.plot(time_axis, loudness_to_plot, color='yellow', linewidth=2, alpha=0.8)
            
            # Set y-limits to [0, 1] so the plotted line spans the full height
            ax2.set_ylim(0, 1)
            
            # Set custom y-tick labels to show the actual loudness values
            num_ticks = 5
            tick_locations = np.linspace(0, 1, num_ticks)
            tick_labels = np.linspace(global_min_loudness, global_max_loudness, num_ticks)
            ax2.set_yticks(tick_locations)
            ax2.set_yticklabels([f'{l:.2f}' for l in tick_labels])
            
            ax2.set_ylabel('Loudness', color='yellow')
            ax2.tick_params(axis='y', labelcolor='yellow')
        
        # Add colored overlay windows using low/chirp intervals
        # - low_ints tinted blue
        # - chirp_intervals tinted red
        # Also mark interval boundaries with white vertical lines
        def _draw_intervals(intervals: list[tuple[int, int]], color: str) -> None:
            for (s, e) in intervals:
                s_clamped = max(s, start)
                e_clamped = min(e, end)
                if s_clamped < e_clamped:
                    t_left = (s_clamped * frame_ms) / 1000.0
                    t_right = (e_clamped * frame_ms) / 1000.0
                    ax.axvspan(t_left, t_right, alpha=0.35, color=color, zorder=0)
                    ax.axvline(x=t_left, color='white', linewidth=1.5, alpha=0.9, zorder=3)
                    ax.axvline(x=t_right, color='white', linewidth=1.5, alpha=0.9, zorder=3)

        _draw_intervals(chirp_intervals, 'red')
        
        ax.set_ylabel('Freq bin')
        # Only put x-label on the last subplot to reduce clutter
        if i == n_rows - 1:
            ax.set_xlabel('Time (s)')
        else:
            ax.set_xticklabels([])
        ax.set_title(f"{t0:0.1f}s – {t1:0.1f}s", loc='left', fontsize=10)

    # No colorbar/legend

    if tight:
        fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def compute_loudness(spec_db: np.ndarray) -> np.ndarray:
    spec_power = np.power(10.0, spec_db / 10.0, dtype=np.float64)
    loudness = np.sum(np.log1p(spec_power), axis=0, dtype=np.float64)
    return np.nan_to_num(loudness, nan=0.0, posinf=0.0, neginf=0.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot .pt spectrogram into stacked 30s windows")
    p.add_argument('--pt', required=True, type=str, help='Path to .pt spectrogram file')
    p.add_argument('--out', required=True, type=str, help='Output PNG path')
    p.add_argument('--frame_ms', type=float, default=10.0,
                   help='Frame duration in milliseconds (default: 10)')
    p.add_argument('--window_s', type=float, default=30.0,
                   help='Window size in seconds per panel (default: 30)')
    p.add_argument('--cmap', type=str, default='viridis', help='Matplotlib colormap')
    p.add_argument('--dpi', type=int, default=200, help='Output PNG DPI')
    args = p.parse_args()

    pt_path = Path(args.pt)
    out_path = Path(args.out)

    if not pt_path.exists():
        raise FileNotFoundError(f".pt file not found: {pt_path}")
    if args.frame_ms <= 0:
        raise ValueError("--frame_ms must be > 0")
    if args.window_s <= 0:
        raise ValueError("--window_s must be > 0")

    spec_db, chirp_intervals = load_spectrogram(pt_path)
    
    plot_stacked_windows(spec_db,
                         chirp_intervals=chirp_intervals,
                         frame_ms=float(args.frame_ms),
                         window_s=float(args.window_s),
                         out_path=out_path,
                         cmap=args.cmap,
                         dpi=int(args.dpi))

    print(f"Saved stacked spectrogram to: {out_path}")


if __name__ == '__main__':
    main()
