import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib import cm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

import extract_embedding  # noqa: E402

MAX_PLOT_TIMEBINS = 1000


def _build_palette(labels, colormap=cm.tab20):
    mask = labels >= 0
    if not mask.any():
        return {}
    uniq = np.unique(labels[mask])
    colors = colormap(np.linspace(0, 1, len(uniq)))
    palette = {}
    for uid, color in zip(uniq.tolist(), colors):
        rgb = np.asarray(color, dtype=np.float32)
        if rgb.shape[0] > 3:
            rgb = rgb[:3]
        palette[int(uid)] = rgb
    return palette


def _scatter(xy, labels, palette, path, title):
    plt.figure(figsize=(8, 8), dpi=300)
    mask = labels >= 0
    if (~mask).any():
        plt.scatter(xy[~mask, 0], xy[~mask, 1], s=10, color="#404040", alpha=0.1, edgecolors="none")
    for lab, color in palette.items():
        idx = labels == lab
        if idx.any():
            plt.scatter(xy[idx, 0], xy[idx, 1], s=10, color=color, alpha=0.15, edgecolors="none")
    plt.title(title, fontsize=28, fontweight="bold", loc="left")
    plt.xlabel("UMAP 1", fontsize=20, fontweight="bold")
    plt.ylabel("UMAP 2", fontsize=20, fontweight="bold")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()


def _fit_umap(embeddings, args):
    reducer_kwargs = {
        "n_components": 2,
        "n_neighbors": args.umap_neighbors,
        "metric": "cosine",
    }
    if args.deterministic:
        reducer_kwargs["random_state"] = 42
    else:
        reducer_kwargs["low_memory"] = True
        reducer_kwargs["n_jobs"] = -1
    reducer = umap.UMAP(**reducer_kwargs)
    return reducer.fit_transform(embeddings)


def _plot_spectrogram_segments(spectrograms, out_dir, max_segments):
    spectrograms = np.asarray(spectrograms, dtype=np.float32)
    if spectrograms.size == 0:
        return []
    total_bins = spectrograms.shape[0]
    segment_length = min(MAX_PLOT_TIMEBINS, total_bins)
    if segment_length <= 0:
        return []

    plotted = []
    start = 0
    segment_idx = 0
    while start < total_bins and len(plotted) < max_segments:
        end = min(start + segment_length, total_bins)
        segment = spectrograms[start:end]
        if segment.size == 0:
            break
        # Transpose for (mel, time) orientation
        image = segment.T
        fig = plt.figure(figsize=(10, 4), dpi=300)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(image, aspect="auto", origin="lower", interpolation="none")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Spectrogram segment {segment_idx:02d} | bins {start}-{end}")
        out_path = out_dir / f"spectrogram_{segment_idx:02d}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        plotted.append(
            {
                "segment_index": segment_idx,
                "path": str(out_path),
                "timebin_start": int(start),
                "timebin_end": int(end),
            }
        )
        segment_idx += 1
        start = end
    return plotted


def main():
    parser = argparse.ArgumentParser(description="Generate TinyBird embedding UMAPs and spectrograms.")
    parser.add_argument("--results_dir", required=True, help="Directory to store plots and metrics")
    parser.add_argument("--spec_dir", required=True, help="Spectrogram directory used for extraction")
    parser.add_argument("--run_dir", required=True, help="Training run directory containing checkpoint/config")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint filename within run_dir")
    parser.add_argument("--npz_path", default=None, help="Optional output npz path; defaults to results_dir/embeddings.npz")
    parser.add_argument("--num_timebins", type=int, default=12400, help="Maximum number of timebins to accumulate during extraction")
    parser.add_argument("--json_path", default=None, help="Event JSON path (optional)")
    parser.add_argument("--bird", default=None, help="Optional bird identifier to filter JSON")
    parser.add_argument("--umap_neighbors", type=int, default=50, help="Number of neighbors for UMAP")
    parser.add_argument("--max_spectrograms", type=int, default=5, help="Maximum number of event spectrograms to save")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic UMAP with random_state")
    args = parser.parse_args()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = Path(args.npz_path) if args.npz_path else (out_dir / "embeddings.npz")

    extract_args = {
        "num_timebins": args.num_timebins,
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "spec_dir": args.spec_dir,
        "npz_dir": str(npz_path),
        "json_path": args.json_path,
        "bird": args.bird,
    }
    extract_embedding.main(extract_args)

    npz = np.load(npz_path, allow_pickle=True)
    labels_down = npz["labels_downsampled"]
    base_palette = _build_palette(labels_down)
    encoded_variants = {
        "after_pos_removal": npz["encoded_embeddings_after_pos_removal"],
        "before_pos_removal": npz["encoded_embeddings_before_pos_removal"],
    }

    umap_dir = out_dir / "umap"
    umap_dir.mkdir(parents=True, exist_ok=True)
    spectrogram_dir = out_dir / "spectrograms"
    spectrogram_dir.mkdir(parents=True, exist_ok=True)

    umap_paths = {}
    for variant_name, embedding_array in encoded_variants.items():
        xy = _fit_umap(embedding_array, args)
        umap_path = umap_dir / f"encoded_{variant_name}.png"
        _scatter(
            xy,
            labels_down,
            base_palette,
            umap_path,
            title=f"encoded | {variant_name}",
        )
        umap_paths[variant_name] = str(umap_path)

    spectrogram_array = npz.get("spectrograms")
    if spectrogram_array is None:
        spectrograms = []
    else:
        spectrograms = _plot_spectrogram_segments(spectrogram_array, spectrogram_dir, args.max_spectrograms)

    metrics = {
        "npz_path": str(npz_path),
        "encoded_umap_plots": umap_paths,
        "spectrograms": spectrograms,
    }
    if spectrogram_array is not None:
        metrics["spectrogram_timebins"] = int(spectrogram_array.shape[0])
        metrics["spectrogram_mels"] = int(spectrogram_array.shape[1]) if spectrogram_array.ndim == 2 else None

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
