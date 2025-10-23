import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib import cm
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

import extract_embedding  # noqa: E402
from data_loader import SpectogramDataset  # noqa: E402
from extract_embedding import load_json_events  # noqa: E402

MAX_PLOT_TIMEBINS = 1000  # cap for rendered spectrogram width

def _to_python(obj):
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_python(v) for v in obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def _to_item(arr):
    return int(arr.item()) if hasattr(arr, "item") else int(arr)

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


def _scatter(xy, labels, path, title=None, palette=None, colormap=cm.tab20, unlabeled_color="#404040"):
    plt.figure(figsize=(8, 8), dpi=300)
    mask = labels >= 0
    if palette is None:
        palette = _build_palette(labels, colormap=colormap)
    if (~mask).any():
        plt.scatter(xy[~mask, 0], xy[~mask, 1], s=10, color=unlabeled_color, alpha=0.1, edgecolors="none")
    for lab, color in palette.items():
        idx = labels == lab
        if idx.any():
            plt.scatter(xy[idx, 0], xy[idx, 1], s=10, color=color, alpha=0.1, edgecolors="none")
    plot_title = title if title is not None else Path(path).stem.replace("_", " ")
    plt.title(plot_title, fontsize=32, fontweight="bold", loc="left")
    plt.xlabel("UMAP 1", fontsize=24, fontweight="bold")
    plt.ylabel("UMAP 2", fontsize=24, fontweight="bold")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    return palette


def _label_bar(labels, palette, default_rgb):
    t = labels.shape[0]
    bar = np.zeros((t, 3), dtype=np.float32)
    for i, lab in enumerate(labels.tolist()):
        color = palette.get(lab, default_rgb)
        color = np.asarray(color, dtype=np.float32)
        if color.shape[0] > 3:
            color = color[:3]
        bar[i] = color
    bar = np.tile(bar[None, :, :], (8, 1, 1))
    return bar


def _calc_pad(length, context):
    if length > context:
        remainder = length % context
        pad = context if remainder == 0 else context - remainder
    elif length < context:
        pad = context - length
    else:
        pad = 0
    return pad


def _gather_event_summaries(npz, args, cluster_time, patch_width, num_embeddings):
    if not args.json_path:
        return []
    audio_params = (
        _to_item(npz["audio_sr"]),
        _to_item(npz["audio_n_mels"]),
        _to_item(npz["audio_hop_size"]),
        _to_item(npz["audio_fft"]),
    )
    context = _to_item(npz["model_num_timebins"])
    labels_original = npz["labels_original"]

    ds = SpectogramDataset(args.spec_dir, n_timebins=None)
    event_map = load_json_events(args.json_path, audio_params=audio_params, selected_bird=args.bird)

    ptr = 0
    collected = []
    for idx in range(len(ds)):
        if ptr >= len(labels_original):
            break
        spec, fname = ds[idx]
        spec = spec.numpy()
        timebins = spec.shape[-1]
        remainder = timebins % patch_width
        if remainder:
            spec = spec[:, :, : timebins - remainder]
        rounded = spec.shape[-1]
        matched = event_map.get(fname, [])
        if not matched:
            continue
        for ev_idx, event in enumerate(matched, start=1):
            time_start = ptr
            event_spec = spec[:, :, event["on_timebins"] : event["off_timebins"]]
            if event_spec.shape[-1] == 0:
                continue
            raw_len = event_spec.shape[-1]
            pad = _calc_pad(raw_len, context)
            padded_len = raw_len + pad
            if ptr + padded_len > len(labels_original) or ptr + padded_len > len(cluster_time):
                return collected
            gt_slice = labels_original[ptr : ptr + padded_len][:raw_len]
            cluster_slice = cluster_time[ptr : ptr + padded_len][:raw_len]
            ptr += padded_len
            embed_start = time_start // patch_width
            embed_end = min(num_embeddings, (time_start + raw_len + patch_width - 1) // patch_width)
            collected.append(
                {
                    "spec": event_spec.squeeze(0),
                    "file": fname,
                    "event_index": ev_idx,
                    "gt": gt_slice,
                    "pred": cluster_slice,
                    "embedding_slice": (int(embed_start), int(embed_end)),
                    "time_length": raw_len,
                }
            )
            if len(collected) >= args.max_spectrograms:
                return collected
    return collected


def _plot_spectrogram(event, out_dir, gt_palette, pred_palette):
    spec = event["spec"]
    gt = event["gt"]
    pred = event["pred"]
    max_time = min(spec.shape[-1], MAX_PLOT_TIMEBINS)
    spec = spec[..., :max_time]
    gt = gt[:max_time]
    pred = pred[:max_time]
    default_rgb = np.array([0.25, 0.25, 0.25], dtype=np.float32)
    gt_bar = _label_bar(gt, gt_palette, default_rgb)
    pred_bar = _label_bar(pred, pred_palette, default_rgb)

    fig = plt.figure(figsize=(10, 6), dpi=300)
    gs = fig.add_gridspec(3, 1, height_ratios=[5, 0.6, 0.6], hspace=0.05)
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_gt = fig.add_subplot(gs[1, 0])
    ax_pred = fig.add_subplot(gs[2, 0])

    ax_spec.imshow(spec, aspect="auto", origin="lower", interpolation="none")
    ax_spec.set_xticks([])
    ax_spec.set_yticks([])
    ax_spec.set_title(f"{event['file']} | event {event['event_index']}")

    ax_gt.imshow(gt_bar, aspect="auto", origin="lower", interpolation="nearest")
    ax_gt.set_xticks([])
    ax_gt.set_yticks([])
    ax_gt.set_ylabel("GT", rotation=0, labelpad=20, ha="right", va="center")

    ax_pred.imshow(pred_bar, aspect="auto", origin="lower", interpolation="nearest")
    ax_pred.set_xticks([])
    ax_pred.set_yticks([])
    ax_pred.set_ylabel("KM", rotation=0, labelpad=10, ha="right", va="center")

    out_path = out_dir / f"{event['file']}_event{event['event_index']:02d}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_cosine_similarity(event, embeddings, out_dir, patch_width, prominence=0.1):
    embed_start, embed_end = event.get("embedding_slice", (None, None))
    if embed_start is None or embed_end is None or embed_end - embed_start < 2:
        return None
    segment = embeddings[embed_start:embed_end]
    if segment.shape[0] < 2:
        return None

    time_length = event.get("time_length")
    if time_length is not None and time_length > MAX_PLOT_TIMEBINS and patch_width:
        crop_bins = MAX_PLOT_TIMEBINS
        max_embeddings = max(2, int(np.ceil(crop_bins / patch_width)) + 1)
        segment = segment[:max_embeddings]

    norms = np.linalg.norm(segment, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = segment / norms
    cos_sim = np.sum(normalized[:-1] * normalized[1:], axis=1)
    similarities = -cos_sim
    peak_kwargs = {}
    if prominence is not None and prominence > 0:
        peak_kwargs["prominence"] = prominence
    peak_indices, _ = find_peaks(similarities, **peak_kwargs) if similarities.size else (np.array([], dtype=int), {})

    peak_timebins = []
    overlay_path = None
    x_overlay = None
    spec = event.get("spec")
    if spec is not None:
        spec_arr = np.asarray(spec)
        if spec_arr.ndim == 3 and spec_arr.shape[0] == 1:
            spec_arr = spec_arr[0]
        timebins = int(spec_arr.shape[-1])
        if timebins > MAX_PLOT_TIMEBINS:
            timebins = MAX_PLOT_TIMEBINS
        spec_arr = spec_arr[..., :timebins]
        if similarities.shape[0] > 1:
            x_overlay = np.linspace(0, timebins, num=similarities.shape[0], endpoint=False)
        else:
            x_overlay = np.zeros_like(similarities)

        fig_overlay, ax_spec = plt.subplots(figsize=(10, 6), dpi=300)
        ax_spec.imshow(spec_arr, aspect="auto", origin="lower", interpolation="none")
        ax_spec.set_xticks([])
        ax_spec.set_yticks([])
        ax_spec.set_xlabel("Timebins")
        ax_spec.set_ylabel("Mel Bins")
        ax_spec.set_title(f"{event['file']} | event {event['event_index']}")

        ax_cos = ax_spec.twinx()
        if similarities.size:
            ymin = float(np.min(similarities))
            ymax = float(np.max(similarities))
        else:
            ymin, ymax = 0.0, 1.0
        if np.isclose(ymin, ymax):
            margin = 0.5 if ymax == 0 else abs(ymax) * 0.1
            ymin -= margin
            ymax += margin
        ax_cos.set_ylim(ymin, ymax)
        if x_overlay.size > 1:
            ax_cos.set_xlim(x_overlay[0], x_overlay[-1])
        ax_cos.plot(x_overlay, similarities, color="tab:red", linewidth=2.5, label="Neg Cosine")
        if peak_indices.size:
            peaks_x = x_overlay[peak_indices]
            ax_cos.scatter(peaks_x, similarities[peak_indices], color="tab:red", s=30, marker="o", edgecolors="black", linewidths=0.5, label="Peaks")
            peak_timebins = peaks_x.astype(int).tolist()
        ax_cos.set_ylabel("Negative Cosine Similarity")
        ax_cos.legend(loc="upper right")

        overlay_path = out_dir / f"{event['file']}_event{event['event_index']:02d}_similarity_overlay.png"
        fig_overlay.tight_layout()
        fig_overlay.savefig(overlay_path, bbox_inches="tight")
        plt.close(fig_overlay)

    if x_overlay is None and similarities.size:
        if patch_width:
            x_overlay = np.arange(similarities.shape[0], dtype=np.float32) * float(patch_width)
        else:
            x_overlay = np.arange(similarities.shape[0], dtype=np.float32)
    if not peak_timebins and x_overlay is not None and peak_indices.size:
        peak_timebins = np.asarray(x_overlay)[peak_indices].astype(int).tolist()

    return {
        "overlay_path": overlay_path,
        "peak_timebins": peak_timebins,
    }


def _analyze_embedding(
    tag,
    embeddings,
    out_dir,
    labels_down,
    labels_original,
    patch_width,
    base_palette,
    unique_gt,
    labeled_mask,
    k,
    npz,
    args,
):
    tag_dir = out_dir / tag
    tag_dir.mkdir(parents=True, exist_ok=True)

    if args.deterministic:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=args.umap_neighbors,
            metric="cosine",
            random_state=42,
        )
    else:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=args.umap_neighbors,
            metric="cosine",
            low_memory=True,
            n_jobs=-1,
        )

    umap_xy = reducer.fit_transform(embeddings)
    umap_path = tag_dir / "umap.png"
    gt_palette = _scatter(umap_xy, labels_down, umap_path, title="Ground Truth", palette=base_palette)

    matched_labels = np.full(embeddings.shape[0], -1, dtype=int)
    kmeans_info = {
        "inertia": None,
        "matched_clusters": 0,
        "downsample_error_rate": None,
        "error_rate": None,
        "error_rate_percent": None,
    }
    if k > 0:
        kmeans_kwargs = {"n_clusters": k, "n_init": 10}
        if args.deterministic:
            kmeans_kwargs["random_state"] = 42
        kmeans = KMeans(**kmeans_kwargs)
        cluster_labels = kmeans.fit_predict(umap_xy)
        kmeans_info["inertia"] = float(kmeans.inertia_)
        unique_pred = np.unique(cluster_labels)
        counts = np.zeros((k, unique_pred.size), dtype=np.int64)
        pred_to_col = {label: idx for idx, label in enumerate(unique_pred)}
        for gt_idx, gt_label in enumerate(unique_gt):
            mask = labeled_mask & (labels_down == gt_label)
            preds = cluster_labels[mask]
            if preds.size == 0:
                continue
            for pred in preds:
                counts[gt_idx, pred_to_col[pred]] += 1
        mapping = {}
        if counts.size:
            row_ind, col_ind = linear_sum_assignment(counts.max() - counts)
            for gt_idx, col_idx in zip(row_ind, col_ind):
                if counts[gt_idx, col_idx] == 0:
                    continue
                mapping[unique_pred[col_idx]] = unique_gt[gt_idx]
        for cluster_label, gt_label in mapping.items():
            matched_labels[cluster_labels == cluster_label] = gt_label
        kmeans_info["matched_clusters"] = int(len(mapping))
        if labeled_mask.any() and len(mapping) > 0:
            labeled_preds = matched_labels[labeled_mask]
            mismatches = np.sum(labeled_preds != labels_down[labeled_mask])
            kmeans_info["downsample_error_rate"] = float(mismatches / labeled_mask.sum())

    cluster_path = tag_dir / "umap_kmeans.png"
    cluster_palette = _scatter(
        umap_xy,
        matched_labels,
        cluster_path,
        title="KMeans (matched)",
        palette=base_palette,
    )

    cluster_time = np.repeat(matched_labels, patch_width)
    if len(cluster_time) < len(labels_original):
        cluster_time = np.pad(cluster_time, (0, len(labels_original) - len(cluster_time)), constant_values=-1)
    else:
        cluster_time = cluster_time[: len(labels_original)]

    valid_bins = labels_original >= 0
    if valid_bins.any():
        mismatches = np.sum(cluster_time[valid_bins] != labels_original[valid_bins])
        total = int(valid_bins.sum())
        error_rate = mismatches / total
        kmeans_info["error_rate"] = float(error_rate)
        kmeans_info["error_rate_percent"] = float(error_rate * 100.0)

    events = _gather_event_summaries(npz, args, cluster_time, patch_width, embeddings.shape[0])
    rendered = []
    similarity_plots = []
    boundary_predictions = []
    for event in events:
        rendered.append(str(_plot_spectrogram(event, tag_dir, gt_palette, cluster_palette)))
        similarity_info = _plot_cosine_similarity(
            event,
            embeddings,
            tag_dir,
            patch_width,
            prominence=args.peak_prominence,
        )
        if similarity_info:
            overlay_path = similarity_info.get("overlay_path")
            if overlay_path:
                similarity_plots.append(str(overlay_path))
            peaks = similarity_info.get("peak_timebins") or []
            if peaks:
                boundary_predictions.append(
                    {
                        "file": event["file"],
                        "event_index": event["event_index"],
                        "timebins": peaks,
                    }
                )

    analysis = {
        "umap_path": str(umap_path),
        "umap_kmeans_path": str(cluster_path),
        "kmeans": kmeans_info,
        "spectrograms": rendered,
        "cosine_similarity_plots": similarity_plots,
        "predicted_boundaries": boundary_predictions,
        "events_processed": int(len(events)),
    }
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Run extraction + evaluation for TinyBird embeddings.")
    parser.add_argument("--results_dir", required=True, help="Directory to store plots and metrics")
    parser.add_argument("--spec_dir", required=True, help="Spectrogram directory used for extraction")
    parser.add_argument("--run_dir", required=True, help="Training run directory containing checkpoint/config")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint filename within run_dir")
    parser.add_argument("--npz_path", default=None, help="Optional output npz path; defaults to results_dir/embeddings.npz")
    parser.add_argument("--num_timebins", type=int, default=12400, help="Maximum number of timebins to accumulate during extraction")
    parser.add_argument("--json_path", default=None, help="Event JSON path (optional)")
    parser.add_argument("--bird", default=None, help="Optional bird identifier to filter JSON")
    parser.add_argument("--umap_neighbors", type=int, default=100)
    parser.add_argument("--max_spectrograms", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic UMAP with random_state")
    parser.add_argument(
        "--peak_prominence",
        type=float,
        default=0.1,
        help="Prominence threshold for boundary peak picking on dissimilarity score",
    )
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
        "peak_prominence": args.peak_prominence,
    }
    extract_embedding.main(extract_args)

    npz = np.load(npz_path, allow_pickle=True)
    labels_down = npz["labels_downsampled"]
    labels_original = npz["labels_original"]
    patch_width = _to_item(npz["patch_width"])

    embedding_sets = {
        "encoded": npz["encoded_embeddings_after_pos_removal"],
        "patch": npz["patch_embeddings_after_pos_removal"],
    }

    labeled_mask = labels_down >= 0
    unique_gt = np.unique(labels_down[labeled_mask]) if labeled_mask.any() else np.array([], dtype=labels_down.dtype)
    k = unique_gt.size

    base_palette = _build_palette(labels_down)
    metrics = {
        "npz_path": str(npz_path),
        "ground_truth_syllables": int(k),
        "peak_prominence": float(args.peak_prominence),
        "analyses": {},
        "embedding_counts": {name: int(arr.shape[0]) for name, arr in embedding_sets.items()},
    }

    for tag, embedding_array in embedding_sets.items():
        analysis = _analyze_embedding(
            tag,
            embedding_array,
            out_dir,
            labels_down,
            labels_original,
            patch_width,
            base_palette,
            unique_gt,
            labeled_mask,
            k,
            npz,
            args,
        )
        metrics["analyses"][tag] = analysis

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(_to_python(metrics), f, indent=2)


if __name__ == "__main__":
    main()
