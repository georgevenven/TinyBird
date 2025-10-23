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


def _plot_metric_overlay(event, values, x_overlay, spec_arr, out_path, color, series_label, y_label, peaks=None):
    values = np.asarray(values, dtype=np.float32)
    x_overlay = np.asarray(x_overlay, dtype=np.float32)
    if values.size == 0 or x_overlay.size == 0:
        return None
    if values.shape[0] != x_overlay.shape[0]:
        return None

    if peaks is None:
        peaks_idx = np.array([], dtype=int)
    else:
        peaks_idx = np.asarray(peaks, dtype=int)

    if spec_arr is not None:
        fig, ax_spec = plt.subplots(figsize=(10, 6), dpi=300)
        ax_spec.imshow(spec_arr, aspect="auto", origin="lower", interpolation="none")
        ax_spec.set_xticks([])
        ax_spec.set_yticks([])
        ax_spec.set_xlabel("Timebins")
        ax_spec.set_ylabel("Mel Bins")
        ax_spec.set_title(f"{event['file']} | event {event['event_index']}")
        metric_ax = ax_spec.twinx()
    else:
        fig, metric_ax = plt.subplots(figsize=(10, 4), dpi=300)
        metric_ax.set_xlabel("Patch Index")
        metric_ax.set_title(f"{event['file']} | event {event['event_index']}")

    ymin = float(np.min(values))
    ymax = float(np.max(values))
    if np.isclose(ymin, ymax):
        margin = 0.5 if np.isclose(ymax, 0.0) else abs(ymax) * 0.1
        ymin -= margin
        ymax += margin
    metric_ax.set_ylim(ymin, ymax)
    if x_overlay.size > 1:
        metric_ax.set_xlim(float(x_overlay[0]), float(x_overlay[-1]))

    metric_ax.plot(x_overlay, values, color=color, linewidth=2.5, label=series_label)
    if peaks_idx.size:
        metric_ax.scatter(
            x_overlay[peaks_idx],
            values[peaks_idx],
            color=color,
            s=30,
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            label="Peaks",
        )
    metric_ax.set_ylabel(y_label)
    if series_label or peaks_idx.size:
        metric_ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_adjacent_metrics(event, embeddings, similarity_dir, difference_dir, patch_width, prominence=0.1):
    embed_start, embed_end = event.get("embedding_slice", (None, None))
    if embed_start is None or embed_end is None or embed_end - embed_start < 2:
        return None
    segment = embeddings[int(embed_start) : int(embed_end)]
    if segment.shape[0] < 2:
        return None

    time_length = event.get("time_length")
    if time_length is not None and time_length > MAX_PLOT_TIMEBINS and patch_width:
        crop_bins = MAX_PLOT_TIMEBINS
        max_embeddings = max(2, int(np.ceil(crop_bins / patch_width)) + 1)
        segment = segment[:max_embeddings]
        if segment.shape[0] < 2:
            return None

    norms = np.linalg.norm(segment, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = segment / norms
    cos_sim = np.sum(normalized[:-1] * normalized[1:], axis=1)
    similarities = -cos_sim
    differences = np.linalg.norm(segment[1:] - segment[:-1], axis=1)

    spec_arr = event.get("spec")
    prepared_spec = None
    metric_x = np.array([], dtype=np.float32)
    if spec_arr is not None:
        spec_arr = np.asarray(spec_arr)
        if spec_arr.ndim == 3 and spec_arr.shape[0] == 1:
            spec_arr = spec_arr[0]
        timebins = int(spec_arr.shape[-1])
        if timebins > MAX_PLOT_TIMEBINS:
            timebins = MAX_PLOT_TIMEBINS
        prepared_spec = spec_arr[..., :timebins]
        if similarities.size:
            if similarities.shape[0] > 1:
                metric_x = np.linspace(0, timebins, num=similarities.shape[0], endpoint=False, dtype=np.float32)
            else:
                metric_x = np.zeros_like(similarities, dtype=np.float32)

    if not metric_x.size and similarities.size:
        if patch_width:
            metric_x = np.arange(similarities.shape[0], dtype=np.float32) * float(patch_width)
        else:
            metric_x = np.arange(similarities.shape[0], dtype=np.float32)

    peak_indices = np.array([], dtype=int)
    peak_timebins = []
    if similarities.size:
        peak_kwargs = {}
        if prominence is not None and prominence > 0:
            peak_kwargs["prominence"] = prominence
        peak_indices, _ = find_peaks(similarities, **peak_kwargs)
        if peak_indices.size and metric_x.size == similarities.size:
            peak_timebins = metric_x[peak_indices].astype(int).tolist()

    diff_peak_indices = np.array([], dtype=int)
    diff_peak_timebins = []
    if differences.size:
        diff_kwargs = {}
        if prominence is not None and prominence > 0:
            diff_kwargs["prominence"] = prominence
        diff_peak_indices, _ = find_peaks(differences, **diff_kwargs)
        if diff_peak_indices.size and metric_x.size == differences.size:
            diff_peak_timebins = metric_x[diff_peak_indices].astype(int).tolist()

    similarity_path = None
    difference_path = None
    if similarities.size and metric_x.size:
        similarity_out = similarity_dir / f"{event['file']}_event{event['event_index']:02d}_neg_cosine.png"
        similarity_path = _plot_metric_overlay(
            event,
            similarities,
            metric_x,
            prepared_spec,
            similarity_out,
            color="tab:red",
            series_label="Neg Cosine",
            y_label="Negative Cosine Similarity",
            peaks=peak_indices,
        )
        difference_out = difference_dir / f"{event['file']}_event{event['event_index']:02d}_difference.png"
        difference_path = _plot_metric_overlay(
            event,
            differences,
            metric_x,
            prepared_spec,
            difference_out,
            color="tab:red",
            series_label="L2 Difference",
            y_label="Adjacent Vector L2 Difference",
            peaks=diff_peak_indices,
        )

    return {
        "similarity_overlay_path": similarity_path,
        "difference_overlay_path": difference_path,
        "peak_timebins": peak_timebins,
        "difference_peak_timebins": diff_peak_timebins,
    }


def _extract_boundaries(labels):
    arr = np.asarray(labels)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size < 2:
        return np.array([], dtype=int)
    changes = np.flatnonzero(arr[1:] != arr[:-1]) + 1
    return changes.astype(int)


def _match_boundaries(predicted, ground_truth, tolerance):
    if tolerance < 0:
        tolerance = 0
    pred = np.sort(np.asarray(predicted, dtype=int))
    gt = np.sort(np.asarray(ground_truth, dtype=int))
    if pred.size == 0:
        return 0, 0, int(gt.size)
    if gt.size == 0:
        return 0, int(pred.size), 0

    matched = np.zeros(gt.shape[0], dtype=bool)
    true_positives = 0
    for value in pred:
        left = np.searchsorted(gt, value - tolerance, side="left")
        right = np.searchsorted(gt, value + tolerance, side="right")
        if left >= right:
            continue
        best_idx = -1
        best_dist = None
        for idx in range(int(left), int(right)):
            if matched[idx]:
                continue
            dist = abs(int(gt[idx]) - int(value))
            if dist <= tolerance:
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx
        if best_idx >= 0:
            matched[best_idx] = True
            true_positives += 1

    false_positives = int(pred.size) - true_positives
    false_negatives = int(gt.size - matched.sum())
    return int(true_positives), int(false_positives), int(false_negatives)


def _init_segmentation_stats():
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "total_predicted": 0,
        "total_ground_truth": 0,
    }


def _accumulate_segmentation(stats, predicted, ground_truth, tolerance):
    predicted = np.asarray(predicted, dtype=int)
    ground_truth = np.asarray(ground_truth, dtype=int)
    stats["total_predicted"] += int(predicted.size)
    stats["total_ground_truth"] += int(ground_truth.size)
    if predicted.size == 0 and ground_truth.size == 0:
        return
    tp, fp, fn = _match_boundaries(predicted, ground_truth, tolerance)
    stats["tp"] += int(tp)
    stats["fp"] += int(fp)
    stats["fn"] += int(fn)


def _finalize_segmentation(stats):
    tp = float(stats["tp"])
    fp = float(stats["fp"])
    fn = float(stats["fn"])
    total_pred = float(stats["total_predicted"])
    total_gt = float(stats["total_ground_truth"])

    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    if precision > 0.0:
        over_segmentation = recall / precision - 1.0
    elif recall == 0.0:
        over_segmentation = 0.0
    else:
        over_segmentation = float("inf")

    if precision > 0.0 or recall > 0.0:
        if np.isfinite(over_segmentation):
            r1 = float(np.sqrt((1.0 - recall) ** 2 + over_segmentation ** 2))
            r2 = float((-over_segmentation + recall - 1.0) / np.sqrt(2.0))
            r_value = 1.0 - (abs(r1) + abs(r2)) / 2.0
        else:
            r_value = 0.0
    else:
        r_value = 0.0

    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "r_value": float(r_value),
        "over_segmentation": float(over_segmentation) if np.isfinite(over_segmentation) else None,
        "tp": int(stats["tp"]),
        "fp": int(stats["fp"]),
        "fn": int(stats["fn"]),
        "total_predicted": int(stats["total_predicted"]),
        "total_ground_truth": int(stats["total_ground_truth"]),
    }


def _majority_label(labels):
    arr = np.asarray(labels)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    arr = arr[arr >= 0]
    if arr.size == 0:
        return -1
    values, counts = np.unique(arr, return_counts=True)
    if counts.size == 0:
        return -1
    return int(values[np.argmax(counts)])


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
    umap_dir = tag_dir / "umap"
    spectrogram_dir = tag_dir / "spectrograms"
    similarity_dir = tag_dir / "similarity"
    difference_dir = tag_dir / "difference"
    segment_umap_dir = tag_dir / "umap_segments"
    for path in (umap_dir, spectrogram_dir, similarity_dir, difference_dir, segment_umap_dir):
        path.mkdir(parents=True, exist_ok=True)

    audio_sr = None
    audio_hop = None
    if "audio_sr" in npz:
        audio_sr = _to_item(npz["audio_sr"])
    if "audio_hop_size" in npz:
        audio_hop = _to_item(npz["audio_hop_size"])
    tolerance_ms = 20.0
    tolerance_bins = 1
    if audio_sr and audio_hop:
        hop_seconds = float(audio_hop) / float(audio_sr)
        if hop_seconds > 0.0:
            tolerance_bins = max(1, int(round((tolerance_ms / 1000.0) / hop_seconds)))

    segmentation_stats = {
        "similarity": _init_segmentation_stats(),
        "difference": _init_segmentation_stats(),
    }
    segment_records = []

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
    umap_path = umap_dir / "ground_truth.png"
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

    cluster_path = umap_dir / "kmeans_matched.png"
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
    difference_plots = []
    boundary_predictions = []
    for event in events:
        rendered_path = _plot_spectrogram(event, spectrogram_dir, gt_palette, cluster_palette)
        rendered.append(str(rendered_path))
        similarity_info = _plot_adjacent_metrics(
            event,
            embeddings,
            similarity_dir,
            difference_dir,
            patch_width,
            prominence=args.peak_prominence,
        ) or {}

        overlay_path = similarity_info.get("similarity_overlay_path")
        if overlay_path:
            similarity_plots.append(str(overlay_path))
        difference_path = similarity_info.get("difference_overlay_path")
        if difference_path:
            difference_plots.append(str(difference_path))

        similarity_peaks = [int(p) for p in (similarity_info.get("peak_timebins") or [])]
        difference_peaks = [int(p) for p in (similarity_info.get("difference_peak_timebins") or [])]
        gt_boundaries = _extract_boundaries(event.get("gt", []))
        _accumulate_segmentation(segmentation_stats["similarity"], similarity_peaks, gt_boundaries, tolerance_bins)
        _accumulate_segmentation(segmentation_stats["difference"], difference_peaks, gt_boundaries, tolerance_bins)

        if similarity_peaks or difference_peaks:
            boundary_predictions.append(
                {
                    "file": event["file"],
                    "event_index": event["event_index"],
                    "timebins": similarity_peaks,
                    "similarity_timebins": similarity_peaks,
                    "difference_timebins": difference_peaks,
                }
            )

        embed_start, embed_end = event.get("embedding_slice", (None, None))
        if embed_start is None or embed_end is None:
            continue
        embed_start = int(embed_start)
        embed_end = int(embed_end)
        if embed_end - embed_start < 1:
            continue
        segment_array = embeddings[embed_start:embed_end]
        if segment_array.size == 0:
            continue

        seg_len = segment_array.shape[0]
        if patch_width and patch_width > 0:
            peak_bins = np.asarray(difference_peaks or similarity_peaks, dtype=np.float32)
            peak_patches = np.rint(peak_bins / float(patch_width)).astype(int) if peak_bins.size else np.array([], dtype=int)
        else:
            peak_patches = np.asarray(difference_peaks or similarity_peaks, dtype=int)
        if peak_patches.size:
            peak_patches = np.clip(peak_patches, 0, seg_len)
        boundaries = [0]
        if peak_patches.size:
            boundaries.extend(sorted(np.unique(peak_patches.tolist())))
        if boundaries[-1] != seg_len:
            boundaries.append(seg_len)

        gt = np.asarray(event.get("gt", []))
        segment_counter = 0
        for idx in range(len(boundaries) - 1):
            seg_start = int(boundaries[idx])
            seg_end = int(boundaries[idx + 1])
            if seg_end <= seg_start:
                continue
            seg_vectors = segment_array[seg_start:seg_end]
            if seg_vectors.size == 0:
                continue
            if patch_width and patch_width > 0:
                time_start = seg_start * patch_width
                time_end = seg_end * patch_width
            else:
                time_start = seg_start
                time_end = seg_end
            time_start = max(0, min(time_start, gt.shape[0]))
            time_end = max(0, min(time_end, gt.shape[0]))
            if time_end > time_start:
                seg_label = _majority_label(gt[time_start:time_end])
            else:
                seg_label = -1
            segment_records.append(
                {
                    "vectors": seg_vectors.copy(),
                    "label": seg_label,
                    "file": event["file"],
                    "event_index": int(event["event_index"]),
                    "segment_index": segment_counter,
                    "start_patch": seg_start,
                    "end_patch": seg_end,
                    "start_timebin": time_start,
                    "end_timebin": time_end,
                }
            )
            segment_counter += 1

    analysis = {
        "umap_path": str(umap_path),
        "umap_kmeans_path": str(cluster_path),
        "kmeans": kmeans_info,
        "spectrograms": rendered,
        "cosine_similarity_plots": similarity_plots,
        "difference_plots": difference_plots,
        "predicted_boundaries": boundary_predictions,
        "events_processed": int(len(events)),
    }
    analysis["segmentation_metrics"] = {
        "tolerance_ms": float(tolerance_ms),
        "tolerance_timebins": int(tolerance_bins),
        "sample_rate": int(audio_sr) if audio_sr is not None else None,
        "hop_size": int(audio_hop) if audio_hop is not None else None,
        "results": {name: _finalize_segmentation(stat) for name, stat in segmentation_stats.items()},
    }

    if segment_records:
        embed_dim = embeddings.shape[1]
        max_segments = max(rec["vectors"].shape[0] for rec in segment_records)
        segment_count = len(segment_records)
        segment_tensor = np.zeros((segment_count, max_segments, embed_dim), dtype=np.float32)
        segment_lengths = []
        segment_labels = np.full(segment_count, -1, dtype=int)
        segment_meta = []
        for idx, rec in enumerate(segment_records):
            data = rec["vectors"]
            length = data.shape[0]
            segment_tensor[idx, :length, :] = data
            segment_lengths.append(int(length))
            segment_labels[idx] = int(rec["label"])
            metadata = {k: v for k, v in rec.items() if k != "vectors"}
            metadata["length_patches"] = int(length)
            segment_meta.append(metadata)
        segment_flat = segment_tensor.reshape(segment_count, max_segments * embed_dim)
        segment_umap_path = None
        if segment_count >= 2:
            if args.deterministic:
                segment_reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=args.umap_neighbors,
                    metric="cosine",
                    random_state=42,
                )
            else:
                segment_reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=args.umap_neighbors,
                    metric="cosine",
                    low_memory=True,
                    n_jobs=-1,
                )
            try:
                segment_xy = segment_reducer.fit_transform(segment_flat)
                segment_umap_path = segment_umap_dir / "segments.png"
                _scatter(segment_xy, segment_labels, segment_umap_path, title="Segments (between peaks)", palette=base_palette)
            except Exception as exc:  # noqa: F841
                segment_umap_path = None
        analysis["segment_umap"] = {
            "umap_path": str(segment_umap_path) if segment_umap_path else None,
            "segment_count": int(segment_count),
            "max_segment_patches": int(max_segments),
            "segment_lengths": segment_lengths,
            "labels": segment_labels.tolist(),
            "metadata": segment_meta,
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
        "encoded": {
            "after_pos_removal": npz["encoded_embeddings_after_pos_removal"],
            "before_pos_removal": npz["encoded_embeddings_before_pos_removal"],
        },
        "patch": {
            "after_pos_removal": npz["patch_embeddings_after_pos_removal"],
            "before_pos_removal": npz["patch_embeddings_before_pos_removal"],
        },
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
        "embedding_counts": {},
    }

    for embedding_type, variants in embedding_sets.items():
        metrics["analyses"][embedding_type] = {}
        metrics["embedding_counts"][embedding_type] = {}
        for variant_name, embedding_array in variants.items():
            metrics["embedding_counts"][embedding_type][variant_name] = int(embedding_array.shape[0])
            analysis = _analyze_embedding(
                f"{embedding_type}/{variant_name}",
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
            metrics["analyses"][embedding_type][variant_name] = analysis

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(_to_python(metrics), f, indent=2)


if __name__ == "__main__":
    main()
