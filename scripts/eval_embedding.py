import argparse
import json
import sys
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib import cm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, v_measure_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import extract_embedding  # noqa: E402
from data_loader import SpectogramDataset  # noqa: E402
from extract_embedding import create_label_arr, load_json_events  # noqa: E402

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


def _gather_event_summaries(npz, args, cluster_time):
    if not args.json_path:
        return []
    audio_params = (
        _to_item(npz["audio_sr"]),
        _to_item(npz["audio_n_mels"]),
        _to_item(npz["audio_hop_size"]),
        _to_item(npz["audio_fft"]),
    )
    patch_width = _to_item(npz["patch_width"])
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
            labels_full = create_label_arr(event, rounded)
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
            collected.append(
                {
                    "spec": event_spec.squeeze(0),
                    "file": fname,
                    "event_index": ev_idx,
                    "gt": gt_slice,
                    "pred": cluster_slice,
                }
            )
            if len(collected) >= args.max_spectrograms:
                return collected
    return collected


def _plot_spectrogram(event, out_dir, gt_palette, pred_palette):
    spec = event["spec"]
    gt = event["gt"]
    pred = event["pred"]
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
    ax_pred.set_ylabel("HDB", rotation=0, labelpad=10, ha="right", va="center")

    out_path = out_dir / f"{event['file']}_event{event['event_index']:02d}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


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
    parser.add_argument("--min_cluster_size", type=int, default=1000)
    parser.add_argument("--min_cluster_samples", type=int, default=10)
    parser.add_argument("--cluster_selection_epsilon", type=float, default=0.0)
    parser.add_argument("--max_spectrograms", type=int, default=5)
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
    embeddings = npz["embeddings_after_pos_removal"]
    labels_down = npz["labels_downsampled"]
    labels_original = npz["labels_original"]
    patch_width = _to_item(npz["patch_width"])

    metrics = {"npz_path": str(npz_path)}
    base_palette = _build_palette(labels_down)

    labeled_mask = labels_down >= 0
    if labeled_mask.sum() >= 2 and np.unique(labels_down[labeled_mask]).size > 1:
        n_comp = min(2, np.unique(labels_down[labeled_mask]).size - 1)
        lda = LinearDiscriminantAnalysis(n_components=n_comp)
        proj = lda.fit_transform(embeddings[labeled_mask], labels_down[labeled_mask])
        pad_proj = np.zeros((embeddings.shape[0], 2), dtype=np.float32)
        pad_proj[labeled_mask, :n_comp] = proj
        pad_proj[~labeled_mask] = np.nan
        lda_path = out_dir / "lda.png"
        _scatter(pad_proj, labels_down, lda_path, title="LDA", palette=base_palette)
        metrics["lda_classes"] = int(np.unique(labels_down[labeled_mask]).size)
    else:
        metrics["lda_classes"] = 0

    if labeled_mask.sum() > 10 and np.unique(labels_down[labeled_mask]).size > 1:
        x_train, x_test, y_train, y_test = train_test_split(
            embeddings[labeled_mask],
            labels_down[labeled_mask],
            test_size=0.2,
            random_state=42,
            stratify=labels_down[labeled_mask],
        )
        clf = LogisticRegression(max_iter=2000, multi_class="auto")
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        metrics["linear_probe_accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["linear_probe_report"] = classification_report(y_test, y_pred, output_dict=True)
    else:
        metrics["linear_probe_accuracy"] = None

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
    umap_path = out_dir / "umap.png"
    gt_palette = _scatter(umap_xy, labels_down, umap_path, title="Ground Truth", palette=base_palette)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_cluster_samples,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
    )
    cluster_labels = clusterer.fit_predict(umap_xy)
    cluster_path = out_dir / "umap_hdbscan.png"
    cluster_palette = _scatter(
        umap_xy,
        cluster_labels,
        cluster_path,
        title="HDBSCAN",
        colormap=cm.nipy_spectral,
    )

    mask_v = (labels_down >= 0) & (cluster_labels >= 0)
    if mask_v.any():
        metrics["v_measure"] = float(v_measure_score(labels_down[mask_v], cluster_labels[mask_v]))
    else:
        metrics["v_measure"] = None

    cluster_time = np.repeat(cluster_labels, patch_width)
    if len(cluster_time) < len(labels_original):
        cluster_time = np.pad(cluster_time, (0, len(labels_original) - len(cluster_time)), constant_values=-1)
    else:
        cluster_time = cluster_time[: len(labels_original)]

    events = _gather_event_summaries(npz, args, cluster_time)
    rendered = []
    for event in events:
        rendered.append(str(_plot_spectrogram(event, out_dir, gt_palette, cluster_palette)))
    metrics["spectrograms"] = rendered

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(_to_python(metrics), f, indent=2)


if __name__ == "__main__":
    main()
