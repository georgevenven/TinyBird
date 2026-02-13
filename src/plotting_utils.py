"""
Utilities for generating consistent visualizations.

These helpers centralize figure sizing, color choices, and plotting logic so
that the rest of the codebase only needs to provide the underlying data.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import umap
from matplotlib import cm

SPEC_FIGSIZE: Tuple[float, float] = (10.0, 6.0)  # matches plot_embedding chunk visuals
SPEC_DPI = 300
SPEC_IMSHOW_KW = {"origin": "lower", "aspect": "auto", "interpolation": "none"}
SPEC_TITLE_KW = {"fontsize": 18, "fontweight": "bold"}
SPEC_TITLE_Y = 1.15
LOSS_DPI = 300

TRAIN_COLOR = "royalblue"
VAL_COLOR = "tomato"
MASK_CMAP = "viridis"

__all__ = [
    "save_reconstruction_plot",
    "plot_loss_curves",
    "save_supervised_prediction_plot",
    "plot_theoretical_resolution_limit",
    "plot_species_f1_curves",
    "generate_umap_plots",
]


def _depatchify(patches: torch.Tensor, *, mels: int, timebins: int, patch_size: Sequence[int]) -> torch.Tensor:
    """Convert (B, T, P) patches back to spectrogram images."""
    fold = nn.Fold(output_size=(mels, timebins), kernel_size=patch_size, stride=patch_size)
    return fold(patches.transpose(1, 2))


def _denormalize_predictions(x_patches: torch.Tensor, pred_patches: torch.Tensor) -> torch.Tensor:
    """Undo per-patch normalization to match the target scale."""
    target_mean = x_patches.mean(dim=-1, keepdim=True)
    target_std = x_patches.std(dim=-1, keepdim=True)
    return pred_patches * (target_std + 1e-6) + target_mean


def _create_overlay(
    x_patches: torch.Tensor, pred_patches: torch.Tensor, bool_mask: torch.Tensor
) -> torch.Tensor:
    """Blend predictions into the masked regions of the original patches."""
    overlay = x_patches.clone()
    overlay[bool_mask] = pred_patches[bool_mask]
    return overlay


def _mask_flat_to_image(mask_flat: np.ndarray, *, patch_size: Sequence[int], spec_shape: Tuple[int, int]) -> np.ndarray:
    """Expand a flattened patch mask to the full spectrogram pixel grid."""
    patch_h, patch_w = patch_size
    spec_h, spec_w = spec_shape
    grid_h = spec_h // patch_h
    grid_w = spec_w // patch_w
    mask_grid = mask_flat.reshape(grid_h, grid_w)
    mask_img = np.repeat(np.repeat(mask_grid, patch_h, axis=0), patch_w, axis=1).astype(bool)
    return mask_img


def _imshow_spec(ax: plt.Axes, image: np.ndarray, *, spec_shape: Tuple[int, int], cmap=None) -> None:
    """Consistently display a spectrogram-sized array."""
    extent = (0, spec_shape[1], 0, spec_shape[0])
    ax.imshow(image, extent=extent, cmap=cmap, **SPEC_IMSHOW_KW)
    ax.set_xlim(0, spec_shape[1])
    ax.set_ylim(0, spec_shape[0])


def _style_spec_ax(ax: plt.Axes, title: str, *, cmap: Optional[str] = None) -> None:
    """Apply shared TinyBird spectrogram styling to the given axes."""
    if cmap is not None:
        ax.images[-1].set_cmap(cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.5,
        SPEC_TITLE_Y,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        **SPEC_TITLE_KW,
    )


def save_reconstruction_plot(
    model: torch.nn.Module,
    batch,
    *,
    config: dict,
    device: torch.device,
    use_amp: bool,
    output_dir: str,
    step_num: int,
    sample_idx: int = 0,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = MASK_CMAP,
) -> str:
    """
    Generate and persist the reconstruction comparison plot for a batch.
    
    Args:
        sample_idx: Index of the sample to visualize from the batch (default: 0)

    Returns:
        The file path of the saved image.
    """
    spectrograms, _ = batch
    x = spectrograms.to(device, non_blocking=True)

    model.eval()
    with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast():
                h, idx_restore, bool_mask, T = model.forward_encoder(x)
                pred = model.forward_decoder(h, idx_restore, T)
        else:
            h, idx_restore, bool_mask, T = model.forward_encoder(x)
            pred = model.forward_decoder(h, idx_restore, T)

    bool_mask = bool_mask.reshape(bool_mask.size(0), -1)
    patch_size = config["patch_size"]
    spec_shape = (config["mels"], config["num_timebins"])

    unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
    x_patches = unfold(x).transpose(1, 2)

    # Disable per-patch denormalization so we visualise raw decoder outputs.
    # pred_denorm = _denormalize_predictions(x_patches, pred)
    pred_denorm = pred.to(dtype=x_patches.dtype)
    overlay_patches = _create_overlay(x_patches, pred_denorm, bool_mask)

    # Renormalize if normalize_patches is active (or default)
    if config.get("normalize_patches", True):
        overlay_mean = overlay_patches.mean(dim=-1, keepdim=True)
        overlay_std = overlay_patches.std(dim=-1, keepdim=True)
        overlay_patches = (overlay_patches - overlay_mean) / (overlay_std + 1e-6)

    overlay_img = _depatchify(overlay_patches, mels=config["mels"], timebins=config["num_timebins"], patch_size=patch_size)

    x_img = x[sample_idx, 0].detach().cpu().numpy()
    overlay_img_np = overlay_img[sample_idx, 0].detach().cpu().numpy()
    mask_flat_np = bool_mask[sample_idx].detach().cpu().numpy().astype(bool)
    mask_img_np = _mask_flat_to_image(mask_flat_np, patch_size=patch_size, spec_shape=spec_shape)

    masked_display = x_img.copy()
    masked_display[mask_img_np] = np.nan

    if isinstance(cmap, str):
        mask_cmap = plt.get_cmap(cmap, 256)
    else:
        mask_cmap = cmap
    if hasattr(mask_cmap, "with_extremes"):
        mask_cmap = mask_cmap.with_extremes(bad="black")
    else:
        mask_cmap = mask_cmap.copy() if hasattr(mask_cmap, "copy") else mask_cmap
        mask_cmap.set_bad("black")

    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=figsize or SPEC_FIGSIZE, dpi=SPEC_DPI)
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 3, 3], hspace=0.7)

    ax1 = fig.add_subplot(gs[0, 0])
    _imshow_spec(ax1, x_img, spec_shape=spec_shape)
    _style_spec_ax(ax1, "Input Spectrogram")

    ax2 = fig.add_subplot(gs[1, 0])
    _imshow_spec(ax2, masked_display, spec_shape=spec_shape, cmap=mask_cmap)
    _style_spec_ax(ax2, "Original with Mask (black = masked patches)", cmap=mask_cmap)

    ax3 = fig.add_subplot(gs[2, 0])
    _imshow_spec(ax3, overlay_img_np, spec_shape=spec_shape)
    _style_spec_ax(ax3, "Overlay: Unmasked Original + Masked Predictions")

    recon_path = os.path.join(output_dir, f"recon_step_{step_num:06d}.png")
    fig.savefig(recon_path, dpi=SPEC_DPI, bbox_inches="tight")
    plt.close(fig)

    return recon_path


def plot_loss_curves(
    *,
    train_steps: Iterable[int],
    train_losses: Iterable[float],
    val_steps: Iterable[int],
    val_losses: Iterable[float],
    loss_log_path: str,
    output_path: str,
    figsize: Tuple[int, int] = (12.0, 6.0),
    yscale: Optional[str] = "log",
) -> str:
    """
    Plot and persist the training/validation loss curves.

    Returns:
        The file path of the saved loss plot.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(train_steps, train_losses, label="Training Loss", alpha=0.7, linewidth=1, color=TRAIN_COLOR)
    ax.plot(val_steps, val_losses, label="Validation Loss", marker="o", markersize=3, linewidth=2, color=VAL_COLOR)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.grid(True, alpha=0.3)
    if yscale:
        ax.set_yscale(yscale)

    if ax.get_lines():
        ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=LOSS_DPI, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_supervised_prediction_plot(
    *,
    spectrogram: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray],
    logits: Optional[np.ndarray],
    filename: str,
    mode: str,
    num_classes: int,
    output_dir: str,
    step_num: int,
    figsize: Optional[Tuple[int, int]] = None,
    logits_top_k: int = 3,
    split: str = "val",
) -> str:
    """
    Generate and persist a supervised prediction visualization.
    
    Args:
        spectrogram: (H, W) spectrogram array
        labels: (W_patches,) ground truth labels
        predictions: (W_patches,) predicted labels
        probabilities: (W_patches, num_classes) class probabilities (optional, used for detect mode)
        logits: (W_patches, num_classes) or (W_patches,) raw logits (optional)
        logits_top_k: number of class logits to plot as lines (multi-class only)
        split: "train" or "val" label for the plot filename/title
        filename: Name of the audio file
        mode: "detect", "unit_detect", or "classify"
        num_classes: Total number of classes
        output_dir: Directory to save the plot
        step_num: Current training step
        figsize: Optional figure size override
        
    Returns:
        The file path of the saved image.
    """
    # Create custom colormap where class 0 is black
    base_cmap = plt.get_cmap('tab20', num_classes)
    colors = base_cmap(np.linspace(0, 1, num_classes))
    colors[0] = [0, 0, 0, 1]  # Set class 0 to black
    custom_cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    # Create figure
    if mode in ["detect", "unit_detect"]:
        fig, axes = plt.subplots(
            4,
            1,
            figsize=figsize or (12, 8),
            gridspec_kw={'height_ratios': [3, 1, 0.7, 0.5]},
        )
    else:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=figsize or (12, 7),
            gridspec_kw={'height_ratios': [3, 0.7, 0.5]},
        )
    
    # Plot spectrogram
    axes[0].imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_ylabel('Mel Bins')
    axes[0].set_title(f'Spectrogram ({split}) - {filename}')
    axes[0].set_xticks([])
    
    # Plot probability line for detect mode
    if mode in ["detect", "unit_detect"] and probabilities is not None:
        vocal_prob = probabilities[:, 1]  # Probability of vocalization (class 1)
        x = np.arange(len(vocal_prob))
        axes[1].plot(x, vocal_prob, 'r-', linewidth=2)
        axes[1].set_ylabel('Vocal Prob')
        axes[1].set_ylim([0, 1])
        axes[1].set_xlim([0, len(vocal_prob)])
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks([])
        
        pred_ax_idx = 2
        gt_ax_idx = 3
    else:
        pred_ax_idx = 1
        gt_ax_idx = 2
    
    # Plot predicted classes as colored bar
    pred_img = predictions.reshape(1, -1)
    axes[pred_ax_idx].imshow(pred_img, aspect='auto', cmap=custom_cmap, 
                             vmin=0, vmax=num_classes-1)
    axes[pred_ax_idx].set_ylabel('Predicted')
    axes[pred_ax_idx].set_yticks([])
    axes[pred_ax_idx].set_xticks([])

    _ = logits, logits_top_k
    
    # Plot ground truth classes as colored bar
    gt_img = labels.reshape(1, -1)
    axes[gt_ax_idx].imshow(gt_img, aspect='auto', cmap=custom_cmap, 
                           vmin=0, vmax=num_classes-1)
    axes[gt_ax_idx].set_ylabel('Ground Truth')
    axes[gt_ax_idx].set_yticks([])
    axes[gt_ax_idx].set_xlabel('Time Patches')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'prediction_{split}_step_{step_num:06d}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

def plot_theoretical_resolution_limit(results_csv: str, output_dir: str) -> None:
    import csv
    import matplotlib.ticker as ticker

    data = []
    try:
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['samples'] = float(row['samples'])
                row['metric_value'] = float(row['metric_value'])
                data.append(row)
    except FileNotFoundError:
        print(f"Results file not found: {results_csv}")
        return

    if not data:
        print("No data found in results CSV.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Filter to only classification results
    classify_data = [d for d in data if d.get('task') == 'classify']
    if not classify_data:
        print("No classification data found.")
        return

    # Group by individual and species
    individuals = sorted(set(d['individual'] for d in classify_data))
    species_list = sorted(set(d['species'] for d in classify_data))
    colors = plt.cm.tab10(np.linspace(0, 1, len(species_list)))
    species_color = dict(zip(species_list, colors))

    # One plot per individual
    for individual in individuals:
        fig, ax = plt.subplots()

        for species in species_list:
            sp_data = [d for d in classify_data if d['species'] == species and d['individual'] == individual]
            if not sp_data:
                continue
            sp_data = sorted(sp_data, key=lambda x: x['samples'])

            x = [d['samples'] for d in sp_data]
            y = [d['metric_value'] for d in sp_data]

            # Compute theoretical resolution in ms (128 hop at 32kHz: 4ms per timebin)
            # We assume samples = seconds of training data. This is a heuristic; adjust if needed.
            ax.plot(x, y, marker='o', linewidth=2, label=species, color=species_color[species])

        ax.set_xlabel("Training Data (seconds)")
        ax.set_ylabel("FER (%)")
        ax.set_title(f"{individual} - Classification FER")
        ax.grid(True, alpha=0.2)
        ax.legend()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        fig.savefig(os.path.join(output_dir, f"theoretical_resolution_limit_{individual}.png"), bbox_inches='tight')
        plt.close(fig)


def plot_species_f1_curves(
    results_dir: str,
    output_dir: Optional[str] = None,
    *,
    mode: str = "classify",
    probe_mode: Optional[str] = None,
    metric: str = "both",
    species_filter: Optional[Sequence[str]] = None,
) -> list[str]:
    import csv
    import matplotlib.ticker as ticker

    if metric not in {"f1", "fer", "both"}:
        raise ValueError("metric must be 'f1', 'fer', or 'both'")

    results_csv = results_dir
    if os.path.isdir(results_dir):
        results_csv = os.path.join(results_dir, "eval_f1.csv")
    if not os.path.isfile(results_csv):
        print(f"Results CSV not found: {results_csv}")
        return []

    data = []
    with open(results_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if mode and row.get("mode") != mode:
                continue
            if probe_mode and row.get("probe_mode") != probe_mode:
                continue
            species = (row.get("species") or "").strip()
            bird = (row.get("bird") or "").strip()
            train_seconds = row.get("train_seconds")
            lr_value = row.get("lr")
            if not species or not train_seconds:
                continue
            train_seconds_token = str(train_seconds).strip()
            if not train_seconds_token:
                continue
            if train_seconds_token.upper() == "MAX":
                train_seconds_label = "MAX"
                train_seconds_sort_key = (1, float("inf"))
            else:
                try:
                    train_seconds_numeric = float(train_seconds_token)
                except (TypeError, ValueError):
                    continue
                train_seconds_label = f"{train_seconds_numeric:g}"
                train_seconds_sort_key = (0, train_seconds_numeric)

            lr_label = None
            lr_sort_key = None
            if lr_value not in (None, ""):
                try:
                    lr_numeric = float(lr_value)
                except (TypeError, ValueError):
                    lr_numeric = None
                if lr_numeric is not None:
                    lr_label = f"{lr_numeric:g}"
                    lr_sort_key = lr_numeric

            f1_value = row.get("f1")
            fer_value = row.get("fer")
            try:
                f1_metric = float(f1_value) if f1_value not in (None, "") else None
            except (TypeError, ValueError):
                f1_metric = None
            try:
                fer_metric = float(fer_value) if fer_value not in (None, "") else None
            except (TypeError, ValueError):
                fer_metric = None

            if metric == "f1" and f1_metric is None:
                continue
            if metric == "fer" and fer_metric is None:
                continue
            if metric == "both" and f1_metric is None and fer_metric is None:
                continue

            data.append(
                {
                    "species": species,
                    "bird": bird or "unknown",
                    "train_seconds": train_seconds_label,
                    "train_seconds_sort_key": train_seconds_sort_key,
                    "lr": lr_label,
                    "lr_sort_key": lr_sort_key,
                    "f1": f1_metric,
                    "fer": fer_metric,
                }
            )

    if not data:
        print("No matching rows found in results CSV.")
        return []

    if output_dir is None:
        if os.path.isdir(results_dir):
            output_dir = os.path.join(results_dir, "plots")
        else:
            output_dir = os.path.join(os.path.dirname(results_csv), "plots")
    os.makedirs(output_dir, exist_ok=True)

    species_set = {row["species"] for row in data}
    if species_filter:
        species_filter_set = {s.strip() for s in species_filter if s.strip()}
        species_set = {s for s in species_set if s in species_filter_set}
    if not species_set:
        print("No species matched the filter.")
        return []

    preferred_species_order = ["Bengalese_Finch", "Zebra_Finch", "Canary"]
    ordered_species = [sp for sp in preferred_species_order if sp in species_set]
    ordered_species.extend(sorted(species_set - set(ordered_species)))

    species_display = {
        "Bengalese_Finch": "Bengalese Finch",
        "Zebra_Finch": "Zebra Finch",
        "Canary": "Canary",
    }

    def _slugify(text: str) -> str:
        import re

        text = text.strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return text.strip("_") or "species"

    def _save_plot_with_svg(fig, png_path: str) -> str:
        fig.savefig(png_path, bbox_inches="tight")
        svg_path = os.path.splitext(png_path)[0] + ".svg"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        return svg_path

    metrics = ["f1", "fer"] if metric == "both" else [metric]
    metric_labels = {"f1": "F1 Score (%)", "fer": "Frame Error Rate (%)"}
    metric_titles = {"f1": "F1", "fer": "FER"}
    # Paper-friendly portrait ratio so two figures can sit side-by-side.
    # Keep width stable and reduce height slightly for tighter page fit.
    single_plot_figsize = (4.8, 6.8)

    def _build_species_metric_series(rows: list[dict], metric_name: str):
        birds = sorted({row["bird"] for row in rows})
        train_levels = {row["train_seconds"] for row in rows}
        lr_levels = {row["lr"] for row in rows if row["lr"] is not None}
        if len(train_levels) > 1:
            x_field = "train_seconds"
            x_label = "# Training Seconds"
        elif len(lr_levels) > 1:
            x_field = "lr"
            x_label = "Learning Rate"
        else:
            x_field = "train_seconds"
            x_label = "# Training Seconds"

        x_sort_keys: dict[str, tuple[int, float]] = {}
        for row in rows:
            x_value = row.get(x_field)
            if x_value is None:
                continue
            if x_field == "train_seconds":
                x_sort_keys[x_value] = row["train_seconds_sort_key"]
            else:
                x_sort_keys[x_value] = (0, row["lr_sort_key"])
        x_levels = sorted(x_sort_keys.keys(), key=lambda ts: x_sort_keys[ts])
        x_index = {ts: i for i, ts in enumerate(x_levels)}

        bird_series: dict[str, dict[str, list[float]]] = {}
        for row in rows:
            metric_value = row[metric_name]
            x_value = row.get(x_field)
            if metric_value is None or x_value is None:
                continue
            bird_series.setdefault(row["bird"], {}).setdefault(x_value, []).append(metric_value)

        if not bird_series:
            return None

        avg_by_ts: dict[str, list[float]] = {}
        for by_ts in bird_series.values():
            for ts, vals in by_ts.items():
                avg_by_ts.setdefault(ts, []).append(float(np.mean(vals)))
        avg_xs = sorted(avg_by_ts.keys(), key=lambda ts: x_sort_keys[ts])
        avg_ys = [float(np.mean(avg_by_ts[x])) for x in avg_xs]
        avg_positions = [x_index[x] for x in avg_xs]

        return {
            "birds": birds,
            "x_label": x_label,
            "x_levels": x_levels,
            "x_sort_keys": x_sort_keys,
            "x_index": x_index,
            "bird_series": bird_series,
            "avg_ys": avg_ys,
            "avg_positions": avg_positions,
        }

    def _draw_species_metric_ax(ax, *, species_idx: int, species: str, metric_name: str, series):
        base_color = plt.cm.tab10((species_idx % 10) / 10)
        for bird in series["birds"]:
            by_ts = series["bird_series"].get(bird, {})
            if not by_ts:
                continue
            xs = sorted(by_ts.keys(), key=lambda ts: series["x_sort_keys"][ts])
            ys = [float(np.mean(by_ts[x])) for x in xs]
            x_positions = [series["x_index"][x] for x in xs]
            ax.plot(
                x_positions,
                ys,
                marker="o",
                markersize=4.5,
                linewidth=1.0,
                alpha=0.35,
                color=base_color,
            )

        ax.plot(
            series["avg_positions"],
            series["avg_ys"],
            marker="o",
            markersize=5.8,
            linewidth=2.5,
            color=base_color,
            alpha=0.95,
        )

        species_title = species_display.get(species, species.replace("_", " "))
        ax.set_title(f"{species_title} - {metric_titles[metric_name]}", fontsize=20, fontweight="bold")
        ax.set_xlabel(series["x_label"], fontsize=16, fontweight="bold")
        ax.set_ylabel(metric_labels[metric_name], fontsize=16, fontweight="bold")
        ax.grid(True, alpha=0.22)
        ax.set_xlim(-0.25, max(0.25, len(series["x_levels"]) - 0.75))
        ax.set_xticks(list(range(len(series["x_levels"]))))
        ax.set_xticklabels(series["x_levels"])
        ax.xaxis.set_major_locator(ticker.FixedLocator(list(range(len(series["x_levels"]))))
        )
        ax.tick_params(axis="both", labelsize=16, width=1.2)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight("bold")
        if metric_name == "f1":
            ax.set_ylim(40.0, 100.0)
        else:
            max_y = max(series["avg_ys"]) if series["avg_ys"] else 0.0
            upper = max(10.0, np.ceil((max_y + 1.0) / 5.0) * 5.0)
            ax.set_ylim(0.0, upper)

        # Keep borders consistent across panels.
        for side in ("top", "bottom", "left", "right"):
            spine = ax.spines[side]
            if spine.get_visible():
                spine.set_linewidth(1.0)
                spine.set_color("#404040")

    saved_paths: list[str] = []
    tag = "all" if not probe_mode else probe_mode

    for metric_name in metrics:
        for species_idx, species in enumerate(ordered_species):
            rows = [row for row in data if row["species"] == species]
            if not rows:
                continue

            series = _build_species_metric_series(rows, metric_name)
            if series is None:
                continue

            fig, ax = plt.subplots(figsize=single_plot_figsize, dpi=SPEC_DPI)
            _draw_species_metric_ax(
                ax,
                species_idx=species_idx,
                species=species,
                metric_name=metric_name,
                series=series,
            )

            mode_tag = mode or "all_modes"
            filename = f"eval_{metric_name}_{mode_tag}_{tag}_{_slugify(species)}.png"
            save_path = os.path.join(output_dir, filename)
            fig.tight_layout()
            svg_path = _save_plot_with_svg(fig, save_path)
            plt.close(fig)
            saved_paths.append(save_path)
            saved_paths.append(svg_path)

    # Additional joined F1 figure with all species in one image.
    if "f1" in metrics and len(ordered_species) > 1:
        # Keep joined subplot height consistent with individual plots.
        joined_figsize = (single_plot_figsize[0] * len(ordered_species), single_plot_figsize[1])
        fig, axes = plt.subplots(
            1,
            len(ordered_species),
            figsize=joined_figsize,
            dpi=SPEC_DPI,
            squeeze=False,
            sharey=True,
        )
        axes_row = axes[0]
        any_drawn = False
        for species_idx, species in enumerate(ordered_species):
            ax = axes_row[species_idx]
            rows = [row for row in data if row["species"] == species]
            if not rows:
                ax.set_visible(False)
                continue
            series = _build_species_metric_series(rows, "f1")
            if series is None:
                ax.set_visible(False)
                continue
            _draw_species_metric_ax(
                ax,
                species_idx=species_idx,
                species=species,
                metric_name="f1",
                series=series,
            )
            if species_idx > 0:
                # Joined layout: keep only the left y-axis to reduce clutter.
                ax.set_ylabel("")
                ax.tick_params(axis="y", left=False, labelleft=False)
            any_drawn = True

        if any_drawn:
            mode_tag = mode or "all_modes"
            joined_path = os.path.join(output_dir, f"eval_f1_{mode_tag}_{tag}_species_joined.png")
            fig.subplots_adjust(left=0.055, right=0.995, bottom=0.14, top=0.90, wspace=0.10)
            joined_svg = _save_plot_with_svg(fig, joined_path)
            saved_paths.append(joined_path)
            saved_paths.append(joined_svg)
        plt.close(fig)

    return saved_paths


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


def _scatter_umap(xy, labels, palette, path, title):
    plt.figure(figsize=(5.5, 5.5), dpi=300)
    mask = labels >= 0
    if (~mask).any():
        plt.scatter(xy[~mask, 0], xy[~mask, 1], s=10, color="#404040", alpha=0.1, edgecolors="none")
    for lab, color in palette.items():
        idx = labels == lab
        if idx.any():
            plt.scatter(xy[idx, 0], xy[idx, 1], s=10, color=color, alpha=0.15, edgecolors="none")
    plt.title(title, fontsize=24, fontweight="bold", loc="center")
    plt.xlabel("UMAP 1", fontsize=20, fontweight="bold")
    plt.ylabel("UMAP 2", fontsize=20, fontweight="bold")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=300, format="png")
    plt.close()


def _fit_umap_embedding(embeddings, n_neighbors, deterministic):
    reducer_kwargs = {
        "n_components": 2,
        "n_neighbors": n_neighbors,
        "metric": "cosine",
        "min_dist": 0.01,
    }
    if deterministic:
        reducer_kwargs["random_state"] = 42
    else:
        reducer_kwargs["low_memory"] = True
        reducer_kwargs["n_jobs"] = -1
    reducer = umap.UMAP(**reducer_kwargs)
    return reducer.fit_transform(embeddings)


def generate_umap_plots(npz_path: str, output_dir: str, umap_neighbors: int = 200, deterministic: bool = False):
    """
    Generate UMAP plots from an embedding NPZ file.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    npz = np.load(npz_path, allow_pickle=True)
    labels_down = npz["labels_downsampled"]
    
    base_palette = _build_palette(labels_down)
    
    # Check available keys
    keys = ["encoded_embeddings_after_pos_removal", "encoded_embeddings_before_pos_removal"]
    encoded_variants = {}
    for k in keys:
        if k in npz:
            encoded_variants[k.replace("encoded_embeddings_", "")] = npz[k]
    
    umap_dir = os.path.join(output_dir, "umap")
    os.makedirs(umap_dir, exist_ok=True)
    
    paths = {}
    
    for variant_name, embedding_array in encoded_variants.items():
        xy = _fit_umap_embedding(embedding_array, umap_neighbors, deterministic)
        umap_path = os.path.join(umap_dir, f"encoded_{variant_name}.png")
        
        title_text = "Pos. Removal" if "after" in variant_name else "No Pos. Removal"
        
        _scatter_umap(
            xy,
            labels_down,
            base_palette,
            umap_path,
            title=title_text,
        )
        paths[variant_name] = umap_path
        print(f"Generated UMAP plot: {umap_path}")
        
    return paths
