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
    "plot_benchmark_results",
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
    filename: str,
    mode: str,
    num_classes: int,
    output_dir: str,
    step_num: int,
    figsize: Optional[Tuple[int, int]] = None,
) -> str:
    """
    Generate and persist a supervised prediction visualization.
    
    Args:
        spectrogram: (H, W) spectrogram array
        labels: (W_patches,) ground truth labels
        predictions: (W_patches,) predicted labels
        probabilities: (W_patches, num_classes) class probabilities (optional, used for detect mode)
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
        fig, axes = plt.subplots(4, 1, figsize=figsize or (12, 8), 
                                gridspec_kw={'height_ratios': [3, 1, 0.5, 0.5]})
    else:
        fig, axes = plt.subplots(3, 1, figsize=figsize or (12, 7), 
                                gridspec_kw={'height_ratios': [3, 0.5, 0.5]})
    
    # Plot spectrogram
    axes[0].imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_ylabel('Mel Bins')
    axes[0].set_title(f'Spectrogram - {filename}')
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
    save_path = os.path.join(output_dir, f'prediction_step_{step_num:06d}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

def plot_benchmark_results(results_csv: str, output_dir: str):
    """
    Plot benchmark results.
    Expected CSV format: task,species,individual,samples,metric_value
    """
    import csv
    import matplotlib.ticker as ticker
    
    data = []
    try:
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['samples'] = int(row['samples'])
                row['metric_value'] = float(row['metric_value'])
                data.append(row)
    except FileNotFoundError:
        print(f"Results file not found: {results_csv}")
        return

    if not data:
        print("No data found in results CSV.")
        return
            
    os.makedirs(output_dir, exist_ok=True)
    
    # Helper to get color map
    species_list = sorted(list(set(d['species'] for d in data)))
    # Use tab10 or viridis if more than 10 species
    if len(species_list) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:len(species_list)]
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(species_list)))
        
    species_color = dict(zip(species_list, colors))
    
    # Apply style changes via context
    with plt.rc_context({
        'font.size': 24,
        'font.weight': 'bold',
        'axes.labelweight': 'bold', 
        'axes.titleweight': 'bold',
        'axes.titlesize': 24,
        'axes.labelsize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
    }):
        
        # Detection Plot
        detect_data = [d for d in data if d['task'] == 'detect']
        if detect_data:
            for species in species_list:
                sp_data = [d for d in detect_data if d['species'] == species]
                if not sp_data: continue
                
                plt.figure(figsize=(10, 6), dpi=SPEC_DPI)
                
                # Sort by samples
                sp_data.sort(key=lambda x: x['samples'])
                x = [d['samples'] for d in sp_data]
                y = [d['metric_value'] for d in sp_data]
                
                plt.plot(x, y, marker='o', label=species, color=species_color[species], linewidth=2)
                
                plt.xscale('log')
                plt.xlabel('Training Samples')
                plt.ylabel('F1 Score (%)')
                plt.title(f'Detection Performance: {species}')
                plt.legend()
                plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(5))
                plt.tick_params(axis='both', which='major', labelsize=18)
                plt.grid(True, alpha=0.2)
                plt.savefig(os.path.join(output_dir, f'detection_benchmark_{species}.png'), bbox_inches='tight')
                plt.close()
            
        # Classification Plot
        classify_data = [d for d in data if d['task'] == 'classify']
        if classify_data:
            # Get unique sample sizes from data and create position mapping
            all_samples = sorted(list(set(d['samples'] for d in classify_data)))
            sample_to_pos = {s: i for i, s in enumerate(all_samples)}
            
            for species in species_list:
                sp_data = [d for d in classify_data if d['species'] == species]
                if not sp_data: continue

                fig, ax = plt.subplots(figsize=(5.5, 6), dpi=SPEC_DPI)
                
                # Group by individual
                individuals = sorted(list(set(d['individual'] for d in sp_data)))
                color = species_color[species]
                
                for ind in individuals:
                    ind_data = [d for d in sp_data if d['individual'] == ind]
                    
                    ind_data.sort(key=lambda x: x['samples'])
                    x = [sample_to_pos[d['samples']] for d in ind_data]
                    y = [d['metric_value'] for d in ind_data]
                    
                    # Plot individual lines faintly
                    ax.plot(x, y, marker='o', color=color, alpha=0.3, linewidth=1)
                
                # Plot species averages
                sample_map = {}
                for d in sp_data:
                    s = d['samples']
                    if s not in sample_map: sample_map[s] = []
                    sample_map[s].append(d['metric_value'])
                
                samples = sorted(sample_map.keys())
                x_pos = [sample_to_pos[s] for s in samples]
                avgs = [np.mean(sample_map[s]) for s in samples]
                
                # No label (no legend)
                ax.plot(x_pos, avgs, marker='o', color=color, linewidth=3)
            
                # Set linear spacing with custom labels
                ax.set_xticks(range(len(all_samples)))
                ax.set_xticklabels([str(s) for s in all_samples])
                
                ax.set_xlabel('# Training Recordings', fontsize=24)
                ax.set_ylabel('Frame Error Rate (%)', fontsize=24)
                ax.set_title(species, fontsize=24)
                ax.set_ylim(0, 50)
                ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.tick_params(axis='both', which='major', labelsize=18)
                ax.grid(True, alpha=0.2)
                
                fig.savefig(os.path.join(output_dir, f'classification_benchmark_{species}.png'), bbox_inches='tight')
                plt.close(fig)


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
