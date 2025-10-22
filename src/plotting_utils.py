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

SPEC_FIGSIZE: Tuple[float, float] = (10.0, 6.0)  # matches plot_embedding chunk visuals
SPEC_DPI = 300
SPEC_IMSHOW_KW = {"origin": "lower", "aspect": "auto", "interpolation": "none"}
SPEC_TITLE_KW = {"fontsize": 18, "fontweight": "bold"}
SPEC_TITLE_Y = 1.15
LOSS_DPI = 300

TRAIN_COLOR = "royalblue"
VAL_COLOR = "tomato"
MASK_CMAP = "viridis"

__all__ = ["save_reconstruction_plot", "plot_loss_curves"]


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
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = MASK_CMAP,
) -> str:
    """
    Generate and persist the reconstruction comparison plot for a batch.

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

    overlay_img = _depatchify(overlay_patches, mels=config["mels"], timebins=config["num_timebins"], patch_size=patch_size)

    x_img = x[0, 0].detach().cpu().numpy()
    overlay_img_np = overlay_img[0, 0].detach().cpu().numpy()
    mask_flat_np = bool_mask[0].detach().cpu().numpy().astype(bool)
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
