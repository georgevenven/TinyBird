"""
Utilities for generating consistent visualizations.

These helpers centralize figure sizing, color choices, and plotting logic so
that the rest of the codebase only needs to provide the underlying data.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

DEFAULT_RECON_FIGSIZE: Tuple[float, float] = (12.0, 4.5)  # width, height in inches
DEFAULT_LOSS_FIGSIZE: Tuple[float, float] = (12.0, 10.0)
LOSS_DPI = 300
RECON_DPI = 150

TRAIN_COLOR = "royalblue"
VAL_COLOR = "tomato"
EMA_TRAIN_COLOR = "navy"
EMA_VAL_COLOR = "maroon"
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


def _create_masked_original(x_patches: torch.Tensor, bool_mask: torch.Tensor) -> torch.Tensor:
    """Zero out masked patches to visualize which regions were hidden."""
    masked = x_patches.clone()
    min_val = masked.min()
    masked[bool_mask] = min_val - 1.0
    return masked


def save_reconstruction_plot(
    model: torch.nn.Module,
    batch,
    *,
    config: dict,
    device: torch.device,
    use_amp: bool,
    output_dir: str,
    step_num: int,
    figsize: Tuple[int, int] = DEFAULT_RECON_FIGSIZE,
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

    unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
    x_patches = unfold(x).transpose(1, 2)

    pred_denorm = _denormalize_predictions(x_patches, pred)
    overlay_patches = _create_overlay(x_patches, pred_denorm, bool_mask)

    overlay_img = _depatchify(overlay_patches, mels=config["mels"], timebins=config["num_timebins"], patch_size=patch_size)
    masked_img = _depatchify(
        _create_masked_original(x_patches, bool_mask), mels=config["mels"], timebins=config["num_timebins"], patch_size=patch_size
    )

    x_img = x[0, 0].detach().cpu().numpy()
    masked_img_np = masked_img[0, 0].detach().cpu().numpy()
    overlay_img_np = overlay_img[0, 0].detach().cpu().numpy()

    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=figsize)

    ax1 = plt.subplot(3, 1, 1)
    ax1.imshow(x_img, origin="lower", aspect="auto")
    ax1.set_title("Input Spectrogram")
    ax1.axis("off")

    ax2 = plt.subplot(3, 1, 2)
    ax2.imshow(masked_img_np, origin="lower", aspect="auto", cmap=cmap)
    ax2.set_title("Original with Mask (black = masked patches)")
    ax2.axis("off")

    ax3 = plt.subplot(3, 1, 3)
    ax3.imshow(overlay_img_np, origin="lower", aspect="auto")
    ax3.set_title("Overlay: Unmasked Original + Masked Predictions")
    ax3.axis("off")

    fig.tight_layout()
    recon_path = os.path.join(output_dir, f"recon_step_{step_num:06d}.png")
    fig.savefig(recon_path, dpi=RECON_DPI)
    plt.close(fig)

    return recon_path


def _read_ema_histories(loss_log_path: str) -> Tuple[Sequence[int], Sequence[float], Sequence[float]]:
    eval_steps = []
    ema_train = []
    ema_val = []
    if not os.path.exists(loss_log_path):
        return eval_steps, ema_train, ema_val

    with open(loss_log_path, "r") as f:
        lines = f.readlines()[1:]  # skip header

    for line in lines:
        parts = line.strip().split(",")
        if len(parts) >= 5:
            eval_steps.append(int(parts[0]))
            ema_train.append(float(parts[2]))
            ema_val.append(float(parts[4]))

    return eval_steps, ema_train, ema_val


def plot_loss_curves(
    *,
    train_steps: Iterable[int],
    train_losses: Iterable[float],
    val_steps: Iterable[int],
    val_losses: Iterable[float],
    loss_log_path: str,
    output_path: str,
    figsize: Tuple[int, int] = DEFAULT_LOSS_FIGSIZE,
    yscale: Optional[str] = "log",
) -> str:
    """
    Plot and persist the training/validation loss curves and EMA histories.

    Returns:
        The file path of the saved loss plot.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    ax1.plot(train_steps, train_losses, label="Training Loss", alpha=0.7, linewidth=1, color=TRAIN_COLOR)
    ax1.plot(val_steps, val_losses, label="Validation Loss", marker="o", markersize=3, linewidth=2, color=VAL_COLOR)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.grid(True, alpha=0.3)
    if yscale:
        ax1.set_yscale(yscale)

    eval_steps, ema_train, ema_val = [], [], []
    try:
        eval_steps, ema_train, ema_val = _read_ema_histories(loss_log_path)
    except Exception as exc:  # pylint: disable=broad-except
        ax2.text(0.5, 0.5, f"Error reading EMA data: {exc}", transform=ax2.transAxes, ha="center", va="center")
    else:
        if eval_steps and ema_train and ema_val:
            ax2.plot(eval_steps, ema_train, label="EMA Training Loss", linewidth=2, color=EMA_TRAIN_COLOR)
            ax2.plot(eval_steps, ema_val, label="EMA Validation Loss", marker="o", markersize=3, linewidth=2, color=EMA_VAL_COLOR)
        else:
            ax2.text(0.5, 0.5, "No EMA data available for plotting", transform=ax2.transAxes, ha="center", va="center")

    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("EMA Loss")
    ax2.set_title("Exponential Moving Average Loss")
    ax2.grid(True, alpha=0.3)
    if yscale:
        ax2.set_yscale(yscale)

    if ax1.get_lines():
        ax1.legend()
    if ax2.get_lines():
        ax2.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=LOSS_DPI, bbox_inches="tight")
    plt.close(fig)

    return output_path
