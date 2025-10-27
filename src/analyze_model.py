#!/usr/bin/env python3
"""
Analyze TinyBird model behavior on spectrogram data.

This script loads a trained TinyBird model and normalized spectrogram data for analysis.
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import load_model_from_checkpoint
from data_loader import SpectogramDataset
import pickle

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path_for(filename: str):
    # Sanitize filename to a safe key
    safe = filename.replace(os.sep, "_").replace(" ", "_")
    return os.path.join(CACHE_DIR, f"{safe}.pkl")


def load_from_cache(filename: str):
    path = _cache_path_for(filename)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: failed to load cache {path}: {e}")
    return None


def save_to_cache(filename: str, data):
    path = _cache_path_for(filename)
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Warning: failed to save cache {path}: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze TinyBird model on spectrogram data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint directory (containing config.json and weights/)")  # fmt: skip
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint file to load (e.g., model_step_005000.pth). If not specified, loads the latest checkpoint.")  # fmt: skip
    parser.add_argument("--data_dir", type=str, default=None, help="Path to directory containing .pt spectrogram files (default: uses val_dir from config.json)")  # fmt: skip
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="Index of the spectrogram file to analyze (>=0). If negative, process ALL files (default: -1)",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"], help="Device to run inference on (default: cuda if available, else cpu)")  # fmt: skip
    parser.add_argument("--start-block", type=int, default=None, help="Start block index for save_reconstruction mode")
    parser.add_argument("--last-block", type=int, default=None, help="Last block index for save_reconstruction mode")
    parser.add_argument(
        "--isolate-block", action="store_true", help="Use isolate_block=True during save_reconstruction"
    )
    parser.add_argument("--test", action="store_true", help="Run the model in test mode (test logic placeholder).")
    return parser.parse_args()


def test_file(model, dataset, index, device):
    """
    Loading sample at index 0
    Filename: 1740770732_USA5483_USA5494.1168.0_300
    Spectrogram shape: torch.Size([1, 1, 128, 2524])
    Chirp intervals shape: torch.Size([1, 109, 2])
    Chirp labels shape: torch.Size([109])
    Chirp feats shape: torch.Size([109, 2, 6])
    Number of valid chirps: 109
    """

    # generate a torch array that goes from 0 to 10
    x = torch.arange(0, 10, device=device, dtype=torch.long)
    x = x.view(1, 1, 1, 10).expand(1, 1, 3, 10).clone()
    x_i = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 10]], device=device, dtype=torch.long)
    x_i = x_i.unsqueeze(0)
    N = torch.tensor([5], device=device, dtype=torch.long)
    print("N shape: ", N.shape)
    print("x shape: ", x.shape)
    print("x[0,0,0,:]:", x[0, 0, 0, :].tolist())
    print("x_i shape: ", x_i.shape)
    print("x_i[0,0,:,0]:", x_i[0, :, 0].tolist())
    print("x_i[0,0,:,1]:", x_i[0, :, 1].tolist())

    print("sample_data_indices")
    x_sdi, x_i_sdi = model.sample_data_indices(x, x_i, N, [3, 4, 0, 1, 2])
    print("x_sdi shape: ", x_sdi.shape)
    print("x_sdi[0,0,0,:]:", x_sdi[0, 0, 0, :].tolist())
    print("x_i_sdi shape: ", x_i_sdi.shape)
    print("x_i_sdi[0,0,:,0]:", x_i_sdi[0, :, 0].tolist())
    print("x_i_sdi[0,0,:,1]:", x_i_sdi[0, :, 1].tolist())

    print("remap_boundaries")
    x_r, x_i_r = model.remap_boundaries(x, x_i, N, move_block=2)
    print("x_r shape: ", x_r.shape)
    print("x_r[0,0,0,:]:", x_r[0, 0, 0, :].tolist())
    print("x_i_r shape: ", x_i_r.shape)
    print("x_i_r[0,0,:,0]:", x_i_r[0, :, 0].tolist())
    print("x_i_r[0,0,:,1]:", x_i_r[0, :, 1].tolist())


def process_xl(x_i, x_l, N, device):
    x_i = x_i.to(device, non_blocking=True).long()
    x_l = x_l.to(device, non_blocking=True).float()

    B = x_i.shape[0]
    max_N = int(N.max().item())
    x_l_out = torch.zeros((B, max_N), dtype=torch.float32, device=device)

    for b in range(B):
        n_valid = int(N[b].item())
        for block in range(n_valid):
            start, end = x_i[b, block, 0].item(), x_i[b, block, 1].item()
            x_l_out[b, block] = x_l[b, start:end].mean()

    return x_l_out


def prepare_sample(model, dataset, index, device):
    x, x_i, x_l, _, N, filename = dataset[index]
    x = x.unsqueeze(0).float().to(device)
    x_i = x_i.unsqueeze(0).to(device)
    x_l = x_l.unsqueeze(0).to(device).float()
    N = torch.tensor([N], dtype=torch.long, device=device)

    # Compute chirp start/end/dt/lt
    starts = x_i[0, : N.item(), 0]
    ends = x_i[0, : N.item(), 1]
    x_dt = torch.cat([torch.tensor([0.0], device=device), starts[1:] - ends[:-1]])
    x_lt = ends - starts

    x_l_mean = process_xl(x_i, x_l, N, device)
    x, x_i = model.compactify_data(x.clone(), x_i.clone(), N.clone())

    return x, x_i, x_l_mean, x_dt, x_lt, N, filename


def compute_loss(model, x, x_i, N, start_block, last_block, x_dt, x_lt):
    """
    Compute masked MSE loss for a given block configuration.
    Returns: (loss, xs, x_is, bool_mask, pred, mblock, indices)
    """

    indices = list(range(last_block + 1, N.max().item())) + list(range(last_block + 1))
    start_index = indices.index(start_block)

    indices = indices[start_index:]
    mblock = [len(indices) - 1]

    dt = x_dt[indices].sum().item()
    lt = x_lt[indices].sum().item()

    if start_block == last_block:
        nan = torch.tensor(float('nan'), device=x.device)
        return nan, nan, nan, dt, lt, None, None, None, None, None, None

    try:
        xs, x_is = model.sample_data_indices(x.clone(), x_i.clone(), N.clone(), indices)
        W = xs.shape[-1]
        h, idx_restore, bool_mask, T = model.forward_encoder(xs, x_is, mblock=mblock)
        pred = model.forward_decoder(h, idx_restore, T)
        reconstruction_loss, channel_losses = model.loss_mse(xs, pred, bool_mask, return_per_channel=True)
        return reconstruction_loss, channel_losses, dt, lt, xs, x_is, bool_mask, pred, mblock, indices

    except RuntimeError as e:
        msg = str(e).lower()
        if ("out of memory" in msg or "cuda" in msg) and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            nan = torch.tensor(float('nan'), device=x.device)
            return nan, nan, nan, dt, lt, None, None, None, None, None, None
        else:
            raise


def _normalize_patch_size(patch_size):
    if isinstance(patch_size, (tuple, list)):
        assert len(patch_size) == 2, "patch_size tuple must have length 2"
        return int(patch_size[0]), int(patch_size[1])
    return int(patch_size), int(patch_size)


def reconstruct_channel_from_tokens(xs, pred, bool_mask, patch_size, channel_idx):
    """
    Reconstruct a single channel by replacing masked tokens with decoder predictions.

    Args:
        xs: (B, C, H, W) normalized spectrogram window.
        pred: (B, T, P) decoder output in normalized space.
        bool_mask: (B, T) boolean mask for tokens that were masked.
        patch_size: tuple or int describing patch height/width.
        channel_idx: which channel to reconstruct.

    Returns:
        Tensor of shape (B, 1, H, W) containing the reconstructed channel.
    """
    ph, pw = _normalize_patch_size(patch_size)
    patch_area = ph * pw

    xs_ch = xs[:, channel_idx : channel_idx + 1]
    unfold = torch.nn.Unfold(kernel_size=(ph, pw), stride=(ph, pw))
    fold = torch.nn.Fold(output_size=xs_ch.shape[-2:], kernel_size=(ph, pw), stride=(ph, pw))

    x_patches = unfold(xs_ch).transpose(1, 2)  # (B, T, patch_area)
    target_mean = x_patches.mean(dim=-1, keepdim=True)
    target_std = x_patches.std(dim=-1, keepdim=True)

    start = channel_idx * patch_area
    stop = start + patch_area
    pred_slice = pred[..., start:stop]
    pred_denorm = pred_slice * (target_std + 1e-6) + target_mean

    out_patches = x_patches.clone()
    out_patches[bool_mask] = pred_denorm[bool_mask]
    out_patches = out_patches.transpose(1, 2)  # (B, patch_area, T)
    x_rec_ch = fold(out_patches)  # (B, 1, H, W)
    return x_rec_ch


def column_mask_from_bool_mask(bool_mask, H, W, patch_size):
    """
    Convert token-level mask to per-column mask in pixel space.

    Args:
        bool_mask: (B, T) boolean tensor.
        H, W: spatial dimensions.
        patch_size: tuple or int describing patch size.

    Returns:
        Tensor of shape (B, W) with True where any patch in the column was masked.
    """
    ph, pw = _normalize_patch_size(patch_size)
    Htok = max(1, H // ph)
    Wtok = max(1, W // pw)
    return bool_mask.reshape(-1, Htok, Wtok).any(dim=1)


def compute_best_context_indices(loss_matrix: torch.Tensor) -> np.ndarray:
    """
    For each last_block (column) return the start_block (row) that achieves the lowest loss.
    Returns an array of shape (N,) with -1 where no finite loss is available.
    """
    mat_np = loss_matrix.detach().cpu().numpy()
    n = mat_np.shape[0]
    best = np.full(n, -1, dtype=np.int32)
    for last in range(n):
        col = mat_np[:, last]
        finite = np.isfinite(col)
        if finite.any():
            best[last] = int(np.nanargmin(np.where(finite, col, np.nan)))
    return best


def compute_reconstructed_channel_preference(
    model,
    x,
    x_i,
    N,
    x_dt,
    x_lt,
    loss_matrix: torch.Tensor,
    dataset_mean: float,
    dataset_std: float,
) -> np.ndarray:
    """
    For each last_block, reconstruct the masked block using the best-loss context and
    compute the mean channel preference (analogous to chirp_labels).

    Returns:
        np.ndarray of shape (N,) with values in [0,1] or NaN where unavailable.
    """
    best_indices = compute_best_context_indices(loss_matrix)
    n_blocks = int(N.max().item())
    results = np.full((n_blocks,), np.nan, dtype=np.float32)
    dataset_mean = float(dataset_mean)
    dataset_std = float(dataset_std)

    for last_block in range(n_blocks):
        start_block = int(best_indices[last_block])
        if start_block < 0 or start_block == last_block:
            continue
        with torch.no_grad():
            loss, channel_losses, dt, lt, xs, x_is, bool_mask, pred, mblock, indices = compute_loss(
                model,
                x,
                x_i,
                N,
                start_block=start_block,
                last_block=last_block,
                x_dt=x_dt,
                x_lt=x_lt,
            )
        if xs is None or pred is None or bool_mask is None or x_is is None or indices is None:
            continue

        B, C, H, W = xs.shape
        if C < 2:
            continue

        target_idx = len(indices) - 1
        bounds = x_is[0, target_idx].detach().cpu().numpy()
        block_start = int(bounds[0])
        block_end = int(bounds[1])
        if block_end <= block_start:
            continue
        block_start = max(0, min(block_start, W - 1))
        block_end = max(block_start + 1, min(block_end, W))

        col_mask_tensor = column_mask_from_bool_mask(bool_mask.bool(), H, W, model.patch_size)
        col_mask = col_mask_tensor[0].detach().cpu().numpy().astype(bool)
        block_columns = np.arange(block_start, block_end)
        if col_mask.any():
            masked_cols = np.where(col_mask)[0]
            overlap = np.intersect1d(block_columns, masked_cols)
            if overlap.size > 0:
                block_columns = overlap
        if block_columns.size == 0:
            continue

        energies = []
        for ch in range(C):
            x_rec_ch = reconstruct_channel_from_tokens(xs, pred, bool_mask, model.patch_size, ch)
            x_rec_np = x_rec_ch[0, 0].detach().cpu().numpy()
            denorm = x_rec_np * dataset_std + dataset_mean
            power = np.power(10.0, denorm / 10.0, dtype=np.float64)
            energies.append(power.sum(axis=0))
        energies = np.stack(energies, axis=0)
        block_energy = energies[:, block_columns]
        if block_energy.shape[0] < 2 or block_energy.shape[1] == 0:
            continue
        labels = (block_energy[1] > block_energy[0]).astype(np.float32)
        results[last_block] = float(labels.mean())
    return results


def save_reconstruction(model, x, x_i, N, last_block, x_dt, x_lt, *, filename, index, images_dir='images'):
    """
    Save reconstruction visualizations for a given block configuration.
    Returns: (list_of_paths, total_loss, per_channel_losses)
    """
    # Choose start_block using cached losses (best per-column), fallback to last_block+1
    default_start = last_block + 1
    best_start = default_start
    try:
        cached = load_from_cache(filename)
        if cached is not None and len(cached) >= 1:
            cached_losses = cached[0]
            if torch.is_tensor(cached_losses):
                col = cached_losses.detach().cpu().numpy()[:, last_block]
            else:
                col = np.array(cached_losses)[:, last_block]
            finite_mask = np.isfinite(col)
            if finite_mask.any():
                best_start = int(np.nanargmin(np.where(finite_mask, col, np.nan)))
    except Exception:
        best_start = default_start

    loss, channel_losses, dt, lt, xs, x_is, bool_mask, pred, mblock, indices = compute_loss(
        model, x, x_i, N, start_block=best_start, last_block=last_block, x_dt=x_dt, x_lt=x_lt
    )

    if torch.is_tensor(channel_losses):
        channel_losses_np = channel_losses.detach().cpu().numpy().reshape(-1)
    elif channel_losses is None:
        channel_losses_np = None
    else:
        channel_losses_np = np.array(channel_losses, dtype=float).reshape(-1)

    if xs is None or pred is None or bool_mask is None:
        return [], loss, channel_losses_np

    def _draw_separators(ax, x_is_b):
        K = x_is_b.shape[0]
        for k in range(K - 1):
            pos = int(x_is_b[k, 1])
            ax.axvline(pos - 0.5, linewidth=0.8, alpha=0.8, color='cyan')

    B, C, H, W = xs.shape
    col_mask = column_mask_from_bool_mask(bool_mask.bool(), H, W, model.patch_size)[0].detach().cpu().numpy()
    col_mask_bool = col_mask.astype(bool)
    true_idxs = np.where(col_mask_bool)[0]
    zoom_has_span = true_idxs.size > 0
    if zoom_has_span:
        span_start = int(true_idxs.min())
        span_end = int(true_idxs.max() + 1)
    else:
        span_start, span_end = 0, 1
    mask_hw = ~np.broadcast_to(col_mask_bool, (H, W))

    total_loss_scalar = None
    if loss is not None:
        if torch.is_tensor(loss):
            total_loss_scalar = float(loss.detach().cpu().item())
        elif isinstance(loss, (np.ndarray, np.generic)):
            total_loss_scalar = float(loss)
        else:
            try:
                total_loss_scalar = float(loss)
            except Exception:
                total_loss_scalar = None

    from matplotlib.gridspec import GridSpec

    paths = []
    os.makedirs(images_dir, exist_ok=True)
    for ch in range(C):
        x_rec_ch = reconstruct_channel_from_tokens(xs, pred, bool_mask, model.patch_size, ch)
        xs_cpu = xs[0, ch].detach().cpu().numpy()
        x_rec_cpu = x_rec_ch[0, 0].detach().cpu().numpy()
        vmin = min(np.nanmin(xs_cpu), np.nanmin(x_rec_cpu))
        vmax = max(np.nanmax(xs_cpu), np.nanmax(x_rec_cpu))

        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, width_ratios=[3.5, 1.4], height_ratios=[1, 1, 1], wspace=0.10, hspace=0.08)

        ax_in_full = fig.add_subplot(gs[0, 0])
        ax_rec_full = fig.add_subplot(gs[1, 0], sharex=ax_in_full, sharey=ax_in_full)
        ax_err_full = fig.add_subplot(gs[2, 0], sharex=ax_in_full, sharey=ax_in_full)

        ax_in_zoom = fig.add_subplot(gs[0, 1], sharey=ax_in_full)
        ax_rec_zoom = fig.add_subplot(gs[1, 1], sharey=ax_in_full)
        ax_err_zoom = fig.add_subplot(gs[2, 1], sharey=ax_in_full)

        cmap = plt.get_cmap('magma')
        im0 = ax_in_full.imshow(
            xs_cpu, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', aspect='auto', interpolation='nearest'
        )
        ax_in_full.set_title(f'Input (channel {ch})')
        im1 = ax_rec_full.imshow(
            x_rec_cpu, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', aspect='auto', interpolation='nearest'
        )
        ax_rec_full.set_title(f'Reconstruction (channel {ch})')

        # Overlay masked span on input
        in_span = False
        start_idx = 0
        for i in range(len(col_mask_bool)):
            if col_mask_bool[i] and not in_span:
                start_idx = i
                in_span = True
            elif not col_mask_bool[i] and in_span:
                ax_in_full.axvspan(start_idx - 0.5, i - 0.5, alpha=0.25, color='yellow')
                in_span = False
        if in_span:
            ax_in_full.axvspan(start_idx - 0.5, len(col_mask_bool) - 0.5, alpha=0.25, color='yellow')

        diff = x_rec_cpu - xs_cpu
        diff_ma = np.ma.masked_array(diff, mask=mask_hw)
        if col_mask_bool.any():
            max_abs = np.nanmax(np.abs(diff[:, col_mask_bool]))
            if not np.isfinite(max_abs) or max_abs == 0:
                max_abs = 1e-6
        else:
            max_abs = np.nanmax(np.abs(diff)) if np.isfinite(np.nanmax(np.abs(diff))) else 1.0
        cmap_diff = plt.get_cmap('coolwarm').copy()
        cmap_diff.set_bad(color='black')
        im2 = ax_err_full.imshow(
            diff_ma, cmap=cmap_diff, vmin=-max_abs, vmax=max_abs, origin='lower', aspect='auto', interpolation='nearest'
        )
        ax_err_full.set_title(f'Error (channel {ch}, masked)')

        if zoom_has_span:
            xs_crop = xs_cpu[:, span_start:span_end]
            xrec_crop = x_rec_cpu[:, span_start:span_end]
            diff_crop = diff[:, span_start:span_end]

            ax_in_zoom.imshow(
                xs_crop, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', aspect='auto', interpolation='nearest'
            )
            ax_rec_zoom.imshow(
                xrec_crop, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', aspect='auto', interpolation='nearest'
            )
            ax_err_zoom.imshow(
                diff_crop,
                cmap=cmap_diff,
                vmin=-max_abs,
                vmax=max_abs,
                origin='lower',
                aspect='auto',
                interpolation='nearest',
            )

            ax_in_zoom.set_title('Masked span (Input)')
            ax_rec_zoom.set_title('Masked span (Reconstruction)')
            ax_err_zoom.set_title('Masked span (Error)')

            ax_in_zoom.set_xlim(0, xs_crop.shape[1])
            ax_rec_zoom.set_xlim(0, xrec_crop.shape[1])
            ax_err_zoom.set_xlim(0, diff_crop.shape[1])

            for axz in (ax_in_zoom, ax_rec_zoom, ax_err_zoom):
                axz.tick_params(axis='y', which='both', left=False, labelleft=False)
        else:
            for axz in (ax_in_zoom, ax_rec_zoom, ax_err_zoom):
                axz.axis('off')

        separators = x_is[0].detach().cpu().numpy()
        _draw_separators(ax_in_full, separators)
        _draw_separators(ax_rec_full, separators)
        _draw_separators(ax_err_full, separators)

        cbar0 = fig.colorbar(im0, ax=ax_in_full, fraction=0.025, pad=0.02)
        cbar0.set_label('Amplitude', rotation=270, labelpad=12)
        cbar1 = fig.colorbar(im1, ax=ax_rec_full, fraction=0.025, pad=0.02)
        cbar1.set_label('Amplitude', rotation=270, labelpad=12)
        cbar2 = fig.colorbar(im2, ax=ax_err_full, fraction=0.025, pad=0.02)
        cbar2.set_label('Diff (recon - input)', rotation=270, labelpad=12)

        mblock_str = mblock[0] if mblock is not None and len(mblock) > 0 else 'N/A'
        total_loss_str = f"{total_loss_scalar:.6f}" if total_loss_scalar is not None else "NaN"
        channel_loss_val = None
        if channel_losses_np is not None and ch < channel_losses_np.shape[0]:
            channel_loss_val = channel_losses_np[ch]
        channel_loss_str = (
            f"{float(channel_loss_val):.6f}" if channel_loss_val is not None and np.isfinite(channel_loss_val) else "NaN"
        )
        title = (
            f"{filename}, idx={index}, last={last_block}, channel={ch}, mblock={mblock_str}, "
            f"loss_total={total_loss_str}, loss_ch={channel_loss_str}"
        )
        fig.suptitle(title, fontsize=11)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        suffix = f"_ch{ch}" if C > 1 else ""
        out_path = os.path.join(images_dir, f"reconstruction_idx{index}_last{last_block}{suffix}.png")
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        paths.append(out_path)

    return paths, loss, channel_losses_np


def process_file(model, dataset, index, device):
    x, x_i, x_l_mean, x_dt, x_lt, N, filename = prepare_sample(model, dataset, index, device)

    print(f"  Filename: {filename}")
    print(f"  Spectrogram shape: {x.shape}")
    print(f"  Chirp intervals shape: {x_i.shape}")
    print(f"  Number of valid chirps: {N.item()}")

    def compute_losses(x, x_i, N, x_dt, x_lt):
        n_valid_chirps = N.max().item()
        channels = int(x.shape[1])
        losses_reconstruction = torch.full((n_valid_chirps, n_valid_chirps), float('nan'), device=device)
        losses_reconstruction_channels = torch.full(
            (channels, n_valid_chirps, n_valid_chirps), float('nan'), device=device
        )
        dt_mat = torch.full((n_valid_chirps, n_valid_chirps), float('nan'), device=device)
        lt_mat = torch.full((n_valid_chirps, n_valid_chirps), float('nan'), device=device)
        total_jobs = max((n_valid_chirps - 1) * (n_valid_chirps - 1), 1)
        print(f"\nComputing losses/dt/lt for {losses_reconstruction.numel()} (rows × starts)...")
        with tqdm(total=total_jobs, desc="Computing losses/dt/lt") as pbar:
            for last_block in range(1, n_valid_chirps):
                for start_block in range(0, n_valid_chirps - 1):
                    with torch.no_grad():
                        loss_reconstruction, channel_losses, dt_val, lt_val, *_ = compute_loss(
                            model, x, x_i, N, start_block, last_block, x_dt, x_lt
                        )
                    losses_reconstruction[start_block, last_block] = (
                        float('nan') if torch.isnan(loss_reconstruction) else loss_reconstruction
                    )
                    losses_reconstruction_channels[:, start_block, last_block] = channel_losses.squeeze(0)
                    dt_mat[start_block, last_block] = dt_val
                    lt_mat[start_block, last_block] = lt_val
                    pbar.update(1)
        return losses_reconstruction, losses_reconstruction_channels, dt_mat, lt_mat

    cached = load_from_cache(filename)
    cache_data = {}
    cache_dirty = False

    def _ensure_tensor(val):
        if isinstance(val, torch.Tensor):
            return val
        if isinstance(val, np.ndarray):
            return torch.from_numpy(val)
        return None

    if isinstance(cached, dict):
        cache_data = dict(cached)
        print(f"Loaded cached data for {filename}")
    elif isinstance(cached, tuple):
        # Legacy tuple format: upgrade to dict
        if len(cached) >= 4:
            cache_data = {
                "loss_matrix": _ensure_tensor(cached[0]),
                "channel_loss_matrix": _ensure_tensor(cached[1]),
                "dt_matrix": _ensure_tensor(cached[2]),
                "lt_matrix": _ensure_tensor(cached[3]),
            }
            if len(cached) >= 5:
                cache_data["recon_channel_pref"] = _ensure_tensor(cached[4])
            cache_dirty = True  # ensure re-saved in new format
            print(f"Upgraded legacy cache for {filename}")
        elif cached is not None:
            print(f"Discarding incompatible cache for {filename}")
    elif cached is not None:
        print(f"Discarding unrecognized cache format for {filename}")

    all_losses_reconstruction = _ensure_tensor(cache_data.get("loss_matrix"))
    all_losses_channels = _ensure_tensor(cache_data.get("channel_loss_matrix"))
    all_dt = _ensure_tensor(cache_data.get("dt_matrix"))
    all_lt = _ensure_tensor(cache_data.get("lt_matrix"))
    recon_channel_pref = _ensure_tensor(cache_data.get("recon_channel_pref"))

    if all_losses_reconstruction is not None:
        all_losses_reconstruction = all_losses_reconstruction.cpu()
    if all_losses_channels is not None:
        all_losses_channels = all_losses_channels.cpu()
    if all_dt is not None:
        all_dt = all_dt.cpu()
    if all_lt is not None:
        all_lt = all_lt.cpu()
    if recon_channel_pref is not None:
        recon_channel_pref = recon_channel_pref.cpu()

    if all_losses_reconstruction is None or all_losses_channels is None or all_dt is None or all_lt is None:
        all_losses_reconstruction, all_losses_channels, all_dt, all_lt = compute_losses(x, x_i, N, x_dt, x_lt)
        all_losses_reconstruction = all_losses_reconstruction.detach().cpu()
        all_losses_channels = all_losses_channels.detach().cpu()
        all_dt = all_dt.detach().cpu()
        all_lt = all_lt.detach().cpu()
        cache_data["loss_matrix"] = all_losses_reconstruction
        cache_data["channel_loss_matrix"] = all_losses_channels
        cache_data["dt_matrix"] = all_dt
        cache_data["lt_matrix"] = all_lt
        cache_dirty = True

    dataset_mean = float(dataset.mean)
    dataset_std = float(dataset.std)

    if recon_channel_pref is None:
        recon_pref_np = compute_reconstructed_channel_preference(
            model,
            x,
            x_i,
            N,
            x_dt,
            x_lt,
            all_losses_reconstruction,
            dataset_mean,
            dataset_std,
        )
        recon_channel_pref = torch.from_numpy(recon_pref_np)
        cache_data["recon_channel_pref"] = recon_channel_pref
        cache_dirty = True

    if cache_dirty:
        # Always persist cache on CPU tensors
        cache_to_store = {
            "loss_matrix": all_losses_reconstruction.cpu(),
            "channel_loss_matrix": all_losses_channels.cpu(),
            "dt_matrix": all_dt.cpu(),
            "lt_matrix": all_lt.cpu(),
            "recon_channel_pref": recon_channel_pref.cpu(),
        }
        save_to_cache(filename, cache_to_store)

    return (
        all_losses_reconstruction,
        all_losses_channels,
        all_dt,
        all_lt,
        filename,
        x_l_mean,
        x_dt,
        x_lt,
        recon_channel_pref,
    )


def compute_lift_matrix(loss_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a (N,N) upper-triangular (wrapped) loss matrix, compute the expected loss per context length,
    its standard deviation, and the lift (actual - expected) for every (start,last) entry.
    """
    n = loss_matrix.shape[0]
    expected = np.full((n,), np.nan, dtype=np.float64)
    expected_std = np.full((n,), np.nan, dtype=np.float64)
    lift = np.full_like(loss_matrix, np.nan, dtype=np.float64)
    for delta in range(1, n):
        vals = []
        coords = []
        for start in range(0, n):
            last = (start + delta) % n
            val = loss_matrix[start, last]
            if np.isfinite(val):
                vals.append(val)
                coords.append((start, last))
        if not vals:
            continue
        baseline = float(np.mean(vals))
        spread = float(np.std(vals))
        expected[delta] = baseline
        expected_std[delta] = spread
        for start, last in coords:
            lift[start, last] = loss_matrix[start, last] - baseline
    return lift, expected, expected_std


def compute_lift_for_channels(loss_matrix: np.ndarray, channel_losses: np.ndarray):
    """
    Computes lift matrices for aggregate loss and each channel-specific loss cube (C,N,N).
    """
    lift_all, expected_all, std_all = compute_lift_matrix(loss_matrix)
    lifts_channels = []
    expected_channels = []
    std_channels = []
    for c in range(channel_losses.shape[0]):
        lift_c, expected_c, std_c = compute_lift_matrix(channel_losses[c])
        lifts_channels.append(lift_c)
        expected_channels.append(expected_c)
        std_channels.append(std_c)
    lifts_channels = np.stack(lifts_channels, axis=0)
    expected_channels = np.stack(expected_channels, axis=0)
    std_channels = np.stack(std_channels, axis=0)
    return lift_all, expected_all, std_all, lifts_channels, expected_channels, std_channels


def build_last_block_profiles(loss_matrix: np.ndarray) -> np.ndarray:
    n = loss_matrix.shape[0]
    profiles = np.full((n, n), np.nan, dtype=np.float64)
    for last in range(n):
        for delta in range(1, n):
            start = (last - delta) % n
            val = loss_matrix[start, last]
            if np.isfinite(val):
                profiles[last, delta] = val
    return profiles


def compute_best_loss_curve(loss_matrix: np.ndarray) -> np.ndarray:
    """
    Returns the lowest achievable loss per last_block (column-wise minima over start_block).
    Matches the "best context" series in the line charts.
    """
    if loss_matrix.ndim != 2:
        raise ValueError("Expected a 2D loss matrix (start_block x last_block).")
    _, cols = loss_matrix.shape
    best_loss = np.full(cols, np.nan, dtype=float)
    for last in range(cols):
        column = loss_matrix[:, last]
        finite = column[np.isfinite(column)]
        if finite.size:
            best_loss[last] = float(np.min(finite))
    return best_loss


def compute_context_gain_matrix(loss_matrix: np.ndarray, expected_curve: np.ndarray | None) -> np.ndarray:
    """
    Measures the incremental change in loss obtained by extending the context one block earlier.
    For each start_block row, compares the loss at start_block vs start_block+1 (i.e., dropping the earliest block).
    Positive values indicate the added block hurts performance; negative values mean the added block lowers loss.
    We subtract the expected improvement for the corresponding context length so values reflect
    "better/worse than average" benefit. Near-zero raw deltas are clamped to zero to avoid noise.
    """
    if loss_matrix.ndim != 2:
        raise ValueError("Expected a 2D loss matrix (start_block x last_block).")
    rows, cols = loss_matrix.shape
    delta = np.full((rows, cols), np.nan, dtype=np.float64)
    if rows == 0 or cols == 0:
        return delta
    for start in range(rows):
        next_start = (start + 1) % rows
        for last in range(cols):
            cur = loss_matrix[start, last]
            next_val = loss_matrix[next_start, last]
            if np.isfinite(cur) and np.isfinite(next_val):
                delta[start, last] = cur - next_val

    finite_mask = np.isfinite(delta)
    if finite_mask.any():
        finite_vals = np.abs(delta[finite_mask])
        scale = float(np.nanpercentile(finite_vals, 90)) if finite_vals.size else 0.0
        zero_tol = max(1e-8, scale * 1e-3)
        small_mask = np.abs(delta) < zero_tol
        delta[small_mask] = 0.0

    benefit = np.full_like(delta, np.nan)
    row_idx = np.arange(rows)[:, None]
    col_idx = np.arange(cols)[None, :]
    context_len = (col_idx - row_idx) % rows

    expected_curve = np.asarray(expected_curve, dtype=float).flatten() if expected_curve is not None else None
    expected_delta = None
    if expected_curve is not None and expected_curve.size:
        expected_delta = np.full_like(expected_curve, np.nan)
        expected_delta[0] = 0.0
        expected_delta[1:] = expected_curve[1:] - expected_curve[:-1]

    for start in range(rows):
        for last in range(cols):
            val = delta[start, last]
            if not np.isfinite(val):
                continue
            if val == 0.0:
                benefit[start, last] = 0.0
                continue
            ctx_len = int(context_len[start, last])
            exp = 0.0
            if expected_delta is not None and 0 <= ctx_len < expected_delta.shape[0]:
                exp = expected_delta[ctx_len]
                if not np.isfinite(exp):
                    exp = 0.0
            benefit[start, last] = val - exp
    return benefit


def select_highlight_profiles(
    loss_matrix: np.ndarray, expected_curve: np.ndarray, expected_std: np.ndarray
) -> list[tuple[str, np.ndarray, str]]:
    profiles = build_last_block_profiles(loss_matrix)
    lift_profiles = profiles - expected_curve[np.newaxis, :]
    mean_lift = np.nanmean(lift_profiles[:, 1:], axis=1)
    if not np.isfinite(mean_lift).any():
        return []
    best_idx = int(np.nanargmin(mean_lift))
    worst_idx = int(np.nanargmax(mean_lift))
    avg_std = float(np.nanmean(expected_std[1:]))

    def closest(target):
        diff = np.abs(mean_lift - target)
        idx = int(np.nanargmin(diff))
        return idx

    better_idx = closest(-avg_std) if np.isfinite(avg_std) else best_idx
    worse_idx = closest(avg_std) if np.isfinite(avg_std) else worst_idx

    return [
        (f"best (last={best_idx})", profiles[best_idx], "better_dark"),
        (f"~ -1σ (last={better_idx})", profiles[better_idx], "better_light"),
        (f"~ +1σ (last={worse_idx})", profiles[worse_idx], "worse_light"),
        (f"worst (last={worst_idx})", profiles[worst_idx], "worse_dark"),
    ]


def plot_expected_curve(expected_curve, expected_std, highlight_profiles, title, filename, index, images_dir, tag):
    lengths = np.arange(expected_curve.shape[0])
    mask = np.isfinite(expected_curve)
    lengths = lengths[mask]
    if lengths.size == 0:
        return None
    values = expected_curve[mask]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lengths, values, marker="o", linewidth=2, color='black', label="Expected loss")
    if expected_std is not None:
        std_vals = expected_std[mask]
        ax.fill_between(lengths, values - std_vals, values + std_vals, alpha=0.2, color='gray', label="±1 std")
    if highlight_profiles:
        color_map = {
            "better_dark": "#1f77b4",
            "better_light": "#87cefa",
            "worse_light": "#ffa500",
            "worse_dark": "#d62728",
        }
        for label, profile, category in highlight_profiles:
            prof_vals = profile[mask]
            ax.plot(
                lengths,
                prof_vals,
                linewidth=1.6,
                color=color_map.get(category, "#666666"),
                label=label,
            )
    ax.set_xlabel("Context length (Δ = last - start)")
    ax.set_ylabel("Expected loss (MSE)")
    ax.set_title(f"{title}\nIndex {index}, File: {filename}")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=7, loc="best")
    out_path = os.path.join(images_dir, f"expected_loss_{tag}_{index}_{filename}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_block_lift_summary(benefit_matrix, filename, index, images_dir, title_prefix="All"):
    """
    Summarize which blocks (by start index) provide the biggest benefit when added to the context.
    """
    if benefit_matrix.size == 0:
        return None
    row_sum = np.nansum(benefit_matrix, axis=1)
    valid_mask = np.isfinite(row_sum)
    if not valid_mask.any():
        return None
    row_sum = np.where(valid_mask, row_sum, np.nan)
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(row_sum.shape[0])
    ax.plot(x, row_sum, linewidth=2, marker='o', markersize=3)
    ax.set_xlabel("Block index (start_block)")
    ax.set_ylabel("Σ Δ advantage vs expected (start vs start+1)")
    ax.set_title(f"{title_prefix} Context Benefit – Index {index}, File: {filename}")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    finite_vals = row_sum[valid_mask]
    if finite_vals.size:
        limit = np.nanmax(np.abs(finite_vals))
        if np.isfinite(limit) and limit > 0:
            ax.set_ylim(-limit, limit)
    ax.axhline(0.0, color='black', linewidth=1.0, linestyle='--', alpha=0.7)
    if row_sum.shape[0] > 0:
        xticks = np.arange(0, row_sum.shape[0], 10) if row_sum.shape[0] > 10 else np.array([0])
        ax.set_xticks(np.unique(xticks))
    out_path = os.path.join(images_dir, f"block_lift_{title_prefix.lower()}_{index}_{filename}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Log top/bottom performers for quick reference
    with np.errstate(invalid="ignore"):
        order = np.argsort(row_sum[valid_mask])
    # Map sorted order back to actual indices
    valid_indices = np.where(valid_mask)[0]
    sorted_blocks = valid_indices[np.argsort(row_sum[valid_mask])]
    best = sorted_blocks[::-1][:5]
    worst = sorted_blocks[:5]
    highlight_indices = np.unique(np.concatenate([best, worst]))
    ax.scatter(highlight_indices, row_sum[highlight_indices], color='black', s=40, zorder=5)
    for idx in highlight_indices:
        val = row_sum[idx]
        ax.text(
            idx,
            val,
            str(int(idx)),
            fontsize=7,
            fontweight='bold',
            ha='center',
            va='bottom' if val >= 0 else 'top',
        )
    print("Top benefit blocks:", [(int(idx), float(row_sum[idx])) for idx in best])
    print("Lowest benefit blocks:", [(int(idx), float(row_sum[idx])) for idx in worst])
    return out_path


def plot_mean_scatter(row_mean, best_loss_curve, filename, index, images_dir, tag="scatter"):
    valid = np.isfinite(row_mean) & np.isfinite(best_loss_curve)
    if not valid.any():
        return None
    indices = np.where(valid)[0]
    xs = row_mean[valid]
    ys = best_loss_curve[valid]
    fig, ax = plt.subplots(figsize=(6, 6))
    norm = (indices - indices.min()) / max(1, (indices.max() - indices.min()))
    scatter = ax.scatter(xs, ys, alpha=0.9, s=30, c=norm, cmap='viridis', edgecolor='none')
    distances = np.abs(xs) + np.abs(ys)
    extreme_mask = np.zeros_like(distances, dtype=bool)
    if distances.size:
        topk = min(10, distances.size)
        extreme_indices = np.argpartition(distances, -topk)[-topk:]
        extreme_mask[extreme_indices] = True
    for idx_pt, (blk, xv, yv) in enumerate(zip(indices, xs, ys)):
        txt = ax.text(
            xv,
            yv,
            str(int(blk)),
            fontsize=6,
            color='black',
            ha='center',
            va='center',
            fontweight='bold' if extreme_mask[idx_pt] else 'normal',
        )
        txt.set_bbox(dict(facecolor='white', alpha=0.65, edgecolor='none', pad=0.5))
    x_limit = max(abs(xs).max(), 1e-6)
    y_mean = float(ys.mean())
    max_dev = float(np.max(np.abs(ys - y_mean))) if ys.size else 0.0
    if not np.isfinite(max_dev) or max_dev == 0.0:
        max_dev = max(abs(y_mean) * 0.05, 1e-6)
    y_min = y_mean - max_dev
    y_max = y_mean + max_dev
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(y_min, y_max)
    ax.axhline(y_mean, color='gray', linewidth=1.0, linestyle='--', alpha=0.8, label='_nolegend_')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Σ Δ vs expected per start_block (context benefit)')
    ax.set_ylabel('Lowest loss per last_block (best context)')
    ax.set_title(
        "Context benefit vs difficulty\n"
        "Left side: blocks whose added context outperforms the average improvement. Bottom: blocks that are easy to predict."
    )
    legend_elements, legend_labels = scatter.legend_elements(num=6)
    sample_blocks = np.linspace(indices.min(), indices.max(), num=min(8, len(indices)), dtype=int)
    sample_blocks = np.unique(sample_blocks)
    legend_handles = []
    legend_labels = []
    from matplotlib.lines import Line2D
    cmap = plt.get_cmap('viridis')
    for blk in sample_blocks:
        norm_val = (blk - indices.min()) / max(1, (indices.max() - indices.min()))
        legend_handles.append(Line2D([0], [0], marker='o', color='none', markerfacecolor=cmap(norm_val), markersize=6))
        legend_labels.append(f"last={blk}")
    ax.legend(legend_handles, legend_labels, title="Block index", loc="upper left", fontsize=7)
    out_path = os.path.join(images_dir, f"mean_scatter_{tag}_{index}_{filename}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("TinyBird Model Analysis")
    print("=" * 60)

    # Load the model
    print(f"\nLoading model from: {args.model_path}")
    if args.checkpoint:
        print(f"Using checkpoint: {args.checkpoint}")

    model, config = load_model_from_checkpoint(run_dir=args.model_path, checkpoint_file=args.checkpoint, fallback_to_random=False)  # fmt: skip

    # Move model to device
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    print(f"\nModel loaded successfully on device: {device}")
    print("Model configuration:")
    print(f"  Encoder hidden dim: {config.get('enc_hidden_d', 'N/A')}")
    print(f"  Decoder hidden dim: {config.get('dec_hidden_d', 'N/A')}")
    print(f"  Patch size: {config.get('patch_size', 'N/A')}")
    print(f"  Max sequence length: {config.get('max_seq', 'N/A')}")

    # Determine data directory: use command line arg if provided, otherwise use val_dir from config
    if args.data_dir is not None:
        data_dir = args.data_dir
        print(f"\nUsing data directory from command line: {data_dir}")
    elif 'val_dir' in config:
        data_dir = config['val_dir']
        print(f"\nUsing validation directory from config: {data_dir}")
    else:
        raise ValueError("No data directory specified. Either provide --data_dir or ensure val_dir is in config.json")

    # Load dataset with normalized spectrograms
    dataset = SpectogramDataset(dir=data_dir, n_mels=config.get('mels', 128), n_timebins=config.get('num_timebins', 1024), pad_crop=True)  # fmt: skip
    print(f"Dataset size: {len(dataset)} files")

    images_dir = "images"
    os.makedirs(images_dir, exist_ok=True)

    # === Test mode branch ===
    if args.test:
        print("\nTest mode enabled.")
        if args.index >= 0:
            if args.index >= len(dataset):
                raise ValueError(f"Index {args.index} out of range. Dataset has {len(dataset)} files.")
            test_file(model, dataset, args.index, device)
        else:
            print(f"\nProcessing all {len(dataset)} files (test mode)...")
            for i in range(len(dataset)):
                test_file(model, dataset, i, device)
        print("\n" + "=" * 60)
        print("Test mode complete!")
        print("=" * 60)
        return

    # === Reconstruction-only mode branch ===
    reconstruction_mode = (args.start_block is not None) and (args.last_block is not None)
    if reconstruction_mode:
        print("\nReconstruction-only mode enabled.")
        print(f"Params: start_block={args.start_block}, last_block={args.last_block}")

        def run_reconstruction(i: int):
            print(f"\nLoading sample at index {i}")
            x, x_i, x_l_mean, x_dt, x_lt, N, filename = prepare_sample(model, dataset, i, device)

            for last_block in range(args.start_block, args.last_block + 1):
                print(f"Generating reconstruction for last_block={last_block}")
                paths, loss, channel_losses = save_reconstruction(
                    model,
                    x,
                    x_i,
                    N,
                    last_block=last_block,
                    x_dt=x_dt,
                    x_lt=x_lt,
                    filename=filename,
                    index=i,
                    images_dir=images_dir,
                )
                if paths:
                    if torch.is_tensor(loss):
                        loss_val = float(loss.detach().cpu().item())
                    else:
                        try:
                            loss_val = float(loss)
                        except (TypeError, ValueError):
                            loss_val = None
                    channel_vals = None
                    if channel_losses is not None and len(channel_losses):
                        channel_vals = []
                        for val in channel_losses:
                            try:
                                channel_vals.append(float(val))
                            except (TypeError, ValueError):
                                channel_vals.append(float('nan'))
                    for idx_path, path in enumerate(paths):
                        extra = ""
                        if channel_vals is not None and idx_path < len(channel_vals):
                            ch_val = channel_vals[idx_path]
                            extra = f", channel_loss={ch_val:.6f}" if np.isfinite(ch_val) else ", channel_loss=NaN"
                        loss_text = f"{loss_val:.6f}" if loss_val is not None else "NaN"
                        print(f"Saved reconstruction → {path} (total_loss={loss_text}{extra})")
                else:
                    print(f"Skipped reconstruction for last_block={last_block} (NaN or error)")

        if args.index >= 0:
            if args.index >= len(dataset):
                raise ValueError(f"Index {args.index} out of range. Dataset has {len(dataset)} files.")
            run_reconstruction(args.index)
        else:
            print(f"\nProcessing all {len(dataset)} files (reconstruction-only)...")
            for i in range(len(dataset)):
                run_reconstruction(i)
        print("\n" + "=" * 60)
        print("Reconstruction complete!")
        print("=" * 60)
        return

    # === Default: loss/plotting mode ===
    def process_and_plot(i: int):
        print(f"\nLoading sample at index {i}")
        (
            all_losses_reconstruction,
            all_losses_channels,
            all_dt,
            all_lt,
            filename,
            x_l_mean,
            x_dt,
            x_lt,
            recon_channel_pref,
        ) = process_file(model, dataset, i, device)

        recon_np = all_losses_reconstruction.detach().cpu().numpy()
        recon_ch_np = all_losses_channels.detach().cpu().numpy()  # (C, N, N)
        dt_np = all_dt.detach().cpu().numpy()
        lt_np = all_lt.detach().cpu().numpy()

        labels_true_np = x_l_mean.detach().cpu().numpy().astype(float).reshape(-1)
        labels_pred_np = (
            recon_channel_pref.detach().cpu().numpy().astype(float).reshape(-1)
            if recon_channel_pref is not None
            else None
        )
        x_dt_np = x_dt.detach().cpu().numpy()
        x_lt_np = x_lt.detach().cpu().numpy()

        lift_np, expected_curve, expected_std, lift_ch_np, expected_curve_ch, expected_std_ch = compute_lift_for_channels(
            recon_np, recon_ch_np
        )
        best_loss_all = compute_best_loss_curve(recon_np)
        best_loss_channels = (
            np.stack([compute_best_loss_curve(recon_ch_np[ch_idx]) for ch_idx in range(recon_ch_np.shape[0])], axis=0)
            if recon_ch_np.size
            else np.empty((0, recon_np.shape[1]))
        )
        context_gain_all = compute_context_gain_matrix(recon_np, expected_curve)
        context_gain_channels = []
        if recon_ch_np.size:
            for ch_idx in range(recon_ch_np.shape[0]):
                gain_ch = compute_context_gain_matrix(recon_ch_np[ch_idx], expected_curve_ch[ch_idx])
                context_gain_channels.append(gain_ch)
            context_gain_channels = np.stack(context_gain_channels, axis=0)
        else:
            context_gain_channels = np.empty((0, recon_np.shape[0], recon_np.shape[1]))

        def plot_heatmap(
            mat_np,
            title: str,
            cbar_label: str,
            tag: str,
            labels_true_np,
            *,
            note: str | None = None,
            cmap_name: str | None = None,
            center_zero: bool = False,
            top_curve: np.ndarray | None = None,
            top_curve_label: str | None = None,
            row_label: str | None = None,
            labels_pred_np: np.ndarray | None = None,
        ):
            fig_hm, ax_hm = plt.subplots(figsize=(12, 8))
            data_ma = np.ma.masked_invalid(mat_np)

            if cmap_name is None:
                cmap_name = 'RdYlGn_r' if tag.startswith('loss') else 'viridis'
            cmap = plt.get_cmap(cmap_name).copy()
            is_binary = False

            cmap.set_bad(color='black')

            if is_binary:
                vmin, vmax = -0.5, 1.5   # crisp separation between 0 and 1
            else:
                vmin = float(np.nanmin(data_ma)) if np.isfinite(np.nanmin(data_ma)) else 0.0
                vmax = float(np.nanmax(data_ma)) if np.isfinite(np.nanmax(data_ma)) else 1.0
                if vmax <= vmin:
                    vmax = vmin + 1.0

            from matplotlib.colors import TwoSlopeNorm
            norm = None
            if center_zero and not is_binary:
                data_vals = mat_np[np.isfinite(mat_np) & (mat_np != 0.0)]
                if data_vals.size:
                    std = float(np.std(data_vals))
                    clip = 2.5 * std if std > 0 else max(abs(vmax), abs(vmin), 1e-6)
                    clip = max(clip, 1e-6)
                    vmax = clip
                    vmin = -clip
                else:
                    vmax = max(vmax, 1e-6)
                    vmin = min(vmin, -1e-6)
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

            im_hm = ax_hm.imshow(
                data_ma,
                aspect='auto',
                cmap=cmap,
                origin='lower',
                vmin=None if norm else vmin,
                vmax=None if norm else vmax,
                norm=norm,
            )
            cbar_hm = plt.colorbar(im_hm, ax=ax_hm)
            cbar_hm.set_label(cbar_label, rotation=270, labelpad=20, fontsize=12)
            if is_binary:
                cbar_hm.set_ticks([0, 1])
                cbar_hm.set_ticklabels(['0', '1'])

            ax_hm.set_xlabel('last_block (end index)', labelpad=10)
            ax_hm.set_ylabel('start_block (start index)', labelpad=10)
            ax_hm.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax_hm.set_title(f'{title}\nPredict block `last_block` (x) using context starting at `start_block` (y)\nIndex {i}, File: {filename}', fontsize=14, pad=20)

            if note:
                ax_hm.annotate(note, xy=(0.99, 0.01), xycoords='axes fraction', fontsize=9, ha='right', va='bottom')

            # === Channel-preference strips (ground truth vs prediction) ===
            Hh, Wh = mat_np.shape
            lbl_true = np.asarray(labels_true_np, dtype=float)
            lbl_true = np.clip(lbl_true, 0.0, 1.0)
            lbl_true = np.nan_to_num(lbl_true, nan=0.5, posinf=1.0, neginf=0.0)
            lbl_true_x = lbl_true[:Wh]
            lbl_true_y = lbl_true[:Hh]

            if labels_pred_np is not None:
                lbl_pred = np.asarray(labels_pred_np, dtype=float)
                lbl_pred = np.clip(lbl_pred, 0.0, 1.0)
                lbl_pred = np.nan_to_num(lbl_pred, nan=0.5, posinf=1.0, neginf=0.0)
                lbl_pred_x = lbl_pred[:Wh]
                lbl_pred_y = lbl_pred[:Hh]
            else:
                lbl_pred = None
                lbl_pred_x = None
                lbl_pred_y = None

            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(ax_hm)
            cmap_lbl = plt.get_cmap('RdYlGn_r').copy()

            pad_x_true = 0.3
            pad_x_pred = pad_x_true + 0.08
            pad_y_true = 0.45
            pad_y_pred = pad_y_true + 0.12

            ax_strip_x_true = divider.append_axes("bottom", size="3%", pad=pad_x_true, sharex=ax_hm)
            ax_strip_x_true.imshow(
                lbl_true_x[np.newaxis, :],
                aspect='auto',
                cmap=cmap_lbl,
                interpolation='nearest',
                origin='lower',
                vmin=0.0,
                vmax=1.0,
            )
            ax_strip_x_true.set_ylabel("GT channel", fontsize=8)
            ax_strip_x_true.set_xlabel("Green=L, Red=R", fontsize=8)

            ax_strip_x_pred = None
            if lbl_pred_x is not None:
                ax_strip_x_pred = divider.append_axes("bottom", size="3%", pad=pad_x_pred, sharex=ax_hm)
                ax_strip_x_pred.imshow(
                    lbl_pred_x[np.newaxis, :],
                    aspect='auto',
                    cmap=cmap_lbl,
                    interpolation='nearest',
                    origin='lower',
                    vmin=0.0,
                    vmax=1.0,
                )
                ax_strip_x_pred.set_ylabel("Pred channel", fontsize=8)

            ax_strip_y_true = divider.append_axes("left", size="3%", pad=pad_y_true, sharey=ax_hm)
            ax_strip_y_true.imshow(
                lbl_true_y[:, np.newaxis],
                aspect='auto',
                cmap=cmap_lbl,
                interpolation='nearest',
                origin='lower',
                vmin=0.0,
                vmax=1.0,
            )
            ax_strip_y_true.set_ylabel("GT (G=L, R=R)", fontsize=8, rotation=90, labelpad=18)

            ax_strip_y_pred = None
            if lbl_pred_y is not None:
                ax_strip_y_pred = divider.append_axes("left", size="3%", pad=pad_y_pred, sharey=ax_hm)
                ax_strip_y_pred.imshow(
                    lbl_pred_y[:, np.newaxis],
                    aspect='auto',
                    cmap=cmap_lbl,
                    interpolation='nearest',
                    origin='lower',
                    vmin=0.0,
                    vmax=1.0,
                )
                ax_strip_y_pred.set_ylabel("Pred (G=L, R=R)", fontsize=8, rotation=90, labelpad=18)

            bottom_axes = [ax_strip_x_true]
            if ax_strip_x_pred is not None:
                bottom_axes.append(ax_strip_x_pred)
            for ax_strip in bottom_axes:
                ax_strip.set_xlim(ax_hm.get_xlim())
                ax_strip.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                ax_strip.tick_params(axis='y', which='both', left=False, labelleft=False)

            left_axes = [ax_strip_y_true]
            if ax_strip_y_pred is not None:
                left_axes.append(ax_strip_y_pred)
            for ax_strip in left_axes:
                ax_strip.set_ylim(ax_hm.get_ylim())
                ax_strip.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                ax_strip.tick_params(axis='y', which='both', left=False, labelleft=False)

            # 2) Turn ON ticks on the main heatmap (they got auto-disabled by sharex/sharey)
            ax_hm.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
            ax_hm.tick_params(axis='y', which='both', left=True,  labelleft=True)

            # 3) Put ticks every 10 (0, 10, 20, …). Fall back to step=1 for tiny matrices.
            Hh, Wh = mat_np.shape
            step = 10
            sx = step if Wh >= step else 1
            sy = step if Hh >= step else 1

            xticks = np.arange(0, Wh, sx)
            yticks = np.arange(0, Hh, sy)

            # Ensure 0 is included and we don't overshoot the last index
            if (Wh - 1) not in xticks:
                xticks = np.append(xticks, Wh - 1)
            if (Hh - 1) not in yticks:
                yticks = np.append(yticks, Hh - 1)

            ax_hm.set_xticks(xticks)
            ax_hm.set_yticks(yticks)
            ax_hm.set_xticklabels([str(int(t)) for t in xticks], fontsize=9)
            ax_hm.set_yticklabels([str(int(t)) for t in yticks], fontsize=9)

            # (optional) keep ticks inside to avoid overlap with strips
            ax_hm.tick_params(axis='both', direction='in')

            # Column/row summary plots
            row_mean = np.nansum(mat_np, axis=1)
            width = mat_np.shape[1]

            def align_curve(curve):
                if curve is None or width == 0:
                    return curve
                arr_curve = np.asarray(curve, dtype=float)
                if arr_curve.shape[0] == width:
                    return arr_curve
                aligned = np.full(width, np.nan, dtype=float)
                take = min(width, arr_curve.shape[0])
                aligned[:take] = arr_curve[:take]
                return aligned

            if width == 0:
                col_curve = np.array([])
            else:
                with np.errstate(all="ignore"):
                    col_curve = align_curve(top_curve) if top_curve is not None else np.nanmin(mat_np, axis=0)

            ax_col_mean = divider.append_axes("top", size="8%", pad=0.7, sharex=ax_hm)
            ax_col_mean.plot(np.arange(width), col_curve, color="black", linewidth=1.5)
            label_top = top_curve_label or ("Lowest loss per last_block" if top_curve is None else "Metric per last_block")
            ax_col_mean.set_ylabel(label_top, fontsize=8, rotation=0, labelpad=25)
            ax_col_mean.tick_params(axis='x', labelbottom=False)
            ax_col_mean.grid(True, alpha=0.2)
            ax_row_mean = divider.append_axes("right", size="8%", pad=0.55, sharey=ax_hm)
            ax_row_mean.plot(row_mean, np.arange(mat_np.shape[0]), color="black", linewidth=1.5)
            row_label_final = row_label or "Σ Δ vs expected per start_block"
            ax_row_mean.set_xlabel(row_label_final, fontsize=8)
            for label in ax_row_mean.get_xticklabels():
                label.set_rotation(90)
            ax_row_mean.tick_params(axis='y', labelleft=False)
            ax_row_mean.grid(True, alpha=0.2)


            plt.tight_layout()
            plt.subplots_adjust(bottom=0.14, left=0.12, top=0.9, right=.95)
            out_path = os.path.join(images_dir, f"heatmap_{tag}_{i}_{filename}.png")
            fig_hm.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig_hm)

            print(f"Saved: {out_path}")
            return col_curve, row_mean

        def add_heatmap(
            mat,
            title,
            cbar_label,
            tag,
            note,
            cmap_name='RdYlGn_r',
            center_zero=False,
            per_channel=False,
            ch_idx=None,
            top_curve=None,
            top_curve_label=None,
            row_label=None,
            labels_pred=None,
        ):
            return plot_heatmap(
                mat,
                title=title if not per_channel else f"{title} (channel {ch_idx})",
                cbar_label=cbar_label,
                tag=tag if not per_channel else f"{tag}_ch{ch_idx}",
                labels_true_np=labels_true_np,
                note=note,
                cmap_name=cmap_name,
                center_zero=center_zero,
                top_curve=top_curve,
                top_curve_label=top_curve_label,
                row_label=row_label,
                labels_pred_np=labels_pred,
            )

        add_heatmap(
            recon_np,
            'Loss (MSE) Heatmap – Reconstruction',
            'Loss (MSE)',
            'loss_recon',
            'NaN = black; low = green; high = red.',
            cmap_name='coolwarm',
            labels_pred=labels_pred_np,
        )
        for ch_idx in range(recon_ch_np.shape[0]):
            add_heatmap(
                recon_ch_np[ch_idx],
                'Loss (MSE) Heatmap – Reconstruction',
                'Loss (MSE)',
                'loss_recon',
                f'Channel {ch_idx} masked-loss matrix.',
                cmap_name='coolwarm',
                per_channel=True,
                ch_idx=ch_idx,
                labels_pred=labels_pred_np,
            )

        gain_means = {}
        gain_means["all"] = add_heatmap(
            context_gain_all,
            'Context Benefit Heatmap – Reconstruction (all channels)',
            'Δ loss advantage vs expected (start vs start+1)',
            'lift_all',
            'Blue = this block reduces loss more than an average added block (better-than-mean benefit). Red = worse than mean.',
            cmap_name='coolwarm',
            center_zero=True,
            top_curve=best_loss_all,
            top_curve_label="Lowest loss per last_block",
            row_label="Σ Δ vs expected per start_block",
            labels_pred=labels_pred_np,
        )
        for ch_idx in range(context_gain_channels.shape[0]):
            gain_means[f"ch{ch_idx}"] = add_heatmap(
                context_gain_channels[ch_idx],
                'Context Benefit Heatmap – Reconstruction',
                'Δ loss advantage vs expected (start vs start+1)',
                'lift',
                'Blue = this block reduces loss more than an average added block (better-than-mean benefit). Red = worse than mean.',
                cmap_name='coolwarm',
                center_zero=True,
                per_channel=True,
                ch_idx=ch_idx,
                top_curve=best_loss_channels[ch_idx],
                top_curve_label="Lowest loss per last_block",
                row_label="Σ Δ vs expected per start_block",
                labels_pred=labels_pred_np,
            )

        highlights_all = select_highlight_profiles(recon_np, expected_curve, expected_std)
        plot_expected_curve(
            expected_curve,
            expected_std,
            highlights_all,
            "Expected Loss vs Context Length (all channels)",
            filename,
            i,
            images_dir,
            tag="all",
        )
        for ch_idx in range(expected_curve_ch.shape[0]):
            highlights_ch = select_highlight_profiles(recon_ch_np[ch_idx], expected_curve_ch[ch_idx], expected_std_ch[ch_idx])
            plot_expected_curve(
                expected_curve_ch[ch_idx],
                expected_std_ch[ch_idx],
                highlights_ch,
                f"Expected Loss vs Context Length (channel {ch_idx})",
                filename,
                i,
                images_dir,
                tag=f"ch{ch_idx}",
            )

        col_curve_all, row_mean_all = gain_means.get("all", (None, None))
        if row_mean_all is not None:
            plot_block_lift_summary(context_gain_all, filename, i, images_dir, title_prefix="All")
            plot_mean_scatter(row_mean_all, best_loss_all, filename, i, images_dir, tag="lift_all")
        for ch_idx in range(context_gain_channels.shape[0]):
            means = gain_means.get(f"ch{ch_idx}")
            if means is None:
                continue
            _, row_mean_ch = means
            best_loss_ch = best_loss_channels[ch_idx] if ch_idx < best_loss_channels.shape[0] else None
            if best_loss_ch is not None:
                plot_mean_scatter(row_mean_ch, best_loss_ch, filename, i, images_dir, tag=f"lift_ch{ch_idx}")
        plot_heatmap(
            dt_np,
            title='Δt Heatmap – gap between blocks',
            cbar_label='Time between blocks',
            tag='dt',
            labels_true_np=labels_true_np,
            labels_pred_np=labels_pred_np,
            note='Rows=start_block, Cols=last_block.',
            cmap_name='viridis',
        )
        plot_heatmap(
            lt_np,
            title='ℓt Heatmap – time within blocks',
            cbar_label='Time within blocks',
            tag='lt',
            labels_true_np=labels_true_np,
            labels_pred_np=labels_pred_np,
            note='Rows=start_block, Cols=last_block.',
            cmap_name='viridis',
        )

        def plot_line_summaries(
            all_np, x_dt_vec, x_lt_vec, filename: str, index: int, images_dir: str = "images", title_prefix: str = ""
        ):
            from matplotlib.gridspec import GridSpec

            rows, cols = all_np.shape
            x = np.arange(cols)

            # Prepare vectors filled with NaN. Matplotlib will skip NaNs.
            min_loss = compute_best_loss_curve(all_np)
            maxctx_loss = np.full(cols, np.nan, dtype=float)  # start = last+1
            len10_loss = np.full(cols, np.nan, dtype=float)  # start = last-10

            # Column-wise computation for the fixed context baselines
            for last in range(cols):
                # largest context: start_block = last + 1
                sb = last + 1
                if 0 <= sb < rows:
                    val = all_np[sb, last]
                    if np.isfinite(val):
                        maxctx_loss[last] = val

                # fixed context length 10: start_block = last - 10
                sb2 = last - 10
                if 0 <= sb2 < rows:
                    val2 = all_np[sb2, last]
                    if np.isfinite(val2):
                        len10_loss[last] = val2

            # Align raw x_dt/x_lt to last_block domain
            # We only plot up to the number of columns available in the loss matrix.
            max_len = min(cols, len(x_dt_vec), len(x_lt_vec))
            x_axis = np.arange(max_len)
            dt_plot = np.array(x_dt_vec[:max_len], dtype=float)
            lt_plot = np.array(x_lt_vec[:max_len], dtype=float)

            # === Figure layout ===
            fig = plt.figure(figsize=(14, 8))
            gs = GridSpec(2, 1, height_ratios=[3.0, 1.6], hspace=0.25)

            # Top panel: three loss curves
            ax_top = fig.add_subplot(gs[0, 0])
            ax_top.set_title(f"{title_prefix}Loss vs last_block – Index {index}, File: {filename}")
            ax_top.plot(
                x,
                min_loss,
                label="Min loss vs last_block (best context)",
                marker='o',
                markersize=3,
                linewidth=1.8,
            )
            ax_top.plot(
                x,
                maxctx_loss,
                label="Loss @ start=last_block+1 (full circle context)",
                marker='s',
                markersize=3,
                linewidth=1.6,
            )
            ax_top.plot(
                x,
                len10_loss,
                label="Loss @ start=last_block-10 (fixed 10 blocks of context)",
                marker='^',
                markersize=3,
                linewidth=1.4,
            )
            ax_top.set_xlabel("last_block (end index)")
            ax_top.set_ylabel("Loss (MSE)")
            ax_top.set_title(
                f"{title_prefix}Loss vs last_block – Index {index}, File: {filename}\n"
                "Each curve shows how loss changes as more context is added before predicting `last_block` "
                "(min: best start; start=last+1: entire circle; start=last-10: fixed-length context)."
            )
            if cols > 0:
                if cols <= 10:
                    xticks = np.array([0])
                else:
                    xticks = np.arange(0, cols, 10)
                    if xticks.size == 0:
                        xticks = np.array([0])
                ax_top.set_xticks(xticks)
            ax_top.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
            ax_top.legend(loc="best")

            # Bottom panel: Δt and ℓt on twin y-axes (both vs last_block)
            ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
            (line_dt,) = ax_bot.plot(x_axis, dt_plot, color='tab:blue', label="Δt (gap)")
            ax_bot.set_ylabel("Δt (gap between blocks)")
            ax_bot.set_xlabel("last_block (block being predicted)")
            ax_bot.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

            ax_bot_r = ax_bot.twinx()
            (line_lt,) = ax_bot_r.plot(x_axis, lt_plot, color='tab:orange', label="ℓt (duration)")
            ax_bot_r.set_ylabel("ℓt (duration of block)")

            # Merge legends for bottom panel
            lines_left, labels_left = ax_bot.get_legend_handles_labels()
            lines_right, labels_right = ax_bot_r.get_legend_handles_labels()
            ax_bot.legend(lines_left + lines_right, labels_left + labels_right, loc="best")

            # Save
            out_path = os.path.join(images_dir, f"lines_{title_prefix.lower().strip()}_{index}_{filename}.png")
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {out_path}")

        plot_line_summaries(recon_np, x_dt_np, x_lt_np, filename, i, images_dir, title_prefix="Recon ")

    # If a single index is specified, only process that file; otherwise process all
    if args.index >= 0:
        if args.index >= len(dataset):
            raise ValueError(f"Index {args.index} out of range. Dataset has {len(dataset)} files.")
        process_and_plot(args.index)
    else:
        print(f"\nProcessing all {len(dataset)} files in the validation dataset...")
        for i in range(len(dataset)):
            process_and_plot(i)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
