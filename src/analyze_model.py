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
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from utils import load_model_from_checkpoint
from data_loader import SpectogramDataset

# For pixel-perfect axis-aligned strips
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
    return parser.parse_args()


def prepare_sample(model, dataset, index, device):
    x, x_i, x_l, x_f, N, filename = dataset[index]
    x = x.unsqueeze(0).float().to(device)
    x_i = x_i.unsqueeze(0).to(device)
    N = torch.tensor([N], dtype=torch.long, device=device)

    # Compute chirp start/end/dt/lt
    starts = x_i[0, : N.item(), 0]
    ends = x_i[0, : N.item(), 1]
    x_dt = torch.cat([torch.tensor([0.0], device=device), starts[1:] - ends[:-1]])
    x_lt = ends - starts

    x, x_i = model.compactify_data(x.clone(), x_i.clone(), N.clone())

    # Ensure chirp labels align in length with chirp boundaries (trim trailing padding labels)
    if isinstance(x_l, torch.Tensor):
        x_l = x_l[: x_i.shape[1]]
    else:
        x_l = torch.as_tensor(x_l)[: x_i.shape[1]]

    # Ensure chirp_feats align in length with chirp boundaries (trim trailing padding)
    if isinstance(x_f, torch.Tensor):
        x_f = x_f[: x_i.shape[1]]
    else:
        x_f = torch.as_tensor(x_f)[: x_i.shape[1]]

    return x, x_i, x_l, x_f, x_dt, x_lt, N, filename


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
        return torch.tensor(float('nan'), device=x.device), dt, lt, None, None, None, None, None, None

    try:
        xs, x_is = model.sample_data_indices(x.clone(), x_i.clone(), N.clone(), indices)
        h, idx_restore, bool_mask, T = model.forward_encoder(xs, x_is, mblock=mblock)
        pred = model.forward_decoder(h, idx_restore, T)
        loss = model.loss_mse(xs, pred, bool_mask)
        return loss, dt, lt, xs, x_is, bool_mask, pred, mblock, indices
    except RuntimeError as e:
        msg = str(e).lower()
        if ("out of memory" in msg or "cuda" in msg) and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return torch.tensor(float('nan'), device=x.device), dt, lt, None, None, None, None, None, None
        else:
            raise


def save_reconstruction(model, x, x_i, N, last_block, x_dt, x_lt, *, filename, index, images_dir='images'):
    """
    Save a side-by-side reconstruction visualization for a given block configuration.
    Returns: (output_path, loss)
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
            # Pick the row index (start_block) with the lowest finite loss for this column
            finite_mask = np.isfinite(col)
            if finite_mask.any():
                best_start = int(np.nanargmin(np.where(finite_mask, col, np.nan)))
    except Exception as e:
        # If anything goes wrong with cache logic, silently fall back
        best_start = default_start

    loss, dt, lt, xs, x_is, bool_mask, pred, mblock, indices = compute_loss(
        model, x, x_i, N, start_block=best_start, last_block=last_block, x_dt=x_dt, x_lt=x_lt
    )
    if xs is None or pred is None or bool_mask is None:
        return None, loss

    def _reconstruct(xs, pred, bool_mask, patch_size):
        """
        xs:   (B, 1, H, W) input spectrogram window
        pred: (B, T, P) decoder output in the same normalized space as loss
        bool_mask: (B, T) which tokens were masked
        Returns: (B, 1, H, W) image where masked patches are replaced with **denormalized** predictions.
        """
        B, C, H, W = xs.shape
        unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
        fold = torch.nn.Fold(output_size=(H, W), kernel_size=patch_size, stride=patch_size)

        # Convert input into patches to compute per-token moments for denormalization
        x_patches = unfold(xs).transpose(1, 2)  # (B, T, P)

        # Per-token mean/std of target patches (match pretrain.py logic)
        target_mean = x_patches.mean(dim=-1, keepdim=True)
        target_std = x_patches.std(dim=-1, keepdim=True)

        # Denormalize predictions: pred_denorm = pred * std + mean
        pred_denorm = pred * (target_std + 1e-6) + target_mean

        # Replace only masked tokens with denormalized predictions
        out_patches = x_patches.clone()
        out_patches[bool_mask] = pred_denorm[bool_mask]

        # Fold back to image
        out_patches = out_patches.transpose(1, 2)  # (B, P, T)
        x_rec = fold(out_patches)  # (B, 1, H, W)
        return x_rec

    def _column_mask_2d(bool_mask, H, W, patch_size):
        """
        Convert a token-level mask (B, T) into a per-column mask (B, W) in pixel space.
        T = H' * W', where H' = H // patch_height, W' = W // patch_width.
        We reduce across token rows (H') to find columns masked anywhere in that column.
        """
        ph, pw = patch_size
        Htok = max(1, H // ph)
        Wtok = max(1, W // pw)
        # bool_mask is (B, T) where T == Htok * Wtok; reshape accordingly
        return bool_mask.view(-1, Htok, Wtok).any(dim=1)

    def _draw_separators(ax, x_is_b):
        # x_is_b: (K, 2)
        K = x_is_b.shape[0]
        for k in range(K - 1):
            pos = int(x_is_b[k, 1])
            ax.axvline(pos - 0.5, linewidth=0.8, alpha=0.8, color='cyan')

    # Assume batch size 1
    xs_cpu = xs[0, 0].detach().cpu().numpy()
    H, W = xs_cpu.shape
    x_rec = _reconstruct(xs, pred, bool_mask, model.patch_size)
    x_rec_cpu = x_rec[0, 0].detach().cpu().numpy()
    vmin = min(np.nanmin(xs_cpu), np.nanmin(x_rec_cpu))
    vmax = max(np.nanmax(xs_cpu), np.nanmax(x_rec_cpu))
    from matplotlib.gridspec import GridSpec

    # Figure with right-hand zoom panes
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
    ax_in_full.set_title('Input')
    im1 = ax_rec_full.imshow(
        x_rec_cpu, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', aspect='auto', interpolation='nearest'
    )
    ax_rec_full.set_title('Reconstruction')
    # Overlay masked region on panel 0 only
    col_mask = _column_mask_2d(bool_mask, H, W, model.patch_size)[0].cpu().numpy()
    # Find contiguous True spans in col_mask and draw axvspan
    in_span = False
    start = 0
    for i in range(len(col_mask)):
        if col_mask[i] and not in_span:
            start = i
            in_span = True
        elif not col_mask[i] and in_span:
            ax_in_full.axvspan(start - 0.5, i - 0.5, alpha=0.25, color='yellow')
            in_span = False
    if in_span:
        ax_in_full.axvspan(start - 0.5, len(col_mask) - 0.5, alpha=0.25, color='yellow')

    # Determine the masked span (contiguous True region). If multiple spans, merge to min..max
    col_mask_bool = col_mask.astype(bool)
    true_idxs = np.where(col_mask_bool)[0]
    zoom_has_span = true_idxs.size > 0
    if zoom_has_span:
        span_start = int(true_idxs.min())
        span_end = int(true_idxs.max() + 1)  # exclusive
    else:
        span_start, span_end = 0, 1  # harmless fallback to 1 column

    # Difference heatmap (reconstruction error) only within masked region
    diff = x_rec_cpu - xs_cpu
    mask_hw = ~np.broadcast_to(col_mask_bool, (H, W))
    diff_ma = np.ma.masked_array(diff, mask=mask_hw)
    # Symmetric color scale based on masked region
    if col_mask_bool.any():
        max_abs = np.nanmax(np.abs(diff[:, col_mask_bool]))
        if not np.isfinite(max_abs) or max_abs == 0:
            max_abs = 1e-6
    else:
        max_abs = np.nanmax(np.abs(diff)) if np.isfinite(np.nanmax(np.abs(diff))) else 1.0
    cmap_diff = plt.get_cmap('coolwarm').copy()
    cmap_diff.set_bad(color='black')  # outside masked region
    im2 = ax_err_full.imshow(
        diff_ma, cmap=cmap_diff, vmin=-max_abs, vmax=max_abs, origin='lower', aspect='auto', interpolation='nearest'
    )
    ax_err_full.set_title('Reconstruction Error (masked region only)')

    # === Right-hand zoom panels: crop to the masked span, keep exact masked width ===
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

        # Tighten x-lims to show exact masked width without padding
        ax_in_zoom.set_xlim(0, xs_crop.shape[1])
        ax_rec_zoom.set_xlim(0, xrec_crop.shape[1])
        ax_err_zoom.set_xlim(0, diff_crop.shape[1])

        # Hide y tick labels on zoom panes; keep y shared with left for alignment
        for axz in (ax_in_zoom, ax_rec_zoom, ax_err_zoom):
            axz.tick_params(axis='y', which='both', left=False, labelleft=False)
    else:
        # No masked span; hide zoom axes content
        for axz in (ax_in_zoom, ax_rec_zoom, ax_err_zoom):
            axz.axis('off')

    # Draw separators
    _draw_separators(ax_in_full, x_is[0].detach().cpu().numpy())
    _draw_separators(ax_rec_full, x_is[0].detach().cpu().numpy())
    _draw_separators(ax_err_full, x_is[0].detach().cpu().numpy())
    # Add colorbars for each panel
    cbar0 = fig.colorbar(im0, ax=ax_in_full, fraction=0.025, pad=0.02)
    cbar0.set_label('Amplitude', rotation=270, labelpad=12)
    cbar1 = fig.colorbar(im1, ax=ax_rec_full, fraction=0.025, pad=0.02)
    cbar1.set_label('Amplitude', rotation=270, labelpad=12)
    cbar2 = fig.colorbar(im2, ax=ax_err_full, fraction=0.025, pad=0.02)
    cbar2.set_label('Diff (recon - input)', rotation=270, labelpad=12)
    # Detailed title
    mblock_str = mblock[0] if mblock is not None and len(mblock) > 0 else 'N/A'
    loss_str = f"{loss.item():.6f}" if loss is not None and torch.isfinite(loss) else "NaN"
    title = f"{filename}, idx={index}, last={last_block}, mblock={mblock_str}, loss={loss_str}"
    fig.suptitle(title, fontsize=11)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(images_dir, exist_ok=True)
    out_path = os.path.join(images_dir, f"reconstruction_idx{index}_last{last_block}.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_path, loss


def process_file(model, dataset, index, device):
    x, x_i, x_l, x_f, x_dt, x_lt, N, filename = prepare_sample(model, dataset, index, device)

    print(f"  Filename: {filename}")
    print(f"  Spectrogram shape: {x.shape}")
    print(f"  Chirp intervals shape: {x_i.shape}")
    print(f"  Chirp labels shape: {x_l.shape}")
    print(f"  Chirp feats shape: {x_f.shape}")
    print(f"  Number of valid chirps: {N.item()}")

    def compute_losses(x, x_i, N, x_dt, x_lt):
        n_valid_chirps = N.max().item()
        losses = torch.full((n_valid_chirps, n_valid_chirps), float('nan'), device=device)
        dt_mat = torch.full((n_valid_chirps, n_valid_chirps), float('nan'), device=device)
        lt_mat = torch.full((n_valid_chirps, n_valid_chirps), float('nan'), device=device)
        print(f"\nComputing losses/dt/lt for {losses.numel()} (rows × starts)...")
        with tqdm(total=((n_valid_chirps - 1) * (n_valid_chirps - 1)), desc="Computing losses/dt/lt") as pbar:
            for last_block in range(1, n_valid_chirps):
                for start_block in range(0, n_valid_chirps - 1):
                    with torch.no_grad():
                        loss, dt_val, lt_val, *_ = compute_loss(model, x, x_i, N, start_block, last_block, x_dt, x_lt)
                    losses[start_block, last_block] = float('nan') if torch.isnan(loss) else loss
                    dt_mat[start_block, last_block] = dt_val
                    lt_mat[start_block, last_block] = lt_val
                    pbar.update(1)
        return losses, dt_mat, lt_mat

    cached = load_from_cache(filename)
    if cached is not None:
        print(f"Loaded cached matrices for {filename}")
        all_losses, all_dt, all_lt = cached
    else:
        all_losses, all_dt, all_lt = compute_losses(x, x_i, N, x_dt, x_lt)
        save_to_cache(filename, (all_losses, all_dt, all_lt))

    return all_losses, all_dt, all_lt, filename, x_l, x_f, x_dt, x_lt


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

    # === Reconstruction-only mode branch ===
    reconstruction_mode = (args.start_block is not None) and (args.last_block is not None)
    if reconstruction_mode:
        print("\nReconstruction-only mode enabled.")
        print(f"Params: start_block={args.start_block}, last_block={args.last_block}")

        def run_reconstruction(i: int):
            print(f"\nLoading sample at index {i}")
            x, x_i, x_l, x_f, x_dt, x_lt, N, filename = prepare_sample(model, dataset, i, device)

            for last_block in range(args.start_block, args.last_block + 1):
                print(f"Generating reconstruction for last_block={last_block}")
                out_path, loss = save_reconstruction(
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
                if out_path is not None:
                    print(f"Saved reconstruction → {out_path} (loss={loss})")
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
        all_losses, all_dt, all_lt, filename, x_l, x_f, x_dt, x_lt = process_file(model, dataset, i, device)

        all_np = all_losses.detach().cpu().numpy()
        dt_np = all_dt.detach().cpu().numpy()
        lt_np = all_lt.detach().cpu().numpy()
        labels_np = x_l.detach().cpu().numpy().astype(int).reshape(-1)
        x_dt_np = x_dt.detach().cpu().numpy()
        x_lt_np = x_lt.detach().cpu().numpy()

        # Diagnostics: summarize chirp labels
        uniq, counts = np.unique(labels_np, return_counts=True)
        print("chirp_labels summary:")
        for u, c in zip(uniq.tolist(), counts.tolist()):
            print(f"  label {u}: {c}")

        def plot_heatmap(
            mat_np,
            title: str,
            cbar_label: str,
            tag: str,
            labels_np,
            x_f_np,
            *,
            note: str | None = None,
            cmap_name: str | None = None,
        ):
            fig_hm, ax_hm = plt.subplots(figsize=(12, 8))
            data_ma = np.ma.masked_invalid(mat_np)
            # Choose colormap (default depends on tag)
            if cmap_name is None:
                cmap_name = 'RdYlGn_r' if tag.startswith('loss') else 'viridis'
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad(color='black')
            # Scale to finite data range
            vmin = float(np.nanmin(data_ma)) if np.isfinite(np.nanmin(data_ma)) else 0.0
            vmax = float(np.nanmax(data_ma)) if np.isfinite(np.nanmax(data_ma)) else 1.0
            if vmax <= vmin:
                vmax = vmin + 1.0
            im_hm = ax_hm.imshow(data_ma, aspect='auto', cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
            cbar_hm = plt.colorbar(im_hm, ax=ax_hm)
            cbar_hm.set_label(cbar_label, rotation=270, labelpad=20, fontsize=12)

            # Axes labels reflect matrix semantics: rows=start_block, cols=last_block
            ax_hm.set_xlabel('last_block (end index)')
            ax_hm.set_ylabel('start_block (start index)')

            ax_hm.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax_hm.set_title(f'{title}\nIndex {i}, File: {filename}', fontsize=14, pad=20)

            if note:
                ax_hm.annotate(note, xy=(0.99, 0.01), xycoords='axes fraction', fontsize=8, ha='right', va='bottom')

            # === Overlay: for each x (column), mark the y with the smallest finite value ===
            arr = np.array(mat_np, dtype=float)
            finite_mask = np.isfinite(arr)
            cols = np.where(finite_mask.any(axis=0))[0]
            if cols.size > 0:
                arr_inf = arr.copy()
                arr_inf[~finite_mask] = np.inf
                ys = np.argmin(arr_inf[:, cols], axis=0)
                fig = ax_hm.figure
                bbox = ax_hm.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                cell_w_in = (bbox.width) / max(1, arr.shape[1])
                cell_h_in = (bbox.height) / max(1, arr.shape[0])
                diam_in = 2.0 * min(cell_w_in, cell_h_in)
                diam_pt = diam_in * 72.0
                s = diam_pt**2
                ax_hm.scatter(cols, ys, s=s, c="#ff1493", marker='o', edgecolors='white', linewidths=0.4, zorder=7)

            # === Chirp label strips (reused for all heatmaps) ===
            Hh, Wh = mat_np.shape
            if isinstance(x_f_np, torch.Tensor):
                x_f_np = x_f_np.detach().cpu().numpy()
            labels_adj = labels_np.copy()
            L_call = x_f_np[:, 0, 3].astype(float)
            R_call = x_f_np[:, 1, 3].astype(float)
            for idx2 in range(min(len(labels_adj), x_f_np.shape[0])):
                if labels_adj[idx2] == 0 and not (L_call[idx2] > 0):
                    labels_adj[idx2] = -1
                elif labels_adj[idx2] == 1 and not (R_call[idx2] > 0):
                    labels_adj[idx2] = -1

            from matplotlib.colors import BoundaryNorm

            lbl_x = labels_adj[:Wh].astype(int)
            lbl_y = labels_adj[:Hh].astype(int)
            cmap_lbl = ListedColormap(["#ffffff", "#add8e6", "#00008b"]).copy()
            bounds = [-1.5, -0.5, 0.5, 1.5]
            norm_lbl = BoundaryNorm(bounds, cmap_lbl.N)

            divider = make_axes_locatable(ax_hm)
            ax_strip_x = divider.append_axes("bottom", size="3%", pad=0.3, sharex=ax_hm)
            ax_strip_y = divider.append_axes("left", size="3%", pad=0.45, sharey=ax_hm)

            ax_strip_x.imshow(
                lbl_x[np.newaxis, :],
                aspect='auto',
                cmap=cmap_lbl,
                norm=norm_lbl,
                interpolation='nearest',
                origin='lower',
            )
            ax_strip_x.set_xlim(ax_hm.get_xlim())
            ax_strip_x.axis('off')

            ax_strip_y.imshow(
                lbl_y[:, np.newaxis],
                aspect='auto',
                cmap=cmap_lbl,
                norm=norm_lbl,
                interpolation='nearest',
                origin='lower',
            )
            ax_strip_y.set_ylim(ax_hm.get_ylim())
            ax_strip_y.axis('off')

            # Additional feature strips (reused)
            if isinstance(x_f_np, torch.Tensor):
                x_f_np = x_f_np.detach().cpu().numpy()
            var_names = ["x", "y", "z"]
            var_indices = [0, 1, 2]

            def _imshow_bottom_strip(values_1d, label_text):
                ax_b = divider.append_axes("bottom", size="3%", pad=0.12, sharex=ax_hm)
                strip = np.ma.masked_invalid(values_1d[np.newaxis, :])
                ax_b.imshow(strip, aspect='auto', cmap=plt.get_cmap('viridis'), interpolation='nearest', origin='lower')
                ax_b.set_xlim(ax_hm.get_xlim())
                ax_b.axis('off')
                ax_b.text(0.0, 0.5, label_text, transform=ax_b.transAxes, fontsize=6, ha='left', va='center')

            def _imshow_left_strip(values_1d, label_text):
                ax_lf = divider.append_axes("left", size="3%", pad=0.18, sharey=ax_hm)
                strip = np.ma.masked_invalid(values_1d[:, np.newaxis])
                ax_lf.imshow(
                    strip, aspect='auto', cmap=plt.get_cmap('viridis'), interpolation='nearest', origin='lower'
                )
                ax_lf.set_ylim(ax_hm.get_ylim())
                ax_lf.axis('off')
                ax_lf.text(
                    0.5, 1.0, label_text, transform=ax_lf.transAxes, fontsize=6, ha='center', va='top', rotation=90
                )

            for bird_idx, bird_tag in enumerate(["L", "R"]):
                for vi, vname in zip(var_indices, var_names):
                    vec = x_f_np[:, bird_idx, vi].astype(float)
                    _imshow_bottom_strip(vec[:Wh], f"{bird_tag} {vname}")
                    _imshow_left_strip(vec[:Hh], f"{bird_tag} {vname}")

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.12, left=0.10)
            out_path = os.path.join(images_dir, f"heatmap_{tag}_{i}_{filename}.png")
            fig_hm.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig_hm)
            print(f"Saved: {out_path}")

        # Heatmaps use the full sparse matrices (with chirp label strips)
        plot_heatmap(
            all_np,
            title='Loss (MSE) Heatmap – All Blocks Mode',
            cbar_label='Loss (MSE)',
            tag='loss_allblocks',
            labels_np=labels_np,
            x_f_np=x_f,
            note='NaN = black; low = green; high = red. Sparse near diagonal due to windowing.',
            cmap_name='RdYlGn_r',
        )
        plot_heatmap(
            dt_np,
            title='Δt Heatmap – gap between blocks (time between masked start and last block)',
            cbar_label='Time between blocks',
            tag='dt',
            labels_np=labels_np,
            x_f_np=x_f,
            note='Δt: amount of time between blocks (gaps). Rows = start_block, Cols = last_block.',
            cmap_name='viridis',
        )
        plot_heatmap(
            lt_np,
            title='ℓt Heatmap – time within blocks (duration of included blocks)',
            cbar_label='Time within blocks',
            tag='lt',
            labels_np=labels_np,
            x_f_np=x_f,
            note='ℓt: amount of time in blocks (durations). Rows = start_block, Cols = last_block.',
            cmap_name='viridis',
        )

        def plot_line_summaries(all_np, x_dt_vec, x_lt_vec, filename: str, index: int, images_dir: str = "images"):
            """
            Create a single figure with two stacked panels:
              (Top) 3 line plots vs last_block (x-axis):
                1) lowest achieved loss across start_block for each last_block
                2) loss for the largest amount of context: start_block = last_block + 1
                3) loss for fixed context length 10: start_block = last_block - 10
              (Bottom) Δt and ℓt vs last_block on twin y-axes, plotting raw x_dt and x_lt by index.

            Any out-of-bounds or invalid entries are left as NaN so no point is shown.
            """
            from matplotlib.gridspec import GridSpec

            rows, cols = all_np.shape
            x = np.arange(cols)

            # Prepare vectors filled with NaN. Matplotlib will skip NaNs.
            min_loss = np.full(cols, np.nan, dtype=float)
            maxctx_loss = np.full(cols, np.nan, dtype=float)  # start = last+1
            len10_loss = np.full(cols, np.nan, dtype=float)  # start = last-10

            # Column-wise computation
            for last in range(cols):
                col = all_np[:, last]
                # (1) lowest achieved loss for this last_block
                if np.isfinite(col).any():
                    min_loss[last] = np.nanmin(col)

                # (2) largest context: start_block = last + 1
                sb = last + 1
                if 0 <= sb < rows:
                    val = all_np[sb, last]
                    if np.isfinite(val):
                        maxctx_loss[last] = val

                # (3) fixed context length 10: start_block = last - 10
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
            ax_top.plot(x, min_loss, label="Min loss vs last_block")
            ax_top.plot(x, maxctx_loss, label="Loss @ start=last_block+1 (max context)")
            ax_top.plot(x, len10_loss, label="Loss @ start=last_block-10 (context len 10)")
            ax_top.set_xlabel("last_block (end index)")
            ax_top.set_ylabel("Loss (MSE)")
            ax_top.set_title(f"Loss vs last_block – Index {index}, File: {filename}")
            ax_top.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
            ax_top.legend(loc="best")

            # Bottom panel: Δt and ℓt on twin y-axes (both vs last_block)
            ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
            (line_dt,) = ax_bot.plot(x_axis, dt_plot, color='tab:blue', label="Δt (gap)")
            ax_bot.set_ylabel("Δt")
            ax_bot.set_xlabel("last_block (end index)")
            ax_bot.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

            ax_bot_r = ax_bot.twinx()
            (line_lt,) = ax_bot_r.plot(x_axis, lt_plot, color='tab:orange', label="ℓt (duration)")
            ax_bot_r.set_ylabel("ℓt")

            # Merge legends for bottom panel
            lines_left, labels_left = ax_bot.get_legend_handles_labels()
            lines_right, labels_right = ax_bot_r.get_legend_handles_labels()
            ax_bot.legend(lines_left + lines_right, labels_left + labels_right, loc="best")

            # Save
            out_path = os.path.join(images_dir, f"summary_lines_{index}_{filename}.png")
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {out_path}")

        # === New: summary line figure (top: 3 loss curves; bottom: Δt & ℓt) ===
        plot_line_summaries(all_np, x_dt_np, x_lt_np, filename, i, images_dir)

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
