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
    x, x_i, x_l, N, filename = dataset[index]
    x = x.unsqueeze(0).float().to(device)
    x_i = x_i.unsqueeze(0).to(device)
    N = torch.tensor([N], dtype=torch.long, device=device)
    x, x_i = model.compactify_data(x.clone(), x_i.clone(), N.clone())
    return x, x_i, N, filename


def compute_loss(model, x, x_i, N, start_block, last_block, n_blocks, isolate_block):
    """
    Compute masked MSE loss for a given block configuration.
    Returns: (loss, xs, x_is, bool_mask, pred, mblock, indices)
    """
    try:
        if start_block == last_block:
            return torch.tensor(float('nan'), device=x.device), None, None, None, None, None, None

        if isolate_block:
            indices = [start_block, last_block]
        else:
            if start_block < last_block:
                indices = list(range(start_block, min(start_block + n_blocks, last_block))) + [last_block]
            else:
                indices = list(range(start_block, min(start_block + n_blocks, N.max().item()))) + [last_block]

        xs, x_is = model.sample_data_indices(x.clone(), x_i.clone(), N.clone(), indices)
        mblock = [len(indices) - 1]
        h, idx_restore, bool_mask, bool_pad, T = model.forward_encoder(xs, x_is, mblock=mblock)
        pred = model.forward_decoder(h, idx_restore, T, bool_pad=bool_pad, attend_to_padded=False)
        loss = model.loss_mse(xs, pred, bool_mask)
        return loss, xs, x_is, bool_mask, pred, mblock, indices
    except RuntimeError as e:
        msg = str(e).lower()
        if ("out of memory" in msg or "cuda" in msg) and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return torch.tensor(float('nan'), device=x.device), None, None, None, None, None, None
        else:
            raise


def save_reconstruction(
    model, x, x_i, N, start_block, last_block, n_blocks, isolate_block, *, filename, index, images_dir='images'
):
    """
    Save a side-by-side reconstruction visualization for a given block configuration.
    Returns: (output_path, loss)
    """
    loss, xs, x_is, bool_mask, pred, mblock, indices = compute_loss(
        model, x, x_i, N, start_block, last_block, n_blocks, isolate_block
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
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True, sharey=True)
    cmap = plt.get_cmap('magma')
    im0 = axs[0].imshow(xs_cpu, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', aspect='auto', interpolation='nearest')
    axs[0].set_title('Input')
    im1 = axs[1].imshow(
        x_rec_cpu, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', aspect='auto', interpolation='nearest'
    )
    axs[1].set_title('Reconstruction')
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
            axs[0].axvspan(start - 0.5, i - 0.5, alpha=0.25, color='yellow')
            in_span = False
    if in_span:
        axs[0].axvspan(start - 0.5, len(col_mask) - 0.5, alpha=0.25, color='yellow')
    # Difference heatmap (reconstruction error) only within masked region
    diff = x_rec_cpu - xs_cpu
    # Broadcast column mask to HxW and mask outside region
    col_mask_bool = col_mask.astype(bool)
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
    im2 = axs[2].imshow(
        diff_ma, cmap=cmap_diff, vmin=-max_abs, vmax=max_abs, origin='lower', aspect='auto', interpolation='nearest'
    )
    axs[2].set_title('Reconstruction Error (masked region only)')
    # Draw separators
    _draw_separators(axs[0], x_is[0].detach().cpu().numpy())
    _draw_separators(axs[1], x_is[0].detach().cpu().numpy())
    _draw_separators(axs[2], x_is[0].detach().cpu().numpy())
    # Add colorbars for each panel
    cbar0 = fig.colorbar(im0, ax=axs[0], fraction=0.025, pad=0.02)
    cbar0.set_label('Amplitude', rotation=270, labelpad=12)
    cbar1 = fig.colorbar(im1, ax=axs[1], fraction=0.025, pad=0.02)
    cbar1.set_label('Amplitude', rotation=270, labelpad=12)
    cbar2 = fig.colorbar(im2, ax=axs[2], fraction=0.025, pad=0.02)
    cbar2.set_label('Diff (recon - input)', rotation=270, labelpad=12)
    # Detailed title
    indices_str = str(indices)
    mblock_str = mblock[0] if mblock is not None and len(mblock) > 0 else 'N/A'
    loss_str = f"{loss.item():.6f}" if loss is not None and torch.isfinite(loss) else "NaN"
    title = (
        f"{filename}, idx={index}, start={start_block}, last={last_block}, n_blocks={n_blocks}, "
        f"isolate_block={isolate_block}, indices={indices_str}, mblock={mblock_str}, loss={loss_str}"
    )
    fig.suptitle(title, fontsize=11)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(images_dir, exist_ok=True)
    out_path = os.path.join(
        images_dir, f"reconstruction_idx{index}_start{start_block}_last{last_block}_iso{int(isolate_block)}.png"
    )
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_path, loss


def process_file(model, dataset, index, device):
    x, x_i, x_l, N, filename = dataset[index]

    x = x.unsqueeze(0).float().to(device)
    x_i = x_i.unsqueeze(0).to(device)
    N = torch.tensor([N], dtype=torch.long, device=device)

    x, x_i = model.compactify_data(x.clone(), x_i.clone(), N.clone())
    # Ensure chirp labels align in length with chirp boundaries (trim trailing padding labels)
    if isinstance(x_l, torch.Tensor):
        x_l = x_l[: x_i.shape[1]]
    else:
        x_l = torch.as_tensor(x_l)[: x_i.shape[1]]

    print(f"  Filename: {filename}")
    print(f"  Spectrogram shape: {x.shape}")
    print(f"  Chirp intervals shape: {x_i.shape}")
    print(f"  Chirp labels shape: {x_l.shape}")
    print(f"  Number of valid chirps: {N.item()}")

    def compute_losses(x, x_i, N, isolate_block=False):
        n_blocks = 8
        n_valid_chirps = N.max().item()
        # Fill with NaNs so missing entries don't bias averages
        losses = torch.full((n_valid_chirps, n_valid_chirps), float('nan'), device=device)
        print(f"\nComputing losses for {losses.numel()} (rows × starts)...")
        # Compute an accurate total for the progress bar
        with tqdm(total=((n_valid_chirps - 1) ^ 2), desc="Computing losses") as pbar:
            for last_block in range(1, n_valid_chirps):
                for start_block in range(0, n_valid_chirps - 1):
                    with torch.no_grad():
                        loss, *_ = compute_loss(model, x, x_i, N, start_block, last_block, n_blocks, isolate_block)
                    val = float('nan') if torch.isnan(loss) else loss.item()
                    losses[start_block, last_block] = val
                    pbar.update(1)
        return losses

    isolated_losses = compute_losses(x, x_i, N, isolate_block=True)
    all_losses = compute_losses(x, x_i, N, isolate_block=False)

    return isolated_losses, all_losses, filename, x_l


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
        print(
            f"Params: start_block={args.start_block}, last_block={args.last_block}, isolate_block={args.isolate_block}"
        )

        def run_reconstruction(i: int):
            print(f"\nLoading sample at index {i}")
            x, x_i, N, filename = prepare_sample(model, dataset, i, device)
            n_blocks = min(8, int(args.last_block - args.start_block))
            out_path, loss = save_reconstruction(
                model,
                x,
                x_i,
                N,
                args.start_block,
                args.last_block,
                n_blocks,
                args.isolate_block,
                filename=filename,
                index=i,
                images_dir=images_dir,
            )
            print(f"Saved reconstruction → {out_path}")

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
        isolated_losses, all_losses, filename, x_l = process_file(model, dataset, i, device)

        iso_np = isolated_losses.detach().cpu().numpy()
        all_np = all_losses.detach().cpu().numpy()
        labels_np = x_l.detach().cpu().numpy().astype(int).reshape(-1)

        # Diagnostics: summarize chirp labels
        uniq, counts = np.unique(labels_np, return_counts=True)
        print("chirp_labels summary:")
        for u, c in zip(uniq.tolist(), counts.tolist()):
            print(f"  label {u}: {c}")

        def plot_heatmap(loss_mat_np, tag: str, labels_np):
            fig_hm, ax_hm = plt.subplots(figsize=(12, 8))
            loss_ma = np.ma.masked_invalid(loss_mat_np)
            # Colormap: low loss = green, high loss = red; NaNs = black
            cmap = plt.get_cmap('RdYlGn_r').copy()
            cmap.set_bad(color='black')
            # Scale to finite data range
            vmin = float(np.nanmin(loss_ma)) if np.isfinite(np.nanmin(loss_ma)) else 0.0
            vmax = float(np.nanmax(loss_ma)) if np.isfinite(np.nanmax(loss_ma)) else 1.0
            if vmax <= vmin:
                vmax = vmin + 1.0
            im_hm = ax_hm.imshow(loss_ma, aspect='auto', cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
            cbar_hm = plt.colorbar(im_hm, ax=ax_hm)
            cbar_hm.set_label('Loss (MSE)', rotation=270, labelpad=20, fontsize=12)
            ax_hm.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            mode_label = (
                'Isolated Block Mode (isolate_block=True)'
                if tag == 'isolated'
                else 'All Blocks Mode (isolate_block=False)'
            )
            ax_hm.set_title(f'Loss (MSE) Heatmap – {mode_label}\nIndex {i}, File: {filename}', fontsize=14, pad=20)
            ax_hm.annotate(
                'NaN = black; low = green; high = red. Sparse near diagonal due to windowing.',
                xy=(0.99, 0.01),
                xycoords='axes fraction',
                fontsize=8,
                ha='right',
                va='bottom',
            )

            # === Overlay: for each x (column), mark the y with the smallest finite loss ===
            arr = np.array(loss_mat_np, dtype=float)
            finite_mask = np.isfinite(arr)
            # Columns that have at least one finite value
            cols = np.where(finite_mask.any(axis=0))[0]
            if cols.size > 0:
                # Replace non-finite with +inf so argmin ignores them
                arr_inf = arr.copy()
                arr_inf[~finite_mask] = np.inf
                ys = np.argmin(arr_inf[:, cols], axis=0)
                # Compute marker size ~ size of one heatmap cell
                fig = ax_hm.figure
                bbox = ax_hm.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                # cell sizes (in inches)
                cell_w_in = (bbox.width) / max(1, arr.shape[1])
                cell_h_in = (bbox.height) / max(1, arr.shape[0])
                # diameter ~ 1.2 of the smaller cell dimension (slightly larger than a cell)
                diam_in = 2.0 * min(cell_w_in, cell_h_in)
                # convert diameter (inches) to points, then to area (points^2)
                diam_pt = diam_in * 72.0
                s = diam_pt**2
                ax_hm.scatter(
                    cols,
                    ys,
                    s=s,
                    c="#ff1493",  # brighter pink
                    marker='o',
                    edgecolors='white',
                    linewidths=0.4,
                    zorder=7,
                )

            # Add chirp label strips along axes using shared axes for pixel-perfect alignment
            Hh, Wh = loss_mat_np.shape
            lbl_x = labels_np[:Wh]
            lbl_y = labels_np[:Hh]
            # Mask any values not in {0,1}
            lbl_x = lbl_x.astype(float)
            lbl_x[(lbl_x != 0) & (lbl_x != 1)] = np.nan
            lbl_y = lbl_y.astype(float)
            lbl_y[(lbl_y != 0) & (lbl_y != 1)] = np.nan

            cmap_lbl = ListedColormap(["#add8e6", "#00008b"]).copy()
            cmap_lbl.set_bad(color='black')

            divider = make_axes_locatable(ax_hm)
            # Increase pad to leave room for axis labels and ticks
            ax_strip_x = divider.append_axes("bottom", size="3%", pad=0.3, sharex=ax_hm)
            ax_strip_y = divider.append_axes("left", size="3%", pad=0.45, sharey=ax_hm)

            # Bottom strip (x-axis): 1 × Wh
            strip_x = np.ma.masked_invalid(lbl_x[np.newaxis, :])
            ax_strip_x.imshow(strip_x, aspect='auto', cmap=cmap_lbl, interpolation='nearest', origin='lower')
            ax_strip_x.set_xlim(ax_hm.get_xlim())
            ax_strip_x.axis('off')

            # Left strip (y-axis): Hh × 1 (origin lower to match heatmap orientation)
            strip_y = np.ma.masked_invalid(lbl_y[:, np.newaxis])
            ax_strip_y.imshow(strip_y, aspect='auto', cmap=cmap_lbl, interpolation='nearest', origin='lower')
            ax_strip_y.set_ylim(ax_hm.get_ylim())
            ax_strip_y.axis('off')

            plt.tight_layout()
            # Add a small extra margin to leave room for axis labels and ticks
            plt.subplots_adjust(bottom=0.12, left=0.10)
            loss_hm_out = os.path.join(images_dir, f"loss_heatmap_{tag}_{i}_{filename}.png")
            fig_hm.savefig(loss_hm_out, dpi=300, bbox_inches='tight')
            plt.close(fig_hm)
            print(f"Saved: {loss_hm_out}")

        # Heatmaps use the full sparse matrices (with chirp label strips)
        plot_heatmap(iso_np, tag='isolated', labels_np=labels_np)
        plot_heatmap(all_np, tag='allblocks', labels_np=labels_np)

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
