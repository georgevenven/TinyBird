#!/usr/bin/env python3
"""
Analyze TinyBird model behavior on spectrogram data.

This script loads a trained TinyBird model and normalized spectrogram data for analysis.
"""

import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import load_model_from_checkpoint
from data_loader import SpectogramDataset


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
    return parser.parse_args()


def process_file(model, dataset, index, device):
    x, x_i, x_l, N, filename = dataset[index]
    x = x.unsqueeze(0).float().to(device)
    x_i = x_i.unsqueeze(0).to(device)
    N = torch.tensor([N], dtype=torch.long, device=device)

    x, x_i = model.compactify_data(x.clone(), x_i.clone(), N.clone())

    print(f"  Filename: {filename}")
    print(f"  Spectrogram shape: {x.shape}")
    print(f"  Chirp intervals shape: {x_i.shape}")
    print(f"  Chirp labels shape: {x_l.shape}")
    print(f"  Number of valid chirps: {N.item()}")

    def mean_column_over_intervals(x: torch.Tensor, xi: torch.Tensor, N: torch.Tensor):
        assert x.dim() == 4, f"x must be (B,C,H,W), got {tuple(x.shape)}"
        assert xi.dim() == 3 and xi.size(-1) == 2, f"xi must be (B,N_max,2), got {tuple(xi.shape)}"

        B, C, H, W = x.shape
        device = x.device

        # Normalize N shape to (B,) long
        if N.dim() == 2 and N.size(1) == 1:
            N = N.view(B)
        N = N.to(device=device, dtype=torch.long)

        # Boolean mask over columns per item: True where the column is inside any interval
        mask = torch.zeros(B, W, dtype=torch.bool, device=device)

        # Clamp interval bounds to [0, W] to be safe
        starts = xi[..., 0].clamp(min=0, max=W)
        ends = xi[..., 1].clamp(min=0, max=W)

        # Fill mask per-batch item for valid intervals
        for b in range(B):
            n_valid = int(N[b].item())
            if n_valid <= 0:
                continue
            s_b = starts[b, :n_valid].to(torch.long)
            e_b = ends[b, :n_valid].to(torch.long)
            # Mark each [s,e) as True
            for s, e in zip(s_b.tolist(), e_b.tolist()):
                if e > s:  # skip empty/invalid
                    mask[b, s:e] = True

        # Count selected columns per item; avoid div-by-zero
        counts = mask.sum(dim=1).clamp_min(1).view(B, 1, 1, 1).to(dtype=x.dtype)

        # Broadcast mask to (B,C,H,W) and compute masked mean across W -> keepdim True for width=1
        mask_bc = mask.view(B, 1, 1, W).to(dtype=x.dtype)
        summed = (x * mask_bc).sum(dim=3, keepdim=True)  # (B,C,H,1)
        mean_x = summed / counts  # (B,C,H,1)

        # For any item with zero selected columns, force zeros (already handled via counts=1 but be explicit)
        zero_items = mask.sum(dim=1) == 0
        if zero_items.any():
            mean_x[zero_items, ...] = 0

        return mean_x

    x_mean = mean_column_over_intervals(x, x_i, N)

    def compute_losses(x, x_i, N, x_mean, isolate_block=False):
        def compute_loss(x, x_i, N, start, x_mean, n_blocks, isolate_block, max_blocks=12):
            windowed_blocks = max_blocks if isolate_block else abs(n_blocks)

            if windowed_blocks <= 1:
                return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

            mblock = windowed_blocks - 1
            if isolate_block:
                iblock = mblock - (n_blocks - 1)
            else:
                iblock = -1

            xs, x_is = model.sample_data(x.clone(), x_i.clone(), N.clone(), n_blocks=windowed_blocks, start=start)
            h, idx_restore, bool_mask, bool_pad, T = model.forward_encoder(xs, x_is, mblock=mblock, iblock=iblock)
            pred = model.forward_decoder(h, idx_restore, T, bool_pad=bool_pad)
            loss = model.loss_mse(xs, pred, bool_mask)
            return loss

        block_max = 12
        n_valid_chirps = N.max().item()
        # Fill with NaNs so missing entries don't bias averages
        losses = torch.full((block_max, n_valid_chirps), float('nan'), device=device)

        print(f"\nComputing losses for {losses.numel()} (rows × starts)...")

        # Compute an accurate total for the progress ba
        with tqdm(total=n_valid_chirps, desc="Computing losses") as pbar:
            for start in range(block_max, n_valid_chirps):
                for n_blocks in range(1, block_max):
                    with torch.no_grad():
                        loss = compute_loss(x, x_i, N, start, x_mean, n_blocks, isolate_block, max_blocks=block_max)
                    losses[n_blocks + block_max, start] = loss.item()
                    pbar.update(1)
        return losses

    losses = compute_losses(x, x_i, N, x_mean, isolate_block=True)
    losses_all_blocks = compute_losses(x, x_i, N, x_mean, isolate_block=False)

    return losses, losses_all_blocks, filename


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

    import os
    import numpy as np

    images_dir = "images"
    os.makedirs(images_dir, exist_ok=True)

    def process_and_plot(i: int):
        print(f"\nLoading sample at index {i}")
        losses_iso, losses_all, filename = process_file(model, dataset, i, device)
        losses_iso_np = losses_iso.cpu().numpy()
        losses_all_np = losses_all.cpu().numpy()

        # Infer rows and baseline dynamically from the matrix shape
        rows = int(losses_iso_np.shape[0])
        # If rows is odd, assume a centered baseline row; else assume 0..K (no negatives)
        if rows % 2 == 1:
            baseline_row = rows // 2
            y_values = list(range(-baseline_row, rows - baseline_row))
        else:
            baseline_row = 0
            y_values = list(range(rows))

        # Helper to plot line summary and heatmap for a given matrix
        def plot_set(loss_mat_np, tag: str):
            # 1) Line plot of average raw loss (MSE) vs n_blocks, excluding baseline row (0 blocks)
            mean_loss = np.nanmean(loss_mat_np, axis=1)
            std_loss = np.nanstd(loss_mat_np, axis=1)

            # Exclude the baseline row only if it exists (rows odd); otherwise plot all rows
            if rows % 2 == 1:
                idxs = [i for i in range(rows) if i != baseline_row]
            else:
                idxs = list(range(rows))
            y_vals_plot = [y_values[i] for i in idxs]
            mean_plot = mean_loss[idxs]
            std_plot = std_loss[idxs]

            fig_line, ax_line = plt.subplots(figsize=(8, 5))
            ax_line.plot(y_vals_plot, mean_plot, marker='o')
            ax_line.fill_between(y_vals_plot, mean_plot - std_plot, mean_plot + std_plot, alpha=0.2)
            ax_line.set_xlabel('n_blocks (index)')
            ax_line.set_ylabel('Average Loss (MSE)')
            ax_line.set_title(f'Average Loss vs n_blocks (index {i}, {tag})\nFile: {filename}')
            ax_line.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            # Add note about missing data
            ax_line.annotate(
                'Note: Some rows/columns may have fewer valid samples due to boundaries.',
                xy=(0.99, 0.01),
                xycoords='axes fraction',
                fontsize=8,
                ha='right',
                va='bottom',
            )
            plt.tight_layout()
            out_line = os.path.join(images_dir, f"avg_loss_vs_nblocks_{tag}_{i}_{filename}.png")
            fig_line.savefig(out_line, dpi=300, bbox_inches='tight')
            plt.close(fig_line)
            print(f"Saved: {out_line}")

            # 2) Heatmap: raw Loss (MSE)
            fig_hm, ax_hm = plt.subplots(figsize=(12, 8))
            loss_ma = np.ma.masked_invalid(loss_mat_np)
            # Set color scale to data range (ignoring NaNs)
            vmin = np.nanmin(loss_ma)
            vmax = np.nanmax(loss_ma)
            if not np.isfinite(vmin):
                vmin = 0.0
            if not np.isfinite(vmax) or vmax == vmin:
                vmax = vmin + 1.0
            im_hm = ax_hm.imshow(loss_ma, aspect='auto', cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
            ax_hm.set_xlabel('Start Position (block index)')
            ax_hm.set_ylabel('n_blocks')
            ax_hm.set_title(f'Loss (MSE) Heatmap (index {i}, {tag})\nFile: {filename}', fontsize=14, pad=20)
            ax_hm.set_yticks(np.arange(rows))
            ax_hm.set_yticklabels([str(v) for v in y_values])
            cbar_hm = plt.colorbar(im_hm, ax=ax_hm)
            cbar_hm.set_label('Loss (MSE)', rotation=270, labelpad=20, fontsize=12)
            ax_hm.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax_hm.annotate(
                'Note: Some rows/columns may have fewer valid samples due to boundaries.',
                xy=(0.99, 0.01),
                xycoords='axes fraction',
                fontsize=8,
                ha='right',
                va='bottom',
            )
            plt.tight_layout()
            loss_hm_out = os.path.join(images_dir, f"loss_heatmap_{tag}_{i}_{filename}.png")
            fig_hm.savefig(loss_hm_out, dpi=300, bbox_inches='tight')
            plt.close(fig_hm)
            print(f"Saved: {loss_hm_out}")

        # Generate both sets
        plot_set(losses_iso_np, tag='isolated')
        plot_set(losses_all_np, tag='allblocks')

        # Line graph of raw loss for rows baseline_row-1 .. baseline_row-5 (interpreted as -1..-5 when baseline exists)
        fig_neg, ax_neg = plt.subplots(figsize=(10, 5))
        for k in range(1, 6):
            row_idx = baseline_row - k
            if 0 <= row_idx < losses_all_np.shape[0]:
                y = losses_all_np[row_idx, :]
                x = np.arange(y.size)
                if np.isnan(y).all():
                    continue
                is_boundary = (k == 1) or (k == 5)
                label = f"{y_values[row_idx]} blocks" if is_boundary else "_nolegend_"
                lw = 1.6 if is_boundary else 0.8
                alpha = 1.0 if is_boundary else 0.35
                marker = 'o' if is_boundary else None
                ax_neg.plot(x, y, marker=marker, linewidth=lw, alpha=alpha, label=label)
        ax_neg.set_xlabel('Start Position (block index)')
        ax_neg.set_ylabel('Loss (MSE)')
        ax_neg.set_title(
            f'Raw Loss vs Start for {"-1..-5" if rows % 2 == 1 else "lower rows"} (index {i}, allblocks)\nFile: {filename}'
        )
        ax_neg.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        if len(ax_neg.lines) > 0:
            ax_neg.legend()
        plt.tight_layout()
        neg_out = os.path.join(images_dir, f"loss_lines_allblocks_neg_{i}_{filename}.png")
        fig_neg.savefig(neg_out, dpi=300, bbox_inches='tight')
        plt.close(fig_neg)
        print(f"Saved: {neg_out}")

        # Line graph of raw loss for rows baseline_row+1 .. baseline_row+5 (interpreted as +1..+5 when baseline exists)
        fig_pos, ax_pos = plt.subplots(figsize=(10, 5))
        for k in range(1, 6):
            row_idx = baseline_row + k
            if 0 <= row_idx < losses_all_np.shape[0]:
                y = losses_all_np[row_idx, :]
                x = np.arange(y.size)
                if np.isnan(y).all():
                    continue
                is_boundary = (k == 1) or (k == 5)
                label = f"{y_values[row_idx]} blocks" if is_boundary else "_nolegend_"
                lw = 1.6 if is_boundary else 0.8
                alpha = 1.0 if is_boundary else 0.35
                marker = 'o' if is_boundary else None
                ax_pos.plot(x, y, marker=marker, linewidth=lw, alpha=alpha, label=label)
        ax_pos.set_xlabel('Start Position (block index)')
        ax_pos.set_ylabel('Loss (MSE)')
        ax_pos.set_title(
            f'Raw Loss vs Start for {"+1..+5" if rows % 2 == 1 else "+ rows"} (index {i}, allblocks)\nFile: {filename}'
        )
        ax_pos.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        if len(ax_pos.lines) > 0:
            ax_pos.legend()
        plt.tight_layout()
        pos_out = os.path.join(images_dir, f"loss_lines_allblocks_pos_{i}_{filename}.png")
        fig_pos.savefig(pos_out, dpi=300, bbox_inches='tight')
        plt.close(fig_pos)
        print(f"Saved: {pos_out}")

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
