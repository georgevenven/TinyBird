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
    parser.add_argument("--index", type=int, default=0, help="Index of the spectrogram file to analyze from the dataset (default: 0)")  # fmt: skip
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
        assert xi.dim() == 3 and xi.size(-1) == 2, f"xi must be (B,N_max,2), got {tuple(x.shape)}"

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

    def compute_loss(x, x_i, N, start, x_mean, n_blocks):
        xs, x_is = model.sample_data(x.clone(), x_i.clone(), N.clone(), n_blocks=n_blocks, start=start)

        if n_blocks == 1:
            x_mean_expanded = x_mean.expand_as(xs)
            return (x_mean_expanded - xs).pow(2).mean()  # (B, T, P)

        # Initialize masked_blocks and frac based on sequence length
        if xs.shape[-1] > 3000:
            masked_blocks, frac = 0, 0.5
        else:
            masked_blocks, frac = 1, 0.0

        h, idx_restore, bool_mask, bool_pad, T = model.forward_encoder(xs, x_is, masked_blocks=masked_blocks, frac=frac)
        pred = model.forward_decoder(h, idx_restore, T, bool_pad=bool_pad)
        loss = model.loss_mse(xs, pred, bool_mask)
        return loss

    block_min, block_max = 1, 13
    n_valid_chirps = N.max().item()
    losses = torch.zeros((block_max - block_min, n_valid_chirps - block_max), device=device)

    print(f"\nComputing losses for {n_valid_chirps - block_max} starting positions...")

    total_iterations = (n_valid_chirps - block_max) * (block_max - block_min)
    with tqdm(total=total_iterations, desc="Computing losses") as pbar:
        for start in range(0, n_valid_chirps - block_max):
            for n_blocks in range(block_min, block_max):
                with torch.no_grad():
                    loss = compute_loss(x, x_i, N, start, x_mean, n_blocks)
                losses[n_blocks - block_min, start] = loss.item()
                pbar.update(1)

    return losses, filename


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

    # Load specific sample
    if args.index >= len(dataset):
        raise ValueError(f"Index {args.index} out of range. Dataset has {len(dataset)} files.")

    print(f"\nLoading sample at index {args.index}")

    losses, filename = process_file(model, dataset, args.index, device)

    # Convert losses to numpy for plotting
    losses_np = losses.cpu().numpy()

    # ------------------ New: Aggregate loss vs blocks & marginal gains ------------------
    import numpy as np

    # losses_np has shape (num_blocks, num_starts) where rows correspond to n_blocks in [block_min, block_max)
    # From process_file(): block_min=1, block_max=13 → rows represent n_blocks = 1..12
    block_min, block_max = 1, 13
    n_blocks_axis = np.arange(block_min, block_max)  # 1..12 inclusive

    # Mean & std across all start positions
    mean_loss_per_blocks = losses_np.mean(axis=1)
    std_loss_per_blocks  = losses_np.std(axis=1)

    # Marginal change (delta) when adding one more block: L(k) - L(k-1)
    # This results in rows aligned to 2..12 (since delta from 1->2 is the first row)
    deltas = losses_np[1:, :] - losses_np[:-1, :]
    delta_blocks_axis = np.arange(block_min + 1, block_max)  # 2..12
    mean_delta_per_add = deltas.mean(axis=1)
    std_delta_per_add  = deltas.std(axis=1)

    # Relative improvement wrt 1-block baseline: (L(k) - L(1)) / L(1)
    baseline = losses_np[0:1, :]  # shape (1, num_starts)
    rel_improve = (losses_np - baseline) / np.maximum(np.abs(baseline), 1e-12)
    mean_rel_improve = rel_improve.mean(axis=1)
    std_rel_improve  = rel_improve.std(axis=1)

    # 1) Plot: Mean loss vs number of blocks with ±1 std band
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(n_blocks_axis, mean_loss_per_blocks, marker='o')
    ax1.fill_between(n_blocks_axis, 
                     mean_loss_per_blocks - std_loss_per_blocks, 
                     mean_loss_per_blocks + std_loss_per_blocks, 
                     alpha=0.2)
    ax1.set_xlabel('Number of Blocks')
    ax1.set_ylabel('Mean MSE Loss (across starts)')
    ax1.set_title(f'Mean Loss vs Blocks\nFile: {filename}')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    out1 = f"loss_vs_blocks_{filename}.png"
    fig1.savefig(out1, dpi=300, bbox_inches='tight')
    print(f"Saved: {out1}")

    # 2) Plot: Marginal change when adding one more block (negative means improvement)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(delta_blocks_axis, mean_delta_per_add)
    ax2.errorbar(delta_blocks_axis, mean_delta_per_add, yerr=std_delta_per_add, fmt='none', capsize=3)
    ax2.axhline(0.0, linestyle='--', linewidth=1)
    ax2.set_xlabel('Blocks After Adding One More (k)')
    ax2.set_ylabel('Mean ΔLoss = L(k) - L(k-1)')
    ax2.set_title(f'Marginal Gain from Adding a Block\nFile: {filename}')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    out2 = f"delta_loss_per_block_{filename}.png"
    fig2.savefig(out2, dpi=300, bbox_inches='tight')
    print(f"Saved: {out2}")

    # 3) Plot: Relative improvement vs 1-block baseline with ±1 std band
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(n_blocks_axis, mean_rel_improve, marker='o')
    ax3.fill_between(n_blocks_axis, 
                     mean_rel_improve - std_rel_improve, 
                     mean_rel_improve + std_rel_improve, 
                     alpha=0.2)
    ax3.axhline(0.0, linestyle='--', linewidth=1)
    ax3.set_xlabel('Number of Blocks')
    ax3.set_ylabel('Relative Improvement vs 1-Block (Δ/|L1|)')
    ax3.set_title(f'Relative Improvement vs Baseline\nFile: {filename}')
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    out3 = f"relative_improvement_{filename}.png"
    fig3.savefig(out3, dpi=300, bbox_inches='tight')
    print(f"Saved: {out3}")
    # --------------------------------------------------------------------

    # Create heatmap visualization
    print("\nGenerating heatmap visualization...")
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(losses_np, aspect='auto', cmap='viridis', origin='lower')

    # Set labels and title
    ax.set_xlabel('Start Position', fontsize=12)
    ax.set_ylabel('Number of Blocks', fontsize=12)
    ax.set_title(f'Reconstruction Loss Heatmap\nFile: {filename}', fontsize=14, pad=20)

    # Set y-axis ticks to show actual block counts
    block_min, block_max = 0, 12
    ax.set_yticks(range(block_max - block_min))
    ax.set_yticklabels(range(block_min, block_max))

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MSE Loss', rotation=270, labelpad=20, fontsize=12)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()

    # Save the figure
    output_filename = f"loss_heatmap_{filename}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_filename}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
