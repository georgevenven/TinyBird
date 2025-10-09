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
    N = N.unsqueeze(0).to(device)

    x, x_i = model.compactify_data(x.clone(), x_i.clone(), N.clone())

    print(f"  Filename: {filename}")
    print(f"  Spectrogram shape: {x.shape}")
    print(f"  Chirp intervals shape: {x_i.shape}")
    print(f"  Chirp labels shape: {x_l.shape}")
    print(f"  Number of valid chirps: {N.item()}")

    def compute_loss(x, x_i, N, start, n_blocks):
        xs, x_is = model.sample_data(x.clone(), x_i.clone(), N.clone(), n_blocks=n_blocks, start=start)

        # Initialize masked_blocks and frac based on sequence length
        if xs.shape[-1] > 3000:
            masked_blocks, frac = 0, 0.5
        else:
            masked_blocks, frac = 1, 0.0

        h, idx_restore, bool_mask, bool_pad, T = model.forward_encoder(xs, x_is, masked_blocks=masked_blocks, frac=frac)
        pred = model.forward_decoder(h, idx_restore, T, bool_pad=bool_pad)
        loss = model.loss_mse(xs, pred, bool_mask)
        return loss

    block_min, block_max = 2, 13
    n_valid_chirps = N.max().item()
    losses = torch.zeros((block_max - block_min, n_valid_chirps - block_max), device=device)

    print(f"\nComputing losses for {n_valid_chirps - block_max} starting positions...")

    total_iterations = (n_valid_chirps - block_max) * (block_max - block_min)
    with tqdm(total=total_iterations, desc="Computing losses") as pbar:
        for start in range(block_max, n_valid_chirps):
            for n_blocks in range(block_min, block_max):
                with torch.no_grad():
                    loss = compute_loss(x, x_i, N, start, n_blocks)
                losses[n_blocks - block_min, start - block_max] = loss.item()
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

    # Create heatmap visualization
    print("\nGenerating heatmap visualization...")
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(losses_np, aspect='auto', cmap='viridis', origin='lower')

    # Set labels and title
    ax.set_xlabel('Start Position', fontsize=12)
    ax.set_ylabel('Number of Blocks', fontsize=12)
    ax.set_title(f'Reconstruction Loss Heatmap\nFile: {filename}', fontsize=14, pad=20)

    # Set y-axis ticks to show actual block counts
    block_min, block_max = 2, 13
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

    # Display the figure
    plt.show()

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
