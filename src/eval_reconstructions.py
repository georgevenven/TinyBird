#!/usr/bin/env python3
import argparse
import os
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local deps
from data_loader import SpectogramDataset
from utils import load_model_from_checkpoint


def depatchify(pred_patches, H, W, patch_size):
    # pred_patches: (B, T, P) → (B, 1, H, W)
    fold = nn.Fold(output_size=(H, W), kernel_size=patch_size, stride=patch_size)
    return fold(pred_patches.transpose(1, 2))


def masked_original(x_patches, bool_mask):
    # x_patches: (B, T, P), bool_mask: (B, T)
    masked = x_patches.clone()
    # Set masked values very low to render as black in viridis colormap
    masked[bool_mask] = -10.0
    return masked


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)


def main():
    parser = argparse.ArgumentParser(description="Reconstruct spectrograms and compute MSE.")
    parser.add_argument("--run_dir", required=True, type=str, help="Run directory or name under ../runs")
    parser.add_argument("--spec_dir", required=True, type=str, help="Directory of spectrogram tensors (val-style)")
    parser.add_argument("--out_dir", required=True, type=str, help="Folder to store results")
    parser.add_argument("--num_samples", type=int, default=10000, help="Max samples to evaluate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint filename to load")
    parser.add_argument("--per_patch_norm", action="store_true", help="Enable per-patch normalization for visualization")
    parser.add_argument("--inference_mode", action="store_true", help="Disable masking (autoencoder-style reconstruction)")
    args = parser.parse_args()

    # Load model + config
    model, config = load_model_from_checkpoint(
        run_dir=args.run_dir,
        checkpoint_file=args.checkpoint,
        fallback_to_random=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Dataset and loader (val-style), batch size = 1
    dataset = SpectogramDataset(
        dir=args.spec_dir,
        n_timebins=config["num_timebins"]
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Output dirs
    mse_root = os.path.join(args.out_dir, "MSE analysis")
    os.makedirs(mse_root, exist_ok=True)
    imgs_dir = os.path.join(mse_root, "imgs")
    os.makedirs(imgs_dir, exist_ok=True)

    # Save a copy of run config for traceability
    with open(os.path.join(mse_root, "eval_config.json"), "w") as f:
        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_dir": args.run_dir,
            "checkpoint": args.checkpoint,
            "spec_dir": args.spec_dir,
            "num_samples": args.num_samples,
            "device": str(device),
            "model_config": config
        }
        json.dump(meta, f, indent=2)

    patch_size = tuple(config["patch_size"])
    H = int(dataset.n_mels)
    W = int(config["num_timebins"])

    unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    # Aggregators for true dataset-wide MSE (sum of squared errors / total elements)
    SSE_all = 0.0
    N_all = 0
    SSE_masked = 0.0
    N_masked = 0

    # Per-sample CSV
    csv_path = os.path.join(mse_root, "per_sample_mse.csv")
    with open(csv_path, "w") as fcsv:
        fcsv.write("index,filename,mse_all,mse_masked\n")

    pbar = tqdm(total=min(args.num_samples, len(loader)), desc="Evaluating", unit="sample")
    evaluated = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if evaluated >= args.num_samples:
                break

            spectrograms, filenames = batch  # shapes: (1, 1, H, W), list[str]
            x = spectrograms.to(device, non_blocking=True)

            # Forward pass: encoder → decoder
            h, idx_restore, bool_mask, T = model.forward_encoder(x, inference_mode=args.inference_mode)
            pred = model.forward_decoder(h, idx_restore, T)  # (1, T, P)

            # Prepare patches of target
            x_patches = unfold(x).transpose(1, 2)  # (1, T, P)

            # Disable per-patch denormalization so we visualise raw decoder outputs.
            # target_mean = x_patches.mean(dim=-1, keepdim=True)
            # target_std = x_patches.std(dim=-1, keepdim=True)
            # pred_denorm = pred * (target_std + 1e-6) + target_mean  # (1, T, P)
            pred_denorm = pred.to(dtype=x_patches.dtype)

            # Overlay image: original for unmasked, prediction for masked.
            # In inference_mode there is no mask, so show full reconstruction.
            if args.inference_mode:
                overlay_patches = pred_denorm
            else:
                overlay_patches = x_patches.clone()
                overlay_patches[bool_mask] = pred_denorm[bool_mask]
            
            # Normalize all patches patch-wise for consistent visualization
            if args.per_patch_norm:
                overlay_mean = overlay_patches.mean(dim=-1, keepdim=True)
                overlay_std = overlay_patches.std(dim=-1, keepdim=True)
                overlay_patches_normalized = (overlay_patches - overlay_mean) / (overlay_std + 1e-6)
            else:
                overlay_patches_normalized = overlay_patches
            
            overlay_img = depatchify(overlay_patches_normalized, H=H, W=W, patch_size=patch_size)

            # Masked-original image for visualization
            masked_patches = masked_original(x_patches, bool_mask)
            masked_img = depatchify(masked_patches, H=H, W=W, patch_size=patch_size)

            # Per-sample MSEs in raw scale
            diff2 = (pred_denorm - x_patches) ** 2  # (1, T, P)
            mse_all = diff2.mean().item()
            # Masked elements count = (#masked tokens) * P
            masked_elems = bool_mask.sum().item() * diff2.size(-1)
            if masked_elems > 0:
                mse_masked = diff2[bool_mask].mean().item()
            else:
                mse_masked = float("nan")  # no masked tokens (should not happen with mask_p>0)

            # Global aggregates
            SSE_all += diff2.sum().item()
            N_all += diff2.numel()
            SSE_masked += diff2[bool_mask].sum().item()
            N_masked += masked_elems

            # Save visualization
            x_img = x[0, 0].detach().cpu().numpy()
            masked_img_np = masked_img[0, 0].detach().cpu().numpy()
            overlay_np = overlay_img[0, 0].detach().cpu().numpy()

            fname = sanitize(filenames[0] if isinstance(filenames, list) else str(filenames))

            # Plot: Overlay
            fig2 = plt.figure(figsize=(7.9, 5.8933))
            ax1 = plt.subplot(3, 1, 1)
            ax1.imshow(x_img, origin="lower", aspect="auto")
            ax1.set_title("Input Spectrogram", fontsize=16, fontweight='bold')
            ax1.axis("off")

            ax2 = plt.subplot(3, 1, 2)
            ax2.imshow(masked_img_np, origin="lower", aspect="auto")
            ax2.set_title("Input Spectrogram With Mask", fontsize=16, fontweight='bold')
            ax2.axis("off")

            ax3 = plt.subplot(3, 1, 3)
            ax3.imshow(overlay_np, origin="lower", aspect="auto")
            ax3.set_title(
                "Decoder Output" if args.inference_mode else "Decoder Predictions and Original Spectrogram",
                fontsize=16,
                fontweight="bold",
            )
            ax3.axis("off")

            fig2.tight_layout()
            out_png2 = os.path.join(imgs_dir, f"{i:06d}_{fname}_overlay.png")
            fig2.savefig(out_png2, dpi=300, facecolor='white', 
                       edgecolor='none')
            plt.close(fig2)

            # Append CSV
            with open(csv_path, "a") as fcsv:
                fcsv.write(f"{i},{fname},{mse_all:.8f},{mse_masked:.8f}\n")

            evaluated += 1
            pbar.set_postfix(mse_all=f"{mse_all:.5g}", mse_masked=f"{mse_masked:.5g}")
            pbar.update(1)

    pbar.close()

    # Final summary
    summary = {
        "evaluated_samples": evaluated,
        "pixels_per_patch": int(patch_size[0] * patch_size[1]),
        "SSE_all": SSE_all,
        "N_all": N_all,
        "MSE_all_dataset_mean": (SSE_all / N_all) if N_all > 0 else float("nan"),
        "SSE_masked": SSE_masked,
        "N_masked": N_masked,
        "MSE_masked_dataset_mean": (SSE_masked / N_masked) if N_masked > 0 else float("nan"),
    }
    with open(os.path.join(mse_root, "summary.json"), "w") as fsum:
        json.dump(summary, fsum, indent=2)

    print("Done.")
    print(f"Summary: {os.path.join(mse_root, 'summary.json')}")
    print(f"Per-sample CSV: {csv_path}")
    print(f"Images dir: {imgs_dir}")


if __name__ == "__main__":
    main()
