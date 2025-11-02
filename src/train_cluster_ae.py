"""Training script for the cluster autoencoder."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:  # pragma: no cover - optional
    wandb = None

from dataset_clusters import ClusterBalancedDataset, collate_blocks
from model_cluster_ae import ClusterAutoEncoder
from registry_utils import load_registry


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train cluster autoencoder.")
    parser.add_argument("--registry", type=str, required=True, help="Path to global cluster registry sqlite.")
    parser.add_argument("--clusters_dir", type=str, required=True, help="Directory containing channel cluster pickles.")
    parser.add_argument("--split", type=str, nargs="*", default=None, help="Optional splits to train on.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--match-threshold", type=float, default=0.35)  # reserved
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--output", type=str, default="../runs/cluster_ae")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape != pred.shape:
        mask = mask.expand_as(pred)
    diff = (pred - target) * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    return (diff.pow(2).sum()) / denom


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output, exist_ok=True)

    dataset = ClusterBalancedDataset(
        registry_path=args.registry,
        clusters_dir=args.clusters_dir,
        split_filter=args.split,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_blocks,
    )

    prototypes_by_m = load_registry(args.registry)
    num_clusters = len(dataset.clusters)
    config = {
        "enc_hidden_d": 256,
        "enc_n_head": 8,
        "enc_dim_ff": 1024,
        "enc_n_layer": 4,
        "dec_hidden_d": 256,
        "dec_n_head": 8,
        "dec_dim_ff": 1024,
        "dec_n_layer": 4,
        "dropout": 0.1,
        "max_seq": 8192,
        "lambda_recon": args.lambda_recon,
        "lambda_cls": args.lambda_cls,
    }

    model = ClusterAutoEncoder(config, prototypes_by_m, num_clusters, dataset.feature_dim).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(opt, T_max=args.steps)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint.get("step", 0)

    if not args.no_wandb and wandb is not None:
        wandb.init(project="cluster_ae", name=args.run_name, config=config)

    model.train()
    dataloader_iterator = iter(dataloader)

    while global_step < args.steps:
        try:
            batch = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(dataloader)
            batch = next(dataloader_iterator)

        x, mask, labels, dims_masks, infos = batch
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp and device.type == "cuda"):
            recon, logits, latent = model(x, mask)
            loss_dict = model.compute_loss(recon, logits, x, mask, labels)
            loss = loss_dict["loss"]

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        scheduler.step()

        if not args.no_wandb and wandb is not None:
            wandb.log(
                {
                    "train/loss": loss_dict["loss"].item(),
                    "train/loss_recon": loss_dict["loss_recon"].item(),
                    "train/loss_cls": loss_dict["loss_cls"].item(),
                    "lr": scheduler.get_last_lr()[0],
                    "step": global_step,
                },
                step=global_step,
            )

        if global_step % args.log_every == 0:
            print(
                f"step {global_step:06d} "
                f"loss={loss_dict['loss'].item():.4f} "
                f"recon={loss_dict['loss_recon'].item():.4f} "
                f"cls={loss_dict['loss_cls'].item():.4f} "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )

        if global_step % args.save_every == 0 and global_step > 0:
            ckpt_dir = Path(args.output) / "weights"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"step_{global_step:06d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": global_step,
                },
                ckpt_path,
            )

        global_step += 1

    if not args.no_wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

