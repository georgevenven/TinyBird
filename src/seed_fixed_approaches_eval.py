from __future__ import annotations

import argparse
import random
import numpy as np
from typing import List, Sequence

from sympy.core.numbers import I
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Mirror your analyze_model.py imports/names exactly
from utils import load_model_from_checkpoint
from data_loader import SpectogramDataset  # note: "SpectogramDataset" (matches your file)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate fixed-context approaches and run GA over teams.")
    p.add_argument("--model_path", type=str, required=True, help="Path to run dir (config.json + weights/).")
    p.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint file (optional).")
    p.add_argument("--data_dir", type=str, default=None, help="Path to .pt spectrograms (else val_dir from config).")
    p.add_argument("--index", type=int, default=0, help="Dataset index of the single file to analyze.")
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device for inference.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")

    return p.parse_args()


class Team:
    def __init__(self, model, x, x_i, N, n_blocks=10, device=torch.device('cuda')):
        """
        Team manages a set of fixed-context approaches and their evaluation.
        Args:
            model: The model to evaluate.
            x: Input data tensor.
            x_i: Index tensor.
            N: Number of chirps (int).
            n_blocks: Number of context blocks.
            device: torch device.
        """
        self.model = model
        self.x = x
        self.x_i = x_i
        self.N = int(N)
        self.n_blocks = int(n_blocks)
        self.device = device

        # Targets are chirps 1..N-1 (since 0 can't be predicted from prior)
        self.N_targets = list(range(1, self.N))
        self.approaches = self.build_approaches(local_blocks=0)
        self.losses = torch.full((len(self.approaches), self.N), float('inf'), device=device)
        self.leaderboard = [-1 for _ in range(self.N)]
        self.winners = set(a.index for a in self.approaches if a.keep)
        print(f"[Team] Init: N={self.N}, n_blocks={self.n_blocks}, approaches={len(self.approaches)}")

    def build_approaches(self, local_blocks=0):
        context = max(0, self.n_blocks - int(local_blocks))
        approaches = []
        for t in range(self.N):  # range(self.N):
            # Keep a fixed prefix [t, t+1, ..., t+context-1] (clamped to N-1)
            right = min(t + context, self.N - 1)
            indices = list(range(t, right))
            approaches.append(Approach(indices=indices, index=len(approaches), n_blocks=self.n_blocks, iteration=0))
        print(f"[Team] Built {len(approaches)} initial approaches (context={context}).")
        return approaches

    def eval_loss(self, indices: Sequence[int]) -> float:
        """
        Run the authoritative TinyBird loss path EXACTLY and return Python float32.
        On CUDA OOM or messages containing 'out of memory'/'cuda', clear cache and return NaN.
        """
        try:
            xs, x_is = self.model.sample_data_indices(
                self.x.clone(), self.x_i.clone(), torch.tensor([self.N], device=self.device), list(indices)
            )
            mblock = [len(indices) - 1]  # last element is the masked target
            h, idx_restore, bool_mask, bool_pad, T = self.model.forward_encoder(xs, x_is, mblock=mblock)
            pred = self.model.forward_decoder(h, idx_restore, T, bool_pad=bool_pad, attend_to_padded=False)
            loss = self.model.loss_mse(xs, pred, bool_mask)
            val = float(loss.detach().item())
            return np.float32(val).item()
        except RuntimeError as e:
            msg = str(e).lower()
            if ("out of memory" in msg or "cuda" in msg) and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                print("[Team] CUDA OOM while eval_loss; returning NaN for indices=", indices)
                return np.float32(float('inf')).item()
            else:
                raise

    def eval_approach(self, approach, t):
        return self.eval_loss(approach.build_indices(t))

    def eval_approach_all(self, approach, iteration=0):
        if approach.keep and approach.iteration == iteration:
            for t in self.N_targets:
                self.losses[approach.index, t] = self.eval_approach(approach, t)
            return True
        elif approach.keep :
            return True
        else:
            self.losses[approach.index, :] = np.float32(float('inf')).item()
            return False

    def eval_all(self, iteration=0):
        valid_approaches = sum(1 for approach in self.approaches if approach.keep )
        print(f"[Team] Evaluating {valid_approaches} kept approaches over {len(self.N_targets)} targets each…")
        with tqdm(total=valid_approaches, desc="Computing losses") as pbar:
            for approach in self.approaches:
                if self.eval_approach_all(approach, iteration=iteration):
                    pbar.update(1)

    def keep_winners(self, iteration=0):
        self.winners = set(a.index for a in self.approaches if a.keep)
        original_length = len(self.winners)

        self.eval_all(iteration=iteration)
        self.new_winners = set(torch.argmin(self.losses, dim=0).tolist())
        self.losers = self.winners - self.new_winners

        print(f"[Team] Winners this round: {sorted(list(self.new_winners))} (total {len(self.new_winners)})")
        print(f"[Team] Approaches to drop: {sorted(list(self.losers))} (total {len(self.losers)})")
        for loser in self.losers:
            loser.keep = False

        keep_mask = torch.tensor([a.keep for a in self.approaches], device=self.losses.device)
        self.losses = self.losses[keep_mask, :]
        self.approaches = [a for a in self.approaches if a.keep]

        print(f"[Team] Condensed to {len(self.approaches)} approaches from {original_length} after pruning losers.")

    def add_new_approaches(self, iteration=0):
        new_approaches = []
        for a in self.approaches:
            if a.iteration == iteration and a.keep and len(a.indices) > 0:
                new_approaches.append(a.prune_context())
        self.approaches.extend(new_approaches)
        for i, a in enumerate(self.approaches):
            a.index = i
        self.losses = torch.cat(
            [self.losses, torch.full((len(new_approaches), self.N), float('inf'), device=self.losses.device)]
        )
        print(
            f"[Team] Added {len(new_approaches)} pruned-context approaches for iteration {iteration}. Total now {len(self.approaches)}."
        )

    def summarize(self, tag=""):
        if self.losses.numel() == 0:
            print("[Team] summarize: no losses yet.")
            return
        with torch.no_grad():
            # Winners per target (we already use +inf for invalid entries)
            argmins = torch.argmin(self.losses, dim=0)
            counts = torch.bincount(argmins, minlength=self.losses.shape[0]).tolist()

            print(f"[Team] Summary {tag}: approaches with ≥1 win (in current order)")
            for i, a in enumerate(self.approaches):
                wins = int(counts[i]) if i < len(counts) else 0
                if wins > 0:
                    print(f"   • {i}: wins={wins} | context={a.indices} | iter={a.iteration}")

    def optimize(self, images_dir="images_seed_eval"):
        os.makedirs(images_dir, exist_ok=True)
        rounds = max(0, self.n_blocks - 1)
        for i in range(rounds):
            print("\n" + "=" * 60)
            print(f"[Team] Optimize round {i+1}/{rounds}")
            print("=" * 60)
            self.keep_winners(iteration=i)
            self.plot_state(images_dir, suffix=f"round{i+1}")
            self.summarize(tag=f"after round {i+1}")
            self.add_new_approaches(iteration=i)
        print("\n[Team] Optimization complete.")

    def _loss_numpy(self):
        return self.losses.detach().cpu().numpy()

    def plot_state(self, images_dir, suffix=""):
        # Heatmap: approaches × targets
        arr = self._loss_numpy()
        arr[arr == np.inf] = np.nan
        fig, ax = plt.subplots(figsize=(12, 6))
        m = np.ma.masked_invalid(arr)
        vmin = float(np.nanmin(m)) if np.isfinite(np.nanmin(m)) else 0.0
        vmax = float(np.nanmax(m)) if np.isfinite(np.nanmax(m)) else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        im = ax.imshow(m, aspect='auto', origin='lower', cmap=plt.get_cmap('RdYlGn_r'), vmin=vmin, vmax=vmax)
        ax.set_xlabel('Target t')
        ax.set_ylabel('Approach index')
        ax.set_title('Approach vs Target Loss (MSE)')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Loss (MSE)', rotation=270, labelpad=20)
        out1 = os.path.join(images_dir, f"loss_heatmap_{suffix}.png")
        fig.tight_layout()
        fig.savefig(out1, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[Plot] Saved {out1}")

        with np.errstate(invalid='ignore'):
            arr_copy = arr.copy()
            arr_copy[~np.isfinite(arr_copy)] = np.inf
            winners = np.argmin(arr_copy, axis=0)

        # Histogram
        unique, counts = np.unique(winners, return_counts=True)
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.bar(unique, counts)
        ax3.set_xlabel('Approach index')
        ax3.set_ylabel('#Targets won')
        ax3.set_title('Win count per approach')
        out3 = os.path.join(images_dir, f"winners_hist_{suffix}.png")
        fig3.tight_layout()
        fig3.savefig(out3, dpi=200, bbox_inches='tight')
        plt.close(fig3)
        print(f"[Plot] Saved {out3}")


class Approach:
    def __init__(self, indices=None, index=-1, n_blocks=10, iteration=0):
        self.indices = list(indices or [])
        self.index = int(index)
        self.keep = True
        self.n_blocks = int(n_blocks)
        self.iteration = int(iteration)

    def build_indices(self, t: int) -> List[int]:
        kept = [int(idx) for idx in self.indices]
        kept = kept[: max(0, int(self.n_blocks))]
        remaining = max(0, int(self.n_blocks) - len(kept))
        if remaining > 0:
            preds = list(range(max(0, t - remaining), t))
            indices = kept + preds + [t]
        else:
            indices = kept + [t]
        return indices

    def prune_context(self):
        # drop the earliest fixed index to free one slot, keep iteration+1
        new_indices = self.indices[1:] if len(self.indices) > 0 else []
        return Approach(indices=new_indices, index=-1, n_blocks=self.n_blocks, iteration=self.iteration + 1)


def main():
    args = parse_args()

    # seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load model
    print("=" * 60)
    print("TinyBird Fixed-Context Approaches Evaluation")
    print("=" * 60)
    print(f"Loading model: {args.model_path}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    model, config = load_model_from_checkpoint(
        run_dir=args.model_path, checkpoint_file=args.checkpoint, fallback_to_random=False
    )
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    # Data dir, Dataset + pick item
    if args.data_dir is not None:
        data_dir = args.data_dir
    elif "val_dir" in config:
        data_dir = config["val_dir"]
    else:
        raise ValueError("No data directory specified. Provide --data_dir or ensure val_dir in config.json")
    dataset = SpectogramDataset(
        dir=data_dir, n_mels=config.get("mels", 128), n_timebins=config.get("num_timebins", 1024), pad_crop=True
    )
    if not (0 <= args.index < len(dataset)):
        raise ValueError(f"--index {args.index} out of range (dataset size={len(dataset)}).")
    x, x_i, x_l, N, file_name = dataset[args.index]
    x = x.unsqueeze(0).float().to(device)
    x_i = x_i.unsqueeze(0).to(device)
    N = int(torch.tensor([int(N)], dtype=torch.long, device=device).max().item())
    x, x_i = model.compactify_data(x.clone(), x_i.clone(), torch.tensor([N], device=device))
    print(f"[Main] compactify_data → x={tuple(x.shape)}, x_i={tuple(x_i.shape)}, N={N}")

    team = Team(model, x, x_i, N, n_blocks=10, device=device)
    team.optimize()
    print("[Main] Seed fixed-approaches evaluation complete. Plots written under images_seed_eval/ .")


if __name__ == "__main__":
    main()
