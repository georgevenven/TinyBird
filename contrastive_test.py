#!/usr/bin/env python3
"""
contrastive_from_npz.py

Train a contrastive projection head from NPZ-encoded embeddings.
Uses 'encoded_embeddings_after_pos_removal' (or reasonable fallback keys).
Memory-maps NPZ files. Produces checkpoints and optional UMAP visualization.

Example:
python contrastive_from_npz.py \
  --npz /home/george/.../zebrafinch/embeddings.npz \
  --npz /home/george/.../canary/embeddings.npz \
  --outdir ./contrastive_out --epochs 3 --window_len 256 --stride 128 --batch_windows 12

Dependencies: numpy, torch, matplotlib (optional), umap-learn (optional)
"""
import os
import argparse
import random
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# --- Utilities ---
def choose_embedding_key(npz):
    """Pick best embedding key from npz file (memory-mapped npz object)."""
    preferred = [
        "encoded_embeddings_after_pos_removal",
        "patch_embeddings_after_pos_removal",
        "encoded_embeddings_before_pos_removal",
        "patch_embeddings_before_pos_removal",
        "encoded_embeddings",
        "patch_embeddings",
    ]
    for k in preferred:
        if k in npz.files:
            return k
    # fallback: any array with 2 dims and reasonable time x dim shape
    for k in npz.files:
        arr = npz[k]
        if arr.ndim >= 2:
            return k
    raise KeyError(f"No embedding-like key found in npz. Available keys: {npz.files}")


# --- Dataset: memory-mapped sliding windows metadata only ---
class NpzSlidingWindowDataset(Dataset):
    """
    Lazily index NPZ-embedded sequences and expose sliding windows.
    Stores metadata of windows (file_path, start_index) and reads slice on __getitem__.
    """
    def __init__(self, npz_paths: List[str], window_len: int = 256, stride: int = 128, downsample: int = 1):
        if downsample < 1:
            raise ValueError("downsample must be >=1")
        self.window_len = int(window_len)
        self.stride = int(stride)
        self.downsample = int(downsample)
        self._files = []  # list of dicts: {'path', 'embedding', 'labels', 'has_labels'}
        self._meta = []   # list of tuples (file_idx, start_idx)
        self.in_dim = None
        # Scan files and build metadata
        for p in npz_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            # memory-map
            npz = np.load(p, mmap_mode='r')
            key = choose_embedding_key(npz)
            arr = npz[key]
            
            # Check for labels (prefer downsampled labels if available)
            has_labels = False
            label_key = None
            if "labels_downsampled" in npz.files:
                label_key = "labels_downsampled"
                has_labels = True
            elif "labels_original" in npz.files:
                label_key = "labels_original"
                has_labels = True
            elif "labels" in npz.files:
                label_key = "labels"
                has_labels = True
            # Normalize shapes: allow (N,T,D) -> flatten to (N*T, D)
            if arr.ndim == 3:
                T = arr.shape[0] * arr.shape[1]
                D = arr.shape[2]
            elif arr.ndim == 2:
                T, D = arr.shape
            else:
                # flatten trailing dims to D
                T = arr.shape[0]
                D = int(np.prod(arr.shape[1:]))
            if self.in_dim is None:
                self.in_dim = D
            elif self.in_dim != D:
                raise ValueError(f"Inconsistent embedding dims. Found {D} but expected {self.in_dim}")
            # Materialize a 2D view for reuse
            if arr.ndim == 3:
                embedding = arr.reshape(-1, arr.shape[-1])
            elif arr.ndim == 2:
                embedding = arr
            else:
                embedding = arr.reshape(arr.shape[0], -1)
            # Load labels once if present and flatten for easy slicing
            label_arr = None
            if has_labels:
                label_arr = npz[label_key]
                if label_arr.ndim > 1:
                    label_arr = label_arr.reshape(-1)
            if self.downsample > 1:
                embedding_view = embedding[::self.downsample]
                if label_arr is not None:
                    label_arr = label_arr[::self.downsample]
            else:
                embedding_view = embedding
            file_index = len(self._files)
            self._files.append({
                "path": p,
                "embedding": embedding_view,
                "has_labels": has_labels,
                "labels": label_arr,
            })
            # compute effective length after downsample
            eff_T = embedding_view.shape[0]
            if eff_T >= self.window_len:
                # create window starts
                for start in range(0, eff_T - self.window_len + 1, self.stride):
                    self._meta.append((file_index, start))
            npz.close()
        if len(self._meta) == 0:
            raise RuntimeError("No windows collected. Reduce window_len or add files.")
    def __len__(self):
        return len(self._meta)
    def __getitem__(self, idx):
        file_idx, start = self._meta[idx]
        entry = self._files[file_idx]
        embedding = entry["embedding"]
        slice_view = embedding[start:start + self.window_len]
        slice_ = slice_view.astype(np.float32, copy=False)
        
        # Load labels if available
        labels = None
        if entry["has_labels"]:
            label_arr = entry["labels"]
            if label_arr is not None:
                labels = label_arr[start:start + self.window_len]
        
        return slice_, labels, (entry["path"], start)
    def get_in_dim(self):
        return self.in_dim


# --- Model and loss ---
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)


def nnclr_loss(proj: torch.Tensor, temperature: float = 0.07, distractors: Optional[int] = None):
    """
    NNCLR-style contrastive loss with in-batch nearest-neighbor positives.

    proj: (N, D) projection vectors.
    Positive for anchor i is its nearest neighbor (cosine similarity) among other samples.
    If distractors is None or >= N use all available negatives.
    Otherwise use top-k = distractors neighbors (including the positive).
    """
    device = proj.device
    N = proj.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)
    features = nn.functional.normalize(proj, dim=1)
    sim = features @ features.t()  # (N, N)
    self_mask = torch.eye(N, device=device, dtype=torch.bool)
    sim_without_self = sim.masked_fill(self_mask, float("-inf"))
    pos_idx = sim_without_self.argmax(dim=1)

    if distractors is None or distractors >= N:
        logits = sim_without_self / float(temperature)
        targets = pos_idx
        loss = nn.functional.cross_entropy(logits, targets)
        return loss

    # Use top-k similarities (including the positive) to limit negatives
    k = max(1, min(distractors, N - 1))
    top_vals, top_idx = torch.topk(sim_without_self, k=k, dim=1)
    logits = top_vals / float(temperature)
    # The nearest neighbor (positive) is in column 0 because topk sorts descending
    targets = torch.zeros(N, dtype=torch.long, device=device)
    loss = nn.functional.cross_entropy(logits, targets)
    return loss


# --- Training logic ---
def collate_windows(batch):
    # batch: list of (window_np, labels, meta)
    windows = [item[0] for item in batch]
    arr = np.stack(windows, axis=0)
    # Labels might be None for some windows, so we just pass them through
    labels = [item[1] for item in batch]
    metas = [item[2] for item in batch]
    return arr, labels, metas


def train_loop(dataset: Dataset,
               in_dim: int,
               out_dir: str,
               device: torch.device,
               epochs: int = 3,
               batch_windows: int = 12,
               lr: float = 1e-3,
               distractors: Optional[int] = None,
               save_every_epoch: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=batch_windows, shuffle=True,
                        num_workers=2, drop_last=True, collate_fn=collate_windows)
    hidden = max(512, in_dim * 2)
    head = ProjectionHead(in_dim, hidden=hidden, out_dim=128).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    history = []
    step = 0
    head.train()
    for ep in range(epochs):
        for batch_np, labels, metas in loader:
            # batch_np: (B, W, D)
            batch_t = torch.from_numpy(batch_np).to(device)
            B, W, D = batch_t.shape
            seq = batch_t.view(B * W, D)  # flatten in time order
            proj = head(seq)  # (N, out_dim)
            loss = nnclr_loss(proj, distractors=distractors)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            history.append(float(loss.item()))
            if step % 20 == 0:
                print(f"[ep {ep+1}/{epochs}] step {step} loss {loss.item():.4f}")
        if save_every_epoch:
            ckpt = os.path.join(out_dir, f"head_epoch{ep+1}.pt")
            torch.save({"state_dict": head.state_dict(), "opt": opt.state_dict(), "history": history}, ckpt)
            print("Saved checkpoint", ckpt)
    # final save
    final_path = os.path.join(out_dir, "head_final.pt")
    torch.save({"state_dict": head.state_dict(), "opt": opt.state_dict(), "history": history}, final_path)
    print("Saved final model", final_path)
    return head, history


# --- Collect projections and optional UMAP visualization ---
def collect_timestep_projections(head: nn.Module, dataset: Dataset, device: torch.device, n_steps: int = 200):
    """Accumulate projection vectors for individual time steps up to n_steps."""
    head.eval()
    projs = []
    step_labels = []
    has_any_labels = False
    collected = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            if collected >= n_steps:
                break
            window_np, labels, meta = dataset[i]
            t = torch.from_numpy(window_np.astype(np.float32)).to(device)
            seq = t.view(-1, t.shape[-1])
            proj_window = head(seq).cpu().numpy()  # (W, out_dim)
            available = proj_window.shape[0]
            take = min(available, n_steps - collected)
            if take <= 0:
                continue
            projs.append(proj_window[:take])
            if labels is not None:
                label_slice = np.asarray(labels[:take])
                step_labels.extend(label_slice.tolist())
                if label_slice.size > 0:
                    has_any_labels = True
            else:
                step_labels.extend([None] * take)
            collected += take
    if not projs:
        raise RuntimeError("No projections collected for UMAP.")
    projs = np.concatenate(projs, axis=0)
    labels_array = None
    if has_any_labels:
        labels_array = np.array(step_labels)
    return projs, labels_array


def try_umap_and_plot(projs: np.ndarray, out_path: str, labels: Optional[np.ndarray] = None):
    try:
        import umap
        import matplotlib.pyplot as plt
    except Exception as e:
        print("UMAP or matplotlib not available:", e)
        return None
    reducer = umap.UMAP(n_components=2, random_state=42)
    emb2 = reducer.fit_transform(projs)
    plt.figure(figsize=(8, 6))
    if labels is not None:
        labels_array = np.array(labels)
        numeric_kinds = {"i", "u", "f", "b"}
        if labels_array.dtype.kind not in numeric_kinds:
            unique_labels, label_codes = np.unique(labels_array, return_inverse=True)
            scatter = plt.scatter(emb2[:, 0], emb2[:, 1], c=label_codes, s=6, alpha=0.8, cmap='tab20')
            cbar = plt.colorbar(scatter, label='Labels')
            cbar.set_ticks(np.arange(len(unique_labels)))
            cbar.set_ticklabels([str(l) for l in unique_labels])
            label_count = len(unique_labels)
        else:
            scatter = plt.scatter(emb2[:, 0], emb2[:, 1], c=labels_array, s=6, alpha=0.8, cmap='tab20')
            plt.colorbar(scatter, label='Labels')
            label_count = len(np.unique(labels_array))
        plt.title(f"UMAP of timestep projections (colored by labels, {label_count} unique)")
    else:
        plt.scatter(emb2[:, 0], emb2[:, 1], s=6, alpha=0.8)
        plt.title("UMAP of timestep projections")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved UMAP figure to", out_path)
    return out_path


# --- CLI ---
def parse_args():
    p = argparse.ArgumentParser(description="Train contrastive projection head from NPZ-encoded embeddings")
    p.add_argument("--npz", required=True, action="append", help="Path to NPZ file. Use multiple times for multiple files.")
    p.add_argument("--window_len", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--downsample", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_windows", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--distractors", type=int, default=8, nargs="?")
    p.add_argument("--outdir", type=str, default="./results/contrastive_out")
    p.add_argument("--device", type=str, default='cuda:0',
                   help="Device string like 'cuda:0' or 'cpu'. Default auto.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--umap_n_steps", type=int, default=100000, help="Number of time steps to collect for UMAP.")
    p.add_argument("--umap_n_windows", type=int, dest="umap_n_steps", help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    print("Device:", device)

    print("Building dataset metadata from NPZ files...")
    dataset = NpzSlidingWindowDataset(args.npz, window_len=args.window_len, stride=args.stride, downsample=args.downsample)
    print("Windows collected:", len(dataset))
    in_dim = dataset.get_in_dim()
    print("Embedding dim:", in_dim)

    head, history = train_loop(dataset, in_dim, args.outdir, device,
                               epochs=args.epochs,
                               batch_windows=args.batch_windows,
                               lr=args.lr,
                               distractors=args.distractors)

    projs, labels = collect_timestep_projections(head, dataset, device, n_steps=args.umap_n_steps)
    umap_path = os.path.join(args.outdir, "umap_windows.png")
    try_umap_and_plot(projs, umap_path, labels=labels)
    print("Done.")


if __name__ == "__main__":
    main()
