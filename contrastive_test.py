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
from typing import Dict, List, Optional, Tuple

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


def flatten_to_time_major(arr: np.ndarray) -> np.ndarray:
    """
    Reshape an array so time is the first axis and features are in the last axis.
    Works for arrays with trailing feature dimensions.
    """
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    leading = int(np.prod(arr.shape[:-1]))
    return arr.reshape(leading, arr.shape[-1])


def align_spectrogram_to_embeddings(spec: np.ndarray, expected_len: int) -> np.ndarray:
    """
    Try different orientation permutations so the resulting spectrogram view matches
    the embedding timeline length. Returns a view with shape (time, features).
    """
    candidates = []

    def add_candidate(arr: np.ndarray):
        view = flatten_to_time_major(arr)
        if view.shape[0] == expected_len:
            candidates.append(view)

    add_candidate(spec)
    if spec.ndim >= 2:
        add_candidate(np.swapaxes(spec, 0, 1))
    if spec.ndim >= 3:
        add_candidate(spec.transpose(1, 0, 2))
        add_candidate(spec.transpose(0, 2, 1))
        add_candidate(spec.transpose(2, 0, 1))
        add_candidate(spec.transpose(1, 2, 0))
        add_candidate(spec.transpose(2, 1, 0))

    if candidates:
        return candidates[0]
    raise ValueError(f"Spectrogram frames {spec.shape} do not match expected length {expected_len}")


def smooth_power_envelope(power: np.ndarray, smoothing: int) -> np.ndarray:
    """Apply simple moving-average smoothing to the power envelope."""
    if smoothing <= 1:
        return power
    kernel = np.ones(int(smoothing), dtype=np.float32)
    kernel /= kernel.sum()
    return np.convolve(power, kernel, mode="same")


def compute_power_envelope_labels(spec_view: np.ndarray,
                                  smoothing: int = 5,
                                  percentile: Optional[float] = 70.0,
                                  min_frames: int = 4,
                                  pad: int = 0,
                                  abs_threshold: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute power envelope labels for a spectrogram view.
    Returns (labels, stats) where labels shape (time,) has segment ids or -1.
    """
    if spec_view.ndim != 2:
        raise ValueError(f"Spectrogram view must be 2D, got shape {spec_view.shape}")

    power = np.square(np.asarray(spec_view, dtype=np.float32)).sum(axis=1)
    smoothed = smooth_power_envelope(power, smoothing)

    if abs_threshold is not None:
        threshold = float(abs_threshold)
    elif percentile is not None:
        threshold = float(np.percentile(smoothed, percentile))
    else:
        threshold = float(np.percentile(smoothed, 50.0))

    active = smoothed >= threshold
    n = active.shape[0]
    segments = []
    idx = 0
    while idx < n:
        if active[idx]:
            start = idx
            while idx < n and active[idx]:
                idx += 1
            segments.append((start, idx))
        else:
            idx += 1

    if pad > 0 and segments:
        padded = []
        for start, end in segments:
            start = max(0, start - pad)
            end = min(n, end + pad)
            if padded and start <= padded[-1][1]:
                padded[-1] = (padded[-1][0], max(padded[-1][1], end))
            else:
                padded.append((start, end))
        segments = padded

    filtered = []
    min_frames = max(1, int(min_frames))
    for start, end in segments:
        if end - start >= min_frames:
            filtered.append((start, end))

    labels = np.full(n, -1, dtype=np.int64)
    positive_frames = 0
    for start, end in filtered:
        labels[start:end] = 1  # use a single positive label
        positive_frames += (end - start)

    stats = {
        "threshold": float(threshold),
        "segments": int(len(filtered)),
        "positive_frames": int(positive_frames),
        "total_frames": int(n),
        "active_ratio": float(positive_frames / n) if n > 0 else 0.0,
    }
    return labels, stats


# --- Dataset: memory-mapped sliding windows metadata only ---
class NpzSlidingWindowDataset(Dataset):
    """
    Lazily index NPZ-embedded sequences and expose sliding windows.
    Stores metadata of windows (file_path, start_index) and reads slice on __getitem__.
    """
    def __init__(self,
                 npz_paths: List[str],
                 window_len: int = 256,
                 stride: int = 128,
                 downsample: int = 1,
                 envelope_percentile: Optional[float] = 70.0,
                 envelope_abs_threshold: Optional[float] = None,
                 envelope_smoothing: int = 5,
                 envelope_min_frames: int = 4,
                 envelope_pad: int = 0):
        if downsample < 1:
            raise ValueError("downsample must be >=1")
        self.window_len = int(window_len)
        self.stride = int(stride)
        self.downsample = int(downsample)
        self.envelope_percentile = envelope_percentile
        self.envelope_abs_threshold = envelope_abs_threshold
        self.envelope_smoothing = int(max(1, envelope_smoothing))
        self.envelope_min_frames = int(max(1, envelope_min_frames))
        self.envelope_pad = int(max(0, envelope_pad))
        self._files = []  # list of dicts: {'path', 'embedding', 'labels', 'has_labels'}
        self._meta = []   # list of tuples (file_idx, start_idx)
        self.in_dim = None
        self.using_envelope = False
        self._envelope_segments = 0
        self._envelope_positive_frames = 0
        self._files_with_envelope = 0
        # Scan files and build metadata
        for p in npz_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            # memory-map
            npz = np.load(p, mmap_mode='r')
            key = choose_embedding_key(npz)
            arr = npz[key]
            embedding_full = flatten_to_time_major(arr)

            embedding_dim = embedding_full.shape[1]
            if self.in_dim is None:
                self.in_dim = embedding_dim
            elif self.in_dim != embedding_dim:
                raise ValueError(f"Inconsistent embedding dims. Found {embedding_dim} but expected {self.in_dim}")

            # Optional labels stored inside NPZ
            label_arr_npz = None
            if "labels_downsampled" in npz.files:
                label_arr_npz = np.asarray(npz["labels_downsampled"])
            elif "labels_original" in npz.files:
                label_arr_npz = np.asarray(npz["labels_original"])
            elif "labels" in npz.files:
                label_arr_npz = np.asarray(npz["labels"])
            if label_arr_npz is not None and label_arr_npz.ndim > 1:
                label_arr_npz = label_arr_npz.reshape(-1)

            # Align spectrograms if available for envelope computation
            spec_view = None
            if "spectrograms" in npz.files:
                try:
                    spec_view = align_spectrogram_to_embeddings(npz["spectrograms"], embedding_full.shape[0])
                    self._files_with_envelope += 1
                except ValueError as exc:
                    print(f"Warning: {p} spectrogram alignment failed: {exc}")
                    spec_view = None

            # Apply downsampling consistently
            if self.downsample > 1:
                embedding_view = embedding_full[::self.downsample]
                if label_arr_npz is not None:
                    label_arr_npz = label_arr_npz[::self.downsample]
                if spec_view is not None:
                    spec_view = spec_view[::self.downsample]
            else:
                embedding_view = embedding_full

            # Derive envelope-based labels if spectrogram available
            envelope_labels = None
            envelope_stats = None
            if spec_view is not None:
                envelope_labels, envelope_stats = compute_power_envelope_labels(
                    spec_view,
                    smoothing=self.envelope_smoothing,
                    percentile=self.envelope_percentile,
                    min_frames=self.envelope_min_frames,
                    pad=self.envelope_pad,
                    abs_threshold=self.envelope_abs_threshold,
                )
                self._envelope_segments += envelope_stats["segments"]
                self._envelope_positive_frames += envelope_stats["positive_frames"]
                if envelope_stats["segments"] > 0:
                    self.using_envelope = True

            label_arr = None
            label_source = None
            if envelope_labels is not None:
                if envelope_labels.shape[0] != embedding_view.shape[0]:
                    raise ValueError(f"Envelope label length {envelope_labels.shape[0]} mismatch with embeddings {embedding_view.shape[0]}")
                label_arr = envelope_labels
                label_source = "power_envelope"
            elif label_arr_npz is not None:
                if label_arr_npz.shape[0] != embedding_view.shape[0]:
                    raise ValueError(f"Label length {label_arr_npz.shape[0]} mismatch with embeddings {embedding_view.shape[0]}")
                label_arr = label_arr_npz.astype(np.int64, copy=False)
                label_source = "npz_labels"

            has_labels = label_arr is not None
            file_index = len(self._files)
            self._files.append({
                "path": p,
                "embedding": embedding_view,
                "has_labels": has_labels,
                "labels": label_arr,
                "label_source": label_source,
                "envelope_stats": envelope_stats,
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
        self.envelope_summary = {
            "using_envelope": bool(self.using_envelope),
            "segments": int(self._envelope_segments),
            "positive_frames": int(self._envelope_positive_frames),
            "files_with_envelope": int(self._files_with_envelope),
        }
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


def envelope_contrastive_loss(proj: torch.Tensor,
                              labels: torch.Tensor,
                              temperature: float = 0.07) -> Tuple[Optional[torch.Tensor], int]:
    """
    Contrastive loss where positives share the same power-envelope label.
    labels: (N,) with -1 for distractors. Returns (loss, positive_pairs).
    """
    device = proj.device
    labels = labels.to(device)
    valid = labels >= 0
    if torch.count_nonzero(valid) <= 1:
        return None, 0

    features = nn.functional.normalize(proj, dim=1)
    logits = features @ features.t() / float(temperature)
    logits = logits.masked_fill(torch.eye(logits.shape[0], device=device, dtype=torch.bool), -1e9)

    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    positive_mask = same_label & valid.unsqueeze(0) & valid.unsqueeze(1)
    positive_mask.fill_diagonal_(False)

    positives_per_anchor = positive_mask.sum(dim=1)
    valid_anchors = positives_per_anchor > 0
    if not torch.any(valid_anchors):
        return None, 0

    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_mask_f = positive_mask.float()
    losses = -(log_prob * positive_mask_f).sum(dim=1) / positives_per_anchor.clamp(min=1)
    loss = losses[valid_anchors].mean()
    positive_pairs = int(positives_per_anchor[valid_anchors].sum().item())
    return loss, positive_pairs


# --- Training logic ---
def collate_windows(batch):
    # batch: list of (window_np, labels, meta)
    windows = [item[0] for item in batch]
    arr = np.stack(windows, axis=0)
    label_list = [item[1] for item in batch]
    if all(lbl is not None for lbl in label_list):
        labels = np.stack(label_list, axis=0)
    else:
        labels = None
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
               temperature: float = 0.07,
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
            positive_pairs = None
            if labels is not None:
                labels_np = np.asarray(labels, dtype=np.int64)
                labels_t = torch.from_numpy(labels_np).to(device)
                labels_flat = labels_t.view(-1)
                loss_tuple = envelope_contrastive_loss(proj, labels_flat, temperature=temperature)
                if loss_tuple[0] is None:
                    continue
                loss, positive_pairs = loss_tuple
            else:
                loss = nnclr_loss(proj, temperature=temperature, distractors=distractors)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            history.append(float(loss.item()))
            if step % 20 == 0:
                extra = f" pos_pairs={positive_pairs}" if positive_pairs is not None else ""
                print(f"[ep {ep+1}/{epochs}] step {step} loss {loss.item():.4f}{extra}")
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
    p.add_argument("--temperature", type=float, default=0.07,
                   help="Softmax temperature for contrastive loss.")
    p.add_argument("--envelope_percentile", type=float, default=70.0,
                   help="Percentile of power used to define active envelope regions (ignored if envelope_threshold is provided).")
    p.add_argument("--envelope_threshold", type=float, default=None,
                   help="Absolute power threshold overriding percentile for envelope segmentation.")
    p.add_argument("--envelope_smoothing", type=int, default=5,
                   help="Moving-average window (frames) for power envelope smoothing before segmentation.")
    p.add_argument("--envelope_min_frames", type=int, default=4,
                   help="Minimum consecutive frames (after downsampling) for a positive envelope segment.")
    p.add_argument("--envelope_pad", type=int, default=0,
                   help="Pad (frames) applied to both sides of detected envelope segments before merging.")
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
    dataset = NpzSlidingWindowDataset(
        args.npz,
        window_len=args.window_len,
        stride=args.stride,
        downsample=args.downsample,
        envelope_percentile=args.envelope_percentile,
        envelope_abs_threshold=args.envelope_threshold,
        envelope_smoothing=args.envelope_smoothing,
        envelope_min_frames=args.envelope_min_frames,
        envelope_pad=args.envelope_pad,
    )
    print("Windows collected:", len(dataset))
    in_dim = dataset.get_in_dim()
    print("Embedding dim:", in_dim)
    if getattr(dataset, "envelope_summary", None):
        summary = dataset.envelope_summary
        if summary["using_envelope"]:
            print(f"Power envelope positives: {summary['segments']} segments, {summary['positive_frames']} frames across {summary['files_with_envelope']} files")
            if summary["positive_frames"] == 0:
                print("Warning: no positive frames found with current envelope settings.")

    head, history = train_loop(dataset, in_dim, args.outdir, device,
                               epochs=args.epochs,
                               batch_windows=args.batch_windows,
                               lr=args.lr,
                               distractors=args.distractors,
                               temperature=args.temperature)

    projs, labels = collect_timestep_projections(head, dataset, device, n_steps=args.umap_n_steps)
    umap_path = os.path.join(args.outdir, "umap_windows.png")
    try_umap_and_plot(projs, umap_path, labels=labels)
    print("Done.")


if __name__ == "__main__":
    main()
