#!/usr/bin/env python3
"""
Filter TinyBird spectrogram blocks by measuring their contextual impact.

This script builds the loss tables used in analyze_model.py (restricted to a
user-specified context length), derives expected loss/benefit statistics, and
marks blocks for removal when they both underperform relative to expectation
and make neighbouring predictions worse.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data_loader import SpectogramDataset
from utils import load_model_from_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze TinyBird model on spectrogram data")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint directory (containing config.json and weights/)",  # fmt: skip
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint file to load (e.g., model_step_005000.pth). If not specified, loads the latest checkpoint.",  # fmt: skip
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to directory containing .pt spectrogram files (default: uses val_dir from config.json)",  # fmt: skip
    )
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="Index of the spectrogram file to analyze (>=0). If negative, process ALL files (default: -1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda if available, else cpu)",  # fmt: skip
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=50,
        help="Maximum number of preceding blocks (per prediction) to evaluate when building the loss array",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Standard-deviation offset applied to the expected benefit when scoring block penalties",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a self-contained demonstration of block impact computation using tiny synthetic data",
    )
    return parser.parse_args()


def process_xl(x_i, x_l, N, device):
    x_i = x_i.to(device, non_blocking=True).long()
    x_l = x_l.to(device, non_blocking=True).float()

    batch = x_i.shape[0]
    max_N = int(N.max().item())
    x_l_out = torch.zeros((batch, max_N), dtype=torch.float32, device=device)

    for b in range(batch):
        n_valid = int(N[b].item())
        for block in range(n_valid):
            start, end = x_i[b, block, 0].item(), x_i[b, block, 1].item()
            x_l_out[b, block] = x_l[b, start:end].mean()

    return x_l_out


def prepare_sample(model, dataset, index, device):
    x, x_i, x_l, _, N, filename = dataset[index]
    x = x.unsqueeze(0).float().to(device)
    x_i = x_i.unsqueeze(0).to(device)
    x_l = x_l.unsqueeze(0).to(device).float()
    N = torch.tensor([N], dtype=torch.long, device=device)

    x_l_mean = process_xl(x_i, x_l, N, device)
    x, x_i = model.compactify_data(x.clone(), x_i.clone(), N.clone())

    return x, x_i, x_l_mean, N, filename


def compute_loss(model, x, x_i, N, start_block, last_block, exclude_blocks=None):
    """
    Compute masked MSE loss for a given block configuration.
    Returns: (loss, channel_losses, dt, lt, xs, x_is, bool_mask, pred, mblock, indices)
    """
    total_blocks = int(N.max().item())
    exclude_set = set(exclude_blocks or [])
    if start_block in exclude_set or last_block in exclude_set:
        raise ValueError("start_block or last_block is in the excluded set.")

    def advance(idx):
        return (idx + 1) % total_blocks

    indices = []
    current = start_block
    visited = 0
    while True:
        if current not in exclude_set:
            indices.append(current)
        if current == last_block:
            break
        current = advance(current)
        visited += 1
        if visited > total_blocks + len(exclude_set):
            raise RuntimeError("Failed to reach last_block without looping indefinitely.")

    if not indices or indices[-1] != last_block:
        raise RuntimeError("Invalid context path computed for compute_loss.")

    mblock = [len(indices) - 1]
    dt = 0.0
    lt = 0.0
    if len(indices) >= 1:
        starts = x_i[0, indices, 0]
        ends = x_i[0, indices, 1]
        durations = ends - starts
        lt = float(durations.sum().item())
        if len(indices) > 1:
            gaps = starts[1:] - ends[:-1]
            dt = float(gaps.sum().item())

    if len(indices) == 1:
        nan = torch.tensor(float("nan"), device=x.device)
        return nan, nan, nan, dt, lt, None, None, None, None, None, None

    try:
        xs, x_is = model.sample_data_indices(x.clone(), x_i.clone(), N.clone(), indices)
        h, idx_restore, bool_mask, T = model.forward_encoder(xs, x_is, mblock=mblock)
        pred = model.forward_decoder(h, idx_restore, T)
        reconstruction_loss, channel_losses = model.loss_mse(xs, pred, bool_mask, return_per_channel=True)
        return reconstruction_loss, channel_losses, dt, lt, xs, x_is, bool_mask, pred, mblock, indices
    except RuntimeError as exc:
        msg = str(exc).lower()
        if ("out of memory" in msg or "cuda" in msg) and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            nan = torch.tensor(float("nan"), device=x.device)
            return nan, nan, nan, dt, lt, None, None, None, None, None, None
        raise


def _safe_float(value: float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Expected scalar tensor for loss value.")
        scalar = float(value.detach().item())
    else:
        scalar = float(value)
    if math.isnan(scalar):
        return float("nan")
    return scalar


def compute_loss_matrix(
    model,
    x,
    x_i,
    N,
    device,
    requested_context: int,
) -> tuple[np.ndarray, int]:
    n_blocks = int(N.max().item())
    if n_blocks <= 0:
        return np.empty((0, requested_context), dtype=np.float64), 0

    loss_matrix = np.full((n_blocks, requested_context), np.nan, dtype=np.float64)
    effective_context = min(requested_context, max(n_blocks - 1, 0))
    total_jobs = n_blocks * effective_context

    if effective_context == 0:
        return loss_matrix, effective_context

    desc = "Computing loss matrix"
    with torch.no_grad(), tqdm(total=total_jobs, desc=desc, leave=False) as pbar:
        for last_block in range(n_blocks):
            for ctx_idx in range(effective_context):
                ctx_len = ctx_idx + 1
                start_block = (last_block - ctx_len) % n_blocks
                loss, *_ = compute_loss(model, x, x_i, N, start_block, last_block)
                loss_matrix[last_block, ctx_idx] = _safe_float(loss)
                pbar.update(1)
    return loss_matrix, effective_context


def compute_expected_metrics(loss_matrix: np.ndarray, effective_context: int):
    rows, cols = loss_matrix.shape
    expected_loss = np.full(cols, np.nan, dtype=np.float64)
    actual_benefit = np.full((rows, cols), np.nan, dtype=np.float64)
    expected_diff = np.full(cols, np.nan, dtype=np.float64)
    expected_diff_std = np.full(cols, np.nan, dtype=np.float64)

    for ctx_idx in range(effective_context):
        column = loss_matrix[:, ctx_idx]
        finite = column[np.isfinite(column)]
        if finite.size:
            expected_loss[ctx_idx] = float(np.mean(finite))

    if effective_context > 0:
        expected_diff[0] = 0.0
        expected_diff_std[0] = 0.0

    for ctx_idx in range(1, effective_context):
        prev_col = loss_matrix[:, ctx_idx - 1]
        cur_col = loss_matrix[:, ctx_idx]
        valid_mask = np.isfinite(prev_col) & np.isfinite(cur_col)
        if not np.any(valid_mask):
            continue
        diffs = cur_col[valid_mask] - prev_col[valid_mask]
        actual_benefit[valid_mask, ctx_idx] = diffs
        expected_diff[ctx_idx] = float(np.mean(diffs))
        expected_diff_std[ctx_idx] = float(np.std(diffs))

    return expected_loss, expected_diff, expected_diff_std, actual_benefit


def compute_block_impacts(
    actual_benefit: np.ndarray,
    expected_diff: np.ndarray,
    expected_diff_std: np.ndarray,
    effective_context: int,
    threshold: float,
) -> np.ndarray:
    n_blocks, _ = actual_benefit.shape
    impact = np.zeros(n_blocks, dtype=np.float64)

    for block in range(n_blocks):
        for ctx_idx in range(1, effective_context):
            ctx_len = ctx_idx + 1
            last_block = (block + ctx_len) % n_blocks
            benefit = actual_benefit[last_block, ctx_idx]
            if not np.isfinite(benefit):
                continue
            exp = expected_diff[ctx_len] if np.isfinite(expected_diff[ctx_len]) else 0.0
            std = expected_diff_std[ctx_len] if np.isfinite(expected_diff_std[ctx_len]) else 0.0
            adjusted = exp + threshold * std
            diff = benefit - adjusted
            impact[block] += diff

    return impact


def run_test_mode(threshold: float):
    print("\n=== Demo: compute_block_impacts ===")
    effective_context = 3  # columns correspond to context lengths 1, 2, 3
    n_blocks = 4

    actual_benefit = np.array(
        [
            [np.nan, 0.20, -0.15],
            [np.nan, -0.05, 0.10],
            [np.nan, 0.30, -0.25],
            [np.nan, -0.10, 0.05],
        ],
        dtype=np.float64,
    )
    expected_diff = np.array([0.0, 0.12, -0.06], dtype=np.float64)
    expected_diff_std = np.array([0.0, 0.03, 0.08], dtype=np.float64)

    print("actual_benefit (rows=predicted block, cols=ctx length-1):")
    print(actual_benefit)
    print("\nexpected_diff:", expected_diff.tolist())
    print("expected_diff_std:", expected_diff_std.tolist())
    print(f"\nUsing threshold = {threshold}\n")

    manual = np.zeros(n_blocks, dtype=np.float64)
    for block in range(n_blocks):
        print(f"Block {block}:")
        for ctx_idx in range(1, effective_context):
            ctx_len = ctx_idx + 1  # number of preceding blocks under evaluation
            last_block = (block + ctx_len) % n_blocks
            benefit = actual_benefit[last_block, ctx_idx]
            if not np.isfinite(benefit):
                print(f"  ctx_len={ctx_len}: skip (NaN benefit)")
                continue
            exp = expected_diff[ctx_idx] if np.isfinite(expected_diff[ctx_idx]) else 0.0
            std = expected_diff_std[ctx_idx] if np.isfinite(expected_diff_std[ctx_idx]) else 0.0
            adjusted = exp + threshold * std
            contribution = benefit - adjusted
            manual[block] += contribution
            print(
                f"  ctx_len={ctx_len}: predict block={last_block}, benefit={benefit:+.3f}, "
                f"expected={exp:+.3f}, std={std:+.3f}, adjusted={adjusted:+.3f}, "
                f"contribution={contribution:+.3f}"
            )
        print(f"  => cumulative impact for block {block}: {manual[block]:+.3f}\n")

    computed = compute_block_impacts(
        actual_benefit=actual_benefit,
        expected_diff=expected_diff,
        expected_diff_std=expected_diff_std,
        effective_context=effective_context,
        threshold=threshold,
    )

    print("Manual accumulation:", manual.tolist())
    print("Function result   :", computed.tolist())
    print("Match?", np.allclose(manual, computed))
    print("=== End Demo ===\n")


def process_file(model, dataset, index, device, args, output_dir: Path):
    x, x_i, _, N, filename = prepare_sample(model, dataset, index, device)
    n_blocks = int(N.item())
    print(f"\nProcessing file {index}: {filename} (blocks={n_blocks})")

    loss_matrix, effective_context = compute_loss_matrix(
        model,
        x,
        x_i,
        N,
        device,
        args.context_length,
    )

    if effective_context < args.context_length:
        print(
            f"  Requested context_length {args.context_length} truncated to {effective_context} "
            f"because only {n_blocks} blocks are available."
        )

    if effective_context == 0:
        print("  Not enough blocks to compute contextual losses; emitting zero mask.")
        mask = np.zeros(n_blocks, dtype=np.int64)
        output_path = build_output_path(dataset, index, output_dir)
        torch.save({"mask": torch.as_tensor(mask, dtype=torch.int64)}, output_path)
        print(f"  Saved mask ({output_path.name})")
        return

    expected_loss, expected_diff, expected_diff_std, actual_benefit = compute_expected_metrics(
        loss_matrix, effective_context
    )
    impacts = compute_block_impacts(
        actual_benefit,
        expected_diff,
        expected_diff_std,
        effective_context,
        args.threshold,
    )

    target_idx = effective_context - 1
    threshold_loss = expected_loss[target_idx]
    loss_at_target = loss_matrix[:, target_idx]

    loss_condition = np.zeros(n_blocks, dtype=bool)
    if np.isfinite(threshold_loss):
        loss_condition = np.isfinite(loss_at_target) & (loss_at_target > threshold_loss)

    penalty_condition = impacts > 0.0
    filtered_mask = loss_condition & penalty_condition

    print(f"  Expected loss (first 5 contexts): {np.round(expected_loss[:5], 4).tolist()}")
    print(f"  Expected diff (first 5 contexts): {np.round(expected_diff[:5], 4).tolist()}")
    print(f"  Expected diff std (first 5): {np.round(expected_diff_std[:5], 4).tolist()}")
    print(f"  Blocks above expected loss @ctx={target_idx + 1}: {np.where(loss_condition)[0].tolist()}")
    print(f"  Blocks with positive penalty: {np.where(penalty_condition)[0].tolist()}")
    print(f"  Filtered blocks: {np.where(filtered_mask)[0].tolist()}")

    mask_tensor = torch.as_tensor(filtered_mask.astype(np.int64))
    output_path = build_output_path(dataset, index, output_dir)
    torch.save({"mask": mask_tensor}, output_path)
    print(f"  Saved mask ({output_path.name})")


def build_output_path(dataset, index: int, output_dir: Path) -> Path:
    src_path = dataset.file_dirs[index]
    suffix = src_path.suffix
    name = f"{src_path.stem}_filter{suffix}"
    return output_dir / name


def main():
    args = parse_args()
    if args.test:
        run_test_mode(args.threshold)
        return
    if args.context_length <= 0:
        raise ValueError("context_length must be a positive integer.")

    model, config = load_model_from_checkpoint(
        run_dir=args.model_path,
        checkpoint_file=args.checkpoint,
        fallback_to_random=False,
    )

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        data_dir = config.get("val_dir")
        if data_dir is None:
            raise ValueError(
                "No data directory specified. Provide --data_dir or configure 'val_dir' in config.json."
            )

    dataset = SpectogramDataset(
        dir=data_dir,
        n_mels=config.get("mels", 128),
        n_timebins=config.get("num_timebins", 1024),
        pad_crop=True,
    )

    output_dir = Path(data_dir) / "filter"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.index >= 0:
        if args.index >= len(dataset):
            raise ValueError(f"Index {args.index} out of range (dataset size: {len(dataset)}).")
        process_file(model, dataset, args.index, device, args, output_dir)
    else:
        print(f"Processing all {len(dataset)} files...")
        for idx in range(len(dataset)):
            process_file(model, dataset, idx, device, args, output_dir)


if __name__ == "__main__":
    main()
