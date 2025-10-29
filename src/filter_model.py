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
        required=False,  # will be True temporarily for testing
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
        "--iterations",
        type=int,
        default=1,
        help="Number of filter iterations to perform (each pass recomputes losses after masking blocks)",
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


def _prepare_mask(filtered_mask: np.ndarray | None, n_blocks: int) -> np.ndarray:
    if filtered_mask is None:
        return np.zeros(n_blocks, dtype=bool)
    mask = np.asarray(filtered_mask, dtype=bool)
    if mask.shape[0] != n_blocks:
        raise ValueError("filtered_mask must have length equal to number of blocks.")
    return mask


def _gather_context_indices(last_block: int, ctx_len: int, mask: np.ndarray) -> list[int] | None:
    if ctx_len <= 0:
        return []
    n_blocks = mask.shape[0]
    indices: list[int] = []
    current = last_block
    visited = 0
    while len(indices) < ctx_len and visited < n_blocks:
        if not mask[current]:
            indices.append(current)
        current = (current - 1) % n_blocks
        visited += 1
    if len(indices) < ctx_len:
        return None
    indices.reverse()
    return indices


def _build_impact_lookup(n_blocks: int, effective_context: int, mask: np.ndarray) -> list[list[tuple[int, int]]]:
    lookup: list[list[tuple[int, int]]] = [[] for _ in range(n_blocks)]
    if effective_context <= 1:
        return lookup
    for last_block in range(n_blocks):
        if mask[last_block]:
            continue
        for ctx_idx in range(1, effective_context):
            ctx_len = ctx_idx + 1
            context = _gather_context_indices(last_block, ctx_len, mask)
            if context is None:
                continue
            block = context[0]
            if mask[block]:
                continue
            lookup[block].append((last_block, ctx_idx))
    return lookup


def _to_numpy(array, dtype=None):
    if array is None:
        return None
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    else:
        array = np.asarray(array)
    if dtype is not None and array is not None:
        array = array.astype(dtype)
    return array


def load_saved_state(path: Path, n_blocks: int, context_length: int) -> dict:
    if not path.exists():
        return {}
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    result: dict = {}
    mask_arr = _to_numpy(data.get("mask"))
    if mask_arr is not None and mask_arr.size == n_blocks:
        result["mask"] = mask_arr.astype(bool)
    saved_context = data.get("context_length")
    saved_n_blocks = data.get("n_blocks", n_blocks)
    if (
        saved_context is not None
        and int(saved_context) == int(context_length)
        and int(saved_n_blocks) == int(n_blocks)
    ):
        loss_matrix = _to_numpy(data.get("loss_matrix"), dtype=np.float64)
        if loss_matrix is not None and loss_matrix.shape[0] == n_blocks:
            result["loss_matrix"] = loss_matrix
            saved_eff = data.get("effective_context")
            if saved_eff is not None:
                result["effective_context"] = int(saved_eff)
    return result


def save_filter_state(
    path: Path,
    mask: np.ndarray,
    context_length: int,
    effective_context: int | None,
    initial_loss_matrix: np.ndarray | None,
    n_blocks: int,
):
    payload: dict = {
        "mask": torch.as_tensor(mask.astype(np.int64)),
        "context_length": int(context_length),
        "effective_context": int(effective_context or 0),
        "n_blocks": int(n_blocks),
    }
    if initial_loss_matrix is not None:
        payload["loss_matrix"] = torch.as_tensor(initial_loss_matrix, dtype=torch.float32)
    torch.save(payload, path)


def compute_loss_matrix(
    model,
    x,
    x_i,
    N,
    device,
    requested_context: int,
    filtered_mask: np.ndarray | None = None,
    iteration: int | None = None,
) -> tuple[np.ndarray, int]:
    n_blocks = int(N.max().item())
    if n_blocks <= 0:
        return np.empty((0, requested_context), dtype=np.float64), 0

    loss_matrix = np.full((n_blocks, requested_context), np.nan, dtype=np.float64)
    mask = _prepare_mask(filtered_mask, n_blocks)
    available_blocks = int(np.count_nonzero(~mask))
    effective_context = min(requested_context, available_blocks) if available_blocks > 0 else 0

    if effective_context == 0:
        return loss_matrix, 0

    combos: list[tuple[int, int, int]] = []
    for last_block in range(n_blocks):
        if mask[last_block]:
            continue
        for ctx_idx in range(effective_context):
            ctx_len = ctx_idx + 1
            context = _gather_context_indices(last_block, ctx_len, mask)
            if context is None:
                continue
            start_block = context[0]
            combos.append((last_block, ctx_idx, start_block))

    if not combos:
        return loss_matrix, effective_context

    exclude_blocks = np.where(mask)[0].tolist()
    iter_label = f" (iteration {iteration})" if iteration is not None else ""
    desc = f"Computing losses{iter_label}"

    with torch.no_grad(), tqdm(total=len(combos), desc=desc, leave=False) as pbar:
        for last_block, ctx_idx, start_block in combos:
            loss, *_ = compute_loss(
                model,
                x,
                x_i,
                N,
                start_block,
                last_block,
                exclude_blocks=exclude_blocks if exclude_blocks else None,
            )
            loss_matrix[last_block, ctx_idx] = _safe_float(loss)
            pbar.update(1)

    return loss_matrix, effective_context


def compute_expected_metrics(loss_matrix: np.ndarray, effective_context: int):
    rows, cols = loss_matrix.shape
    expected_loss = np.full(cols, np.nan, dtype=np.float64)
    actual_benefit = np.full((rows, cols), np.nan, dtype=np.float64)
    expected_diff = np.full(cols, np.nan, dtype=np.float64)
    expected_diff_std = np.full(cols, np.nan, dtype=np.float64)

    if cols > 0:
        first_col = loss_matrix[:, 0]
        actual_benefit[:, 0] = first_col - 1.0

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
    filtered_mask: np.ndarray | None = None,
) -> np.ndarray:
    n_blocks, _ = actual_benefit.shape
    mask = _prepare_mask(filtered_mask, n_blocks)
    impact = np.zeros(n_blocks, dtype=np.float64)

    if effective_context <= 1:
        return impact

    lookup = _build_impact_lookup(n_blocks, effective_context, mask)

    for block, targets in enumerate(lookup):
        if mask[block] or not targets:
            continue
        for last_block, ctx_idx in targets:
            benefit = actual_benefit[last_block, ctx_idx]
            if not np.isfinite(benefit):
                continue
            exp = expected_diff[ctx_idx] if np.isfinite(expected_diff[ctx_idx]) else 0.0
            std = expected_diff_std[ctx_idx] if np.isfinite(expected_diff_std[ctx_idx]) else 0.0
            adjusted = exp + threshold * std
            impact[block] += benefit - adjusted

    return impact


def run_test_mode(threshold: float):
    print("\n=== Demo: compute_block_impacts ===")
    effective_context = 6  # columns correspond to context lengths 1→6
    loss_matrix = np.array(
        [
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
            [1.1, 0.9, 0.7, 0.5, 0.3, 0.1],
            [1.0, 0.9, 0.7, 0.5, 0.3, 0.1],
            [1.0, 0.8, 0.7, 0.5, 0.3, 0.1],
            [1.0, 0.8, 0.6, 0.5, 0.3, 0.1],
            [1.0, 0.8, 0.6, 0.4, 0.3, 0.1],
        ],
        dtype=np.float64,
    )
    n_blocks = loss_matrix.shape[0]

    print(f"loss_matrix (rows=predicted block, cols=context length 1→{effective_context}):")
    print(loss_matrix)

    expected_loss, expected_diff, expected_diff_std, actual_benefit = compute_expected_metrics(
        loss_matrix, effective_context
    )

    print("\nexpected_loss by context length:", expected_loss.tolist())
    print("expected_diff (mean delta when adding a block):", expected_diff.tolist())
    print("expected_diff_std (std of that delta):", expected_diff_std.tolist())
    print("\nactual_benefit (per predicted block, per delta context):")
    print(actual_benefit)
    print(f"\nUsing threshold = {threshold}\n")

    target_idx = effective_context - 1
    loss_at_target = loss_matrix[:, target_idx]
    threshold_loss = expected_loss[target_idx]
    loss_condition = np.isfinite(loss_at_target) & (loss_at_target > threshold_loss)
    print(f"Loss at target context (len={target_idx + 1}): {loss_at_target.tolist()}")
    print(f"Expected loss threshold @context {target_idx + 1}: {threshold_loss}")
    print("Blocks above threshold:", np.where(loss_condition)[0].tolist(), "\n")

    manual = np.zeros(n_blocks, dtype=np.float64)
    mask_demo = np.zeros(n_blocks, dtype=bool)
    lookup_demo = _build_impact_lookup(n_blocks, effective_context, mask_demo)
    for block, targets in enumerate(lookup_demo):
        print(f"Block {block}:")
        if not targets:
            print("  no eligible contexts")
        for last_block, ctx_idx in targets:
            ctx_len = ctx_idx + 1
            context = _gather_context_indices(last_block, ctx_len, mask_demo) or []
            benefit = actual_benefit[last_block, ctx_idx]
            if not np.isfinite(benefit):
                print(f"  ctx_len={ctx_len}, last_block={last_block}, context={context}: skip (NaN benefit)")
                continue
            exp = expected_diff[ctx_idx] if np.isfinite(expected_diff[ctx_idx]) else 0.0
            std = expected_diff_std[ctx_idx] if np.isfinite(expected_diff_std[ctx_idx]) else 0.0
            adjusted = exp + threshold * std
            contribution = benefit - adjusted
            manual[block] += contribution
            print(
                f"  ctx_len={ctx_len}, last_block={last_block}, context={context}: benefit={benefit:+.3f}, "
                f"expected={exp:+.3f}, std={std:+.3f}, adjusted={adjusted:+.3f}, contribution={contribution:+.3f}"
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
    penalty_condition = manual > 0.0
    filtered = loss_condition & penalty_condition
    print("Blocks with positive penalty:", np.where(penalty_condition)[0].tolist())
    print("=> Blocks to filter (loss condition ∧ penalty):", np.where(filtered)[0].tolist())
    print("=== End Demo ===\n")


def process_file(model, dataset, index, device, args, output_dir: Path):
    x, x_i, _, N, filename = prepare_sample(model, dataset, index, device)
    n_blocks = int(N.item())
    print(f"\nProcessing file {index}: {filename} (blocks={n_blocks})")
    output_path = build_output_path(dataset, index, output_dir)

    if n_blocks == 0:
        print("  No blocks available in sample; emitting zero mask.")
        empty_mask = np.zeros(0, dtype=bool)
        save_filter_state(
            output_path,
            empty_mask,
            args.context_length,
            effective_context=0,
            initial_loss_matrix=None,
            n_blocks=0,
        )
        print(f"  Saved results ({output_path.name})")
        return

    filtered_mask = np.zeros(n_blocks, dtype=bool)
    initial_loss_matrix: np.ndarray | None = None
    initial_effective_context: int | None = None

    cached_state = load_saved_state(output_path, n_blocks, args.context_length)
    cached_loss_matrix = cached_state.get("loss_matrix")
    cached_effective_context = cached_state.get("effective_context")

    if cached_loss_matrix is not None:
        print("  Reusing cached loss matrix from previous run.")
        loss_matrix = cached_loss_matrix.copy()
        effective_context = int(cached_effective_context or args.context_length)
        effective_context = max(0, min(effective_context, loss_matrix.shape[1]))
        initial_loss_matrix = cached_loss_matrix.copy()
        initial_effective_context = effective_context
    else:
        loss_matrix, effective_context = compute_loss_matrix(
            model,
            x,
            x_i,
            N,
            device,
            args.context_length,
            filtered_mask=filtered_mask,
            iteration=1,
        )
        initial_loss_matrix = loss_matrix.copy()
        initial_effective_context = effective_context

    iteration = 1
    max_iterations = max(1, args.iterations)
    while iteration <= max_iterations:
        print(f"\n  Iteration {iteration}/{max_iterations}")

        available_blocks = int(np.count_nonzero(~filtered_mask))
        if effective_context < args.context_length:
            print(
                f"    Context truncated to {effective_context} (available unfiltered blocks: {available_blocks})"
            )

        if effective_context == 0:
            print("    Insufficient unfiltered blocks to compute losses; stopping.")
            break

        expected_loss, expected_diff, expected_diff_std, actual_benefit = compute_expected_metrics(
            loss_matrix, effective_context
        )

        impacts = compute_block_impacts(
            actual_benefit,
            expected_diff,
            expected_diff_std,
            effective_context,
            args.threshold,
            filtered_mask=filtered_mask,
        )

        target_idx = effective_context - 1
        threshold_loss = expected_loss[target_idx] if target_idx >= 0 else float("nan")
        loss_at_target = loss_matrix[:, target_idx] if target_idx >= 0 else np.full(n_blocks, np.nan)

        loss_condition = np.zeros(n_blocks, dtype=bool)
        if np.isfinite(threshold_loss):
            candidate = np.isfinite(loss_at_target) & (loss_at_target > threshold_loss)
            loss_condition = candidate & (~filtered_mask)

        penalty_condition = (impacts > 0.0) & (~filtered_mask)
        filtered_this_iter = loss_condition & penalty_condition

        print(f"    Expected loss (first 5 contexts): {np.round(expected_loss[:5], 4).tolist()}")
        print(f"    Expected diff (first 5 contexts): {np.round(expected_diff[:5], 4).tolist()}")
        print(f"    Expected diff std (first 5): {np.round(expected_diff_std[:5], 4).tolist()}")
        print(f"    Blocks above expected loss @ctx={target_idx + 1}: {np.where(loss_condition)[0].tolist()}")
        print(f"    Blocks with positive penalty: {np.where(penalty_condition)[0].tolist()}")
        print(f"    Newly filtered blocks: {np.where(filtered_this_iter)[0].tolist()}")

        valid_before = (~filtered_mask) & np.isfinite(loss_at_target)
        before_mean = float(np.nan) if target_idx < 0 or not np.any(valid_before) else float(
            np.nanmean(loss_at_target[valid_before])
        )
        print(
            f"    Mean loss @ max context before filtering: {before_mean:.4f}"
            if np.isfinite(before_mean)
            else "    Mean loss @ max context before filtering: nan"
        )

        if not filtered_this_iter.any():
            print("    No new blocks met filtering criteria; stopping iterations.")
            if np.isfinite(before_mean):
                print(f"    Mean loss @ max context after filtering: {before_mean:.4f}")
            else:
                print("    Mean loss @ max context after filtering: nan")
            break

        filtered_mask |= filtered_this_iter

        loss_matrix_after, effective_context_after = compute_loss_matrix(
            model,
            x,
            x_i,
            N,
            device,
            args.context_length,
            filtered_mask=filtered_mask,
            iteration=iteration + 1,
        )

        if effective_context_after > 0:
            loss_after_target_idx = effective_context_after - 1
            loss_after = loss_matrix_after[:, loss_after_target_idx]
            valid_after = (~filtered_mask) & np.isfinite(loss_after)
            after_mean = (
                float(np.nanmean(loss_after[valid_after])) if np.any(valid_after) else float("nan")
            )
        else:
            after_mean = float("nan")

        if np.isfinite(after_mean):
            print(f"    Mean loss @ max context after filtering: {after_mean:.4f}")
        else:
            print("    Mean loss @ max context after filtering: nan")

        print(f"    Cumulative filtered blocks: {np.where(filtered_mask)[0].tolist()}")

        loss_matrix = loss_matrix_after
        effective_context = effective_context_after
        iteration += 1

    save_filter_state(
        output_path,
        filtered_mask,
        args.context_length,
        effective_context=initial_effective_context,
        initial_loss_matrix=initial_loss_matrix,
        n_blocks=n_blocks,
    )
    print(f"\n  Saved results ({output_path.name})")


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
        run_dir=args.model_path, checkpoint_file=args.checkpoint, fallback_to_random=False
    )

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        data_dir = config.get("val_dir")
        if data_dir is None:
            raise ValueError("No data directory specified. Provide --data_dir or configure 'val_dir' in config.json.")

    dataset = SpectogramDataset(
        dir=data_dir, n_mels=config.get("mels", 128), n_timebins=config.get("num_timebins", 1024), pad_crop=True
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
