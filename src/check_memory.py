#!/usr/bin/env python3
"""
Probe the maximum seq_len (time columns) that fits in GPU memory for a range of batch sizes.

Usage:
  python check_memory.py --config /path/to/config.json \
      --batches 2,3,4,5,6 --start_len 4096 --min_len 256 --cap 262144 --amp

Notes:
- This uses your actual TinyBird.forward_encoder/decoder/loss paths, so results are realistic.
- We bypass compactify/sample_data to probe the full window (worst-case sizing).
- We ensure seq_len is divisible by patch_width (Conv2d stride).
"""

import os
import json
import argparse
import contextlib
from typing import List, Tuple, Dict

import torch
from torch.cuda import OutOfMemoryError

# Import your model
# If you run this from project root, ensure:  PYTHONPATH=src  python tools/probe_max_seq_len.py ...
from model import TinyBird


# ---------- small helpers ----------


@contextlib.contextmanager
def amp_autocast_if(amp_enabled: bool):
    if amp_enabled and torch.cuda.is_available():
        with torch.amp.autocast(device_type="cuda"):
            yield
    else:
        yield


def round_down_to_multiple(x: int, multiple: int) -> int:
    return (x // multiple) * multiple


def try_one_step(
    model: TinyBird, config: Dict, seq_len: int, batch_size: int, device: torch.device
) -> Tuple[bool, int]:
    """
    Do a full forward + loss + backward for a synthetic batch and report if it fits.

    Returns:
      (fits, peak_mem_bytes)
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    H = config["mels"]
    W = seq_len  # time columns (must be divisible by patch_width)
    B = batch_size

    # Synthetic batch that exercises the same shapes your code expects
    # xi: 1 block covering the entire window (so masking paths run)
    x = torch.randn(B, 1, H, W, device=device, dtype=torch.float32)
    xi = torch.tensor([[[0, W]]], device=device, dtype=torch.long).expand(B, 1, 2).contiguous()
    # N unused in probe (forward_encoder signature doesn’t need it)
    # n_blocks=0 and frac=mask_p triggers fractional column masking only
    use_amp = config.get("amp", False)

    model.train()
    model.zero_grad(set_to_none=True)

    try:
        with amp_autocast_if(use_amp):
            h, idx_restore, bool_mask, T = model.forward_encoder(x, xi, mblock=[0])
            pred = model.forward_decoder(h, idx_restore, T)
            loss = model.loss_mse(x, pred, bool_mask)

        if use_amp:
            scaler = torch.amp.GradScaler("cuda")
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # No optimizer step needed — backward is the heavy lift
        del x, xi, h, idx_restore, bool_mask, pred, loss
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        return True, int(peak)

    except OutOfMemoryError:
        torch.cuda.empty_cache()
        return False, 0
    except RuntimeError as e:
        # Catch generic OOM messages thrown by some kernels
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return False, 0
        raise  # real error: bubble up


def find_max_seq_len(
    model: TinyBird, config: Dict, batch_size: int, device: torch.device, min_len: int, start_len: int, max_len_cap: int
) -> Tuple[int, int]:
    """
    Find the largest seq_len that fits for a given batch size.
    Strategy: probe min_len (warmup) -> exp grow to find upper bound -> binary search.

    Returns: (best_seq_len, peak_mem_at_best)
    """
    patch_w = int(config["patch_width"])
    # Ensure candidates are legal for Conv2d stride
    min_len = max(patch_w, round_down_to_multiple(min_len, patch_w))
    start_len = max(patch_w, round_down_to_multiple(start_len, patch_w))
    max_len_cap = max(patch_w, round_down_to_multiple(max_len_cap, patch_w))

    # Warmup tiny pass (helps with kernel init and gives more stable memory)
    try_one_step(model, config, seq_len=min_len, batch_size=batch_size, device=device)

    # 1) Exponentially grow until failure (or cap)
    lo = min_len
    hi = start_len
    fits, _ = try_one_step(model, config, seq_len=hi, batch_size=batch_size, device=device)
    if not fits:
        # If even start_len fails, shrink by halves until it fits or hits min
        while hi > lo:
            hi = max(min_len, round_down_to_multiple(hi // 2, patch_w))
            fits, _ = try_one_step(model, config, seq_len=hi, batch_size=batch_size, device=device)
            if fits:
                lo = hi
                break
        if not fits:
            # Try min_len explicitly one last time
            fits, peak = try_one_step(model, config, seq_len=min_len, batch_size=batch_size, device=device)
            return (min_len if fits else 0), (peak if fits else 0)
    else:
        # Grow
        while hi < max_len_cap:
            nxt = min(max_len_cap, round_down_to_multiple(hi * 2, patch_w))
            ok, _ = try_one_step(model, config, seq_len=nxt, batch_size=batch_size, device=device)
            if ok:
                lo = nxt
                hi = nxt
            else:
                break

    # 2) Binary search (lo fits, hi fails or is just above lo)
    best_len = lo
    best_peak = 0
    left, right = lo, min(max_len_cap, max(lo + patch_w, hi))
    while left + patch_w <= right:
        mid = round_down_to_multiple((left + right) // 2, patch_w)
        if mid == left:
            mid = min(right, left + patch_w)
        ok, peak = try_one_step(model, config, seq_len=mid, batch_size=batch_size, device=device)
        if ok:
            best_len, best_peak = mid, peak
            left = mid
        else:
            right = mid - patch_w

    return best_len, best_peak


def parse_batches_arg(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def pretty_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    i = 0
    while x >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {units[i]}"


# ---------- main ----------


def main():
    ap = argparse.ArgumentParser(description="Probe max seq_len per batch size using TinyBird.")
    ap.add_argument("--config", required=True, type=str, help="Path to JSON config (same format as pretrain.py)")
    ap.add_argument(
        "--batches", type=str, default="8,16,32,64", help="Comma-separated batch sizes to probe, e.g. '8,16,32,64'"
    )
    ap.add_argument("--min_len", type=int, default=256, help="Minimum seq_len (time columns) to start probing")
    ap.add_argument("--start_len", type=int, default=4096, help="Initial seq_len to try before growing/shrinking")
    ap.add_argument("--cap", type=int, default=262144, help="Upper cap on seq_len during search")
    ap.add_argument("--amp", action="store_true", help="Enable AMP during probing (must match your planned training)")
    ap.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., 'cuda:0')")
    args = ap.parse_args()

    # Load config JSON
    with open(args.config, "r") as f:
        config = json.load(f)

    # Respect CLI AMP override for probing (you want this to match training)
    if args.amp:
        config["amp"] = True

    # Required config fields: ensure we have what TinyBird expects
    required = [
        "mels",
        "num_timebins",
        "patch_height",
        "patch_width",
        "enc_hidden_d",
        "enc_n_head",
        "enc_n_layer",
        "enc_dim_ff",
        "dec_hidden_d",
        "dec_n_head",
        "dec_n_layer",
        "dec_dim_ff",
        "dropout",
        "mask_p",
    ]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    # Geometry sanity
    assert config["mels"] % config["patch_height"] == 0, "mels must be divisible by patch_height"
    assert isinstance(config["patch_width"], int) and config["patch_width"] >= 1

    config["patch_size"] = (int(config["patch_height"]), int(config["patch_width"]))

    # We will instantiate the model ONCE with a large enough max_seq so all candidate seq_len fit.
    # Let max_seq correspond to the user-provided --cap.
    Hp = config["mels"] // config["patch_height"]
    capW = round_down_to_multiple(int(args.cap), config["patch_width"])
    config["max_seq"] = (capW // config["patch_width"]) * Hp

    # Create model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARN] CUDA not available; this probe is designed for GPU sizing. Running on CPU will be very slow.")

    model = TinyBird(config).to(device).float()

    # Probe each requested batch size
    batch_sizes = parse_batches_arg(args.batches)

    print(f"\n[probe] Device: {device}, AMP: {bool(config.get('amp', False))}")
    print(f"[probe] Patch: {config['patch_size']}  (H'={Hp}, stride_w={config['patch_width']})")
    print(f"[probe] Cap seq_len: {capW} → max_seq tokens: {config['max_seq']}\n")

    results = []
    for B in batch_sizes:
        best_len, peak = find_max_seq_len(
            model, config, batch_size=B, device=device, min_len=args.min_len, start_len=args.start_len, max_len_cap=capW
        )
        results.append((B, best_len, peak))
        print(f"[probe] batch={B:>5} → max_seq_len={best_len:>7} cols  |  peak={pretty_bytes(peak)}")

    # Summary table
    print("\n=== SUMMARY ===")
    colw = 12
    print(f"{'Batch':>{colw}} | {'Max seq_len (cols)':>{colw}} | {'Peak memory':>{colw}}")
    print("-" * (colw * 3 + 6))
    for B, best_len, peak in results:
        print(f"{B:>{colw}} | {best_len:>{colw}} | {pretty_bytes(peak):>{colw}}")

    # Optional: write JSON report next to the config
    out_path = os.path.splitext(args.config)[0] + "_probe_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "device": str(device),
                "amp": bool(config.get("amp", False)),
                "patch_size": list(config["patch_size"]),
                "cap_seq_len": int(capW),
                "results": [
                    {"batch": int(B), "max_seq_len": int(best_len), "peak_bytes": int(peak)}
                    for (B, best_len, peak) in results
                ],
            },
            f,
            indent=2,
        )
    print(f"\n[probe] Wrote report: {out_path}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # helps for repeated convs
    main()
