#!/usr/bin/env python3
import argparse
import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch.nn.functional as F

# Add `src/` to path because internal modules use absolute imports like `import utils`
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
sys.path.insert(0, _SRC_ROOT)

from data_loader import SpectogramDataset
from utils import load_model_from_checkpoint


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)


def _final_layer_attention_inference(model, x):
    """
    Compute final encoder layer self-attention weights in inference mode.

    We avoid calling `model.encoder(z_seq)` directly because PyTorch may take an
    optimized path that doesn't surface attention weights.

    Returns:
      attn_w: (B, heads, T, T)
      T: number of patch tokens
    """
    if not hasattr(model, "patch_projection") or not hasattr(model, "pos_enc"):
        raise ValueError("Model missing patch projection / positional encoding.")
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layers") or len(model.encoder.layers) == 0:
        raise ValueError("Model does not look like it has an encoder with layers.")

    # Patchify + pos enc (matches TinyBird.forward_encoder_inference)
    z = model.patch_projection(x)  # (B, D_enc, H', W')
    B, D, H, W = z.shape
    pos_enc = model.pos_enc[:, :, :H, :W]
    z = z + pos_enc
    z_seq = z.flatten(2).transpose(1, 2)  # (B, T, D_enc)
    T = z_seq.size(1)

    layers = model.encoder.layers
    y = z_seq
    for layer in layers[:-1]:
        y = layer(y)

    last = layers[-1]

    # TransformerEncoderLayer was created with norm_first=True in src/model.py
    if getattr(last, "norm_first", False):
        y_attn_in = last.norm1(y)
    else:
        y_attn_in = y

    # Get per-head attention weights
    attn_out, attn_w = last.self_attn(
        y_attn_in,
        y_attn_in,
        y_attn_in,
        need_weights=True,
        average_attn_weights=False,
    )

    return attn_w.detach(), T


def _final_encoder_embeddings_inference(model, x):
    """
    Compute final encoder embeddings (per patch token) in inference mode.

    Returns:
      y: (B, T, D)
      T: number of patch tokens
    """
    if not hasattr(model, "patch_projection") or not hasattr(model, "pos_enc"):
        raise ValueError("Model missing patch projection / positional encoding.")
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layers") or len(model.encoder.layers) == 0:
        raise ValueError("Model does not look like it has an encoder with layers.")

    z = model.patch_projection(x)  # (B, D_enc, H', W')
    B, D, H, W = z.shape
    pos_enc = model.pos_enc[:, :, :H, :W]
    z = z + pos_enc
    z_seq = z.flatten(2).transpose(1, 2)  # (B, T, D_enc)
    T = z_seq.size(1)

    y = z_seq
    for layer in model.encoder.layers:
        y = layer(y)

    # TinyBird.encoder was created without a final norm module, so y is final.
    return y, T


def _compute_position_mean_embeddings_2d(
    model,
    dataset,
    indices,
    device,
    patch_h: int,
    patch_w: int,
):
    """
    Compute mean embedding per (y, x) patch position across a set of specs.

    Returns:
      means_np: (grid_h, grid_w, D) float32
      grid_h, grid_w, D
    """
    sums = None
    n_used = 0
    grid_h = grid_w = D = None

    with torch.no_grad():
        for idx in indices:
            x, _ = dataset[idx]  # (1, H, W)
            H = int(x.shape[-2])
            W = int(x.shape[-1])
            gh = H // patch_h
            gw = W // patch_w

            x = x.unsqueeze(0).to(device)  # (1, 1, H, W)
            y, T = _final_encoder_embeddings_inference(model, x)  # (1, T, D)
            y0 = y[0]  # (T, D)
            if gh * gw != T:
                raise RuntimeError(f"Patch grid mismatch during mean calc: {gh}*{gw} != T={T}")

            if sums is None:
                D = int(y0.shape[-1])
                grid_h, grid_w = gh, gw
                sums = np.zeros((grid_h, grid_w, D), dtype=np.float64)
            else:
                if gh != grid_h or gw != grid_w:
                    raise RuntimeError(
                        f"Inconsistent patch grid across specs during mean calc: "
                        f"expected {grid_h}x{grid_w}, got {gh}x{gw}"
                    )

            y_grid = y0.reshape(grid_h, grid_w, D).detach().float().cpu().numpy()
            sums += y_grid
            n_used += 1

    if n_used == 0:
        raise RuntimeError("No specs were used to compute position-mean embeddings.")

    means = (sums / float(n_used)).astype(np.float32)
    return means, grid_h, grid_w, D


def _save_attn_map_png(attn_map_2d: np.ndarray, out_path: str, title: str):
    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    im = ax.imshow(attn_map_2d, origin="lower", aspect="equal", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_attn_overlay_on_spec_png(
    spec_2d: np.ndarray,
    attn_grid: np.ndarray,
    patch_h: int,
    patch_w: int,
    query_y: int,
    query_x: int,
    out_path: str,
    title: str,
    alpha: float,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Overlay a patch-grid scalar map on top of the spectrogram (pixel grid).
    """
    if spec_2d.ndim != 2:
        raise ValueError(f"spec_2d must be 2D, got shape {spec_2d.shape}")
    if attn_grid.ndim != 2:
        raise ValueError(f"attn_grid must be 2D, got shape {attn_grid.shape}")

    H, W = spec_2d.shape
    gh, gw = attn_grid.shape
    if gh * patch_h != H or gw * patch_w != W:
        raise ValueError(
            f"Cannot align attention grid to spectrogram: "
            f"attn_grid={attn_grid.shape}, patch=({patch_h},{patch_w}), spec=({H},{W})"
        )

    # Expand attention grid back to spectrogram resolution
    attn_img = np.repeat(np.repeat(attn_grid, patch_h, axis=0), patch_w, axis=1)

    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.imshow(spec_2d, origin="lower", aspect="auto", cmap="gray")
    im = ax.imshow(
        attn_img,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
    )

    # Highlight the query patch
    rect = Rectangle(
        (query_x * patch_w, query_y * patch_h),
        patch_w,
        patch_h,
        linewidth=2.0,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(rect)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize final-layer encoder attention for random latent tokens/patches."
    )
    parser.add_argument("--run_dir", required=True, type=str, help="Run directory or name under ../runs")
    parser.add_argument("--spec_dir", required=True, type=str, help="Directory containing spectrogram .npy + audio_params.json")
    parser.add_argument("--out_dir", required=True, type=str, help="Directory to save PNG attention maps")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint filename (defaults to latest)")
    parser.add_argument("--num_specs", type=int, default=50, help="How many spectrograms to process")
    parser.add_argument("--queries_per_spec", type=int, default=8, help="Random query tokens per spectrogram")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling specs and queries")
    parser.add_argument("--max_heads", type=int, default=None, help="Optionally limit number of heads visualized (default: all)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for overlaying attention on spectrogram (default: 0.5)")
    parser.add_argument(
        "--mode",
        type=str,
        default="attn",
        choices=["attn", "cosine"],
        help="Visualization mode: 'attn' (final layer attention) or 'cosine' (cosine sim in final embeddings).",
    )
    parser.add_argument(
        "--cosine_pos_mean_specs",
        type=int,
        default=1000,
        help="(cosine mode) If >0, subtract mean embedding per (y,x) position computed over this many specs before cosine sim.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, config = load_model_from_checkpoint(
        run_dir=args.run_dir,
        checkpoint_file=args.checkpoint,
        fallback_to_random=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Dataset (same conventions used elsewhere in repo)
    dataset = SpectogramDataset(dir=args.spec_dir, n_timebins=config["num_timebins"])
    total = min(args.num_specs, len(dataset))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose which specs to process (random subset)
    spec_indices = list(range(len(dataset)))
    random.shuffle(spec_indices)
    spec_indices = spec_indices[:total]

    patch_h = int(config["patch_size"][0])
    patch_w = int(config["patch_size"][1])

    pos_mean_tensor = None
    pos_mean_gh = None
    pos_mean_gw = None

    if args.mode == "cosine" and int(args.cosine_pos_mean_specs) > 0:
        mean_count = min(int(args.cosine_pos_mean_specs), len(dataset))
        mean_indices = list(range(len(dataset)))
        random.shuffle(mean_indices)
        mean_indices = mean_indices[:mean_count]

        print(f"Computing (y,x) position mean embeddings over {mean_count} specs...")
        means_np, pos_mean_gh, pos_mean_gw, _D = _compute_position_mean_embeddings_2d(
            model=model,
            dataset=dataset,
            indices=mean_indices,
            device=device,
            patch_h=patch_h,
            patch_w=patch_w,
        )
        pos_mean_tensor = torch.from_numpy(means_np).to(device)
        print("Done computing position means.")

    with torch.no_grad():
        for n, idx in enumerate(spec_indices):
            x, fname = dataset[idx]  # x: (1, H, W) (z-scored)
            spec_img = x[0].cpu().numpy()
            x = x.unsqueeze(0).to(device)  # (1, 1, H, W)

            H = int(x.shape[-2])
            W = int(x.shape[-1])
            grid_h = H // patch_h
            grid_w = W // patch_w

            # Pick random query token indices
            safe_fname = _sanitize(str(fname))
            spec_out = out_dir / safe_fname
            spec_out.mkdir(parents=True, exist_ok=True)

            if args.mode == "attn":
                attn, T = _final_layer_attention_inference(model, x)

                # attn: (B, heads, T, T) when average_attn_weights=False
                if attn.dim() != 4:
                    raise RuntimeError(f"Unexpected attention shape: {tuple(attn.shape)}")

                B, n_heads, Tq, Tk = attn.shape
                if B != 1 or Tq != Tk or Tq != T:
                    raise RuntimeError(
                        f"Unexpected attention dims: attn={tuple(attn.shape)} vs T={T}"
                    )
                if grid_h * grid_w != T:
                    raise RuntimeError(f"Patch grid mismatch: {grid_h}*{grid_w} != T={T}")

                q_count = min(args.queries_per_spec, T)
                q_indices = random.sample(range(T), k=q_count)
                heads_to_show = n_heads if args.max_heads is None else min(args.max_heads, n_heads)

                # Save per-head + mean-over-heads attention maps
                attn_cpu = attn[0].float().cpu().numpy()  # (heads, T, T)
                mean_attn = attn_cpu[:heads_to_show].mean(axis=0)  # (T, T)

                for q in q_indices:
                    qy, qx = divmod(q, grid_w)

                    # Mean over heads
                    mean_row = mean_attn[q].reshape(grid_h, grid_w)
                    out_path = spec_out / f"spec{n:04d}_q{q:04d}_qy{qy:03d}_qx{qx:03d}_mean.png"
                    _save_attn_overlay_on_spec_png(
                        spec_img,
                        mean_row,
                        patch_h=patch_h,
                        patch_w=patch_w,
                        query_y=qy,
                        query_x=qx,
                        out_path=str(out_path),
                        title=f"{safe_fname} | query={q} (y={qy}, x={qx}) | mean over {heads_to_show} heads",
                        alpha=float(args.alpha),
                        cmap="viridis",
                    )

                    # Individual heads
                    for head in range(heads_to_show):
                        row = attn_cpu[head, q].reshape(grid_h, grid_w)
                        out_path = spec_out / f"spec{n:04d}_q{q:04d}_qy{qy:03d}_qx{qx:03d}_head{head:02d}.png"
                        _save_attn_overlay_on_spec_png(
                            spec_img,
                            row,
                            patch_h=patch_h,
                            patch_w=patch_w,
                            query_y=qy,
                            query_x=qx,
                            out_path=str(out_path),
                            title=f"{safe_fname} | query={q} (y={qy}, x={qx}) | head={head}",
                            alpha=float(args.alpha),
                            cmap="viridis",
                        )

            else:  # args.mode == "cosine"
                y, T = _final_encoder_embeddings_inference(model, x)  # (1, T, D)
                if y.dim() != 3 or y.size(0) != 1:
                    raise RuntimeError(f"Unexpected encoder embeddings shape: {tuple(y.shape)}")
                if grid_h * grid_w != T:
                    raise RuntimeError(f"Patch grid mismatch: {grid_h}*{grid_w} != T={T}")

                q_count = min(args.queries_per_spec, T)
                q_indices = random.sample(range(T), k=q_count)

                y0 = y[0]  # (T, D)
                if pos_mean_tensor is not None:
                    if pos_mean_gh != grid_h or pos_mean_gw != grid_w:
                        raise RuntimeError(
                            f"Position-mean grid mismatch: means={pos_mean_gh}x{pos_mean_gw} vs spec={grid_h}x{grid_w}"
                        )
                    y0_grid = y0.reshape(grid_h, grid_w, -1)
                    y0 = (y0_grid - pos_mean_tensor).reshape(T, -1)

                for q in q_indices:
                    qy, qx = divmod(q, grid_w)
                    q_vec = y0[q].unsqueeze(0)  # (1, D)
                    sims = F.cosine_similarity(y0, q_vec.expand_as(y0), dim=1)  # (T,)
                    sim_grid = sims.float().cpu().numpy().reshape(grid_h, grid_w)

                    out_path = spec_out / f"spec{n:04d}_q{q:04d}_qy{qy:03d}_qx{qx:03d}_cosine.png"
                    _save_attn_overlay_on_spec_png(
                        spec_img,
                        sim_grid,
                        patch_h=patch_h,
                        patch_w=patch_w,
                        query_y=qy,
                        query_x=qx,
                        out_path=str(out_path),
                        title=f"{safe_fname} | query={q} (y={qy}, x={qx}) | cosine sim (final embeddings)",
                        alpha=float(args.alpha),
                        cmap="coolwarm",
                        vmin=-1.0,
                        vmax=1.0,
                    )


if __name__ == "__main__":
    main()

