#!/usr/bin/env python3
import argparse
from pathlib import Path
import datetime
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector, Button
from matplotlib.path import Path as MplPath

from model import TinyBird
from data_loader import SpectogramDataset
from utils import load_model_from_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMG_DIR = PROJECT_ROOT / "img"

def build_embeddings(model, dataset, config, max_timebins, subtract_pos):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    latent_list = []
    index_map = []         # dicts: {"spec_id", "t_idx", "t_start", "t_end"}
    specs_cache = []       # 2D tensors [mels, time]

    total_timebins = 0
    i = 0
    patch_h = int(config["patch_height"])
    patch_w = int(config["patch_width"])
    mels     = int(config["mels"])
    max_tb   = int(config["num_timebins"])

    with torch.inference_mode():
        while i < len(dataset) and total_timebins < max_timebins:
            spec, _ = dataset[i]                      # expect [mels, time] or [1,mels,time]
            if spec.ndim == 3 and spec.shape[0] == 1:
                spec = spec[0]
            if spec.ndim != 2:
                raise ValueError(f"Expected 2D spectrogram, got shape {tuple(spec.shape)}")

            # enforce training crop
            spec = spec[:mels, :max_tb]
            spec_timebins = spec.shape[-1]

            # make time divisible by patch_w
            remainder = spec_timebins % patch_w
            if remainder != 0:
                spec = spec[:, :spec_timebins - remainder]
            spec_timebins = spec.shape[-1]

            # cache for later visualization
            specs_cache.append(spec.clone())
            spec_id = len(specs_cache) - 1

            # encode
            spec_b = spec.unsqueeze(0).unsqueeze(0).to(device)   # [1,1,mels,time]
            z = model.forward_encoder_inference(spec_b)                  # [1, S, D]
            z = z.squeeze(0).cpu()                                       # [S, D]
            S, D = z.shape

            H = int(spec.shape[0] // patch_h)                            # patches along mel
            W = int(spec.shape[1] // patch_w)                            # patches along time
            if H * W != S:
                # fallback: skip this file to keep indexing exact
                print(f"Skip spec {spec_id}: tokens S={S} != H*W={H*W}")
                i += 1
                continue

            # [H,W,D] -> [W,H,D] -> [W,H*D]
            z_grid = z.view(H, W, D).permute(1, 0, 2).contiguous()       # [W,H,D]
            z_freq_stack = z_grid.view(W, H * D)                          # [W,H*D]

            if subtract_pos:
                z_freq_stack = z_freq_stack - z_freq_stack.mean(dim=0, keepdim=True)

            latent_list.append(z_freq_stack)

            # index rows 1:1 with z_freq_stack rows
            for w in range(W):
                t_start = w * patch_w
                t_end   = t_start + patch_w
                index_map.append({"spec_id": spec_id, "t_idx": w, "t_start": t_start, "t_end": t_end})

            total_timebins += spec_timebins
            i += 1

    if len(latent_list) == 0:
        raise RuntimeError("No embeddings produced. Check shapes or config.")
    Z = torch.cat(latent_list, dim=0).contiguous()    # [N_points, H*D]
    return Z, np.array(index_map, dtype=object), specs_cache

def umap_embed(Z, seed):
    reducer = umap.UMAP(
        n_components=2, metric="cosine",
        n_jobs=-1, low_memory=True, random_state=None
    )
    emb2d = reducer.fit_transform(Z.numpy())
    return emb2d

def make_snippet_fig(snippets, titles, cols, dpi, add_colorbar=False):
    count = len(snippets)
    cols = max(1, cols)
    rows = int(np.ceil(count / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.2*cols, 2.8*rows),
                            squeeze=False, dpi=dpi)

    mappable = None
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axs[r, c]
        if idx < count:
            img = snippets[idx]
            im = ax.imshow(img, origin="lower", aspect="auto")
            if mappable is None:
                mappable = im
            ax.set_title(titles[idx], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.axis("off")

    if add_colorbar and mappable is not None:
        fig.colorbar(mappable, ax=axs.ravel().tolist(), fraction=0.025, pad=0.02)

    fig.tight_layout()
    return fig

def collect_snippets(selected_idxs, index_map, specs_cache, args):
    """Return list of 2D arrays to plot and titles."""
    snippets, titles = [], []
    take = selected_idxs[:args.max_show]
    k = max(0, int(args.context_patches))
    for j in take:
        m = index_map[j]
        spec2d = specs_cache[m["spec_id"]]                 # [mels, time], torch.Tensor
        if args.display == "full":
            patch = spec2d.cpu().numpy()
            title = f"spec{m['spec_id']} full"
        else:
            # patch or context
            pw = int(args.patch_width) if args.patch_width > 0 else (m["t_end"] - m["t_start"])
            t_start = m["t_start"]
            t_end   = m["t_end"]
            if args.display == "context" and k > 0:
                t_start = max(0, t_start - k * pw)
                t_end   = min(spec2d.shape[1], t_end + k * pw)
            patch = spec2d[:, t_start:t_end].cpu().numpy()
            title = f"spec{m['spec_id']} t[{t_start}:{t_end}]"

        if args.normalize:
            lo, hi = np.percentile(patch, 2), np.percentile(patch, 98)
            if hi > lo:
                patch = (patch - lo) / (hi - lo)
        snippets.append(patch)
        titles.append(title)
    return snippets, titles

def _ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def interactive_lasso(emb2d, index_map, specs_cache, args):
    outdir = _ensure_outdir(Path(args.out_dir) if args.out_dir else DEFAULT_IMG_DIR)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=args.dpi)
    pts = ax.scatter(emb2d[:, 0], emb2d[:, 1], s=3, alpha=0.5)
    ax.set_title("UMAP: lasso-select points")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.3)

    # Save a static overview once
    overview_path = outdir / f"umap_overview_{_timestamp()}.png"
    fig.savefig(overview_path, bbox_inches="tight", dpi=args.dpi)
    print(f"Saved: {overview_path}")

    # buttons
    btn_ax_clear = plt.axes([0.13, 0.01, 0.1, 0.05])
    btn_clear = Button(btn_ax_clear, "Clear")
    def on_clear(_):
        ax.set_title("UMAP: lasso-select points")
        fig.canvas.draw_idle()
    btn_clear.on_clicked(on_clear)

    def onselect(verts):
        path = MplPath(verts)
        inside = path.contains_points(emb2d)
        idxs = np.flatnonzero(inside)
        if len(idxs) == 0:
            ax.set_title("UMAP: 0 selected")
            fig.canvas.draw_idle()
            return

        snippets, titles = collect_snippets(
            selected_idxs=idxs,
            index_map=index_map,
            specs_cache=specs_cache,
            args=args
        )
        ax.set_title(f"UMAP: selected {len(idxs)} | showing {len(snippets)}")
        fig.canvas.draw_idle()

        fig_snip = make_snippet_fig(
            snippets, titles,
            cols=args.grid_cols, dpi=args.dpi,
            add_colorbar=getattr(args, "colorbar", False)
        )
        sel_name = f"umap_selection_{len(idxs)}pts_{_timestamp()}.png"
        sel_path = outdir / sel_name
        fig_snip.savefig(sel_path, bbox_inches="tight", dpi=args.dpi)
        print(f"Saved: {sel_path}")
        plt.close(fig_snip)

        # Optional: save raw selection indices and coordinates for reproducibility
        if args.save_npz:
            npz_path = outdir / f"umap_selection_{len(idxs)}pts_{_timestamp()}.npz"
            np.savez_compressed(
                npz_path,
                selected_indices=idxs,
                embedding=emb2d,
                grid_titles=np.array(titles, dtype=object)
            )
            print(f"Saved: {npz_path}")

    try:
        lasso = LassoSelector(ax, onselect=onselect, props={"linewidth": 1.0})
    except TypeError:
        lasso = LassoSelector(ax, onselect=onselect, lineprops={"linewidth": 1.0})
    plt.show()

def main():
    ap = argparse.ArgumentParser(description="Lasso-select in UMAP and save spectrogram grids")
    ap.add_argument("--num_timebins", type=int, required=True, help="Total timebins budget to scan from dataset")
    ap.add_argument("--run_dir", type=str, required=True, help="Run directory to load model from")
    ap.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint filename")
    ap.add_argument("--spec_dir", type=str, required=True, help="Directory of spectrogram .pt files")
    ap.add_argument("--subtract_pos", action="store_true", help="Subtract mean across positions")
    ap.add_argument("--max_show", type=int, default=36, help="Max spectrograms to show per selection")
    ap.add_argument("--grid_cols", type=int, default=6, help="Grid columns for display")
    ap.add_argument("--normalize", action="store_true", help="Per-image robust scaling for display")
    ap.add_argument("--dpi", type=int, default=110, help="Matplotlib DPI")
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_IMG_DIR),
                    help="Directory to save PNG/NPZ (default: project_root/img)")
    ap.add_argument("--save_npz", action="store_true",
                    help="Also save selected indices and coords as NPZ")
    ap.add_argument("--umap_seed", type=int, default=0, help="Random seed for UMAP")
    ap.add_argument("--display", choices=["patch","context","full"], default="patch",
                    help="Show one patch, a context window, or full spectrogram")
    ap.add_argument("--context_patches", type=int, default=2, help="Extra patches on each side for 'context' view")
    ap.add_argument("--patch_width", type=int, default=0, help="Override patch width in timebins (0=auto from config)")
    ap.add_argument("--colorbar", action="store_true", help="Show a colorbar on snippet grids")
    args = ap.parse_args()

    model, config = load_model_from_checkpoint(run_dir=args.run_dir, checkpoint_file=args.checkpoint)

    ds = SpectogramDataset(
        dir=args.spec_dir,
        n_mels=int(config["mels"]),
        n_timebins=int(config["num_timebins"]),
        pad_crop=False
    )

    Z, index_map, specs_cache = build_embeddings(
        model=model,
        dataset=ds,
        config=config,
        max_timebins=args.num_timebins,
        subtract_pos=args.subtract_pos
    )

    emb2d = umap_embed(Z, seed=args.umap_seed)

    # store patch_width for later use
    if args.patch_width == 0:
        args.patch_width = int(config["patch_width"])

    interactive_lasso(emb2d, index_map, specs_cache, args)

if __name__ == "__main__":
    main()
