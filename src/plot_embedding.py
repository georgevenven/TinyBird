import argparse
from model import TinyBird
from data_loader import SpectogramDataset
from utils import load_model_from_checkpoint
import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import torch.nn.functional as F
from math import floor, ceil
from matplotlib import colors as mcolors
import re

# should be parameter, or somehow automatically calculated 
def _ms_to_tb(ms):
    """2 ms per timebin."""
    return int(round(ms / 2.0))

def _slug(s):
    """Safe filename slug. GEORGE SAYS THIS IS TEMPORARY."""
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in str(s))

def load_json_events(json_path, selected_bird=None):
    """
    Json format 
    {
      "metadata": {"units":"ms"},
      "recordings":[
        {"recording":{"filename":...,"bird_id":...},
         "detected_events":[
            {"onset_ms":...,"offset_ms":...,
             "units":[{"onset_ms":...,"offset_ms":...,"id":int}, ...]}]}]}
    Returns: list of entries:
      {"stem": <lower filename stem>, "bird_id": <lower bird_id>, "events": [event,...]}
      event = {"on_tb": int, "off_tb": int, "units": [(on_tb, off_tb, id), ...]}

    filename will always contain stem 
    
    Args:
        json_path: Path to the JSON file
        selected_bird: is the bird_id, must be a direct match 
    """
    with open(json_path, "r") as f:
        jd = json.load(f)
    # Build exact map keyed by (bird_num, clip_num) -> [events]
    event_map = {}
    for rec in jd.get("recordings", []):
        file_name = rec.get("recording", {}).get("filename", "") or ""
        bird_id = rec.get("recording", {}).get("bird_id", "") or ""

        file_name = file_name.split(".")[:-1]
        file_name = ".".join(file_name)
        # print(file_name)
        # return 

        # # Filter by selected bird if specified
        if selected_bird is not None and bird_id != selected_bird:
            continue

        # if we have selected a certain bird, that the event_map will only contain that said bird 
        ev_list = []
        for ev in rec.get("detected_events", []):
            e_on = _ms_to_tb(ev["onset_ms"])
            e_off = _ms_to_tb(ev["offset_ms"])
            units = []
            for u in ev.get("units", []):
                u_on = _ms_to_tb(u["onset_ms"])
                u_off = _ms_to_tb(u["offset_ms"])
                units.append((u_on, u_off, int(u["id"])))
            ev_list.append({"on_tb": e_on, "off_tb": e_off, "units": units})
        event_map.setdefault((file_name), []).extend(ev_list)
    
    return event_map

def fill_between_equals(labels, song_mask=None, max_gap=80):
    """Fill None runs bounded by the SAME label on both sides (<= max_gap)."""
    lab = list(labels)
    W = len(lab)
    i = 0
    while i < W:
        if lab[i] is not None:
            i += 1
            continue
        start = i
        while i < W and lab[i] is None:
            i += 1
        end = i  # [start:end) is a None-run
        left  = lab[start-1] if start-1 >= 0 else None
        right = lab[end]     if end < W     else None
        if left is not None and right is not None and left == right and (end-start) <= max_gap:
            if song_mask is None or all(song_mask[j] for j in range(start, end)):
                for j in range(start, end):
                    lab[j] = left
    return lab

def _to_tb(v, ref_ms=None):
    """
    Convert a time value to timebins (2 ms per bin).
    Treat very small numeric values as **seconds** (common in your JSON),
    otherwise as milliseconds. If ref_ms is given, prefer 'seconds' if v << ref_ms.
    """
    if v is None:
        return None
    # if clearly small or much smaller than a known ms reference, assume seconds
    is_seconds = (v < 20.0) or (ref_ms is not None and v < 0.5 * ref_ms)
    ms = v * 1000.0 if is_seconds else v
    tb = int(round(ms / 2.0))
    return tb

def get_syllable_labels_for_patches(filename, song_data_dict, W, patch_width, spec_timebins, start_timebin=0):
    """
    Determine syllable labels for patches and which patches contain song segments.
    
    Args:
        filename: Name of the spectrogram file
        song_data_dict: Dictionary mapping filename stems to song data (segments + syllable_labels)
        W: Number of patches in the spectrogram
        patch_width: Width of each patch in timebins
        spec_timebins: Total timebins in the spectrogram (max 1024)
    
    Returns:
        Tuple of (song_mask, syllable_labels):
        - song_mask: List of boolean values indicating which patches contain song segments
        - syllable_labels: List of syllable labels for each patch (None for non-song patches)
    """
    filename_stem = Path(filename).stem
    patch_song_mask = [False] * W  # default: no song
    patch_syllable_labels = [None] * W  # default: no label
    
    if filename_stem in song_data_dict:
        entry = song_data_dict[filename_stem]
        segments = entry["segments"]
        syllable_labels = entry.get("syllable_labels", {})
        
        # First, mark all song segments
        for segment in segments:
            # Prefer explicit timebins. Else convert ms→tb robustly.
            if "onset_timebin" in segment and "offset_timebin" in segment:
                seg_on_tb = int(segment["onset_timebin"])
                seg_off_tb = int(segment["offset_timebin"])
            else:
                seg_on_tb = _to_tb(segment.get("onset_ms", 0))
                seg_off_tb = _to_tb(segment.get("offset_ms", max(1, segment.get("onset_ms", 0)+1)))

            # Intersect with current chunk [start_timebin, start_timebin+spec_timebins)
            chunk_start = start_timebin
            chunk_end   = start_timebin + spec_timebins
            if seg_off_tb <= chunk_start or seg_on_tb >= chunk_end:
                continue
            # chunk-relative tb
            segment_start = max(0, seg_on_tb - chunk_start)
            segment_end   = min(spec_timebins, seg_off_tb - chunk_start)
            
            # Convert timebin indices to patch indices within the cropped spectrogram
            onset_patch = max(0, segment_start // patch_width)
            offset_patch = min(W, segment_end // patch_width + 1)
            
            # Mark patches that overlap with song segment
            for p in range(onset_patch, offset_patch):
                if p < W:
                    patch_song_mask[p] = True
        
        # Now assign syllable labels. Handle seconds vs ms and segment-relative cases.
        for syllable_type, syllable_instances in syllable_labels.items():
            for on_v, off_v in syllable_instances:
                # Build candidate intervals in tb:
                # 1) absolute (treat values as ms or seconds)
                abs_on = _to_tb(on_v)
                abs_off = max(abs_on + 1, _to_tb(off_v))
                candidates = [(abs_on, abs_off)]
                # 2) segment-relative: add each segment onset_ms if present
                for seg in segments:
                    seg_on_ms = seg.get("onset_ms", None)
                    if seg_on_ms is not None:
                        rel_on = _to_tb(on_v + seg_on_ms, ref_ms=seg_on_ms)
                        rel_off = max(rel_on + 1, _to_tb(off_v + seg_on_ms, ref_ms=seg_on_ms))
                        candidates.append((rel_on, rel_off))

                # choose candidate with max overlap with current chunk window
                best = None
                best_ov = 0
                for a, b in candidates:
                    s = max(0, a - start_timebin); e = min(spec_timebins, b - start_timebin)
                    ov = max(0, e - s)
                    if ov > best_ov:
                        best = (s, e); best_ov = ov
                if best is None or best_ov == 0:
                    continue

                s_tb, e_tb = best
                # Convert to patch indices and assign within song mask only
                on_p = max(0, s_tb // patch_width)
                off_p = min(W, e_tb // patch_width + 1)
                for p in range(on_p, off_p):
                    if p < W and patch_song_mask[p]:
                        patch_syllable_labels[p] = syllable_type
    
    return patch_song_mask, patch_syllable_labels

def main(args):
    # Load Bengal-finch JSON with detected events
    event_map = {}
    if args.get("json_path"):
        event_map = load_json_events(args["json_path"], selected_bird=args.get("bird"))
    
    # Load model and config from checkpoint
    model, config = load_model_from_checkpoint(
        run_dir=args["run_dir"],
        checkpoint_file=args["checkpoint"]
    )
    model.eval()
    
    # Create embedding dataset using a large max_context, then batch-chunk per file
    embedding_dataset = SpectogramDataset(
        dir=args["spec_dir"],
        n_timebins=args.get("max_context", config["num_timebins"] * 64),
        pad_crop=False
    )
    
    latent_list = []
    spec_batches_all = []
    pos_ids = []          # absolute time index per patch within its clip (0..W-1)
    rel_bins = []         # optional: relative-position bins (0..K-1)
    syllable_labels = []  # syllable labels for each patch
    K = 64                # change if you want coarser/finer relative bins
    total_timebins = 0
    i = 0
    # store full CHUNK windows and their patch labels for later visualization
    all_chunks = []        # list of tensors [1, mels, CHUNK]
    all_chunk_labels = []  # list of lists length W_const with unit ids or None
    
    while i < len(embedding_dataset) and total_timebins < args["num_timebins"]:
        spec, file_name = embedding_dataset[i]
        
        patch_height, patch_width, mels, num_timebins = config["patch_height"], config["patch_width"], config["mels"], config["num_timebins"]
        # keep full spectrogram up to max_context
        spec = spec[:,:mels,:]
        
        spec_timebins = spec.shape[-1]
        spec_mels = spec.shape[-2]
        
        remainder = spec_timebins % patch_width
        if remainder != 0: 
            spec = spec[:,:,:spec_timebins-remainder]
        spec_timebins = spec.shape[-1]
        spec_mels = spec.shape[-2]
        
        # --- Only pass detected events; crop non-events before the model ---
        CHUNK = config["num_timebins"]                         # context length
        H = int(spec_mels / config["patch_height"])
        W_const = int(CHUNK / config["patch_width"])
        batch_chunks = []              # tensors [1, mels, CHUNK]
        batch_song_indices = []        # list[list[int]]
        batch_labels = []              # list[list[object]]; here: unit ids

        matched_events = event_map.get(file_name, [])

        # ugly 
        if matched_events == []:
            i+=1
            continue

        if matched_events:
            for ev in matched_events:
                e_on, e_off = ev["on_tb"], ev["off_tb"]
                if e_off <= e_on:
                    continue
                # split long events into CHUNK-sized windows
                win_start = e_on
                while win_start < e_off:
                    win_end = min(win_start + CHUNK, e_off)
                    chunk = spec[:, :, win_start:win_end]      # [1, mels, Tci]
                    Tci = chunk.shape[-1]
                    if Tci < config["patch_width"]:
                        break
                    if Tci < CHUNK:
                        pad = CHUNK - Tci
                        chunk = F.pad(chunk, (0, pad, 0, 0, 0, 0))
                    # label patches by unit id
                    W_chunk = W_const
                    labs = [None] * W_chunk
                    # compute best-overlap unit per patch
                    for (u_on, u_off, uid) in ev["units"]:
                        # overlap with this window
                        a = max(u_on, win_start)
                        b = min(u_off, win_end)
                        if b <= a:
                            continue
                        # window-relative timebins
                        a_rel = a - win_start
                        b_rel = b - win_start
                        p0 = max(0, a_rel // config["patch_width"])
                        p1 = min(W_chunk, ceil(b_rel / config["patch_width"]))
                        for p in range(p0, p1):
                            labs[p] = uid
                    # mask is simply patches inside the event window (all True)
                    idxs = [j for j in range(W_chunk)]
                    batch_chunks.append(chunk.unsqueeze(0))
                    batch_song_indices.append(idxs)
                    batch_labels.append(labs)
                    # keep a copy for spectrogram visualization later
                    all_chunks.append(chunk.detach().cpu())     # [1, mels, CHUNK]
                    all_chunk_labels.append(labs[:])            # copy
                    total_timebins += (win_end - win_start)
                    if total_timebins >= args["num_timebins"]:
                        break
                    win_start += CHUNK
        else:
            # No exact match -> skip this spec to avoid wrong labels
            i += 1
            continue

        if not batch_chunks:
            i += 1
            continue

        batch = torch.cat(batch_chunks, dim=0)               # [B,1,mels,CHUNK]
        spec_batches_all.append(batch.cpu())
        with torch.no_grad():
            z = model.forward_encoder_inference(batch)       # [B,S,D]
        B, S, D = z.shape
        z_grid = z.reshape(B, H, W_const, D).permute(0,2,1,3)      # [B,W,H,D]
        z_freq = z_grid.reshape(B, W_const, H * D)                 # [B,W,HD]
    
        # defer any normalization until after concatenation
        for b in range(B):
            idxs = batch_song_indices[b]
            latent_list.append(z_freq[b, idxs, :].detach().cpu())
            pos_ids.extend(idxs)
            if len(idxs) > 1:
                rel = np.round(np.linspace(0, K-1, num=len(idxs))).astype(int)
            else:
                rel = np.array([0], dtype=int)
            rel_bins.extend(rel.tolist())
            song_patch_labels = [batch_labels[b][j] for j in idxs]
            syllable_labels.extend(song_patch_labels)
        i += 1
    
    Z = torch.cat(latent_list, dim=0)  # (N_rows, H*D)
    pos_ids = np.asarray(pos_ids, dtype=int)
    rel_bins = np.asarray(rel_bins, dtype=int)
    syllable_labels = np.asarray(syllable_labels, dtype=object)
    print(f"Shape of embeddings: {Z.shape}")
    
    # Count syllable types
    unique_syllables, counts = np.unique(syllable_labels[syllable_labels != None], return_counts=True)
    unlabeled_count = np.sum(syllable_labels == None)
    print(f"Syllable type counts: {dict(zip(unique_syllables, counts))}")
    print(f"Unlabeled song patches: {unlabeled_count}")
    
    # --- remove average vector per position (choose one) ---
    mode = None  # {"absolute", "relative", None}
    Z_np = Z.cpu().numpy()
    if mode == "absolute":
        uniq = np.unique(pos_ids)
        means = np.zeros((uniq.max()+1, Z_np.shape[1]), dtype=np.float32)
        for p in uniq:
            means[p] = Z_np[pos_ids == p].mean(axis=0)
        Z_np = Z_np - means[pos_ids]
    elif mode == "relative":
        means = np.zeros((K, Z_np.shape[1]), dtype=np.float32)
        for b in np.unique(rel_bins):
            means[b] = Z_np[rel_bins == b].mean(axis=0)
        Z_np = Z_np - means[rel_bins]
    Z = torch.from_numpy(Z_np)
    
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=512)
    # Z_pca = pca.fit_transform(Z_np)
    # print(f"Shape after PCA: {Z_pca.shape}")
    
    # Apply UMAP to the PCA-reduced data
    reducer_enc = umap.UMAP(n_components=2, n_neighbors=100, metric='cosine')
    emb_enc = reducer_enc.fit_transform(Z)
    print("UMAP done")

    Z_pca = Z ## place holder 

    # --- save arrays to NPZ (labels, spec, embeddings, pos_ids) ---
    if args.get("npz_out"):
        spec_arr = (
            torch.cat(spec_batches_all, dim=0).numpy()
            if spec_batches_all else
            np.empty((0, 1, config["mels"], config["num_timebins"]), dtype=np.float32)
        )
        np.savez(
            args["npz_out"],
            labels=syllable_labels,     # shape: (N_patches,)
            spec=spec_arr,              # shape: (total_chunks, 1, mels, CHUNK)
            embeddings=Z_np,            # shape: (N_patches, H*D), freq-stacked
            embeddings_pca=Z_pca,       # shape: (N_patches, 32), PCA-reduced
            embeddings_umap=emb_enc,    # shape: (N_patches, 2), UMAP 2D embedding
            pos_ids=pos_ids             # shape: (N_patches,)
        )
        print(f"NPZ saved to {args['npz_out']}")
    
    # Create scatter plot colored by syllable types
    plt.figure(figsize=(8, 8), dpi=300)
    
    # Determine title based on depositioning mode
    if mode is None:
        title = "A"
    else:
        title = "B "
    
    if args.get("json_path") and len(event_map) > 0:
        # Define colors for different syllable types
        import matplotlib.cm as cm
        mask_labeled = syllable_labels != None
        labeled_vals = syllable_labels[mask_labeled]
        unique_syllables = np.unique(labeled_vals) if labeled_vals.size > 0 else np.array([], dtype=object)
        colors = cm.tab20(np.linspace(0, 1, len(unique_syllables))) if len(unique_syllables) > 0 else []
        color_map = dict(zip(unique_syllables, colors))

        # Plot unlabeled song patches first (in background)
        unlabeled_indices = ~mask_labeled
        if unlabeled_indices.any():
            plt.scatter(emb_enc[unlabeled_indices, 0], emb_enc[unlabeled_indices, 1], 
                       alpha=0.1, s=10, color='#404040', edgecolors='none')  # light black instead of lightgray
        
        # Plot each syllable type with different colors
        for syllable_type in unique_syllables:
            syllable_indices = syllable_labels == syllable_type
            if syllable_indices.any():
                plt.scatter(emb_enc[syllable_indices, 0], emb_enc[syllable_indices, 1], 
                           alpha=0.1, s=10, color=color_map[syllable_type], edgecolors='none')
    else:
        # Fallback to original single-color plot if no JSON provided
        plt.scatter(emb_enc[:, 0], emb_enc[:, 1], alpha=0.1, s=10, edgecolors='none')
        color_map = {}
    
    plt.title(title, fontsize=48, fontweight='bold', loc='left')
    plt.xlabel('UMAP 1', fontsize=24, fontweight='bold')
    plt.ylabel('UMAP 2', fontsize=24, fontweight='bold')
    plt.xticks([])  # Remove tick marks but keep axis
    plt.yticks([])  # Remove tick marks but keep axis
    
    # Always save under root/imgs relative to this file. No CLI arg.
    # GEORGE SAYS THIS IS TEMPORARY: name = spec_dir + run_dir + timebins
    out_dir = (Path(__file__).resolve().parent.parent / "imgs")
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_slug = _slug(Path(args["spec_dir"]).name or "spec")
    run_slug  = _slug(Path(args["run_dir"]).name or "run")
    img_path  = out_dir / f"{spec_slug}__{run_slug}__tb{args['num_timebins']}.png"
    plt.savefig(img_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Figure saved to {img_path}")

    # --- Generate 5 random CHUNK-length spectrograms with aligned 1D color bars ---
    # GEORGE SAYS THIS IS TEMPORARY: uses same color palette as UMAP points.
    if all_chunks:
        rnd_count = min(5, len(all_chunks))
        sel = np.random.choice(len(all_chunks), size=rnd_count, replace=False)
        unlabeled_rgb = np.array(mcolors.to_rgb("#404040"))  # light black for non-song portions
        for k, idx in enumerate(sel, start=1):
            chunk = all_chunks[idx]        # [1, mels, CHUNK]
            labs  = all_chunk_labels[idx]  # length W_const
            mels, T = chunk.shape[-2], chunk.shape[-1]
            # derive patch geometry
            patch_width = config["patch_width"]
            W_const = int(T // patch_width) if T % patch_width == 0 else int(np.ceil(T / patch_width))
            # build bar image [h, T, 3], repeat per-timebin color from per-patch labels
            bar_h = 10
            bar = np.ones((bar_h, T, 3), dtype=np.float32) * unlabeled_rgb.reshape(1, 1, 3)
            for p in range(min(W_const, len(labs))):
                uid = labs[p]
                # rgba->rgb if present; else unlabeled
                rgb = np.array(color_map.get(uid, (*unlabeled_rgb, 1.0)))[:3]
                a = p * patch_width
                b = min(T, (p + 1) * patch_width)
                bar[:, a:b, :] = rgb
            # plot spectrogram + bar
            fig = plt.figure(figsize=(10, 6), dpi=300)
            gs = fig.add_gridspec(2, 1, height_ratios=[4, 0.3], hspace=0.05)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            ax1.imshow(chunk.squeeze(0).numpy(), aspect="auto", origin="lower", interpolation="none")
            ax1.set_xticks([]); ax1.set_yticks([])
            ax2.imshow(bar, aspect="auto", origin="lower", interpolation="nearest")
            ax2.set_xticks([]); ax2.set_yticks([])
            # save
            sp_path = out_dir / f"{spec_slug}__{run_slug}__tb{args['num_timebins']}__spec{k}.png"
            fig.savefig(sp_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
            print(f"Spectrogram saved to {sp_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple UMAP embedding visualization")
    parser.add_argument("--num_timebins", type=int, default=12400, help="Number of time bins")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file (optional)")
    parser.add_argument("--spec_dir", type=str, required=True, help="Directory of specs to plot the embedding of")
    parser.add_argument("--npz_out", type=str, default=None, help="Save arrays to this .npz path")
    parser.add_argument("--json_path", type=str, default=None, help="to provide song snippets + syllable labels")
    parser.add_argument("--bird", type=str, default=None, help="select specific bird number (e.g., 1 for bird1, 2 for bird2)")
    parser.add_argument("--max_gap", type=int, default=100, help="max patch-length of unlabeled gap to infill")
    parser.add_argument("--max_context", type=int, default=None, help="max timebins to load per clip before chunking; defaults to 64×context")

    args = parser.parse_args()
    main(vars(args))