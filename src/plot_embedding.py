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
    # Load and process JSON file if provided
    song_data_dict = {}
    if args.get("json_path"):
        print("Loading JSON file...")
        with open(args["json_path"], 'r') as f:
            json_data = json.load(f)
        
        # Filter for entries with song_present=true and build lookup dict
        for entry in json_data:
            if entry.get("song_present", False):
                # Use filename stem (without extension) for matching
                filename_stem = Path(entry["filename"]).stem
                song_data_dict[filename_stem] = {
                    "segments": entry["segments"],
                    "syllable_labels": entry.get("syllable_labels", {})
                }
        
        print(f"Found {len(song_data_dict)} files with songs")
    
    # Load model and config from checkpoint
    model, config = load_model_from_checkpoint(
        run_dir=args["run_dir"],
        checkpoint_file=args["checkpoint"]
    )
    model.eval()
    
    # Create embedding dataset using a large max_context, then batch-chunk per file
    embedding_dataset = SpectogramDataset(
        dir=args["spec_dir"],
        n_mels=config["mels"],
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
        
        # --- chunk into fixed context batches and process only chunks with song ---
        CHUNK = config["num_timebins"]                       # context length
        H = int(spec_mels / config["patch_height"])
        W_const = int(CHUNK / config["patch_width"])
        batch_chunks = []            # tensors [1, mels, CHUNK]
        batch_song_indices = []      # list[list[int]]
        batch_labels = []            # list[list[object]]

        full_T = spec.shape[-1]
        n_chunks = (full_T + CHUNK - 1) // CHUNK
        for ci in range(n_chunks):
            start_tb = ci * CHUNK
            end_tb   = min(start_tb + CHUNK, full_T)
            chunk = spec[:, :, start_tb:end_tb]              # [1, mels, Tci]
            Tci = chunk.shape[-1]
            if Tci < config["patch_width"]:
                continue
            # right-pad to CHUNK so we can batch
            if Tci < CHUNK:
                pad = CHUNK - Tci
                chunk = F.pad(chunk, (0, pad, 0, 0, 0, 0))
            # song mask + labels for this chunk (before encoder)
            W_chunk = W_const                                  # after pad
            if args.get("json_path") and song_data_dict:
                mask, labs = get_syllable_labels_for_patches(
                    file_name, song_data_dict, W_chunk, config["patch_width"], CHUNK, start_timebin=start_tb
                )
                labs = fill_between_equals(labs, song_mask=mask, max_gap=args.get("max_gap"))
            else:
                mask = [True] * W_chunk
                labs = [None] * W_chunk
            idxs = [j for j, m in enumerate(mask) if m]
            if not idxs:
                continue  # skip chunks with no song
            batch_chunks.append(chunk.unsqueeze(0))           # -> [1,1,mels,CHUNK]
            batch_song_indices.append(idxs)
            batch_labels.append(labs)
            total_timebins += (end_tb - start_tb)
            if total_timebins >= args["num_timebins"]:
                break

        if not batch_chunks:
            print(f"Skip {Path(file_name).stem}: no song patches")
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
    
    if args.get("json_path") and len(song_data_dict) > 0:
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
                       alpha=0.1, s=10, color='lightgray', edgecolors='none')
        
        # Plot each syllable type with different colors
        for syllable_type in unique_syllables:
            syllable_indices = syllable_labels == syllable_type
            if syllable_indices.any():
                plt.scatter(emb_enc[syllable_indices, 0], emb_enc[syllable_indices, 1], 
                           alpha=0.1, s=10, color=color_map[syllable_type], edgecolors='none')
    else:
        # Fallback to original single-color plot if no JSON provided
        plt.scatter(emb_enc[:, 0], emb_enc[:, 1], alpha=0.1, s=10, edgecolors='none')
    
    plt.title(title, fontsize=48, fontweight='bold', loc='left')
    plt.xlabel('UMAP 1', fontsize=24, fontweight='bold')
    plt.ylabel('UMAP 2', fontsize=24, fontweight='bold')
    plt.xticks([])  # Remove tick marks but keep axis
    plt.yticks([])  # Remove tick marks but keep axis
    
    # Save figure if specified
    if args.get("save_path"):
        plt.savefig(args["save_path"], bbox_inches='tight', dpi=300)
        print(f"Figure saved to {args['save_path']}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple UMAP embedding visualization")
    parser.add_argument("--num_timebins", type=int, required=True, help="Number of time bins")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file (optional)")
    parser.add_argument("--spec_dir", type=str, required=True, help="Directory of specs to plot the embedding of")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the figure (optional)")
    parser.add_argument("--npz_out", type=str, default=None, help="Save arrays to this .npz path")
    parser.add_argument("--json_path", type=str, default=None, help="to provide song snippets + syllable labels")
    parser.add_argument("--max_gap", type=int, default=100, help="max patch-length of unlabeled gap to infill")
    parser.add_argument("--max_context", type=int, default=None, help="max timebins to load per clip before chunking; defaults to 64×context")
    # removed auxiliary visualization args

    args = parser.parse_args()
    main(vars(args))