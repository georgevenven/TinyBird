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

###
# To Do #
# Remove padding before UMAPING and account for that in the total number of timebins 

###

def _slug(s):
    """Safe filename slug. GEORGE SAYS THIS IS TEMPORARY."""
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in str(s))

def load_json_events(json_path, audio_params, selected_bird=None):
    """
    Json format 
    {
      "metadata": {"units":"ms"},
      "recordings":[
        {"recording":{"filename": str ,"bird_id": str "detected_vocalizations": int},
         "detected_events":[
            {"onset_ms":float,"offset_ms":float,
             "units":[{"onset_ms":...,"offset_ms":...,"id":int}, ...]}]}]}

    Returns: dict mapping filename -> list of events:
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
    
        file_name = rec.get("recording", {}).get("filename", "") 
        bird_id = rec.get("recording", {}).get("bird_id", "") 

        file_name = file_name.split(".")[:-1]
        file_name = ".".join(file_name)

        # Filter by selected bird if specified
        if selected_bird is not None and bird_id != selected_bird:
            continue

        event_list = []
        for event in rec.get("detected_events", []):
            event_on_ms = event["onset_ms"]
            event_off_ms = event["offset_ms"]
            event_on_timebins = ms_to_timebins(event_on_ms, audio_params)
            event_off_timebins = ms_to_timebins(event_off_ms, audio_params)
            units = []
            for unit in event.get("units", []):
                units_on_ms = unit["onset_ms"]
                units_off_ms = unit["offset_ms"]
                units_on_timebins = ms_to_timebins(units_on_ms, audio_params)
                units_off_timebins = ms_to_timebins(units_off_ms, audio_params)
                units.append((units_on_timebins, units_off_timebins, int(unit["id"])))
            event_list.append({"on_timebins": event_on_timebins, "off_timebins": event_off_timebins, "units": units})
        event_map.setdefault((file_name), []).extend(event_list)
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

def ms_to_timebins(ms_value, audio_params):
    """
    purpose: converts ms value to timebin value 

    the formula of converting ms to timebins:
    time_bin = (time_ms / 1000) × sample_rate / hop_length

    audio_params (tuple) = sr, n_mels, hop_size, fft
    """

    sr = audio_params[0]
    hop_size = audio_params[2]

    # int rounding is floor rounding ... could be a point of error 
    return int((ms_value / 1000) * sr / hop_size)

def create_label_arr(matched_event, rounded_spec_length):
    # create an array of labels 
    labels = torch.full((rounded_spec_length,), fill_value=-1) # -1 represents silence / non song element 

    units = matched_event["units"]

    for start, end, unit_id in units:
        labels[start:end+1] = unit_id # the +1 is because its inclusive 

    return labels

def main(args):

    # Load model and config from checkpoint
    model, config = load_model_from_checkpoint(
        run_dir=args["run_dir"],
        checkpoint_file=args["checkpoint"]
    )

    patch_height, patch_width, mels, model_num_timebins = config["patch_height"], config["patch_width"], config["mels"], config["num_timebins"]
    # number of time patches

    # not sure why its getting casted into a float 
    num_patches_time= int(model_num_timebins / patch_width)
    num_patches_height = int(config["max_seq"] / num_patches_time) 
    
    # this has to be done first to get the audio params etc 
    # Create embedding dataset using a large max_context, then batch-chunk per file
    embedding_dataset = SpectogramDataset(
        dir=args["spec_dir"],
        n_timebins=None
    )

    audio_params = embedding_dataset.sr, embedding_dataset.n_mels, embedding_dataset.hop_size, embedding_dataset.fft

    # Load JSON with labels and detected events 
    event_map = {}
    if args.get("json_path"):
        event_map = load_json_events(args["json_path"], selected_bird=args.get("bird"), audio_params=audio_params)
    
    model.eval()

    pos_ids = []          # absolute time index per patch within its clip (0..W-1)
    total_timebins = 0
    i = 0
    
    latent_list = []
    label_list = []
    while i < len(embedding_dataset) and total_timebins < args["num_timebins"]:
        spec, file_name = embedding_dataset[i]
        
        # spec shape 1, mels, time
        spec_timebins = spec.shape[-1]
        spec_mels = spec.shape[-2]
        
        remainder = spec_timebins % patch_width

        # cut off small parts of end of spectrograms as they don't fit in the patch dim 
        if remainder != 0: 
            spec = spec[:,:,:spec_timebins-remainder]

        # new spec length, cuts off bits that wont fit in the patches neatly 
        rounded_spec_length = spec.shape[-1]

        # match the current spec file with the labels, gets list of detected vocalizations 
        matched_events = event_map.get(file_name, [])

        # skip the file if it has no labels 
        if not matched_events:
            i += 1
            continue

        if matched_events:
            for matched_event in matched_events: # could be multiple songs for each file 
                # crop the non song elements 
                labels = create_label_arr(matched_event, rounded_spec_length)

                spec_detected = spec[:,:,matched_event["on_timebins"]:matched_event["off_timebins"]]
                # since the labels are based off the rounded spec, but not based off the detected event, we crop it further to match the dim of spec
                labels = labels[matched_event["on_timebins"]:matched_event["off_timebins"]]                
                # reshape spec into batches to fit into context window

                # if spec longer than the context 
                if spec_detected.shape[-1] > model_num_timebins:
                    batch_size = (spec_detected.shape[-1] // model_num_timebins) + 1 # +1 to account for the padding creating an additional batch (important)
                    pad_amnt = model_num_timebins - (spec_detected.shape[-1] % model_num_timebins)

                elif spec_detected.shape[-1] < model_num_timebins:
                    pad_amnt = model_num_timebins - spec_detected.shape[-1]
                    batch_size = 1 

                # context equal exactly to spec size 
                else:
                    pad_amnt = 0
                    batch_size = 1 
                
                if pad_amnt > 0: 
                    spec_detected = torch.nn.functional.pad(spec_detected, (0, pad_amnt), mode='constant', value=0)
                    labels = torch.nn.functional.pad(labels, (0, pad_amnt), mode='constant', value=-1) # we gotta pad this shi to match abv

                
                channel, mel, time = spec_detected.shape
                # Correctly batch by slicing along time dimension
                # spec_detected shape: (channel, mel, time)
                # We want: (batch_size, channel, mel, model_num_timebins)
                batched_spec_detected = spec_detected.reshape(channel, mel, batch_size, model_num_timebins)
                batched_spec_detected = batched_spec_detected.permute(2, 0, 1, 3)  # (batch_size, channel, mel, model_num_timebins)
                with torch.no_grad():
                    z = model.forward_encoder_inference(batched_spec_detected)       # [B,S,D]
                    B, NP, D = z.shape # batch, num patches, dim
                
                # reshape z into something resembling a grid
                print(z.shape)
                z = z.permute(0, 2, 1) # now batch, d  s 
                print(z.shape)

                z = z.reshape(B, D, num_patches_height, num_patches_time)

                latent_list.append(z)
                label_list.append(labels)
                total_timebins+=batched_spec_detected.shape[0] * batched_spec_detected.shape[-1] # add the number of timebins 
        i += 1

    z = torch.cat(latent_list, dim=0)  # shape is batch, d , h, w 
    print(f"z after cat: {z.shape}")
    batches = z.shape[0]
    z = z.permute(0,3,1,2) # now batch, w, h, d 
    print(f"z after permute: {z.shape}")
    z = z.flatten(0,1) # samples x h, w
    print(f"z after flatten(0,1): {z.shape}")

    # stack height patches onto temporal patches 
    # (3072, 384, 4) will be 3072, 384 * 4 
    z = z.flatten(1,2)
    print(f"z after flatten(1,2): {z.shape}")

    syllable_labels = torch.cat(label_list, dim=0)  # samples, labels
    pos_ids = torch.arange(0, 1024).repeat(batches)
    pos_ids = np.asarray(pos_ids, dtype=int)[:z.shape[0]]
    
    syllable_labels = syllable_labels.cpu().numpy()
    z = z.numpy()

    print(f"Shape of embeddings: {z.shape}")
    
    # Count syllable types
    unique_syllables, counts = np.unique(syllable_labels, return_counts=True)
    unlabeled_count = np.sum(syllable_labels == -1)

    print(f"Syllable type counts: {dict(zip(unique_syllables, counts))}")
    print(f"Non syllable song patches: {unlabeled_count}")
    
    # removal of average vector per position 
    if args.get("mode") == "absolute":
        uniq = np.unique(pos_ids)
        means = np.zeros((uniq.max()+1, z.shape[1]), dtype=np.float32)
        for p in uniq:
            means[p] = z[pos_ids == p].mean(axis=0)
        z = z - means[pos_ids]
        print("Applied absolute position removal")
    
    print(z.shape)

    song_index = np.where(syllable_labels != -1)[0]
    z = z[song_index]
    syllable_labels = syllable_labels[song_index]

    # Apply UMAP to the PCA-reduced datas
    reducer_enc = umap.UMAP(n_components=2, n_neighbors=100, metric='cosine')
    emb_enc = reducer_enc.fit_transform(z)
    print("UMAP done")

    # # --- save arrays to NPZ (labels, spec, embeddings, pos_ids) ---
    # if args.get("npz_out"):
    #     spec_arr = (
    #         torch.cat(spec_batches_all, dim=0).numpy()
    #         if spec_batches_all else
    #         np.empty((0, 1, config["mels"], config["num_timebins"]), dtype=np.float32)
    #     )
    #     np.savez(
    #         args["npz_out"],
    #         labels=syllable_labels,     # shape: (N_patches,)
    #         spec=spec_arr,              # shape: (total_chunks, 1, mels, CHUNK)
    #         embeddings=Z_np,            # shape: (N_patches, H*D), freq-stacked
    #         embeddings_pca=Z_pca,       # shape: (N_patches, 32), PCA-reduced
    #         embeddings_umap=emb_enc,    # shape: (N_patches, 2), UMAP 2D embedding
    #         pos_ids=pos_ids             # shape: (N_patches,)
    #     )
    #     print(f"NPZ saved to {args['npz_out']}")
    
    # Create scatter plot colored by syllable types
    plt.figure(figsize=(8, 8), dpi=300)
    
    # Determine title based on depositioning mode
    if args.get("mode") == "absolute":
        title = "B"
    else:
        title = "A"
    
    if args.get("json_path") and len(event_map) > 0:
        # Define colors for different syllable types
        import matplotlib.cm as cm
        mask_labeled = syllable_labels != -1
        labeled_vals = syllable_labels[mask_labeled]
        unique_syllables = np.unique(labeled_vals) if labeled_vals.size > 0 else np.array([], dtype=int)
        colors = cm.tab20(np.linspace(0, 1, len(unique_syllables))) if len(unique_syllables) > 0 else []
        color_map = dict(zip(unique_syllables.tolist(), colors))

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

    # we gotta adapt this later 
    # # --- Generate 5 random CHUNK-length spectrograms with aligned 1D color bars ---
    # # GEORGE SAYS THIS IS TEMPORARY: uses same color palette as UMAP points.
    # if all_chunks:
    #     rnd_count = min(5, len(all_chunks))
    #     sel = np.random.choice(len(all_chunks), size=rnd_count, replace=False)
    #     unlabeled_rgb = np.array(mcolors.to_rgb("#404040"))  # light black for non-song portions
    #     for k, idx in enumerate(sel, start=1):
    #         chunk = all_chunks[idx]        # [1, mels, CHUNK]
    #         labs  = all_chunk_labels[idx]  # length W_const
    #         mels, T = chunk.shape[-2], chunk.shape[-1]
    #         # derive patch geometry
    #         patch_width = config["patch_width"]
    #         W_const = int(T // patch_width) if T % patch_width == 0 else int(np.ceil(T / patch_width))
    #         # build bar image [h, T, 3], repeat per-timebin color from per-patch labels
    #         bar_h = 10
    #         bar = np.ones((bar_h, T, 3), dtype=np.float32) * unlabeled_rgb.reshape(1, 1, 3)
    #         for p in range(min(W_const, len(labs))):
    #             uid = labs[p]
    #             # rgba->rgb if present; else unlabeled
    #             rgb = np.array(color_map.get(uid, (*unlabeled_rgb, 1.0)))[:3]
    #             a = p * patch_width
    #             b = min(T, (p + 1) * patch_width)
    #             bar[:, a:b, :] = rgb
    #         # plot spectrogram + bar
    #         fig = plt.figure(figsize=(10, 6), dpi=300)
    #         gs = fig.add_gridspec(2, 1, height_ratios=[4, 0.3], hspace=0.05)
    #         ax1 = fig.add_subplot(gs[0, 0])
    #         ax2 = fig.add_subplot(gs[1, 0])
    #         ax1.imshow(chunk.squeeze(0).numpy(), aspect="auto", origin="lower", interpolation="none")
    #         ax1.set_xticks([]); ax1.set_yticks([])
    #         ax2.imshow(bar, aspect="auto", origin="lower", interpolation="nearest")
    #         ax2.set_xticks([]); ax2.set_yticks([])
    #         # save
    #         sp_path = out_dir / f"{spec_slug}__{run_slug}__tb{args['num_timebins']}__spec{k}.png"
    #         fig.savefig(sp_path, bbox_inches="tight", dpi=300)
    #         plt.close(fig)
    #         print(f"Spectrogram saved to {sp_path}")

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
    parser.add_argument("--mode", type=str, default="absolute", help="mode for position-based removal: 'absolute' or 'relative'")

    args = parser.parse_args()
    main(vars(args))
