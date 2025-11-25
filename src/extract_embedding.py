import argparse
from model import TinyBird
from data_loader import SpectogramDataset
from utils import load_model_from_checkpoint
import torch
import torch.nn.functional as F
import umap
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import torch.nn.functional as F
from matplotlib import colors as mcolors

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

    # Lists to store results
    latent_list = []
    patch_list = []
    label_list_original = []
    label_list_downsampled = []
    spec_list = []
    pos_ids_list = []

    total_timebins = 0
    i = 0

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

                # converts to spectrogram timebins 
                spec_detected = spec[:,:,matched_event["on_timebins"]:matched_event["off_timebins"]]
                # since the labels are based off the rounded spec, but not based off the detected event, we crop it further to match the dim of spec
                labels = labels[matched_event["on_timebins"]:matched_event["off_timebins"]]                
                
                if spec_detected.shape[-1] == 0:
                    continue

                # reshape spec into batches to fit into context window
                # Calculate batch size and padding to avoid omitting data
                num_full_batches = spec_detected.shape[-1] // model_num_timebins
                remainder_bins = spec_detected.shape[-1] % model_num_timebins
                
                if remainder_bins > 0:
                    batch_size = num_full_batches + 1
                    pad_amnt = model_num_timebins - remainder_bins
                else:
                    batch_size = num_full_batches
                    pad_amnt = 0
                    
                if batch_size == 0:
                    batch_size = 1
                    pad_amnt = model_num_timebins - spec_detected.shape[-1]

                # Pad if necessary
                if pad_amnt > 0: 
                    spec_detected = torch.nn.functional.pad(spec_detected, (0, pad_amnt), mode='constant', value=0)
                    labels = torch.nn.functional.pad(labels, (0, pad_amnt), mode='constant', value=-1)
                
                channel, mel, time = spec_detected.shape
                
                # Correctly batch by slicing along time dimension
                batched_spec_detected = spec_detected.reshape(channel, mel, batch_size, model_num_timebins)
                batched_spec_detected = batched_spec_detected.permute(2, 0, 1, 3)  # (batch_size, channel, mel, model_num_timebins)
                
                with torch.no_grad():
                    h, z_seq = model.forward_encoder_inference(batched_spec_detected)
                    # h: encoded embeddings [B, NP, D]
                    # z_seq: patch embeddings [B, NP, D]
                    B, NP, D = h.shape

                # Reshape and process embeddings
                # Reconstruct grid [B, D, H, W]
                h_grid = h.permute(0, 2, 1).reshape(B, D, num_patches_height, num_patches_time)
                z_grid = z_seq.permute(0, 2, 1).reshape(B, D, num_patches_height, num_patches_time)

                # Permute to [B, W, H, D] to align with time
                h_time = h_grid.permute(0, 3, 2, 1) 
                z_time = z_grid.permute(0, 3, 2, 1)

                # Flatten batch and time dimensions: [B*W, H, D] -> flatten H -> [B*W, H*D]
                # We flatten B and W first to easily trim padding
                h_flat = h_time.reshape(-1, num_patches_height * D)
                z_flat = z_time.reshape(-1, num_patches_height * D)
                
                # Prepare spectrograms for saving: flatten B and T -> [B*T, M]
                # batched_spec_detected: [B, C, M, T]
                spec_flat = batched_spec_detected.squeeze(1).permute(0, 2, 1).reshape(-1, mel)

                # Prepare labels
                # labels is currently [B*T] (since it was padded to match spec)
                
                # Generate Pos IDs
                # 0..W-1 repeated B times
                current_pos_ids = torch.arange(0, num_patches_time).repeat(batch_size)

                # TRIM PADDING
                if pad_amnt > 0:
                    # Trim embeddings (patches)
                    pad_patches = pad_amnt // patch_width
                    h_flat = h_flat[:-pad_patches]
                    z_flat = z_flat[:-pad_patches]
                    current_pos_ids = current_pos_ids[:-pad_patches]
                    
                    # Trim spectrograms and labels (timebins)
                    spec_flat = spec_flat[:-pad_amnt]
                    labels = labels[:-pad_amnt]

                # Downsample labels to patch resolution
                # We do this after trimming to ensure alignment
                # labels is 1D [Timebins]
                labels_reshaped = labels.float().view(1, 1, -1)
                labels_down = F.max_pool1d(labels_reshaped, kernel_size=patch_width, stride=patch_width).view(-1).long()

                # Append to lists
                latent_list.append(h_flat.cpu())
                patch_list.append(z_flat.cpu())
                label_list_original.append(labels.cpu())
                label_list_downsampled.append(labels_down.cpu())
                spec_list.append(spec_flat.cpu())
                pos_ids_list.append(current_pos_ids.cpu())
                
                total_timebins += spec_flat.shape[0]

        i += 1

    # Process encoded embeddings (h)
    h = torch.cat(latent_list, dim=0)  # [TotalPatches, H*D]
    
    # Process patch embeddings (z_seq)
    z_seq = torch.cat(patch_list, dim=0) # [TotalPatches, H*D]
    
    # Process labels
    labels_original = torch.cat(label_list_original, dim=0).numpy()
    syllable_labels_downsampled = torch.cat(label_list_downsampled, dim=0).numpy()
    
    # Process spectrograms
    spectrograms = torch.cat(spec_list, dim=0) # [TotalTimebins, Mel]
    spectrograms_np = spectrograms.numpy()
    
    # Process pos_ids
    pos_ids = torch.cat(pos_ids_list, dim=0)
    pos_ids = np.asarray(pos_ids.numpy(), dtype=int)
    
    # Convert embeddings to NP
    h_np = h.numpy()
    z_seq_np = z_seq.numpy()
    
    # Save embeddings before position removal
    encoded_embeddings_before_pos_removal = h_np.copy()
    patch_embeddings_before_pos_removal = z_seq_np.copy()
    
    # removal of average vector per position for encoded embeddings
    uniq = np.unique(pos_ids)
    means_h = np.zeros((uniq.max()+1, h_np.shape[1]), dtype=np.float32)
    for p in uniq:
        means_h[p] = h_np[pos_ids == p].mean(axis=0)
    h_np = h_np - means_h[pos_ids]
    
    # removal of average vector per position for patch embeddings
    means_z = np.zeros((uniq.max()+1, z_seq_np.shape[1]), dtype=np.float32)
    for p in uniq:
        means_z[p] = z_seq_np[pos_ids == p].mean(axis=0)
    z_seq_np = z_seq_np - means_z[pos_ids]

    # Save embeddings after position removal
    encoded_embeddings_after_pos_removal = h_np.copy()
    patch_embeddings_after_pos_removal = z_seq_np.copy()
    
    # Save embeddings before position removal
    encoded_embeddings_before_pos_removal = h_np.copy()
    patch_embeddings_before_pos_removal = z_seq_np.copy()
    
    # removal of average vector per position for encoded embeddings
    uniq = np.unique(pos_ids)
    means_h = np.zeros((uniq.max()+1, h_np.shape[1]), dtype=np.float32)
    for p in uniq:
        means_h[p] = h_np[pos_ids == p].mean(axis=0)
    h_np = h_np - means_h[pos_ids]
    
    # removal of average vector per position for patch embeddings
    means_z = np.zeros((uniq.max()+1, z_seq_np.shape[1]), dtype=np.float32)
    for p in uniq:
        means_z[p] = z_seq_np[pos_ids == p].mean(axis=0)
    z_seq_np = z_seq_np - means_z[pos_ids]

    # Save embeddings after position removal
    encoded_embeddings_after_pos_removal = h_np.copy()
    patch_embeddings_after_pos_removal = z_seq_np.copy()

    # we need audio params, patch stuff, checkpoint, spec, labels original, labels_downsampled, embedding before and after pos removal 
    np.savez(
        args["npz_dir"],
        # Spectrograms
        spectrograms=spectrograms_np,                           # shape: (total_timebins, mels) - 2D spectrogram array
        # Labels
        labels_original=labels_original,                        # shape: (total_timebins,) - original labels before downsampling
        labels_downsampled=syllable_labels_downsampled,         # shape: (N_patches,) - labels downsampled to patch resolution
        # Encoded Embeddings (after encoder)
        encoded_embeddings_before_pos_removal=encoded_embeddings_before_pos_removal,  # shape: (N_patches, H*D) - encoded embeddings before position removal
        encoded_embeddings_after_pos_removal=encoded_embeddings_after_pos_removal,    # shape: (N_patches, H*D) - encoded embeddings after position removal
        # Patch Embeddings (before encoder)
        patch_embeddings_before_pos_removal=patch_embeddings_before_pos_removal,      # shape: (N_patches, H*D) - patch embeddings before position removal
        patch_embeddings_after_pos_removal=patch_embeddings_after_pos_removal,        # shape: (N_patches, H*D) - patch embeddings after position removal
        pos_ids=pos_ids,                                        # shape: (N_patches,) - positional indices
        # Audio parameters (sr, n_mels, hop_size, fft)
        audio_sr=np.array(audio_params[0]),
        audio_n_mels=np.array(audio_params[1]),
        audio_hop_size=np.array(audio_params[2]),
        audio_fft=np.array(audio_params[3]),
        # Patch parameters
        patch_height=np.array(patch_height),
        patch_width=np.array(patch_width),
        num_patches_time=np.array(num_patches_time),
        num_patches_height=np.array(num_patches_height),
        # Model parameters
        checkpoint=np.array(args["checkpoint"] if args["checkpoint"] else ""),
        model_num_timebins=np.array(model_num_timebins),
        mels=np.array(mels),
    )
    print(f"NPZ saved to {args['npz_dir']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple UMAP embedding visualization")
    parser.add_argument("--num_timebins", type=int, default=12400, help="Number of time bins")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file (optional)")
    parser.add_argument("--spec_dir", type=str, required=True, help="Directory of specs to plot the embedding of")
    parser.add_argument("--npz_dir", type=str, required=True, help="Save arrays to this .npz path")
    parser.add_argument("--json_path", type=str, default=None, help="to provide song snippets + syllable labels")
    parser.add_argument("--bird", type=str, default=None, help="select specific bird number (e.g., 1 for bird1, 2 for bird2)")

    args = parser.parse_args()
    main(vars(args))
