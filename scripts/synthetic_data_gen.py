"""
Description:
Takes two dirs full of npy spectrograms shaped mels x time, one is foreground, one is background
We Z-score both the background and foreground independetly, then we re zscore the foreground wrt to the background
We take a snippet (size n timebins) with a detected event of the foreground, and place it onto the background (random location), we apply 1std (or whatever is declared) to the new generated spec
We then create a new json with the detected events, and save the synthetic spec in its own location
"""

import argparse
import json
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm


def get_spectrogram_paths(spec_dir):
    """Get paths to all .npy spectrograms in directory."""
    return {p.stem: p for p in Path(spec_dir).glob("*.npy")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fg_spec_dir", required=True, help="Foreground spectrogram directory")
    parser.add_argument("--bg_spec_dir", required=True, help="Background spectrogram directory")
    parser.add_argument("--fg_json", required=True, help="Foreground annotations JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory for synthetic data")
    parser.add_argument("--snippet_size", type=int, required=True, help="Snippet size in time bins")
    parser.add_argument("--noise_std", type=float, default=1.0, help="Noise std to apply")
    parser.add_argument("--num_files", type=int, default=100, help="Number of synthetic files to generate")
    args = parser.parse_args()

    # Load audio params from both directories
    fg_params_path = Path(args.fg_spec_dir) / "audio_params.json"
    bg_params_path = Path(args.bg_spec_dir) / "audio_params.json"
    
    with open(fg_params_path) as f:
        fg_params = json.load(f)
    with open(bg_params_path) as f:
        bg_params = json.load(f)
    
    ms_per_frame = (fg_params["hop_size"] / fg_params["sr"]) * 1000
    
    fg_mean, fg_std = fg_params["mean"], fg_params["std"]
    bg_mean, bg_std = bg_params["mean"], bg_params["std"]

    # Get spectrogram paths
    fg_spec_paths = get_spectrogram_paths(args.fg_spec_dir)
    bg_spec_paths = get_spectrogram_paths(args.bg_spec_dir)
    bg_names = list(bg_spec_paths.keys())
    
    with open(args.fg_json) as f:
        fg_data = json.load(f)

    # Collect all foreground recordings with their events
    fg_recordings = []
    for rec in fg_data["recordings"]:
        fg_filename = Path(rec["recording"]["filename"]).stem
        fg_recordings.append((fg_filename, rec["detected_events"]))

    # Generate synthetic data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    synthetic_recordings = []
    synthetic_idx = 0
    
    # For computing statistics on the fly
    total_sum = 0.0
    total_sq_sum = 0.0
    total_values = 0
    
    # Count total chunks we'll generate
    total_chunks = 0
    for fg_filename, _ in fg_recordings:
        fg_spec = np.load(fg_spec_paths[fg_filename])
        num_chunks = fg_spec.shape[1] // args.snippet_size
        total_chunks += num_chunks
    
    # Limit to num_files if specified
    chunks_to_generate = min(total_chunks, args.num_files)
    
    with tqdm(total=chunks_to_generate, desc="Generating synthetic specs") as pbar:
        for fg_filename, events in fg_recordings:
            if synthetic_idx >= args.num_files:
                break
                
            # Load foreground
            fg_spec = np.load(fg_spec_paths[fg_filename])
            
            # Z-score foreground independently
            fg_mean_local = fg_spec.mean()
            fg_std_local = fg_spec.std()
            fg_spec_zscore = (fg_spec - fg_mean_local) / fg_std_local
            
            # Split into chunks
            num_chunks = fg_spec.shape[1] // args.snippet_size
            
            for chunk_idx in range(num_chunks):
                if synthetic_idx >= args.num_files:
                    break
                    
                # Extract chunk
                chunk_start = chunk_idx * args.snippet_size
                chunk_end = chunk_start + args.snippet_size
                fg_chunk_zscore = fg_spec_zscore[:, chunk_start:chunk_end]
                
                # Pick random background that's large enough
                attempts = 0
                max_attempts = 50
                while attempts < max_attempts:
                    bg_name = np.random.choice(bg_names)
                    bg_spec = np.load(bg_spec_paths[bg_name])
                    
                    if bg_spec.shape[1] >= fg_chunk_zscore.shape[1]:
                        break
                    attempts += 1
                
                # Skip if no suitable background found
                if bg_spec.shape[1] < fg_chunk_zscore.shape[1]:
                    continue
                
                # Compute background stats
                bg_mean_local = bg_spec.mean()
                bg_std_local = bg_spec.std()
                
                # Re-zscore foreground chunk wrt background
                fg_chunk_renorm = fg_chunk_zscore * bg_std_local + bg_mean_local
                
                # Place chunk at random location in background
                max_start = bg_spec.shape[1] - fg_chunk_renorm.shape[1]
                if max_start > 0:
                    insert_pos = np.random.randint(0, max_start)
                else:
                    insert_pos = 0
                
                bg_spec[:, insert_pos:insert_pos + fg_chunk_renorm.shape[1]] = fg_chunk_renorm
                
                # Add noise
                noise = np.random.randn(*bg_spec.shape) * args.noise_std * bg_std_local
                synthetic_spec = bg_spec + noise
                
                # Save synthetic spectrogram
                synthetic_name = f"synthetic_{synthetic_idx:05d}"
                np.save(output_dir / f"{synthetic_name}.npy", synthetic_spec)
                
                # Update statistics
                total_sum += synthetic_spec.sum()
                total_sq_sum += np.square(synthetic_spec).sum()
                total_values += synthetic_spec.size
                
                # Find events that overlap with this chunk
                chunk_start_ms = chunk_start * ms_per_frame
                chunk_end_ms = chunk_end * ms_per_frame
                
                chunk_events = []
                for event in events:
                    # Check if event overlaps with chunk (not just entirely within)
                    if event["offset_ms"] > chunk_start_ms and event["onset_ms"] < chunk_end_ms:
                        # Clip event to chunk boundaries
                        event_onset_clipped = max(event["onset_ms"], chunk_start_ms)
                        event_offset_clipped = min(event["offset_ms"], chunk_end_ms)
                        
                        # Adjust event times to new position in background
                        new_onset_ms = (insert_pos * ms_per_frame) + (event_onset_clipped - chunk_start_ms)
                        new_offset_ms = (insert_pos * ms_per_frame) + (event_offset_clipped - chunk_start_ms)
                        
                        # Process units within this event
                        new_units = []
                        for unit in event.get("units", []):
                            # Check if unit overlaps with chunk
                            if unit["offset_ms"] > chunk_start_ms and unit["onset_ms"] < chunk_end_ms:
                                # Clip unit to chunk boundaries
                                unit_onset_clipped = max(unit["onset_ms"], chunk_start_ms)
                                unit_offset_clipped = min(unit["offset_ms"], chunk_end_ms)
                                
                                # Adjust unit times to new position
                                new_unit_onset = (insert_pos * ms_per_frame) + (unit_onset_clipped - chunk_start_ms)
                                new_unit_offset = (insert_pos * ms_per_frame) + (unit_offset_clipped - chunk_start_ms)
                                
                                new_units.append({
                                    "onset_ms": new_unit_onset,
                                    "offset_ms": new_unit_offset,
                                    "id": unit["id"]
                                })
                        
                        # Only add event if it has units
                        if new_units:
                            chunk_events.append({
                                "onset_ms": new_onset_ms,
                                "offset_ms": new_offset_ms,
                                "units": new_units
                            })
                
                synthetic_recordings.append({
                    "recording": {
                        "filename": f"{synthetic_name}.npy",
                        "source_fg": fg_filename,
                        "source_bg": bg_name
                    },
                    "detected_events": chunk_events
                })
                
                synthetic_idx += 1
                pbar.update(1)
    
    # Save annotations
    output_json = {
        "metadata": {"units": "ms"},
        "recordings": synthetic_recordings
    }
    
    with open(output_dir / "annotations.json", "w") as f:
        json.dump(output_json, f, indent=2)
    
    # Calculate statistics from synthetic data
    synthetic_mean = total_sum / total_values
    variance = (total_sq_sum / total_values) - (synthetic_mean * synthetic_mean)
    synthetic_std = math.sqrt(max(variance, 0))
    
    # Create audio_params.json from background params
    audio_params_output = {
        "fft": bg_params["fft"],
        "hop_size": bg_params["hop_size"],
        "mels": bg_params["mels"],
        "sr": bg_params["sr"],
        "mean": synthetic_mean,
        "std": synthetic_std
    }
    
    with open(output_dir / "audio_params.json", "w") as f:
        json.dump(audio_params_output, f, indent=2)
    
    print(f"Generated {synthetic_idx} synthetic spectrograms in {output_dir}")
    print(f"Synthetic data stats - Mean: {synthetic_mean:.4f}, Std: {synthetic_std:.4f}")


if __name__ == "__main__":
    main()



    