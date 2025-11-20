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
    parser.add_argument("--output_length", type=int, default=5000, help="Output length in time bins")
    parser.add_argument("--fg_proportion", type=float, default=0.1, help="Proportion of output that is foreground (0.0-1.0)")
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

    # Build lightweight index of foreground chunks (don't load data yet)
    fg_chunk_index = []  # List of (fg_filename, chunk_idx, events)
    for fg_filename, events in fg_recordings:
        # Load just to get the shape
        fg_spec_shape = np.load(fg_spec_paths[fg_filename], mmap_mode='r').shape
        num_chunks = fg_spec_shape[1] // args.snippet_size
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * args.snippet_size
            chunk_end = chunk_start + args.snippet_size
            
            # Find events that overlap with this chunk
            chunk_start_ms = chunk_start * ms_per_frame
            chunk_end_ms = chunk_end * ms_per_frame
            
            chunk_events = []
            for event in events:
                if event["offset_ms"] > chunk_start_ms and event["onset_ms"] < chunk_end_ms:
                    # Clip event to chunk boundaries
                    event_onset_clipped = max(event["onset_ms"], chunk_start_ms)
                    event_offset_clipped = min(event["offset_ms"], chunk_end_ms)
                    
                    # Store event relative to chunk start
                    relative_event = {
                        "onset_ms": event_onset_clipped - chunk_start_ms,
                        "offset_ms": event_offset_clipped - chunk_start_ms
                    }
                    
                    # Check if original event has units
                    if "units" in event:
                        relative_units = []
                        for unit in event["units"]:
                            if unit["offset_ms"] > chunk_start_ms and unit["onset_ms"] < chunk_end_ms:
                                unit_onset_clipped = max(unit["onset_ms"], chunk_start_ms)
                                unit_offset_clipped = min(unit["offset_ms"], chunk_end_ms)
                                
                                relative_units.append({
                                    "onset_ms": unit_onset_clipped - chunk_start_ms,
                                    "offset_ms": unit_offset_clipped - chunk_start_ms,
                                    "id": unit["id"]
                                })
                        
                        if relative_units:
                            relative_event["units"] = relative_units
                            chunk_events.append(relative_event)
                    else:
                        chunk_events.append(relative_event)
            
            # Only add chunks that have detected events
            if chunk_events:
                fg_chunk_index.append((fg_filename, chunk_idx, chunk_events))
    
    print(f"Found {len(fg_chunk_index)} foreground chunks with detected events")
    
    if len(fg_chunk_index) == 0:
        print("ERROR: No foreground chunks with events found. Check your annotations JSON.")
        return
    
    def load_fg_chunk(fg_filename, chunk_idx):
        """Load and process a foreground chunk on-demand."""
        fg_spec = np.load(fg_spec_paths[fg_filename])
        
        # Z-score foreground independently
        fg_mean_local = fg_spec.mean()
        fg_std_local = fg_spec.std()
        fg_spec_zscore = (fg_spec - fg_mean_local) / fg_std_local
        
        # Extract chunk
        chunk_start = chunk_idx * args.snippet_size
        chunk_end = chunk_start + args.snippet_size
        return fg_spec_zscore[:, chunk_start:chunk_end]
    
    # Generate synthetic data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    synthetic_recordings = []
    
    # For computing statistics on the fly
    total_sum = 0.0
    total_sq_sum = 0.0
    total_values = 0
    
    with tqdm(total=args.num_files, desc="Generating synthetic specs") as pbar:
        for synthetic_idx in range(args.num_files):
            # Pick random background that's large enough
            attempts = 0
            max_attempts = 50
            while attempts < max_attempts:
                bg_name = np.random.choice(bg_names)
                bg_spec = np.load(bg_spec_paths[bg_name])
                
                if bg_spec.shape[1] >= args.output_length:
                    break
                attempts += 1
            
            # Skip if no suitable background found
            if bg_spec.shape[1] < args.output_length:
                print(f"\nWarning: No background large enough for output_length={args.output_length}")
                continue
            
            # Get number of mel bins from background
            n_mels = bg_spec.shape[0]
            
            # Extract a segment of background at random position
            max_bg_start = bg_spec.shape[1] - args.output_length
            bg_start = np.random.randint(0, max_bg_start + 1) if max_bg_start > 0 else 0
            bg_segment = bg_spec[:, bg_start:bg_start + args.output_length].copy()
            
            # Free up background spec memory
            del bg_spec
            
            # Compute background stats
            bg_mean_local = bg_segment.mean()
            bg_std_local = bg_segment.std()
            
            # Calculate how many timebins should be foreground
            total_fg_bins = int(args.output_length * args.fg_proportion)
            
            # Track which positions are already occupied and collect events
            occupied = np.zeros(args.output_length, dtype=bool)
            all_events = []
            fg_sources = []
            
            # Keep adding foreground chunks until we reach desired proportion
            bins_placed = 0
            while bins_placed < total_fg_bins and len(fg_chunk_index) > 0:
                # Pick random foreground chunk
                chunk_idx_in_list = np.random.randint(0, len(fg_chunk_index))
                fg_filename, chunk_idx, chunk_events = fg_chunk_index[chunk_idx_in_list]
                
                # Load chunk on-demand
                fg_chunk_zscore = load_fg_chunk(fg_filename, chunk_idx)
                chunk_width = fg_chunk_zscore.shape[1]
                
                # Find available positions
                available_positions = []
                for pos in range(args.output_length - chunk_width + 1):
                    if not occupied[pos:pos + chunk_width].any():
                        available_positions.append(pos)
                
                if not available_positions:
                    break
                
                # Pick random position
                insert_pos = available_positions[np.random.randint(0, len(available_positions))]
                
                # Re-zscore foreground chunk wrt background
                fg_chunk_renorm = fg_chunk_zscore * bg_std_local + bg_mean_local
                
                # Place chunk
                bg_segment[:, insert_pos:insert_pos + chunk_width] = fg_chunk_renorm
                occupied[insert_pos:insert_pos + chunk_width] = True
                bins_placed += chunk_width
                fg_sources.append(fg_filename)
                
                # Free up chunk memory
                del fg_chunk_zscore, fg_chunk_renorm
                
                # Adjust event times to new position
                for event in chunk_events:
                    new_event = {
                        "onset_ms": (insert_pos * ms_per_frame) + event["onset_ms"],
                        "offset_ms": (insert_pos * ms_per_frame) + event["offset_ms"]
                    }
                    
                    if "units" in event:
                        new_units = []
                        for unit in event["units"]:
                            new_units.append({
                                "onset_ms": (insert_pos * ms_per_frame) + unit["onset_ms"],
                                "offset_ms": (insert_pos * ms_per_frame) + unit["offset_ms"],
                                "id": unit["id"]
                            })
                        new_event["units"] = new_units
                    
                    all_events.append(new_event)
            
            # Add noise
            noise = np.random.randn(*bg_segment.shape) * args.noise_std * bg_std_local
            synthetic_spec = bg_segment + noise
            
            # Save synthetic spectrogram
            synthetic_name = f"synthetic_{synthetic_idx:05d}"
            np.save(output_dir / f"{synthetic_name}.npy", synthetic_spec)
            
            # Update statistics
            total_sum += synthetic_spec.sum()
            total_sq_sum += np.square(synthetic_spec).sum()
            total_values += synthetic_spec.size
            
            # Sort events by onset time
            all_events.sort(key=lambda e: e["onset_ms"])
            
            synthetic_recordings.append({
                "recording": {
                    "filename": f"{synthetic_name}.npy",
                    "source_fg": list(set(fg_sources)),
                    "source_bg": bg_name,
                    "fg_proportion_actual": bins_placed / args.output_length
                },
                "detected_events": all_events
            })
            
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
    
    print(f"Generated {len(synthetic_recordings)} synthetic spectrograms in {output_dir}")
    print(f"Output length: {args.output_length} timebins")
    print(f"Target FG proportion: {args.fg_proportion:.2%}")
    print(f"Synthetic data stats - Mean: {synthetic_mean:.4f}, Std: {synthetic_std:.4f}")


if __name__ == "__main__":
    main()



    