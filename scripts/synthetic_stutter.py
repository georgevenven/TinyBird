import argparse
import json
import random
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

def ms_to_timebins(ms, sr, hop_length):
    return int((ms / 1000) * sr / hop_length)

def timebins_to_ms(bins, sr, hop_length):
    return (bins * hop_length / sr) * 1000

def load_audio_params(src_dir):
    params_path = src_dir / "audio_params.json"
    if params_path.exists():
        with open(params_path, 'r') as f:
            params = json.load(f)
            return params.get("sr", 32000), params.get("hop_size", 64)
    return 32000, 64

def process_single_file(spec_path, recording_entry, dst_dir, stutter_count, sr, hop_length):
    # Load spectrogram
    try:
        spec = np.load(spec_path)
    except Exception as e:
        print(f"Error loading {spec_path}: {e}")
        return None

    # Collect all units from all events
    all_units = []
    for event in recording_entry.get("detected_events", []):
        for unit in event.get("units", []):
            all_units.append(unit)
    
    # Sort units by onset
    all_units.sort(key=lambda x: x["onset_ms"])

    new_spec_parts = []
    new_units = []
    
    current_idx = 0
    time_shift_ms = 0
    
    # We will reconstruct events. Since we are modifying units globally, 
    # we might break the event structure if we just flatten.
    # However, to stutter, we repeat units. 
    # Let's reconstruct the spectrogram linearly and build a new single "detected_event" 
    # or try to preserve event structure if possible. 
    # Given the prompt "modifed json will reflect the adjustment", 
    # let's try to keep the event structure but with repeated units.
    
    # Actually, simpler to treat the whole file as a timeline.
    # We will build:
    # 1. new_spec (concatenated arrays)
    # 2. new_detected_events (list of events with adjusted times)
    
    # To avoid complex event logic, let's iterate through units and gaps.
    # But we need to group them back into events for the JSON.
    # Strategy: 
    # - Calculate the shift for each unit.
    # - Apply shift to unit and repeat it.
    # - Apply shift to the gaps between units.
    
    # Let's iterate through the original timeline
    last_original_end_bin = 0
    last_original_end_ms = 0.0
    
    current_write_ms = 0.0
    
    # We need to preserve the event grouping.
    # New recording entry structure
    new_recording_entry = {
        "recording": recording_entry["recording"],
        "detected_events": []
    }
    
    # We process events in order
    sorted_events = sorted(recording_entry.get("detected_events", []), key=lambda x: x["onset_ms"])
    
    for event in sorted_events:
        event_onset_ms = event["onset_ms"]
        event_offset_ms = event["offset_ms"]
        
        # Add gap before this event (if any)
        gap_start_bin = ms_to_timebins(last_original_end_ms, sr, hop_length)
        gap_end_bin = ms_to_timebins(event_onset_ms, sr, hop_length)
        
        if gap_end_bin > gap_start_bin:
            # Append gap from spec
            # Check bounds
            gap_start_bin = max(0, gap_start_bin)
            gap_end_bin = min(spec.shape[1], gap_end_bin)
            if gap_end_bin > gap_start_bin:
                new_spec_parts.append(spec[:, gap_start_bin:gap_end_bin])
                
        # Update current write time by adding the gap duration
        current_write_ms += (event_onset_ms - last_original_end_ms)
        
        # Now process units within this event
        units = sorted(event.get("units", []), key=lambda x: x["onset_ms"])
        
        new_event_units = []
        new_event_onset = current_write_ms # Event starts here (roughly) - will be refined by first unit? 
        # Actually event onset should probably be adjusted by the cumulative shift so far.
        # But "gap" calculation above handles the shift implicitly by appending to new_spec_parts.
        # So current_write_ms tracks the end of the reconstructed spectrogram in ms.
        
        # Wait, current_write_ms calculation above is slightly wrong if we just added bins.
        # Better to track bins for spec and ms for JSON separately, 
        # but ensure they stay synced.
        
        # Let's track everything in MS for JSON, and slice spec using MS->Bins.
        
        # Re-initialize for safety
        new_spec_parts = []
        current_spec_ms = 0.0 # Position in original spec
        cumulative_shift_ms = 0.0
        
        new_events_list = []

    # RESTART LOOP WITH BETTER LOGIC
    
    # 1. Flatten everything into a list of "segments": (start_ms, end_ms, type, data)
    # type: 'gap' or 'unit'
    # data: unit_dict or None
    
    segments = []
    
    # Sort events
    events = sorted(recording_entry.get("detected_events", []), key=lambda x: x["onset_ms"])
    
    curr_t = 0.0
    for event in events:
        # Gap before event
        if event["onset_ms"] > curr_t:
            segments.append({"type": "gap", "start": curr_t, "end": event["onset_ms"]})
            curr_t = event["onset_ms"]
            
        # Process units in event
        event_units = sorted(event.get("units", []), key=lambda x: x["onset_ms"])
        
        # Gap between event start and first unit
        if event_units and event_units[0]["onset_ms"] > curr_t:
             segments.append({"type": "gap", "start": curr_t, "end": event_units[0]["onset_ms"]})
             curr_t = event_units[0]["onset_ms"]
        elif not event_units:
            # No units, just gap until event end? Or treat whole event as gap?
            # If event has no units, we just copy it as is?
            # Let's assume we just copy the spectrogram part.
            pass

        for i, unit in enumerate(event_units):
            # Gap before unit (inside event)
            if unit["onset_ms"] > curr_t:
                segments.append({"type": "gap", "start": curr_t, "end": unit["onset_ms"]})
            
            # The Unit itself
            segments.append({"type": "unit", "start": unit["onset_ms"], "end": unit["offset_ms"], "data": unit})
            curr_t = unit["offset_ms"]
            
        # Gap after last unit to event end
        if event["offset_ms"] > curr_t:
             segments.append({"type": "gap", "start": curr_t, "end": event["offset_ms"]})
             curr_t = event["offset_ms"]
             
    # Gap after last event to end of file (we don't know end of file MS exactly without spec len)
    # We'll handle tail after loop
    
    new_spec_parts = []
    new_detected_events = []
    
    # We need to group new units into events. 
    # Strategy: Create a new event whenever we encounter a unit belonging to a new original event?
    # Or just simplify: 1 original event -> 1 new event (expanded).
    
    current_new_ms = 0.0
    
    # Iterate original events again to preserve structure
    curr_orig_t = 0.0
    
    for event in events:
        # 1. Handle gap before event
        gap_dur = event["onset_ms"] - curr_orig_t
        if gap_dur > 0:
            start_bin = ms_to_timebins(curr_orig_t, sr, hop_length)
            end_bin = ms_to_timebins(event["onset_ms"], sr, hop_length)
            if end_bin > start_bin:
                 # Check bounds
                end_bin = min(spec.shape[1], end_bin)
                new_spec_parts.append(spec[:, start_bin:end_bin])
            current_new_ms += gap_dur
            curr_orig_t = event["onset_ms"]
            
        # 2. Start new event
        new_event = {
            "onset_ms": current_new_ms,
            "offset_ms": 0.0, # will set later
            "units": []
        }
        
        event_units = sorted(event.get("units", []), key=lambda x: x["onset_ms"])
        if not event_units:
             # Event without units? Just copy the duration
             dur = event["offset_ms"] - event["onset_ms"]
             start_bin = ms_to_timebins(event["onset_ms"], sr, hop_length)
             end_bin = ms_to_timebins(event["offset_ms"], sr, hop_length)
             if end_bin > start_bin:
                end_bin = min(spec.shape[1], end_bin)
                new_spec_parts.append(spec[:, start_bin:end_bin])
             
             current_new_ms += dur
             curr_orig_t = event["offset_ms"]
             new_event["offset_ms"] = current_new_ms
             new_detected_events.append(new_event)
             continue

        # Process units
        for unit in event_units:
            # Gap before unit
            gap_dur = unit["onset_ms"] - curr_orig_t
            if gap_dur > 0:
                start_bin = ms_to_timebins(curr_orig_t, sr, hop_length)
                end_bin = ms_to_timebins(unit["onset_ms"], sr, hop_length)
                if end_bin > start_bin:
                    end_bin = min(spec.shape[1], end_bin)
                    new_spec_parts.append(spec[:, start_bin:end_bin])
                current_new_ms += gap_dur
                curr_orig_t = unit["onset_ms"]
            
            # The Unit - REPEAT stutter_count TIMES
            unit_dur = unit["offset_ms"] - unit["onset_ms"]
            start_bin = ms_to_timebins(unit["onset_ms"], sr, hop_length)
            end_bin = ms_to_timebins(unit["offset_ms"], sr, hop_length)
            
            unit_spec_slice = None
            if end_bin > start_bin:
                 end_bin = min(spec.shape[1], end_bin)
                 unit_spec_slice = spec[:, start_bin:end_bin]
            
            for _ in range(stutter_count):
                if unit_spec_slice is not None:
                    new_spec_parts.append(unit_spec_slice)
                
                new_unit = {
                    "onset_ms": current_new_ms,
                    "offset_ms": current_new_ms + unit_dur,
                    "id": unit["id"]
                }
                new_event["units"].append(new_unit)
                current_new_ms += unit_dur
            
            curr_orig_t = unit["offset_ms"]
            
        # Gap after last unit to event end
        gap_dur = event["offset_ms"] - curr_orig_t
        if gap_dur > 0:
            start_bin = ms_to_timebins(curr_orig_t, sr, hop_length)
            end_bin = ms_to_timebins(event["offset_ms"], sr, hop_length)
            if end_bin > start_bin:
                end_bin = min(spec.shape[1], end_bin)
                new_spec_parts.append(spec[:, start_bin:end_bin])
            current_new_ms += gap_dur
            curr_orig_t = event["offset_ms"]
            
        new_event["offset_ms"] = current_new_ms
        new_detected_events.append(new_event)
        
    # Handle tail of file
    start_bin = ms_to_timebins(curr_orig_t, sr, hop_length)
    if start_bin < spec.shape[1]:
        new_spec_parts.append(spec[:, start_bin:])
        
    # Concatenate spec
    if new_spec_parts:
        new_spec = np.concatenate(new_spec_parts, axis=1)
    else:
        new_spec = np.zeros((spec.shape[0], 0))
        
    # Save
    stem = Path(spec_path).stem
    out_name = f"{stem}_stuttered.npy"
    out_path = dst_dir / out_name
    np.save(out_path, new_spec)
    
    # Update recording entry
    new_recording_entry["recording"]["filename"] = out_name
    new_recording_entry["detected_events"] = new_detected_events
    
    return new_recording_entry

def main():
    parser = argparse.ArgumentParser(description="Synthetic stutter generator")
    parser.add_argument("--src_dir", type=Path, required=True, help="Directory with spectrograms (.npy)")
    parser.add_argument("--json_path", type=Path, required=True, help="Path to annotations.json")
    parser.add_argument("--dst_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--n_files", type=int, default=10, help="Number of files to process")
    parser.add_argument("--stutter_count", type=int, default=2, help="Number of times to repeat each unit")
    parser.add_argument("--sr", type=int, default=None, help="Sample rate (optional)")
    parser.add_argument("--hop_length", type=int, default=None, help="Hop length (optional)")

    args = parser.parse_args()
    
    args.dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy audio_params.json if it exists
    src_params = args.src_dir / "audio_params.json"
    if src_params.exists():
        shutil.copy(src_params, args.dst_dir / "audio_params.json")
    
    # Load audio params
    sr, hop_length = load_audio_params(args.src_dir)
    if args.sr: sr = args.sr
    if args.hop_length: hop_length = args.hop_length
    
    print(f"Using SR: {sr}, Hop Length: {hop_length}")
    
    # Load JSON
    with open(args.json_path, 'r') as f:
        data = json.load(f)
        
    recordings = data.get("recordings", [])
    
    # Filter recordings that have corresponding files
    available_files = list(args.src_dir.glob("*.npy"))
    available_stems = {f.stem: f for f in available_files}
    
    valid_recordings = []
    for rec in recordings:
        fname = rec["recording"]["filename"]
        # Handle cases where filename has extension or not
        stem = Path(fname).stem
        if stem in available_stems:
            valid_recordings.append((available_stems[stem], rec))
            
    if not valid_recordings:
        print("No matching files found between JSON and src_dir")
        return

    # Select random sample
    n = min(args.n_files, len(valid_recordings))
    selected = random.sample(valid_recordings, n)
    
    new_recordings = []
    
    print(f"Processing {n} files...")
    for spec_path, rec_entry in tqdm(selected):
        new_entry = process_single_file(spec_path, rec_entry, args.dst_dir, args.stutter_count, sr, hop_length)
        if new_entry:
            new_recordings.append(new_entry)
            
    # Save new JSON
    new_json = {
        "metadata": data.get("metadata", {"units": "ms"}),
        "recordings": new_recordings
    }
    
    out_json_path = args.dst_dir / "annotations_stuttered.json"
    with open(out_json_path, 'w') as f:
        json.dump(new_json, f, indent=2)
        
    print(f"Done. Saved to {args.dst_dir}")

if __name__ == "__main__":
    main()
