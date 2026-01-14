import json
import argparse
import os
import shutil
import glob
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
import random

def calculate_ms(detected_events):
    """Calculate total milliseconds in detected_events"""
    total_ms = 0
    for event in detected_events:
        total_ms += event['offset_ms'] - event['onset_ms']
    return total_ms

def split_data_random(input_file, spec_dir, train_dir, test_dir, train_percent=80):
    """Randomly split recordings without considering bird_id"""
    
    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Get all recordings and shuffle them
    recordings = data['recordings'].copy()
    random.shuffle(recordings)
    
    # Calculate split point
    split_idx = int(len(recordings) * (train_percent / 100))
    train_recordings = recordings[:split_idx]
    test_recordings = recordings[split_idx:]
    
    # Calculate statistics
    train_ms = sum(calculate_ms(r['detected_events']) for r in train_recordings)
    test_ms = sum(calculate_ms(r['detected_events']) for r in test_recordings)
    total_ms = train_ms + test_ms
    
    # Copy train spec files
    print("Copying train files...")
    for recording in tqdm(train_recordings):
        filename = recording['recording']['filename']
        base_name = os.path.splitext(filename)[0]
        matches = glob.glob(os.path.join(spec_dir, f"{base_name}.*"))
        for src in matches:
            dst = os.path.join(train_dir, os.path.basename(src))
            shutil.copy2(src, dst)
    
    # Copy test spec files
    print("Copying test files...")
    for recording in tqdm(test_recordings):
        filename = recording['recording']['filename']
        base_name = os.path.splitext(filename)[0]
        matches = glob.glob(os.path.join(spec_dir, f"{base_name}.*"))
        for src in matches:
            dst = os.path.join(test_dir, os.path.basename(src))
            shutil.copy2(src, dst)
    
    # Copy audio_params.json to both directories
    audio_params = os.path.join(spec_dir, "audio_params.json")
    if os.path.exists(audio_params):
        shutil.copy2(audio_params, os.path.join(train_dir, "audio_params.json"))
        shutil.copy2(audio_params, os.path.join(test_dir, "audio_params.json"))
    
    # Print stats
    print(f"\n=== Split Statistics (Random) ===")
    print(f"Total MS: {total_ms:.2f}")
    print(f"Train MS: {train_ms:.2f} ({train_ms/total_ms*100:.1f}%)")
    print(f"Test MS: {test_ms:.2f} ({test_ms/total_ms*100:.1f}%)")
    print(f"Train recordings: {len(train_recordings)}")
    print(f"Test recordings: {len(test_recordings)}")

def split_data(input_file, spec_dir, train_dir, test_dir, train_percent=80):
    """Split recordings by bird_id, aiming for train_percent of ms in train set"""
    
    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Group recordings by bird_id and calculate ms
    bird_data = defaultdict(lambda: {'recordings': [], 'total_ms': 0})
    
    for recording in data['recordings']:
        bird_id = recording['recording']['bird_id']
        ms = calculate_ms(recording['detected_events'])
        bird_data[bird_id]['recordings'].append(recording)
        bird_data[bird_id]['total_ms'] += ms
    
    # Sort birds by total_ms (helps with allocation)
    birds = sorted(bird_data.items(), key=lambda x: x[1]['total_ms'], reverse=True)
    
    print(f"Total unique birds: {len(birds)}")
    
    # Allocate birds to train/test
    total_ms = sum(b[1]['total_ms'] for b in birds)
    target_train_ms = total_ms * (train_percent / 100)
    
    # Find best allocation by trying all possible combinations
    best_train_birds = []
    best_diff = float('inf')
    
    # Try all possible subsets of birds for train set
    for r in range(len(birds) + 1):
        for train_combo in combinations(range(len(birds)), r):
            train_ms = sum(birds[i][1]['total_ms'] for i in train_combo)
            diff = abs(train_ms - target_train_ms)
            if diff < best_diff:
                best_diff = diff
                best_train_birds = list(train_combo)
    
    # Build train and test sets
    train_recordings = []
    test_recordings = []
    train_ms = 0
    train_birds = []
    test_birds = []
    
    for i, (bird_id, bird_info) in enumerate(birds):
        if i in best_train_birds:
            train_recordings.extend(bird_info['recordings'])
            train_ms += bird_info['total_ms']
            train_birds.append(bird_id)
        else:
            test_recordings.extend(bird_info['recordings'])
            test_birds.append(bird_id)
    
    # Copy train spec files
    print("Copying train files...")
    for recording in tqdm(train_recordings):
        filename = recording['recording']['filename']
        base_name = os.path.splitext(filename)[0]
        matches = glob.glob(os.path.join(spec_dir, f"{base_name}.*"))
        for src in matches:
            dst = os.path.join(train_dir, os.path.basename(src))
            shutil.copy2(src, dst)
    
    # Copy test spec files
    print("Copying test files...")
    for recording in tqdm(test_recordings):
        filename = recording['recording']['filename']
        base_name = os.path.splitext(filename)[0]
        matches = glob.glob(os.path.join(spec_dir, f"{base_name}.*"))
        for src in matches:
            dst = os.path.join(test_dir, os.path.basename(src))
            shutil.copy2(src, dst)
    
    # Copy audio_params.json to both directories
    audio_params = os.path.join(spec_dir, "audio_params.json")
    if os.path.exists(audio_params):
        shutil.copy2(audio_params, os.path.join(train_dir, "audio_params.json"))
        shutil.copy2(audio_params, os.path.join(test_dir, "audio_params.json"))
    
    # Print stats
    test_ms = total_ms - train_ms
    print(f"\n=== Split Statistics ===")
    print(f"Total MS: {total_ms:.2f}")
    print(f"Train MS: {train_ms:.2f} ({train_ms/total_ms*100:.1f}%)")
    print(f"Test MS: {test_ms:.2f} ({test_ms/total_ms*100:.1f}%)")
    print(f"Train recordings: {len(train_recordings)} from {len(train_birds)} birds")
    print(f"Test recordings: {len(test_recordings)} from {len(test_birds)} birds")
    print(f"Train birds: {train_birds}")
    print(f"Test birds: {test_birds}")

def filter_by_bird(input_file, spec_dir, output_dir, bird_id):
    """Copy all files for a specific bird_id from spec_dir to output_dir"""
    os.makedirs(output_dir, exist_ok=True)
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    bird_recordings = []
    files_to_copy = []
    for recording in data['recordings']:
        if recording['recording']['bird_id'] == bird_id:
            bird_recordings.append(recording)
            filename = recording['recording']['filename']
            base_name = os.path.splitext(filename)[0]
            files_to_copy.append(base_name)
    
    print(f"Found {len(files_to_copy)} recordings for bird {bird_id}")
    
    count = 0
    for base_name in tqdm(files_to_copy):
        matches = glob.glob(os.path.join(spec_dir, f"{base_name}.*"))
        for src in matches:
            dst = os.path.join(output_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            count += 1
            
    # Copy audio_params if exists
    audio_params = os.path.join(spec_dir, "audio_params.json")
    if os.path.exists(audio_params):
        shutil.copy2(audio_params, os.path.join(output_dir, "audio_params.json"))
        
    print(f"Copied {count} files to {output_dir}")

    # Write a filtered annotation JSON for downstream split steps (avoids scanning/copying unrelated recordings)
    filtered = dict(data)
    filtered["recordings"] = bird_recordings
    filtered_path = os.path.join(output_dir, "annotations_filtered.json")
    with open(filtered_path, "w") as f:
        json.dump(filtered, f, indent=2)
    print(f"Wrote filtered annotations: {filtered_path}")

def sample_files(spec_dir, output_dir, n_samples):
    """Randomly sample n_samples .npy files from spec_dir and copy to output_dir"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .npy files
    all_files = glob.glob(os.path.join(spec_dir, "*.npy"))
    if len(all_files) < n_samples:
        print(f"Warning: Requested {n_samples} samples but only found {len(all_files)}")
        selected_files = all_files
    else:
        selected_files = random.sample(all_files, n_samples)
        
    print(f"Sampling {len(selected_files)} files...")
    
    for src in tqdm(selected_files):
        dst = os.path.join(output_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        
    # Copy audio_params if exists
    audio_params = os.path.join(spec_dir, "audio_params.json")
    if os.path.exists(audio_params):
        shutil.copy2(audio_params, os.path.join(output_dir, "audio_params.json"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_dir', required=True)
    parser.add_argument('--train_dir') # Optional depending on mode
    parser.add_argument('--test_dir') # Optional depending on mode
    parser.add_argument('--annotation_json') # Optional depending on mode
    parser.add_argument('--train_percent', type=float, default=80)
    parser.add_argument('--ignore_bird_id', action='store_true', 
                        help='Randomly split files without grouping by bird_id')
    
    # New arguments
    parser.add_argument('--mode', type=str, default='split', choices=['split', 'filter_bird', 'sample'],
                        help='Mode: split (default), filter_bird, sample')
    parser.add_argument('--bird_id', type=str, help='Bird ID for filter_bird mode')
    parser.add_argument('--n_samples', type=int, help='Number of samples for sample mode')
    
    args = parser.parse_args()
    
    if args.mode == 'split':
        if not args.train_dir or not args.test_dir or not args.annotation_json:
             parser.error("--mode split requires --train_dir, --test_dir, and --annotation_json")
        if args.ignore_bird_id:
            split_data_random(args.annotation_json, args.spec_dir, args.train_dir, args.test_dir, args.train_percent)
        else:
            split_data(args.annotation_json, args.spec_dir, args.train_dir, args.test_dir, args.train_percent)
            
    elif args.mode == 'filter_bird':
        if not args.train_dir:
             parser.error("--mode filter_bird requires --train_dir (as output)")
        if not args.annotation_json or not args.bird_id:
             parser.error("--mode filter_bird requires --annotation_json and --bird_id")
        filter_by_bird(args.annotation_json, args.spec_dir, args.train_dir, args.bird_id)
        
    elif args.mode == 'sample':
        if not args.train_dir:
             parser.error("--mode sample requires --train_dir (as output)")
        if not args.n_samples:
             parser.error("--mode sample requires --n_samples")
        sample_files(args.spec_dir, args.train_dir, args.n_samples)
