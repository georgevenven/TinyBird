import json
import argparse
import os
import shutil
import glob
from collections import defaultdict
from tqdm import tqdm

def calculate_ms(detected_events):
    """Calculate total milliseconds in detected_events"""
    total_ms = 0
    for event in detected_events:
        total_ms += event['offset_ms'] - event['onset_ms']
    return total_ms

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
    
    # Allocate birds to train/test
    total_ms = sum(b[1]['total_ms'] for b in birds)
    target_train_ms = total_ms * (train_percent / 100)
    
    train_recordings = []
    test_recordings = []
    train_ms = 0
    
    for bird_id, bird_info in birds:
        if train_ms < target_train_ms:
            train_recordings.extend(bird_info['recordings'])
            train_ms += bird_info['total_ms']
        else:
            test_recordings.extend(bird_info['recordings'])
    
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
    
    # Print stats
    test_ms = total_ms - train_ms
    print(f"Total MS: {total_ms:.2f}")
    print(f"Train MS: {train_ms:.2f} ({train_ms/total_ms*100:.1f}%)")
    print(f"Test MS: {test_ms:.2f} ({test_ms/total_ms*100:.1f}%)")
    print(f"Train recordings: {len(train_recordings)}")
    print(f"Test recordings: {len(test_recordings)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_dir', required=True)
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--annotation_json', required=True)
    parser.add_argument('--train_percent', type=float, default=80)
    args = parser.parse_args()
    
    split_data(args.annotation_json, args.spec_dir, args.train_dir, args.test_dir, args.train_percent)

