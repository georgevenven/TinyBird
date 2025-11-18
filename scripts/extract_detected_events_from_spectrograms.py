#!/usr/bin/env python3
"""
Extracts detected event segments from full spectrograms.
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
from tqdm import tqdm

def extract(args):
    src_file, events, dst_dir, sr, hop = args
    try:
        # mmap_mode='r' for efficient slicing without full load
        spec = np.load(src_file, mmap_mode='r')
    except FileNotFoundError:
        return 0

    stem = src_file.stem
    count = 0
    for i, ev in enumerate(events):
        # Convert ms to spectrogram frames
        start = int((ev['onset_ms'] * sr) / (1000 * hop))
        end = int((ev['offset_ms'] * sr) / (1000 * hop))
        
        if start < spec.shape[1] and start < end:
            seg = np.array(spec[:, start:end])
            np.save(dst_dir / f"{stem}_frag_{i}.npy", seg)
            count += 1
    return count

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", type=Path, required=True, help="Source directory with .npy files")
    p.add_argument("--annotations", type=Path, required=True, help="Annotations JSON file")
    p.add_argument("--dst_dir", type=Path, required=True, help="Destination directory")
    p.add_argument("--workers", type=int, default=mp.cpu_count(), help="Number of parallel workers")
    args = p.parse_args()

    args.dst_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.src_dir / "audio_params.json") as f:
        cfg = json.load(f)
    
    with open(args.annotations) as f:
        recs = json.load(f)["recordings"]

    # Build task list
    tasks = []
    for r in recs:
        if r["detected_events"]:
            stem = Path(r["recording"]["filename"]).stem
            tasks.append((args.src_dir / f"{stem}.npy", r["detected_events"], args.dst_dir, cfg["sr"], cfg["hop_size"]))

    print(f"Extracting from {len(tasks)} recordings...")
    with mp.Pool(args.workers) as pool:
        total = sum(tqdm(pool.imap_unordered(extract, tasks), total=len(tasks)))
    print(f"Done. Extracted {total} segments.")

if __name__ == "__main__":
    main()

