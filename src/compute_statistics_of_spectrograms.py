import argparse
import json
import math
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm


def compute_statistics(spec_dir, sample_fraction=0.1, seed=0):
    """Compute mean and std of spectrogram files."""
    rng = random.Random(seed)
    
    # Stream through files without loading all paths into memory
    all_files = list(spec_dir.glob("*.npy"))
    if not all_files:
        raise ValueError(f"No .npy files found in {spec_dir}")
    
    files = [path for path in all_files if rng.random() < sample_fraction]
    if not files:
        files = all_files
    
    print(f"Processing {len(files)} files...")
    
    # Compute statistics
    total_sum = 0.0
    total_sq_sum = 0.0
    total_values = 0
    
    for path in tqdm(files, desc="Computing statistics"):
        # Load tensor
        tensor = np.load(path).astype(np.float32, copy=False)
        
        # Update statistics
        total_sum += tensor.sum().item()
        total_sq_sum += np.square(tensor, out=None).sum().item()
        total_values += tensor.size
    
    # Calculate mean and std
    mean = total_sum / total_values
    variance = (total_sq_sum / total_values) - (mean * mean)
    std = math.sqrt(max(variance, 0))
    
    return mean, std, len(files)


def main():
    parser = argparse.ArgumentParser(description="Compute mean/std for spectrogram .pt files")
    parser.add_argument("--spec_dir", type=Path, help="Directory with .pt files")
    parser.add_argument("--sample_fraction", type=float, default=0.1, help="Fraction of files to sample (default: 0.1)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    args = parser.parse_args()
    spec_dir = args.spec_dir.expanduser()
    
    # Compute statistics
    mean, std, num_files = compute_statistics(
        spec_dir, 
        args.sample_fraction,
        args.seed
    )
    
    print(f"Processed: {num_files} files")
    print(f"Mean: {mean:.6f}")
    print(f"Std: {std:.6f}")
    
    # Always use audio_params.json in the spec directory
    output_path = spec_dir / "audio_params.json"
    
    # Load existing data if present
    metadata = {}
    if output_path.exists():
        with open(output_path) as f:
            metadata = json.load(f)
    
    # Update with new values
    metadata["mean"] = float(mean)
    metadata["std"] = float(std)
    
    # Save
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
