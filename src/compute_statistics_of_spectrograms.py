###############
## THIS FILE NEEDS TO BE REWRITTEN TO BE MUCH SIMPLER
###############
import argparse
import json
import math
import random
import sys
from pathlib import Path
import torch

def iter_pt_files(spec_dir: Path):
    for path in spec_dir.glob("*.pt"):
        if path.is_file():
            yield path


def reservoir_sample(paths, count, rng):
    # Reservoir sampling keeps a uniform subset without storing all candidates.
    sample = []
    for idx, path in enumerate(paths):
        if idx < count:
            sample.append(path)
        else:
            replace_idx = rng.randint(0, idx)
            if replace_idx < count:
                sample[replace_idx] = path
    return sample


def load_spectrogram(path: Path) -> torch.Tensor:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, torch.Tensor):
        tensor = loaded.detach()
    elif isinstance(loaded, dict):
        if "s" not in loaded:
            raise KeyError("missing 's' entry")
        tensor = loaded["s"]
    else:
        raise TypeError(f"Unsupported data format in {path}")
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    return tensor.detach().to(dtype=torch.float64, device="cpu")


def compute_statistics(paths, max_files=None):
    total_sum = 0.0
    total_sq_sum = 0.0
    total_values = 0
    processed = 0

    for path in paths:
        try:
            tensor = load_spectrogram(path)
        except Exception as exc:
            print(f"Skipping {path}: {exc}", file=sys.stderr)
            continue

        total_sum += tensor.sum().item()
        total_sq_sum += tensor.square().sum().item()
        total_values += tensor.numel()
        processed += 1

        if max_files is not None and processed >= max_files:
            break

    if processed == 0 or total_values == 0:
        return processed, total_values, math.nan, math.nan

    mean = total_sum / total_values
    variance = max((total_sq_sum / total_values) - (mean * mean), 0.0)
    std = math.sqrt(variance)
    return processed, total_values, mean, std


def write_metadata(path: Path, mean: float, std: float) -> bool:
    try:
        existing = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                existing = json.load(fh) or {}
            if not isinstance(existing, dict):
                print(
                    f"Warning: {path} contained non-object JSON, replacing with new object.",
                    file=sys.stderr,
                )
                existing = {}
    except Exception as exc:
        print(f"Unable to read {path}: {exc}", file=sys.stderr)
        existing = {}

    existing["mean"] = float(mean)
    existing["std"] = float(std)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2, sort_keys=True)
            fh.write("\n")
        return True
    except Exception as exc:
        print(f"Failed to write {path}: {exc}", file=sys.stderr)
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Compute mean and standard deviation for spectrogram tensors stored as .pt files.")
    parser.add_argument("spec_dir", type=Path, help="Directory containing spectrogram .pt files.")
    parser.add_argument("--sample-fraction", type=float, default=None, help="Process approximately this fraction of files (0 < fraction <= 1).")
    parser.add_argument("--sample-count", type=int, default=None, help="Process this many files sampled uniformly at random.")
    parser.add_argument("--max-files", type=int, default=None, help="Hard limit on the number of files to load after sampling.")
    parser.add_argument("--metadata-path", type=Path, default=None, help="JSON file to update with computed mean/std (default: spec_dir/audio_processing.json).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for sampling.")
    return parser.parse_args()


def main():
    args = parse_args()
    spec_dir = args.spec_dir.expanduser()

    rng = random.Random(args.seed)

    if args.sample_count is not None:
        selected = reservoir_sample(iter_pt_files(spec_dir), args.sample_count, rng)
        if not selected:
            print(f"No spectrogram files found in {spec_dir}.", file=sys.stderr)
            return 1
        if len(selected) < args.sample_count:
            print(
                f"Found only {len(selected)} files (requested {args.sample_count}).",
                file=sys.stderr,
            )
        print(f"Computing statistics on {len(selected)} sampled files.")
        paths = selected
    else:
        if args.sample_fraction is not None:
            print(f"Computing statistics with sampling fraction {args.sample_fraction}.")
        else:
            print("Computing statistics on all spectrogram files.")

        def filtered_paths():
            for path in iter_pt_files(spec_dir):
                if args.sample_fraction is not None and rng.random() > args.sample_fraction:
                    continue
                yield path

        paths = filtered_paths()

    processed, total_values, mean, std = compute_statistics(paths, args.max_files)

    if processed == 0:
        print("No spectrograms were processed. Check your sampling settings.", file=sys.stderr)
        return 1

    print(f"Processed {processed} files ({total_values} total values).")
    print(f"Mean: {mean:.6f}")
    print(f"Std: {std:.6f}")

    metadata_path = (
        args.metadata_path.expanduser() if args.metadata_path else spec_dir / "audio_processing.json"
    )
    if write_metadata(metadata_path, mean, std):
        print(f"Wrote mean/std to {metadata_path}.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

