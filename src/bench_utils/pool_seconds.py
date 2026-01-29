import argparse
import json
from pathlib import Path

import numpy as np


def load_audio_params(spec_dir: Path):
    audio_path = spec_dir / "audio_params.json"
    if not audio_path.exists():
        raise SystemExit(f"audio_params.json not found in {spec_dir}")
    with open(audio_path, "r") as f:
        audio = json.load(f)
    for key in ("sr", "hop_size"):
        if key not in audio:
            raise SystemExit(f"Missing {key} in {audio_path}")
    return int(audio["sr"]), int(audio["hop_size"])


def pool_seconds(spec_dir: Path) -> float:
    sr, hop_size = load_audio_params(spec_dir)
    total_bins = 0
    for path in spec_dir.glob("*.npy"):
        arr = np.load(path, mmap_mode="r")
        total_bins += int(arr.shape[1])
    return total_bins * hop_size / sr


def main():
    parser = argparse.ArgumentParser(description="Compute total seconds in a spec directory.")
    parser.add_argument("--spec_dir", required=True, help="Directory containing .npy files.")
    args = parser.parse_args()

    spec_dir = Path(args.spec_dir)
    total = pool_seconds(spec_dir)
    print(f"{total:.6f}")


if __name__ == "__main__":
    main()
