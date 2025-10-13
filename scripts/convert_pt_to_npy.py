import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def convert(pt_path: Path, dst_dir: Path, overwrite: bool) -> None:
    out_path = dst_dir / (pt_path.stem + ".npy")
    if out_path.exists() and not overwrite:
        return

    data = torch.load(pt_path, map_location="cpu")
    spec = data["s"] if isinstance(data, dict) else data

    if torch.is_tensor(spec):
        spec = spec.detach().cpu().numpy()

    np.save(out_path, spec.astype(np.float32, copy=False))
    if overwrite and dst_dir == pt_path.parent:
        pt_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert spectrogram .pt files to .npy.")
    parser.add_argument("--src_dir", type=Path, help="Directory containing .pt files.")
    parser.add_argument("--dst_dir", type=Path, help="Output directory (default: src_dir).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    src_dir = args.src_dir.expanduser()
    dst_dir = args.dst_dir.expanduser() if args.dst_dir else src_dir
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.pt"))
    for pt_path in tqdm(files, desc="Converting", unit="file"):
        convert(pt_path, dst_dir, args.overwrite)


if __name__ == "__main__":
    main()
