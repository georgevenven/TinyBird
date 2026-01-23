import argparse
import shutil
from pathlib import Path

import numpy as np

from utils import load_audio_params


def seconds_to_timebins(seconds, sr, hop_size):
    return max(1, int(round(float(seconds) * sr / hop_size)))


def timebins_to_ms(timebins, sr, hop_size):
    return int(round(timebins * hop_size / sr * 1000.0))


def chunk_spectrogram_dir(
    spec_dir,
    out_dir,
    *,
    chunk_timebins=None,
    chunk_seconds=None,
    stride_timebins=None,
    stride_seconds=None,
    pad_remainder=False,
    overwrite=False,
):
    spec_dir = Path(spec_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_params = load_audio_params(spec_dir)
    sr = int(audio_params["sr"])
    hop_size = int(audio_params["hop_size"])

    if chunk_timebins is None:
        if chunk_seconds is None:
            raise ValueError("chunk_timebins or chunk_seconds must be provided")
        chunk_timebins = seconds_to_timebins(chunk_seconds, sr, hop_size)
    else:
        chunk_timebins = int(chunk_timebins)
        if chunk_timebins <= 0:
            raise ValueError("chunk_timebins must be > 0")

    if stride_timebins is None:
        if stride_seconds is not None:
            stride_timebins = seconds_to_timebins(stride_seconds, sr, hop_size)
        else:
            stride_timebins = chunk_timebins
    else:
        stride_timebins = int(stride_timebins)
        if stride_timebins <= 0:
            raise ValueError("stride_timebins must be > 0")

    audio_params_path = spec_dir / "audio_params.json"
    if audio_params_path.exists():
        shutil.copy2(audio_params_path, out_dir / "audio_params.json")

    npy_files = sorted(spec_dir.glob("*.npy"))
    for npy_path in npy_files:
        arr = np.load(npy_path, mmap_mode="r")
        total_timebins = arr.shape[1]

        start = 0
        while start < total_timebins:
            end = start + chunk_timebins
            pad = 0
            actual_end = end
            if end > total_timebins:
                if not pad_remainder:
                    break
                actual_end = total_timebins
                pad = end - total_timebins

            chunk = np.array(arr[:, start:actual_end], dtype=np.float32)
            if pad > 0:
                chunk = np.pad(chunk, ((0, 0), (0, pad)), mode="constant")

            start_ms = timebins_to_ms(start, sr, hop_size)
            end_ms = timebins_to_ms(actual_end, sr, hop_size)
            out_name = f"{npy_path.stem}__ms_{start_ms}_{end_ms}.npy"
            out_path = out_dir / out_name

            if overwrite or not out_path.exists():
                np.save(out_path, chunk)

            start += stride_timebins


def main():
    parser = argparse.ArgumentParser(description="Chunk spectrogram .npy files into fixed-time windows.")
    parser.add_argument("--spec_dir", required=True, help="Input directory with .npy spectrograms")
    parser.add_argument("--out_dir", required=True, help="Output directory for chunked .npy files")
    parser.add_argument("--chunk_seconds", type=float, default=None, help="Chunk size in seconds")
    parser.add_argument("--chunk_timebins", type=int, default=None, help="Chunk size in timebins")
    parser.add_argument("--stride_seconds", type=float, default=None, help="Stride in seconds (defaults to chunk size)")
    parser.add_argument("--stride_timebins", type=int, default=None, help="Stride in timebins (defaults to chunk size)")
    parser.add_argument("--pad_remainder", action="store_true", help="Pad final chunk to full length")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing chunk files")
    args = parser.parse_args()

    if args.chunk_seconds is None and args.chunk_timebins is None:
        parser.error("Must provide --chunk_seconds or --chunk_timebins")

    if args.chunk_seconds is not None and args.chunk_timebins is not None:
        print("Both chunk_seconds and chunk_timebins provided; using chunk_timebins.")

    chunk_spectrogram_dir(
        args.spec_dir,
        args.out_dir,
        chunk_timebins=args.chunk_timebins,
        chunk_seconds=args.chunk_seconds,
        stride_timebins=args.stride_timebins,
        stride_seconds=args.stride_seconds,
        pad_remainder=args.pad_remainder,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
