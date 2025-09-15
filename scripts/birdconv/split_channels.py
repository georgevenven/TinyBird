#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

import soundfile as sf
import numpy as np

def parse_participants_and_suffix(basename: str) -> Tuple[str, List[str], str]:
    """
    Given '1740770732_USA5494_USA5483.2354.0_300', return:
      timestamp='1740770732'
      participants=['USA5494','USA5483']
      suffix='.2354.0_300'   (everything from the first '.' in the basename, or '' if none)
    """
    # Separate extension already removed before calling this function
    # Split on underscores to find tokens
    tokens = basename.split('_')
    if not tokens:
        return basename, [], ""

    timestamp = tokens[0]

    # Find suffix as "from the first '.' in the entire basename"
    first_dot = basename.find('.')
    suffix = basename[first_dot:] if first_dot != -1 else ""

    # Participants are the underscore-separated tokens after the timestamp,
    # but we stop at the token where the first '.' occurs (that token may contain 'NAME.rest')
    participants = []
    # Reconstruct positions to know when we reach/overlap the first dot
    consumed = len(tokens[0])  # length of timestamp
    # tokens are joined by underscores; track cumulative position in basename
    for idx, tok in enumerate(tokens[1:], start=1):
        # If we've already reached/overlapped first_dot, stop parsing participants
        # Account for an underscore before each token (except the first, handled above)
        start_pos = consumed + 1  # underscore
        end_pos = start_pos + len(tok)
        # Decide if this token intersects or follows the earliest dot
        overlaps_dot = (first_dot != -1) and (start_pos <= first_dot <= end_pos)
        # A participant name is the portion before any dot in this token
        name = tok.split('.', 1)[0]
        if name and any(ch.isalpha() for ch in name):
            participants.append(name)
        # If this token contains the first dot position, we stop after recording its clean name
        if overlaps_dot:
            break
        consumed = end_pos

    return timestamp, participants, suffix

def split_wav_to_mono(src_wav: Path, dst_dir: Path) -> Optional[str]:
    """
    Read a multi-channel WAV and write per-channel mono WAVs using the naming rule.
    Returns error string on failure, or None on success.
    """
    try:
        info = sf.info(str(src_wav))
        # Always read as (n_frames, n_channels)
        data, sr = sf.read(str(src_wav), always_2d=True)
        n_ch = data.shape[1]

        base = src_wav.stem  # without extension
        timestamp, participants, suffix = parse_participants_and_suffix(base)

        # If no participants parsed, fall back to generic CH0..CHn
        if not participants:
            participants = [f"CH{i}" for i in range(n_ch)]

        # If more participants than channels (or fewer), align to the min
        n_out = min(len(participants), n_ch)
        if n_out == 0:
            return None  # nothing to do

        dst_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(n_out):
            name = f"{timestamp}_{participants[idx]}{suffix}.wav"
            out_path = dst_dir / name
            # Write mono channel (copy to ensure contiguous array)
            mono = np.ascontiguousarray(data[:, idx])
            sf.write(str(out_path), mono, sr, subtype=info.subtype)
        return None
    except Exception as e:
        return f"{src_wav}: {e}"

def find_wavs(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.wav") if p.is_file()]

def main():
    ap = argparse.ArgumentParser(
        description="Recurse src dir; for each multi-participant WAV named like "
                    "'<ts>_USA5494_USA5483.<rest>.wav', write mono files "
                    "to dst as '<ts>_USA5494<suffix>.wav', '<ts>_USA5483<suffix>.wav', ..."
    )
    ap.add_argument("--src", required=True, type=Path, help="Source directory (recurse)")
    ap.add_argument("--dst", required=True, type=Path, help="Output directory for mono WAVs")
    ap.add_argument("--dry_run", action="store_true", help="List actions without writing files")
    args = ap.parse_args()

    wavs = find_wavs(args.src)
    if not wavs:
        print("No .wav files found.")
        return

    errors = 0
    for wav in wavs:
        if args.dry_run:
            base = wav.stem
            timestamp, participants, suffix = parse_participants_and_suffix(base)
            try:
                info = sf.info(str(wav))
                n_ch = info.channels
            except Exception:
                n_ch = "?"
            n_out = min(len(participants) or n_ch, n_ch if isinstance(n_ch, int) else len(participants) or 0)
            print(f"[DRY] {wav}  ->  {n_out} mono files under {args.dst}")
        else:
            err = split_wav_to_mono(wav, args.dst)
            if err:
                errors += 1
                print(err)

    if not args.dry_run:
        if errors:
            print(f"Done with {errors} file(s) failing.")
        else:
            print("Done.")

if __name__ == "__main__":
    main()