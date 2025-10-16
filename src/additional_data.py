#!/usr/bin/env python3
"""
window_birdconv_csv.py

Given an audio filename of the form:
  <timestamp>_<bird0>_<bird1>.<starttime>_<length>.wav
e.g. 1725649219_USA5494_USA5499.1247.0_300.wav

1) Parse timestamp, bird0, bird1, starttime (s), length (s).
2) Load CSV from <csv_dir>/<timestamp>-<bird0>-<bird1>.csv
   Default csv_dir = /mnt/birdconv/data_csv
3) Window rows where starttime <= timestamp <= starttime+length.
4) Print windowed rows to stdout (CSV), or save via --out-csv.
"""

import argparse
import sys
import re
from pathlib import Path
import pandas as pd


FILENAME_RE = re.compile(
    r"""
    ^(?P<ts>\d+)_                               # timestamp (epoch-like)
    (?P<bird0>[^_]+)_                           # bird0
    (?P<bird1>[^.]+)\.                          # bird1 (up to dot)
    (?P<start>[0-9]+(?:\.[0-9]+)?)_             # starttime (float)
    (?P<len>[0-9]+(?:\.[0-9]+)?)                # length (float)
    \.wav$                                      # extension
    """,
    re.VERBOSE | re.IGNORECASE,
)


def parse_audio_filename(name: str):
    """Extract fields from the audio filename."""
    m = FILENAME_RE.match(Path(name).name)
    if not m:
        raise ValueError(
            f"Filename does not match expected pattern: {name}\n"
            "Expected: <ts>_<bird0>_<bird1>.<start>_<len>.wav"
        )
    ts = m.group("ts")
    bird0 = m.group("bird0")
    bird1 = m.group("bird1")
    start = float(m.group("start"))
    length = float(m.group("len"))
    return ts, bird0, bird1, start, length


def build_csv_path(csv_dir: Path, ts: str, bird0: str, bird1: str) -> Path:
    """timestamp-bird0-bird1.csv in the given directory."""
    return csv_dir / f"{ts}-{bird0}-{bird1}.csv"


def window_dataframe(df: pd.DataFrame, start: float, length: float) -> pd.DataFrame:
    """Filter rows with start <= timestamp <= start+length (seconds)."""
    if "timestamp" not in df.columns:
        raise KeyError("CSV is missing required 'timestamp' column.")
    # Ensure numeric timestamps
    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    t0 = start
    t1 = start + length
    return df[(df["timestamp"] >= t0) & (df["timestamp"] <= t1)]


def main():
    ap = argparse.ArgumentParser(
        description="Window a bird conversation CSV based on an audio filename."
    )
    ap.add_argument(
        "audio_filename",
        help="Filename like 1725649219_USA5494_USA5499.1247.0_300.wav",
    )
    ap.add_argument(
        "--csv-dir",
        default="/mnt/birdconv/data_csv",
        help="Directory containing timestamp-bird0-bird1.csv files "
             "(default: /mnt/birdconv/data_csv)",
    )
    ap.add_argument(
        "--out-csv",
        default=None,
        help="Optional path to save the windowed CSV. If omitted, prints to stdout.",
    )
    ap.add_argument(
        "--no-header",
        action="store_true",
        help="When printing to stdout, omit the CSV header.",
    )
    args = ap.parse_args()

    ts, bird0, bird1, start, length = parse_audio_filename(args.audio_filename)
    csv_path = build_csv_path(Path(args.csv_dir), ts, bird0, bird1)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    win = window_dataframe(df, start, length)

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        win.to_csv(out_path, index=False)
        print(f"Saved windowed CSV: {out_path} (rows: {len(win)})")
    else:
        # Print to stdout as CSV
        win.to_csv(sys.stdout, index=False, header=not args.no_header)


if __name__ == "__main__":
    main()