#!/usr/bin/env python3
"""
additional_data.py

Provides both:
  1) A CLI that windows a conversation CSV given an audio filename, and
  2) An importable API returning an object you can query from other programs.

Audio filename format:
  <timestamp>_<bird0>_<bird1>.<starttime>_<length>.wav
Example:
  1725649219_USA5494_USA5499.1247.0_300.wav

CSV expected at (by default):
  /mnt/birdconv/data_csv/<timestamp>-<bird0>-<bird1>.csv

Windowing rule:
  starttime <= timestamp <= starttime + length (seconds)
"""

import argparse
import sys
import re
from pathlib import Path
import pandas as pd

from dataclasses import dataclass
from typing import Optional

# Default location of CSVs unless overridden by caller
DEFAULT_CSV_DIR = Path("/mnt/birdconv/data_csv")


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


@dataclass(frozen=True)
class AudioSpec:
    ts: str
    bird0: str
    bird1: str
    start: float
    length: float

    @property
    def t0(self) -> float:
        return self.start

    @property
    def t1(self) -> float:
        return self.start + self.length


class AdditionalData:
    """
    Wraps a conversation CSV as a pandas DataFrame and provides
    convenient query methods tied to an audio filename spec.

    Required columns:
      - 'timestamp' (seconds, numeric)
    """

    def __init__(self, df: pd.DataFrame, csv_path: Path, spec: AudioSpec):
        self._df = df.copy()
        self._csv_path = Path(csv_path)
        self._spec = spec

        if "timestamp" not in self._df.columns:
            raise KeyError("CSV is missing required 'timestamp' column.")
        self._df["timestamp"] = pd.to_numeric(self._df["timestamp"], errors="coerce")
        self._df = self._df.dropna(subset=["timestamp"])  # keep only rows with numeric timestamps

    # ---------- Public properties ----------
    @property
    def df(self) -> pd.DataFrame:
        """Return a copy of the full DataFrame."""
        return self._df.copy()

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    @property
    def spec(self) -> AudioSpec:
        return self._spec

    # ---------- Core queries ----------
    def window(self) -> pd.DataFrame:
        """Rows where 0 <= timestamp <= length (timestamps are shifted to start)."""
        return self.between(0.0, self._spec.length, inclusive="both")
    def row_for_time_and_side(self, t: float, side: int) -> Optional[pd.Series]:
        """
        Query a single row using a float time in seconds and a side indicator.
        side: 0 -> left/first bird in filename (bird0), 1 -> right/last bird (bird1)
        The float time is converted to an integer number of seconds.

        Returns a pandas Series for the first matching row, or None if not found.
        """
        if side not in (0, 1):
            raise ValueError("side must be 0 (left/bird0) or 1 (right/bird1)")

        if "timestamp" not in self._df.columns:
            return None

        # Convert to integer seconds (floor by Python's int casting)
        sec = int(t)
        target_bird = self._spec.bird0 if side == 0 else self._spec.bird1

        mask = (self._df["timestamp"].astype(int) == sec)
        if "bird" in self._df.columns:
            mask = mask & (self._df["bird"] == target_bird)

        matches = self._df.loc[mask]
        if matches.empty:
            return None
        # Return the first matching row as a copy
        return matches.iloc[0].copy()

    def between(self, t0: float, t1: float, inclusive: str = "both") -> pd.DataFrame:
        """
        Return rows with timestamps between t0 and t1.
        inclusive: 'both' | 'left' | 'right' | 'neither'
        """
        mask = self._df["timestamp"].between(t0, t1, inclusive=inclusive)
        return self._df.loc[mask].copy()

    def at(self, t: float, tol: float = 0.0) -> pd.DataFrame:
        """
        Return rows exactly at time t (or within +/- tol if tol > 0).
        """
        if tol <= 0:
            return self._df.loc[self._df["timestamp"] == t].copy()
        lo, hi = t - tol, t + tol
        return self.between(lo, hi, inclusive="both")

    def nearest(self, t: float) -> Optional[pd.Series]:
        """
        Return the single row nearest to time t (tie breaks by first occurrence).
        Returns a Series, or None if the DataFrame is empty.
        """
        if self._df.empty:
            return None
        idx = (self._df["timestamp"] - t).abs().idxmin()
        return self._df.loc[idx].copy()  # type: ignore[return-value]


# ---------- Public module API ----------

def load_additional_data(audio_filename: str, csv_dir: Path | str = DEFAULT_CSV_DIR) -> AdditionalData:
    """
    Parse an audio filename, load the corresponding CSV, and return
    an AdditionalData object for convenient querying.

    Example:
        from additional_data import load_additional_data
        data = load_additional_data("1725649219_USA5494_USA5499.1247.0_300.wav")
        window_df = data.window()
    """
    ts, bird0, bird1, start, length = parse_audio_filename(audio_filename)
    spec = AudioSpec(ts=ts, bird0=bird0, bird1=bird1, start=start, length=length)

    csv_dir = Path(csv_dir)
    csv_path = build_csv_path(csv_dir, ts, bird0, bird1)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Window to [start, start+length] and shift timestamps to be relative to start
    df_win = window_dataframe(df, start, length).copy()
    df_win["timestamp"] = df_win["timestamp"] - start
    return AdditionalData(df=df_win, csv_path=csv_path, spec=spec)


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
        default=str(DEFAULT_CSV_DIR),
        help=f"Directory containing timestamp-bird0-bird1.csv files (default: {DEFAULT_CSV_DIR})",
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