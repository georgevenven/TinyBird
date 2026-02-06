#!/usr/bin/env python3
"""Concatenate all eval_f1.csv files across nested results directories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List


DEFAULT_BASE_DIR = Path("/media/george/DATA/lambda_results/results")
DEFAULT_OUTPUT = Path("scripts/combined_eval_f1.csv")


def find_eval_files(base_dir: Path) -> List[Path]:
    """Return sorted list of all eval_f1.csv files under the base directory."""
    return sorted({p.resolve() for p in base_dir.rglob("eval_f1.csv") if p.is_file()})


def concat_csvs(files: List[Path], output_path: Path) -> int:
    """Concatenate CSV files, keeping the first header. Returns row count."""
    if not files:
        raise FileNotFoundError("No eval_f1.csv files found to concatenate.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    header: List[str] | None = None

    with output_path.open("w", newline="") as out_f:
        writer: csv.DictWriter | None = None
        for csv_path in files:
            with csv_path.open("r", newline="") as in_f:
                reader = csv.DictReader(in_f)
                if reader.fieldnames is None:
                    continue

                if header is None:
                    header = list(reader.fieldnames)
                    writer = csv.DictWriter(out_f, fieldnames=header)
                    writer.writeheader()
                elif list(reader.fieldnames) != header:
                    raise ValueError(
                        f"Header mismatch in {csv_path}. "
                        f"Expected {header}, got {reader.fieldnames}."
                    )

                assert writer is not None  # for type checkers
                for row in reader:
                    writer.writerow(row)
                    total_rows += 1

    return total_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate eval_f1.csv files across nested results directories."
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        help="Optional explicit eval_f1.csv files to concatenate.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help=f"Base directory to search (default: {DEFAULT_BASE_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def resolve_input_files(csv_files: List[Path], base_dir: Path) -> List[Path]:
    if csv_files:
        resolved = sorted({p.resolve() for p in csv_files})
        missing = [p for p in resolved if not p.exists()]
        if missing:
            raise SystemExit(f"Missing input CSV file(s): {', '.join(str(p) for p in missing)}")
        non_files = [p for p in resolved if not p.is_file()]
        if non_files:
            raise SystemExit(f"Input path(s) are not files: {', '.join(str(p) for p in non_files)}")
        return resolved
    return find_eval_files(base_dir)


def main() -> None:
    args = parse_args()
    base_dir: Path = args.base_dir
    output_path: Path = args.output

    files = resolve_input_files(args.csv_files, base_dir)
    if not files:
        if args.csv_files:
            raise SystemExit("No input CSV files provided.")
        raise SystemExit(f"No eval_f1.csv files found under {base_dir}")

    row_count = concat_csvs(files, output_path)
    print(
        f"Concatenated {len(files)} files with {row_count} rows into: {output_path}"
    )


if __name__ == "__main__":
    main()
