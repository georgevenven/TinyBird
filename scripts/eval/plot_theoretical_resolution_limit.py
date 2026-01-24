#!/usr/bin/env python3
import argparse
import os
import sys

# Add `src/` to path because internal modules use absolute imports.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
sys.path.insert(0, _SRC_ROOT)

from plotting_utils import plot_theoretical_resolution_limit


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot theoretical resolution limit curves.")
    parser.add_argument("--results_csv", type=str, required=True, help="Path to results CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for plots")
    args = parser.parse_args()

    plot_theoretical_resolution_limit(args.results_csv, args.output_dir)


if __name__ == "__main__":
    main()
