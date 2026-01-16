#!/usr/bin/env python3
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.plotting_utils import plot_theoretical_resolution_limit


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot theoretical resolution limit results from CSV")
    parser.add_argument("--results_csv", type=str, required=True, help="Path to results CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    args = parser.parse_args()

    plot_theoretical_resolution_limit(args.results_csv, args.output_dir)


if __name__ == "__main__":
    main()

