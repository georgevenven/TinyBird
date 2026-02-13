#!/usr/bin/env python3
import argparse
import os
import sys

# Add `src/` to path because internal modules use absolute imports.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
sys.path.insert(0, _SRC_ROOT)

from plotting_utils import plot_species_f1_curves


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-species F1/FER curves from eval_f1.csv results."
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Results directory containing eval_f1.csv (or a direct path to eval_f1.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: <results_dir>/plots)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="classify",
        help="Filter by mode (default: classify)",
    )
    parser.add_argument(
        "--probe_mode",
        type=str,
        default=None,
        help="Filter by probe mode (e.g. finetune, linear, lora)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="both",
        choices=["f1", "fer", "both"],
        help="Metric to plot (default: both)",
    )
    parser.add_argument(
        "--species",
        type=str,
        default=None,
        help="Comma-separated list of species to plot (default: all)",
    )
    args = parser.parse_args()

    species_filter = None
    if args.species:
        species_filter = [s.strip() for s in args.species.split(",") if s.strip()]

    plot_species_f1_curves(
        args.results_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        probe_mode=args.probe_mode,
        metric=args.metric,
        species_filter=species_filter,
    )


if __name__ == "__main__":
    main()
