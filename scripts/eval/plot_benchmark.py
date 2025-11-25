#!/usr/bin/env python3
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.plotting_utils import plot_benchmark_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot benchmark results from CSV")
    
    parser.add_argument("--results_csv", type=str, required=True, help="Path to results CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    
    args = parser.parse_args()
    
    plot_benchmark_results(args.results_csv, args.output_dir)

