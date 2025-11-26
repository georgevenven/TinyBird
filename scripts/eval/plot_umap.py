#!/usr/bin/env python3
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.plotting_utils import generate_umap_plots

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate UMAP plots from embedding NPZ")
    
    parser.add_argument("--npz_path", type=str, required=True, help="Path to embeddings NPZ file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    parser.add_argument("--neighbors", type=int, default=100, help="Number of UMAP neighbors (default: 200)")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic UMAP")
    
    args = parser.parse_args()
    
    generate_umap_plots(args.npz_path, args.output_dir, args.neighbors, args.deterministic)

