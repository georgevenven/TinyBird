#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
from pathlib import Path

def get_checkpoints(run_dir):
    run_path = Path(run_dir)
    weights_dir = run_path / "weights"
    
    # Support both run_dir/weights/*.pth and run_dir/*.pth (though weights/ is standard)
    if weights_dir.exists():
        checkpoints = list(weights_dir.glob("*.pth"))
    else:
        checkpoints = list(run_path.glob("*.pth"))
        
    # Sort by step number
    def extract_step(p):
        match = re.search(r"step_(\d+)", p.name)
        return int(match.group(1)) if match else -1
    
    checkpoints.sort(key=extract_step)
    return checkpoints

def main():
    parser = argparse.ArgumentParser(description="Run eval_embedding on all checkpoints in a run to generate UMAP series.")
    parser.add_argument("--run_dir", required=True, help="Run directory containing weights/")
    parser.add_argument("--spec_dir", required=True, help="Spectrogram directory")
    parser.add_argument("--json_path", required=True, help="Annotation JSON path")
    parser.add_argument("--bird", required=True, help="Bird ID")
    parser.add_argument("--results_dir", required=True, help="Root output directory")
    parser.add_argument("--num_timebins", type=int, default=10_000, help="Number of timebins per evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    
    args = parser.parse_args()
    
    checkpoints = get_checkpoints(args.run_dir)
    if not checkpoints:
        print(f"No checkpoints found in {args.run_dir} (or {args.run_dir}/weights)")
        sys.exit(1)
        
    print(f"Found {len(checkpoints)} checkpoints in {args.run_dir}")
    
    # Resolve path to eval_embedding.py relative to this script
    script_path = Path(__file__).parent / "eval_embedding.py"
    if not script_path.exists():
        print(f"Error: Could not find {script_path}")
        sys.exit(1)
        
    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    
    for ckpt in checkpoints:
        # Extract step name for folder
        step_match = re.search(r"step_(\d+)", ckpt.name)
        step_name = step_match.group(0) if step_match else ckpt.stem
        
        # Create specific output directory for this step
        out_dir = results_root / step_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[eval] Processing {ckpt.name} -> {out_dir}")
        
        cmd = [
            sys.executable,
            str(script_path),
            "--run_dir", args.run_dir,
            "--checkpoint", ckpt.name,
            "--spec_dir", args.spec_dir,
            "--json_path", args.json_path,
            "--bird", args.bird,
            "--results_dir", str(out_dir),
            "--num_timebins", str(args.num_timebins),
            "--deterministic" # Use deterministic UMAP for consistent comparisons
        ]
        
        if args.dry_run:
            print(" ".join(cmd))
        else:
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()

