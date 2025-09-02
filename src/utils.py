import os
import json
import glob
import torch
from model import TinyBird

def load_model_from_checkpoint(run_dir="", checkpoint_file=None):
    """
    Load a TinyBird model from a checkpoint directory.
    
    Args:
        run_dir (str): Either:
            - Absolute path to the run directory
            - Relative path to a folder in runs/ (e.g., "my_run_name")
            - Empty string (will raise error)
        checkpoint_file (str, optional): Specific checkpoint filename to load.
            If None, loads the latest checkpoint. Can be either:
            - Just the filename (e.g., "model_step_005000.pth")
            - Full path to the checkpoint file
    
    Returns:
        TinyBird: Loaded model with weights from the specified checkpoint
    """
    if not run_dir:
        raise ValueError("run_dir cannot be empty. Provide either absolute path or relative path to runs/")
    
    # Handle path - check if it's absolute or relative
    if os.path.isabs(run_dir):
        # Absolute path provided
        run_path = run_dir
    else:
        # Relative path - assume it's a folder name in runs/
        runs_base = os.path.join("..", "runs")
        run_path = os.path.join(runs_base, run_dir)
    
    # Check if run directory exists
    if not os.path.exists(run_path):
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    
    # Load config.json
    config_path = os.path.join(run_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model with config
    tinybird = TinyBird(config)
    
    # Determine which checkpoint to load
    weights_dir = os.path.join(run_path, "weights")
    if not os.path.exists(weights_dir):
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")
    
    if checkpoint_file is not None:
        # Manual checkpoint specified
        if os.path.isabs(checkpoint_file):
            # Full path provided
            checkpoint_path = checkpoint_file
        else:
            # Just filename provided, combine with weights directory
            checkpoint_path = os.path.join(weights_dir, checkpoint_file)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Specified checkpoint file not found: {checkpoint_path}")
        
        print(f"Loading specified checkpoint: {checkpoint_path}")
    else:
        # Find the latest checkpoint automatically
        checkpoint_pattern = os.path.join(weights_dir, "model_step_*.pth")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in: {weights_dir}")
        
        # Find the latest checkpoint (highest step number)
        checkpoint_path = max(checkpoint_files, key=lambda x: int(x.split('_step_')[1].split('.pth')[0]))
        print(f"Loading latest checkpoint: {checkpoint_path}")
    
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    tinybird.load_state_dict(state_dict)
    
    # Extract step number for info
    try:
        step_num = int(checkpoint_path.split('_step_')[1].split('.pth')[0])
        print(f"Model loaded from step {step_num}")
    except (IndexError, ValueError):
        print(f"Model loaded from: {os.path.basename(checkpoint_path)}")
    
    return tinybird, config