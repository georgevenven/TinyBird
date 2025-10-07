import os
import json
import glob
import torch
from pathlib import Path
from model import TinyBird

def count_parameters(model):
    """
    Count and print the number of parameters in the encoder, decoder, and total model.
    
    Args:
        model (TinyBird): The TinyBird model instance
    
    Returns:
        dict: Dictionary containing parameter counts for encoder, decoder, and total
    """
    # Count encoder parameters
    encoder_params = 0
    encoder_components = [
        model.patch_projection,
        model.encoder,
        model.pos_enc
    ]
    
    for component in encoder_components:
        if hasattr(component, 'parameters'):
            encoder_params += sum(p.numel() for p in component.parameters())
        else:
            encoder_params += component.numel()  # For nn.Parameter
    
    # Count decoder parameters  
    decoder_params = 0
    decoder_components = [
        model.decoder,
        model.encoder_to_decoder,
        model.decoder_to_pixel,
        model.mask_token
    ]
    
    for component in decoder_components:
        if hasattr(component, 'parameters'):
            decoder_params += sum(p.numel() for p in component.parameters())
        else:
            decoder_params += component.numel()  # For nn.Parameter
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Print the results
    print("=" * 60)
    print("MODEL PARAMETER COUNT")
    print("=" * 60)
    print(f"Encoder parameters:    {encoder_params:,}")
    print(f"Decoder parameters:    {decoder_params:,}")
    print(f"Total parameters:      {total_params:,}")
    print("=" * 60)
    
    return {
        'encoder': encoder_params,
        'decoder': decoder_params, 
        'total': total_params
    }

def load_model_from_checkpoint(run_dir="", checkpoint_file=None, fallback_to_random=True):
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
        fallback_to_random (bool, optional): If True, initialize with random weights
            when checkpoint is not found instead of raising an error. Default: False
    
    Returns:
        TinyBird: Loaded model with weights from the specified checkpoint, or random weights if fallback enabled
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
        if fallback_to_random:
            raise ValueError(f"Cannot fallback to random weights: run directory not found and config unavailable: {run_path}")
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    
    # Load config.json
    config_path = os.path.join(run_path, "config.json")
    if not os.path.exists(config_path):
        if fallback_to_random:
            raise ValueError(f"Cannot fallback to random weights: config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model with config
    tinybird = TinyBird(config)
    
    # Determine which checkpoint to load
    weights_dir = os.path.join(run_path, "weights")
    if not os.path.exists(weights_dir):
        if fallback_to_random:
            print(f"Weights directory not found: {weights_dir}")
            print("Initializing model with random weights")
            return tinybird, config
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")
    
    checkpoint_path = None
    if checkpoint_file is not None:
        # Manual checkpoint specified
        if os.path.isabs(checkpoint_file):
            # Full path provided
            checkpoint_path = checkpoint_file
        else:
            # Just filename provided, combine with weights directory
            checkpoint_path = os.path.join(weights_dir, checkpoint_file)
        
        if not os.path.exists(checkpoint_path):
            if fallback_to_random:
                print(f"Specified checkpoint file not found: {checkpoint_path}")
                print("Initializing model with random weights")
                return tinybird, config
            raise FileNotFoundError(f"Specified checkpoint file not found: {checkpoint_path}")
        
        print(f"Loading specified checkpoint: {checkpoint_path}")
    else:
        # Find the latest checkpoint automatically
        checkpoint_pattern = os.path.join(weights_dir, "model_step_*.pth")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            if fallback_to_random:
                print(f"No checkpoint files found in: {weights_dir}")
                print("Initializing model with random weights")
                return tinybird, config
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

def load_audio_params(data_dir):
    """
    Load audio parameters from audio_params.json in the specified directory.
    
    Args:
        data_dir (str): Path to directory containing audio_params.json
    
    Returns:
        dict: Dictionary containing audio parameters (mels, sr, hop_size, fft, mean, std)
    
    Raises:
        SystemExit: If audio_params.json is missing or lacks required keys
    """
    audio_json_path = Path(data_dir) / "audio_params.json"
    
    if not audio_json_path.exists():
        raise SystemExit(f"audio_params.json not found in directory: {data_dir}")
    
    with open(audio_json_path, "r") as f:
        audio_data_json = json.load(f)
    
    # Validate required keys
    required_keys = ["mels", "sr", "hop_size", "fft", "mean", "std"]
    for key in required_keys:
        if key not in audio_data_json:
            raise SystemExit(f"Missing required key '{key}' in audio_params.json. Exiting.")
    
    return audio_data_json