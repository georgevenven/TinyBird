import argparse
import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Add src to path
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
sys.path.insert(0, SRC_DIR)

from data_loader import SupervisedSpectogramDataset
from supervised_train import SupervisedTinyBird
from utils import load_model_from_checkpoint
from torch.utils.data import DataLoader

def sliding_window_max_vote(predictions, window_size=100):
    """Apply max vote over sliding window to enforce minimum song length"""
    N = len(predictions)
    smoothed = np.zeros_like(predictions)
    
    for i in range(N):
        start = max(0, i - window_size // 2)
        end = min(N, i + window_size // 2)
        window = predictions[start:end]
        smoothed[i] = 1 if np.sum(window) > window_size // 2 else 0
    
    return smoothed

def calculate_frame_accuracy(predictions, labels):
    """Calculate frame-level accuracy"""
    correct = np.sum(predictions == labels)
    total = len(labels)
    accuracy = 100.0 * correct / total
    return accuracy

def calculate_f1_score(predictions, labels):
    """Calculate F1 score for binary classification"""
    # Positive class is 1 (vocalization detected)
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    # Compute precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Compute F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1 * 100.0  # Convert to percentage

def plot_detection(spec, labels, predictions, smoothed, filename, output_dir, raw_acc, smoothed_acc, raw_f1, smoothed_f1):
    """Visualize spectrogram with ground truth and predictions"""
    binary_cmap = ListedColormap(["#1f2933", "#38bdf8"])
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(4, 1, height_ratios=[6, 1, 1, 1], hspace=0.08)
    
    spec_ax = fig.add_subplot(gs[0, 0])
    spec_ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
    spec_ax.set_ylabel('Frequency')
    spec_ax.set_title(f'{filename} | Raw Acc: {raw_acc:.2f}% F1: {raw_f1:.2f}% | Smoothed Acc: {smoothed_acc:.2f}% F1: {smoothed_f1:.2f}%')
    spec_ax.set_xticks([])
    
    strip_data = [
        ("Ground Truth", labels),
        (f'Raw Pred (Acc:{raw_acc:.2f}% F1:{raw_f1:.2f}%)', predictions),
        (f'Smoothed (Acc:{smoothed_acc:.2f}% F1:{smoothed_f1:.2f}%)', smoothed),
    ]
    
    strip_axes = []
    spec_width = spec.shape[1]
    for idx, (label, data) in enumerate(strip_data, start=1):
        ax = fig.add_subplot(gs[idx, 0], sharex=spec_ax)
        strip_axes.append(ax)
        strip = data.reshape(1, -1)
        ax.imshow(
            strip,
            aspect='auto',
            cmap=binary_cmap,
            vmin=0,
            vmax=1,
            interpolation='nearest',
            extent=(0, spec_width, 0, 1),
        )
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, va='center', ha='right', labelpad=45)
        if idx < len(strip_data):
            ax.set_xticks([])
        ax.set_xlim(0, spec_width)
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    strip_axes[-1].set_xlabel('Time (bins)')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{filename}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path

def main(args):
    # Setup output directory
    PROJECT_ROOT = os.path.dirname(SRC_DIR)
    output_dir = os.path.join(PROJECT_ROOT, "results", "detect_debug")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    config_path = os.path.join(args["run_dir"], "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Determine annotation file path
    if args["annotation_file"]:
        annotation_file = args["annotation_file"]
    else:
        # Look for annotation JSON in spec_dir
        spec_dir_path = Path(args["spec_dir"])
        annotation_files = list(spec_dir_path.glob("*_annotations.json"))
        if len(annotation_files) == 0:
            raise ValueError(f"No annotation JSON found in {args['spec_dir']}. Please specify --annotation_file")
        elif len(annotation_files) > 1:
            print(f"Warning: Multiple annotation files found in {args['spec_dir']}: {[f.name for f in annotation_files]}")
            print(f"Using: {annotation_files[0].name}")
        annotation_file = str(annotation_files[0])
    
    print(f"Using annotation file: {annotation_file}")
    
    # Load pretrained encoder config for model architecture
    pretrained_config_path = os.path.join(config["pretrained_run"], "config.json")
    with open(pretrained_config_path, 'r') as f:
        pretrained_config = json.load(f)
    
    # Load pretrained model first
    from pretrain import resolve_run_path
    pretrained_path = resolve_run_path(config["pretrained_run"])
    pretrained_model, _ = load_model_from_checkpoint(pretrained_path, fallback_to_random=False)
    
    # Create supervised model
    model = SupervisedTinyBird(
        pretrained_model=pretrained_model,
        config=config,
        num_classes=config["num_classes"],
        freeze_encoder=config.get("freeze_encoder", True),
        mode=config["mode"]
    )
    
    # Load weights
    checkpoint = args["checkpoint"] if args["checkpoint"] else "latest"
    if checkpoint == "latest":
        weight_files = sorted(Path(os.path.join(args["run_dir"], "weights")).glob("*.pth"))
        if not weight_files:
            raise ValueError("No checkpoint files found")
        checkpoint_path = str(weight_files[-1])
    else:
        checkpoint_path = os.path.join(args["run_dir"], "weights", checkpoint)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Loaded model from: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    
    # Load dataset with full spectrograms (no cropping)
    dataset = SupervisedSpectogramDataset(
        dir=args["spec_dir"],
        annotation_file_path=annotation_file,
        n_timebins=None,  # Load full files
        mode=config["mode"]
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Process samples
    patch_width = config["patch_width"]
    model_num_timebins = config["num_timebins"]
    window_size = args["window_size"]
    
    for i, (spec, labels, filename) in enumerate(loader):
        if i >= args["num_samples"]:
            break
        
        # spec shape: (1, 1, H, W)
        spec_timebins = spec.shape[-1]
        
        # Round to patch width
        remainder = spec_timebins % patch_width
        if remainder != 0:
            spec = spec[:, :, :, :spec_timebins - remainder]
            labels = labels[:, :spec_timebins - remainder]
        
        rounded_spec_length = spec.shape[-1]
        
        # Chunk into context windows (like extract_embedding.py)
        if rounded_spec_length > model_num_timebins:
            batch_size = (rounded_spec_length // model_num_timebins) + 1
            pad_amnt = model_num_timebins - (rounded_spec_length % model_num_timebins)
        elif rounded_spec_length < model_num_timebins:
            pad_amnt = model_num_timebins - rounded_spec_length
            batch_size = 1
        else:
            pad_amnt = 0
            batch_size = 1
        
        if pad_amnt > 0:
            spec = torch.nn.functional.pad(spec, (0, pad_amnt), mode='constant', value=0)
            labels = torch.nn.functional.pad(labels, (0, pad_amnt), mode='constant', value=0)
        
        # Reshape into batches: (1, 1, H, total_time) -> (batch_size, 1, H, model_num_timebins)
        B, C, H, total_time = spec.shape
        spec_batched = spec.reshape(C, H, batch_size, model_num_timebins)
        spec_batched = spec_batched.permute(2, 0, 1, 3)  # (batch_size, 1, H, model_num_timebins)
        
        spec_batched = spec_batched.to(device)
        
        # Process all chunks
        logits_list = []
        with torch.no_grad():
            for batch_idx in range(batch_size):
                chunk = spec_batched[batch_idx:batch_idx+1]  # (1, 1, H, model_num_timebins)
                logits = model(chunk)  # (1, W_patches, num_classes) or (1, W_patches, 1) for binary
                
                # Handle binary vs multi-class classification
                if config["num_classes"] == 2:
                    # Binary classification with BCE: logits shape (1, W_patches, 1)
                    logits_flat = logits.reshape(-1)  # (W_patches,)
                    probs = torch.sigmoid(logits_flat)  # (W_patches,)
                    preds = (probs > 0.5).long()  # (W_patches,)
                else:
                    # Multi-class classification: logits shape (1, W_patches, num_classes)
                    preds = torch.argmax(logits, dim=-1)[0]  # (W_patches,)
                
                logits_list.append(preds)
        
        # Concatenate predictions from all chunks
        preds_all = torch.cat(logits_list, dim=0).cpu().numpy()  # (total_patches,)
        
        # Convert to numpy for visualization
        spec_full = spec[0, 0].cpu().numpy()  # (H, total_time)
        labels_np = labels[0].cpu().numpy()  # (total_time,)
        
        # Remove padding from visualization
        spec_full = spec_full[:, :rounded_spec_length]
        labels_np = labels_np[:rounded_spec_length]
        preds_all = preds_all[:rounded_spec_length // patch_width]
        
        # Upsample predictions to match original resolution
        preds_upsampled = np.repeat(preds_all, patch_width)[:rounded_spec_length]
        
        # Apply sliding window smoothing
        smoothed = sliding_window_max_vote(preds_upsampled, window_size=window_size)
        
        # Split into chunks of model_num_timebins and save separately
        num_chunks = (rounded_spec_length + model_num_timebins - 1) // model_num_timebins
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * model_num_timebins
            end_idx = min(start_idx + model_num_timebins, rounded_spec_length)
            
            # Extract chunk
            spec_chunk = spec_full[:, start_idx:end_idx]
            labels_chunk = labels_np[start_idx:end_idx]
            preds_chunk = preds_upsampled[start_idx:end_idx]
            smoothed_chunk = smoothed[start_idx:end_idx]
            
            # Calculate accuracy and F1 for this chunk
            chunk_raw_acc = calculate_frame_accuracy(preds_chunk, labels_chunk)
            chunk_smoothed_acc = calculate_frame_accuracy(smoothed_chunk, labels_chunk)
            chunk_raw_f1 = calculate_f1_score(preds_chunk, labels_chunk)
            chunk_smoothed_f1 = calculate_f1_score(smoothed_chunk, labels_chunk)
            
            # Create chunk filename
            chunk_filename = f"{filename[0]}_chunk{chunk_idx}"
            
            # Plot and save
            save_path = plot_detection(spec_chunk, labels_chunk, preds_chunk, smoothed_chunk,
                                       chunk_filename, output_dir, chunk_raw_acc, chunk_smoothed_acc,
                                       chunk_raw_f1, chunk_smoothed_f1)
            print(f"Saved: {save_path} | Raw Acc: {chunk_raw_acc:.2f}% F1: {chunk_raw_f1:.2f}% | Smoothed Acc: {chunk_smoothed_acc:.2f}% F1: {chunk_smoothed_f1:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate supervised detection model")
    parser.add_argument("--run_dir", type=str, required=True, help="Supervised training run directory")
    parser.add_argument("--spec_dir", type=str, required=True, help="Directory of specs to evaluate")
    parser.add_argument("--annotation_file", type=str, default=None, help="Path to annotation JSON (default: auto-detect in spec_dir)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint filename (default: latest)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--window_size", type=int, default=100, help="Sliding window size for smoothing")
    
    args = parser.parse_args()
    main(vars(args))
