import argparse
from model import TinyBird
from data_loader import SpectogramDataset
from utils import load_model_from_checkpoint
import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.decomposition import PCA

def main(args):
    # Load model and config from checkpoint
    model, config = load_model_from_checkpoint(
        run_dir=args["run_dir"],
        checkpoint_file=args["checkpoint"]
    )
    
    # Create embedding dataset using config parameters
    embedding_dataset = SpectogramDataset(
        dir=args["spec_dir"],
        n_mels=config["mels"],
        n_timebins=config["num_timebins"],
        pad_crop=False
    )
    
    latent_list = []
    pos_ids = []          # absolute time index per patch within its clip (0..W-1)
    rel_bins = []         # optional: relative-position bins (0..K-1)
    K = 64                # change if you want coarser/finer relative bins
    spec_list = []
    total_timebins = 0
    i = 0
    
    while i < len(embedding_dataset) and total_timebins < args["num_timebins"]:
        spec, labels = embedding_dataset[i]
        
        patch_height, patch_width, mels, num_timebins = config["patch_height"], config["patch_width"], config["mels"], config["num_timebins"]
        spec = spec[:,:mels,:num_timebins]
        
        spec_timebins = spec.shape[-1]
        spec_mels = spec.shape[-2]
        
        remainder = spec_timebins % patch_width
        if remainder != 0: 
            spec = spec[:,:,:spec_timebins-remainder]
        spec_timebins = spec.shape[-1]
        spec_mels = spec.shape[-2]
        
        # Store the spectrogram for visualization
        spec_list.append(spec.squeeze(0).detach().cpu())
        
        spec = spec.unsqueeze(0)  # add a batch dimension
        
        # Using original spec dtype
        z = model.forward_encoder_inference(spec)
        B, S, D = z.shape
        
        H = int(spec_mels / config["patch_height"])
        W = int(spec_timebins / config["patch_width"])
        
        z_grid = z.reshape(B, H, W, D)
        
        z_grid = z_grid.permute(0,2,1,3)
        z_freq_stack = z_grid.reshape(B, W, H * D)
        
        # get rid of the batch shape (its 1 anyway)
        z_freq_stack = z_freq_stack.squeeze(0)

        # shape is W, H*D
        # Store raw features for later whitening across full dataset
        latent_list.append(z_freq_stack.detach().cpu())
        # record indices for de-pos-encoding
        pos_ids.extend(range(W))
        if W > 1:
            rel = np.round(np.linspace(0, K-1, num=W)).astype(int)
        else:
            rel = np.array([0], dtype=int)
        rel_bins.extend(rel.tolist())
        
        total_timebins += spec_timebins
        i += 1
    
    Z = torch.cat(latent_list, dim=0)  # (N_rows, H*D)
    pos_ids = np.asarray(pos_ids, dtype=int)
    rel_bins = np.asarray(rel_bins, dtype=int)
    print(f"shape of dim reduction op {Z.shape}")
    
    # --- remove average vector per position (choose one) ---
    mode = "absolute"  # {"absolute", "relative", None}
    Z_np = Z.cpu().numpy()
    if mode == "absolute":
        uniq = np.unique(pos_ids)
        means = np.zeros((uniq.max()+1, Z_np.shape[1]), dtype=np.float32)
        for p in uniq:
            means[p] = Z_np[pos_ids == p].mean(axis=0)
        Z_np = Z_np - means[pos_ids]
    elif mode == "relative":
        means = np.zeros((K, Z_np.shape[1]), dtype=np.float32)
        for b in np.unique(rel_bins):
            means[b] = Z_np[rel_bins == b].mean(axis=0)
        Z_np = Z_np - means[rel_bins]
    Z = torch.from_numpy(Z_np)
    # Apply whitening + L2 normalization (new style)
    Z_whitened = (Z_np - Z_np.mean(0)) / (Z_np.std(0) + 1e-6)
    Z_whitened /= np.linalg.norm(Z_whitened, axis=1, keepdims=True) + 1e-9
    print(f"shape after whitening {Z_whitened.shape}")
    
    # Apply UMAP to the whitened data
    reducer_enc = umap.UMAP(n_components=2, n_neighbors=50, metric='cosine')
    emb_enc = reducer_enc.fit_transform(Z_whitened)
    print("umap done")
    
    # Concatenate all spectrograms
    concat_spec = torch.cat(spec_list, dim=-1)  # Concatenate along time dimension
    concat_spec_np = concat_spec.numpy()
    
    # Create color mapping based on UMAP embedding
    # Map UMAP coordinates to colors using a 2D colormap
    norm_x = Normalize(vmin=emb_enc[:, 0].min(), vmax=emb_enc[:, 0].max())
    norm_y = Normalize(vmin=emb_enc[:, 1].min(), vmax=emb_enc[:, 1].max())
    # Create colors based on UMAP position (using hue for x, lightness for y)
    from matplotlib.colors import hsv_to_rgb
    colors = np.zeros((len(emb_enc), 3))
    for idx in range(len(emb_enc)):
        # Map UMAP coordinates to HSV color space
        hue = norm_x(emb_enc[idx, 0])  # x-coordinate maps to hue
        saturation = 0.8  # Keep saturation high
        value = 0.3 + 0.7 * norm_y(emb_enc[idx, 1])  # y-coordinate maps to brightness
        colors[idx] = hsv_to_rgb([hue, saturation, value])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: UMAP Embedding
    ax1 = plt.subplot(2, 1, 1)
    scatter = ax1.scatter(emb_enc[:, 0], emb_enc[:, 1], c=colors, alpha=0.6, s=20)
    ax1.set_xlabel('UMAP 1', fontsize=12)
    ax1.set_ylabel('UMAP 2', fontsize=12)
    ax1.set_title('UMAP Embedding of Spectrogram Patches', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Concatenated Spectrogram with color-coded timebins
    ax2 = plt.subplot(2, 1, 2)
    
    # Display the spectrogram
    im = ax2.imshow(concat_spec_np, aspect='auto', origin='lower', cmap='viridis', alpha=0.7)
    
    # Overlay color bars for each timebin based on UMAP position
    patch_width = config["patch_width"]
    num_patches = len(emb_enc)
    
    # Create overlay showing UMAP-based colors for each patch
    overlay = np.zeros((concat_spec_np.shape[0], num_patches * patch_width, 4))
    
    for patch_idx in range(num_patches):
        start_idx = patch_idx * patch_width
        end_idx = start_idx + patch_width
        
        # Create a colored bar at the top of the spectrogram for each patch
        bar_height = int(concat_spec_np.shape[0] * 0.05)  # 5% of spectrogram height
        overlay[-bar_height:, start_idx:end_idx, :3] = colors[patch_idx]
        overlay[-bar_height:, start_idx:end_idx, 3] = 0.8  # Alpha channel
    
    # Overlay the color bars
    ax2.imshow(overlay, aspect='auto', origin='lower', extent=[0, concat_spec_np.shape[1], 0, concat_spec_np.shape[0]])
    
    ax2.set_xlabel('Time (bins)', fontsize=12)
    ax2.set_ylabel('Frequency (mel bins)', fontsize=12)
    ax2.set_title('Concatenated Spectrogram with UMAP-colored Timebins', fontsize=14)
    
    # Add colorbar for spectrogram intensity
    cbar = plt.colorbar(im, ax=ax2, pad=0.01)
    cbar.set_label('Intensity', rotation=270, labelpad=15)
    
    # Add text annotation
    ax2.text(0.02, 0.98, f'Total patches: {num_patches}', 
            transform=ax2.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure if specified
    if args.get("save_path"):
        plt.savefig(args["save_path"], dpi=300, bbox_inches='tight')
        print(f"Figure saved to {args['save_path']}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plotting embedding args")
    parser.add_argument("--num_timebins", type=int, required=True, help="Number of time bins")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file (optional)")
    parser.add_argument("--spec_dir", type=str, required=True, help="Directory of specs to plot the embedding of")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the figure (optional)")
    
    args = parser.parse_args()
    main(vars(args))