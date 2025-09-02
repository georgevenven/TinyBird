import argparse 
from model import TinyBird
from data_loader import SpectogramDataset
from utils import load_model_from_checkpoint
import torch
import umap

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
    total_timebins = 0
    i = 0
    while i < len(embedding_dataset) and total_timebins < args["num_timebins"]:
        spec, labels = embedding_dataset[i]
        
        patch_height, patch_width, mels, num_timebins = config["patch_height"], config["patch_width"], config["mels"], config["num_timebins"]
        spec = spec[:,:mels,:num_timebins]

        spec_timebins = spec.shape[-1]
        spec_mels = spec.shape[-2]

        remainder = spec_timebins % patch_width
        if remainder != 0: spec = spec[:,:,:spec_timebins-remainder]
        spec_timebins = spec.shape[-1]
        spec_mels = spec.shape[-2]

        spec = spec.unsqueeze(0) # add a batch dimension
        
        # TODO: temp conversion to float32, should be removed later
        spec = spec.float()
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
        # Remove positional encoding bias by subtracting mean across positions
        if args["subtract_pos"]:
            z_mean = z_freq_stack.mean(dim=0, keepdim=True)  # mean across time dimension
            z_centered = z_freq_stack - z_mean
        else:
            z_centered = z_freq_stack

        latent_list.append(z_centered)

        total_timebins += spec_timebins
        i += 1

    Z = torch.cat(latent_list, dim=0)
    reducer_enc = umap.UMAP(n_components=2, metric='cosine', n_jobs=-1, low_memory=True)
    emb_enc = reducer_enc.fit_transform(Z.detach().cpu().numpy())

    # Plot the embedding
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    plt.scatter(emb_enc[:, 0], emb_enc[:, 1], alpha=0.1)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('UMAP Embedding of Spectrogram Patches')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plotting embedding args")
    parser.add_argument("--num_timebins", type=int, required=True, help="Number of time bins")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file (optional)")
    parser.add_argument("--spec_dir", type=str, required=True, help="Directory of specs to plot the embedding of")
    parser.add_argument("--subtract_pos", action="store_true", help="Subtract positional encoding bias")
    args = parser.parse_args()
    main(vars(args))
