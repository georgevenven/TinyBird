import argparse 
import os
import shutil
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import AdamW
import matplotlib.pyplot as plt
from model import TinyBird

class Trainer():
    def __init__(self, config):
        self.config = config
        
        # Setup run directory - move existing to archive if it exists
        runs_base = os.path.join("..", "runs")
        os.makedirs(runs_base, exist_ok=True)
        
        self.run_path = os.path.join(runs_base, config["run_name"])
        if os.path.exists(self.run_path):
            archive_dir = os.path.join(runs_base, "archive")
            os.makedirs(archive_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived_path = os.path.join(archive_dir, f"{config['run_name']}_{timestamp}")
            shutil.move(self.run_path, archived_path)
            print(f"Moved existing run directory to: {archived_path}")
        
        os.makedirs(self.run_path, exist_ok=True)
        
        # Create subdirectories
        self.weights_path = os.path.join(self.run_path, "weights")
        self.imgs_path = os.path.join(self.run_path, "imgs")
        os.makedirs(self.weights_path, exist_ok=True)
        os.makedirs(self.imgs_path, exist_ok=True)
        
        # Save config as JSON
        config_path = os.path.join(self.run_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.tinybird = TinyBird(config).to(self.device)
        
        # Initialize optimizer
        self.optimizer = AdamW(self.tinybird.parameters(), lr=config["lr"], weight_decay=0.0)
        
        # Loss tracking
        self.ema_train_loss = None
        self.ema_val_loss = None
        self.ema_alpha = 0.99
        
        # Loss history for plotting
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_steps = []
        self.val_steps = []

    def step(self, batch, is_training=True):
        """
        Perform one forward pass and optionally backward pass.
        
        Args:
            batch: Input batch (spectrograms, labels)
            is_training: If True, perform gradient update. If False, no gradients.
            
        Returns:
            loss: Scalar loss value
        """
        spectrograms, _ = batch
        x = spectrograms.float().to(self.device, non_blocking=True)  # (B, 1, H, W)
        
        if is_training:
            self.tinybird.train()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.tinybird.eval()
        
        # Forward pass through encoder-decoder
        with torch.set_grad_enabled(is_training):
            h, idx_restore, bool_mask, T = self.tinybird.forward_encoder(x)
            pred = self.tinybird.forward_decoder(h, idx_restore, T)
            loss = self.tinybird.loss_mse(x, pred, bool_mask)
        
        # Backward pass only for training
        if is_training:
            loss.backward()
            self.optimizer.step()
            
            # Update EMA train loss
            if self.ema_train_loss is None:
                self.ema_train_loss = loss.item()
            else:
                self.ema_train_loss = self.ema_alpha * self.ema_train_loss + (1 - self.ema_alpha) * loss.item()
        else:
            # Update EMA val loss
            if self.ema_val_loss is None:
                self.ema_val_loss = loss.item()
            else:
                self.ema_val_loss = self.ema_alpha * self.ema_val_loss + (1 - self.ema_alpha) * loss.item()
        
        return loss.item()

    def save_reconstruction(self, batch, step_num):
        """Save reconstruction visualization comparing input and output spectrograms."""
        spectrograms, _ = batch
        x = spectrograms.float().to(self.device, non_blocking=True)  # (B, 1, H, W)
        
        # Get model prediction
        self.tinybird.eval()
        with torch.no_grad():
            h, idx_restore, bool_mask, T = self.tinybird.forward_encoder(x)
            pred = self.tinybird.forward_decoder(h, idx_restore, T)
        
        # Depatchify prediction to get back (B, 1, H, W) format
        def depatchify(pred_patches):
            # pred_patches: (B, T, P) → (B, 1, H, W)
            H, W = self.config["mels"], self.config["num_timebins"]
            patch_size = self.config["patch_size"]
            fold = nn.Fold(output_size=(H, W), kernel_size=patch_size, stride=patch_size)
            return fold(pred_patches.transpose(1, 2))
        
        # Save reconstruction comparison
        x_img = x[0, 0].detach().cpu().numpy()  # First sample, first channel
        r_img = depatchify(pred)[0, 0].detach().cpu().numpy()
        
        fig = plt.figure(figsize=(12, 3))  # Wide rectangular figure for time dimension
        ax1 = plt.subplot(2, 1, 1)
        ax1.imshow(x_img, origin="lower", aspect="auto")
        ax1.set_title("Input Spectrogram")
        ax1.axis("off")
        
        ax2 = plt.subplot(2, 1, 2)
        ax2.imshow(r_img, origin="lower", aspect="auto")
        ax2.set_title("Reconstructed Spectrogram")
        ax2.axis("off")
        
        fig.tight_layout()
        recon_path = os.path.join(self.imgs_path, f"recon_step_{step_num:06d}.png")
        fig.savefig(recon_path, dpi=150)
        plt.close(fig)

    def train(self):
        from data_loader import SpectogramDataset
        from torch.utils.data import DataLoader
        
        # Initialize datasets
        train_dataset = SpectogramDataset(
            dir=self.config["train_dir"],
            n_mels=self.config["mels"],
            n_timebins=self.config["num_timebins"]
        )
        
        val_dataset = SpectogramDataset(
            dir=self.config["val_dir"],
            n_mels=self.config["mels"],
            n_timebins=self.config["num_timebins"]
        )
        
        # Initialize dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Training loop
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)
        
        for step_num in range(self.config["steps"]):
            try:
                train_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                train_batch = next(train_iter)
            
            # Training step
            train_loss = self.step(train_batch, is_training=True)
            
            # Store training loss every step
            self.train_loss_history.append(train_loss)
            self.train_steps.append(step_num)
            
            # Evaluation and checkpointing
            if step_num % self.config["eval_every"] == 0:
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_batch = next(val_iter)
                
                # Validation step (no gradients)
                val_loss = self.step(val_batch, is_training=False)
                
                # Store validation loss
                self.val_loss_history.append(val_loss)
                self.val_steps.append(step_num)
                
                # Print progress
                print(f"Step {step_num}: Train Loss = {train_loss:.6f}, "
                      f"EMA Train = {self.ema_train_loss:.6f}, "
                      f"Val Loss = {val_loss:.6f}, "
                      f"EMA Val = {self.ema_val_loss:.6f}")
                
                # Save model weights
                weight_path = os.path.join(self.weights_path, f"model_step_{step_num:06d}.pth")
                torch.save(self.tinybird.state_dict(), weight_path)
                
                # Save reconstruction visualization
                self.save_reconstruction(val_batch, step_num)
        
        # Generate loss plot at the end of training
        self.end_of_train_viz()

    def end_of_train_viz(self):
        """Generate and save loss plot showing training and validation curves."""
        plt.figure(figsize=(12, 8))
        
        # Plot training loss (more data points)
        plt.plot(self.train_steps, self.train_loss_history, 
                label='Training Loss', alpha=0.7, linewidth=1, color='blue')
        
        # Plot validation loss (fewer data points)
        plt.plot(self.val_steps, self.val_loss_history, 
                label='Validation Loss', marker='o', markersize=3, 
                linewidth=2, color='red')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for loss values
        
        # Save the plot
        plot_path = os.path.join(self.imgs_path, 'loss_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Loss plot saved to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pretrain args")

    # Required argparse
    parser.add_argument("--train_dir", type=str, required=True, help="training directory")
    parser.add_argument("--val_dir", type=str, required=True, help="validation directory")
    parser.add_argument("--run_name", type=str, required=True, help="directory name inside /runs to store train run details")

    # Defaults 
    parser.add_argument("--steps", type=int, default=100000, help="number of training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--patch_height", type=int, default=32, help="patch height")
    parser.add_argument("--patch_width", type=int, default=8, help="patch width")
    parser.add_argument("--mels", type=int, default=128, help="number of mel bins")
    parser.add_argument("--num_timebins", type=int, default=512, help="number of time bins")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--mask_p", type=float, default=0.5, help="mask probability")
    parser.add_argument("--eval_every", type=int, default=50, help="evaluate every N steps")

    # Encoder Model
    parser.add_argument("--enc_hidden_d", type=int, default=192, help="encoder hidden dimension")
    parser.add_argument("--enc_n_head", type=int, default=6, help="encoder number of attention heads")
    parser.add_argument("--enc_n_layer", type=int, default=6, help="encoder number of transformer layers")
    parser.add_argument("--enc_dim_ff", type=int, default=768, help="encoder feed-forward dimension")

    # Decoder Model
    parser.add_argument("--dec_hidden_d", type=int, default=192, help="decoder hidden dimension")
    parser.add_argument("--dec_n_head", type=int, default=6, help="decoder number of attention heads")
    parser.add_argument("--dec_n_layer", type=int, default=3, help="decoder number of transformer layers")
    parser.add_argument("--dec_dim_ff", type=int, default=768, help="decoder feed-forward dimension")

    args = parser.parse_args()
    config = vars(args)

    # Calculate seq_len from num_timebins and patch dimensions  
    assert config["num_timebins"] % config["patch_width"] == 0, f"num_timebins ({config['num_timebins']}) must be divisible by patch_width ({config['patch_width']})"
    assert config["mels"] % config["patch_height"] == 0, f"mels ({config['mels']}) must be divisible by patch_height ({config['patch_height']})"
    
    # Configure patch size and max sequence length for model
    config["patch_size"] = (config["patch_height"], config["patch_width"])
    config["max_seq"] = (config["num_timebins"] // config["patch_width"]) * (config["mels"] // config["patch_height"])
    
    trainer = Trainer(config)
    trainer.train()
