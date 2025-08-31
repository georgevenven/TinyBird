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
        
        # Setup loss logging file
        self.loss_log_path = os.path.join(self.run_path, "loss_log.txt")
        with open(self.loss_log_path, 'w') as f:
            f.write("step,train_loss,ema_train_loss,val_loss,ema_val_loss\n")

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
        
        # Denormalize predictions to match original patch scale
        def denormalize_predictions(x_patches, pred_patches):
            # x_patches: (B, T, P), pred_patches: (B, T, P)
            # Apply same normalization as loss function, then reverse it on predictions
            target_mean = x_patches.mean(dim=-1, keepdim=True)
            target_std = x_patches.std(dim=-1, keepdim=True)
            # Denormalize: pred_denorm = pred * std + mean
            pred_denorm = pred_patches * (target_std + 1e-6) + target_mean
            return pred_denorm
        
        # Create overlay: unmasked original + masked predictions
        def create_overlay(x_patches, pred_patches, bool_mask):
            # x_patches: (B, T, P), pred_patches: (B, T, P), bool_mask: (B, T)
            overlay_patches = x_patches.clone()
            overlay_patches[bool_mask] = pred_patches[bool_mask]
            return overlay_patches
        
        # Convert input to patches for overlay
        unfold = nn.Unfold(kernel_size=self.config["patch_size"], stride=self.config["patch_size"])
        x_patches = unfold(x).transpose(1, 2)  # (B, T, P)
        
        # Denormalize predictions to original scale
        pred_denorm = denormalize_predictions(x_patches, pred)
        
        # Create overlay patches
        overlay_patches = create_overlay(x_patches, pred_denorm, bool_mask)
        
        # Save reconstruction comparison
        x_img = x[0, 0].detach().cpu().numpy()  # First sample, first channel
        r_img = depatchify(pred_denorm)[0, 0].detach().cpu().numpy()  # Use denormalized predictions
        overlay_img = depatchify(overlay_patches)[0, 0].detach().cpu().numpy()
        
        fig = plt.figure(figsize=(12, 4.5))  # Taller figure for 3 rows
        
        ax1 = plt.subplot(3, 1, 1)
        ax1.imshow(x_img, origin="lower", aspect="auto")
        ax1.set_title("Input Spectrogram")
        ax1.axis("off")
        
        ax2 = plt.subplot(3, 1, 2)
        ax2.imshow(r_img, origin="lower", aspect="auto")
        ax2.set_title("Reconstructed Spectrogram")
        ax2.axis("off")
        
        ax3 = plt.subplot(3, 1, 3)
        ax3.imshow(overlay_img, origin="lower", aspect="auto")
        ax3.set_title("Overlay: Unmasked Original + Masked Predictions")
        ax3.axis("off")
        
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
                
                # Log losses to file
                with open(self.loss_log_path, 'a') as f:
                    f.write(f"{step_num},{train_loss:.6f},{self.ema_train_loss:.6f},{val_loss:.6f},{self.ema_val_loss:.6f}\n")
                
                # Save model weights
                weight_path = os.path.join(self.weights_path, f"model_step_{step_num:06d}.pth")
                torch.save(self.tinybird.state_dict(), weight_path)
                
                # Save reconstruction visualization
                self.save_reconstruction(val_batch, step_num)
        
        # Generate loss plot at the end of training
        self.end_of_train_viz()

    def end_of_train_viz(self):
        """Generate and save loss plots showing training and validation curves."""
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # First panel: All losses
        ax1.plot(self.train_steps, self.train_loss_history, 
                label='Training Loss', alpha=0.7, linewidth=1, color='blue')
        ax1.plot(self.val_steps, self.val_loss_history, 
                label='Validation Loss', marker='o', markersize=3, 
                linewidth=2, color='red')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale for loss values
        
        # Second panel: EMA losses only
        # Calculate EMA train loss history for plotting
        ema_train_history = []
        ema_alpha = self.ema_alpha
        ema_val = None
        for loss in self.train_loss_history:
            if ema_val is None:
                ema_val = loss
            else:
                ema_val = ema_alpha * ema_val + (1 - ema_alpha) * loss
            ema_train_history.append(ema_val)
        
        # Calculate EMA val loss history
        ema_val_history = []
        ema_val = None
        for i, loss in enumerate(self.val_loss_history):
            if ema_val is None:
                ema_val = loss
            else:
                ema_val = ema_alpha * ema_val + (1 - ema_alpha) * loss
            ema_val_history.append(ema_val)
        
        ax2.plot(self.train_steps, ema_train_history, 
                label='EMA Training Loss', linewidth=2, color='darkblue')
        ax2.plot(self.val_steps, ema_val_history, 
                label='EMA Validation Loss', marker='o', markersize=3, 
                linewidth=2, color='darkred')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('EMA Loss')
        ax2.set_title('Exponential Moving Average Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')  # Log scale for loss values
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.imgs_path, 'loss_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Loss plot saved to: {plot_path}")
        print(f"Loss log saved to: {self.loss_log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pretrain args")

    # Required argparse
    parser.add_argument("--train_dir", type=str, required=True, help="training directory")
    parser.add_argument("--val_dir", type=str, required=True, help="validation directory")
    parser.add_argument("--run_name", type=str, required=True, help="directory name inside /runs to store train run details")

    # Defaults 
    parser.add_argument("--steps", type=int, default=100_000, help="number of training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--patch_height", type=int, default=32, help="patch height")
    parser.add_argument("--patch_width", type=int, default=8, help="patch width")
    parser.add_argument("--mels", type=int, default=128, help="number of mel bins")
    parser.add_argument("--num_timebins", type=int, default=512, help="number of time bins")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--mask_p", type=float, default=0.25, help="mask probability")
    parser.add_argument("--eval_every", type=int, default=500, help="evaluate every N steps")

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
