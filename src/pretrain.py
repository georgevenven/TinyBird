import argparse
import os
import shutil
import json
from datetime import datetime
import time

# Set matplotlib backend BEFORE importing plotting_utils
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import TinyBird
from plotting_utils import plot_loss_curves, save_reconstruction_plot
from utils import load_training_state

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")


def resolve_run_path(path_fragment):
    """Return absolute path to run directory under project root."""
    if os.path.isabs(path_fragment):
        return path_fragment

    project_relative = os.path.abspath(os.path.join(PROJECT_ROOT, path_fragment))
    if os.path.exists(project_relative):
        return project_relative

    return os.path.abspath(os.path.join(RUNS_ROOT, path_fragment))

class Trainer():
    def __init__(self, config, pretrained_model=None):
        self.config = config
        
        # Handle continue mode vs new training
        if config.get('is_continuing', False):
            # Continue training mode - use existing run directory
            continue_from = config['continue_from']
            self.run_path = resolve_run_path(continue_from)
            
            if not os.path.exists(self.run_path):
                raise FileNotFoundError(f"Continue directory not found: {self.run_path}")
            
            print(f"Continuing training from: {self.run_path}")
            
        else:
            # New training mode - setup run directory
            os.makedirs(RUNS_ROOT, exist_ok=True)
            
            self.run_path = os.path.join(RUNS_ROOT, config["run_name"])
            if os.path.exists(self.run_path):
                archive_dir = os.path.join(RUNS_ROOT, "archive")
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
        
        # Save config as JSON (only for new runs)
        if not config.get('is_continuing', False):
            config_path = os.path.join(self.run_path, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        if pretrained_model is not None:
            # Use the loaded model from continue mode
            self.tinybird = pretrained_model.to(self.device)
            print("Using loaded model from checkpoint")
        else:
            # Initialize new model
            self.tinybird = TinyBird(config).to(self.device)
            print("Initialized new model")
        
        # Print parameter counts
        from utils import count_parameters
        count_parameters(self.tinybird)
        
        # Initialize optimizer
        self.optimizer = AdamW(self.tinybird.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        
        # Initialize cosine annealing scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config["steps"])
        
        # Initialize AMP scaler if AMP is enabled
        self.use_amp = config.get("amp", False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Loss history for plotting
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_steps = []
        self.val_steps = []
        
        # Initialize step tracking
        self.starting_step = 0
        if config.get('is_continuing', False):
            # Load existing step count and loss history
            training_state = load_training_state(self.run_path, config.get('eval_every', 500))
            
            # Apply loaded state to trainer
            self.starting_step = training_state['starting_step']
            self.train_steps = training_state['steps']
            self.train_loss_history = training_state['train_losses']
            self.val_steps = training_state['steps']
            self.val_loss_history = training_state['val_losses']
            
            # Advance scheduler to correct step if training state was found
            if training_state['found_state']:
                for _ in range(self.starting_step):
                    self.scheduler.step()
        
        # Setup loss logging file
        self.loss_log_path = os.path.join(self.run_path, "loss_log.txt")
        if not config.get('is_continuing', False):
            # Create new loss log for new training
            with open(self.loss_log_path, 'w') as f:
                f.write("step,train_loss,val_loss,gnorm,samples_processed,steps_per_sec,samples_per_sec\n")
        else:
            # Verify loss log exists for continuing training
            if not os.path.exists(self.loss_log_path):
                print(f"Warning: Loss log not found at {self.loss_log_path}, starting fresh")


    def step(self, batch, is_training=True):
        """
        Perform one forward pass and optionally backward pass.
        
        Args:
            batch: Input batch (spectrograms, filenames)
            is_training: If True, perform gradient update. If False, no gradients.
            
        Returns:
            loss: Scalar loss value
            gnorm: Gradient norm (only for training, else None)
        """
        spectrograms, _ = batch
        x = spectrograms.to(self.device, non_blocking=True)  # (B, 1, H, W)
        
        if is_training:
            self.tinybird.train()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.tinybird.eval()
        
        # Forward pass through encoder-decoder
        with torch.set_grad_enabled(is_training):
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    h, idx_restore, bool_mask, T = self.tinybird.forward_encoder(x)
                    pred = self.tinybird.forward_decoder(h, idx_restore, T)
                    loss = self.tinybird.loss_mse(x, pred, bool_mask)
            else:
                h, idx_restore, bool_mask, T = self.tinybird.forward_encoder(x)
                pred = self.tinybird.forward_decoder(h, idx_restore, T)
                loss = self.tinybird.loss_mse(x, pred, bool_mask)
        
        # Backward pass only for training
        gnorm = None
        if is_training:
            if self.use_amp:
                self.scaler.scale(loss).backward()
                # Unscale gradients to compute true gradient norm
                self.scaler.unscale_(self.optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(self.tinybird.parameters(), float('inf'))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(self.tinybird.parameters(), float('inf'))
                self.optimizer.step()
            
            # Update learning rate scheduler
            self.scheduler.step()
        
        return loss.item(), gnorm.item() if gnorm is not None else None

    def save_reconstruction(self, batch, step_num):
        """Save reconstruction visualization comparing input and output spectrograms."""
        save_reconstruction_plot(
            self.tinybird,
            batch,
            config=self.config,
            device=self.device,
            use_amp=self.use_amp,
            output_dir=self.imgs_path,
            step_num=step_num,
        )

    def train(self):
        from data_loader import SpectogramDataset
        from torch.utils.data import DataLoader
        
        # Initialize datasets
        train_dataset = SpectogramDataset(
            dir=self.config["train_dir"],
            n_timebins=self.config["num_timebins"]
        )
        
        val_dataset = SpectogramDataset(
            dir=self.config["val_dir"],
            n_timebins=self.config["num_timebins"]
        )
        
        # Initialize dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=True
        )
        
        # Training loop
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)
        
        # Calculate total steps and range
        total_steps = self.config["steps"]
        end_step = self.starting_step + total_steps
        
        # Initialize timing for batches per second calculation
        last_eval_time = time.time()
        last_eval_step = self.starting_step
        
        print(f"Training from step {self.starting_step} to {end_step}")
        
        for step_num in range(self.starting_step, end_step):
            try:
                train_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                train_batch = next(train_iter)
            
            # Training step
            train_loss, gnorm = self.step(train_batch, is_training=True)
            
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
                val_loss, _ = self.step(val_batch, is_training=False)
                
                # Store validation loss
                self.val_loss_history.append(val_loss)
                self.val_steps.append(step_num)
                
                # Calculate samples processed
                samples_processed = self.config["batch_size"] * (step_num + 1)
                
                # Calculate percentage complete
                progress_pct = ((step_num - self.starting_step + 1) / total_steps) * 100
                
                # Calculate steps per second and samples per second
                current_time = time.time()
                elapsed_time = current_time - last_eval_time
                steps_since_last_eval = step_num - last_eval_step
                steps_per_sec = steps_since_last_eval / elapsed_time if elapsed_time > 0 else 0
                samples_per_sec = steps_per_sec * self.config["batch_size"]
                
                # Update timing trackers
                last_eval_time = current_time
                last_eval_step = step_num
                
                # Print progress
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Step {step_num} ({progress_pct:.1f}%): Train Loss = {train_loss:.6f}, "
                      f"Val Loss = {val_loss:.6f}, "
                      f"Gnorm = {gnorm:.6f}, "
                      f"Samples = {samples_processed}, "
                      f"LR = {current_lr:.2e}, "
                      f"Steps/sec = {steps_per_sec:.2f}, "
                      f"Samples/sec = {samples_per_sec:.1f}")
                
                # Log losses to file
                with open(self.loss_log_path, 'a') as f:
                    f.write(f"{step_num},{train_loss:.6f},{val_loss:.6f},{gnorm:.6f},{samples_processed},{steps_per_sec:.2f},{samples_per_sec:.1f}\n")
                
                # Save model weights
                weight_path = os.path.join(self.weights_path, f"model_step_{step_num:06d}.pth")
                torch.save(self.tinybird.state_dict(), weight_path)
                
                # Save reconstruction visualization
                self.save_reconstruction(val_batch, step_num)
        
        # Save final model weights
        final_step = self.starting_step + self.config['steps'] - 1
        final_weight_path = os.path.join(self.weights_path, f"model_step_{final_step:06d}.pth")
        torch.save(self.tinybird.state_dict(), final_weight_path)
        
        # Generate loss plot at the end of training
        self.end_of_train_viz()

    def end_of_train_viz(self):
        """Generate and save loss plots showing training and validation curves."""
        plot_path = os.path.join(self.imgs_path, 'loss_plot.png')
        plot_loss_curves(
            train_steps=self.train_steps,
            train_losses=self.train_loss_history,
            val_steps=self.val_steps,
            val_losses=self.val_loss_history,
            loss_log_path=self.loss_log_path,
            output_path=plot_path,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pretrain args")

    # Required argparse
    parser.add_argument("--train_dir", type=str, help="training directory")
    parser.add_argument("--val_dir", type=str, help="validation directory")
    parser.add_argument("--run_name", type=str, help="directory name inside /runs to store train run details")

    # Defaults 
    parser.add_argument("--steps", type=int, default=500_000, help="number of training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="number of DataLoader worker processes")

    parser.add_argument("--patch_height", type=int, default=32, help="patch height")
    parser.add_argument("--patch_width", type=int, default=1, help="patch width")
    parser.add_argument("--num_timebins", type=int, default=1024, help="n number of time bins")
    
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--mask_p", type=float, default=0.75, help="mask probability")
    parser.add_argument("--mask_c", type=float, default=0.1, help="seed probability for Voronoi mask")
    parser.add_argument("--eval_every", type=int, default=500, help="evaluate every N steps")
    parser.add_argument("--amp", action="store_true", help="enable automatic mixed precision training")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay")
    parser.add_argument("--continue_from", type=str, help="continue training from existing run directory (path to run dir)")

    # Encoder 
    parser.add_argument("--enc_hidden_d", type=int, default=384, help="encoder hidden dimension")
    parser.add_argument("--enc_n_head", type=int, default=6, help="encoder number of attention heads")
    parser.add_argument("--enc_n_layer", type=int, default=6, help="encoder number of transformer layers")
    parser.add_argument("--enc_dim_ff", type=int, default=1536, help="encoder feed-forward dimension")

    # Decoder Model
    parser.add_argument("--dec_hidden_d", type=int, default=192, help="decoder hidden dimension")
    parser.add_argument("--dec_n_head", type=int, default=6, help="decoder number of attention heads")
    parser.add_argument("--dec_n_layer", type=int, default=2, help="decoder number of transformer layers")
    parser.add_argument("--dec_dim_ff", type=int, default=768, help="decoder feed-forward dimension")

    args = parser.parse_args()
    
    # Handle continue mode vs new training
    if args.continue_from:
        # Continue training mode - load config from existing run
        from utils import load_model_from_checkpoint
        
        # Load existing config and model
        resolved_continue = resolve_run_path(args.continue_from)
        model, config = load_model_from_checkpoint(resolved_continue, fallback_to_random=False)
        
        config['continue_from'] = resolved_continue
        config['is_continuing'] = True
        config.setdefault("mask_c", args.mask_c)

    else:
        # New training mode - validate required args
        if not args.train_dir or not args.val_dir or not args.run_name:
            parser.error("--train_dir, --val_dir, and --run_name are required when not using --continue_from")
        
        config = vars(args)
        config['is_continuing'] = False
    
    # Load audio params from training directory
    from utils import load_audio_params
    
    audio_params = load_audio_params(config["train_dir"])
    config["mels"] = audio_params["mels"]
    # Configure patch size and max sequence length for model
    config["patch_size"] = (config["patch_height"], config["patch_width"])
    config["max_seq"] = (config["num_timebins"] // config["patch_width"]) * (config["mels"] // config["patch_height"])

    # Calculate seq_len from num_timebins and patch dimensions  
    assert config["num_timebins"] % config["patch_width"] == 0, f"num_timebins ({config['num_timebins']}) must be divisible by patch_width ({config['patch_width']})"
    assert config["mels"] % config["patch_height"] == 0, f"mels ({config['mels']}) must be divisible by patch_height ({config['patch_height']})"

    # Create trainer with loaded model if continuing
    if config.get('is_continuing', False):
        trainer = Trainer(config, pretrained_model=model)
    else:
        trainer = Trainer(config)
    
    trainer.train()
