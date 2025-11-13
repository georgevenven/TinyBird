### For Linear probe, or supervised detection and classification tasks ###

import argparse
import os
import shutil
import json
from datetime import datetime
import time

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

class SupervisedTinyBird(nn.Module):
    def __init__(self, pretrained_model, config, num_classes=2, freeze_encoder=True, mode="detect"):
        """
        Supervised classification/detection model built on top of pretrained TinyBird encoder.
        
        Args:
            pretrained_model: Pretrained TinyBird model
            config: Configuration dict with patch_size, etc.
            num_classes: Number of output classes (2 for detection, N for classification)
            freeze_encoder: If True, freeze encoder weights (linear probe mode)
            mode: "detect" for binary detection, "classify" for multi-class classification
        """
        super().__init__()
        
        self.encoder = pretrained_model
        self.patch_height = config["patch_height"]
        self.patch_width = config["patch_width"]
        self.num_classes = num_classes
        self.enc_hidden_d = config["enc_hidden_d"]
        self.mels = config["mels"]
        self.mode = mode
        
        # Freeze encoder if in linear probe mode
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen - training linear probe only")
        else:
            print("Encoder unfrozen - finetuning entire model")
        
        # Calculate input dimension after concatenating height patches
        # H_patches * D_enc since we concatenate height-wise
        H_patches = self.mels // self.patch_height
        input_dim = H_patches * self.enc_hidden_d
        
        # Simple MLP classifier: input -> hidden -> output
        hidden_dim = 256
        # For binary classification (2 classes), output 1 logit (BCE)
        # For multi-class, output num_classes logits (CrossEntropy)
        output_dim = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Loss function - BCE for binary (2 classes), cross-entropy for multi-class
        # Class 0 is silence in both cases, explicitly trained
        if num_classes == 2:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x):
        """
        Forward pass for supervised training.
        
        Args:
            x: Input spectrogram (B, 1, H, W) where W is n_timebins
            
        Returns:
            logits: (B, W_patches, 1) for binary classification or (B, W_patches, num_classes) for multi-class
                   - classification logits per time patch
        """
        B, _, H, W = x.shape
        
        # Get encoder embeddings (no masking in inference mode)
        with torch.set_grad_enabled(not self.encoder.training or self.training):
            h, z_seq = self.encoder.forward_encoder_inference(x)  # (B, T, D_enc)
        
        # h has shape (B, T, D_enc) where T = (H/patch_height) * (W/patch_width)
        # Reshape back to spatial grid (B, H_patches, W_patches, D_enc)
        H_patches = H // self.patch_height
        W_patches = W // self.patch_width
        h_grid = h.view(B, H_patches, W_patches, self.enc_hidden_d)
        
        # Concatenate height patches to get (B, W_patches, H_patches * D_enc)
        # This is like extract_embedding.py where we flatten height into feature dim
        h_time = h_grid.permute(0, 2, 1, 3)  # (B, W_patches, H_patches, D_enc)
        h_time = h_time.flatten(2, 3)  # (B, W_patches, H_patches * D_enc)
        
        # Apply MLP classifier to get (B, W_patches, num_classes)
        logits = self.classifier(h_time)
        
        return logits
    
    def compute_loss(self, logits, labels):
        """
        Compute loss between predictions and labels.
        
        Args:
            logits: (B, W_patches, 1) for binary or (B, W_patches, num_classes) for multi-class
            labels: (B, W) where W is n_timebins
                    Class 0 = silence, Class 1+ = vocalizations/syllables
            
        Returns:
            loss: scalar loss value
        """
        B_label, W = labels.shape
        
        # Downsample labels to match patch width
        # Each patch covers patch_width timebins
        labels_downsampled = labels[:, ::self.patch_width]  # (B, W_patches)
        
        if self.num_classes == 2:
            # Binary classification: BCE loss
            # logits shape: (B, W_patches, 1)
            logits_flat = logits.reshape(-1)  # (B * W_patches,)
            labels_flat = labels_downsampled.reshape(-1).float()  # (B * W_patches,) as float for BCE
            loss = self.loss_fn(logits_flat, labels_flat)
        else:
            # Multi-class classification: CrossEntropy loss
            # logits shape: (B, W_patches, num_classes)
            B, W_patches, num_classes = logits.shape
            logits_flat = logits.reshape(-1, num_classes)  # (B * W_patches, num_classes)
            labels_flat = labels_downsampled.reshape(-1)  # (B * W_patches,)
            loss = self.loss_fn(logits_flat, labels_flat)
        
        return loss
    
    def compute_accuracy(self, logits, labels):
        """
        Compute per-timebin accuracy.
        
        Args:
            logits: (B, W_patches, 1) for binary or (B, W_patches, num_classes) for multi-class
            labels: (B, W) where W is n_timebins
            
        Returns:
            accuracy: scalar accuracy value (0-100%)
        """
        B_label, W = labels.shape
        
        # Downsample labels to match patch width
        labels_downsampled = labels[:, ::self.patch_width]
        
        if self.num_classes == 2:
            # Binary classification: use sigmoid + threshold at 0.5
            # logits shape: (B, W_patches, 1)
            logits_flat = logits.reshape(-1)  # (B * W_patches,)
            probs = torch.sigmoid(logits_flat)  # (B * W_patches,)
            preds = (probs > 0.5).long().reshape(labels_downsampled.shape)  # (B, W_patches)
        else:
            # Multi-class classification: use argmax
            # logits shape: (B, W_patches, num_classes)
            B, W_patches, num_classes = logits.shape
            preds = torch.argmax(logits, dim=-1)  # (B, W_patches)
        
        # Compute accuracy
        correct = (preds == labels_downsampled).sum().item()
        total = labels_downsampled.numel()
        accuracy = 100.0 * correct / total
        
        return accuracy
    
    def compute_f1_score(self, logits, labels):
        """
        Compute F1 score for detection (binary classification only).
        
        Args:
            logits: (B, W_patches, 1) for binary or (B, W_patches, num_classes) for multi-class
            labels: (B, W) where W is n_timebins
            
        Returns:
            f1: F1 score (0-100%), or None if not in binary detection mode
        """
        if self.num_classes != 2:
            return None
        
        B_label, W = labels.shape
        
        # Downsample labels to match patch width
        labels_downsampled = labels[:, ::self.patch_width]
        
        # Binary classification: use sigmoid + threshold at 0.5
        # logits shape: (B, W_patches, 1)
        logits_flat = logits.reshape(-1)  # (B * W_patches,)
        probs = torch.sigmoid(logits_flat)  # (B * W_patches,)
        preds = (probs > 0.5).long()  # (B * W_patches,)
        labels_flat = labels_downsampled.reshape(-1)  # (B * W_patches,)
        
        # Calculate TP, FP, FN
        # Positive class is 1 (vocalization detected)
        tp = ((preds == 1) & (labels_flat == 1)).sum().item()
        fp = ((preds == 1) & (labels_flat == 0)).sum().item()
        fn = ((preds == 0) & (labels_flat == 1)).sum().item()
        
        # Compute precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Compute F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1 * 100.0  # Convert to percentage


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")


class Trainer():
    def __init__(self, model, config):
        self.config = config
        self.model = model
        
        # Setup run directory
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
        
        # Save config as JSON
        config_path = os.path.join(self.run_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Print parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        
        # Initialize optimizer (only optimize parameters that require gradients)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=config["lr"], weight_decay=config["weight_decay"])
        
        # Initialize cosine annealing scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config["steps"])
        
        # Initialize AMP scaler if AMP is enabled
        self.use_amp = config.get("amp", False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Loss history for tracking
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.train_f1_history = []
        self.val_f1_history = []
        self.train_steps = []
        self.val_steps = []
        
        # Setup loss logging file
        self.loss_log_path = os.path.join(self.run_path, "loss_log.txt")
        with open(self.loss_log_path, 'w') as f:
            if config["mode"] == "detect":
                f.write("step,train_loss,val_loss,train_acc,val_acc,train_f1,val_f1,samples_processed,steps_per_sec,samples_per_sec\n")
            else:
                f.write("step,train_loss,val_loss,train_acc,val_acc,samples_processed,steps_per_sec,samples_per_sec\n")
    
    def step(self, batch, is_training=True):
        """
        Perform one forward pass and optionally backward pass.
        
        Args:
            batch: Input batch (spectrograms, labels, filenames)
            is_training: If True, perform gradient update. If False, no gradients.
            
        Returns:
            loss: Scalar loss value
            accuracy: Accuracy percentage
            f1: F1 score (only for detection mode, else None)
            logits: Model predictions (for visualization)
        """
        spectrograms, labels, _ = batch
        x = spectrograms.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        
        if is_training:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.model.eval()
        
        # Forward pass
        with torch.set_grad_enabled(is_training):
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(x)
                    loss = self.model.compute_loss(logits, labels)
                    accuracy = self.model.compute_accuracy(logits, labels)
                    f1 = self.model.compute_f1_score(logits, labels)
            else:
                logits = self.model(x)
                loss = self.model.compute_loss(logits, labels)
                accuracy = self.model.compute_accuracy(logits, labels)
                f1 = self.model.compute_f1_score(logits, labels)
        
        # Backward pass only for training
        if is_training:
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
        
        return loss.item(), accuracy, f1, logits
    
    def save_prediction_visualization(self, batch, logits, step_num):
        """
        Save visualization showing spectrogram, predictions, and ground truth.
        
        Args:
            batch: Input batch (spectrograms, labels, filenames)
            logits: Model predictions (B, W_patches, 1) for binary or (B, W_patches, num_classes) for multi-class
            step_num: Current training step
        """
        from plotting_utils import save_supervised_prediction_plot
        
        spectrograms, labels, filenames = batch
        
        # Select a sample from the batch (cycle through to show variety)
        batch_size = spectrograms.shape[0]
        sample_idx = (step_num // self.config["eval_every"]) % batch_size
        
        # Move to CPU and convert to numpy
        spec = spectrograms[sample_idx, 0].cpu().numpy()  # Selected sample, remove channel dim
        labels_np = labels[sample_idx].cpu().numpy()  # (W,)
        logits_np = logits[sample_idx].cpu().numpy()  # (W_patches, 1) or (W_patches, num_classes)
        
        # Get predictions and probabilities
        if self.model.num_classes == 2:
            # Binary classification: use sigmoid + threshold
            logits_flat = logits_np.reshape(-1)  # (W_patches,)
            probs_flat = torch.sigmoid(torch.from_numpy(logits_flat)).numpy()  # (W_patches,)
            preds = (probs_flat > 0.5).astype(int)  # (W_patches,)
            # For visualization, create 2-class probability array
            probs = np.stack([1 - probs_flat, probs_flat], axis=-1)  # (W_patches, 2)
        else:
            # Multi-class classification: use argmax and softmax
            preds = np.argmax(logits_np, axis=-1)  # (W_patches,)
            probs = torch.softmax(torch.from_numpy(logits_np), dim=-1).numpy()  # (W_patches, num_classes)
        
        # Downsample labels to match predictions
        labels_downsampled = labels_np[::self.config["patch_width"]]
        
        # Call plotting utility
        save_path = save_supervised_prediction_plot(
            spectrogram=spec,
            labels=labels_downsampled,
            predictions=preds,
            probabilities=probs if self.config["mode"] == "detect" else None,
            filename=filenames[sample_idx],
            mode=self.config["mode"],
            num_classes=self.model.num_classes,
            output_dir=self.imgs_path,
            step_num=step_num
        )
    
    def train(self):
        from data_loader import SupervisedSpectogramDataset
        from torch.utils.data import DataLoader
        
        # Initialize datasets
        train_dataset = SupervisedSpectogramDataset(
            dir=self.config["train_dir"],
            annotation_file_path=self.config["annotation_file"],
            n_timebins=self.config["num_timebins"],
            mode=self.config["mode"]
        )
        
        val_dataset = SupervisedSpectogramDataset(
            dir=self.config["val_dir"],
            annotation_file_path=self.config["annotation_file"],
            n_timebins=self.config["num_timebins"],
            mode=self.config["mode"]
        )
        
        # Verify num_classes matches dataset
        if train_dataset.num_classes != self.model.num_classes:
            print(f"Warning: Model was initialized with {self.model.num_classes} classes, "
                  f"but dataset has {train_dataset.num_classes} classes")
        
        print(f"Dataset has {train_dataset.num_classes} classes (mode: {self.config['mode']})")
        
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
        
        total_steps = self.config["steps"]
        
        # Initialize timing
        last_eval_time = time.time()
        last_eval_step = 0
        
        print(f"Training for {total_steps} steps")
        
        for step_num in range(total_steps):
            try:
                train_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                train_batch = next(train_iter)
            
            # Training step
            train_loss, train_acc, train_f1, _ = self.step(train_batch, is_training=True)
            
            # Store training loss and accuracy every step
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)
            if train_f1 is not None:
                self.train_f1_history.append(train_f1)
            self.train_steps.append(step_num)
            
            # Evaluation and checkpointing
            if step_num % self.config["eval_every"] == 0:
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_batch = next(val_iter)
                
                # Validation step (no gradients)
                val_loss, val_acc, val_f1, val_logits = self.step(val_batch, is_training=False)
                
                # Store validation loss and accuracy
                self.val_loss_history.append(val_loss)
                self.val_acc_history.append(val_acc)
                if val_f1 is not None:
                    self.val_f1_history.append(val_f1)
                self.val_steps.append(step_num)
                
                # Calculate samples processed
                samples_processed = self.config["batch_size"] * (step_num + 1)
                
                # Calculate percentage complete
                progress_pct = ((step_num + 1) / total_steps) * 100
                
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
                if self.config["mode"] == "detect":
                    print(f"Step {step_num} ({progress_pct:.1f}%): "
                          f"Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, "
                          f"Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%, "
                          f"Train F1 = {train_f1:.2f}%, Val F1 = {val_f1:.2f}%, "
                          f"Samples = {samples_processed}, "
                          f"LR = {current_lr:.2e}, "
                          f"Steps/sec = {steps_per_sec:.2f}, "
                          f"Samples/sec = {samples_per_sec:.1f}")
                else:
                    print(f"Step {step_num} ({progress_pct:.1f}%): "
                          f"Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, "
                          f"Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%, "
                          f"Samples = {samples_processed}, "
                          f"LR = {current_lr:.2e}, "
                          f"Steps/sec = {steps_per_sec:.2f}, "
                          f"Samples/sec = {samples_per_sec:.1f}")
                
                # Log losses and accuracies to file
                with open(self.loss_log_path, 'a') as f:
                    if self.config["mode"] == "detect":
                        f.write(f"{step_num},{train_loss:.6f},{val_loss:.6f},{train_acc:.2f},{val_acc:.2f},{train_f1:.2f},{val_f1:.2f},{samples_processed},{steps_per_sec:.2f},{samples_per_sec:.1f}\n")
                    else:
                        f.write(f"{step_num},{train_loss:.6f},{val_loss:.6f},{train_acc:.2f},{val_acc:.2f},{samples_processed},{steps_per_sec:.2f},{samples_per_sec:.1f}\n")
                
                # Save model weights
                weight_path = os.path.join(self.weights_path, f"model_step_{step_num:06d}.pth")
                torch.save(self.model.state_dict(), weight_path)
                
                # Save prediction visualization
                self.save_prediction_visualization(val_batch, val_logits, step_num)
        
        # Save final model weights
        final_step = self.config['steps'] - 1
        final_weight_path = os.path.join(self.weights_path, f"model_step_{final_step:06d}.pth")
        torch.save(self.model.state_dict(), final_weight_path)
        
        print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="supervised training args")
    
    # Required arguments
    parser.add_argument("--train_dir", type=str, required=True, help="training directory")
    parser.add_argument("--val_dir", type=str, required=True, help="validation directory")
    parser.add_argument("--run_name", type=str, required=True, help="directory name inside /runs to store train run details")
    parser.add_argument("--pretrained_run", type=str, required=True, help="path to pretrained run directory")
    parser.add_argument("--annotation_file", type=str, required=True, help="path to annotation JSON file")
    parser.add_argument("--mode", type=str, required=True, choices=["detect", "classify"], help="detect or classify mode")
    
    # Training hyperparameters
    parser.add_argument("--steps", type=int, default=50_000, help="number of training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="number of DataLoader worker processes")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay")
    parser.add_argument("--eval_every", type=int, default=500, help="evaluate every N steps")
    
    # Model configuration
    parser.add_argument("--freeze_encoder", action="store_true", help="freeze encoder weights (linear probe mode)")
    parser.add_argument("--amp", action="store_true", help="enable automatic mixed precision training")
    
    args = parser.parse_args()
    
    # Load pretrained model and config
    from utils import load_model_from_checkpoint, get_num_classes_from_annotations
    from pretrain import resolve_run_path
    
    pretrained_path = resolve_run_path(args.pretrained_run)
    pretrained_model, pretrained_config = load_model_from_checkpoint(pretrained_path, fallback_to_random=False)
    
    print(f"Loaded pretrained model from: {pretrained_path}")
    
    # Create supervised config
    config = vars(args)
    
    # Add necessary info from pretrained config
    config["num_timebins"] = pretrained_config["num_timebins"]
    config["patch_height"] = pretrained_config["patch_height"]
    config["patch_width"] = pretrained_config["patch_width"]
    config["enc_hidden_d"] = pretrained_config["enc_hidden_d"]
    config["mels"] = pretrained_config["mels"]
    config["pretrained_run"] = pretrained_path
    
    # Automatically determine number of classes from annotations
    num_classes = get_num_classes_from_annotations(config["annotation_file"], config["mode"])
    config["num_classes"] = num_classes
    
    # Create supervised model
    supervised_model = SupervisedTinyBird(
        pretrained_model=pretrained_model,
        config=config,
        num_classes=num_classes,
        freeze_encoder=config["freeze_encoder"],
        mode=config["mode"]
    )
    
    # Create trainer and train
    trainer = Trainer(supervised_model, config)
    trainer.train()
