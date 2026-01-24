### For MLP classifier, or supervised detection and classification tasks ###

import argparse
import os
import shutil
import json
from datetime import datetime
import time
from pathlib import Path
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from model import apply_lora_to_encoder, LoRALinear

class SupervisedTinyBird(nn.Module):
    def __init__(
        self,
        pretrained_model,
        config,
        num_classes=2,
        freeze_encoder=True,
        freeze_encoder_up_to=None,
        mode="detect",
        linear_probe=False,
        lora_rank=0,
        lora_alpha=1.0,
        lora_dropout=0.0,
    ):
        """
        Supervised classification/detection model built on top of pretrained TinyBird encoder.
        
        Args:
            pretrained_model: Pretrained TinyBird model
            config: Configuration dict with patch_size, etc.
            num_classes: Number of output classes (2 for detection, N for classification)
            freeze_encoder: If True, freeze encoder weights (train MLP classifier only)
            freeze_encoder_up_to: If set, freeze encoder layers up to this index (inclusive)
            mode: "detect" for binary detection, "classify" for multi-class classification
            linear_probe: If True, use a single linear layer instead of MLP
            lora_rank: If > 0, apply LoRA adapters to encoder FFN layers
            lora_alpha: LoRA alpha scaling factor
            lora_dropout: LoRA dropout (applied before low-rank projection)
        """
        super().__init__()
        
        self.encoder = pretrained_model
        self.patch_height = config["patch_height"]
        self.patch_width = config["patch_width"]
        self.num_classes = num_classes
        self.enc_hidden_d = config["enc_hidden_d"]
        self.mels = config["mels"]
        self.mode = mode
        self.encoder_layer_idx = config.get("encoder_layer_idx", None)
        self.class_weighting = bool(config.get("class_weighting", False))
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        
        # LoRA mode: inject adapters and train only them (plus classifier).
        if self.lora_rank > 0:
            replaced = apply_lora_to_encoder(
                self.encoder,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            )
            if replaced == 0:
                print("Warning: LoRA enabled but no FFN layers were replaced.")

            # Freeze all encoder params, then unfreeze only LoRA params.
            for param in self.encoder.parameters():
                param.requires_grad = False
            lora_param_count = 0
            for module in self.encoder.modules():
                if isinstance(module, LoRALinear):
                    for param in module.lora_parameters():
                        param.requires_grad = True
                        lora_param_count += param.numel()
            print(
                "LoRA enabled - training adapters only "
                f"(rank={self.lora_rank}, alpha={self.lora_alpha}, dropout={self.lora_dropout}, "
                f"lora_params={lora_param_count:,})"
            )
            if freeze_encoder or freeze_encoder_up_to is not None:
                print("LoRA enabled - ignoring freeze_encoder/freeze_encoder_up_to")
        # Freeze encoder if in MLP classifier mode
        elif freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen - training classifier only")
        elif freeze_encoder_up_to is not None:
            layers = getattr(self.encoder.encoder, "layers", None)
            if layers is None:
                raise RuntimeError("TinyBird.encoder does not expose .layers; cannot freeze by layer.")
            num_layers = len(layers)
            idx = int(freeze_encoder_up_to)
            if idx < 0:
                idx = num_layers + idx
            if idx < 0 or idx >= num_layers:
                raise ValueError(
                    f"freeze_encoder_up_to out of range: {freeze_encoder_up_to} (num_layers={num_layers})"
                )

            # Freeze patch projection and positional embeddings (lowest encoder components).
            for param in self.encoder.patch_projection.parameters():
                param.requires_grad = False
            self.encoder.pos_enc.requires_grad = False

            # Freeze encoder transformer layers up to idx (inclusive).
            for layer in layers[: idx + 1]:
                for param in layer.parameters():
                    param.requires_grad = False

            print(f"Encoder partially frozen - frozen layers [0..{idx}]")
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
        
        if linear_probe:
            print("Using Linear Probe (single linear layer)")
            self.classifier = nn.Linear(input_dim, output_dim)
        else:
            print("Using MLP Classifier")
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
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
            h, z_seq = self.encoder.forward_encoder_inference(
                x, encoder_layer_idx=self.encoder_layer_idx
            )  # (B, T, D_enc)
        
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
            if self.class_weighting:
                counts = torch.bincount(labels_flat, minlength=num_classes).float()
                counts_safe = torch.where(counts > 0, counts, torch.ones_like(counts))
                weights = counts.sum() / (counts_safe * num_classes)
                weights = torch.where(counts > 0, weights, torch.zeros_like(weights))
                loss = F.cross_entropy(logits_flat, labels_flat, weight=weights)
            else:
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
        Compute F1 score.

        - For binary tasks (detect/unit_detect): F1 for the positive class (1), returned as 0-100%.
        - For multi-class classify: macro-F1 across all classes (including silence=0), returned as 0-100%.
        
        Args:
            logits: (B, W_patches, 1) for binary or (B, W_patches, num_classes) for multi-class
            labels: (B, W) where W is n_timebins
            
        Returns:
            f1: F1 score (0-100%)
        """
        B_label, W = labels.shape
        
        # Downsample labels to match patch width
        labels_downsampled = labels[:, ::self.patch_width]

        if self.num_classes == 2:
            # Binary classification: use sigmoid + threshold at 0.5
            # logits shape: (B, W_patches, 1)
            logits_flat = logits.reshape(-1)  # (B * W_patches,)
            probs = torch.sigmoid(logits_flat)  # (B * W_patches,)
            preds = (probs > 0.5).long()  # (B * W_patches,)
            labels_flat = labels_downsampled.reshape(-1)  # (B * W_patches,)

            # Calculate TP, FP, FN for positive class 1
            tp = ((preds == 1) & (labels_flat == 1)).sum().item()
            fp = ((preds == 1) & (labels_flat == 0)).sum().item()
            fn = ((preds == 0) & (labels_flat == 1)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            return f1 * 100.0

        # Multi-class macro-F1 (include silence=0)
        preds = torch.argmax(logits, dim=-1)  # (B, W_patches)
        labels_mc = labels_downsampled  # (B, W_patches)

        f1s = []
        for c in range(int(self.num_classes)):
            tp = ((preds == c) & (labels_mc == c)).sum().item()
            fp = ((preds == c) & (labels_mc != c)).sum().item()
            fn = ((preds != c) & (labels_mc == c)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1s.append(f1)

        macro_f1 = float(np.mean(f1s)) if len(f1s) else 0.0
        return macro_f1 * 100.0


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")


class Trainer():
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.compute_f1 = config["mode"] in ["detect", "unit_detect"] or config.get("log_f1", False)
        
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
        
        # Initialize LR scheduler (optional warmup + decay to min_lr)
        warmup_steps = int(config.get("warmup_steps") or 0)
        min_lr = float(config.get("min_lr") or 0.0)
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0. Got {warmup_steps}")
        if min_lr < 0:
            raise ValueError(f"min_lr must be >= 0. Got {min_lr}")

        if warmup_steps > 0 or min_lr > 0.0:
            total_steps = int(config["steps"])
            base_lr = float(config["lr"])
            decay_steps = max(1, total_steps - warmup_steps)

            def lr_lambda(step_idx):
                # LambdaLR passes 0 on the first scheduler.step() call.
                step_num = step_idx + 1
                if warmup_steps > 0 and step_num <= warmup_steps:
                    return step_num / float(warmup_steps)
                # Cosine decay from base_lr to min_lr
                decay_step = step_num - warmup_steps
                decay_step = min(max(decay_step, 0), decay_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * decay_step / float(decay_steps)))
                target_lr = min_lr + (base_lr - min_lr) * cosine
                return target_lr / base_lr if base_lr > 0 else 1.0

            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        else:
            # Default behavior: constant learning rate (no scheduler)
            self.scheduler = None
        
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
        
        # Early stopping state (initialized in train())
        self._es_ema_val_loss = None
        self._es_best_ema_val_loss = None
        self._es_bad_evals = 0

        # Setup loss logging file
        self.loss_log_path = os.path.join(self.run_path, "loss_log.txt")
        with open(self.loss_log_path, 'w') as f:
            if self.compute_f1:
                f.write("step,train_loss,val_loss,train_acc,val_acc,train_f1,val_f1,samples_processed,steps_per_sec,samples_per_sec\n")
            else:
                f.write("step,train_loss,val_loss,train_acc,val_acc,samples_processed,steps_per_sec,samples_per_sec\n")
    
    def export_validation_outputs(self, val_loader, step_num, final_weight_path=None):
        """
        Export validation logits + labels + filenames so metrics can be computed posthoc.

        Runs inference once at the end of training (optionally reloading the final saved weights),
        then saves consolidated arrays to: runs/<run_name>/val_outputs/

        Outputs:
          - logits.npy: float32, shape (N_windows, W_patches, C_or_1)
          - labels_timebins.npy: int64, shape (N_windows, W_timebins)
          - labels_patches.npy: int64, shape (N_windows, W_patches)
          - window_starts.npy: int64, shape (N_windows,)   (start timebin in the original file)
          - window_lengths.npy: int64, shape (N_windows,)  (unpadded length in timebins, <= W_timebins)
          - filenames.json: list[str], length N_windows (filename repeated per window)
          - meta.json
        """
        out_dir = Path(self.run_path) / "val_outputs"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        from utils import parse_chunk_ms, clip_labels_to_chunk

        # Ensure we're exporting logits from the final saved weights (not an in-memory intermediate state)
        if final_weight_path is not None:
            state_dict = torch.load(final_weight_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

        # IMPORTANT: we do NOT want a single random crop to represent a whole file.
        # Export uses deterministic full-file coverage by tiling each file into sequential windows.
        ds = val_loader.dataset
        window_timebins = int(self.config["num_timebins"])
        if window_timebins <= 0:
            raise ValueError(f"num_timebins must be > 0 for export. Got {window_timebins}")

        meta = {
            "run_name": self.config.get("run_name"),
            "mode": self.config.get("mode"),
            "num_classes": int(getattr(self.model, "num_classes", -1)),
            "patch_width": int(self.config.get("patch_width")),
            "n_timebins": int(self.config.get("num_timebins")),
            "step_num": int(step_num),
            "final_weight_path": str(final_weight_path) if final_weight_path is not None else None,
            "export_strategy": "tiled_full_file",
            "export_stride_timebins": int(window_timebins),
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        self.model.eval()
        patch_width = int(self.config["patch_width"])
        w_timebins = int(window_timebins)
        w_patches = w_timebins // patch_width

        # Infer output channels from a single sample to keep export inference low-memory
        first_batch = next(iter(val_loader))
        x0 = first_batch[0][:1].to(self.device, non_blocking=True)
        with torch.no_grad():
            logits0 = self.model(x0)
        c_out = int(logits0.shape[-1])

        # Precompute all windows across all files (deterministic, full coverage)
        window_index = []  # (path, filename, start, length, total_timebins)
        for path in ds.file_dirs:
            filename = path.stem
            arr = np.load(path, mmap_mode="r")
            total_t = int(arr.shape[1])
            # starts at 0, stride = window length, include final partial window (padded)
            start = 0
            while start < total_t:
                length = min(w_timebins, total_t - start)
                window_index.append((path, filename, int(start), int(length), int(total_t)))
                start += w_timebins
            if total_t == 0:
                # Degenerate case: still emit a single all-pad window
                window_index.append((path, filename, 0, 0, 0))

        n_windows = len(window_index)

        logits_mm = np.lib.format.open_memmap(
            out_dir / "logits.npy", mode="w+", dtype=np.float32, shape=(n_windows, w_patches, c_out)
        )
        labels_time_mm = np.lib.format.open_memmap(
            out_dir / "labels_timebins.npy", mode="w+", dtype=np.int64, shape=(n_windows, w_timebins)
        )
        labels_patches_mm = np.lib.format.open_memmap(
            out_dir / "labels_patches.npy", mode="w+", dtype=np.int64, shape=(n_windows, w_patches)
        )
        window_starts_mm = np.lib.format.open_memmap(
            out_dir / "window_starts.npy", mode="w+", dtype=np.int64, shape=(n_windows,)
        )
        window_lengths_mm = np.lib.format.open_memmap(
            out_dir / "window_lengths.npy", mode="w+", dtype=np.int64, shape=(n_windows,)
        )
        filenames_all = []  # repeated per window

        # Always use batch size 1 for export inference to minimize GPU memory
        batch_size = 1

        with torch.no_grad():
            write_idx = 0
            for i in range(0, n_windows, batch_size):
                batch_windows = window_index[i:i + batch_size]
                bsz = len(batch_windows)

                specs_np = np.zeros((bsz, 1, int(ds.n_mels), w_timebins), dtype=np.float32)
                labels_np = np.zeros((bsz, w_timebins), dtype=np.int64)
                starts_np = np.zeros((bsz,), dtype=np.int64)
                lengths_np = np.zeros((bsz,), dtype=np.int64)
                fns = []

                for j, (path, filename, start, length, total_t) in enumerate(batch_windows):
                    arr = np.load(path, mmap_mode="r")  # (mels, time)
                    # create labels for the full file, then slice the window
                    base_filename, chunk_start_ms, chunk_end_ms = parse_chunk_ms(filename)
                    labels = ds._label_index.get(base_filename)
                    if labels is None:
                        raise ValueError(f"No matching recording found for: {base_filename}")
                    labels = clip_labels_to_chunk(labels, chunk_start_ms, chunk_end_ms)
                    full_labels = ds.create_label_array(labels, 0, int(arr.shape[1]))

                    # slice + pad if needed
                    end = start + length
                    if length > 0:
                        window = np.array(arr[:, start:end], dtype=np.float32)
                        lab_w = np.array(full_labels[start:end], dtype=np.int64)
                    else:
                        window = np.zeros((int(ds.n_mels), 0), dtype=np.float32)
                        lab_w = np.zeros((0,), dtype=np.int64)

                    if length < w_timebins:
                        pad = w_timebins - length
                        window = np.pad(window, ((0, 0), (0, pad)), mode="constant")
                        lab_w = np.pad(lab_w, (0, pad), mode="constant", constant_values=0)

                    # normalize like the dataset does (no noise in val)
                    window -= ds.mean
                    window /= ds.std

                    specs_np[j, 0] = window
                    labels_np[j] = lab_w
                    starts_np[j] = start
                    lengths_np[j] = length
                    fns.append(filename)

                x = torch.from_numpy(specs_np).to(self.device, non_blocking=True)
                labels_t = torch.from_numpy(labels_np).to(self.device, non_blocking=True)
                logits_t = self.model(x)
                labels_patches_t = labels_t[:, ::patch_width]

                logits_out = logits_t.detach().cpu().numpy().astype(np.float32, copy=False)
                labels_time_out = labels_t.detach().cpu().numpy().astype(np.int64, copy=False)
                labels_patches_out = labels_patches_t.detach().cpu().numpy().astype(np.int64, copy=False)

                logits_mm[write_idx:write_idx + bsz] = logits_out
                labels_time_mm[write_idx:write_idx + bsz] = labels_time_out
                labels_patches_mm[write_idx:write_idx + bsz] = labels_patches_out
                window_starts_mm[write_idx:write_idx + bsz] = starts_np
                window_lengths_mm[write_idx:write_idx + bsz] = lengths_np
                filenames_all.extend(fns)
                write_idx += bsz

        with open(out_dir / "filenames.json", "w") as f:
            json.dump(filenames_all, f, indent=2)

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
            grad_norm: L2 norm of gradients for the step (training only)
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
                    f1 = self.model.compute_f1_score(logits, labels) if self.compute_f1 else None
            else:
                logits = self.model(x)
                loss = self.model.compute_loss(logits, labels)
                accuracy = self.model.compute_accuracy(logits, labels)
                f1 = self.model.compute_f1_score(logits, labels) if self.compute_f1 else None
        
        grad_norm = None
        # Backward pass only for training
        if is_training:
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()

            grad_clip = float(self.config.get("grad_clip", 0.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            grads = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grads.append(param.grad.detach().norm(2))
            if grads:
                grad_norm = float(torch.norm(torch.stack(grads), 2).item())

            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        
        return loss.item(), accuracy, f1, logits, grad_norm
    
    def save_prediction_visualization(self, batch, logits, step_num, split="val"):
        """
        Save visualization showing spectrogram, predictions, and ground truth.
        
        Args:
            batch: Input batch (spectrograms, labels, filenames)
            logits: Model predictions (B, W_patches, 1) for binary or (B, W_patches, num_classes) for multi-class
            step_num: Current training step
            split: "train" or "val" label for the plot filename/title
        """
        from plotting_utils import save_supervised_prediction_plot
        
        spectrograms, labels, filenames = batch
        
        # Select a sample from the batch (cycle through to show variety)
        batch_size = spectrograms.shape[0]
        sample_idx = (step_num // self.config["eval_every"]) % batch_size
        
        # Move to CPU and convert to numpy
        spec = spectrograms[sample_idx, 0].cpu().numpy()  # Selected sample, remove channel dim
        labels_np = labels[sample_idx].cpu().numpy()  # (W,)
        logits_np = logits[sample_idx].detach().cpu().numpy()  # (W_patches, 1) or (W_patches, num_classes)
        
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
            probabilities=probs if self.config["mode"] in ["detect", "unit_detect"] else None,
            logits=logits_np,
            filename=filenames[sample_idx],
            mode=self.config["mode"],
            num_classes=self.model.num_classes,
            output_dir=self.imgs_path,
            step_num=step_num,
            split=split,
        )
    
    def train(self):
        from data_loader import SupervisedSpectogramDataset
        from torch.utils.data import DataLoader
        
        # Initialize datasets
        train_dataset = SupervisedSpectogramDataset(
            dir=self.config["train_dir"],
            annotation_file_path=self.config["annotation_file"],
            n_timebins=self.config["num_timebins"],
            mode=self.config["mode"],
            white_noise=self.config.get("white_noise", 0.0),
            audio_params_override=self.config.get("audio_params_override"),
        )
        
        val_dataset = SupervisedSpectogramDataset(
            dir=self.config["val_dir"],
            annotation_file_path=self.config["annotation_file"],
            n_timebins=self.config["num_timebins"],
            mode=self.config["mode"],
            white_noise=0.0,  # No augmentation on validation set
            audio_params_override=self.config.get("audio_params_override"),
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
        
        val_batch_size = int(self.config.get("val_batch_size", 0))
        if val_batch_size <= 0:
            val_batch_size = int(self.config["batch_size"])
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=True
        )
        
        # Training loop
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)
        
        total_steps = self.config["steps"]

        # Early stopping config:
        # Stop if EMA-smoothed val_loss does not improve for N consecutive eval checks.
        # Set early_stop_patience=0 to disable.
        es_patience = int(self.config.get("early_stop_patience", 8))
        es_alpha = float(self.config.get("early_stop_ema_alpha", 0.9))
        es_min_delta = float(self.config.get("early_stop_min_delta", 0.0))
        if not (0.0 <= es_alpha < 1.0):
            raise ValueError(f"early_stop_ema_alpha must be in [0, 1). Got {es_alpha}")
        if es_patience < 0:
            raise ValueError(f"early_stop_patience must be >= 0. Got {es_patience}")
        
        # Initialize timing
        last_eval_time = time.time()
        last_eval_step = 0
        
        print(f"Training for {total_steps} steps")
        
        last_step_num = -1
        for step_num in range(total_steps):
            last_step_num = step_num
            try:
                train_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                train_batch = next(train_iter)
            
            # Training step
            train_loss, train_acc, train_f1, train_logits, train_grad_norm = self.step(train_batch, is_training=True)
            
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
                val_loss, val_acc, val_f1, val_logits, _ = self.step(val_batch, is_training=False)

                # Early stopping update (EMA-smoothed val loss)
                if es_patience > 0:
                    if self._es_ema_val_loss is None:
                        self._es_ema_val_loss = float(val_loss)
                        self._es_best_ema_val_loss = float(val_loss)
                        self._es_bad_evals = 0
                    else:
                        self._es_ema_val_loss = (es_alpha * self._es_ema_val_loss) + ((1.0 - es_alpha) * float(val_loss))

                    if self._es_best_ema_val_loss is None:
                        self._es_best_ema_val_loss = float(self._es_ema_val_loss)

                    if float(self._es_ema_val_loss) < (float(self._es_best_ema_val_loss) - es_min_delta):
                        self._es_best_ema_val_loss = float(self._es_ema_val_loss)
                        self._es_bad_evals = 0
                    else:
                        self._es_bad_evals += 1
                
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
                if self.scheduler is not None:
                    current_lr = self.scheduler.get_last_lr()[0]
                else:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                grad_str = f", Grad Norm = {train_grad_norm:.4f}" if train_grad_norm is not None else ""
                if self.config["mode"] in ["detect", "unit_detect"] or self.config.get("log_f1", False):
                    # train_f1 / val_f1 are expected to be non-None when log_f1 is enabled
                    print(f"Step {step_num} ({progress_pct:.1f}%): "
                          f"Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, "
                          f"Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%, "
                          f"Train F1 = {train_f1:.2f}%, Val F1 = {val_f1:.2f}%, "
                          f"Samples = {samples_processed}, "
                          f"LR = {current_lr:.2e}{grad_str}, "
                          f"Steps/sec = {steps_per_sec:.2f}, "
                          f"Samples/sec = {samples_per_sec:.1f}")
                else:
                    print(f"Step {step_num} ({progress_pct:.1f}%): "
                          f"Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, "
                          f"Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%, "
                          f"Samples = {samples_processed}, "
                          f"LR = {current_lr:.2e}{grad_str}, "
                          f"Steps/sec = {steps_per_sec:.2f}, "
                          f"Samples/sec = {samples_per_sec:.1f}")
                
                # Log losses and accuracies to file
                with open(self.loss_log_path, 'a') as f:
                    if self.config["mode"] in ["detect", "unit_detect"] or self.config.get("log_f1", False):
                        f.write(f"{step_num},{train_loss:.6f},{val_loss:.6f},{train_acc:.2f},{val_acc:.2f},{train_f1:.2f},{val_f1:.2f},{samples_processed},{steps_per_sec:.2f},{samples_per_sec:.1f}\n")
                    else:
                        f.write(f"{step_num},{train_loss:.6f},{val_loss:.6f},{train_acc:.2f},{val_acc:.2f},{samples_processed},{steps_per_sec:.2f},{samples_per_sec:.1f}\n")
                
                # Save intermediate model weights (optional; final checkpoint is always saved at end)
                if self.config.get("save_intermediate_checkpoints", True):
                    weight_path = os.path.join(self.weights_path, f"model_step_{step_num:06d}.pth")
                    torch.save(self.model.state_dict(), weight_path)
                
                # Save prediction visualization
                self.save_prediction_visualization(val_batch, val_logits, step_num, split="val")
                self.save_prediction_visualization(train_batch, train_logits, step_num, split="train")

                # Potentially stop training
                if es_patience > 0 and self._es_bad_evals >= es_patience:
                    print(
                        "Early stopping: EMA-smoothed validation loss did not improve for "
                        f"{es_patience} consecutive validation checks "
                        f"(ema_val_loss={self._es_ema_val_loss:.6f}, best_ema_val_loss={self._es_best_ema_val_loss:.6f}). "
                        "Halting training."
                    )
                    break
        
        # Save final model weights
        final_step = last_step_num if last_step_num >= 0 else 0
        final_weight_path = os.path.join(self.weights_path, f"model_step_{final_step:06d}.pth")
        torch.save(self.model.state_dict(), final_weight_path)

        # Export validation outputs by default (can be disabled via CLI)
        if self.config.get("save_val_logits", True):
            print("Exporting validation logits/labels for posthoc metrics...")
            self.export_validation_outputs(val_loader, final_step, final_weight_path=final_weight_path)
        
        print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="supervised training args")
    
    # Required arguments
    parser.add_argument("--train_dir", type=str, required=True, help="training directory")
    parser.add_argument("--val_dir", type=str, required=True, help="validation directory")
    parser.add_argument("--run_name", type=str, required=True, help="directory name inside /runs to store train run details")
    parser.add_argument("--pretrained_run", type=str, required=True, help="path to pretrained run directory")
    parser.add_argument("--annotation_file", type=str, required=True, help="path to annotation JSON file")
    parser.add_argument("--mode", type=str, required=True, choices=["detect", "unit_detect", "classify"], help="detect, unit_detect, or classify mode")
    
    # Training hyperparameters
    parser.add_argument("--steps", type=int, default=50_000, help="number of training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=0,
        help="validation/inference batch size (0 uses --batch_size)",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="number of DataLoader worker processes")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay")
    parser.add_argument("--eval_every", type=int, default=25, help="evaluate every N steps")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="linear warmup steps before decay (omit to disable scheduler)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=None,
        help="minimum learning rate after decay (omit to disable scheduler)",
    )

    # Early stopping
    parser.add_argument("--early_stop_patience", type=int, default=8, help="stop if EMA-smoothed val_loss does not improve for N consecutive eval checks (0 disables)")
    parser.add_argument("--early_stop_ema_alpha", type=float, default=0.9, help="EMA alpha for smoothing val_loss (higher = smoother)")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0, help="minimum decrease in EMA val_loss required to count as improvement")

    # Export val logits/labels for posthoc metrics (default: on)
    parser.add_argument(
        "--save_val_logits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save validation logits/labels/filenames to runs/<run_name>/val_outputs/ (default: enabled)",
    )

    # Checkpointing
    parser.add_argument(
        "--save_intermediate_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save intermediate checkpoints during training at eval steps (final checkpoint is always saved)",
    )
    
    # Model configuration
    parser.add_argument("--freeze_encoder", action="store_true", help="freeze encoder weights (train classifier only)")
    parser.add_argument(
        "--freeze_encoder_up_to",
        type=int,
        default=None,
        help="freeze encoder layers up to this index (inclusive); negative allowed",
    )
    parser.add_argument("--lora_rank", type=int, default=0, help="LoRA rank for encoder FFN (0 disables)")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout before low-rank projection")
    parser.add_argument("--linear_probe", action="store_true", help="use single linear layer instead of MLP")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable automatic mixed precision training (default: enabled)",
    )
    parser.add_argument("--grad_clip", type=float, default=5.0, help="clip gradient norm (0 disables)")
    parser.add_argument("--encoder_layer_idx", type=int, default=None, help="encoder layer index to probe (0..enc_n_layer-1). If omitted, uses full encoder output.")
    parser.add_argument("--log_f1", action="store_true", help="log (macro) F1 to loss_log.txt (useful for classify)")
    parser.add_argument(
        "--class_weighting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="weight CE by inverse class frequency per batch (classify only; default: enabled)",
    )
    
    # Data augmentation
    parser.add_argument("--white_noise", type=float, default=0.0, help="standard deviation of white noise to add after normalization (0.0 = no noise)")
    
    args = parser.parse_args()
    
    # Load pretrained model and config
    from utils import load_model_from_checkpoint, get_num_classes_from_annotations, load_audio_params
    from pretrain import resolve_run_path
    
    pretrained_path = resolve_run_path(args.pretrained_run)
    pretrained_model, pretrained_config = load_model_from_checkpoint(pretrained_path, fallback_to_random=False)
    
    print(f"Loaded pretrained model from: {pretrained_path}")
    
    # Create supervised config
    config = vars(args)
    if config.get("warmup_steps") is None:
        config.pop("warmup_steps", None)
    if config.get("min_lr") is None:
        config.pop("min_lr", None)
    
    # Add necessary info from pretrained config
    config["num_timebins"] = pretrained_config["num_timebins"]
    config["patch_height"] = pretrained_config["patch_height"]
    config["patch_width"] = pretrained_config["patch_width"]
    config["enc_hidden_d"] = pretrained_config["enc_hidden_d"]
    config["mels"] = pretrained_config["mels"]
    config["pretrained_run"] = pretrained_path

    # Use audio params (mean/std, sr/hop_size, mels) from the pretrain run for supervised scaling
    config["audio_params_override"] = load_audio_params(pretrained_path)
    config["audio_params_source"] = pretrained_path
    
    # Automatically determine number of classes from annotations
    num_classes = get_num_classes_from_annotations(config["annotation_file"], config["mode"])
    config["num_classes"] = num_classes
    
    # Create supervised model
    supervised_model = SupervisedTinyBird(
        pretrained_model=pretrained_model,
        config=config,
        num_classes=num_classes,
        freeze_encoder=config["freeze_encoder"],
        freeze_encoder_up_to=config.get("freeze_encoder_up_to", None),
        mode=config["mode"],
        linear_probe=config.get("linear_probe", False),
        lora_rank=config.get("lora_rank", 0),
        lora_alpha=config.get("lora_alpha", 16.0),
        lora_dropout=config.get("lora_dropout", 0.0),
    )
    
    # Create trainer and train
    trainer = Trainer(supervised_model, config)
    trainer.train()
