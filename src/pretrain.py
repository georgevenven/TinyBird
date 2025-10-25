import argparse
import os
import json
import shutil
import warnings
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import TinyBird
import wandb
import psutil
import random
from utils import load_model_from_checkpoint, count_parameters
from data_loader import SpectogramDataset

import contextlib
import time

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    i = 0
    while x >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {units[i]}"


@contextlib.contextmanager
def cuda_mem_scope(device, step=None):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    try:
        yield
    finally:
        dur = time.time() - start
        if torch.cuda.is_available():
            alloc = torch.cuda.max_memory_allocated(device)
            resv = torch.cuda.max_memory_reserved(device)
            free_b, total_b = torch.cuda.mem_get_info()
            print(
                f"alloc={human_bytes(alloc)}  reserved={human_bytes(resv)}  "
                f"free={human_bytes(free_b)}  total={human_bytes(total_b)}  dur={dur:.3f}s"
            )
            # also log to W&B
            try:
                payload = {
                    "cuda/peak_alloc": int(alloc),
                    "cuda/peak_reserved": int(resv),
                    "cuda/free": int(free_b),
                    "cuda/total": int(total_b),
                }
                wandb.log(payload, step=int(step) if step is not None else None, commit=False)
            except Exception:
                pass


def log_batch_shapes(tag, step_num, **kwargs):
    # Print all key=value pairs on one line
    print(", ".join(f"{k}={v}" for k, v in kwargs.items()))

    try:
        import wandb

        # Log only numeric values to wandb
        log_dict = {f"{tag}/{k}": v for k, v in kwargs.items() if isinstance(v, (int, float))}
        if log_dict:
            wandb.log(log_dict, step=int(step_num) if step_num is not None else None, commit=False)
    except Exception:
        pass


def dump_cuda_summary():
    if torch.cuda.is_available():
        try:
            print(torch.cuda.memory_summary())
        except Exception as e:
            print(f"[cuda] memory_summary failed: {e}")


class Trainer:
    def __init__(self, config, pretrained_model=None):
        self.config = config

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Handle continue mode vs new training
        if config.get('is_continuing', False):
            # Continue training mode - use existing run directory
            continue_from = config['continue_from']
            if os.path.isabs(continue_from):
                self.run_path = continue_from
            else:
                runs_base = os.path.join("..", "runs")
                self.run_path = os.path.join(runs_base, continue_from)

            if not os.path.exists(self.run_path):
                raise FileNotFoundError(f"Continue directory not found: {self.run_path}")

            print(f"Continuing training from: {self.run_path}")

        else:
            # New training mode - setup run directory
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

        # Save config as JSON (only for new runs)
        if not config.get('is_continuing', False):
            config_path = os.path.join(self.run_path, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            # Save audio processing parameters
            audio_params = {
                "sr": 32000,  # sample rate
                "mels": config["mels"],  # number of mel bins
                "hop_size": 160,  # hop length
                "n_fft": 1024,  # FFT size
            }
            audio_params_path = os.path.join(self.run_path, "audio_params.json")
            with open(audio_params_path, 'w') as f:
                json.dump(audio_params, f, indent=2)

        # --- Weights & Biases initialization (Step 1) ---
        run_name = self.config.get("run_name") or os.path.basename(self.run_path.rstrip(os.sep))
        wandb.init(
            project=self.config.get("wandb_project", "tinybird"),
            entity=self.config.get("wandb_entity", None),
            name=run_name,
            mode=self.config.get("wandb_mode", "online"),
            config=self.config,
        )
        self.wandb_run = wandb.run
        # Ensure nice metric plots (x-axis = step)
        wandb.define_metric("step")
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("val/*", step_metric="step")
        wandb.define_metric("lr", step_metric="step")
        wandb.define_metric("cuda/*", step_metric="step")
        wandb.define_metric("system/*", step_metric="step")
        wandb.define_metric("recon/*", step_metric="step")

        # --- End W&B initialization ---

        # Memory reporting (one-time, after first training step)
        self._mem_reported = False
        self._process = psutil.Process(os.getpid())

        def _human_bytes(n: int) -> str:
            units = ["B", "KB", "MB", "GB", "TB"]
            i = 0
            x = float(n)
            while x >= 1024 and i < len(units) - 1:
                x /= 1024.0
                i += 1
            return f"{x:.2f} {units[i]}"

        self._human_bytes = _human_bytes

        # Initialize model
        if pretrained_model is not None:
            # Use the loaded model from continue mode
            self.tinybird = pretrained_model.to(self.device).float()
            print("Using loaded model from checkpoint")
        else:
            # Initialize new model
            self.tinybird = TinyBird(config).to(self.device).float()
            print("Initialized new model")

        if self.config.get("detect_anomaly", False):
            torch.autograd.set_detect_anomaly(True)
            print("Anomaly detection enabled")

        # Watch model for gradients/param histograms (lightweight frequency)
        wandb.watch(self.tinybird, log="gradients", log_freq=200)
        # Print parameter counts
        count_parameters(self.tinybird)

        # Initialize optimizer
        self.optimizer = AdamW(self.tinybird.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

        # Initialize cosine annealing scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config["steps"])

        # Initialize AMP scaler if AMP is enabled
        self.use_amp = bool(config.get("amp", False)) and torch.cuda.is_available()

        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # Loss tracking
        self.ema_train_loss = None
        self.ema_val_loss = None
        self.ema_alpha = 0.99

        # Loss history for plotting
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_steps = []
        self.val_steps = []

        # Initialize step tracking
        self.starting_step = 0
        if config.get('is_continuing', False):
            # Load existing step count and loss history
            self._load_training_state()

        # Setup loss logging file
        self.loss_log_path = os.path.join(self.run_path, "loss_log.txt")
        if not config.get('is_continuing', False):
            # Create new loss log for new training
            with open(self.loss_log_path, 'w') as f:
                f.write("step,train_loss,ema_train_loss,val_loss,ema_val_loss\n")
        else:
            # Verify loss log exists for continuing training
            if not os.path.exists(self.loss_log_path):
                print(f"Warning: Loss log not found at {self.loss_log_path}, starting fresh")

    def _load_training_state(self):
        """Load training state from existing run for continuing training."""
        # Load loss history from loss log
        if os.path.exists(self.loss_log_path):
            try:
                # Read CSV manually to avoid pandas dependency
                with open(self.loss_log_path, 'r') as f:
                    lines = f.readlines()[1:]  # Skip header

                if lines:
                    # Parse the last line to get the last step
                    last_line = lines[-1].strip().split(',')
                    last_step = int(last_line[0])
                    self.starting_step = last_step + self.config.get('eval_every', 500)

                    # Load all loss history
                    steps = []
                    train_losses = []
                    val_losses = []
                    ema_train_losses = []
                    ema_val_losses = []

                    for line in lines:
                        parts = line.strip().split(',')
                        if len(parts) >= 5:
                            steps.append(int(parts[0]))
                            train_losses.append(float(parts[1]))
                            ema_train_losses.append(float(parts[2]))
                            val_losses.append(float(parts[3]))
                            ema_val_losses.append(float(parts[4]))

                    # Store loss history
                    self.train_steps = steps
                    self.train_loss_history = train_losses
                    self.val_steps = steps
                    self.val_loss_history = val_losses

                    # Set EMA losses to last values
                    if ema_train_losses and ema_val_losses:
                        self.ema_train_loss = ema_train_losses[-1]
                        self.ema_val_loss = ema_val_losses[-1]

                    print(f"Loaded training state. Continuing from step {self.starting_step}")
                    print(f"Previous EMA train loss: {self.ema_train_loss:.6f}")
                    print(f"Previous EMA val loss: {self.ema_val_loss:.6f}")

                    # Advance scheduler to correct step
                    for _ in range(self.starting_step):
                        self.scheduler.step()
                else:
                    print("Loss log file is empty, starting from step 0")
            except Exception as e:
                print(f"Error loading training state: {e}")
                print("Starting from step 0")
        else:
            print("No loss log found, starting from step 0")

    # --- Metrics & logging helpers -------------------------------------------------
    # ---------- Metrics & logging helpers (modular & concise) ----------

    def _valid_label_eval_tensors(self, logits_label, x_l, bool_mask, W):
        """
        Returns y_true, y_pred restricted to masked, non-separator columns.
        Also returns 'acc' over those columns and a 'has_valid' flag.
        """
        with torch.no_grad():
            B, T, C = logits_label.shape
            H = T // W  # assumes patch_width == 1
            # Per-column logits → predictions
            preds_col = logits_label.view(B, H, W, C).mean(dim=1).argmax(dim=-1)  # (B, W)

            # Valid columns: masked anywhere & not a separator in ground truth
            masked_cols = bool_mask.view(B, H, W).any(dim=1)  # (B, W)
            is_sep = x_l == self.tinybird.sep_class_id  # (B, W)
            valid = masked_cols & (~is_sep)
            if not valid.any():
                return None, None, float("nan"), False

            y_true = x_l[valid].long()
            y_pred = preds_col[valid].long()
            acc = (y_pred == y_true).float().mean().item()
            return y_true, y_pred, acc, True

    def _compute_label_metrics(self, logits_label, x_l, bool_mask, W):
        """
        Only the metrics we care about:
        - acc: overall accuracy on valid columns
        - pct_true1_pred1: P(pred=1 | true=1)
        - pct_true0_pred0: P(pred=0 | true=0)

        Also returns y_true/y_pred for W&B's confusion-matrix plot, plus internal counts
        used to do weighted averaging across many batches. These counts are NOT logged.
        """
        y_true, y_pred, acc, has_valid = self._valid_label_eval_tensors(logits_label, x_l, bool_mask, W)
        metrics = {
            "has_valid": has_valid,
            "acc": float("nan") if not has_valid else acc,
            "pct_true1_pred1": float("nan"),
            "pct_true0_pred0": float("nan"),
            "y_true": None,
            "y_pred": None,
            # internal counts for aggregation (not logged)
            "_n_valid": 0,
            "_n_true1": 0,
            "_n_true0": 0,
            "_n_correct": 0,
            "_n_correct1": 0,
            "_n_correct0": 0,
        }
        if not has_valid:
            return metrics

        mask_true1 = y_true == 1
        mask_true0 = y_true == 0

        n_true1 = int(mask_true1.sum().item())
        n_true0 = int(mask_true0.sum().item())
        n_valid = int(y_true.numel())
        n_correct = int((y_pred == y_true).sum().item())
        n_correct1 = int((y_pred[mask_true1] == 1).sum().item()) if n_true1 > 0 else 0
        n_correct0 = int((y_pred[mask_true0] == 0).sum().item()) if n_true0 > 0 else 0

        pct_true1_pred1 = n_correct1 / n_true1 if n_true1 > 0 else float("nan")
        pct_true0_pred0 = n_correct0 / n_true0 if n_true0 > 0 else float("nan")

        metrics.update(
            {
                "pct_true1_pred1": pct_true1_pred1,
                "pct_true0_pred0": pct_true0_pred0,
                "y_true": y_true.detach().cpu(),
                "y_pred": y_pred.detach().cpu(),
                "_n_valid": n_valid,
                "_n_true1": n_true1,
                "_n_true0": n_true0,
                "_n_correct": n_correct,
                "_n_correct1": n_correct1,
                "_n_correct0": n_correct0,
            }
        )
        return metrics

    def _log_losses_and_metrics(self, tag, step_num, loss_total, loss_recon, loss_label, metrics):
        """
        Log scalar losses + requested percentages; keep the W&B confusion-matrix plot.
        """
        log_batch_shapes(
            f"{tag}",
            step_num,
            loss_total=float(loss_total),
            loss_recon=float(loss_recon),
            loss_label=float(loss_label),
            label_acc=float(metrics["acc"]) if metrics["acc"] == metrics["acc"] else 0.0,
            pct_true1_pred1=float(metrics["pct_true1_pred1"])
            if metrics["pct_true1_pred1"] == metrics["pct_true1_pred1"]
            else 0.0,
            pct_true0_pred0=float(metrics["pct_true0_pred0"])
            if metrics["pct_true0_pred0"] == metrics["pct_true0_pred0"]
            else 0.0,
        )

        # Confusion matrix (pretty plot only; no raw counts)
        try:
            if metrics["has_valid"] and metrics["y_true"] is not None:
                wandb.log(
                    {
                        f"{tag}/label_confmat": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=metrics["y_true"].tolist(),
                            preds=metrics["y_pred"].tolist(),
                            class_names=["class0", "class1"],
                        )
                    },
                    step=int(step_num),
                    commit=False,
                )
        except Exception:
            pass

    def _aggregate_val_metrics(self, items):
        """
        items: list of dicts, each with keys:
        loss_total, loss_recon, loss_lbl, metrics (with internal *_count fields)
        Returns: (loss_total_avg, loss_recon_avg, loss_lbl_avg, agg_metrics)
        """
        if not items:
            return float('nan'), float('nan'), float('nan'), {
                "has_valid": False, "acc": float('nan'),
                "pct_true1_pred1": float('nan'), "pct_true0_pred0": float('nan'),
                "y_true": None, "y_pred": None
            }

        # Loss averages
        loss_total_avg = float(torch.tensor([it["loss_total"] for it in items]).mean().item())
        loss_recon_avg = float(torch.tensor([it["loss_recon"] for it in items]).mean().item())
        loss_lbl_avg   = float(torch.tensor([it["loss_lbl"]   for it in items]).mean().item())

        # Counts for weighted averages
        n_valid   = sum(it["metrics"]["_n_valid"]   for it in items)
        n_correct = sum(it["metrics"]["_n_correct"] for it in items)
        n_true1   = sum(it["metrics"]["_n_true1"]   for it in items)
        n_true0   = sum(it["metrics"]["_n_true0"]   for it in items)
        n_corr1   = sum(it["metrics"]["_n_correct1"] for it in items)
        n_corr0   = sum(it["metrics"]["_n_correct0"] for it in items)

        acc = (n_correct / n_valid) if n_valid > 0 else float('nan')
        pct_true1_pred1 = (n_corr1 / n_true1) if n_true1 > 0 else float('nan')
        pct_true0_pred0 = (n_corr0 / n_true0) if n_true0 > 0 else float('nan')


        # For the W&B confusion matrix, aggregate across batches with a cap to keep payload small
        y_true_list, y_pred_list = [], []
        cap = 10_000  # max points to include in the plot
        for it in items:
            yt = it["metrics"]["y_true"]
            yp = it["metrics"]["y_pred"]
            if yt is None or yp is None:
                continue
            take = min(cap - len(y_true_list), len(yt))
            if take <= 0:
                break
            y_true_list.extend(yt[:take].tolist())
            y_pred_list.extend(yp[:take].tolist())

        agg_metrics = {
            "has_valid": n_valid > 0,
            "acc": acc,
            "pct_true1_pred1": pct_true1_pred1,
            "pct_true0_pred0": pct_true0_pred0,
            "y_true": torch.tensor(y_true_list) if y_true_list else None,
            "y_pred": torch.tensor(y_pred_list) if y_pred_list else None,
        }
        return loss_total_avg, loss_recon_avg, loss_lbl_avg, agg_metrics


    def _forward_encode_decode(self, x, x_i, N, x_l, step_num, is_training: bool, log_metrics: bool = True):
        """
        Runs the full forward path (data shaping → encoder/decoder → losses → metrics),
        logs batch shapes + losses/metrics, and returns losses for the caller.

        Returns:
            loss_total, loss_recon, loss_lbl, H, W
        """
        # --- data shaping pipeline ---
        x, x_i, x_l = self.tinybird.compactify_data(x, x_i, N, xl=x_l)

        if is_training:
            x_l = self.tinybird.randomize_label(x_l)
            x, x_i, x_l = self.tinybird.remap_boundaries(x, x_i, N, xl=x_l)  # randomized remap

        x, x_i, x_l = self.tinybird.sample_data_seq_length(x, x_i, N, seq_len=8000, xl=x_l)

        B, H, W, N_blocks = x.shape[0], x.shape[-2], x.shape[-1], x_i.shape[1]

        # choose masking strategy
        if random.random() < 0.5:
            mblock = [N_blocks - 1]
            masked_blocks = 0
        else:
            mblock = []
            masked_blocks = max(1, int(0.5 * N_blocks))

        # shapes + masking fraction
        tag = "train_batch" if is_training else "val_batch"
        log_batch_shapes(tag, step_num, B=B, W=W, N=N_blocks)

        h, idx_restore, bool_mask, T, x_l_tok_cond = self.tinybird.forward_encoder(
            x, x_i, mblock=mblock, masked_blocks=masked_blocks, xl=x_l
        )

        masked_fraction = bool_mask.float().mean().item()
        log_batch_shapes(tag, step_num, masked_fraction=masked_fraction)

        # decoder + heads
        pred, logits_label = self.tinybird.forward_decoder(h, idx_restore, T, xl_tok_cond=x_l_tok_cond, W=W)

        lambda_label = 0.3
        loss_recon = self.tinybird.loss_mse(x, pred, bool_mask)
        loss_lbl = self.tinybird.loss_label(logits_label, x_l, bool_mask, W)
        loss_total = loss_recon + lambda_label * loss_lbl

        # metrics
        metrics = self._compute_label_metrics(logits_label, x_l, bool_mask, W)
        if log_metrics:
            run_tag = "train" if is_training else "val"
            self._log_losses_and_metrics(
                run_tag, step_num, loss_total.item(), loss_recon.item(), loss_lbl.item(), metrics
            )

        return loss_total, loss_recon, loss_lbl, H, W, metrics

    def step(self, step_num, batch, is_training=True):
        """
        Perform one forward pass and optionally backward pass.

        Args:
            batch: Input batch (spectrograms, chirp_intervals, N, filenames)
            is_training: If True, perform gradient update. If False, no gradients.

        Returns:
            loss: Scalar loss value
        """
        # --- Memory tracking: capture baseline before the step ---
        rss_before = self._process.memory_info().rss  # resident set size (bytes)
        if torch.cuda.is_available() and not self._mem_reported:
            torch.cuda.reset_peak_memory_stats(self.device)

        # spec, chirp_intervals, chirp_labels_pad, chirp_feats_pad, N, filename
        spectrograms, chirp_intervals, chirp_labels, _, N, _ = batch

        x = spectrograms.to(self.device, non_blocking=True).float()  # (B, 1, H, W)
        x_i = chirp_intervals.to(self.device, non_blocking=True)  # (B, N, 2)
        N = N.to(self.device, non_blocking=True)  # (B, 1) # number of chirp intervals
        x_l = chirp_labels.to(self.device, non_blocking=True).long()  # (B, N) # channel labels

        if is_training:
            self.tinybird.train()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.tinybird.eval()

        # Forward pass through encoder-decoder
        with cuda_mem_scope(self.device, step=step_num):
            with torch.set_grad_enabled(is_training):
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        loss, loss_recon, loss_lbl, H, W, _ = self._forward_encode_decode(
                            x, x_i, N, x_l, step_num, is_training=is_training, log_metrics=is_training
                        )

                else:
                    loss, loss_recon, loss_lbl, H, W, _ = self._forward_encode_decode(
                        x, x_i, N, x_l, step_num, is_training=is_training, log_metrics=is_training
                    )

        # Backward pass only for training
        if is_training:
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Update learning rate scheduler
            self.scheduler.step()

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

        # --- Memory tracking: report once after the first completed training step ---
        if not self._mem_reported:
            rss_after = self._process.memory_info().rss
            cpu_delta = rss_after - rss_before

            cuda_alloc = None
            cuda_reserved = None
            if torch.cuda.is_available():
                # Ensure all CUDA work is done before querying
                torch.cuda.synchronize(self.device)
                cuda_alloc = torch.cuda.max_memory_allocated(self.device)
                cuda_reserved = torch.cuda.max_memory_reserved(self.device)

            # Print a concise, human-readable summary
            print("=" * 60)
            print("MEMORY USAGE (first training step)")
            print("=" * 60)
            print(f"CPU RSS before:  {self._human_bytes(rss_before)}")
            print(f"CPU RSS after:   {self._human_bytes(rss_after)}  (+{self._human_bytes(max(0, cpu_delta))})")
            if cuda_alloc is not None:
                print(f"CUDA max allocated: {self._human_bytes(cuda_alloc)}")
                print(f"CUDA max reserved:  {self._human_bytes(cuda_reserved)}")
                # Also log raw numbers to W&B for later inspection
            print("=" * 60)

            # Avoid spamming every step; only report once per run init
            self._mem_reported = True

        return loss.item()

    def save_reconstruction(self, batch, step_num):
        """Save reconstruction visualization comparing input and output spectrograms."""
        # spec, chirp_intervals, chirp_labels_pad, chirp_feats_pad, N, filename
        spectrograms, chirp_intervals, _, _, N, _ = batch
        x = spectrograms.to(self.device, non_blocking=True).float()  # (B, 1, H, W)
        x_i = chirp_intervals.to(self.device, non_blocking=True)  # (B, N, 2)
        N = N.to(self.device, non_blocking=True)  # (B, 1) # number of chirp intervals

        # Get model prediction
        self.tinybird.eval()
        with torch.no_grad():
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    x, x_i = self.tinybird.compactify_data(x, x_i, N)
                    x, x_i = self.tinybird.sample_data(x, x_i, N, n_blocks=3)
                    h, idx_restore, bool_mask, T = self.tinybird.forward_encoder(x, x_i, masked_blocks=1)
                    pred = self.tinybird.forward_decoder(h, idx_restore, T)
            else:
                x, x_i = self.tinybird.compactify_data(x, x_i, N)
                x, x_i = self.tinybird.sample_data(x, x_i, N, n_blocks=3)
                h, idx_restore, bool_mask, T = self.tinybird.forward_encoder(x, x_i, masked_blocks=1)
                pred = self.tinybird.forward_decoder(h, idx_restore, T)

        # Depatchify prediction to get back (B, 1, H, W) format
        def depatchify(pred_patches):
            """pred_patches: (B, T, P) → (B, 1, H, W) using the CURRENT x dims."""
            H, W = x.shape[2], x.shape[3]  # after compactify/sample
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

        # Compute robust display limits from the original spectrogram
        import numpy as np

        def _auto_limits(img, lo=1.0, hi=99.0):
            vals = np.asarray(img).ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return None
            vmin, vmax = np.percentile(vals, [lo, hi])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    vmin, vmax = 0.0, 1.0
            return float(vmin), float(vmax)

        x_img = x[0, 0].detach().cpu().numpy()  # First sample, first channel
        limits = _auto_limits(x_img)
        if limits is None:
            limits = (float(np.nanmin(x_img)), float(np.nanmax(x_img)))
        vmin, vmax = limits

        # Create overlay patches
        overlay_patches = create_overlay(x_patches, pred_denorm, bool_mask)

        # Create masked original: original with black (below vmin) where masked
        def create_masked_original(x_patches, bool_mask):
            masked_patches = x_patches.clone()
            # Use a value slightly below the display vmin so masked areas appear black
            min_val = torch.tensor(
                vmin - (vmax - vmin) * 0.05, dtype=masked_patches.dtype, device=masked_patches.device
            )
            masked_patches[bool_mask] = min_val
            return masked_patches

        masked_img = depatchify(create_masked_original(x_patches, bool_mask))[0, 0].detach().cpu().numpy()
        overlay_img = depatchify(overlay_patches)[0, 0].detach().cpu().numpy()

        fig = plt.figure(figsize=(12, 4.5))  # Taller figure for 3 rows

        ax1 = plt.subplot(3, 1, 1)
        ax1.imshow(x_img, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        ax1.set_title("Input Spectrogram")
        ax1.axis("off")

        ax2 = plt.subplot(3, 1, 2)
        ax2.imshow(masked_img, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        ax2.set_title("Original with Mask (black = masked patches)")
        ax2.axis("off")

        ax3 = plt.subplot(3, 1, 3)
        ax3.imshow(overlay_img, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        ax3.set_title("Overlay: Unmasked Original + Masked Predictions")
        ax3.axis("off")

        fig.tight_layout()
        recon_path = os.path.join(self.imgs_path, f"recon_step_{step_num:06d}.png")
        fig.savefig(recon_path, dpi=150)
        plt.close(fig)
        # W&B: log recon images
        wandb.log(
            {
                "recon/input": wandb.Image(x_img, caption=f"step={step_num} input"),
                "recon/masked": wandb.Image(masked_img, caption=f"step={step_num} masked"),
                "recon/overlay": wandb.Image(overlay_img, caption=f"step={step_num} overlay"),
            },
            step=int(step_num),
            commit=False,
        )

    def train(self):
        # Initialize datasets
        train_dataset = SpectogramDataset(
            dir=self.config["train_dir"], n_mels=self.config["mels"], n_timebins=self.config["num_timebins"]
        )

        val_dataset = SpectogramDataset(
            dir=self.config["val_dir"], n_mels=self.config["mels"], n_timebins=self.config["num_timebins"]
        )

        # Initialize dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,  # Ensure no undersized batches
        )
        # train_loader = ChunkingLoader(base_loader)

        val_loader = DataLoader(
            val_dataset, batch_size=self.config["batch_size"], shuffle=False, num_workers=4, pin_memory=True
        )

        # Training loop
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)

        # Calculate total steps and range
        total_steps = self.config["steps"]
        end_step = self.starting_step + total_steps

        print(f"Training from step {self.starting_step} to {end_step}")

        for step_num in range(self.starting_step, end_step):
            try:
                train_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                train_batch = next(train_iter)

            try:
                train_loss = self.step(step_num, train_batch, is_training=True)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("\n[OOM] CUDA out of memory caught. Dumping diagnostics...")
                    dump_cuda_summary()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    else:
                        raise
                else:
                    raise
            # Store training loss every step
            self.train_loss_history.append(train_loss)
            self.train_steps.append(step_num)

            # Evaluation and checkpointing
            current_lr = self.scheduler.get_last_lr()[0]

            if step_num % self.config["eval_every"] == 0:
                self.tinybird.eval()
                val_items = []
                with torch.no_grad():
                    for _ in range(20):
                        try:
                            val_batch = next(val_iter)
                        except StopIteration:
                            val_iter = iter(val_loader)
                            val_batch = next(val_iter)

                        # unpack + device (same as in step())
                        spectrograms, chirp_intervals, chirp_labels, _, N, _ = val_batch
                        x   = spectrograms.to(self.device, non_blocking=True).float()
                        x_i = chirp_intervals.to(self.device, non_blocking=True)
                        Nv  = N.to(self.device, non_blocking=True)
                        x_l = chirp_labels.to(self.device, non_blocking=True).long()

                        with cuda_mem_scope(self.device, step=step_num):
                            if self.use_amp:
                                with torch.amp.autocast('cuda'):
                                    loss, loss_recon, loss_lbl, H, W, metrics = self._forward_encode_decode(
                                        x, x_i, Nv, x_l, step_num, is_training=False, log_metrics=False
                                    )
                            else:
                                loss, loss_recon, loss_lbl, H, W, metrics = self._forward_encode_decode(
                                    x, x_i, Nv, x_l, step_num, is_training=False, log_metrics=False
                                )

                        val_items.append({
                            "loss_total": float(loss.item()),
                            "loss_recon": float(loss_recon.item()),
                            "loss_lbl":   float(loss_lbl.item()),
                            "metrics":    metrics,
                        })

                # Aggregate across 20 batches
                val_loss, val_loss_recon, val_loss_lbl, val_metrics = self._aggregate_val_metrics(val_items)

                # Update EMA val loss (mirror the training EMA logic)
                if self.ema_val_loss is None:
                    self.ema_val_loss = float(val_loss)
                else:
                    self.ema_val_loss = self.ema_alpha * float(self.ema_val_loss) + (1 - self.ema_alpha) * float(val_loss)
                    
                # Book-keeping / EMA
                self.val_loss_history.append(val_loss)
                self.val_steps.append(step_num)
                current_lr = self.scheduler.get_last_lr()[0]

                print(
                    f"Step {step_num}: Train Loss = {train_loss:.6f}, "
                    f"EMA Train = {self.ema_train_loss:.6f}, "
                    f"Val Loss = {val_loss:.6f}, "
                    f"EMA Val = {self.ema_val_loss:.6f}, "
                    f"LR = {current_lr:.2e}"
                )
                with open(self.loss_log_path, 'a') as f:
                    f.write(f"{step_num},{train_loss:.6f},{self.ema_train_loss:.6f},{val_loss:.6f},{self.ema_val_loss:.6f}\\n")

                # Single consolidated log for validation (losses + averaged metrics)
                self._log_losses_and_metrics("val", step_num, val_loss, val_loss_recon, val_loss_lbl, val_metrics)

                # Save weights + W&B scalars
                weight_path = os.path.join(self.weights_path, f"model_step_{step_num:06d}.pth")
                torch.save(self.tinybird.state_dict(), weight_path)
                wandb.log(
                    {
                        "lr": float(current_lr),
                        "train/loss": float(train_loss),
                        "train/ema_loss": float(self.ema_train_loss if self.ema_train_loss is not None else train_loss),
                        "val/loss": float(val_loss),
                        "val/ema_loss": float(self.ema_val_loss),
                    },
                    step=int(step_num),
                    commit=True,
                )
            else:
                current_lr = self.scheduler.get_last_lr()[0]
                wandb.log(
                    {
                        "lr": float(current_lr),
                        "train/loss": float(train_loss),
                        "train/ema_loss": float(self.ema_train_loss if self.ema_train_loss is not None else train_loss),
                    },
                    step=int(step_num),
                    commit=True,
                )

        # Save final model weights
        final_step = self.starting_step + self.config['steps'] - 1
        final_weight_path = os.path.join(self.weights_path, f"model_step_{final_step:06d}.pth")
        torch.save(self.tinybird.state_dict(), final_weight_path)

        # Generate loss plot at the end of training
        self.end_of_train_viz()

    def end_of_train_viz(self):
        """Generate and save loss plots showing training and validation curves."""
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # First panel: All losses
        ax1.plot(self.train_steps, self.train_loss_history, label='Training Loss', alpha=0.7, linewidth=1, color='blue')
        ax1.plot(
            self.val_steps,
            self.val_loss_history,
            label='Validation Loss',
            marker='o',
            markersize=3,
            linewidth=2,
            color='red',
        )
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

        ax2.plot(self.train_steps, ema_train_history, label='EMA Training Loss', linewidth=2, color='darkblue')
        ax2.plot(
            self.val_steps,
            ema_val_history,
            label='EMA Validation Loss',
            marker='o',
            markersize=3,
            linewidth=2,
            color='darkred',
        )
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
        # W&B: log final loss curves and attach the CSV
        wandb.log(
            {"final/loss_plot": wandb.Image(plot_path)},
            step=int(self.starting_step + self.config['steps'] - 1),
            commit=True,
        )
        if os.path.exists(self.loss_log_path):
            dest = os.path.join(wandb.run.dir, "loss_log.txt")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(self.loss_log_path, dest)
            wandb.save(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pretrain args")

    # Required argparse
    parser.add_argument("--train_dir", type=str, help="training directory")
    parser.add_argument("--val_dir", type=str, help="validation directory")
    parser.add_argument("--run_name", type=str, help="directory name inside /runs to store train run details")

    # Defaults
    parser.add_argument("--steps", type=int, default=500_000, help="number of training steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--patch_height", type=int, default=32, help="patch height")
    parser.add_argument("--patch_width", type=int, default=1, help="patch width")
    parser.add_argument("--mels", type=int, default=128, help="number of mel bins")
    parser.add_argument("--num_timebins", type=int, default=1024, help="n number of time bins")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--mask_p", type=float, default=0.75, help="mask probability")
    parser.add_argument("--eval_every", type=int, default=500, help="evaluate every N steps")
    parser.add_argument("--amp", action="store_true", help="enable automatic mixed precision training")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument(
        "--continue_from", type=str, help="continue training from existing run directory (path to run dir)"
    )
    parser.add_argument(
        "--fallback_random",
        action="store_true",
        help="if checkpoint not found when continuing, initialize with random weights instead of raising error",
    )

    # Encoder
    parser.add_argument("--enc_hidden_d", type=int, default=384, help="encoder hidden dimension")
    parser.add_argument("--enc_n_head", type=int, default=6, help="encoder number of attention heads")
    parser.add_argument("--enc_n_layer", type=int, default=6, help="encoder number of transformer layers")
    parser.add_argument("--enc_dim_ff", type=int, default=1536, help="encoder feed-forward dimension")

    # Decoder Model
    parser.add_argument("--dec_hidden_d", type=int, default=192, help="decoder hidden dimension")
    parser.add_argument("--dec_n_head", type=int, default=6, help="decoder number of attention heads")
    parser.add_argument("--dec_n_layer", type=int, default=3, help="decoder number of transformer layers")
    parser.add_argument("--dec_dim_ff", type=int, default=768, help="decoder feed-forward dimension")

    # W&B configuration (Step 1)
    parser.add_argument("--wandb_project", type=str, default="tinybird", help="Weights & Biases project")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity (team/user)")
    parser.add_argument(
        "--wandb_mode", type=str, default="online", help="Weights & Biases mode: online|offline|disabled"
    )

    parser.add_argument("--config_json", type=str, default=None, help="config json file")
    parser.add_argument("--detect_anomaly", action="store_true", help="detect anomalies in the training loop")

    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

    # Handle continue mode vs new training
    if args.continue_from:
        # Continue training mode - load config from existing run

        # Load existing config and model
        model, config = load_model_from_checkpoint(args.continue_from, fallback_to_random=args.fallback_random)

        # Override with any command line args that were provided
        for key, value in vars(args).items():
            if value is not None and key not in ['continue_from']:
                config[key] = value

        config['continue_from'] = args.continue_from
        config['is_continuing'] = True

    else:
        if args.config_json:
            print(f"Loading config from {args.config_json}")
            with open(args.config_json, "r") as f:
                config = json.load(f)

            for k, v in vars(args).items():
                if k in ["continue_from", "config_json"]:
                    continue
                # Only override if the CLI value differs from the parser default
                if v is not None and v != parser.get_default(k):
                    config[k] = v

            # validate required fields after merge
            for req in ["train_dir", "val_dir", "run_name"]:
                assert req in config and config[req], f"--{req} is required (via JSON or CLI)"
        else:
            if not args.train_dir or not args.val_dir or not args.run_name:
                parser.error("--train_dir, --val_dir, and --run_name are required when not using --continue_from")
            config = vars(args)
        config['is_continuing'] = False

    # Calculate seq_len from num_timebins and patch dimensions
    assert config["num_timebins"] % config["patch_width"] == 0, (
        f"num_timebins ({config['num_timebins']}) must be divisible by patch_width ({config['patch_width']})"
    )
    assert config["mels"] % config["patch_height"] == 0, (
        f"mels ({config['mels']}) must be divisible by patch_height ({config['patch_height']})"
    )

    # Configure patch size and max sequence length for model
    config["patch_size"] = (config["patch_height"], config["patch_width"])
    config["max_seq"] = (config["num_timebins"] // config["patch_width"]) * (config["mels"] // config["patch_height"])

    # Create trainer with loaded model if continuing
    if config.get('is_continuing', False):
        trainer = Trainer(config, pretrained_model=model)
    else:
        trainer = Trainer(config)

    trainer.train()
