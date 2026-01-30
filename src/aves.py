import argparse
import json
import math
import os
import random
import shutil
import time
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from utils import (
    parse_chunk_ms,
    clip_labels_to_chunk,
    get_num_classes_from_annotations,
)

try:
    import torchaudio
    from torchaudio.models import wav2vec2_model
except Exception:
    torchaudio = None
    wav2vec2_model = None


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")


def _require_torchaudio():
    if torchaudio is None or wav2vec2_model is None:
        raise RuntimeError(
            "torchaudio is required for AVES. Install torchaudio and try again."
        )


_WAV_INDEX_CACHE = {}


def _load_wav_manifest(manifest_path):
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"wav_manifest not found: {manifest_path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise FileNotFoundError(f"wav_manifest is empty: {manifest_path}")

    data = None
    try:
        data = json.loads(text)
    except Exception:
        data = None

    items = []
    if data is not None:
        if isinstance(data, dict):
            items = list(data.items())
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if "stem" in item and ("path" in item or "wav" in item):
                        items.append((item["stem"], item.get("path") or item.get("wav")))
                    else:
                        raise ValueError(f"Invalid manifest entry: {item}")
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    items.append((item[0], item[1]))
                else:
                    raise ValueError(f"Invalid manifest entry: {item}")
        else:
            raise ValueError("wav_manifest must be a JSON dict or list.")
    else:
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                stem, wav_path = line.split("\t", 1)
            else:
                wav_path = line
                stem = Path(wav_path).stem
            items.append((stem, wav_path))

    index = {}
    dupes = {}
    for stem, wav_path in items:
        if not stem or not wav_path:
            raise ValueError("wav_manifest entries must include a stem and a path.")
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"Wav not found in manifest for {stem}: {wav_path}")
        if stem in index:
            dupes.setdefault(stem, []).append(str(wav_path))
            continue
        index[stem] = wav_path
    if dupes:
        sample = ", ".join(sorted(dupes)[:5])
        raise ValueError(
            f"Duplicate stems in wav_manifest: {sample} (total={len(dupes)})"
        )
    if not index:
        raise FileNotFoundError(f"No wav entries found in manifest: {manifest_path}")
    return index


def build_wav_index(wav_root, exts=(".wav", ".flac", ".ogg", ".mp3"), manifest_path=None):
    if manifest_path:
        return _load_wav_manifest(manifest_path)
    root = Path(wav_root)
    if not root.exists():
        raise FileNotFoundError(f"wav_root not found: {wav_root}")
    exts = tuple(e.lower() for e in exts)
    cache_key = (str(root.resolve()), exts)
    if cache_key in _WAV_INDEX_CACHE:
        return _WAV_INDEX_CACHE[cache_key]
    index = {}
    dupes = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        stem = path.stem
        if stem in index:
            dupes.setdefault(stem, []).append(path)
            continue
        index[stem] = path
    if dupes:
        print(f"Warning: {len(dupes)} duplicate wav stems found; using first occurrence only.")
    if not index:
        raise FileNotFoundError(f"No audio files found under: {wav_root}")
    _WAV_INDEX_CACHE[cache_key] = index
    return index


def _build_label_index(annotation_file, mode):
    if mode not in ("detect", "unit_detect", "classify"):
        raise ValueError("mode must be detect, unit_detect, or classify")
    data = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    label_index = {}
    for rec in data.get("recordings", []):
        rec_filename = Path(rec.get("recording", {}).get("filename", "")).stem
        events = rec.get("detected_events", [])
        if mode == "detect":
            labels = [
                {"onset_ms": event["onset_ms"], "offset_ms": event["offset_ms"]}
                for event in events
            ]
        else:
            labels = [unit for event in events for unit in event.get("units", [])]
        label_index[rec_filename] = labels
    return label_index


def _clip_or_pad_audio(wav, target_len, is_train):
    cur_len = wav.shape[0]
    if cur_len == target_len:
        return wav, 0
    if cur_len > target_len:
        if is_train:
            start = random.randint(0, cur_len - target_len)
        else:
            start = 0
        return wav[start : start + target_len], start
    pad_len = target_len - cur_len
    wav = F.pad(wav, (0, pad_len), mode="constant")
    return wav, 0


class AvesSupervisedDataset(Dataset):
    def __init__(
        self,
        spec_dir,
        wav_root,
        annotation_file,
        mode="detect",
        audio_sr=16000,
        wav_exts=(".wav", ".flac", ".ogg", ".mp3"),
        wav_manifest=None,
        clip_seconds=None,
        is_train=False,
        max_files=None,
    ):
        _require_torchaudio()
        self.spec_dir = Path(spec_dir)
        self.file_dirs = sorted(self.spec_dir.glob("*.npy"))
        if max_files is not None:
            self.file_dirs = self.file_dirs[: int(max_files)]
        if len(self.file_dirs) == 0:
            raise SystemExit(f"No .npy files found in: {spec_dir}")

        self.wav_index = build_wav_index(
            wav_root, exts=wav_exts, manifest_path=wav_manifest
        )
        self.label_index = _build_label_index(annotation_file, mode)
        self.mode = mode
        self.audio_sr = int(audio_sr)
        self.clip_seconds = float(clip_seconds) if clip_seconds else None
        self.is_train = bool(is_train)

        self.num_classes = get_num_classes_from_annotations(annotation_file, mode)

    def __len__(self):
        return len(self.file_dirs)

    def __getitem__(self, index):
        spec_path = self.file_dirs[index]
        filename = spec_path.stem
        base_filename, chunk_start_ms, chunk_end_ms = parse_chunk_ms(filename)

        labels = self.label_index.get(base_filename)
        if labels is None:
            raise ValueError(f"No matching recording found for: {base_filename}")
        labels = clip_labels_to_chunk(labels, chunk_start_ms, chunk_end_ms)

        wav_path = self.wav_index.get(base_filename)
        if wav_path is None:
            raise FileNotFoundError(f"Wav not found for stem: {base_filename}")

        wav, sr = torchaudio.load(str(wav_path))
        if wav.ndim == 2:
            wav = wav[0]
        if sr != self.audio_sr:
            wav = torchaudio.functional.resample(wav, sr, self.audio_sr)

        if chunk_start_ms is not None:
            start = int(round(float(chunk_start_ms) / 1000.0 * self.audio_sr))
            if chunk_end_ms is None:
                end = wav.shape[0]
            else:
                end = int(round(float(chunk_end_ms) / 1000.0 * self.audio_sr))
            wav = wav[start:end]

        if self.clip_seconds:
            target_len = int(round(self.clip_seconds * self.audio_sr))
            wav, crop_start = _clip_or_pad_audio(wav, target_len, self.is_train)
            if crop_start > 0:
                crop_start_ms = crop_start / self.audio_sr * 1000.0
                crop_end_ms = crop_start_ms + self.clip_seconds * 1000.0
                labels = clip_labels_to_chunk(labels, crop_start_ms, crop_end_ms)

        wav = wav.to(torch.float32)
        return {
            "audio": wav,
            "labels": labels,
            "filename": filename,
        }


def aves_collate(batch):
    audios = [b["audio"] for b in batch]
    lengths = torch.tensor([a.shape[0] for a in audios], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.zeros((len(audios), max_len), dtype=audios[0].dtype)
    for i, a in enumerate(audios):
        padded[i, : a.shape[0]] = a
    return {
        "audio": padded,
        "lengths": lengths,
        "labels": [b["labels"] for b in batch],
        "filenames": [b["filename"] for b in batch],
    }


def freeze_embedding_weights(model, trainable):
    model.feature_extractor.requires_grad_(False)
    model.feature_extractor.eval()
    for param in model.encoder.parameters():
        param.requires_grad = bool(trainable)
    if not trainable:
        model.encoder.eval()


class AvesClassifier(nn.Module):
    def __init__(
        self,
        config_path,
        model_path,
        num_classes,
        mode="classify",
        embedding_dim=768,
        trainable=False,
        linear_probe=False,
        encoder_layer_idx=None,
        hidden_dim=256,
    ):
        super().__init__()
        _require_torchaudio()
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.encoder = wav2vec2_model(**self.config, aux_num_out=None)
        state = torch.load(model_path, map_location="cpu")
        self.encoder.load_state_dict(state)

        self.trainable = bool(trainable)
        freeze_embedding_weights(self.encoder, self.trainable)

        self.num_classes = int(num_classes)
        self.mode = str(mode)
        self.encoder_layer_idx = encoder_layer_idx

        if embedding_dim is None:
            embedding_dim = int(self.config.get("encoder_embed_dim", 768))
        self.embedding_dim = int(embedding_dim)

        if self.mode == "classify" and self.trainable:
            linear_probe = True

        if self.mode == "classify":
            out_dim = self.num_classes
        else:
            out_dim = 1 if self.num_classes == 2 else self.num_classes

        if linear_probe:
            self.classifier = nn.Linear(self.embedding_dim, out_dim)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def set_train_mode(self):
        self.classifier.train()
        if self.trainable:
            self.encoder.encoder.train()
        else:
            self.encoder.encoder.eval()
        self.encoder.feature_extractor.eval()

    def set_eval_mode(self):
        self.classifier.eval()
        self.encoder.encoder.eval()
        self.encoder.feature_extractor.eval()

    def _get_output_lengths(self, lengths):
        if hasattr(self.encoder, "_get_feat_extract_output_lengths"):
            return self.encoder._get_feat_extract_output_lengths(lengths)
        return None

    def _extract_features(self, wav, lengths=None):
        if lengths is not None:
            try:
                feats, out_lengths = self.encoder.extract_features(wav, lengths)
            except TypeError:
                feats = self.encoder.extract_features(wav)[0]
                out_lengths = self._get_output_lengths(lengths)
        else:
            feats = self.encoder.extract_features(wav)[0]
            out_lengths = None

        if out_lengths is None:
            t = feats[-1].shape[1]
            out_lengths = torch.full(
                (feats[-1].shape[0],), t, dtype=torch.long, device=feats[-1].device
            )
        return feats, out_lengths

    def min_input_samples(self):
        layers = getattr(self.encoder.feature_extractor, "conv_layers", None)
        if not layers:
            return 1
        length = 1
        for layer in reversed(layers):
            conv = getattr(layer, "conv", layer)
            kernel = getattr(conv, "kernel_size", 1)
            stride = getattr(conv, "stride", 1)
            if isinstance(kernel, tuple):
                kernel = kernel[0]
            if isinstance(stride, tuple):
                stride = stride[0]
            length = (length - 1) * int(stride) + int(kernel)
        return int(length)

    def forward(self, wav, lengths=None):
        feats, out_lengths = self._extract_features(wav, lengths)
        if self.encoder_layer_idx is None:
            out = feats[-1]
        else:
            idx = int(self.encoder_layer_idx)
            if idx < 0:
                idx = len(feats) + idx
            if idx < 0 or idx >= len(feats):
                raise ValueError(
                    f"encoder_layer_idx out of range: {self.encoder_layer_idx} (num_layers={len(feats)})"
                )
            out = feats[idx]
        logits = self.classifier(out)
        if logits.dim() == 3 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return logits, out_lengths


def build_frame_targets(labels_list, in_lengths, out_lengths, sr, mode, num_classes, device):
    bsz = len(labels_list)
    max_t = int(out_lengths.max().item()) if bsz else 0
    targets = torch.zeros((bsz, max_t), dtype=torch.long, device=device)
    mask = torch.zeros((bsz, max_t), dtype=torch.bool, device=device)
    for i in range(bsz):
        out_len = int(out_lengths[i].item())
        if out_len <= 0:
            continue
        mask[i, :out_len] = True
        duration_ms = float(in_lengths[i].item()) / float(sr) * 1000.0
        if duration_ms <= 0:
            continue
        for label in labels_list[i]:
            onset = float(label.get("onset_ms", 0.0))
            offset = float(label.get("offset_ms", 0.0))
            if offset <= 0 or onset >= duration_ms:
                continue
            onset = max(0.0, min(onset, duration_ms))
            offset = max(0.0, min(offset, duration_ms))
            if offset <= onset:
                continue
            start = int(math.floor(onset / duration_ms * out_len))
            end = int(math.ceil(offset / duration_ms * out_len))
            start = max(0, min(start, out_len))
            end = max(0, min(end, out_len))
            if end <= start:
                continue
            if mode in ("detect", "unit_detect"):
                targets[i, start:end] = 1
            else:
                cls = int(label.get("id", 0)) + 1
                targets[i, start:end] = cls
    return targets, mask


def compute_loss(logits, targets, mask, num_classes, class_weighting, mode):
    if num_classes == 2 and mode in ("detect", "unit_detect"):
        logits_flat = logits.reshape(-1)
        targets_flat = targets.reshape(-1).float()
        mask_flat = mask.reshape(-1)
        if mask_flat.sum().item() == 0:
            return torch.tensor(0.0, device=logits.device)
        loss = F.binary_cross_entropy_with_logits(
            logits_flat[mask_flat], targets_flat[mask_flat], reduction="mean"
        )
        return loss

    bsz, t, c = logits.shape
    logits_flat = logits.reshape(-1, c)
    targets_flat = targets.reshape(-1)
    mask_flat = mask.reshape(-1)
    if mask_flat.sum().item() == 0:
        return torch.tensor(0.0, device=logits.device)

    if class_weighting:
        counts = torch.bincount(targets_flat[mask_flat], minlength=num_classes).float()
        counts_safe = torch.where(counts > 0, counts, torch.ones_like(counts))
        weights = counts.sum() / (counts_safe * num_classes)
        weights = torch.where(counts > 0, weights, torch.zeros_like(weights))
        loss = F.cross_entropy(
            logits_flat[mask_flat], targets_flat[mask_flat], weight=weights
        )
    else:
        loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])
    return loss


def compute_accuracy(logits, targets, mask, num_classes, mode):
    if num_classes == 2 and mode in ("detect", "unit_detect"):
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
    else:
        preds = torch.argmax(logits, dim=-1)
    mask_flat = mask.reshape(-1)
    if mask_flat.sum().item() == 0:
        return 0.0
    preds_flat = preds.reshape(-1)[mask_flat]
    targets_flat = targets.reshape(-1)[mask_flat]
    correct = (preds_flat == targets_flat).sum().item()
    total = mask_flat.sum().item()
    return 100.0 * correct / total


def compute_f1(logits, targets, mask, num_classes, mode):
    mask_flat = mask.reshape(-1)
    if mask_flat.sum().item() == 0:
        return 0.0
    if num_classes == 2 and mode in ("detect", "unit_detect"):
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        preds_flat = preds.reshape(-1)[mask_flat]
        targets_flat = targets.reshape(-1)[mask_flat]
        tp = ((preds_flat == 1) & (targets_flat == 1)).sum().item()
        fp = ((preds_flat == 1) & (targets_flat == 0)).sum().item()
        fn = ((preds_flat == 0) & (targets_flat == 1)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if (precision + recall) == 0:
            return 0.0
        return 100.0 * (2.0 * precision * recall / (precision + recall))

    preds = torch.argmax(logits, dim=-1)
    preds_flat = preds.reshape(-1)[mask_flat]
    targets_flat = targets.reshape(-1)[mask_flat]
    f1s = []
    for c in range(int(num_classes)):
        tp = ((preds_flat == c) & (targets_flat == c)).sum().item()
        fp = ((preds_flat == c) & (targets_flat != c)).sum().item()
        fn = ((preds_flat != c) & (targets_flat == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1s.append(f1)
    if not f1s:
        return 0.0
    return 100.0 * float(np.mean(f1s))


class AvesTrainer:
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.compute_f1 = config["mode"] in ("detect", "unit_detect") or config.get(
            "log_f1", False
        )

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
        self.weights_path = os.path.join(self.run_path, "weights")
        os.makedirs(self.weights_path, exist_ok=True)

        with open(os.path.join(self.run_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = self.model.to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        self.min_samples = int(self.model.min_input_samples())
        self._short_clip_warned = False

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=config["lr"], weight_decay=config["weight_decay"]
        )

        self.scheduler = None

        self.use_amp = bool(config.get("amp", False))
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        self.loss_log_path = os.path.join(self.run_path, "loss_log.txt")
        with open(self.loss_log_path, "w") as f:
            if self.compute_f1:
                f.write(
                    "step,train_loss,val_loss,train_acc,val_acc,train_f1,val_f1,samples_processed,steps_per_sec,samples_per_sec\n"
                )
            else:
                f.write(
                    "step,train_loss,val_loss,train_acc,val_acc,samples_processed,steps_per_sec,samples_per_sec\n"
                )

    def _run_batch(self, batch, train=True):
        audio = batch["audio"].to(self.device, non_blocking=True)
        lengths = batch["lengths"].to(self.device, non_blocking=True)
        labels = batch["labels"]
        min_samples = int(self.min_samples)

        if lengths.numel() > 0:
            valid = (lengths >= min_samples)
            if not torch.all(valid):
                valid_list = valid.detach().cpu().tolist()
                drop_n = int(len(valid_list) - sum(1 for v in valid_list if v))
                if drop_n > 0 and not self._short_clip_warned:
                    print(
                        f"Warning: dropping {drop_n} clips shorter than min_input_samples={min_samples}."
                    )
                    self._short_clip_warned = True
                if not any(valid_list):
                    zero = torch.tensor(0.0, device=self.device)
                    return zero, 0.0, 0.0, 0
                keep_idx = [i for i, v in enumerate(valid_list) if v]
                audio = audio[keep_idx]
                lengths = lengths[keep_idx]
                labels = [labels[i] for i in keep_idx]

        if train:
            self.model.set_train_mode()
        else:
            self.model.set_eval_mode()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logits, out_lengths = self.model(audio, lengths)
            targets, mask = build_frame_targets(
                labels,
                lengths,
                out_lengths,
                self.config["audio_sr"],
                self.config["mode"],
                self.model.num_classes,
                device=logits.device,
            )
            loss = compute_loss(
                logits,
                targets,
                mask,
                self.model.num_classes,
                self.config.get("class_weighting", True),
                self.config["mode"],
            )

        mode = self.config["mode"]
        acc = compute_accuracy(logits, targets, mask, self.model.num_classes, mode)
        f1 = compute_f1(logits, targets, mask, self.model.num_classes, mode) if self.compute_f1 else 0.0
        return loss, acc, f1, int(mask.sum().item())

    def evaluate(self, loader):
        self.model.set_eval_mode()
        losses = []
        accs = []
        f1s = []
        with torch.no_grad():
            for batch in loader:
                loss, acc, f1, _ = self._run_batch(batch, train=False)
                losses.append(float(loss.item()))
                accs.append(acc)
                if self.compute_f1:
                    f1s.append(f1)
        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_acc = float(np.mean(accs)) if accs else 0.0
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        return mean_loss, mean_acc, mean_f1

    def train(self, train_loader, val_loader):
        steps = int(self.config["steps"])
        eval_every = int(self.config["eval_every"])
        grad_clip = float(self.config.get("grad_clip") or 0.0)
        save_intermediate = bool(self.config.get("save_intermediate_checkpoints", True))

        train_iter = iter(train_loader)
        samples_processed = 0
        last_eval_time = time.time()
        last_eval_step = 0

        train_losses = []
        train_accs = []
        train_f1s = []

        for step in range(1, steps + 1):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            loss, acc, f1, batch_samples = self._run_batch(batch, train=True)
            samples_processed += int(batch["audio"].shape[0])

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                self.scheduler.step()

            train_losses.append(float(loss.item()))
            train_accs.append(acc)
            if self.compute_f1:
                train_f1s.append(f1)

            if step % eval_every == 0:
                train_loss = float(np.mean(train_losses)) if train_losses else 0.0
                train_acc = float(np.mean(train_accs)) if train_accs else 0.0
                train_f1 = float(np.mean(train_f1s)) if train_f1s else 0.0

                val_loss, val_acc, val_f1 = self.evaluate(val_loader)
                current_time = time.time()
                elapsed_time = current_time - last_eval_time
                steps_since_last_eval = step - last_eval_step
                steps_per_sec = steps_since_last_eval / elapsed_time if elapsed_time > 0 else 0.0
                samples_per_sec = (steps_since_last_eval * train_loader.batch_size) / elapsed_time if elapsed_time > 0 else 0.0

                last_eval_time = current_time
                last_eval_step = step

                with open(self.loss_log_path, "a") as f:
                    if self.compute_f1:
                        f.write(
                            f"{step},{train_loss:.6f},{val_loss:.6f},{train_acc:.3f},{val_acc:.3f},{train_f1:.3f},{val_f1:.3f},{samples_processed},{steps_per_sec:.4f},{samples_per_sec:.2f}\n"
                        )
                    else:
                        f.write(
                            f"{step},{train_loss:.6f},{val_loss:.6f},{train_acc:.3f},{val_acc:.3f},{samples_processed},{steps_per_sec:.4f},{samples_per_sec:.2f}\n"
                        )

                print(
                    f"Step {step:>6} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.2f}"
                )

                if save_intermediate:
                    ckpt_path = os.path.join(self.weights_path, f"model_step_{step:06d}.pth")
                    torch.save(self.model.state_dict(), ckpt_path)

                train_losses = []
                train_accs = []
                train_f1s = []

        final_path = os.path.join(self.weights_path, "model_final.pth")
        torch.save(self.model.state_dict(), final_path)
        print(f"Training complete. Final weights: {final_path}")

        if self.config.get("save_val_logits", True):
            print("Exporting validation logits/labels for posthoc metrics...")
            self.export_validation_outputs(val_loader.dataset, final_path=final_path)

    def export_validation_outputs(self, val_dataset, final_path=None):
        out_dir = Path(self.run_path) / "val_outputs"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if final_path is not None:
            state_dict = torch.load(final_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

        window_timebins = int(self.config.get("num_timebins") or 0)
        token_lengths = {}
        sample_lengths = {}
        chunk_bounds = {}
        lengths = []
        min_samples = int(self.model.min_input_samples())

        self.model.set_eval_mode()

        for path in val_dataset.file_dirs:
            filename = path.stem
            base_filename, chunk_start_ms, chunk_end_ms = parse_chunk_ms(filename)
            chunk_bounds[filename] = (base_filename, chunk_start_ms, chunk_end_ms)

            wav_path = val_dataset.wav_index.get(base_filename)
            if wav_path is None:
                raise FileNotFoundError(f"Wav not found for stem: {base_filename}")

            wav, wav_sr = torchaudio.load(str(wav_path))
            if wav.ndim == 2:
                wav = wav[0]
            if wav_sr != val_dataset.audio_sr:
                wav = torchaudio.functional.resample(wav, wav_sr, val_dataset.audio_sr)
                wav_sr = val_dataset.audio_sr

            if chunk_start_ms is not None:
                start_sample = int(
                    round(float(chunk_start_ms) / 1000.0 * val_dataset.audio_sr)
                )
                if chunk_end_ms is None:
                    end_sample = wav.shape[0]
                else:
                    end_sample = int(
                        round(float(chunk_end_ms) / 1000.0 * val_dataset.audio_sr)
                    )
                end_sample = max(start_sample, min(end_sample, wav.shape[0]))
                wav = wav[start_sample:end_sample]

            orig_len = int(wav.shape[0])
            sample_lengths[filename] = orig_len
            if orig_len <= 0:
                token_len = 0
            else:
                if wav.shape[0] < min_samples:
                    wav = F.pad(wav, (0, min_samples - wav.shape[0]))
                audio = wav.unsqueeze(0).to(self.device)
                lengths_tensor = torch.tensor([orig_len], device=self.device, dtype=torch.long)
                with torch.no_grad():
                    logits, _ = self.model(audio, lengths_tensor)
                token_len = int(logits.shape[1]) if logits.dim() >= 2 else 0
            token_lengths[filename] = token_len
            if token_len > 0:
                lengths.append(token_len)

        if window_timebins <= 0:
            if not lengths:
                raise RuntimeError("Unable to infer num_timebins for export; no files found.")
            window_timebins = int(np.median(lengths))
            if window_timebins <= 0:
                window_timebins = int(max(lengths))
            print(
                f"Warning: num_timebins not set; using inferred window_timebins={window_timebins} (AVES tokens)"
            )

        window_index = []
        for path in val_dataset.file_dirs:
            filename = path.stem
            total_t = int(token_lengths.get(filename, 0) or 0)
            start = 0
            while start < total_t:
                length = min(window_timebins, total_t - start)
                window_index.append((path, filename, int(start), int(length), int(total_t)))
                start += window_timebins
            if total_t == 0:
                window_index.append((path, filename, 0, 0, 0))

        n_windows = len(window_index)
        c_out = 1 if self.model.num_classes == 2 else int(self.model.num_classes)
        logits_mm = np.lib.format.open_memmap(
            out_dir / "logits.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_windows, window_timebins, c_out),
        )
        # Keep the same val_outputs contract as supervised_train.py.
        # For AVES, patch_width=1 so "patches" align with timebins/tokens.
        labels_patches_mm = np.lib.format.open_memmap(
            out_dir / "labels_patches.npy",
            mode="w+",
            dtype=np.int64,
            shape=(n_windows, window_timebins),
        )
        window_starts_mm = np.lib.format.open_memmap(
            out_dir / "window_starts.npy",
            mode="w+",
            dtype=np.int64,
            shape=(n_windows,),
        )
        window_lengths_mm = np.lib.format.open_memmap(
            out_dir / "window_lengths.npy",
            mode="w+",
            dtype=np.int64,
            shape=(n_windows,),
        )
        filenames = []

        current_filename = None
        current_logits = None
        current_labels = []
        current_ms_per_timebin = 20.0
        ms_per_timebin_by_file = {}

        for idx, (path, filename, start, length, _total_t) in enumerate(window_index):
            if filename != current_filename:
                base_filename, chunk_start_ms, chunk_end_ms = chunk_bounds.get(
                    filename, (None, None, None)
                )
                if base_filename is None:
                    base_filename, chunk_start_ms, chunk_end_ms = parse_chunk_ms(filename)

                wav_path = val_dataset.wav_index.get(base_filename)
                if wav_path is None:
                    raise FileNotFoundError(f"Wav not found for stem: {base_filename}")

                wav, wav_sr = torchaudio.load(str(wav_path))
                if wav.ndim == 2:
                    wav = wav[0]
                if wav_sr != val_dataset.audio_sr:
                    wav = torchaudio.functional.resample(wav, wav_sr, val_dataset.audio_sr)
                    wav_sr = val_dataset.audio_sr

                if chunk_start_ms is not None:
                    start_sample = int(
                        round(float(chunk_start_ms) / 1000.0 * val_dataset.audio_sr)
                    )
                    if chunk_end_ms is None:
                        end_sample = wav.shape[0]
                    else:
                        end_sample = int(
                            round(float(chunk_end_ms) / 1000.0 * val_dataset.audio_sr)
                        )
                    end_sample = max(start_sample, min(end_sample, wav.shape[0]))
                    wav = wav[start_sample:end_sample]

                orig_len = int(sample_lengths.get(filename, 0) or wav.shape[0])
                if orig_len > 0:
                    if wav.shape[0] < min_samples:
                        wav = F.pad(wav, (0, min_samples - wav.shape[0]))

                    audio = wav.unsqueeze(0).to(self.device)
                    lengths = torch.tensor([orig_len], device=self.device, dtype=torch.long)
                    with torch.no_grad():
                        logits, _ = self.model(audio, lengths)

                    if logits.dim() == 2:
                        logits = logits.unsqueeze(-1)
                    logits = logits.squeeze(0).detach().cpu()

                    token_len = int(token_lengths.get(filename, 0) or logits.shape[0])
                    if token_len > 0 and logits.shape[0] > token_len:
                        logits = logits[:token_len]
                    current_logits = logits
                else:
                    token_len = 0
                    current_logits = torch.zeros((0, c_out), dtype=torch.float32)

                duration_ms = float(orig_len) / float(val_dataset.audio_sr) * 1000.0
                if token_len > 0 and duration_ms > 0:
                    current_ms_per_timebin = float(duration_ms) / float(token_len)
                else:
                    current_ms_per_timebin = 20.0
                ms_per_timebin_by_file[filename] = current_ms_per_timebin

                labels_src = val_dataset.label_index.get(base_filename, [])
                if chunk_start_ms is not None:
                    labels_src = clip_labels_to_chunk(labels_src, chunk_start_ms, chunk_end_ms)
                current_labels = labels_src
                current_filename = filename

            labels_win = np.zeros((window_timebins,), dtype=np.int64)
            if length > 0 and current_labels:
                win_start_ms = float(start) * current_ms_per_timebin
                win_end_ms = float(start + length) * current_ms_per_timebin
                for label in current_labels:
                    onset = float(label.get("onset_ms", 0.0))
                    offset = float(label.get("offset_ms", 0.0))
                    if offset <= win_start_ms or onset >= win_end_ms:
                        continue
                    onset_rel = max(onset, win_start_ms) - win_start_ms
                    offset_rel = min(offset, win_end_ms) - win_start_ms
                    start_i = int(math.floor(onset_rel / current_ms_per_timebin))
                    end_i = int(math.ceil(offset_rel / current_ms_per_timebin))
                    start_i = max(0, min(start_i, length))
                    end_i = max(start_i + 1, min(end_i, length))
                    if val_dataset.mode in ("detect", "unit_detect"):
                        labels_win[start_i:end_i] = 1
                    else:
                        cls = int(label.get("id", 0)) + 1
                        labels_win[start_i:end_i] = cls
            labels_patches_mm[idx, :] = labels_win

            logits_win = np.zeros((window_timebins, c_out), dtype=np.float32)
            if length > 0 and current_logits is not None:
                logits_slice = current_logits[start : start + length]
                logits_np = logits_slice.numpy() if isinstance(logits_slice, torch.Tensor) else np.asarray(logits_slice)
                if logits_np.ndim == 1:
                    logits_np = logits_np[:, None]
                slice_len = min(logits_np.shape[0], window_timebins)
                logits_win[:slice_len, : logits_np.shape[1]] = logits_np[:slice_len]

            logits_mm[idx, :, :] = logits_win
            window_starts_mm[idx] = int(start)
            window_lengths_mm[idx] = int(length)
            filenames.append(filename)

        with open(out_dir / "filenames.json", "w") as f:
            json.dump(filenames, f, indent=2)
        with open(out_dir / "ms_per_timebin_by_file.json", "w") as f:
            json.dump(ms_per_timebin_by_file, f, indent=2)

        ms_per_timebin_default = 20.0
        if ms_per_timebin_by_file:
            ms_per_timebin_default = float(
                np.median(list(ms_per_timebin_by_file.values()))
            )

        meta = {
            "run_name": self.config.get("run_name"),
            "mode": self.config.get("mode"),
            "num_classes": int(self.model.num_classes),
            "patch_width": 1,
            "n_timebins": int(window_timebins),
            "final_weight_path": str(final_path) if final_path is not None else None,
            "export_strategy": "tiled_full_file",
            "export_stride_timebins": int(window_timebins),
            "ms_per_timebin_default": float(ms_per_timebin_default),
            "ms_per_timebin_by_file": "ms_per_timebin_by_file.json",
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="AVES supervised training")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--annotation_file", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["detect", "unit_detect", "classify"])
    parser.add_argument("--wav_root", type=str, required=True)
    parser.add_argument(
        "--wav_manifest",
        type=str,
        default=None,
        help="Optional manifest mapping stems to wav paths (JSON dict/list or TSV).",
    )
    parser.add_argument("--wav_exts", type=str, default=".wav,.flac,.ogg,.mp3")
    parser.add_argument("--aves_model_path", type=str, required=True)
    parser.add_argument("--aves_config_path", type=str, required=True)

    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=25)
    # warmup/min_lr/early_stop removed: keep training schedule fixed.

    parser.add_argument("--audio_sr", type=int, default=16000)
    parser.add_argument("--num_timebins", type=int, default=0, help="window size in spectrogram timebins for val export (0 = infer)")
    parser.add_argument(
        "--ms_per_timebin",
        type=float,
        default=20.0,
        help="Fallback ms per spectrogram timebin for val export.",
    )
    parser.add_argument("--clip_seconds", type=float, default=None)
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument("--encoder_layer_idx", type=int, default=None)
    parser.add_argument("--linear_probe", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_val_files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_f1", action="store_true")
    parser.add_argument(
        "--class_weighting",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save_intermediate_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save_val_logits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="export val_outputs/ for eval CSV machinery (default: enabled)",
    )

    args = parser.parse_args()

    if "--lr" not in sys.argv:
        if args.linear_probe and not args.finetune:
            args.lr = 1e-2

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    wav_exts = tuple(e.strip() for e in args.wav_exts.split(",") if e.strip())

    train_ds = AvesSupervisedDataset(
        spec_dir=args.train_dir,
        wav_root=args.wav_root,
        annotation_file=args.annotation_file,
        mode=args.mode,
        audio_sr=args.audio_sr,
        wav_exts=wav_exts,
        wav_manifest=args.wav_manifest,
        clip_seconds=args.clip_seconds,
        is_train=True,
        max_files=args.max_train_files,
    )
    val_ds = AvesSupervisedDataset(
        spec_dir=args.val_dir,
        wav_root=args.wav_root,
        annotation_file=args.annotation_file,
        mode=args.mode,
        audio_sr=args.audio_sr,
        wav_exts=wav_exts,
        wav_manifest=args.wav_manifest,
        clip_seconds=args.clip_seconds,
        is_train=False,
        max_files=args.max_val_files,
    )

    val_batch_size = args.val_batch_size if args.val_batch_size > 0 else args.batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=aves_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=aves_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    model = AvesClassifier(
        config_path=args.aves_config_path,
        model_path=args.aves_model_path,
        num_classes=train_ds.num_classes,
        mode=args.mode,
        embedding_dim=args.embedding_dim,
        trainable=args.finetune,
        linear_probe=args.linear_probe,
        encoder_layer_idx=args.encoder_layer_idx,
        hidden_dim=args.hidden_dim,
    )

    config = vars(args)
    config["num_classes"] = int(train_ds.num_classes)

    trainer = AvesTrainer(model, config)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
