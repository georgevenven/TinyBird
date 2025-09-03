from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pathlib import Path
import torch
import random
from multiprocessing import Pool
import os

class SpectogramDataset(Dataset):
    def __init__(self, dir, n_mels=128, n_timebins=1024, pad_crop=True, stats_batch_size=2048):
        self.file_dirs = list(Path(dir).glob("*.pt"))
        self.n_mels = n_mels
        self.n_timebins = n_timebins
        self.pad_crop = pad_crop
        self.stats_batch_size = stats_batch_size
        if len(self.file_dirs) == 0: raise("no files!!")
        
        # Compute dataset statistics
        self.mean, self.std = self._compute_stats()

    def _stats_worker(self, path):
        """Worker function for computing per-mel statistics"""
        try:
            x = torch.load(path, map_location="cpu", weights_only=False)["s"].to(torch.float32)  # [mels, time]
            s1 = x.sum(dim=1).to(torch.float64)           # per-mel sum
            s2 = (x * x).sum(dim=1).to(torch.float64)     # per-mel sum of squares
            n  = x.shape[1]                                # timebins per mel
            return s1, s2, n
        except Exception:
            return None

    # GPT Generated function  
    def _compute_stats(self):
        """Compute per-mel mean and std across the entire dataset for z-score normalization"""
        print(f"Computing per-mel dataset statistics across {len(self.file_dirs)} files...")
        
        workers = min(os.cpu_count() or 1, 8)
        sum1 = torch.zeros(self.n_mels, dtype=torch.float64)
        sum2 = torch.zeros(self.n_mels, dtype=torch.float64)
        count_timebins = 0
        
        with Pool(processes=workers) as pool:
            for res in pool.imap_unordered(self._stats_worker, self.file_dirs, chunksize=self.stats_batch_size):
                if res is None:
                    continue
                s1, s2, n = res
                # allow variable time length; all mels must match n_mels
                if s1.numel() != self.n_mels:
                    continue
                sum1 += s1
                sum2 += s2
                count_timebins += n
        
        if count_timebins == 0:
            raise RuntimeError("No readable tensors to compute stats")
        
        mean = (sum1 / count_timebins).to(torch.float32)                    # [mels]
        var  = (sum2 / count_timebins - (sum1 / count_timebins) ** 2)       # [mels], float64
        var.clamp_(min=0.0)
        std  = var.sqrt().to(torch.float32)
        std = torch.maximum(std, torch.tensor(1e-8, dtype=torch.float32))
        
        print(f"Dataset statistics computed - Mean shape: {mean.shape}, Std shape: {std.shape}")
        return mean, std

    # time only crop / pads, if mels are wrong assert will catch
    def crop_or_pad(self, spec):
        frq, time = spec.shape

        if time < self.n_timebins:
            padding_amnt = self.n_timebins - time 
            spec = F.pad(spec, (0, padding_amnt, 0, 0)) # 0 pad to the left, padding to the right, nothing on top or bottom 

        elif time > self.n_timebins:
            start = random.randint(0, time - self.n_timebins)
            end = start + self.n_timebins
            spec = spec[:, start:end]
        
        return spec

    def __getitem__(self, index):
        path = self.file_dirs[index]   # pick actual .pt path

        try:
            f=torch.load(path, map_location="cpu",weights_only=False)
        except:
            index = random.randint(0, len(self.file_dirs)-1)
            path = self.file_dirs[index]
            f=torch.load(path, map_location="cpu",weights_only=False)
        spec = f['s']
        filename = path.stem

        # Apply per-mel z-score normalization
        spec = (spec - self.mean.unsqueeze(1)) / self.std.unsqueeze(1)

        if self.pad_crop:
            spec = self.crop_or_pad(spec)
            assert spec.shape[0] == self.n_mels and spec.shape[1] == self.n_timebins

        spec = spec.unsqueeze(0) # since we are dealing with image data, conv requires channels 

        return spec, filename 

    def __len__(self):
        return len(self.file_dirs)