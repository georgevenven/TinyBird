from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pathlib import Path
import torch
import random

class SpectogramDataset(Dataset):
    def __init__(self, dir, n_mels=128, n_timebins=1024, pad_crop=True):
        self.file_dirs = list(Path(dir).glob("*.pt"))
        self.n_mels = n_mels
        self.n_timebins = n_timebins
        self.pad_crop = pad_crop
        if len(self.file_dirs) == 0: raise("no files!!")
        
        # Compute dataset statistics
        self.mean, self.std = self._compute_stats()

    # GPT Generated function  
    def _compute_stats(self):
        """Compute mean and std across 2.5% of the dataset for z-score normalization"""
        all_values = []
        
        # Select 2.5% of files randomly for statistics computation
        subset_size = max(1, int(len(self.file_dirs) * 0.025))
        subset_files = random.sample(self.file_dirs, subset_size)
        
        print(f"Computing dataset statistics on {subset_size} files (2.5% of {len(self.file_dirs)} total files)...")
        for path in subset_files:
            f = torch.load(path, map_location="cpu", weights_only=False)
            spec = f['s']
            all_values.append(spec.flatten())
        
        # Concatenate all values and compute statistics
        all_values = torch.cat(all_values)
        mean = all_values.mean()
        std = all_values.std()
        
        print(f"Dataset statistics (from 2.5% subset) - Mean: {mean:.4f}, Std: {std:.4f}")
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

    def pad_chirp_intervals(self, chirp_intervals, max_N, pad_value=-1):
        """
        Pad chirp_intervals (N, 2) to shape (max_N, 2) with pad_value.
        
        Args:
            chirp_intervals (torch.Tensor): Tensor of shape (N, 2).
            max_N (int): Target number of rows after padding.
            pad_value (float or int): Value to use for padding.

        Returns:
            padded (torch.Tensor): Tensor of shape (max_N, 2).
            length (int): Original number of intervals before padding.
        """
        N, dim = chirp_intervals.shape
        assert dim == 2, f"Expected shape (N, 2), got {chirp_intervals.shape}"

        # Allocate with pad_value
        padded = torch.full((max_N, 2), pad_value, dtype=chirp_intervals.dtype)

        # Copy as much as fits
        n_copy = min(N, max_N)
        padded[:n_copy] = chirp_intervals[:n_copy]

        return padded, N


    def __getitem__(self, index):
        path = self.file_dirs[index]   # pick actual .pt path

        try:
            f=torch.load(path, map_location="cpu",weights_only=False)
        except:
            index = random.randint(0, len(self.file_dirs)-1)
            path = self.file_dirs[index]
            f=torch.load(path, map_location="cpu",weights_only=False)
        spec = f['s']
        chirp_intervals , N  =   self.pad_chirp_intervals(f['chirp_intervals'], self.n_timebins)
        filename = path.stem

        # Apply z-score normalization
        spec = (spec - self.mean) / self.std

        if self.pad_crop:
            spec = self.crop_or_pad(spec)
            assert spec.shape[0] == self.n_mels and spec.shape[1] == self.n_timebins

        spec = spec.unsqueeze(0) # since we are dealing with image data, conv requires channels 

        return spec, chirp_intervals, N , filename 

    def __len__(self):
        return len(self.file_dirs)