from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pathlib import Path
import torch
import random

# all we have to do is return images of size n_mel x n_timebins , all other ops will be handled by HF 
class SpectogramDataset(Dataset):
    def __init__(self, dir, n_mels=128, n_timebins=1000):
        self.file_dirs = list(Path(dir).glob("*.pt"))
        self.n_mels = n_mels
        self.n_timebins = n_timebins
        if len(self.file_dirs) == 0: raise("no files!!")
        
        # Compute dataset statistics
        self.mean, self.std = self._compute_stats()

    # GPT Generated function  
    def _compute_stats(self):
        """Compute mean and std across the entire dataset for z-score normalization"""
        all_values = []
        
        print(f"Computing dataset statistics across {len(self.file_dirs)} files...")
        for path in self.file_dirs:
            f = torch.load(path, map_location="cpu", weights_only=False)
            spec = f['s']
            all_values.append(spec.flatten())
        
        # Concatenate all values and compute statistics
        all_values = torch.cat(all_values)
        mean = all_values.mean()
        std = all_values.std()
        
        print(f"Dataset statistics - Mean: {mean:.4f}, Std: {std:.4f}")
        return mean, std

    # time only crop / pads, if mels are wrong assert will catch
    def crop_or_pad(self, spec, labels):
        frq, time = spec.shape

        if time < self.n_timebins:
            padding_amnt = self.n_timebins - time 
            spec = F.pad(spec, (0, padding_amnt, 0, 0)) # 0 pad to the left, padding to the right, nothing on top or bottom 
            labels = F.pad(labels, (0, padding_amnt)) 

        elif time > self.n_timebins:
            start = random.randint(0, time - self.n_timebins)
            end = start + self.n_timebins
            spec, labels = spec[:, start:end], labels[start:end]
        
        return spec, labels

    def __getitem__(self, index):
        path = self.file_dirs[index]   # pick actual .pt path

        f=torch.load(path, map_location="cpu",weights_only=False)
        spec = f['s']
        labels = f['labels']
        if labels is None:
            labels = torch.zeros(spec.shape[1], dtype=torch.int32)

        spec, labels = self.crop_or_pad(spec, labels)

        # Apply z-score normalization
        spec = (spec - self.mean) / self.std

        assert spec.shape[0] == self.n_mels and spec.shape[1] == self.n_timebins and labels.shape[0] == self.n_timebins

        spec = spec.unsqueeze(0) # since we are dealing with image data 

        return spec, labels 

    def __len__(self):
        return len(self.file_dirs)