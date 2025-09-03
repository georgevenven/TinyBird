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

    def _load_file(self, path):
        """Helper function to load a single file"""
        try:
            f = torch.load(path, map_location="cpu", weights_only=False)
            return f['s'].flatten()
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    # GPT Generated function  
    def _compute_stats(self):
        """Compute mean and std across the entire dataset for z-score normalization"""
        print(f"Computing dataset statistics across {len(self.file_dirs)} files...")
        
        # Initialize running statistics
        running_sum = 0.0
        running_sum_sq = 0.0
        total_elements = 0
        
        # Use multiprocessing for parallel file loading
        num_workers = min(os.cpu_count(), 8)  # Don't use too many cores
        
        # Process files in actual batches to be fast AND memory efficient
        for i in range(0, len(self.file_dirs), self.stats_batch_size):
            batch_paths = self.file_dirs[i:i+self.stats_batch_size]
            
            # Load batch of files in parallel
            with Pool(num_workers) as pool:
                batch_specs = pool.map(self._load_file, batch_paths)
            
            # Filter out None values (failed loads)
            batch_specs = [spec for spec in batch_specs if spec is not None]
            
            # Process entire batch at once (vectorized operations)
            if batch_specs:
                batch_tensor = torch.cat(batch_specs)
                running_sum += batch_tensor.sum().item()
                running_sum_sq += (batch_tensor ** 2).sum().item()
                total_elements += batch_tensor.numel()
                
                # Clear batch memory
                del batch_specs, batch_tensor
            
            # Progress update
            if (i // self.stats_batch_size) % 10 == 0:
                progress = min(i + self.stats_batch_size, len(self.file_dirs))
                print(f"Processed {progress}/{len(self.file_dirs)} files...")
        
        # Compute final statistics
        mean = running_sum / total_elements
        variance = (running_sum_sq / total_elements) - (mean ** 2)
        std = variance ** 0.5
        
        print(f"Dataset statistics - Mean: {mean:.4f}, Std: {std:.4f}")
        return torch.tensor(mean), torch.tensor(std)

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

        # Apply z-score normalization
        spec = (spec - self.mean) / self.std

        if self.pad_crop:
            spec = self.crop_or_pad(spec)
            assert spec.shape[0] == self.n_mels and spec.shape[1] == self.n_timebins

        spec = spec.unsqueeze(0) # since we are dealing with image data, conv requires channels 

        return spec, filename 

    def __len__(self):
        return len(self.file_dirs)