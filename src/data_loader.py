def batch_size_of(batch):
    if isinstance(batch, torch.Tensor):
        return batch.size(0)
    if isinstance(batch, (list, tuple)):
        return batch_size_of(batch[0])
    try:
        return len(batch)
    except Exception:
        raise TypeError(f"Unsupported batch element type for size detection: {type(batch)}")
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


def slice_batch_range(batch, start, end):
    """Slice batch elements along the batch dimension in [start:end]."""
    if isinstance(batch, torch.Tensor):
        return batch[start:end]
    if isinstance(batch, (list, tuple)):
        return type(batch)(slice_batch_range(x, start, end) for x in batch)
    # e.g. list[str] for filenames
    try:
        return batch[start:end]
    except Exception:
        raise TypeError(f"Unsupported batch element type: {type(batch)}")
     

class ChunkingLoader:
    """
    Wraps a base DataLoader(bsz=16). On iteration, yields either:
      - two consecutive micro-batches of size 8 (each with k∈[4,12]), or
      - one micro-batch of size 16 (with k∈[1,3]).
    Yields tuples: (micro_batch, k)
    """
    def __init__(self, base_loader, p_full16=0.5, k_small=(2,3), k_large=(4,12)):
        """
        p_full16: probability to emit a full-16 batch instead of two 8s.
        k_small:  inclusive range for k when batch size is 16
        k_large:  inclusive range for k when batch size is 8
        """
        self.base_loader = base_loader
        batch_size = self.base_loader.batch_size
        assert batch_size % 2 == 0, "Batch size must be even"
        self.full_size = batch_size
        self.half_size = batch_size // 2
        self.p_full16 = float(p_full16)
        self.k_small = tuple(k_small)
        self.k_large = tuple(k_large)

    def __iter__(self):
        base_iter = iter(self.base_loader)
        pending_8 = None  # cache second half if we decide to split

        while True:
            if pending_8 is not None:
                # Emit the cached second-8
                mb = pending_8
                pending_8 = None
                k = random.randint(self.k_large[0], self.k_large[1])
                yield mb, k
                continue

            # Fetch a fresh batch
            try:
                batch16 = next(base_iter)
            except StopIteration:
                # Restart epoch
                base_iter = iter(self.base_loader)
                batch16 = next(base_iter)

            bsz = batch_size_of(batch16)
            if bsz <= 0:
                continue
            if bsz < self.full_size:
                half = bsz // 2
                if half > 0 and bsz - half > 0:
                    mb1 = slice_batch_range(batch16, 0, half)
                    mb2 = slice_batch_range(batch16, half, bsz)
                    k1 = random.randint(self.k_large[0], self.k_large[1])
                    pending_8 = mb2
                    yield mb1, k1
                else:
                    k = random.randint(self.k_small[0], self.k_small[1])
                    yield batch16, k
            else:
                # Decide: full-size or split into two halves
                if random.random() < self.p_full16:
                    # Full-size → pick small k
                    k = random.randint(self.k_small[0], self.k_small[1])
                    yield batch16, k
                else:
                    # Split into two halves → pick large k for each half
                    mb1 = slice_batch_range(batch16, 0, self.half_size)
                    mb2 = slice_batch_range(batch16, self.half_size, self.full_size)
                    k1 = random.randint(self.k_large[0], self.k_large[1])
                    pending_8 = mb2
                    yield mb1, k1