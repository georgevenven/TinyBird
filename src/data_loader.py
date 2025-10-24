from torch.utils.data import Dataset
import torch.nn.functional as F
from pathlib import Path
import torch
import random

def batch_size_of(batch):
    if isinstance(batch, torch.Tensor):
        return batch.size(0)
    if isinstance(batch, (list, tuple)):
        return batch_size_of(batch[0])
    try:
        return len(batch)
    except Exception:
        raise TypeError(f"Unsupported batch element type for size detection: {type(batch)}")

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

    def crop_or_pad_pair(self, spec, labels_1d, pad_value_labels=-1):
        """
        Crop/pad spectrogram and its 1-D per-time labels together to n_timebins.
        spec: (F, T), labels_1d: (T,)
        Returns: (spec_out: (F, n_timebins), labels_out: (n_timebins,))
        """
        frq, time = spec.shape
        assert labels_1d.ndim == 1, f"labels must be 1-D, got {labels_1d.shape}"
        assert labels_1d.shape[0] == time, \
            f"labels length {labels_1d.shape[0]} must match spec time {time}"

        if time < self.n_timebins:
            padding_amnt = self.n_timebins - time
            spec_out = F.pad(spec, (0, padding_amnt, 0, 0))
            labels_out = F.pad(labels_1d, (0, padding_amnt), value=pad_value_labels)
            return spec_out, labels_out

        elif time > self.n_timebins:
            start = random.randint(0, time - self.n_timebins)
            end = start + self.n_timebins
            return spec[:, start:end], labels_1d[start:end]

        else:
            return spec, labels_1d     

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

    def pad_vector(self, vec, max_N, pad_value=-1):
        """Pad 1-D tensor vec to length max_N with pad_value. Returns (padded, length)."""
        vec = torch.as_tensor(vec)
        assert vec.ndim == 1, f"Expected 1-D vector, got {vec.shape}"
        N = vec.shape[0]
        padded = torch.full((max_N,), pad_value, dtype=vec.dtype)
        n_copy = min(N, max_N)
        padded[:n_copy] = vec[:n_copy]
        return padded, N

    def pad_chirp_feats(self, chirp_feats, max_N, pad_value=-1.0):
        """
        Pad chirp_feats (N, 2, 6) to shape (max_N, 2, 6) with pad_value.

        Args:
            chirp_feats (torch.Tensor): Tensor of shape (N, 2, 6).
            max_N (int): Target number of rows after padding.
            pad_value (float or int): Value to use for padding.

        Returns:
            padded (torch.Tensor): Tensor of shape (max_N, 2, 6).
            length (int): Original number of rows N before padding.
        """
        feats = torch.as_tensor(chirp_feats)
        assert feats.ndim == 3 and feats.shape[1] == 2 and feats.shape[2] == 6, \
            f"Expected (N, 2, 6), got {feats.shape}"
        N = feats.shape[0]
        padded = torch.full((max_N, 2, 6), pad_value, dtype=feats.dtype)
        n_copy = min(N, max_N)
        padded[:n_copy] = feats[:n_copy]
        return padded, N


    def __getitem__(self, index):
        path = self.file_dirs[index]   # pick actual .pt path

        try:
            f = torch.load(path, map_location="cpu", weights_only=False)
        except:
            index = random.randint(0, len(self.file_dirs)-1)
            path = self.file_dirs[index]
            f = torch.load(path, map_location="cpu", weights_only=False)

        spec = f['s']
        # Convert to torch if numpy
        if not isinstance(spec, torch.Tensor):
            spec = torch.as_tensor(spec)

        # # Intervals (Nx2)
        chirp_int_np = f['chirp_intervals']
        if isinstance(chirp_int_np, torch.Tensor):
            chirp_int = chirp_int_np
        else:
            chirp_int = torch.as_tensor(chirp_int_np)
        chirp_intervals, N = self.pad_chirp_intervals(chirp_int, self.n_timebins)

        # --- Chirp labels (per-time, length T) ---
        if 'chirp_labels' in f:
            cl_np = f['chirp_labels']
            chirp_labels_time = cl_np if isinstance(cl_np, torch.Tensor) else torch.as_tensor(cl_np)
            chirp_labels_time = chirp_labels_time.to(dtype=torch.int32)
        else:
            # Backward-compat: if missing, default to zeros of current spec width
            chirp_labels_time = torch.zeros((spec.shape[1],), dtype=torch.int32)


        # Chirp features remain per-interval; pad as before
        if 'chirp_feats' in f:
            cf_np = f['chirp_feats']
            chirp_feats = cf_np if isinstance(cf_np, torch.Tensor) else torch.as_tensor(cf_np)
            assert chirp_feats.ndim == 3 and chirp_feats.shape[1] == 2 and chirp_feats.shape[2] == 6, \
                f"chirp_feats must be 3D (N,2,6), got {chirp_feats.shape}"
        else:
            orig_N = chirp_int.shape[0]
            chirp_feats = torch.zeros((orig_N, 2, 6), dtype=torch.float32)
        chirp_feats_pad, N_feats = self.pad_chirp_feats(chirp_feats, self.n_timebins, pad_value=-1.0)

        # Apply z-score normalization to spectrogram
        spec = (spec - self.mean) / self.std

        # Pairwise crop/pad spec and labels together (time-aligned)
        if self.pad_crop:
            spec, chirp_labels_pad = self.crop_or_pad_pair(spec, chirp_labels_time, pad_value_labels=-1)
            assert spec.shape[0] == self.n_mels and spec.shape[1] == self.n_timebins
            assert chirp_labels_pad.shape[0] == self.n_timebins
        else:
            # If not cropping/padding, still ensure equal time length for safety
            T = spec.shape[1]
            if chirp_labels_time.shape[0] != T:
                # Hard align by simple right-pad/truncate to spec T (shouldn't normally happen)
                chirp_labels_pad, _ = self.pad_vector(chirp_labels_time, T, pad_value=-1)
            else:
                chirp_labels_pad = chirp_labels_time

        # Final consistency for interval-derived tensors (labels are per-time now)
        N = min(N, N_feats)
        chirp_intervals = chirp_intervals[:self.n_timebins]
        chirp_feats_pad = chirp_feats_pad[:self.n_timebins]

        filename = path.stem
        spec = spec.unsqueeze(0)
        return spec, chirp_intervals, chirp_labels_pad, chirp_feats_pad, N, filename

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