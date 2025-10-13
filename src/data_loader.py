from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
import random
import numpy as np
from utils import load_audio_params

class SpectogramDataset(Dataset):
    def __init__(self, dir, n_timebins=1024):
        self.file_dirs = list(Path(dir).glob("*.npy"))

        # Load audio parameters using utility function
        self.audio_data_json = load_audio_params(dir)
        
        self.n_mels = self.audio_data_json["mels"]
        self.sr = self.audio_data_json["sr"]
        self.hop_size = self.audio_data_json["hop_size"]
        self.fft = self.audio_data_json["fft"]
        self.mean = self.audio_data_json["mean"]
        self.std = self.audio_data_json["std"]
        self.n_timebins = n_timebins
        self._mmaps = {}
        self._mean32 = np.float32(self.mean)
        self._std32 = np.float32(self.std)

        if len(self.file_dirs) == 0:
            raise SystemExit("no files!!")

    def _open(self, path: Path):
        key = str(path)
        mm = self._mmaps.get(key)
        if mm is None:
            mm = np.load(path, mmap_mode="r")
            self._mmaps[key] = mm
        return mm
            
    def __getitem__(self, index):
        path = self.file_dirs[index]   # pick actual .npy path

        try:
            arr = self._open(path)
        except Exception:
            index = random.randint(0, len(self.file_dirs)-1)
            path = self.file_dirs[index]
            arr = self._open(path)
        filename = path.stem

        time = arr.shape[1]
        if time > self.n_timebins:
            start = random.randint(0, time - self.n_timebins)
            end = start + self.n_timebins
            view = arr[:, start:end]
        else:
            view = arr[:, :self.n_timebins]

        out = np.empty((self.n_mels, self.n_timebins), dtype=np.float32)
        out.fill(0.0)
        out[:, :view.shape[1]] = view
        assert out.shape[0] == self.n_mels and out.shape[1] == self.n_timebins

        # Apply z-score normalization in-place
        out -= self._mean32
        out /= self._std32

        spec = torch.from_numpy(out).unsqueeze(0)  # since we are dealing with image data, conv requires channels 

        return spec, filename 

    def __len__(self):
        return len(self.file_dirs)
