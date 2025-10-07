from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pathlib import Path
import torch
import random
from utils import load_audio_params

class SpectogramDataset(Dataset):
    def __init__(self, dir, n_timebins=1024, pad_crop=True):
        self.file_dirs = list(Path(dir).glob("*.pt"))

        # Load audio parameters using utility function
        self.audio_data_json = load_audio_params(dir)
        
        self.n_mels = self.audio_data_json["mels"]
        self.sr = self.audio_data_json["sr"]
        self.hop_size = self.audio_data_json["hop_size"]
        self.fft = self.audio_data_json["fft"]
        self.mean = self.audio_data_json["mean"]
        self.std = self.audio_data_json["std"]
        self.n_timebins = n_timebins
        self.pad_crop = pad_crop

        if len(self.file_dirs) == 0:
            raise SystemExit("no files!!")
            
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