from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
import random
import numpy as np
from utils import load_audio_params

class SpectogramDataset(Dataset):
    def __init__(self, dir, n_timebins=1024):
        """
        n_timebins = None means no cropping
        """
        self.file_dirs = sorted(list(Path(dir).glob("*.npy")))

        # Load audio parameters using utility function
        self.audio_data_json = load_audio_params(dir)
        
        self.n_mels = self.audio_data_json["mels"]
        self.sr = self.audio_data_json["sr"]
        self.hop_size = self.audio_data_json["hop_size"]
        self.fft = self.audio_data_json["fft"]
        self.mean = self.audio_data_json["mean"]
        self.std = self.audio_data_json["std"]
        self.n_timebins = n_timebins
        self.mean = np.float32(self.mean)
        self.std = np.float32(self.std)

        if len(self.file_dirs) == 0:
            raise SystemExit("no files!!")
            
    def __getitem__(self, index):
        path = self.file_dirs[index]  

        # if file not possible to open pick a random file from the list and try again, this actually shoudl recursively recall getitem 
        filename = path.stem
        arr = np.load(path, mmap_mode="r")
        time = arr.shape[1]

        # we want to load the whole file if n_timebins is None
        if self.n_timebins is None:
            start = 0
            end = time

        # loading a chunk 
        else:
            # crop 
            if time > self.n_timebins:
                start = random.randint(0, time - self.n_timebins)
                end = start + self.n_timebins
                arr = arr[:,start:end]

            # do nothing 
            if time == self.n_timebins:
                pass 

            # pad 
            if time < self.n_timebins:
                start = 0
                end = self.n_timebins
                pad_amount = self.n_timebins - arr.shape[1]
                arr = np.pad(arr, ((0, 0), (0, pad_amount)), mode='constant')

        arr = np.array(arr, dtype=np.float32)

        # Apply z-score normalization in-place
        arr -= self.mean
        arr /= self.std

        spec = torch.from_numpy(arr).unsqueeze(0)  # since we are dealing with image data, conv requires channels 

        return spec, filename 

    def __len__(self):
        return len(self.file_dirs)
