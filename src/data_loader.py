## LATER TO DO:
### THESE TWO CLASSES SHOULD JUST INHERIT A MORE BASIC CLASS
##

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
import random
import numpy as np
from utils import load_audio_params, load_audio_labels

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

class SupervisedSpectogramDataset(Dataset):
    def __init__(self, dir, annotation_file_path, n_timebins=1024, mode="detect", white_noise=0.0):
        """
        n_timebins = None means no cropping
        white_noise: standard deviation of white noise to add after normalization (0.0 = no noise)
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

        self.mode = mode ## if classify, means classify syllable labels, if detect, means detect onset offset of vocalizations 
        self.annotation_file_path = annotation_file_path
        self.white_noise = white_noise
        
        # Automatically determine number of classes from annotations
        from utils import get_num_classes_from_annotations
        self.num_classes = get_num_classes_from_annotations(annotation_file_path, mode)

        if len(self.file_dirs) == 0:
            raise SystemExit("no files!!")

    def ms_to_timebins(self, ms_value):
        """
        purpose: converts ms value to timebin value 

        the formula of converting ms to timebins:
        time_bin = (time_ms / 1000) × sample_rate / hop_length

        audio_params (tuple) = sr, n_mels, hop_size, fft
        """

        # int rounding is floor rounding ... could be a point of error 
        return int((ms_value / 1000) * self.sr / self.hop_size)

    def create_label_array(self, filename, time_length):
        """
        Create a 1D array of labels matching the spectrogram time dimension
        """
        labels = load_audio_labels(self.annotation_file_path, filename, mode=self.mode)
        
        # Initialize label array with silence (class 0)
        if self.mode == "detect":
            label_arr = np.full(time_length, 0, dtype=np.int64)  # 0 = silence
        else:  # classify
            label_arr = np.full(time_length, 0, dtype=np.int64)  # 0 = silence
        
        # Fill in labels based on onset/offset
        for label in labels:
            onset_bin = self.ms_to_timebins(label["onset_ms"])
            offset_bin = self.ms_to_timebins(label["offset_ms"])
            
            if self.mode == "detect":
                label_arr[onset_bin:offset_bin] = 1  # 1 = vocalization
            else:  # classify
                label_arr[onset_bin:offset_bin] = label["id"] + 1  # shift by +1, so classes are 1, 2, 3, ...
        
        return label_arr
            
    def __getitem__(self, index):
        path = self.file_dirs[index]  

        filename = path.stem
        arr = np.load(path, mmap_mode="r")
        time = arr.shape[1]

        # Create label array matching spectrogram time dimension
        labels = self.create_label_array(filename, time)

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
                labels = labels[start:end]

            # do nothing 
            if time == self.n_timebins:
                pass 

            # pad 
            if time < self.n_timebins:
                start = 0
                end = self.n_timebins
                pad_amount = self.n_timebins - arr.shape[1]
                arr = np.pad(arr, ((0, 0), (0, pad_amount)), mode='constant')
                labels = np.pad(labels, (0, pad_amount), mode='constant', constant_values=0)  # pad with silence (class 0)

        arr = np.array(arr, dtype=np.float32)

        # Apply z-score normalization in-place
        arr -= self.mean
        arr /= self.std

        # Apply white noise augmentation if enabled
        if self.white_noise > 0.0:
            noise = np.random.normal(0, self.white_noise, arr.shape).astype(np.float32)
            arr += noise

        spec = torch.from_numpy(arr).unsqueeze(0)
        labels = torch.from_numpy(labels)

        return spec, labels, filename

    def __len__(self):
        return len(self.file_dirs)
