## LATER TO DO:
### THESE TWO CLASSES SHOULD JUST INHERIT A MORE BASIC CLASS
##

import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
import random
import numpy as np
from utils import load_audio_params, parse_chunk_ms, clip_labels_to_chunk

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
    def __init__(
        self,
        dir,
        annotation_file_path,
        n_timebins=1024,
        mode="detect",
        white_noise=0.0,
        audio_params_override=None,
    ):
        """
        n_timebins = None means no cropping
        white_noise: standard deviation of white noise to add after normalization (0.0 = no noise)
        """
        self.file_dirs = sorted(list(Path(dir).glob("*.npy")))

        # Load audio parameters using utility function (or override with pretrain params)
        if audio_params_override is None:
            self.audio_data_json = load_audio_params(dir)
        else:
            self.audio_data_json = audio_params_override
        
        self.n_mels = self.audio_data_json["mels"]
        self.sr = self.audio_data_json["sr"]
        self.hop_size = self.audio_data_json["hop_size"]
        self.fft = self.audio_data_json["fft"]
        self.mean = self.audio_data_json["mean"]
        self.std = self.audio_data_json["std"]
        self.n_timebins = n_timebins
        self.mean = np.float32(self.mean)
        self.std = np.float32(self.std)

        self.mode = mode ## detect = vocalization present/absent, unit_detect = syllable present/absent, classify = syllable class
        self.annotation_file_path = annotation_file_path
        self.white_noise = white_noise

        with open(annotation_file_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        self._label_index = self._build_label_index(annotations, mode)
        
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

    @staticmethod
    def _build_label_index(annotations, mode):
        if mode not in ["detect", "classify", "unit_detect"]:
            raise ValueError("mode must be 'detect', 'classify', or 'unit_detect'")

        label_index = {}
        for rec in annotations.get("recordings", []):
            rec_filename = Path(rec["recording"]["filename"]).stem
            events = rec.get("detected_events", [])
            if mode == "detect":
                labels = [
                    {"onset_ms": event["onset_ms"], "offset_ms": event["offset_ms"]}
                    for event in events
                ]
            else:
                labels = [unit for event in events for unit in event.get("units", [])]
            label_index[rec_filename] = labels
        return label_index

    def create_label_array(self, labels, start_bin, end_bin):
        """
        Create a 1D array of labels matching the spectrogram time dimension
        """
        window_len = end_bin - start_bin
        if window_len <= 0:
            return np.zeros(0, dtype=np.int64)
        
        # Initialize label array with silence (class 0)
        if self.mode in ["detect", "unit_detect"]:
            label_arr = np.full(window_len, 0, dtype=np.int64)  # 0 = silence
        else:  # classify
            label_arr = np.full(window_len, 0, dtype=np.int64)  # 0 = silence
        
        # Fill in labels based on onset/offset
        for label in labels:
            onset_bin = self.ms_to_timebins(label["onset_ms"])
            offset_bin = self.ms_to_timebins(label["offset_ms"])
            if offset_bin <= start_bin or onset_bin >= end_bin:
                continue
            onset_bin = max(onset_bin, start_bin) - start_bin
            offset_bin = min(offset_bin, end_bin) - start_bin
            
            if self.mode in ["detect", "unit_detect"]:
                label_arr[onset_bin:offset_bin] = 1  # 1 = present (vocalization or unit)
            else:  # classify
                label_arr[onset_bin:offset_bin] = label["id"] + 1  # shift by +1, so classes are 1, 2, 3, ...
        
        return label_arr
            
    def __getitem__(self, index):
        path = self.file_dirs[index]  

        filename = path.stem
        arr = np.load(path, mmap_mode="r")
        time = arr.shape[1]

        base_filename, chunk_start_ms, chunk_end_ms = parse_chunk_ms(filename)
        labels = self._label_index.get(base_filename)
        if labels is None:
            raise ValueError(f"No matching recording found for: {base_filename}")
        labels = clip_labels_to_chunk(labels, chunk_start_ms, chunk_end_ms)

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
            elif time == self.n_timebins:
                start = 0
                end = time
            else:
                start = 0
                end = time

        # Create label array matching spectrogram time dimension
        labels = self.create_label_array(labels, start, end)

        # Crop/pad spectrograms and labels
        if self.n_timebins is not None:
            if time > self.n_timebins:
                arr = arr[:, start:end]
            elif time < self.n_timebins:
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
