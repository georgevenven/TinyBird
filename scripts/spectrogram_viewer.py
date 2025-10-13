#!/usr/bin/env python3
"""
Spectrogram Viewer - Navigate through spectrograms with arrow keys

Usage:
    python scripts/spectrogram_viewer.py <spectrogram_directory>

Controls:
    Left/Right Arrow: Navigate between files
    Up/Down Arrow: Zoom in/out frequency axis
    Q/Escape: Quit
    H: Show help
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Cursor


class SpectrogramViewer:
    def __init__(self, spec_dir: str):
        self.spec_dir = Path(spec_dir)
        if not self.spec_dir.exists():
            raise FileNotFoundError(f"Directory not found: {spec_dir}")
        
        # Load audio parameters
        self.audio_params = self._load_audio_params()
        
        # Find all spectrogram files
        self.files = self._find_spectrogram_files()
        if not self.files:
            raise FileNotFoundError(f"No spectrogram files found in {spec_dir}")
        
        self.current_idx = 0
        self.freq_zoom = 1.0  # Frequency zoom level
        self.time_zoom = 1.0  # Time zoom level
        
        # Set up the plot
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Add cursor for coordinate display
        self.cursor = Cursor(self.ax, useblit=True, color='white', linewidth=1)
        
        # Initialize colorbar placeholder
        self.cbar = None
        
        # Initialize normalization state and statistics
        self.show_normalized = False
        self.dataset_stats = None
        
        # Load and display first spectrogram
        self._update_display()
        
        print("Spectrogram Viewer Controls:")
        print("  Left/Right Arrow: Navigate between files")
        print("  Up/Down Arrow: Zoom frequency axis")
        print("  Shift+Up/Down: Zoom time axis")
        print("  N: Toggle normalization (raw vs z-score normalized)")
        print("  R: Reset zoom")
        print("  H: Show help")
        print("  Q/Escape: Quit")
        print(f"\nLoaded {len(self.files)} spectrograms from {spec_dir}")
        print("\nTransformations applied in audio2spec.py:")
        print("  • Mel-scale frequency conversion (if enabled)")
        print("  • Power spectrum (power=2.0)")
        print("  • dB conversion (ref=max)")
        print("\nNote: Use 'N' to see dataset z-score normalization preview")
    
    def _load_audio_params(self) -> Dict[str, Any]:
        """Load audio processing parameters from audio_params.json"""
        params_file = self.spec_dir / "audio_params.json"
        if params_file.exists():
            with open(params_file, 'r') as f:
                return json.load(f)
        else:
            # Default parameters if file doesn't exist
            print("Warning: audio_params.json not found, using defaults")
            return {
                "sr": 32000,
                "mels": 128,
                "hop_size": 64,
                "fft": 1024
            }
    
    def _find_spectrogram_files(self) -> List[Path]:
        """Find all .npy files in the directory"""
        return sorted(self.spec_dir.glob("*.npy"))
    
    def _load_spectrogram(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load spectrogram from .npy file"""
        spec = np.load(file_path).astype(np.float32)
        labels = np.zeros(spec.shape[1], dtype=np.int32)
        return spec, labels
    
    def _get_time_axis(self, n_frames: int) -> np.ndarray:
        """Generate time axis in seconds"""
        hop_size = self.audio_params.get("hop_size", 64)
        sr = self.audio_params.get("sr", 32000)
        hop_duration = hop_size / sr
        return np.arange(n_frames) * hop_duration
    
    def _get_freq_axis(self, n_freq_bins: int) -> np.ndarray:
        """Generate frequency axis"""
        sr = self.audio_params.get("sr", 32000)
        if "mels" in self.audio_params and n_freq_bins == self.audio_params["mels"]:
            # Mel scale - approximate frequency mapping
            mel_freqs = np.linspace(0, sr/2, n_freq_bins)
            return mel_freqs
        else:
            # Linear frequency scale
            return np.linspace(0, sr/2, n_freq_bins)
    
    def _compute_dataset_stats(self) -> Tuple[float, float]:
        """Compute dataset statistics for z-score normalization (similar to data_loader.py)"""
        if self.dataset_stats is not None:
            return self.dataset_stats
        
        print("Computing dataset statistics (2.5% sample)...")
        all_values = []
        
        # Select 2.5% of files randomly for statistics computation
        subset_size = max(1, int(len(self.files) * 0.025))
        subset_files = np.random.choice(self.files, size=subset_size, replace=False)
        
        for file_path in subset_files:
            try:
                spec, _ = self._load_spectrogram(file_path)
                all_values.append(spec.flatten())
            except Exception as e:
                print(f"Warning: Could not load {file_path.name}: {e}")
                continue
        
        if not all_values:
            print("Warning: No files could be loaded for statistics")
            return 0.0, 1.0
        
        # Concatenate all values and compute statistics
        all_values = np.concatenate(all_values)
        mean = np.mean(all_values)
        std = np.std(all_values)
        
        self.dataset_stats = (mean, std)
        print(f"Dataset statistics - Mean: {mean:.4f}, Std: {std:.4f}")
        return mean, std
    
    def _update_display(self):
        """Update the display with current spectrogram"""
        if not self.files:
            return
        
        current_file = self.files[self.current_idx]
        
        try:
            spec, labels = self._load_spectrogram(current_file)
        except Exception as e:
            print(f"Error loading {current_file.name}: {e}")
            return
        
        # Apply z-score normalization if requested
        if self.show_normalized:
            mean, std = self._compute_dataset_stats()
            spec = (spec - mean) / std
        
        # Clear the figure completely to avoid colorbar issues
        self.fig.clear()
        
        # Recreate the main axes
        self.ax = self.fig.add_subplot(111)
        
        # Recreate cursor for coordinate display
        self.cursor = Cursor(self.ax, useblit=True, color='white', linewidth=1)
        
        # Generate axes
        time_axis = self._get_time_axis(spec.shape[1])
        freq_axis = self._get_freq_axis(spec.shape[0])
        
        # Display spectrogram
        im = self.ax.imshow(
            spec, 
            aspect='auto', 
            origin='lower',
            extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
            cmap='viridis',
            interpolation='nearest'
        )
        
        # Apply zoom
        if self.freq_zoom != 1.0:
            ylim = self.ax.get_ylim()
            center_freq = (ylim[0] + ylim[1]) / 2
            freq_range = (ylim[1] - ylim[0]) / self.freq_zoom
            self.ax.set_ylim(center_freq - freq_range/2, center_freq + freq_range/2)
        
        if self.time_zoom != 1.0:
            xlim = self.ax.get_xlim()
            center_time = (xlim[0] + xlim[1]) / 2
            time_range = (xlim[1] - xlim[0]) / self.time_zoom
            self.ax.set_xlim(center_time - time_range/2, center_time + time_range/2)
        
        # Add colorbar (create fresh each time)
        self.cbar = self.fig.colorbar(im, ax=self.ax, label='Magnitude (dB)')
        
        # Labels and title
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Frequency (Hz)')
        
        # Create title with file info
        mode_str = "Z-score Normalized" if self.show_normalized else "Raw (dB)"
        title = f"{current_file.name} ({self.current_idx + 1}/{len(self.files)}) | {mode_str}"
        if np.any(labels > 0):
            unique_labels = np.unique(labels[labels > 0])
            title += f" | Labels: {list(unique_labels)}"
        title += f" | Shape: {spec.shape}"
        
        self.ax.set_title(title, fontsize=10)
        
        # Add grid
        self.ax.grid(True, alpha=0.3)
        
        # Refresh display
        self.fig.canvas.draw()
    
    def _on_key_press(self, event):
        """Handle key press events"""
        if event.key == 'right':
            self.current_idx = (self.current_idx + 1) % len(self.files)
            self._update_display()
        
        elif event.key == 'left':
            self.current_idx = (self.current_idx - 1) % len(self.files)
            self._update_display()
        
        elif event.key == 'up':
            if 'shift' in str(event.key):
                self.time_zoom *= 1.2
            else:
                self.freq_zoom *= 1.2
            self._update_display()
        
        elif event.key == 'down':
            if 'shift' in str(event.key):
                self.time_zoom /= 1.2
            else:
                self.freq_zoom /= 1.2
            self._update_display()
        
        elif event.key == 'shift+up':
            self.time_zoom *= 1.2
            self._update_display()
        
        elif event.key == 'shift+down':
            self.time_zoom /= 1.2
            self._update_display()
        
        elif event.key == 'r':
            self.freq_zoom = 1.0
            self.time_zoom = 1.0
            self._update_display()
        
        elif event.key == 'n':
            self.show_normalized = not self.show_normalized
            print(f"Switched to {'Z-score normalized' if self.show_normalized else 'raw (dB)'} view")
            self._update_display()
        
        elif event.key in ['q', 'escape']:
            plt.close(self.fig)
            sys.exit(0)
        
        elif event.key == 'h':
            self._show_help()
    
    def _show_help(self):
        """Display help information"""
        help_text = """
Spectrogram Viewer Controls:
  Left/Right Arrow: Navigate between files
  Up/Down Arrow: Zoom frequency axis in/out
  Shift+Up/Down: Zoom time axis in/out
  N: Toggle normalization (raw dB vs z-score normalized)
  R: Reset zoom to original view
  H: Show this help
  Q/Escape: Quit viewer

Current file: {current_file}
Total files: {total_files}
Audio params: SR={sr}Hz, Hop={hop_size}, FFT={fft}
Current mode: {mode}

Transformations applied:
  • Mel-scale frequency conversion (128 bands, 20Hz-16kHz)
  • Power spectrum (power=2.0)
  • dB conversion (ref=max)
  • Z-score normalization (when toggled with 'N')
        """.format(
            current_file=self.files[self.current_idx].name,
            total_files=len(self.files),
            sr=self.audio_params.get("sr", "unknown"),
            hop_size=self.audio_params.get("hop_size", "unknown"),
            fft=self.audio_params.get("fft", "unknown"),
            mode="Z-score Normalized" if self.show_normalized else "Raw (dB)"
        )
        print(help_text)
    
    def show(self):
        """Display the viewer"""
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="View spectrograms generated by audio2spec.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "spec_dir",
        help="Directory containing spectrogram files (.pt or .npz)"
    )
    
    args = parser.parse_args()
    
    try:
        viewer = SpectrogramViewer(args.spec_dir)
        viewer.show()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
