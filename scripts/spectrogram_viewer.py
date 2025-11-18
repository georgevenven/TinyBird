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
    def __init__(self, spec_dir: str, json_path: Optional[str] = None):
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
        
        # Load annotations if provided
        self.annotations = None
        if json_path:
            self.annotations = self._load_annotations(json_path)
        
        # Set up the plot with or without annotation bars
        if self.annotations:
            self.fig, (self.ax, self.ax_events, self.ax_units) = plt.subplots(
                3, 1, figsize=(12, 8), 
                gridspec_kw={'height_ratios': [10, 1, 1], 'hspace': 0.05}
            )
        else:
            self.fig, self.ax = plt.subplots(figsize=(12, 6))
            self.ax_events = None
            self.ax_units = None
        
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
        if self.annotations:
            print(f"Loaded annotations from JSON with {len(self.annotations)} recordings")
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
    
    def _load_annotations(self, json_path: str) -> Dict[str, Dict]:
        """Load annotations from JSON file and index by filename"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Create a dictionary indexed by filename (without extension)
        annotations = {}
        for recording in data.get("recordings", []):
            filename = Path(recording["recording"]["filename"]).stem
            annotations[filename] = recording
        
        return annotations
    
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
        
        # Recreate the axes
        if self.annotations:
            self.ax, self.ax_events, self.ax_units = self.fig.subplots(
                3, 1, gridspec_kw={'height_ratios': [10, 1, 1], 'hspace': 0.05}
            )
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax_events = None
            self.ax_units = None
        
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
        
        # Labels and title
        self.ax.set_xlabel('Time (s)' if not self.annotations else '')
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
        
        # Draw annotation bars if available
        if self.annotations:
            self._draw_annotation_bars(current_file, time_axis)
        
        # Refresh display
        self.fig.canvas.draw()
    
    def _draw_annotation_bars(self, current_file: Path, time_axis: np.ndarray):
        """Draw detected events and units as bars below the spectrogram"""
        filename = current_file.stem
        
        # Get annotations for this file
        if filename not in self.annotations:
            # Clear the annotation axes if no annotations for this file
            self.ax_events.clear()
            self.ax_units.clear()
            self.ax_events.set_xlim(time_axis[0], time_axis[-1])
            self.ax_units.set_xlim(time_axis[0], time_axis[-1])
            self.ax_events.set_ylim(0, 1)
            self.ax_units.set_ylim(0, 1)
            self.ax_events.set_ylabel('Events', fontsize=8)
            self.ax_units.set_ylabel('Units', fontsize=8)
            self.ax_units.set_xlabel('Time (s)')
            self.ax_events.set_yticks([])
            self.ax_units.set_yticks([])
            return
        
        recording = self.annotations[filename]
        detected_events = recording.get("detected_events", [])
        
        # Clear the axes
        self.ax_events.clear()
        self.ax_units.clear()
        
        # Set limits
        self.ax_events.set_xlim(time_axis[0], time_axis[-1])
        self.ax_units.set_xlim(time_axis[0], time_axis[-1])
        self.ax_events.set_ylim(0, 1)
        self.ax_units.set_ylim(0, 1)
        
        # Apply time zoom to annotation axes
        if self.time_zoom != 1.0:
            xlim = self.ax.get_xlim()
            self.ax_events.set_xlim(xlim)
            self.ax_units.set_xlim(xlim)
        
        # Draw detected events
        for i, event in enumerate(detected_events):
            onset_s = event["onset_ms"] / 1000.0
            offset_s = event["offset_ms"] / 1000.0
            
            # Color events with alternating colors
            color = plt.cm.Set3(i % 12)
            
            rect = patches.Rectangle(
                (onset_s, 0), offset_s - onset_s, 1,
                linewidth=1, edgecolor='black', facecolor=color, alpha=0.7
            )
            self.ax_events.add_patch(rect)
            
            # Draw units within this event
            for unit in event.get("units", []):
                unit_onset_s = unit["onset_ms"] / 1000.0
                unit_offset_s = unit["offset_ms"] / 1000.0
                unit_id = unit.get("id", 0)
                
                # Color units by their ID
                unit_color = plt.cm.tab20(unit_id % 20)
                
                rect = patches.Rectangle(
                    (unit_onset_s, 0), unit_offset_s - unit_onset_s, 1,
                    linewidth=1, edgecolor='black', facecolor=unit_color, alpha=0.7
                )
                self.ax_units.add_patch(rect)
        
        # Labels and styling
        self.ax_events.set_ylabel('Events', fontsize=8)
        self.ax_units.set_ylabel('Units', fontsize=8)
        self.ax_units.set_xlabel('Time (s)')
        self.ax_events.set_yticks([])
        self.ax_units.set_yticks([])
        self.ax_events.grid(True, alpha=0.3, axis='x')
        self.ax_units.grid(True, alpha=0.3, axis='x')
    
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
        "--spec_dir",
        help="Directory containing spectrogram files (.npy)"
    )
    parser.add_argument(
        "--json",
        help="Optional JSON file with detected events and labels",
        default=None
    )
    
    args = parser.parse_args()
    
    try:
        viewer = SpectrogramViewer(args.spec_dir, json_path=args.json)
        viewer.show()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
