#!/usr/bin/env python3
"""
Fast labeling tool for annotating song boundaries in audio recordings.
Uses tkinter for blazing fast drag-and-select operations.
"""

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import json
import os
import threading
import time
from collections import deque
import librosa
from datasets import Audio, load_dataset
from matplotlib import cm


class SpectrogramLabeler:
    def __init__(self, root, json_path="xcm_annotations.json"):
        self.root = root
        self.json_path = os.path.abspath(json_path)
        print(f"Annotations will be saved to: {self.json_path}")
        
        self.annotations = self.load_or_create_json()
        
        # Track which files have been labeled (filter out old placeholder names)
        self.labeled_filenames = {
            rec["recording"]["filename"] 
            for rec in self.annotations["recordings"]
            if not rec["recording"]["filename"].startswith("recording_")  # Skip old placeholder names
        }
        self.session_labeled_count = 0
        print(f"Found {len(self.labeled_filenames)} already labeled recordings (filtered out old placeholders)")
        
        self.detected_events = []
        
        # Mouse interaction state
        self.selecting = False
        self.selection_start = None
        self.selection_end = None
        self.display_scale = 2  # Scale factor for display
        
        # Prefetch configuration (can override via LABEL_XCM_* env vars)
        default_workers = min(4, max(1, (os.cpu_count() or 4)))
        self.num_preload_workers = max(1, int(os.getenv("LABEL_XCM_WORKERS", default_workers)))
        self.max_buffer_size = max(self.num_preload_workers * 6, int(os.getenv("LABEL_XCM_BUFFER", 40)))
        self.prefetch_buffer = deque()
        self.prefetch_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.first_recording_ready = threading.Event()
        self.total_processed = 0
        self.total_loaded = 0
        self.total_skipped = 0
        self.total_errors = 0
        self.active_workers = 0
        
        # Current recording data
        self.current_spec_image = None
        self.current_audio = None
        self.current_sr = None
        self.current_filename = None
        self.current_metadata = {}
        self.current_recording_idx = 0
        self.selection_mask = None
        
        # Load dataset
        print("Loading BirdSet XCM dataset...")
        self.root.title("Loading BirdSet XCM...")
        self.root.update()
        
        try:
            ds = load_dataset("DBD-research-group/BirdSet", "XCM", streaming=True)
            self.ds = ds.cast_column("audio", Audio(sampling_rate=32_000))
            self.train = self.ds["train"]
            self.dataset_iter = iter(self.train)
            self.dataset_lock = threading.Lock()
            self.dataset_exhausted = False
            self.dataset_exhaust_logged = False
            self.active_workers = self.num_preload_workers
            print("Dataset loaded successfully! XCM should have ~89,798 training recordings")
            print(f"Already labeled: {len(self.labeled_filenames)} recordings")
            print(f"Prefetch setup -> workers: {self.num_preload_workers}, buffer size: {self.max_buffer_size}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Setup UI
        self.setup_ui()
        
        # Start preloading
        self.preloader_running = True
        self.preloader_threads = []
        for worker_id in range(self.num_preload_workers):
            thread = threading.Thread(target=self.preload_worker, args=(worker_id,), daemon=True)
            self.preloader_threads.append(thread)
            thread.start()
        
        # Auto-save
        self.auto_save_interval = 30  # seconds
        self.auto_save_thread = threading.Thread(target=self.auto_save_worker, daemon=True)
        self.auto_save_thread.start()
    
    def setup_ui(self):
        """Setup tkinter UI."""
        self.root.title("Spectrogram Labeler - Loading...")
        
        # Make fullscreen
        self.root.attributes('-fullscreen', True)
        self.is_fullscreen = True
        
        # Toggle fullscreen with F11 or Escape
        def toggle_fullscreen(event=None):
            self.is_fullscreen = not self.is_fullscreen
            self.root.attributes('-fullscreen', self.is_fullscreen)
        
        self.root.bind("<F11>", toggle_fullscreen)
        self.root.bind("<Escape>", toggle_fullscreen)
        
        # Info labels at top
        info_frame = tk.Frame(self.root, bg="black")
        info_frame.pack(side="top", fill="x")
        
        self.filename_label = tk.Label(info_frame, text="Loading...", fg="white", bg="black", 
                                       font=("Courier", 14, "bold"), anchor="w")
        self.filename_label.pack(side="top", fill="x", padx=10, pady=2)
        
        self.status_label = tk.Label(info_frame, text="Prefetching spectrograms...", fg="white", bg="black",
                                     font=("Courier", 12), anchor="w")
        self.status_label.pack(side="top", fill="x", padx=10, pady=2)
        
        self.counter_label = tk.Label(info_frame, text="Annotated this session: 0 | Total annotated: 0",
                                      fg="white", bg="black", font=("Courier", 12), anchor="w")
        self.counter_label.pack(side="top", fill="x", padx=10, pady=2)
        self.counter_label.config(
            text=f"Annotated this session: {self.session_labeled_count} | Total annotated: {len(self.labeled_filenames)}"
        )
        
        # Canvas for spectrogram
        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True)
        
        # Keyboard bindings at bottom
        bindings_text = (
            "Controls: [Click+Drag: Select] [X: Delete Last] [S: Save] "
            "[N: Next] [←/→: Scroll] [Q: Quit] [F11/Esc: Toggle Fullscreen]"
        )
        self.bindings_label = tk.Label(self.root, text=bindings_text, fg="white", bg="black",
                                       font=("Courier", 11, "bold"))
        self.bindings_label.pack(side="bottom", fill="x", padx=10, pady=5)
        
        # Bind events
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_scroll)
        self.canvas.bind("<Button-4>", self.on_scroll)
        self.canvas.bind("<Button-5>", self.on_scroll)
        
        self.root.bind("<x>", lambda e: self.delete_last())
        self.root.bind("<X>", lambda e: self.delete_last())
        self.root.bind("<s>", lambda e: self.save_current_recording(manual=True))
        self.root.bind("<S>", lambda e: self.save_current_recording(manual=True))
        self.root.bind("<n>", lambda e: self.next_recording())
        self.root.bind("<N>", lambda e: self.next_recording())
        self.root.bind("<q>", lambda e: self.quit_app())
        self.root.bind("<Q>", lambda e: self.quit_app())
        self.root.bind("<Left>", lambda e: self.canvas.xview_scroll(-1, "units"))
        self.root.bind("<Right>", lambda e: self.canvas.xview_scroll(1, "units"))
        
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
    
    def preload_worker(self, worker_id):
        """Background worker to fetch and preprocess recordings."""
        local_errors = 0
        
        if worker_id == 0:
            print("=== Preloader threads started ===", flush=True)
            print("Attempting to fetch recordings from BirdSet...", flush=True)
            print("NOTE: First fetch may take 30-60s to download from HuggingFace...", flush=True)
        
        while self.preloader_running:
            # Avoid racing to append when buffer is saturated
            with self.prefetch_lock:
                buffer_is_full = len(self.prefetch_buffer) >= self.max_buffer_size
            if buffer_is_full:
                time.sleep(0.1)
                continue
            
            try:
                with self.dataset_lock:
                    if self.dataset_exhausted:
                        ex = None
                    else:
                        try:
                            ex = next(self.dataset_iter)
                        except StopIteration:
                            self.dataset_exhausted = True
                            ex = None
            except StopIteration:
                ex = None
            except Exception as e:
                print(f"[worker {worker_id}] ERROR fetching from dataset: {e}", flush=True)
                import traceback
                traceback.print_exc()
                time.sleep(1.0)
                continue
            
            if ex is None:
                log_message = False
                with self.stats_lock:
                    if self.active_workers > 0:
                        self.active_workers -= 1
                    remaining = self.active_workers
                    processed = self.total_processed
                    loaded = self.total_loaded
                    skipped = self.total_skipped
                    if not self.dataset_exhaust_logged:
                        self.dataset_exhaust_logged = True
                        log_message = True
                if log_message or remaining == 0:
                    print(f"[worker {worker_id}] Dataset exhausted (processed={processed}, loaded={loaded}, skipped={skipped})")
                if remaining == 0:
                    print(f"=== Dataset exhausted after processing {processed} recordings ===")
                    print(f"Loaded: {loaded}, Skipped: {skipped}")
                break
            
            with self.stats_lock:
                self.total_processed += 1
                total_processed = self.total_processed
            
            if total_processed % 100 == 0 and worker_id == 0:
                with self.stats_lock:
                    print(f"=== Progress: Processed {self.total_processed} recordings "
                          f"(loaded: {self.total_loaded}, skipped: {self.total_skipped}) ===")
            
            filepath = ex.get("filepath", f"recording_worker{worker_id}_{total_processed}.ogg")
            ebird_code = ex.get("ebird_code", None)
            ebird_code_multilabel = ex.get("ebird_code_multilabel", [])
            
            # Skip previously labeled recordings
            if filepath in self.labeled_filenames:
                with self.stats_lock:
                    self.total_skipped += 1
                    skipped = self.total_skipped
                if skipped % 100 == 0 and worker_id == 0:
                    print(f"Skipped {skipped} already-labeled recordings")
                continue
            
            try:
                start_time = time.time()
                
                audio = ex["audio"]["array"]
                sr = ex["audio"]["sampling_rate"]
                
                if len(audio) == 0:
                    print(f"[worker {worker_id}] WARNING: Empty audio for {filepath}, skipping")
                    with self.stats_lock:
                        self.total_errors += 1
                    continue
                
                decode_time = time.time()
                
                S = librosa.feature.melspectrogram(
                    y=audio,
                    sr=sr,
                    n_mels=128,
                    fmax=8000,
                    hop_length=1024,
                    n_fft=2048
                )
                S_db = librosa.power_to_db(S, ref=np.max)
                
                spec_time = time.time()
                
                S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
                S_norm = np.flipud(S_norm)
                
                colored = cm.viridis(S_norm)
                spec_image = Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8))
                
                image_time = time.time()
                
                data = {
                    'spec_image': spec_image,
                    'audio': audio,
                    'sr': sr,
                    'filename': filepath,
                    'metadata': {
                        'filename': filepath,
                        'ebird_code': ebird_code,
                        'ebird_code_multilabel': ebird_code_multilabel,
                        'lat': ex.get("lat", None),
                        'long': ex.get("long", None),
                        'source': ex.get("source", "xenocanto"),
                        'quality': ex.get("quality", None),
                        'recordist': ex.get("recordist", None),
                        'license': ex.get("license", None)
                    }
                }
                
                appended = False
                buffer_size = 0
                while self.preloader_running and not appended:
                    with self.prefetch_lock:
                        if len(self.prefetch_buffer) < self.max_buffer_size:
                            self.prefetch_buffer.append(data)
                            buffer_size = len(self.prefetch_buffer)
                            appended = True
                        else:
                            buffer_size = len(self.prefetch_buffer)
                    if not appended:
                        time.sleep(0.05)
                if not appended:
                    break
                
                total_time = image_time - start_time
                decode_t = decode_time - start_time
                spec_t = spec_time - decode_time
                image_t = image_time - spec_time
                
                with self.stats_lock:
                    self.total_loaded += 1
                    count = self.total_loaded
                    self.total_errors = max(self.total_errors - 1, 0)
                
                if count <= 5 or count % 50 == 0:
                    print(f"[worker {worker_id}] Loaded: {filepath} ({len(audio)/sr:.1f}s) -> "
                          f"Buffer: {buffer_size}/{self.max_buffer_size} | "
                          f"Time: {total_time:.2f}s (decode:{decode_t:.2f}s, spec:{spec_t:.2f}s, img:{image_t:.2f}s)")
                else:
                    print(f"[worker {worker_id}] Loaded: {filepath} ({len(audio)/sr:.1f}s) -> "
                          f"Buffer: {buffer_size}/{self.max_buffer_size} | Time: {total_time:.2f}s")
                
                if not self.first_recording_ready.is_set():
                    self.first_recording_ready.set()
                    print("  -> Loading first recording into UI")
                    self.root.after(0, self.load_next_recording)
                
                local_errors = 0
            
            except Exception as e:
                local_errors += 1
                with self.stats_lock:
                    self.total_errors += 1
                print(f"[worker {worker_id}] ERROR processing recording: {e}")
                import traceback
                traceback.print_exc()
                if local_errors > 3:
                    time.sleep(0.5)
                continue
        
        if worker_id == 0:
            with self.stats_lock:
                print(f"=== Preloader stopped. Total processed: {self.total_processed}, "
                      f"Loaded: {self.total_loaded}, Skipped: {self.total_skipped} ===")
    
    def load_or_create_json(self):
        """Load existing JSON or create new structure."""
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "metadata": {"units": "ms"},
                "recordings": []
            }
    
    def save_json(self):
        """Save annotations to JSON file."""
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        annotations_clean = convert_to_native(self.annotations)
        
        with open(self.json_path, 'w') as f:
            json.dump(annotations_clean, indent=2, fp=f)
        print(f"Saved annotations to {self.json_path}")
    
    def load_next_recording(self):
        """Load next recording from prefetch buffer."""
        # Wait for buffer to have something (up to 30 seconds)
        max_wait = 30
        waited = 0.0
        status_interval = 5.0
        next_status_time = status_interval
        sleep_interval = 0.1
        
        while waited < max_wait:
            with self.prefetch_lock:
                if self.prefetch_buffer:
                    data = self.prefetch_buffer.popleft()
                    break
                buffer_size = len(self.prefetch_buffer)
            
            # Buffer is empty, wait a bit for preloader
            if waited == 0:
                print(f"Buffer empty, waiting for preloader to load more recordings...")
                self.status_label.config(text=f"Downloading from HuggingFace... (first fetch can take 30-60s)")
                self.root.update()
            
            time.sleep(sleep_interval)
            waited += sleep_interval
            
            if waited >= next_status_time:
                print(f"Still waiting... ({waited:.1f}s elapsed)")
                self.status_label.config(
                    text=f"Loading... ({int(waited)}s elapsed, preloader is fetching from dataset)"
                )
                self.root.update()
                next_status_time += status_interval
        else:
            # Waited too long, no more recordings
            messagebox.showinfo("Complete", "No more recordings available! The preloader may have finished or encountered an error.")
            return False
        
        # Load from prefetched data
        self.current_spec_image = data['spec_image']
        self.current_audio = data['audio']
        self.current_sr = data['sr']
        self.current_filename = data['filename']
        self.current_metadata = data['metadata']
        
        # Check if this recording already has annotations
        self.detected_events = []
        for rec in self.annotations["recordings"]:
            if rec["recording"]["filename"] == self.current_filename:
                self.detected_events = rec["detected_events"]
                break
        
        # Create selection mask
        self.selection_mask = Image.new('L', self.current_spec_image.size, 0)
        
        # Draw existing events on mask
        draw = ImageDraw.Draw(self.selection_mask)
        duration_s = len(self.current_audio) / self.current_sr
        for event in self.detected_events:
            onset_s = event["onset_ms"] / 1000.0
            offset_s = event["offset_ms"] / 1000.0
            
            # Convert time to pixels
            start_x = int((onset_s / duration_s) * self.current_spec_image.width)
            end_x = int((offset_s / duration_s) * self.current_spec_image.width)
            
            draw.rectangle([start_x, 0, end_x, self.current_spec_image.height], fill=255)
        
        self.current_recording_idx += 1
        self.display_spectrogram()
        self.update_status()
        
        return True
    
    def display_spectrogram(self):
        """Display spectrogram with annotations."""
        if self.current_spec_image is None:
            return
        
        # Blend mask with image
        display_image = self.current_spec_image.copy()
        
        # Apply green overlay for selections
        mask_rgba = Image.new('RGBA', display_image.size, (0, 255, 0, 0))
        draw = ImageDraw.Draw(mask_rgba)
        
        # Draw mask areas
        for x in range(self.selection_mask.width):
            for y in range(self.selection_mask.height):
                if self.selection_mask.getpixel((x, y)) > 0:
                    draw.point((x, y), fill=(0, 255, 0, 100))
        
        display_image = display_image.convert('RGBA')
        display_image = Image.alpha_composite(display_image, mask_rgba)
        display_image = display_image.convert('RGB')
        
        # Scale up 4x for better visibility (was 2x)
        scale_factor = 4
        scaled_width = display_image.width * scale_factor
        scaled_height = display_image.height * scale_factor
        display_image = display_image.resize((scaled_width, scaled_height), Image.Resampling.NEAREST)
        
        # Store scale factor for coordinate conversion
        self.display_scale = scale_factor
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, display_image.width, display_image.height))
        
        self.tk_image = ImageTk.PhotoImage(display_image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
    
    def update_status(self):
        """Update status labels."""
        duration_s = len(self.current_audio) / self.current_sr if self.current_audio is not None else 0
        
        # Get metadata info
        ebird_code = self.current_metadata.get('ebird_code', 'N/A')
        quality = self.current_metadata.get('quality', 'N/A')
        source = self.current_metadata.get('source', 'N/A')
        
        # Build info string
        meta_str = f" | eBird: {ebird_code}"
        if quality and quality != 'N/A':
            meta_str += f" | Quality: {quality}"
        
        self.filename_label.config(
            text=f"Recording: {self.current_filename}{meta_str} | Duration: {duration_s:.1f}s | Events: {len(self.detected_events)}"
        )
        self.status_label.config(
            text=f"Labeled: {len(self.labeled_filenames)} total | Buffer: {len(self.prefetch_buffer)}/{self.max_buffer_size} ready | Source: {source}"
        )
        self.counter_label.config(
            text=f"Annotated this session: {self.session_labeled_count} | Total annotated: {len(self.labeled_filenames)}"
        )
        self.root.title(f"Spectrogram Labeler - {self.current_filename}")
    
    def on_press(self, event):
        """Handle mouse press."""
        self.selecting = True
        self.selection_start = self.canvas.canvasx(event.x)
        self.selection_end = self.selection_start
        self.draw_selection_box()
    
    def on_drag(self, event):
        """Handle mouse drag - just update selection box."""
        if self.selecting:
            self.selection_end = self.canvas.canvasx(event.x)
            self.draw_selection_box()
    
    def on_release(self, event):
        """Handle mouse release - add the selection."""
        if not self.selecting:
            return
        
        self.selecting = False
        self.selection_end = self.canvas.canvasx(event.x)
        
        # Remove selection box
        self.canvas.delete("selection_box")
        
        if self.current_spec_image is None:
            return
        
        # Calculate time bounds (convert from scaled display coords to original image coords)
        start_x = min(self.selection_start, self.selection_end) / self.display_scale
        end_x = max(self.selection_start, self.selection_end) / self.display_scale
        
        # Clamp to image bounds
        start_x = max(0, min(start_x, self.current_spec_image.width))
        end_x = max(0, min(end_x, self.current_spec_image.width))
        
        # Minimum selection size
        if abs(end_x - start_x) < 5:
            return
        
        # Convert pixels to time
        duration_s = len(self.current_audio) / self.current_sr
        onset_s = (start_x / self.current_spec_image.width) * duration_s
        offset_s = (end_x / self.current_spec_image.width) * duration_s
        
        # Add event
        self.detected_events.append({
            "onset_ms": onset_s * 1000.0,
            "offset_ms": offset_s * 1000.0
        })
        
        print(f"Added event: {onset_s*1000:.1f} - {offset_s*1000:.1f} ms")
        
        # Update mask
        draw = ImageDraw.Draw(self.selection_mask)
        draw.rectangle([int(start_x), 0, int(end_x), self.current_spec_image.height], fill=255)
        
        # Redisplay
        self.display_spectrogram()
        self.update_status()
    
    def draw_selection_box(self):
        """Draw temporary selection box."""
        self.canvas.delete("selection_box")
        if self.selection_start is not None and self.selection_end is not None:
            self.canvas.create_rectangle(
                self.selection_start, 0,
                self.selection_end, self.canvas.winfo_height(),
                outline="yellow", width=2, tags="selection_box"
            )
    
    def on_scroll(self, event):
        """Handle mouse wheel scrolling."""
        if event.delta:
            delta = -1 if event.delta > 0 else 1
        else:
            delta = -1 if event.num == 4 else 1
        self.canvas.xview_scroll(delta, "units")
    
    def delete_last(self):
        """Delete last annotation."""
        if self.detected_events:
            removed = self.detected_events.pop()
            print(f"Removed event: {removed['onset_ms']:.1f} - {removed['offset_ms']:.1f} ms")
            
            # Rebuild mask
            self.selection_mask = Image.new('L', self.current_spec_image.size, 0)
            draw = ImageDraw.Draw(self.selection_mask)
            duration_s = len(self.current_audio) / self.current_sr
            
            for event in self.detected_events:
                onset_s = event["onset_ms"] / 1000.0
                offset_s = event["offset_ms"] / 1000.0
                start_x = int((onset_s / duration_s) * self.current_spec_image.width)
                end_x = int((offset_s / duration_s) * self.current_spec_image.width)
                draw.rectangle([start_x, 0, end_x, self.current_spec_image.height], fill=255)
            
            self.display_spectrogram()
            self.update_status()
    
    def merge_overlapping_events(self, events):
        """Merge overlapping or adjacent detected events."""
        if not events:
            return []
        
        sorted_events = sorted(events, key=lambda x: x["onset_ms"])
        merged = []
        current = sorted_events[0].copy()
        
        for next_event in sorted_events[1:]:
            if next_event["onset_ms"] <= current["offset_ms"] + 50:
                current["offset_ms"] = max(current["offset_ms"], next_event["offset_ms"])
            else:
                merged.append(current)
                current = next_event.copy()
        
        merged.append(current)
        return merged
    
    def save_current_recording(self, manual=False):
        """Save current recording annotations with merged overlapping events."""
        if self.current_filename is None:
            return
        
        # Merge overlapping events
        merged_events = self.merge_overlapping_events(self.detected_events)
        
        if len(merged_events) < len(self.detected_events):
            print(f"Merged {len(self.detected_events)} events into {len(merged_events)} non-overlapping events")
        
        # Remove existing entry (if any)
        self.annotations["recordings"] = [
            rec for rec in self.annotations["recordings"]
            if rec["recording"]["filename"] != self.current_filename
        ]
        
        previously_labeled = self.current_filename in self.labeled_filenames
        should_store = manual or previously_labeled or len(merged_events) > 0
        
        if should_store:
            # Add current recording
            self.annotations["recordings"].append({
                "recording": self.current_metadata,
                "detected_events": merged_events
            })
            
            self.labeled_filenames.add(self.current_filename)
            if not previously_labeled:
                self.session_labeled_count += 1
        else:
            if not previously_labeled and self.current_filename in self.labeled_filenames:
                self.labeled_filenames.remove(self.current_filename)
        
        if should_store or previously_labeled:
            self.save_json()
        
        # Refresh status labels on the main thread
        if threading.current_thread() is threading.main_thread():
            self.update_status()
        else:
            self.root.after(0, self.update_status)
    
    def next_recording(self):
        """Save and move to next recording."""
        self.save_current_recording(manual=True)
        self.load_next_recording()
    
    def auto_save_worker(self):
        """Background thread for auto-saving."""
        while True:
            time.sleep(self.auto_save_interval)
            if self.current_filename:
                self.save_current_recording()
                print("Auto-saved")
    
    def quit_app(self):
        """Clean exit."""
        if messagebox.askokcancel("Quit", "Save and quit?"):
            self.save_current_recording(manual=True)
            self.preloader_running = False
            self.root.quit()
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SpectrogramLabeler(root, "xcm_annotations.json")
    root.mainloop()
