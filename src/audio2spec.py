# ──────────────────────────────────────────────────────────────────────────────
# audio2spec.py  ‑  simple .wav /.mp3/.ogg ➜ spectrogram (.npy) converter
# ──────────────────────────────────────────────────────────────────────────────
import os, json, time, gc, argparse, logging, random, psutil, signal, sys, shutil
import multiprocessing as mp
from pathlib import Path

import numpy as np
import librosa
import librosa.display                       # noqa: F401  (kept for future plots)
from tqdm import tqdm


class AudioEvent:
    __slots__ = ("path", "start", "end", "name", "waveform", "sample_rate")

    def __init__(self, path, start, end, name):
        self.path = Path(path)
        self.start = float(start)
        self.end = float(end)
        self.name = str(name)
        self.waveform = None  # Will be set for BirdSet events
        self.sample_rate = None  # Will be set for BirdSet events

# ══════════════════════════════════════════════════════════════════════════════
# helper: STFT → log‑magnitude
# ══════════════════════════════════════════════════════════════════════════════
def compute_spectrogram(wav, sr, n_fft, hop, *, mel, n_mels):
    """
    Returns log‑magnitude spectrogram in **dB**.
    • linear STFT  → shape (n_fft//2 + 1, T)   (default 513 × T for n_fft=1024)  
    • mel filter‑bank → shape (n_mels, T)
    
    Optimized version with minimal dtype conversions and efficient power calculation.
    """
    # wav uses default dtype from the loader
    if mel:
        # melspectrogram already computes power spectrum internally
        S = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=hop, power=2.0, n_mels=n_mels, fmin=20, fmax=sr // 2)
    else:
        # More efficient power calculation for linear STFT
        stft_complex = librosa.stft(wav, n_fft=n_fft, hop_length=hop, window="hann", dtype=np.complex64)
        # Efficient power calculation using real and imaginary parts
        S = stft_complex.real**2 + stft_complex.imag**2
        # Using default dtype
        del stft_complex  # free complex array memory immediately

    # Convert to dB with efficient reference calculation
    S_db = librosa.power_to_db(S, ref=np.max(S), top_db=None)
    return S_db


# ══════════════════════════════════════════════════════════════════════════════
# standalone worker function (picklable)
# ══════════════════════════════════════════════════════════════════════════════
def process_audio_file(obj, dst_dir, sr, n_fft, step, use_mel, n_mels, min_len_ms, min_timebins, skipped_counter):
    """
    Standalone worker function that processes a single audio file.
    Returns None on success, error message on failure.
    """
    try:
        event = obj if isinstance(obj, AudioEvent) else None
        fp = event.path if event else obj

        stem = event.name if event else fp.stem
        # ─── check if output already exists ─────────────────────────
        out_path = dst_dir / (stem + ".npy")
        
        if out_path.exists():
            return None  # Skip already processed files
        
        if event:
            duration = max(event.end - event.start, 0.0)
            if duration <= 0:
                skipped_counter.value += 1
                return None
            
            # Use pre-loaded waveform if available (BirdSet streaming)
            if hasattr(event, 'waveform') and event.waveform is not None:
                full_wav = event.waveform
                actual_sr = event.sample_rate
                # Extract the event segment
                start_idx = int(max(event.start, 0.0) * actual_sr)
                end_idx = int(event.end * actual_sr)
                wav = full_wav[start_idx:end_idx]
                # Resample if needed
                if actual_sr != sr:
                    wav = librosa.resample(wav, orig_sr=actual_sr, target_sr=sr)
                    actual_sr = sr
            else:
                # Fallback to file loading for non-BirdSet events
                wav, actual_sr = librosa.load(fp, sr=sr, mono=True, offset=max(event.start, 0.0), duration=duration)
        else:
            # ─── fast duration check before loading ─────────────────
            try:
                duration_sec = librosa.get_duration(path=fp)
                if duration_sec * 1000 < min_len_ms:
                    skipped_counter.value += 1
                    return None
            except Exception:
                # Fallback: if duration check fails, proceed with loading
                pass

            # ─── smart loading with sample rate detection ───────────
            try:
                native_sr = librosa.get_samplerate(fp)
                needs_resampling = (native_sr != sr)
            except Exception:
                needs_resampling = True
                native_sr = None

            if needs_resampling:
                wav, actual_sr = librosa.load(fp, sr=sr, mono=True)
            else:
                wav, actual_sr = librosa.load(fp, sr=None, mono=True)
                if actual_sr != sr:
                    wav = librosa.resample(wav, orig_sr=actual_sr, target_sr=sr)
                    actual_sr = sr

        # Double-check length after loading (in case duration detection was inaccurate)
        if len(wav) / actual_sr * 1000 < min_len_ms:
            skipped_counter.value += 1
            return None

        # ─── spectrogram ─────────────────────────────────────────────
        S = compute_spectrogram(wav, actual_sr, n_fft, step, mel=use_mel, n_mels=n_mels)

        if S.shape[1] < min_timebins:
            skipped_counter.value += 1
            return None

        # ─── optimized output with minimal conversions ───────────────
        np.save(out_path, S.astype(np.float32))

        # free memory fast in workers
        del wav, S
        gc.collect()
        return None
        
    except Exception as e:
        skipped_counter.value += 1
        stem = obj.name if isinstance(obj, AudioEvent) else Path(obj).stem
        return f"{stem}: {e}"


def calculate_optimal_workers(total_files, avg_file_size_mb=50):
    """
    Calculate optimal number of workers based on available memory and CPU cores.
    Assumes each worker needs ~200MB + file_size for processing.
    """
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    
    # Reserve 2GB for system + main process
    worker_memory_gb = available_gb - 2.0
    if worker_memory_gb <= 0:
        return 1
    
    # Estimate memory per worker (base overhead + file processing)
    memory_per_worker_mb = 200 + avg_file_size_mb
    max_workers_by_memory = int(worker_memory_gb * 1024 / memory_per_worker_mb)
    
    # Don't exceed CPU cores
    max_workers_by_cpu = mp.cpu_count()
    
    # Use conservative approach
    optimal = min(max_workers_by_memory, max_workers_by_cpu, total_files)
    return max(1, optimal)


# ══════════════════════════════════════════════════════════════════════════════
# main worker class
# ══════════════════════════════════════════════════════════════════════════════
class WavToSpec:
    """Convert audio files into spectrogram .npy dumps."""

    def __init__(self, src_dir, dst_dir, *, file_list=None, birdset=None, birdset_split="train", step_size=160, n_fft=1024, sr=32_000, take_n_random=None, single_threaded=True, min_len_ms=25, min_timebins=25, mel=True, n_mels=128, max_workers=None):
        self.src_dir = Path(src_dir) if src_dir is not None else None
        self.dst_dir = Path(dst_dir)
        self.dst_dir.mkdir(parents=True, exist_ok=True)

        self.file_list = Path(file_list) if file_list else None
        self.birdset = birdset
        self.birdset_split = birdset_split
        self.step = step_size
        self.n_fft = n_fft
        self.sr = sr
        self.take_n_random = take_n_random
        self.single = single_threaded
        self.min_len_ms = min_len_ms
        self.min_timebins = min_timebins
        self.use_mel = mel
        self.n_mels = n_mels
        self.max_workers = max_workers

        self._setup_logging()
        # Remove unpicklable Manager().Value - will create in run() if needed

        self.audio_files = self._gather_files()
        
        # Save audio parameters to destination directory
        self._save_audio_params()

    # ──────────────────────────────────────────────────────────────────────
    # misc
    # ──────────────────────────────────────────────────────────────────────
    def _setup_logging(self):
        log_dir = self.src_dir / "logs" if self.src_dir else Path("logs")
        log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            filename=log_dir / "run.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def _save_audio_params(self):
        """Save audio processing parameters to JSON file in destination directory."""
        params = {
            "sr": self.sr,
            "mels": self.n_mels,
            "hop_size": self.step,
            "fft": self.n_fft,
        }
        
        params_file = self.dst_dir / "audio_params.json"
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)

    def _gather_files(self):
        if self.birdset:
            # For BirdSet, process samples immediately instead of collecting them
            self._process_birdset_samples()
            return []  # Return empty since we processed everything already
        elif self.file_list:
            files = [Path(line.strip()) for line in self.file_list.read_text().splitlines() if line.strip()]
        else:
            exts = (".wav", ".mp3", ".ogg", ".flac")
            files = [
                Path(root) / f
                for root, _, fs in os.walk(self.src_dir)
                for f in fs if f.lower().endswith(exts)
            ]

        if not files:
            if self.birdset:
                print(f"no detected events found in BirdSet {self.birdset} ({self.birdset_split}) - nothing to do.")
            else:
                print("no audio files matched ‑ nothing to do.")
            return []

        # For BirdSet streaming, we collect all files first, then sample if needed
        if self.take_n_random and self.take_n_random < len(files):
            files = random.sample(files, self.take_n_random)

        return files

    def _process_birdset_samples(self):
        """Process BirdSet samples immediately, parsing detected events"""
        try:
            from datasets import load_dataset, Audio  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install the 'datasets' package to use --birdset") from exc

        print(f"Loading BirdSet dataset: {self.birdset}, split: {self.birdset_split}")
        ds = load_dataset("DBD-research-group/BirdSet", self.birdset, 
                         split=self.birdset_split, streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=self.sr))
        
        print("Starting to process samples...")
        start_time = time.time()
        processed_spectrograms = 0
        processed_samples = 0
        skipped_count = 0
        
        for idx, sample in enumerate(ds):
            processed_samples += 1
            
            # Print progress every 250 samples
            if processed_samples % 250 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {processed_samples} samples → {processed_spectrograms} spectrograms in {elapsed:.1f}s")
            
            # Safety limit for testing
            if self.take_n_random and processed_spectrograms >= self.take_n_random:
                print(f"Reached take_n_random limit: {processed_spectrograms} spectrograms created")
                break
                
            # Get audio info directly from HF
            audio_info = sample.get("audio", {})
            if not audio_info:
                skipped_count += 1
                continue
                
            waveform = audio_info.get("array")
            actual_sr = audio_info.get("sampling_rate", self.sr)
            audio_path = audio_info.get("path")
            
            if waveform is None:
                skipped_count += 1
                continue

            # Event information is available in sample.get("detected_events", [])
            # Each event is a tuple of (start_time, end_time) in seconds
            # detected_events = sample.get("detected_events", [])
            # for event_idx, event in enumerate(detected_events):
            #     start, end = event  # start and end times in seconds
            
            # Get label if available, otherwise use 0
            try:
                label_idx = int(sample.get("ebird_code", 0))
            except (TypeError, ValueError):
                label_idx = 0

            # Extract recording ID from filepath for naming
            filepath = sample.get("filepath", "")
            
            # Create base name - ebird_code + recording ID
            if filepath:
                recording_id = Path(filepath).stem  # e.g., "XC1229031"
                name = f"{label_idx}_{recording_id}"
            elif audio_path:
                recording_id = Path(audio_path).stem
                name = f"{label_idx}_{recording_id}"
            else:
                name = f"{label_idx}_sample_{idx:06d}"

            # Handle missing or None audio_path
            if audio_path is None:
                audio_path = f"{name}.wav"
                fp = Path(audio_path)
            else:
                fp = Path(audio_path)

            # Process full sample (no event snipping)
            duration = len(waveform) / actual_sr
            
            audio_event = AudioEvent(path=fp, start=0.0, end=duration, name=name)
            audio_event.waveform = waveform
            audio_event.sample_rate = actual_sr
            
            # Process this sample immediately
            result = self._safe_process(audio_event)
            if result is None:
                processed_spectrograms += 1
            else:
                skipped_count += 1

        # Final statistics
        elapsed = time.time() - start_time
        print(f"BirdSet processing complete: {processed_spectrograms} spectrograms created from {processed_samples} samples, {skipped_count} skipped in {elapsed:.1f}s")



    # ──────────────────────────────────────────────────────────────────────
    # public entry
    # ──────────────────────────────────────────────────────────────────────
    def run(self):
        if not self.audio_files:
            # For BirdSet, processing is already done in _process_birdset_samples
            if self.birdset:
                print("BirdSet processing completed.")
            return                       # exit 0, no fuss
        
        # Set up signal handler for graceful shutdown
        original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
        
        skipped_count = 0
        pbar = tqdm(total=len(self.audio_files), desc="processing files")

        try:
            if self.single:
                # ───── single threaded ─────
                for fp in self.audio_files:
                    result = self._safe_process(fp)
                    if result is not None:  # error occurred
                        skipped_count += 1
                        logging.error(result)
                    pbar.update()

            else:
                # ───── multi‑process pool with memory-aware workers ─────
                optimal_workers = calculate_optimal_workers(len(self.audio_files))
                if self.max_workers is not None:
                    num_workers = min(self.max_workers, optimal_workers)
                    print(f"Using {num_workers} workers (limited by max_workers={self.max_workers}, optimal would be {optimal_workers})")
                else:
                    num_workers = optimal_workers
                print(f"Workers: {num_workers} (CPU cores: {mp.cpu_count()}, Available RAM: {psutil.virtual_memory().available // (1024**3):.1f}GB)")
                
                ctx = mp.get_context('spawn')  # Use spawn for better isolation
                mgr = ctx.Manager()
                skipped_counter = mgr.Value('i', 0)
                
                # Prepare worker arguments
                worker_args = (
                    self.dst_dir, self.sr, self.n_fft, self.step,
                    self.use_mel, self.n_mels, self.min_len_ms,
                    self.min_timebins, skipped_counter
                )
                
                failed_files = []
                
                with ctx.Pool(processes=num_workers, maxtasksperchild=10, initargs=()) as pool:
                    
                    # Use imap_unordered for better memory control
                    task_args = [(fp,) + worker_args for fp in self.audio_files]
                    
                    try:
                        for i, result in enumerate(pool.imap_unordered(self._worker_wrapper, task_args, chunksize=1)):
                            if result is not None:  # error occurred
                                failed_files.append(result)
                                logging.error(result)
                            
                            pbar.update()
                            
                            # Memory monitoring during processing
                            if (i + 1) % 50 == 0:  # Check every 50 files
                                mem = psutil.virtual_memory()
                                if mem.available < 1.0 * 1024**3:  # Less than 1GB available
                                    print(f"\nLow memory warning: {mem.available // (1024**2)}MB available")
                                    gc.collect()  # Force garbage collection in main process
                                    time.sleep(2)  # Brief pause to let system recover
                    
                    except KeyboardInterrupt:
                        print("\nReceived interrupt signal, shutting down workers...")
                        pool.terminate()
                        pool.join()
                        raise
                
                skipped_count = skipped_counter.value
                
                # Retry failed files single-threaded
                if failed_files:
                    print(f"\nRetrying {len(failed_files)} failed files single-threaded...")
                    for error_msg in failed_files:
                        # Extract file path from error message
                        fp_str = error_msg.split(": ")[0]
                        fp = Path(fp_str)
                        if fp.exists():
                            result = self._safe_process(fp)
                            if result is None:
                                skipped_count -= 1  # Successfully processed on retry

        except KeyboardInterrupt:
            print("\nOperation interrupted by user")
            sys.exit(1)
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_sigint_handler)
            pbar.close()
            
        processed_count = len(self.audio_files) - skipped_count
        print(f"Total processed: {processed_count}")
        print(f"Total skipped  : {skipped_count}")
    
    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown on interrupt signals"""
        print(f"\nReceived signal {signum}, initiating graceful shutdown...")
        # The actual cleanup will be handled in the run() method
        raise KeyboardInterrupt()
    
    @staticmethod
    def _worker_wrapper(args):
        """Wrapper function to call the standalone worker function"""
        return process_audio_file(*args)

    # ──────────────────────────────────────────────────────────────────────
    # helpers
    # ──────────────────────────────────────────────────────────────────────
    def _safe_process(self, obj):
        """
        Single-threaded processing wrapper with optimizations.
        Returns None on success, error message on failure.
        """
        try:
            event = obj if isinstance(obj, AudioEvent) else None
            fp = event.path if event else obj
            stem = event.name if event else fp.stem
            # ─── check if output already exists ─────────────────────────
            out_path = self.dst_dir / (stem + ".npy")
            
            if out_path.exists():
                return None  # Skip already processed files
            
            if event:
                duration = max(event.end - event.start, 0.0)
                if duration <= 0:
                    return None
                
                # Use pre-loaded waveform if available (BirdSet streaming)
                if hasattr(event, 'waveform') and event.waveform is not None:
                    full_wav = event.waveform
                    actual_sr = event.sample_rate
                    # Extract the event segment
                    start_idx = int(max(event.start, 0.0) * actual_sr)
                    end_idx = int(event.end * actual_sr)
                    wav = full_wav[start_idx:end_idx]
                    # Resample if needed
                    if actual_sr != self.sr:
                        wav = librosa.resample(wav, orig_sr=actual_sr, target_sr=self.sr)
                        actual_sr = self.sr
                else:
                    # Fallback to file loading for non-BirdSet events
                    wav, actual_sr = librosa.load(fp, sr=self.sr, mono=True, offset=max(event.start, 0.0), duration=duration)
            else:
                # ─── fast duration check before loading ────────────────
                try:
                    duration_sec = librosa.get_duration(path=fp)
                    if duration_sec * 1000 < self.min_len_ms:
                        return None  # Skip, but not an error
                except Exception:
                    pass

                # ─── smart loading with sample rate detection ──────────
                try:
                    native_sr = librosa.get_samplerate(fp)
                    needs_resampling = (native_sr != self.sr)
                except Exception:
                    needs_resampling = True
                    native_sr = None

                if needs_resampling:
                    wav, actual_sr = librosa.load(fp, sr=self.sr, mono=True)
                else:
                    wav, actual_sr = librosa.load(fp, sr=None, mono=True)
                    if actual_sr != self.sr:
                        wav = librosa.resample(wav, orig_sr=actual_sr, target_sr=self.sr)
                        actual_sr = self.sr

            # Double-check length after loading (in case duration detection was inaccurate)
            if len(wav) / actual_sr * 1000 < self.min_len_ms:
                return None  # Skip, but not an error

            # ─── spectrogram ─────────────────────────────────────────────
            S = compute_spectrogram(wav, actual_sr, self.n_fft, self.step, mel=self.use_mel, n_mels=self.n_mels)

            if S.shape[1] < self.min_timebins:
                return None  # Skip, but not an error

            # ─── optimized output with minimal conversions ───────────────
            np.save(out_path, S.astype(np.float32))

            # free memory fast
            del wav, S
            gc.collect()
            return None
            
        except Exception as e:
            return f"{stem}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def cli():
    p = argparse.ArgumentParser(
        description="Convert audio → log‑spectrogram files (.npy).")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--src_dir",   type=str,
                     help="Root folder with wav/mp3/ogg files (searched recursively).")
    grp.add_argument("--file_list", type=str,
                     help="Text file with absolute/relative paths, one per line.")
    grp.add_argument("--birdset", type=str,
                     help="BirdSet configuration name (e.g. HSN or HSN_xc) to load via Hugging Face.")
    p.add_argument("--dst_dir",  type=str, required=True,
                   help="Where outputs go.")
    p.add_argument("--birdset_split", type=str, default="train",
                   help="Dataset split to use with --birdset (default: train) for XCL that is the only one")
    p.add_argument("--sr", type=int, default=32_000,
                   help="Sample rate in Hz (default: 32000).")
    p.add_argument("--step_size", type=int, default=64,
                   help="STFT hop length (samples at target sample rate).")
    p.add_argument("--nfft",      type=int, default=1024,
                   help="FFT size.")
    p.add_argument("--take_n_random", type=int, default=None,
                   help="Pick N random files instead of the full set.")
    p.add_argument("--single_threaded",
                   choices=["true", "false", "1", "0", "yes", "no"],
                   default="true",
                   help="Force single‑thread. Default true.")
    mel_grp = p.add_mutually_exclusive_group()
    mel_grp.add_argument("--mel", action="store_true",
                         help="Output log‑mel (default).")
    mel_grp.add_argument("--linear", action="store_true",
                         help="Output linear‑frequency STFT bins.")
    p.add_argument("--n_mels", type=int, default=128,
                   help="Number of mel bands (default: 128)")
    p.add_argument("--max_workers", type=int, default=None,
                   help="Maximum number of worker processes (default: auto-detect)")
    args = p.parse_args()

    single = args.single_threaded.lower() in {"true", "1", "yes"}

    converter = WavToSpec(src_dir=args.src_dir, dst_dir=args.dst_dir, file_list=args.file_list, birdset=args.birdset, birdset_split=args.birdset_split, step_size=args.step_size, n_fft=args.nfft, sr=args.sr, take_n_random=args.take_n_random, single_threaded=single, mel=not args.linear, n_mels=args.n_mels, max_workers=args.max_workers)
    converter.run()


if __name__ == "__main__":
    cli()
