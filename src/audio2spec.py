# ──────────────────────────────────────────────────────────────────────────────
# audio2spec.py  ‑  simple .wav /.mp3/.ogg ➜ spectrogram (.npz / .pt) converter
# ──────────────────────────────────────────────────────────────────────────────
import os, json, time, gc, argparse, logging, random, psutil, signal, sys
import multiprocessing as mp
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional

import numpy as np
import librosa
import librosa.display                       # noqa: F401  (kept for future plots)
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# helper: STFT → log‑magnitude
# ══════════════════════════════════════════════════════════════════════════════
def compute_spectrogram(
    wav: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    *,
    mel: bool,
    n_mels: int
) -> np.ndarray:
    """
    Returns log‑magnitude spectrogram in **dB**.
    • linear STFT  → shape (n_fft//2 + 1, T)   (default 513 × T for n_fft=1024)  
    • mel filter‑bank → shape (n_mels, T)
    
    Optimized version with minimal dtype conversions and efficient power calculation.
    """
    # wav uses default dtype from the loader
    if mel:
        # melspectrogram already computes power spectrum internally
        S = librosa.feature.melspectrogram(
            y=wav,  # no dtype conversion needed
            sr=sr,
            n_fft=n_fft,
            hop_length=hop,
            power=2.0,         # power‑spectrogram
            n_mels=n_mels,
            fmin=20,
            fmax=sr // 2,
            # Using default dtype
        )
    else:
        # More efficient power calculation for linear STFT
        stft_complex = librosa.stft(
            wav,  # no dtype conversion needed
            n_fft=n_fft,
            hop_length=hop,
            window="hann",
            dtype=np.complex64  # use complex64 for memory efficiency
        )
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
def process_audio_file(
    fp: Path,
    dst_dir: Path,
    sr: int,
    n_fft: int,
    step: int,
    use_mel: bool,
    n_mels: int,
    min_len_ms: int,
    min_timebins: int,
    fmt: str,
    lab_map: Dict[str, List[Tuple[int, int, int]]],
    skipped_counter: Any
) -> Optional[str]:
    """
    Standalone worker function that processes a single audio file.
    Returns None on success, error message on failure.
    """
    try:
        # ─── check if output already exists ─────────────────────────
        if fmt == "pt":
            out_path = dst_dir / (fp.stem + ".pt")
        else:
            out_path = dst_dir / (fp.stem + ".npz")
        
        if out_path.exists():
            return None  # Skip already processed files
        
        # ─── fast duration check before loading ─────────────────────
        try:
            duration_sec = librosa.get_duration(path=fp)
            if duration_sec * 1000 < min_len_ms:
                skipped_counter.value += 1
                return None
        except Exception:
            # Fallback: if duration check fails, proceed with loading
            pass

        # ─── smart loading with sample rate detection ────────────────
        # First, detect native sample rate without loading audio
        try:
            native_sr = librosa.get_samplerate(fp)
            needs_resampling = (native_sr != sr)
        except Exception:
            # Fallback: assume resampling needed
            needs_resampling = True
            native_sr = None

        # Load audio with optimal settings
        if needs_resampling:
            wav, actual_sr = librosa.load(fp, sr=sr, mono=True)
        else:
            # Load at native rate, avoid unnecessary resampling
            wav, actual_sr = librosa.load(fp, sr=None, mono=True)
            if actual_sr != sr:
                # Resample only if detection was wrong
                wav = librosa.resample(wav, orig_sr=actual_sr, target_sr=sr)
                actual_sr = sr

        # Double-check length after loading (in case duration detection was inaccurate)
        if len(wav) / actual_sr * 1000 < min_len_ms:
            skipped_counter.value += 1
            return None

        # ─── spectrogram ─────────────────────────────────────────────
        S = compute_spectrogram(
            wav, actual_sr, n_fft, step,
            mel=use_mel, n_mels=n_mels)

        if S.shape[1] < min_timebins:
            skipped_counter.value += 1
            return None

        labels = np.zeros(S.shape[1], dtype=np.int32)
        for lab, tb_on, tb_off in lab_map.get(fp.stem, []):
            labels[tb_on:tb_off] = lab

        # ─── optimized output with minimal conversions ───────────────
        if fmt == "pt":
            import torch
            out = dst_dir / (fp.stem + ".pt")
            # S uses default dtype, avoid unnecessary conversion
            torch.save({"s": torch.from_numpy(S),
                        "labels": torch.from_numpy(labels)}, out)
        else:  # npz (uncompressed)
            out = dst_dir / (fp.stem + ".npz")
            # S uses default dtype from compute_spectrogram
            np.savez(out, s=S, labels=labels)

        # free memory fast in workers
        del wav, S, labels
        gc.collect()
        return None
        
    except Exception as e:
        skipped_counter.value += 1
        return f"{fp}: {e}"


def calculate_optimal_workers(total_files: int, avg_file_size_mb: float = 50) -> int:
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


def batch_write_outputs(outputs: List[Tuple[Path, np.ndarray, np.ndarray]], fmt: str) -> None:
    """
    Batch write multiple outputs for better I/O efficiency.
    This is a simple optimization - more complex batching could be implemented.
    """
    for out_path, S, labels in outputs:
        if fmt == "pt":
            import torch
            torch.save({"s": torch.from_numpy(S),
                        "labels": torch.from_numpy(labels)}, out_path)
        else:  # npz
            np.savez(out_path, s=S, labels=labels)


# ══════════════════════════════════════════════════════════════════════════════
# main worker class
# ══════════════════════════════════════════════════════════════════════════════
class WavToSpec:
    """
    Convert a directory (or explicit list) of audio files to .npz spectrograms.
    Keys inside the .npz **match what BirdSpectrogramDataset expects**:
        s       -> (F,T)   log spectrogram
        labels  -> int32   (T,)    all zeros (placeholder)
    """

    def __init__(
        self,
        src_dir: str | None,
        dst_dir: str,
        *,
        file_list: str | None = None,
        step_size: int = 160,
        n_fft: int = 1024,
        sr: int = 32_000,
        take_n_random: int | None = None,
        single_threaded: bool = True,
        min_len_ms: int = 25,
        min_timebins: int = 25,
        fmt: str = "pt",
        mel: bool = True,
        n_mels: int = 128,
        json_path: str | None = None,
        max_workers: int | None = None,
    ) -> None:
        self.src_dir = Path(src_dir) if src_dir is not None else None
        self.dst_dir = Path(dst_dir)
        self.dst_dir.mkdir(parents=True, exist_ok=True)

        self.file_list = Path(file_list) if file_list else None
        self.step = step_size
        self.n_fft = n_fft
        self.sr = sr
        self.take_n_random = take_n_random
        self.single = single_threaded
        self.min_len_ms = min_len_ms
        self.min_timebins = min_timebins
        self.fmt = fmt
        self.use_mel = mel
        self.n_mels = n_mels
        self.max_workers = max_workers

        self._setup_logging()
        # Remove unpicklable Manager().Value - will create in run() if needed

        self.audio_files = self._gather_files()
        
        # Save audio parameters to destination directory
        self._save_audio_params()

        # Build label map if json_path is provided
        if json_path is not None:
            self.lab_map = {}
            p = Path(json_path)
            json_files = [p] if p.is_file() else list(p.glob("*.json"))

            for jfp in json_files:
                text = jfp.read_text()
                # Allow [ {...}, {...} ]    OR    {"filename": ...}    OR   NDJSON
                to_parse = text.strip()
                items = []
                if to_parse.startswith('['):                       # big list
                    items = json.loads(to_parse)
                elif to_parse.startswith('{'):                     # single object
                    items = [json.loads(to_parse)]
                else:                                              # NDJSON
                    items = [json.loads(line) for line in to_parse.splitlines() if line.strip()]

                for jo in items:
                    fname = jo["filename"]
                    hop_ms = 1e3 * self.step / self.sr
                    tmp = []
                    for lab, spans in jo.get("syllable_labels", {}).items():
                        for on, off in spans:
                            tb_on  = int(round(on  * 1e3 / hop_ms))
                            tb_off = int(round(off * 1e3 / hop_ms))
                            tmp.append((int(lab), tb_on, tb_off))
                    self.lab_map[Path(fname).stem] = tmp            # key on *stem*
        else:
            self.lab_map = {}

    # ──────────────────────────────────────────────────────────────────────
    # misc
    # ──────────────────────────────────────────────────────────────────────
    def _setup_logging(self) -> None:
        logging.basicConfig(
            filename="error_log.log",
            level=logging.ERROR,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def _save_audio_params(self) -> None:
        """Save audio processing parameters to JSON file in destination directory."""
        params = {
            "sr": self.sr,
            "mels": self.n_mels,
            "hop_size": self.step,
            "fft": self.n_fft
        }
        
        params_file = self.dst_dir / "audio_params.json"
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)

    def _gather_files(self) -> list[Path]:
        if self.file_list:
            files = [Path(line.strip()) for line in self.file_list.read_text().splitlines() if line.strip()]
        else:
            exts = (".wav", ".mp3", ".ogg", ".flac")
            files = [
                Path(root) / f
                for root, _, fs in os.walk(self.src_dir)
                for f in fs if f.lower().endswith(exts)
            ]

        if not files:
            print("no audio files matched ‑ nothing to do.")
            return []

        if self.take_n_random and self.take_n_random < len(files):
            files = random.sample(files, self.take_n_random)

        return files

    # ──────────────────────────────────────────────────────────────────────
    # public entry
    # ──────────────────────────────────────────────────────────────────────
    def run(self) -> None:
        if not self.audio_files:
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
                    self.min_timebins, self.fmt, self.lab_map, skipped_counter
                )
                
                failed_files = []
                
                with ctx.Pool(
                    processes=num_workers, 
                    maxtasksperchild=10,  # Lower value to prevent memory accumulation
                    initargs=()
                ) as pool:
                    
                    # Use imap_unordered for better memory control
                    task_args = [(fp,) + worker_args for fp in self.audio_files]
                    
                    try:
                        for i, result in enumerate(pool.imap_unordered(
                            self._worker_wrapper, task_args, chunksize=1
                        )):
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
    def _safe_process(self, fp: Path) -> Optional[str]:
        """
        Single-threaded processing wrapper with optimizations.
        Returns None on success, error message on failure.
        """
        try:
            # ─── check if output already exists ─────────────────────────
            if self.fmt == "pt":
                out_path = self.dst_dir / (fp.stem + ".pt")
            else:
                out_path = self.dst_dir / (fp.stem + ".npz")
            
            if out_path.exists():
                return None  # Skip already processed files
            
            # ─── fast duration check before loading ─────────────────────
            try:
                duration_sec = librosa.get_duration(path=fp)
                if duration_sec * 1000 < self.min_len_ms:
                    return None  # Skip, but not an error
            except Exception:
                # Fallback: if duration check fails, proceed with loading
                pass

            # ─── smart loading with sample rate detection ────────────────
            # First, detect native sample rate without loading audio
            try:
                native_sr = librosa.get_samplerate(fp)
                needs_resampling = (native_sr != self.sr)
            except Exception:
                # Fallback: assume resampling needed
                needs_resampling = True
                native_sr = None

            # Load audio with optimal settings
            if needs_resampling:
                wav, actual_sr = librosa.load(fp, sr=self.sr, mono=True)
            else:
                # Load at native rate, avoid unnecessary resampling
                wav, actual_sr = librosa.load(fp, sr=None, mono=True)
                if actual_sr != self.sr:
                    # Resample only if detection was wrong
                    wav = librosa.resample(wav, orig_sr=actual_sr, target_sr=self.sr)
                    actual_sr = self.sr

            # Double-check length after loading (in case duration detection was inaccurate)
            if len(wav) / actual_sr * 1000 < self.min_len_ms:
                return None  # Skip, but not an error

            # ─── spectrogram ─────────────────────────────────────────────
            S = compute_spectrogram(
                    wav, actual_sr, self.n_fft, self.step,
                    mel=self.use_mel, n_mels=self.n_mels)

            if S.shape[1] < self.min_timebins:
                return None  # Skip, but not an error

            labels = np.zeros(S.shape[1], dtype=np.int32)
            for lab, tb_on, tb_off in self.lab_map.get(fp.stem, []):
                labels[tb_on:tb_off] = lab

            # ─── optimized output with minimal conversions ───────────────
            if self.fmt == "pt":
                import torch
                out = self.dst_dir / (fp.stem + ".pt")
                # S uses default dtype, avoid unnecessary conversion
                torch.save({"s": torch.from_numpy(S),
                            "labels": torch.from_numpy(labels)}, out)
            else:  # npz (uncompressed)
                out = self.dst_dir / (fp.stem + ".npz")
                # S uses default dtype from compute_spectrogram
                np.savez(out, s=S, labels=labels)

            # free memory fast
            del wav, S, labels
            gc.collect()
            return None
            
        except Exception as e:
            return f"{fp}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def cli() -> None:
    p = argparse.ArgumentParser(
        description="Convert audio → log‑spectrogram .npz (no JSON, no filtering).")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--src_dir",   type=str,
                     help="Root folder with wav/mp3/ogg files (searched recursively).")
    grp.add_argument("--file_list", type=str,
                     help="Text file with absolute/relative paths, one per line.")
    p.add_argument("--dst_dir",  type=str, required=True,
                   help="Where outputs go.")
    p.add_argument("--format", choices=["pt","npz"], default="pt",
                   help="output format (default: pt, fp32)")

    p.add_argument("--sr", type=int, default=32_000,
                   rehelp="Sample rate in Hz (default: 32000).")
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
    p.add_argument("--json_path", type=str, default=None,
                   help="Directory containing label JSON files (optional)")
    p.add_argument("--max_workers", type=int, default=None,
                   help="Maximum number of worker processes (default: auto-detect)")
    args = p.parse_args()

    single = args.single_threaded.lower() in {"true", "1", "yes"}

    converter = WavToSpec(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        file_list=args.file_list,
        step_size=args.step_size,
        n_fft=args.nfft,
        sr=args.sr,
        take_n_random=args.take_n_random,
        single_threaded=single,
        fmt=args.format,
        mel=not args.linear,
        n_mels=args.n_mels,
        json_path=args.json_path,
        max_workers=args.max_workers,
    )
    converter.run()


if __name__ == "__main__":
    cli()