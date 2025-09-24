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

from scipy import ndimage

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
) -> tuple[np.ndarray, np.ndarray]:
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
    frame_ms = hop / sr * 1000.0
    S_db = librosa.power_to_db(S, ref=np.max(S), top_db=None)
    chirp_intervals, _ , _ = classify_loudness(S_db, frame_ms)
    return S_db, np.asarray(chirp_intervals, dtype=np.int32).reshape(-1, 2)

def classify_loudness(spec_db: np.ndarray, frame_ms: float, merge_ms: float = 30.0) -> tuple[list[tuple[int, int]], np.ndarray, float]:
    
    def compute_loudness(spec_db: np.ndarray) -> np.ndarray:
        spec_power = np.power(10.0, spec_db / 10.0, dtype=np.float64)
        loudness = np.sum(np.log1p(spec_power), axis=0, dtype=np.float64)
        return np.nan_to_num(loudness, nan=0.0, posinf=0.0, neginf=0.0)

    def otsu_threshold_lower(x: np.ndarray, nbins: int = 512, rounds: int = 4) -> float:
        """
        Recursively apply Otsu on the lower (<= threshold) class to ignore loud tails.
        'rounds' = how many times to refine (2 is usually enough).
        """
        x = np.asarray(x, np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return 0.0

        thr = otsu_threshold(x, nbins=nbins)
        for _ in range(max(0, rounds - 1)):
            lower = x[x <= thr]
            if lower.size < 16:  # too few samples to re-estimate a histogram robustly
                break
            new_thr = otsu_threshold(lower, nbins=nbins)
            if not np.isfinite(new_thr) or abs(new_thr - thr) < 1e-12:
                break
            thr = new_thr
        return float(thr)

    def otsu_threshold(x: np.ndarray, nbins: int = 512) -> float:
        """Return scalar threshold (Otsu). Works on 1-D array."""
        x = np.asarray(x, np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return 0.0
        hist, edges = np.histogram(x, bins=nbins)
        hist = hist.astype(np.float64)
        p = hist / hist.sum()
        w_cum = np.cumsum(p)
        mu = np.cumsum(p * (edges[:-1] + edges[1:]) * 0.5)
        mu_t = mu[-1]
        # between-class variance
        sigma_b2 = (mu_t * w_cum - mu)**2 / (w_cum * (1.0 - w_cum) + 1e-12)
        k = np.nanargmax(sigma_b2)
        # threshold is bin edge between k and k+1
        return (edges[k] + edges[k+1]) * 0.5

    def intervals_from_mask(mask: np.ndarray) -> list[tuple[int,int]]:
        """Turn a boolean mask (T,) into a list of (start,end) with end exclusive."""
        out = []
        in_run = False
        s = 0
        for i, v in enumerate(mask):
            if v and not in_run:
                in_run = True
                s = i
            elif not v and in_run:
                in_run = False
                out.append((s, i))
        if in_run:
            out.append((s, mask.size))
        return out

    loudness = compute_loudness(spec_db)
    loudness = ndimage.median_filter(loudness, size=5)
    thr = otsu_threshold_lower(loudness)
    chirp_intervals = intervals_from_mask((loudness > thr))

    # Merge chirps that are closer than 30 ms apart
    gap_frames = max(1, int(round(merge_ms/ max(frame_ms, 1e-9))))
    merged_chirps: list[tuple[int, int]] = []
    for s, e in chirp_intervals:
        if not merged_chirps:
            merged_chirps.append((s, e))
            continue
        ps, pe = merged_chirps[-1]
        if s - pe <= gap_frames:
            merged_chirps[-1] = (ps, max(pe, e))
        else:
            merged_chirps.append((s, e))

    return merged_chirps, loudness, float(thr)


# ──────────────────────────────────────────────────────────────────────────────
# stats helpers
# ──────────────────────────────────────────────────────────────────────────────
def _reduce_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {
            "count": 0,
            "sum": 0.0,
            "sumsq": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    s = float(np.sum(x))
    ss = float(np.sum(x * x))
    n = int(x.size)
    mu = s / n
    var = max(0.0, ss / n - mu * mu)
    return {
        "count": n,
        "sum": s,
        "sumsq": ss,
        "mean": float(mu),
        "std": float(np.sqrt(var)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def compute_chirp_stats(chirp_intervals: np.ndarray, frame_ms: float) -> dict:
    """Compute per-file chirp duration statistics in **array positions** (frame columns).

    Args:
        chirp_intervals: array-like of shape (N, 2) with [start, end) frame indices.
        frame_ms: duration of one time frame in milliseconds. (Ignored here.)

    Returns:
        dict with keys:
            - "num_chirps": int
            - "dur": reduce over individual chirp durations **in columns**
            - "seq": {L: reduce over sliding-window sums of L consecutive chirps}
    """
    if chirp_intervals is None:
        chirp_intervals = np.empty((0, 2), dtype=np.int32)
    ci = np.asarray(chirp_intervals, dtype=np.int64).reshape(-1, 2)
    # durations per chirp in array positions (columns), including 1 for separator token
    dur_cols = ((ci[:, 1] - ci[:, 0]).astype(np.int64) + 1).astype(np.float64) if ci.size else np.empty((0,), dtype=np.float64)

    # Compute max chirp length and its start column
    if dur_cols.size:
        max_idx = int(np.argmax(dur_cols))
        max_len_cols = int(dur_cols[max_idx])
        max_start_col = int(ci[max_idx, 0])
    else:
        max_idx = -1
        max_len_cols = -1
        max_start_col = -1

    stats = {
        "num_chirps": int(dur_cols.size),
        "dur": _reduce_stats(dur_cols),
        "seq": {},
        "max_chirp_len_cols": max_len_cols,
        "max_chirp_start_col": max_start_col,
    }

    # Precompute cumulative sums for fast sliding-window totals
    if dur_cols.size:
        csum = np.concatenate([[0.0], np.cumsum(dur_cols)])
    else:
        csum = np.array([0.0], dtype=np.float64)

    for L in range(1, 26):
        if dur_cols.size >= L:
            # sliding window sums of length L
            totals = csum[L:] - csum[:-L]
            stats["seq"][L] = _reduce_stats(totals)
        else:
            stats["seq"][L] = {
                "count": 0,
                "sum": 0.0,
                "sumsq": 0.0,
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }

    return stats


def _agg_from_rows(rows: list[dict], key: str) -> dict:
    """Aggregate count/sum/sumsq/min/max across rows for a given key.
    Each row[key] is a dict with the same fields.
    """
    total_count = 0
    total_sum = 0.0
    total_sumsq = 0.0
    mins = []
    maxs = []
    for r in rows:
        s = r[key]
        c = int(s["count"]) if np.isfinite(s.get("count", 0)) else 0
        total_count += c
        total_sum += float(s.get("sum", 0.0))
        total_sumsq += float(s.get("sumsq", 0.0))
        if np.isfinite(s.get("min", np.nan)):
            mins.append(float(s["min"]))
        if np.isfinite(s.get("max", np.nan)):
            maxs.append(float(s["max"]))
    if total_count == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    mu = total_sum / total_count
    var = max(0.0, total_sumsq / total_count - mu * mu)
    return {
        "count": int(total_count),
        "mean": float(mu),
        "std": float(np.sqrt(var)),
        "min": float(np.min(mins)) if mins else float("nan"),
        "max": float(np.max(maxs)) if maxs else float("nan"),
    }


def aggregate_and_print_summary(summary_rows: list[dict], dst_dir: Path) -> None:
    """Save per-file stats to summary.npy and print global aggregates."""
    # Save rows for later analysis
    np.save(dst_dir / "summary.npy", np.array(summary_rows, dtype=object))

    if not summary_rows:
        print("No files processed; nothing to summarize.")
        return

    total_files = len(summary_rows)
    total_chirps = int(sum(r.get("num_chirps", 0) for r in summary_rows))
    print(f"\nSummary across {total_files} files (total chirps: {total_chirps}):")

    # Single-chirp duration aggregate
    dur_rows = [{"count": r["dur"]["count"],
                 "sum": r["dur"]["sum"],
                 "sumsq": r["dur"]["sumsq"],
                 "min": r["dur"]["min"],
                 "max": r["dur"]["max"]} for r in summary_rows]
    dur_agg = _agg_from_rows([{"count": d["count"], "sum": d["sum"], "sumsq": d["sumsq"], "min": d["min"], "max": d["max"]} for d in dur_rows], key=None) if False else None
    # simpler: directly compute from r["dur"] structures
    total_count = sum(int(r["dur"]["count"]) for r in summary_rows)
    total_sum = sum(float(r["dur"]["sum"]) for r in summary_rows)
    total_sumsq = sum(float(r["dur"]["sumsq"]) for r in summary_rows)
    mins = [float(r["dur"]["min"]) for r in summary_rows if np.isfinite(r["dur"]["min"]) ]
    maxs = [float(r["dur"]["max"]) for r in summary_rows if np.isfinite(r["dur"]["max"]) ]
    if total_count > 0:
        mu = total_sum / total_count
        var = max(0.0, total_sumsq / total_count - mu * mu)
        print(f"Single chirp duration (cols): mean={mu:.2f}, std={np.sqrt(var):.2f}, min={np.min(mins) if mins else float('nan'):.2f}, max={np.max(maxs) if maxs else float('nan'):.2f}, count={total_count}")
    else:
        print("Single chirp duration (cols): no data")

    # Sequence aggregates 1..25
    print("\nSliding window totals over consecutive chirps:")
    print("L\tmean(cols)\tstd(cols)\tmin\tmax\tcount")
    for L in range(1, 26):
        # collect rows that have this L
        rows_L = [r["seq"][L] for r in summary_rows if "seq" in r and L in r["seq"]]
        if not rows_L:
            continue
        c = sum(int(x["count"]) for x in rows_L)
        if c == 0:
            continue
        s = sum(float(x["sum"]) for x in rows_L)
        ss = sum(float(x["sumsq"]) for x in rows_L)
        mins = [float(x["min"]) for x in rows_L if np.isfinite(x["min"]) ]
        maxs = [float(x["max"]) for x in rows_L if np.isfinite(x["max"]) ]
        mu = s / c
        var = max(0.0, ss / c - mu * mu)
        mn = np.min(mins) if mins else float('nan')
        mx = np.max(maxs) if maxs else float('nan')
        print(f"{L}\t{mu:.2f}\t{np.sqrt(var):.2f}\t{mn:.2f}\t{mx:.2f}\t{c}")


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
    skipped_counter: Any,
    remake: bool = False,
) -> Optional[dict]:
    """
    Standalone worker function that processes a single audio file.
    Returns None on success, error message on failure.
    If remake is False, skips files that already exist; otherwise overwrites.
    """
    try:
        # ─── check if output already exists ─────────────────────────
        if fmt == "pt":
            out_path = dst_dir / (fp.stem + ".pt")
        else:
            out_path = dst_dir / (fp.stem + ".npz")

        if out_path.exists() and not remake:
            return {"file": str(fp), "skipped": True}

        # ─── fast duration check before loading ─────────────────────
        try:
            duration_sec = librosa.get_duration(path=fp)
            if duration_sec * 1000 < min_len_ms:
                skipped_counter.value += 1
                return {"file": str(fp), "skipped": True}
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
            return {"file": str(fp), "skipped": True}

        # ─── spectrogram ─────────────────────────────────────────────
        S, chirp_intervals = compute_spectrogram(
            wav, actual_sr, n_fft, step,
            mel=use_mel, n_mels=n_mels)

        if S.shape[1] < min_timebins:
            skipped_counter.value += 1
            return {"file": str(fp), "skipped": True}

        labels = np.zeros(S.shape[1], dtype=np.int32)
        for lab, tb_on, tb_off in lab_map.get(fp.stem, []):
            labels[tb_on:tb_off] = lab

        frame_ms = 1000.0 * step / actual_sr
        stats = compute_chirp_stats(chirp_intervals, frame_ms)
        file_stats = {"file": Path(fp).stem, "path": str(fp), **stats}

        # ─── optimized output with minimal conversions ───────────────
        if fmt == "pt":
            import torch
            out = dst_dir / (fp.stem + ".pt")
            # S uses default dtype, avoid unnecessary conversion
            torch.save({"s": torch.from_numpy(S),
                        "chirp_intervals": torch.from_numpy(chirp_intervals),
                        "labels": torch.from_numpy(labels)}, out)
        else:  # npz (uncompressed)
            out = dst_dir / (fp.stem + ".npz")
            # S uses default dtype from compute_spectrogram
            np.savez(out, s=S, chirp_intervals=chirp_intervals, labels=labels)

        # free memory fast in workers
        del wav, S, labels
        gc.collect()
        return file_stats

    except Exception as e:
        skipped_counter.value += 1
        return {"error": f"{fp}: {e}", "file": str(fp)}


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
        remake: bool = False,
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
        self.remake = remake
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

        summary_rows = []
        error_count = 0
        skipped_count = 0
        pbar = tqdm(total=len(self.audio_files), desc="processing files")

        try:
            if self.single:
                # ───── single threaded ─────
                for fp in self.audio_files:
                    result = self._safe_process(fp)
                    if not isinstance(result, dict):
                        pbar.update(); continue
                    if result.get("error"):
                        error_count += 1; skipped_count += 1; logging.error(result["error"]) ; pbar.update(); continue
                    if result.get("skipped"):
                        skipped_count += 1; pbar.update(); continue
                    summary_rows.append(result)
                    print(f"MAX CHIRP: len={result.get('max_chirp_len_cols', -1)} cols, start_col={result.get('max_chirp_start_col', -1)}, file={result.get('file')}")
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
                    self.min_timebins, self.fmt, self.lab_map, skipped_counter, self.remake
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
                            if isinstance(result, dict):
                                if result.get("error"):
                                    failed_files.append(result["error"])  # keep string for retry parsing
                                    logging.error(result["error"])
                                elif result.get("skipped"):
                                    pass  # count later from mgr counter
                                else:
                                    print(f"MAX CHIRP: len={result.get('max_chirp_len_cols', -1)} cols, start_col={result.get('max_chirp_start_col', -1)}, file={result.get('file')}")
                                    summary_rows.append(result)
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
                            if isinstance(result, dict):
                                if result.get("error"):
                                    error_count += 1; skipped_count += 1; logging.error(result["error"])
                                elif result.get("skipped"):
                                    skipped_count += 1
                                else:
                                    print(f"MAX CHIRP: len={result.get('max_chirp_len_cols', -1)} cols, start_col={result.get('max_chirp_start_col', -1)}, file={result.get('file')}")
                                    summary_rows.append(result)
                                    skipped_count -= 1  # Successfully processed on retry

        except KeyboardInterrupt:
            print("\nOperation interrupted by user")
            sys.exit(1)
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_sigint_handler)
            pbar.close()

        aggregate_and_print_summary(summary_rows, self.dst_dir)
        processed_count = len(summary_rows)
        print(f"Total processed: {processed_count}")
        print(f"Total skipped  : {skipped_count}")
        if error_count:
            print(f"Total errors   : {error_count}")
    
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
    def _safe_process(self, fp: Path) -> Optional[dict]:
        """
        Single-threaded processing wrapper with optimizations.
        Returns dict with stats or skip/error info.
        """
        try:
            # ─── check if output already exists ─────────────────────────
            if self.fmt == "pt":
                out_path = self.dst_dir / (fp.stem + ".pt")
            else:
                out_path = self.dst_dir / (fp.stem + ".npz")

            if out_path.exists() and not self.remake:
                return {"file": str(fp), "skipped": True}

            # ─── fast duration check before loading ─────────────────────
            try:
                duration_sec = librosa.get_duration(path=fp)
                if duration_sec * 1000 < self.min_len_ms:
                    return {"file": str(fp), "skipped": True}
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
                return {"file": str(fp), "skipped": True}

            # ─── spectrogram ─────────────────────────────────────────────
            S, chirp_intervals = compute_spectrogram(
                    wav, actual_sr, self.n_fft, self.step,
                    mel=self.use_mel, n_mels=self.n_mels)

            if S.shape[1] < self.min_timebins:
                return {"file": str(fp), "skipped": True}

            labels = np.zeros(S.shape[1], dtype=np.int32)
            for lab, tb_on, tb_off in self.lab_map.get(fp.stem, []):
                labels[tb_on:tb_off] = lab

            frame_ms = 1000.0 * self.step / actual_sr
            stats = compute_chirp_stats(chirp_intervals, frame_ms)
            file_stats = {"file": Path(fp).stem, "path": str(fp), **stats}

            # ─── optimized output with minimal conversions ───────────────
            if self.fmt == "pt":
                import torch
                out = self.dst_dir / (fp.stem + ".pt")
                # S uses default dtype, avoid unnecessary conversion
                torch.save({"s": torch.from_numpy(S),
                            "chirp_intervals": torch.from_numpy(chirp_intervals),
                            "labels": torch.from_numpy(labels)}, out)
            else:  # npz (uncompressed)
                out = self.dst_dir / (fp.stem + ".npz")
                # S uses default dtype from compute_spectrogram
                np.savez(out, s=S, chirp_intervals=chirp_intervals, labels=labels)

            # free memory fast
            del wav, S, labels
            gc.collect()
            return file_stats

        except Exception as e:
            return {"error": f"{fp}: {e}", "file": str(fp)}


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
    p.add_argument("--json_path", type=str, default=None,
                   help="Directory containing label JSON files (optional)")
    p.add_argument("--max_workers", type=int, default=None,
                   help="Maximum number of worker processes (default: auto-detect)")
    p.add_argument("--remake", action="store_true",
                   help="Recompute and overwrite outputs even if they already exist.")                   
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
        remake=args.remake,
    )
    converter.run()


if __name__ == "__main__":
    cli()