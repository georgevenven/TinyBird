# ──────────────────────────────────────────────────────────────────────────────
# audio2spec.py  ‑  simple .wav /.mp3/.ogg ➜ spectrogram (.npz / .pt) converter
# ──────────────────────────────────────────────────────────────────────────────
import torch

import os
import json
import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Any, Optional
from types import SimpleNamespace

import time

import numpy as np
import librosa
from tqdm import tqdm

from scipy import ndimage
import scipy.signal as ss

class SingleChannelProcessor:
    __slots__ = (
        "args",
        "S_db",
        "chirp_intervals",
        "ref",
        "mel_spectrogram",
        "mfcc",
        "block_lengths",
        "stage_timings",
    )

    def __init__(self, args: SimpleNamespace) -> None:
        self.args = args
        self.S_db = np.empty((0, 0), dtype=np.float32)
        self.chirp_intervals = np.empty((0, 2), dtype=np.int32)
        self.ref = 0.0
        self.mel_spectrogram = np.empty((0, 0), dtype=np.float32)
        self.mfcc = np.empty((0, 0), dtype=np.float32)
        self.block_lengths = np.empty((0,), dtype=np.int32)
        self.stage_timings: dict[str, float] = {}

        stage = time.perf_counter()
        self._compute_spectrogram()
        self._record_stage("spectrogram", stage, extra=f"shape={self.S_db.shape}")

        stage = time.perf_counter()
        self._compute_mfcc()
        self._record_stage("mfcc", stage, extra=f"shape={self.mfcc.shape}")

        stage = time.perf_counter()
        self._classify_loudness()
        self._record_stage("classify_loudness", stage, extra=f"blocks={self.chirp_intervals.shape[0]}")

        if logging.getLogger().isEnabledFor(logging.INFO):
            summary = {k: round(v, 3) for k, v in self.stage_timings.items()}
            logging.info("Channel %s: timing summary %s", getattr(self.args, "channel_index", -1), summary)

    def _record_stage(self, name: str, start_time: float, *, extra: str = "") -> None:
        duration = time.perf_counter() - start_time
        self.stage_timings[name] = duration
        channel = getattr(self.args, "channel_index", -1)
        extras = f" ({extra})" if extra else ""
        if logging.getLogger().isEnabledFor(logging.INFO):
            logging.info("Channel %s: %s completed in %.3fs%s", channel, name, duration, extras)

    def _compute_spectrogram(self) -> None:
        S = librosa.feature.melspectrogram(
            y=self.args.wav,
            sr=self.args.sr,
            n_fft=self.args.n_fft,
            hop_length=self.args.hop_length,
            power=2.0,
            n_mels=self.args.n_mels,
            fmin=20,
            fmax=self.args.sr // 2,
        )
        self.ref = float(np.max(S))
        self.mel_spectrogram = S.astype(np.float32, copy=False)
        self.S_db = librosa.power_to_db(S, ref=self.ref, top_db=None).astype(np.float32, copy=False)

    def _compute_mfcc(self) -> None:
        n_mfcc = int(getattr(self.args, "n_mfcc", 20))
        n_mfcc = max(n_mfcc, 1)
        mfcc = librosa.feature.mfcc(
            y=self.args.wav,
            sr=self.args.sr,
            n_mfcc=n_mfcc,
            hop_length=self.args.hop_length,
            n_fft=self.args.n_fft,
            htk=False,
        )
        if mfcc.shape[0] > 1:
            mfcc = mfcc[1:, :]
        else:
            mfcc = np.empty((0, mfcc.shape[1]), dtype=mfcc.dtype)
        self.mfcc = mfcc.astype(np.float32, copy=False)

    def _classify_loudness(self, merge_ms: float = 200.0) -> None:
        frame_ms = self.args.hop_length / self.args.sr * 1000.0

        def compute_loudness(spec_db: np.ndarray) -> np.ndarray:
            spec_power = np.power(10.0, spec_db / 10.0, dtype=np.float64)
            loudness = np.sum(np.log1p(spec_power), axis=0, dtype=np.float64)
            return np.nan_to_num(loudness, nan=0.0, posinf=0.0, neginf=0.0)

        def otsu_threshold_lower(x: np.ndarray, nbins: int = 512, rounds: int = 4) -> float:
            x = np.asarray(x, np.float64)
            x = x[np.isfinite(x)]
            if x.size == 0:
                return 0.0

            thr = otsu_threshold(x, nbins=nbins)
            for _ in range(max(0, rounds - 1)):
                lower = x[x <= thr]
                if lower.size < 16:
                    break
                new_thr = otsu_threshold(lower, nbins=nbins)
                if not np.isfinite(new_thr) or abs(new_thr - thr) < 1e-12:
                    break
                thr = new_thr
            return float(thr)

        def otsu_threshold(x: np.ndarray, nbins: int = 512) -> float:
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
            sigma_b2 = (mu_t * w_cum - mu) ** 2 / (w_cum * (1.0 - w_cum) + 1e-12)
            k = np.nanargmax(sigma_b2)
            return (edges[k] + edges[k + 1]) * 0.5

        def intervals_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
            out: list[tuple[int, int]] = []
            in_run = False
            start = 0
            for idx, value in enumerate(mask):
                if value and not in_run:
                    in_run = True
                    start = idx
                elif not value and in_run:
                    in_run = False
                    out.append((start, idx))
            if in_run:
                out.append((start, mask.size))
            return out

        if self.S_db.size == 0:
            self.chirp_intervals = np.empty((0, 2), dtype=np.int32)
            self._initialize_block_metadata()
            return

        loudness = compute_loudness(self.S_db)
        loudness = ndimage.median_filter(loudness, size=5)
        thr = otsu_threshold_lower(loudness)
        chirp_intervals = intervals_from_mask(loudness > thr)

        self.chirp_intervals = np.asarray(chirp_intervals, dtype=np.int32).reshape(-1, 2)

        self._initialize_block_metadata()

    def _initialize_block_metadata(self) -> None:
        intervals = np.asarray(self.chirp_intervals, dtype=np.int32).reshape(-1, 2)
        if intervals.size == 0:
            self.block_lengths = np.empty((0,), dtype=np.int32)
            return

        lengths = np.maximum(intervals[:, 1] - intervals[:, 0], 0)
        self.block_lengths = lengths.astype(np.int32, copy=False)
        self.chirp_intervals = intervals

class TwoChannelFileProcessor:
    desired_channels = 2

    def __init__(self, args: SimpleNamespace) -> None:
        base = SimpleNamespace(**vars(args))
        base.fp = Path(base.fp)
        base.dst_dir = Path(base.dst_dir)
        base.dst_dir.mkdir(parents=True, exist_ok=True)
        base.hop_length = getattr(base, "hop_length", base.step)
        base.s_ref = getattr(base, "s_ref", None)
        self.args = base
        self.out_path = base.dst_dir / (base.fp.stem + ".pt")
        self.actual_sr: int = base.sr
        self.channel_processors: list[SingleChannelProcessor] = []
        self.chirp_intervals = np.empty((0, 2), dtype=np.int32)
        self.chirp_lengths = np.empty((0,), dtype=np.int32)
        self.S_stack = np.empty((0, 0, 0), dtype=np.float32)
        self.frame_ms: float = 0.0

    @staticmethod
    def _high_pass_filter(
        audio_signal: np.ndarray, *, sample_rate: int = 32000, cutoff: int = 512, order: int = 5
    ) -> np.ndarray:
        sos = ss.butter(order, cutoff, btype="high", fs=sample_rate, output="sos")
        return ss.sosfilt(sos, audio_signal, axis=-1)

    @staticmethod
    def _detect_and_load_audio(fp: Path, target_sr: int, channel: int | str = -1) -> tuple[np.ndarray, int, int]:
        try:
            duration_sec = librosa.get_duration(path=fp)
            if duration_sec * 1000 < 0:
                pass
        except Exception:
            pass

        try:
            native_sr = librosa.get_samplerate(fp)
        except Exception:
            native_sr = None
        needs_resampling = (native_sr != target_sr) if native_sr else True

        try:
            import soundfile as sf

            with sf.SoundFile(fp) as f:
                channel_count = int(f.channels)
        except Exception:
            try:
                y_probe, sr_probe = librosa.load(fp, sr=None, mono=False, duration=0.01)
                channel_count = 1 if np.ndim(y_probe) == 1 else int(y_probe.shape[0])
            except Exception:
                channel_count = 1

        take_all_channels = channel == "all"
        mono = channel == -1
        wav, actual_sr = librosa.load(
            fp, sr=target_sr if needs_resampling else None, mono=mono if not take_all_channels else False
        )
        if take_all_channels:
            if wav.ndim == 1:
                wav = wav[np.newaxis, :]
        elif not mono:
            wav = wav[int(channel), :]

        if not needs_resampling and actual_sr != target_sr:
            wav = librosa.resample(wav, orig_sr=actual_sr, target_sr=target_sr)
            actual_sr = target_sr

        wav = TwoChannelFileProcessor._high_pass_filter(wav, sample_rate=actual_sr, cutoff=512, order=5)
        return wav, actual_sr, channel_count

    def process(self) -> Optional[dict]:
        try:
            file_start = time.perf_counter()
            logging.info("Starting processing for %s", self.args.fp)
            skip = self._maybe_skip_existing()
            if skip:
                return skip

            duration_skip = self._maybe_skip_short_duration()
            if duration_skip:
                return duration_skip

            load_start = time.perf_counter()
            wav_multi, self.actual_sr, channel_count = self._detect_and_load_audio(
                self.args.fp, self.args.sr, channel="all"
            )
            duration_sec = wav_multi.shape[-1] / self.actual_sr if wav_multi.size else 0.0
            logging.info(
                "Loaded audio %s: shape=%s, duration=%.2fs (%.2fs elapsed)",
                self.args.fp.name,
                wav_multi.shape,
                duration_sec,
                time.perf_counter() - load_start,
            )
            if channel_count < self.desired_channels:
                return self._skip(reason="mono_audio")

            self._prepare_channels(wav_multi)
            if self.S_stack.size == 0:
                raise ValueError(f"{self.args.fp}: failed to build channel spectrograms")

            # self.chirp_intervals = self._merge_chirps(proc.chirp_intervals for proc in self.channel_processors)
            stacked, lengths = self._stack_channel_chirps()
            self.chirp_intervals = stacked
            self.chirp_lengths = lengths

            self.frame_ms = 1000.0 * self.args.hop_length / self.actual_sr
            stats = self._compute_chirp_stats()
            file_stats = {"file": self.args.fp.stem, "path": str(self.args.fp), "frame_ms": self.frame_ms, **stats}
            self._save_outputs()
            logging.info("Finished processing %s in %.3fs", self.args.fp.name, time.perf_counter() - file_start)
            return file_stats
        except Exception as exc:
            return {"error": f"{self.args.fp}: {exc}", "file": str(self.args.fp)}

    def _maybe_skip_existing(self) -> Optional[dict]:
        if self.out_path.exists() and not self.args.remake:
            return self._skip()
        return None

    def _maybe_skip_short_duration(self) -> Optional[dict]:
        try:
            duration_sec = librosa.get_duration(path=self.args.fp)
            if duration_sec * 1000 < self.args.min_len_ms:
                return self._skip()
        except Exception:
            pass
        return None

    def _prepare_channels(self, wav_multi: np.ndarray) -> None:
        if wav_multi.ndim == 1:
            wav_multi = wav_multi[np.newaxis, :]
        available = int(wav_multi.shape[0])
        if available == 0:
            raise ValueError(f"{self.args.fp}: no audio channels detected")

        self.channel_processors.clear()
        channel_start = time.perf_counter()
        for idx in range(min(self.desired_channels, available)):
            wav_ch = np.ascontiguousarray(wav_multi[idx])
            per_channel_start = time.perf_counter()
            processor = SingleChannelProcessor(
                SimpleNamespace(
                    wav=wav_ch,
                    sr=self.actual_sr,
                    n_fft=self.args.n_fft,
                    hop_length=self.args.hop_length,
                    n_mels=self.args.n_mels,
                    channel_index=idx,
                    file_stem=self.args.fp.stem,
                    file_path=str(self.args.fp),
                )
            )
            per_channel_duration = time.perf_counter() - per_channel_start
            logging.info("File %s: channel %d processed in %.3fs", self.args.fp.name, idx, per_channel_duration)
            self.channel_processors.append(processor)

        logging.info(
            "File %s: processed %d channel(s) in %.3fs",
            self.args.fp.name,
            len(self.channel_processors),
            time.perf_counter() - channel_start,
        )

        if not self.channel_processors:
            raise ValueError(f"{self.args.fp}: unable to initialize channel processors")

        self.S_stack = np.stack([proc.S_db for proc in self.channel_processors], axis=0).astype(np.float32, copy=False)

    def _stack_channel_chirps(self) -> tuple[np.ndarray, np.ndarray]:
        per_channel = [
            np.asarray(proc.chirp_intervals, dtype=np.int32).reshape(-1, 2)
            for proc in self.channel_processors
        ]
        if not per_channel:
            return np.empty((0, 0, 2), dtype=np.int32), np.empty((0,), dtype=np.int32)
        max_len = max(arr.shape[0] for arr in per_channel)
        if max_len == 0:
            return np.zeros((len(per_channel), 0, 2), dtype=np.int32), np.zeros(
                len(per_channel), dtype=np.int32
            )
        lengths = np.array([arr.shape[0] for arr in per_channel], dtype=np.int32)
        stacked = np.full((len(per_channel), max_len, 2), -1, dtype=np.int32)
        for idx, arr in enumerate(per_channel):
            if arr.size:
                stacked[idx, : arr.shape[0], :] = arr
        return stacked, lengths

    def _merge_chirps(self, interval_iterable: Any) -> np.ndarray:
        arrays = [
            np.asarray(intervals, dtype=np.int64).reshape(-1, 2)
            for intervals in interval_iterable
            if intervals is not None
        ]
        non_empty = [arr for arr in arrays if arr.size]
        if not non_empty:
            return np.empty((0, 2), dtype=np.int32)

        combined = np.vstack(non_empty)
        combined = combined[np.argsort(combined[:, 0])]
        merged: list[tuple[int, int]] = []
        for start, end in combined:
            start_i = int(start)
            end_i = int(end)
            if not merged or start_i > merged[-1][1]:
                merged.append((start_i, end_i))
            else:
                ps, pe = merged[-1]
                merged[-1] = (ps, max(pe, end_i))
        merged_arr = np.asarray(merged, dtype=np.int32)
        if merged_arr.size:
            if np.any(merged_arr[:, 1] <= merged_arr[:, 0]):
                raise ValueError(f"{self.args.fp}: invalid chirp interval detected: {merged_arr}")
            if merged_arr.shape[0] > 1 and np.any(merged_arr[1:, 0] < merged_arr[:-1, 1]):
                raise ValueError(f"{self.args.fp}: overlapping chirp intervals after merge: {merged_arr}")
        return merged_arr

    @staticmethod
    def _reduce_stats(x: np.ndarray) -> dict:
        arr = np.asarray(x, dtype=np.float64)
        if arr.size == 0:
            return {
                "count": 0,
                "sum": 0.0,
                "sumsq": 0.0,
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        total = float(np.sum(arr))
        total_sq = float(np.sum(arr * arr))
        count = int(arr.size)
        mean = total / count
        var = max(0.0, total_sq / count - mean * mean)
        return {
            "count": count,
            "sum": total,
            "sumsq": total_sq,
            "mean": float(mean),
            "std": float(np.sqrt(var)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    def _compute_chirp_stats(self) -> dict:
        ci = np.asarray(self.chirp_intervals, dtype=np.int64)
        if ci.ndim == 3:
            flattened: list[np.ndarray] = []
            lengths = np.asarray(self.chirp_lengths, dtype=np.int64)
            for channel_idx, channel in enumerate(ci):
                if channel.size == 0:
                    continue
                limit = int(lengths[channel_idx]) if channel_idx < lengths.size else channel.shape[0]
                if limit <= 0:
                    continue
                flattened.append(channel[:limit])
            if flattened:
                ci = np.vstack(flattened)
            else:
                ci = np.empty((0, 2), dtype=np.int64)
        else:
            ci = ci.reshape(-1, 2)
        dur_cols = (
            ((ci[:, 1] - ci[:, 0]).astype(np.int64) + 1).astype(np.float64)
            if ci.size
            else np.empty((0,), dtype=np.float64)
        )

        if dur_cols.size:
            max_idx = int(np.argmax(dur_cols))
            max_len_cols = int(dur_cols[max_idx])
            max_start_col = int(ci[max_idx, 0])
        else:
            max_len_cols = -1
            max_start_col = -1

        stats = {
            "num_chirps": int(dur_cols.size),
            "dur": TwoChannelFileProcessor._reduce_stats(dur_cols),
            "seq": {},
            "max_chirp_len_cols": max_len_cols,
            "max_chirp_start_col": max_start_col,
        }

        if dur_cols.size:
            csum = np.concatenate([[0.0], np.cumsum(dur_cols)])
        else:
            csum = np.array([0.0], dtype=np.float64)

        for L in range(1, 26):
            if dur_cols.size >= L:
                totals = csum[L:] - csum[:-L]
                stats["seq"][L] = TwoChannelFileProcessor._reduce_stats(totals)
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

    def _save_outputs(self) -> None:
        hop_length = int(getattr(self.args, "hop_length", 0))
        sr = int(getattr(self, "actual_sr", getattr(self.args, "sr", 0)))
        frame_step = float(hop_length) / float(sr) if sr > 0 else None
        meta = {
            "sr": sr,
            "sample_rate": sr,
            "hop_length": hop_length,
            "step_size": hop_length,
            "frame_step": frame_step,
            "n_fft": int(getattr(self.args, "n_fft", 0)),
            "n_mels": int(getattr(self.args, "n_mels", 0)),
            "channels": self.desired_channels,
            "file_path": str(getattr(self.args, "fp", "")),
        }
        payload = {
            "s": torch.from_numpy(self.S_stack),
            "chirp_intervals": torch.from_numpy(self.chirp_intervals),
            "chirp_lengths": torch.from_numpy(self.chirp_lengths.astype(np.int32)),
            "meta": meta,
        }
        torch.save(payload, self.out_path)

    def _skip(self, *, reason: str | None = None) -> dict:
        data = {"file": str(self.args.fp), "skipped": True}
        if reason:
            data["reason"] = reason
        return data


# ══════════════════════════════════════════════════════════════════════════════
# main worker class
# ══════════════════════════════════════════════════════════════════════════════
class WavToSpec:
    """
    Convert a directory (or explicit list) of audio files to .npz spectrograms.
    Keys inside the .npz **match what BirdSpectrogramDataset expects**:
        s             -> (F,T)   log spectrogram
        chirp_labels  -> int32  per-frame channel dominance indicator
    """

    def __init__(self, args: SimpleNamespace) -> None:
        raw = SimpleNamespace(**vars(args))
        dst_dir = Path(getattr(raw, "dst_dir"))
        dst_dir.mkdir(parents=True, exist_ok=True)
        src_dir = getattr(raw, "src_dir", None)
        file_list = getattr(raw, "file_list", None)
        self.args = SimpleNamespace(
            src_dir=Path(src_dir) if src_dir is not None else None,
            dst_dir=dst_dir,
            file_list=Path(file_list) if file_list else None,
            step=getattr(raw, "step_size", getattr(raw, "step", 160)),
            n_fft=getattr(raw, "nfft", getattr(raw, "n_fft", 1024)),
            sr=getattr(raw, "sr", 32_000),
            min_len_ms=getattr(raw, "min_len_ms", 25),
            n_mels=getattr(raw, "n_mels", 128),
            remake=getattr(raw, "remake", False),
        )
        self._setup_logging()
        self.audio_files = self._gather_files()
        self._save_audio_params()

    # ──────────────────────────────────────────────────────────────────────
    # misc
    # ──────────────────────────────────────────────────────────────────────
    def _setup_logging(self) -> None:
        logger = logging.getLogger()
        if not logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
            logger.addHandler(console_handler)

            error_handler = logging.FileHandler("error_log.log")
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logger.addHandler(error_handler)

        logger.setLevel(logging.INFO)

    def _save_audio_params(self) -> None:
        """Save audio processing parameters to JSON file in destination directory."""
        params = {"sr": self.args.sr, "mels": self.args.n_mels, "hop_size": self.args.step, "fft": self.args.n_fft}

        params_file = self.args.dst_dir / "audio_params.json"
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)

    @staticmethod
    def _aggregate_and_print_summary(summary_rows: list[dict], dst_dir: Path) -> None:
        np.save(dst_dir / "summary.npy", np.array(summary_rows, dtype=object))

        if not summary_rows:
            print("No files processed; nothing to summarize.")
            return

        total_files = len(summary_rows)
        total_chirps = int(sum(r.get("num_chirps", 0) for r in summary_rows))
        print(f"\nSummary across {total_files} files (total chirps: {total_chirps}):")

        total_count = sum(int(r["dur"]["count"]) for r in summary_rows)
        total_sum = sum(float(r["dur"]["sum"]) for r in summary_rows)
        total_sumsq = sum(float(r["dur"]["sumsq"]) for r in summary_rows)
        mins = [float(r["dur"]["min"]) for r in summary_rows if np.isfinite(r["dur"]["min"])]
        maxs = [float(r["dur"]["max"]) for r in summary_rows if np.isfinite(r["dur"]["max"])]
        if total_count > 0:
            mu = total_sum / total_count
            var = max(0.0, total_sumsq / total_count - mu * mu)
            print(
                f"Single chirp duration (cols): mean={mu:.2f}, std={np.sqrt(var):.2f}, min={np.min(mins) if mins else float('nan'):.2f}, max={np.max(maxs) if maxs else float('nan'):.2f}, count={total_count}"
            )
        else:
            print("Single chirp duration (cols): no data")

        print("\nSliding window totals over consecutive chirps:")
        print("L\tmean(cols)\tstd(cols)\tmin\tmax\tcount")
        for L in range(1, 26):
            rows_L = [r["seq"][L] for r in summary_rows if "seq" in r and L in r["seq"]]
            if not rows_L:
                continue
            count = sum(int(x["count"]) for x in rows_L)
            if count == 0:
                continue
            total = sum(float(x["sum"]) for x in rows_L)
            total_sq = sum(float(x["sumsq"]) for x in rows_L)
            mins = [float(x["min"]) for x in rows_L if np.isfinite(x["min"])]
            maxs = [float(x["max"]) for x in rows_L if np.isfinite(x["max"])]
            mu = total / count
            var = max(0.0, total_sq / count - mu * mu)
            mn = np.min(mins) if mins else float('nan')
            mx = np.max(maxs) if maxs else float('nan')
            print(f"{L}\t{mu:.2f}\t{np.sqrt(var):.2f}\t{mn:.2f}\t{mx:.2f}\t{count}")

    def _gather_files(self) -> list[Path]:
        if self.args.file_list:
            file_list_path = self.args.file_list
            audio_exts = {".wav", ".mp3", ".ogg", ".flac"}
            suffix = file_list_path.suffix.lower()
            if suffix in audio_exts and file_list_path.exists():
                files = [file_list_path]
            else:
                try:
                    text = file_list_path.read_text()
                except UnicodeDecodeError:
                    # If we fail to decode as text, assume the path itself points to an audio file.
                    if file_list_path.exists():
                        files = [file_list_path]
                    else:
                        raise
                else:
                    files = [Path(line.strip()) for line in text.splitlines() if line.strip()]
        elif self.args.src_dir is not None:
            exts = (".wav", ".mp3", ".ogg", ".flac")
            files = [
                Path(root) / f for root, _, fs in os.walk(self.args.src_dir) for f in fs if f.lower().endswith(exts)
            ]
        else:
            files = []

        if not files:
            print("no audio files matched ‑ nothing to do.")
            return []

        return files

    # ──────────────────────────────────────────────────────────────────────
    # public entry
    # ──────────────────────────────────────────────────────────────────────
    def run(self) -> None:
        if not self.audio_files:
            return  # exit 0, no fuss

        # Set up signal handler for graceful shutdown
        original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)

        summary_rows = []
        error_count = 0
        skipped_count = 0
        pbar = tqdm(total=len(self.audio_files), desc="processing files")

        try:
            for fp in self.audio_files:
                file_args = SimpleNamespace(**vars(self.args))
                file_args.fp = Path(fp)
                processor = TwoChannelFileProcessor(file_args)
                result = processor.process()
                if not isinstance(result, dict):
                    pbar.update()
                    continue
                if result.get("error"):
                    error_count += 1
                    skipped_count += 1
                    logging.error(result["error"])
                    pbar.update()
                    continue
                if result.get("skipped"):
                    skipped_count += 1
                    pbar.update()
                    continue
                summary_rows.append(result)
                print(
                    f"MAX CHIRP: len={result.get('max_chirp_len_cols', -1)} cols, start_col={result.get('max_chirp_start_col', -1)}, file={result.get('file')}"
                )
                pbar.update()

        except KeyboardInterrupt:
            print("\nOperation interrupted by user")
            sys.exit(1)
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_sigint_handler)
            pbar.close()

        self._aggregate_and_print_summary(summary_rows, self.args.dst_dir)
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


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def cli() -> None:
    p = argparse.ArgumentParser(description="Convert audio → log‑spectrogram .npz (no JSON, no filtering).")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--src_dir", type=str, help="Root folder with wav/mp3/ogg files (searched recursively).")
    grp.add_argument("--file_list", type=str, help="Text file with absolute/relative paths, one per line.")
    p.add_argument("--dst_dir", type=str, required=True, help="Where outputs go.")

    p.add_argument("--sr", type=int, default=48_000, help="Sample rate in Hz (default: 32000).")
    p.add_argument("--step_size", type=int, default=240, help="STFT hop length (samples at target sample rate).")
    p.add_argument("--nfft", type=int, default=1024, help="FFT size.")
    p.add_argument("--n_mels", type=int, default=128, help="Number of mel bands (default: 128)")
    p.add_argument("--min_len_ms", type=int, default=25, help="Minimum clip length in milliseconds.")
    args = p.parse_args()

    converter = WavToSpec(args)
    converter.run()


if __name__ == "__main__":
    cli()
