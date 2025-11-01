# ──────────────────────────────────────────────────────────────────────────────
# audio2spec.py  ‑  simple .wav /.mp3/.ogg ➜ spectrogram (.npz / .pt) converter
# ──────────────────────────────────────────────────────────────────────────────
import os
import json
import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Any, Optional
from types import SimpleNamespace

import pickle
import time

import numpy as np
import librosa
from tqdm import tqdm

from scipy import ndimage
import scipy.signal as ss

import torch
import stumpy

torch.save({"test": torch.from_numpy(np.zeros((10, 10)))}, "./test.pt")


class SingleChannelProcessor:
    __slots__ = (
        "args",
        "S_db",
        "chirp_intervals",
        "ref",
        "mel_spectrogram",
        "mfcc",
        "features",
        "profile",
        "profile_indices",
        "joint_profile",
        "motif_sets",
        "block_lengths",
        "block_cluster_ids",
        "block_covered",
        "cluster_exemplars",
        "window_grid",
        "block_index_map",
        "block_labels",
        "stage_timings",
    )

    def __init__(self, args: SimpleNamespace) -> None:
        self.args = args
        self.S_db = np.empty((0, 0), dtype=np.float32)
        self.chirp_intervals = np.empty((0, 2), dtype=np.int32)
        self.ref = 0.0
        self.mel_spectrogram = np.empty((0, 0), dtype=np.float32)
        self.mfcc = np.empty((0, 0), dtype=np.float32)
        self.features = np.empty((0, 0), dtype=np.float32)
        self.profile: dict[int, np.ndarray] = {}
        self.profile_indices: dict[int, np.ndarray] = {}
        self.joint_profile: dict[int, np.ndarray] = {}
        self.motif_sets: list[dict[str, Any]] = []
        self.block_lengths = np.empty((0,), dtype=np.int32)
        self.block_cluster_ids = np.empty((0,), dtype=np.int32)
        self.block_covered = np.empty((0,), dtype=bool)
        self.cluster_exemplars: list[dict[str, Any]] = []
        self.window_grid: list[int] = []
        self.block_index_map = np.empty((0,), dtype=np.int32)
        self.block_labels: dict[int, int] = {}
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

        stage = time.perf_counter()
        self._prepare_features()
        self._record_stage("prepare_features", stage, extra=f"features_shape={self.features.shape}")

        stage = time.perf_counter()
        self._run_mstump()
        self._record_stage("mstump", stage, extra=f"clusters={len(self.motif_sets)}")

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

        gap_frames = max(1, int(round(merge_ms / max(frame_ms, 1e-9))))
        merged: list[tuple[int, int]] = []
        for start, end in chirp_intervals:
            if not merged:
                merged.append((start, end))
                continue
            ps, pe = merged[-1]
            if start - pe <= gap_frames:
                merged[-1] = (ps, max(pe, end))
            else:
                merged.append((start, end))

        self.chirp_intervals = np.asarray(merged, dtype=np.int32).reshape(-1, 2)
        self._initialize_block_metadata()

    def _initialize_block_metadata(self) -> None:
        intervals = np.asarray(self.chirp_intervals, dtype=np.int32).reshape(-1, 2)
        if intervals.size == 0:
            self.block_lengths = np.empty((0,), dtype=np.int32)
            self.block_cluster_ids = np.empty((0,), dtype=np.int32)
            self.block_covered = np.empty((0,), dtype=bool)
            self.block_labels = {}
            return

        lengths = np.maximum(intervals[:, 1] - intervals[:, 0], 0)
        self.block_lengths = lengths.astype(np.int32, copy=False)
        n_blocks = int(lengths.size)
        self.block_cluster_ids = np.full(n_blocks, -1, dtype=np.int32)
        self.block_covered = np.zeros(n_blocks, dtype=bool)
        self.block_labels = {int(intervals[idx, 0]): -1 for idx in range(n_blocks)}
        self.chirp_intervals = intervals

    def _prepare_features(self) -> None:
        if self.S_db.size == 0:
            self.features = np.empty((0, 0), dtype=np.float32)
            self.block_index_map = np.empty((0,), dtype=np.int32)
            self.window_grid = []
            return

        mode = getattr(self.args, "features_mode", "mel_mfcc")
        feature_blocks: list[np.ndarray] = []
        if mode != "mfcc_only":
            feature_blocks.append(self.S_db.astype(np.float32, copy=True))
        if self.mfcc.size:
            feature_blocks.append(self.mfcc.astype(np.float32, copy=False))

        if not feature_blocks:
            logging.warning(
                "Channel %s: no feature blocks available (mode=%s); defaulting to mel spectrogram",
                getattr(self.args, "channel_index", -1),
                mode,
            )
            feature_blocks = [self.S_db.astype(np.float32, copy=True)]

        features = np.vstack(feature_blocks) if feature_blocks else np.empty((0, self.S_db.shape[1]), dtype=np.float32)
        if features.size == 0:
            self.features = features
            self.block_index_map = np.empty((0,), dtype=np.int32)
            self.window_grid = []
            logging.warning("Channel %s: feature matrix is empty", getattr(self.args, "channel_index", -1))
            return

        time_bins = features.shape[1]
        mask = np.zeros(time_bins, dtype=bool)
        for start, end in self.chirp_intervals:
            if end > start:
                mask[start:end] = True

        if np.any(mask):
            in_block = features[:, mask]
            means = np.nanmean(in_block, axis=1)
            stds = np.nanstd(in_block, axis=1)
            for row_idx, (mean_val, std_val) in enumerate(zip(means, stds)):
                if not np.isfinite(std_val) or std_val < 1e-8:
                    features[row_idx, mask] = 0.0
                else:
                    features[row_idx, mask] = (features[row_idx, mask] - mean_val) / std_val
        else:
            features[:] = np.nan

        features[:, ~mask] = np.nan
        self.features = features

        if time_bins:
            self.block_index_map = np.full(time_bins, -1, dtype=np.int32)
            for block_idx, (start, end) in enumerate(self.chirp_intervals):
                start_i = int(start)
                end_i = int(end)
                if end_i > start_i:
                    self.block_index_map[start_i:end_i] = block_idx
        else:
            self.block_index_map = np.empty((0,), dtype=np.int32)

        self.window_grid = self._determine_window_grid()

    def _run_mstump(self) -> None:
        self.profile = {}
        self.profile_indices = {}
        self.joint_profile = {}
        self.motif_sets = []
        if self.block_cluster_ids.size:
            self.block_cluster_ids.fill(-1)
        if self.block_covered.size:
            self.block_covered.fill(False)
        self.cluster_exemplars = []

        if self.features.size == 0 or not np.any(np.isfinite(self.features)):
            self._finalize_block_labels()
            return
        if stumpy is None:
            self._finalize_block_labels()
            return

        if not self.window_grid:
            self.window_grid = self._determine_window_grid()
        if not self.window_grid:
            self._finalize_block_labels()
            return

        series = np.ascontiguousarray(self.features.astype(np.float64, copy=False))
        aggregated_series = np.linalg.norm(np.nan_to_num(series, nan=0.0), axis=0)
        if aggregated_series.size == 0:
            self._finalize_block_labels()
            return

        cluster_threshold = float(getattr(self.args, "cluster_threshold", 1.0))
        mass_threshold_arg = getattr(self.args, "mass_threshold", None)
        mass_threshold = float(cluster_threshold if mass_threshold_arg is None else mass_threshold_arg)
        max_seeds = int(getattr(self.args, "max_motifs", 5))

        cluster_records: dict[int, dict[str, Any]] = {}
        next_cluster_id = 0

        mode = getattr(self.args, "features_mode", "mel_mfcc")
        logging.info(
            "Channel %s: starting mstump across %d window sizes (mode=%s)",
            getattr(self.args, "channel_index", -1),
            len(self.window_grid),
            mode,
        )

        window_progress = None
        if len(self.window_grid) > 1 and getattr(self.args, "progress", True):
            window_progress = tqdm(total=len(self.window_grid), desc="mstump windows", leave=False)

        try:
            for window in self.window_grid:
                loop_start = time.perf_counter()
                clusters_before = next_cluster_id
                window_matches = 0
                seed_count = 0

                if window < 2 or series.shape[1] < window:
                    if window_progress:
                        window_progress.update(1)
                    logging.info(
                        "Channel %s: window %d skipped (series shorter than window)",
                        getattr(self.args, "channel_index", -1),
                        window,
                    )
                    continue
                candidate_starts = self._candidate_window_starts(window)
                if candidate_starts.size == 0:
                    if window_progress:
                        window_progress.update(1)
                    logging.info(
                        "Channel %s: window %d skipped (no candidate starts)",
                        getattr(self.args, "channel_index", -1),
                        window,
                    )
                    continue
                try:
                    P, I = stumpy.mstump(series, window)
                except Exception:
                    if window_progress:
                        window_progress.update(1)
                    logging.exception(
                        "Channel %s: mstump failed for window %d", getattr(self.args, "channel_index", -1), window
                    )
                    continue

                self.profile[window] = P
                self.profile_indices[window] = I
                finite_profile = np.where(np.isfinite(P), P, np.nan)
                joint_profile = np.nanmean(finite_profile, axis=0)
                self.joint_profile[window] = joint_profile

                seeds = self._select_seeds(joint_profile, candidate_starts, window, max_seeds)
                seed_count = len(seeds)
                seed_progress = None
                if seeds and getattr(self.args, "progress", True):
                    seed_progress = tqdm(total=len(seeds), desc=f"window {window}", leave=False)

                try:
                    for seed_start in seeds:
                        block_id = self._block_for_window(seed_start, window)
                        if block_id < 0 or self.block_covered[block_id]:
                            if seed_progress:
                                seed_progress.update(1)
                            continue

                        seed_features = series[:, seed_start : seed_start + window]
                        cluster_id = self._match_existing_cluster(seed_features, window, cluster_threshold)
                        if cluster_id is None:
                            cluster_id = next_cluster_id
                            next_cluster_id += 1
                            self.cluster_exemplars.append(
                                {"id": cluster_id, "window": window, "features": seed_features.copy()}
                            )

                        record = cluster_records.setdefault(
                            cluster_id,
                            {"cluster_id": cluster_id, "window": window, "exemplar_start": seed_start, "matches": []},
                        )

                        matches = self._collect_matches_with_mass(seed_start, window, mass_threshold, aggregated_series)
                        if not matches:
                            if block_id >= 0 and not self.block_covered[block_id]:
                                self.block_cluster_ids[block_id] = cluster_id
                                self.block_covered[block_id] = True
                                record["matches"].append({"start": seed_start, "block": block_id, "distance": 0.0})
                                window_matches += 1
                            if seed_progress:
                                seed_progress.update(1)
                            continue

                        for start_idx, block_idx, distance in matches:
                            if self.block_covered[block_idx]:
                                continue
                            self.block_cluster_ids[block_idx] = cluster_id
                            self.block_covered[block_idx] = True
                            record["matches"].append(
                                {"start": start_idx, "block": block_idx, "distance": float(distance)}
                            )
                            window_matches += 1
                        if seed_progress:
                            seed_progress.update(1)
                finally:
                    if seed_progress:
                        seed_progress.close()

                if window_progress:
                    window_progress.update(1)

                loop_duration = time.perf_counter() - loop_start
                logging.info(
                    "Channel %s: window %d processed in %.3fs (seeds=%d, matches=%d, new_clusters=%d)",
                    getattr(self.args, "channel_index", -1),
                    window,
                    loop_duration,
                    seed_count,
                    window_matches,
                    next_cluster_id - clusters_before,
                )
        finally:
            if window_progress:
                window_progress.close()

        assigned = int(np.sum(self.block_cluster_ids >= 0)) if self.block_cluster_ids.size else 0
        unassigned = int(np.sum(self.block_cluster_ids < 0)) if self.block_cluster_ids.size else 0
        logging.info(
            "Channel %s: mstump completed (clusters=%d, assigned_blocks=%d, noise_blocks=%d)",
            getattr(self.args, "channel_index", -1),
            next_cluster_id,
            assigned,
            unassigned,
        )

        self.motif_sets = [
            {**record, "matches": sorted(record["matches"], key=lambda item: (item["block"], item["start"]))}
            for record in sorted(cluster_records.values(), key=lambda r: r["cluster_id"])
        ]
        self._finalize_block_labels()

    def _determine_window_grid(self) -> list[int]:
        if self.features.size == 0:
            return []
        valid_lengths = self.block_lengths[self.block_lengths >= 2]
        if valid_lengths.size == 0:
            default_window = int(getattr(self.args, "mstump_window", 0))
            if 2 <= default_window <= self.features.shape[1]:
                return [default_window]
            return []

        quantiles = np.quantile(valid_lengths.astype(np.float64), [0.9, 0.75, 0.5, 0.25])
        candidate = {int(round(q)) for q in quantiles if np.isfinite(q) and q >= 2}
        candidate.add(int(valid_lengths.max()))
        default_window = int(getattr(self.args, "mstump_window", 0))
        if default_window >= 2:
            candidate.add(default_window)

        min_window = int(getattr(self.args, "mstump_min_window", 2))
        candidate = {int(max(min_window, value)) for value in candidate}
        windows = sorted(
            {value for value in candidate if value <= self.features.shape[1] and np.any(valid_lengths >= value)},
            reverse=True,
        )
        return windows

    def _candidate_window_starts(self, window: int) -> np.ndarray:
        if self.chirp_intervals.size == 0:
            return np.empty((0,), dtype=np.int32)

        candidates: list[np.ndarray] = []
        for block_idx, (start, end) in enumerate(self.chirp_intervals):
            if self.block_covered.size and self.block_covered[block_idx]:
                continue
            length = int(end - start)
            if length < window:
                continue
            stop = int(end - window + 1)
            if stop <= start:
                continue
            candidates.append(np.arange(int(start), stop, dtype=np.int32))

        if not candidates:
            return np.empty((0,), dtype=np.int32)
        return np.concatenate(candidates)

    def _select_seeds(
        self, joint_profile: np.ndarray, candidates: np.ndarray, window: int, max_seeds: int
    ) -> list[int]:
        if joint_profile.size == 0 or candidates.size == 0:
            return []

        max_index = joint_profile.shape[0]
        ranked: list[tuple[float, int]] = []
        for start in candidates:
            if start < 0 or start >= max_index:
                continue
            value = joint_profile[start]
            if not np.isfinite(value):
                continue
            ranked.append((float(value), int(start)))

        ranked.sort(key=lambda item: item[0])
        seeds: list[int] = []
        for _, start in ranked:
            block_id = self._block_for_window(start, window)
            if block_id < 0 or self.block_covered[block_id]:
                continue
            seeds.append(start)
            if len(seeds) >= max_seeds:
                break
        return seeds

    def _block_for_window(self, start: int, window: int) -> int:
        if self.block_index_map.size == 0 or start < 0 or (start + window) > self.block_index_map.size:
            return -1
        block_id = int(self.block_index_map[start])
        if block_id < 0:
            return -1
        if not self._window_within_block(start, window, block_id):
            return -1
        return block_id

    def _window_within_block(self, start: int, window: int, block_id: int) -> bool:
        if block_id < 0 or block_id >= self.chirp_intervals.shape[0]:
            return False
        block_start = int(self.chirp_intervals[block_id, 0])
        block_end = int(self.chirp_intervals[block_id, 1])
        return block_start <= start and (start + window) <= block_end

    def _match_existing_cluster(self, seed_features: np.ndarray, window: int, threshold: float) -> int | None:
        for cluster in self.cluster_exemplars:
            if cluster["window"] != window:
                continue
            dist = self._subsequence_distance(seed_features, cluster["features"])
            if dist <= threshold:
                cluster["features"] = 0.5 * (cluster["features"] + seed_features)
                return int(cluster["id"])
        return None

    @staticmethod
    def _subsequence_distance(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            return float("inf")
        diff = a - b
        return float(np.sqrt(np.mean(diff * diff)))

    def _collect_matches_with_mass(
        self, seed_start: int, window: int, threshold: float, aggregated_series: np.ndarray
    ) -> list[tuple[int, int, float]]:
        if aggregated_series.size < window or seed_start + window > aggregated_series.size:
            return []

        query = aggregated_series[seed_start : seed_start + window]
        if query.size < window:
            return []

        try:
            profile = stumpy.mass(query, aggregated_series)
        except Exception:
            return []

        order = np.argsort(profile)
        matches: list[tuple[int, int, float]] = []
        seen_blocks: set[int] = set()
        for idx in order:
            dist = profile[idx]
            if not np.isfinite(dist) or dist > threshold:
                break
            block_id = self._block_for_window(int(idx), window)
            if block_id < 0 or block_id in seen_blocks or self.block_covered[block_id]:
                continue
            matches.append((int(idx), block_id, float(dist)))
            seen_blocks.add(block_id)
        return matches

    def _finalize_block_labels(self) -> None:
        if self.block_cluster_ids.size:
            noise_label = int(getattr(self.args, "noise_label", -1))
            for idx in range(self.block_cluster_ids.size):
                if self.block_cluster_ids[idx] < 0:
                    self.block_cluster_ids[idx] = noise_label

        if self.chirp_intervals.size == 0:
            self.block_labels = {}
            return

        labels: dict[int, int] = {}
        for idx, interval in enumerate(self.chirp_intervals):
            start = int(interval[0])
            cluster_id = int(self.block_cluster_ids[idx]) if idx < self.block_cluster_ids.size else -1
            labels[start] = cluster_id
        self.block_labels = labels

    def to_pickle_payload(self) -> dict[str, Any]:
        meta = {
            "sr": getattr(self.args, "sr", None),
            "hop_length": getattr(self.args, "hop_length", None),
            "n_fft": getattr(self.args, "n_fft", None),
            "n_mels": getattr(self.args, "n_mels", None),
            "n_mfcc": getattr(self.args, "n_mfcc", None),
        }

        profile = {int(window): np.array(values, copy=True) for window, values in self.profile.items()}
        profile_indices = {int(window): np.array(values, copy=True) for window, values in self.profile_indices.items()}
        joint_profile = {int(window): np.array(values, copy=True) for window, values in self.joint_profile.items()}

        exemplars = [
            {
                "id": int(exemplar["id"]),
                "window": int(exemplar["window"]),
                "features": np.array(exemplar["features"], copy=True),
            }
            for exemplar in self.cluster_exemplars
        ]

        payload = {
            "meta": meta,
            "chirp_intervals": np.array(self.chirp_intervals, copy=True),
            "block_lengths": np.array(self.block_lengths, copy=True),
            "block_cluster_ids": np.array(self.block_cluster_ids, copy=True),
            "block_labels": dict(self.block_labels),
            "motif_sets": list(self.motif_sets),
            "window_grid": list(self.window_grid),
            "cluster_exemplars": exemplars,
            "profile": profile,
            "profile_indices": profile_indices,
            "joint_profile": joint_profile,
            "features": np.array(self.features, copy=True),
            "mel_spectrogram": np.array(self.mel_spectrogram, copy=True),
            "S_db": np.array(self.S_db, copy=True),
        }
        return payload


class TwoChannelFileProcessor:
    desired_channels = 2

    def __init__(self, args: SimpleNamespace) -> None:
        base = SimpleNamespace(**vars(args))
        base.fp = Path(base.fp)
        base.dst_dir = Path(base.dst_dir)
        base.dst_dir.mkdir(parents=True, exist_ok=True)
        base.hop_length = getattr(base, "hop_length", base.step)
        base.s_ref = getattr(base, "s_ref", None)
        base.features_mode = getattr(base, "features_mode", "mel_mfcc")
        base.progress = getattr(base, "progress", True)
        self.args = base
        self.out_path = base.dst_dir / (base.fp.stem + ".pt")
        self.actual_sr: int = base.sr
        self.channel_processors: list[SingleChannelProcessor] = []
        self.chirp_intervals = np.empty((0, 2), dtype=np.int32)
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
            logging.info(
                "File %s configuration: features_mode=%s, cluster_threshold=%.3f, mass_threshold=%s",
                self.args.fp.name,
                self.args.features_mode,
                float(self.args.cluster_threshold),
                "auto" if self.args.mass_threshold is None else f"{float(self.args.mass_threshold):.3f}",
            )
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

            self.chirp_intervals = self._merge_chirps(proc.chirp_intervals for proc in self.channel_processors)
            if self.S_stack.shape[-1] < self.args.min_timebins:
                return self._skip()

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
                    features_mode=self.args.features_mode,
                    progress=self.args.progress,
                    channel_index=idx,
                    file_stem=self.args.fp.stem,
                    cluster_threshold=getattr(self.args, "cluster_threshold", 1.0),
                    mass_threshold=getattr(self.args, "mass_threshold", None),
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
        ci = np.asarray(self.chirp_intervals, dtype=np.int64).reshape(-1, 2)
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
        torch.save(
            {"s": torch.from_numpy(self.S_stack), "chirp_intervals": torch.from_numpy(self.chirp_intervals)},
            self.out_path,
        )
        cluster_dir = self.args.dst_dir / "cluster"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        for idx, processor in enumerate(self.channel_processors):
            payload = processor.to_pickle_payload()
            payload_meta = payload.get("meta", {})
            payload_meta.update(
                {"channel_index": idx, "file_stem": self.args.fp.stem, "source_path": str(self.args.fp)}
            )
            payload["meta"] = payload_meta
            cluster_path = cluster_dir / f"{self.args.fp.stem}_ch{idx}.pkl"
            with cluster_path.open("wb") as fh:
                pickle.dump(payload, fh)

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
            min_timebins=getattr(raw, "min_timebins", 25),
            n_mels=getattr(raw, "n_mels", 128),
            remake=getattr(raw, "remake", False),
            features_mode=getattr(raw, "features_mode", "mel_mfcc"),
            progress=not getattr(raw, "no_progress", False),
            cluster_threshold=getattr(raw, "cluster_threshold", 1.0),
            mass_threshold=getattr(raw, "mass_threshold", None),
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

    p.add_argument("--sr", type=int, default=32_000, help="Sample rate in Hz (default: 32000).")
    p.add_argument("--step_size", type=int, default=320, help="STFT hop length (samples at target sample rate).")
    p.add_argument("--nfft", type=int, default=1024, help="FFT size.")
    p.add_argument("--n_mels", type=int, default=128, help="Number of mel bands (default: 128)")
    p.add_argument("--min_len_ms", type=int, default=25, help="Minimum clip length in milliseconds.")
    p.add_argument("--min_timebins", type=int, default=25, help="Minimum number of spectrogram time bins.")
    p.add_argument(
        "--features_mode",
        type=str,
        choices=["mel_mfcc", "mfcc_only"],
        default="mel_mfcc",
        help="Feature stack to use for clustering (default: mel_mfcc).",
    )
    p.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars during clustering.")
    p.add_argument(
        "--cluster_threshold",
        type=float,
        default=1.0,
        help="Distance threshold for assigning motifs to existing clusters (default: 1.0)",
    )
    p.add_argument(
        "--mass_threshold",
        type=float,
        default=None,
        help="Override MASS distance threshold (defaults to cluster_threshold).",
    )
    args = p.parse_args()

    converter = WavToSpec(args)
    converter.run()


if __name__ == "__main__":
    cli()
