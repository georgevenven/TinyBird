#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

import extract_embedding  # noqa: E402


def _resolve_run_dir(run_arg: str) -> Path:
    run_path = Path(run_arg)
    if run_path.is_absolute() and run_path.is_dir():
        return run_path
    project_relative = ROOT / run_path
    if project_relative.is_dir():
        return project_relative.resolve()
    runs_relative = ROOT / "runs" / run_path
    if runs_relative.is_dir():
        return runs_relative.resolve()
    raise SystemExit(f"Unable to resolve run_dir: {run_arg}")


def _sanitize_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def _parse_csv_ints(token: str) -> list[int]:
    vals = []
    for part in token.split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(int(p))
    vals = sorted(set(v for v in vals if v > 0))
    if not vals:
        raise SystemExit("No valid positive pooling/window integers were provided.")
    return vals


def _sanitize_embedding_key(key: str) -> str:
    return _sanitize_token(key).replace("_embeddings", "")


def _load_recording_stems_by_bird(annotation_json: Path) -> dict[str, list[str]]:
    data = json.loads(annotation_json.read_text(encoding="utf-8"))
    by_bird: dict[str, set[str]] = {}
    for rec in data.get("recordings", []):
        rec_info = rec.get("recording", {})
        bird_id = str(rec_info.get("bird_id", "")).strip()
        filename = str(rec_info.get("filename", "")).strip()
        stem = Path(filename).stem
        if not bird_id or not stem:
            continue
        by_bird.setdefault(bird_id, set()).add(stem)
    return {bird: sorted(stems) for bird, stems in by_bird.items()}


def _pick_recordings(stems: list[str], songs_per_bird: int, seed: int, bird_id: str) -> list[str]:
    if len(stems) < songs_per_bird:
        return []
    bird_hash = int(hashlib.sha1(bird_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed + bird_hash)
    idx = rng.choice(len(stems), size=songs_per_bird, replace=False)
    idx.sort()
    return [stems[i] for i in idx]


def _find_spec_file(spec_dir: Path, stem: str) -> Path | None:
    direct = spec_dir / f"{stem}.npy"
    if direct.exists():
        return direct
    matches = sorted(spec_dir.glob(f"{stem}.*"))
    for match in matches:
        if match.suffix == ".npy":
            return match
    return None


def _materialize_subset(
    spec_dir: Path,
    stems: list[str],
    subset_dir: Path,
    copy_mode: str,
) -> tuple[list[str], list[str]]:
    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    subset_dir.mkdir(parents=True, exist_ok=True)

    kept_stems: list[str] = []
    missing_stems: list[str] = []
    for stem in stems:
        src = _find_spec_file(spec_dir, stem)
        if src is None:
            missing_stems.append(stem)
            continue
        dst = subset_dir / src.name
        if copy_mode == "copy":
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src.resolve())
        kept_stems.append(stem)

    audio_params_src = spec_dir / "audio_params.json"
    if audio_params_src.exists():
        shutil.copy2(audio_params_src, subset_dir / "audio_params.json")
    else:
        raise SystemExit(f"Missing audio_params.json in spec_dir: {spec_dir}")

    return kept_stems, missing_stems


def _run_extract(
    run_dir: Path,
    checkpoint: str | None,
    spec_subset_dir: Path,
    annotation_json: Path,
    bird_id: str,
    npz_path: Path,
    num_timebins: int,
    encoder_layer_idx: int | None,
):
    args = {
        "num_timebins": int(num_timebins),
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "spec_dir": str(spec_subset_dir),
        "npz_dir": str(npz_path),
        "json_path": str(annotation_json),
        "bird": bird_id,
        "encoder_layer_idx": encoder_layer_idx,
    }
    extract_embedding.main(args)


def _pool_embeddings(emb: np.ndarray, window: int, mode: str, hop: int | None = None) -> np.ndarray:
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape={emb.shape}")
    if emb.shape[0] == 0:
        return emb.astype(np.float32, copy=False)
    if window <= 1:
        return emb.astype(np.float32, copy=False)
    if hop is None:
        hop = window
    hop = max(1, int(hop))

    out = []
    for start in range(0, emb.shape[0] - window + 1, hop):
        chunk = emb[start : start + window]
        if chunk.size == 0:
            continue
        if mode == "max":
            pooled = chunk.max(axis=0)
        elif mode == "sum":
            pooled = chunk.sum(axis=0)
        else:
            pooled = chunk.mean(axis=0)
        out.append(pooled.astype(np.float32, copy=False))
    if not out and emb.shape[0] > 0:
        # If sequence is shorter than window, pool once over the full available sequence.
        chunk = emb
        if mode == "max":
            pooled = chunk.max(axis=0)
        elif mode == "sum":
            pooled = chunk.sum(axis=0)
        else:
            pooled = chunk.mean(axis=0)
        out.append(pooled.astype(np.float32, copy=False))
    if not out:
        return np.zeros((0, emb.shape[1]), dtype=np.float32)
    return np.vstack(out)


def _window_spectrogram(
    spec: np.ndarray,
    window_bins: int,
    hop_bins: int,
    feature_mode: str = "flatten",
) -> np.ndarray:
    if spec.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram, got shape={spec.shape}")
    if feature_mode == "flatten":
        out_dim = spec.shape[1] * window_bins
    elif feature_mode == "mean_freq":
        out_dim = window_bins
    else:
        raise ValueError(f"Unknown spectrogram feature_mode={feature_mode}")
    if spec.shape[0] == 0:
        return np.zeros((0, out_dim), dtype=np.float32)

    def _reduce_chunk(chunk: np.ndarray) -> np.ndarray:
        if feature_mode == "flatten":
            return chunk.reshape(-1).astype(np.float32, copy=False)
        return chunk.mean(axis=1).astype(np.float32, copy=False)

    if spec.shape[0] < window_bins:
        pad = np.zeros((window_bins - spec.shape[0], spec.shape[1]), dtype=spec.dtype)
        chunk = np.vstack([spec, pad])
        return _reduce_chunk(chunk)[None, :]

    out = []
    for start in range(0, spec.shape[0] - window_bins + 1, hop_bins):
        chunk = spec[start : start + window_bins]
        out.append(_reduce_chunk(chunk))
    if not out:
        return np.zeros((0, out_dim), dtype=np.float32)
    return np.vstack(out).astype(np.float32, copy=False)


def _sample_indices(n_rows: int, max_rows: int, seed: int, key: str) -> np.ndarray:
    if n_rows <= 0:
        return np.asarray([], dtype=np.int64)
    if max_rows <= 0 or n_rows <= max_rows:
        return np.arange(n_rows, dtype=np.int64)
    key_hash = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed + key_hash)
    idx = rng.choice(n_rows, size=max_rows, replace=False)
    idx.sort()
    return np.asarray(idx, dtype=np.int64)


def _subsample_rows(arr: np.ndarray, max_rows: int, seed: int, key: str) -> np.ndarray:
    idx = _sample_indices(arr.shape[0], max_rows=max_rows, seed=seed, key=key)
    if idx.size == arr.shape[0]:
        return arr
    return arr[idx]


def _pool_labels(labels: np.ndarray, window: int, hop: int | None = None) -> np.ndarray:
    """
    Pool per-frame labels over windows using majority vote.
    """
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape={labels.shape}")
    if labels.shape[0] == 0:
        return labels.astype(np.int64, copy=False)
    if window <= 1:
        return labels.astype(np.int64, copy=False)
    if hop is None:
        hop = window
    hop = max(1, int(hop))

    pooled = []
    for start in range(0, labels.shape[0] - window + 1, hop):
        chunk = labels[start : start + window]
        if chunk.size == 0:
            continue
        values, counts = np.unique(chunk, return_counts=True)
        pooled.append(int(values[np.argmax(counts)]))

    if not pooled and labels.shape[0] > 0:
        values, counts = np.unique(labels, return_counts=True)
        pooled.append(int(values[np.argmax(counts)]))

    return np.asarray(pooled, dtype=np.int64)


def _fit_umap(x: np.ndarray, n_neighbors: int, min_dist: float, metric: str, deterministic: bool, seed: int) -> np.ndarray:
    kwargs = {
        "n_components": 2,
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "metric": metric,
    }
    if deterministic:
        kwargs["random_state"] = int(seed)
    else:
        kwargs["low_memory"] = True
        kwargs["n_jobs"] = -1
    reducer = umap.UMAP(**kwargs)
    return reducer.fit_transform(x)


def _bird_palette(birds: list[str]) -> dict[str, np.ndarray]:
    uniq = sorted(set(birds))
    if not uniq:
        return {}
    cmap = plt.get_cmap("tab20", len(uniq))
    out = {}
    for idx, bird in enumerate(uniq):
        color = np.asarray(cmap(idx), dtype=np.float32)
        if color.shape[0] > 3:
            color = color[:3]
        out[bird] = color
    return out


def _scatter_umap(xy: np.ndarray, labels: np.ndarray, title: str, out_base: Path):
    birds = sorted(set(labels.tolist()))
    palette = _bird_palette(birds)

    fig = plt.figure(figsize=(9.5, 7.5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)

    for bird in birds:
        idx = labels == bird
        if not np.any(idx):
            continue
        ax.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=10,
            alpha=0.1,
            color=palette[bird],
            label=bird,
            edgecolors="none",
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
        markerscale=1.6,
        ncol=1,
    )
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def _scatter_umap_syllables(
    xy: np.ndarray,
    labels: np.ndarray,
    bird_labels: np.ndarray,
    title: str,
    out_base: Path,
):
    if labels.ndim != 1 or labels.shape[0] != xy.shape[0]:
        raise ValueError("Syllable labels must be 1D and match UMAP point count.")
    if bird_labels.ndim != 1 or bird_labels.shape[0] != xy.shape[0]:
        raise ValueError("Bird labels must be 1D and match UMAP point count.")

    # Bird-specific syllable identity: same integer across birds is a different class.
    categories = []
    for bird, syl in zip(bird_labels.tolist(), labels.tolist()):
        sid = int(syl)
        if sid < 0:
            categories.append("silence")
        else:
            categories.append(f"{bird}:{sid}")

    uniq = sorted(set(categories))
    non_silence = [u for u in uniq if u != "silence"]
    palette = {}
    if non_silence:
        cmap = plt.get_cmap("gist_ncar", max(1, len(non_silence)))
        for idx, cat in enumerate(non_silence):
            color = np.asarray(cmap(idx), dtype=np.float32)
            if color.shape[0] > 3:
                color = color[:3]
            palette[cat] = color
    if "silence" in uniq:
        palette["silence"] = np.asarray([0.55, 0.55, 0.55], dtype=np.float32)

    fig = plt.figure(figsize=(9.5, 7.5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    categories_arr = np.asarray(categories, dtype=object)

    for cat in uniq:
        idx = categories_arr == cat
        if not np.any(idx):
            continue
        alpha = 0.1
        ax.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=10,
            alpha=alpha,
            color=palette[cat],
            label=cat,
            edgecolors="none",
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    # Always show syllable legend. Use multiple columns when many labels exist.
    legend_cols = 1 if len(uniq) <= 30 else 2
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=7,
        markerscale=1.6,
        ncol=legend_cols,
        title="Syllable",
    )
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def _build_songmae_representation(
    per_bird_emb: dict[str, np.ndarray],
    per_bird_lbl: dict[str, np.ndarray],
    pooling_window: int,
    pooling_mode: str,
    pooling_hop: int,
    max_points_per_bird: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_parts = []
    y_parts = []
    s_parts = []
    for bird_id, emb in sorted(per_bird_emb.items()):
        lbl = per_bird_lbl.get(bird_id)
        if lbl is None or lbl.ndim != 1:
            lbl = np.full((emb.shape[0],), fill_value=-1, dtype=np.int64)
        n = min(int(emb.shape[0]), int(lbl.shape[0]))
        if n <= 0:
            continue
        emb = emb[:n]
        lbl = lbl[:n]
        pooled = _pool_embeddings(emb, pooling_window, pooling_mode, hop=pooling_hop)
        pooled_lbl = _pool_labels(lbl, window=pooling_window, hop=pooling_hop)
        m = min(int(pooled.shape[0]), int(pooled_lbl.shape[0]))
        if m <= 0:
            continue
        pooled = pooled[:m]
        pooled_lbl = pooled_lbl[:m]
        idx = _sample_indices(
            pooled.shape[0],
            max_rows=max_points_per_bird,
            seed=seed,
            key=f"songmae::{bird_id}::w{pooling_window}::h{pooling_hop}::{pooling_mode}",
        )
        if idx.size == 0:
            continue
        pooled = pooled[idx]
        pooled_lbl = pooled_lbl[idx]
        if pooled.shape[0] == 0:
            continue
        x_parts.append(pooled)
        y_parts.extend([bird_id] * pooled.shape[0])
        s_parts.append(pooled_lbl)
    if not x_parts:
        return (
            np.zeros((0, 1), dtype=np.float32),
            np.asarray([], dtype=object),
            np.asarray([], dtype=np.int64),
        )
    return (
        np.vstack(x_parts),
        np.asarray(y_parts, dtype=object),
        np.concatenate(s_parts, axis=0),
    )


def _build_spectrogram_representation(
    per_bird_spec: dict[str, np.ndarray],
    per_bird_lbl: dict[str, np.ndarray],
    window_bins: int,
    hop_bins: int,
    feature_mode: str,
    max_points_per_bird: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_parts = []
    y_parts = []
    s_parts = []
    for bird_id, spec in sorted(per_bird_spec.items()):
        lbl = per_bird_lbl.get(bird_id)
        if lbl is None or lbl.ndim != 1:
            lbl = np.full((spec.shape[0],), fill_value=-1, dtype=np.int64)
        n = min(int(spec.shape[0]), int(lbl.shape[0]))
        if n <= 0:
            continue
        spec = spec[:n]
        lbl = lbl[:n]
        windowed = _window_spectrogram(
            spec,
            window_bins=window_bins,
            hop_bins=hop_bins,
            feature_mode=feature_mode,
        )
        windowed_lbl = _pool_labels(lbl, window=window_bins, hop=hop_bins)
        m = min(int(windowed.shape[0]), int(windowed_lbl.shape[0]))
        if m <= 0:
            continue
        windowed = windowed[:m]
        windowed_lbl = windowed_lbl[:m]
        idx = _sample_indices(
            windowed.shape[0],
            max_rows=max_points_per_bird,
            seed=seed,
            key=f"spec::{bird_id}::w{window_bins}::h{hop_bins}",
        )
        if idx.size == 0:
            continue
        windowed = windowed[idx]
        windowed_lbl = windowed_lbl[idx]
        if windowed.shape[0] == 0:
            continue
        x_parts.append(windowed)
        y_parts.extend([bird_id] * windowed.shape[0])
        s_parts.append(windowed_lbl)
    if not x_parts:
        return (
            np.zeros((0, 1), dtype=np.float32),
            np.asarray([], dtype=object),
            np.asarray([], dtype=np.int64),
        )
    return np.vstack(x_parts), np.asarray(y_parts, dtype=object), np.concatenate(s_parts, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-individual UMAP comparison: SongMAE pooling variants vs spectrogram window baseline."
    )
    parser.add_argument("--annotation_json", required=True, help="Annotation JSON containing recording.bird_id and filename.")
    parser.add_argument("--spec_dir", required=True, help="Directory with spectrogram .npy files.")
    parser.add_argument("--run_dir", required=True, help="SongMAE run directory, run name, or project-relative run path.")
    parser.add_argument("--out_dir", required=True, help="Output directory for per-bird NPZs, UMAPs, and metadata.")
    parser.add_argument("--species", default="Zebra_Finch", help="Species label for metadata only.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint filename.")
    parser.add_argument("--encoder_layer_idx", type=int, default=None, help="Optional encoder layer index for extraction.")
    parser.add_argument("--songs_per_bird", type=int, default=5, help="How many recordings to sample per bird.")
    parser.add_argument("--max_birds", type=int, default=0, help="Optional cap on number of birds (0 means all).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    parser.add_argument(
        "--num_timebins",
        type=int,
        default=5000000,
        help="Max timebins passed to extract_embedding for each bird subset.",
    )
    parser.add_argument(
        "--pool_windows",
        default="1,8,32,128",
        help="Comma-separated pooling window sizes in patch-time space.",
    )
    parser.add_argument(
        "--pool_mode",
        default="mean",
        choices=["mean", "max", "sum"],
        help="Pooling mode for SongMAE embeddings.",
    )
    parser.add_argument(
        "--pool_hop_ratio",
        type=float,
        default=0.5,
        help="SongMAE pooling hop as a ratio of window size (default 0.5 = half-window stride).",
    )
    parser.add_argument("--spec_window_bins", type=int, default=32, help="Raw spectrogram window size in timebins.")
    parser.add_argument("--spec_window_hop_bins", type=int, default=16, help="Raw spectrogram window hop in timebins.")
    parser.add_argument(
        "--spec_scale_by_patch_width",
        action="store_true",
        help="Scale spectrogram window/hop by the SongMAE patch width stored in the extracted NPZs.",
    )
    parser.add_argument(
        "--spec_feature_mode",
        default="flatten",
        choices=["flatten", "mean_freq"],
        help="How to convert each spectrogram window into a feature vector.",
    )
    parser.add_argument(
        "--songmae_embedding_key",
        default="encoded_embeddings_after_pos_removal",
        choices=[
            "encoded_embeddings_before_pos_removal",
            "encoded_embeddings_after_pos_removal",
            "patch_embeddings_before_pos_removal",
            "patch_embeddings_after_pos_removal",
        ],
        help="Which NPZ embedding tensor to pool for the SongMAE representation.",
    )
    parser.add_argument(
        "--no_spec_baseline",
        action="store_true",
        help="Skip spectrogram-window baseline UMAP generation.",
    )
    parser.add_argument(
        "--max_points_per_bird",
        type=int,
        default=1500,
        help="Max vectors per bird per representation before UMAP.",
    )
    parser.add_argument("--umap_neighbors", type=int, default=100, help="UMAP n_neighbors.")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--umap_metric", default="cosine", help="UMAP metric.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic UMAP random_state.")
    parser.add_argument(
        "--syllable_plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate secondary UMAPs colored by syllable id when labels are available.",
    )
    parser.add_argument("--copy_mode", default="symlink", choices=["symlink", "copy"], help="How to materialize per-bird subset specs.")
    parser.add_argument("--force_reextract", action="store_true", help="Re-run per-bird extraction even when NPZ exists.")
    args = parser.parse_args()

    annotation_json = Path(args.annotation_json).resolve()
    spec_dir = Path(args.spec_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    run_dir = _resolve_run_dir(args.run_dir)

    if not annotation_json.exists():
        raise SystemExit(f"annotation_json not found: {annotation_json}")
    if not spec_dir.is_dir():
        raise SystemExit(f"spec_dir not found: {spec_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "tmp_subsets"
    run_tag = _sanitize_token(run_dir.name or "run")
    if args.checkpoint:
        run_tag = f"{run_tag}__{_sanitize_token(Path(args.checkpoint).stem)}"
    per_bird_dir = out_dir / "per_bird" / run_tag
    umap_dir = out_dir / "umap"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    per_bird_dir.mkdir(parents=True, exist_ok=True)
    umap_dir.mkdir(parents=True, exist_ok=True)

    pool_windows = _parse_csv_ints(args.pool_windows)
    if args.pool_hop_ratio <= 0:
        raise SystemExit("--pool_hop_ratio must be > 0")

    stems_by_bird = _load_recording_stems_by_bird(annotation_json)
    bird_ids = sorted(stems_by_bird.keys())
    if args.max_birds > 0:
        bird_ids = bird_ids[: args.max_birds]

    sampled_recordings = {}
    skipped_birds = {}
    for bird_id in bird_ids:
        picks = _pick_recordings(stems_by_bird[bird_id], songs_per_bird=args.songs_per_bird, seed=args.seed, bird_id=bird_id)
        if not picks:
            skipped_birds[bird_id] = f"fewer than songs_per_bird={args.songs_per_bird}"
            continue
        sampled_recordings[bird_id] = picks

    if not sampled_recordings:
        raise SystemExit("No birds have enough recordings after filtering.")

    per_bird_emb: dict[str, np.ndarray] = {}
    per_bird_patch_lbl: dict[str, np.ndarray] = {}
    per_bird_spec: dict[str, np.ndarray] = {}
    per_bird_spec_lbl: dict[str, np.ndarray] = {}
    per_bird_patch_width: dict[str, int] = {}
    extraction_meta = {}

    for bird_id in sorted(sampled_recordings.keys()):
        subset_dir = tmp_dir / bird_id / "spec"
        npz_path = per_bird_dir / f"{bird_id}.npz"

        kept, missing = _materialize_subset(
            spec_dir=spec_dir,
            stems=sampled_recordings[bird_id],
            subset_dir=subset_dir,
            copy_mode=args.copy_mode,
        )

        if len(kept) == 0:
            skipped_birds[bird_id] = "no selected stems were found in spec_dir"
            continue

        extraction_meta[bird_id] = {
            "selected_recording_stems": sampled_recordings[bird_id],
            "kept_recording_stems": kept,
            "missing_recording_stems": missing,
            "npz_path": str(npz_path),
        }

        if args.force_reextract or (not npz_path.exists()):
            print(f"[extract] bird={bird_id} files={len(kept)} -> {npz_path}")
            _run_extract(
                run_dir=run_dir,
                checkpoint=args.checkpoint,
                spec_subset_dir=subset_dir,
                annotation_json=annotation_json,
                bird_id=bird_id,
                npz_path=npz_path,
                num_timebins=args.num_timebins,
                encoder_layer_idx=args.encoder_layer_idx,
            )
        else:
            print(f"[reuse] bird={bird_id} npz={npz_path}")

        with np.load(npz_path, allow_pickle=True) as npz:
            if args.songmae_embedding_key not in npz:
                raise SystemExit(
                    f"Embedding key '{args.songmae_embedding_key}' not found in {npz_path}"
                )
            emb = np.asarray(npz[args.songmae_embedding_key], dtype=np.float32)
            spec = np.asarray(npz["spectrograms"], dtype=np.float32)
            if "patch_width" in npz:
                patch_width = int(np.asarray(npz["patch_width"]).reshape(-1)[0])
            else:
                patch_width = 1
            if "labels_downsampled" in npz:
                patch_lbl = np.asarray(npz["labels_downsampled"], dtype=np.int64)
            else:
                patch_lbl = np.full((emb.shape[0],), fill_value=-1, dtype=np.int64)
            if "labels_original" in npz:
                spec_lbl = np.asarray(npz["labels_original"], dtype=np.int64)
            elif patch_lbl.shape[0] == spec.shape[0]:
                spec_lbl = patch_lbl.copy()
            else:
                spec_lbl = np.full((spec.shape[0],), fill_value=-1, dtype=np.int64)
        if emb.ndim != 2 or emb.shape[0] == 0:
            skipped_birds[bird_id] = f"empty or invalid {args.songmae_embedding_key}"
            continue
        if spec.ndim != 2 or spec.shape[0] == 0:
            skipped_birds[bird_id] = "empty or invalid spectrograms"
            continue
        per_bird_emb[bird_id] = emb
        per_bird_patch_lbl[bird_id] = patch_lbl
        per_bird_spec[bird_id] = spec
        per_bird_spec_lbl[bird_id] = spec_lbl
        per_bird_patch_width[bird_id] = patch_width

    for bird_id in list(skipped_birds.keys()):
        if bird_id in per_bird_emb:
            del per_bird_emb[bird_id]
        if bird_id in per_bird_patch_lbl:
            del per_bird_patch_lbl[bird_id]
        if bird_id in per_bird_spec:
            del per_bird_spec[bird_id]
        if bird_id in per_bird_spec_lbl:
            del per_bird_spec_lbl[bird_id]
        if bird_id in per_bird_patch_width:
            del per_bird_patch_width[bird_id]

    if not per_bird_emb or not per_bird_spec:
        raise SystemExit("No valid per-bird embeddings/spectrograms available after extraction.")

    spec_patch_width_scale = 1
    if not args.no_spec_baseline and args.spec_scale_by_patch_width:
        patch_widths = sorted({int(per_bird_patch_width[bird_id]) for bird_id in per_bird_spec.keys()})
        if not patch_widths:
            raise SystemExit("Unable to determine patch_width for spectrogram scaling.")
        if len(patch_widths) != 1:
            raise SystemExit(
                f"Expected one patch_width for spectrogram scaling, found: {patch_widths}"
            )
        spec_patch_width_scale = patch_widths[0]

    effective_spec_window_bins = int(args.spec_window_bins * spec_patch_width_scale)
    effective_spec_hop_bins = int(args.spec_window_hop_bins * spec_patch_width_scale)

    generated = {}
    songmae_embedding_tag = _sanitize_embedding_key(args.songmae_embedding_key)

    for window in pool_windows:
        pooling_hop = max(1, int(round(window * args.pool_hop_ratio)))
        rep_name = f"songmae_{songmae_embedding_tag}_pool_{args.pool_mode}_w{window}_h{pooling_hop}"
        x, y, s = _build_songmae_representation(
            per_bird_emb=per_bird_emb,
            per_bird_lbl=per_bird_patch_lbl,
            pooling_window=window,
            pooling_mode=args.pool_mode,
            pooling_hop=pooling_hop,
            max_points_per_bird=args.max_points_per_bird,
            seed=args.seed,
        )
        if x.shape[0] < 2:
            print(f"[skip] {rep_name}: not enough points ({x.shape[0]})")
            continue
        print(f"[umap] {rep_name}: points={x.shape[0]} dim={x.shape[1]}")
        xy = _fit_umap(
            x=x,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            deterministic=args.deterministic,
            seed=args.seed,
        )
        out_base = umap_dir / rep_name
        _scatter_umap(xy=xy, labels=y, title=f"{args.species} | {rep_name}", out_base=out_base)
        rep_meta = {
            "points": int(x.shape[0]),
            "dims": int(x.shape[1]),
            "png": str(out_base.with_suffix(".png")),
            "pdf": str(out_base.with_suffix(".pdf")),
        }
        if args.syllable_plot and s.shape[0] == x.shape[0]:
            syll_base = umap_dir / f"{rep_name}_syllable"
            _scatter_umap_syllables(
                xy=xy,
                labels=s,
                bird_labels=y,
                title=f"{args.species} | {rep_name} | syllable",
                out_base=syll_base,
            )
            rep_meta["syllable_png"] = str(syll_base.with_suffix(".png"))
            rep_meta["syllable_pdf"] = str(syll_base.with_suffix(".pdf"))
        generated[rep_name] = rep_meta

    if not args.no_spec_baseline:
        if args.spec_feature_mode == "flatten":
            spec_rep_name = f"spectrogram_windows_w{effective_spec_window_bins}_h{effective_spec_hop_bins}"
        else:
            spec_rep_name = (
                f"spectrogram_{args.spec_feature_mode}_windows_w"
                f"{effective_spec_window_bins}_h{effective_spec_hop_bins}"
            )
        x_spec, y_spec, s_spec = _build_spectrogram_representation(
            per_bird_spec=per_bird_spec,
            per_bird_lbl=per_bird_spec_lbl,
            window_bins=effective_spec_window_bins,
            hop_bins=effective_spec_hop_bins,
            feature_mode=args.spec_feature_mode,
            max_points_per_bird=args.max_points_per_bird,
            seed=args.seed,
        )
        if x_spec.shape[0] >= 2:
            print(f"[umap] {spec_rep_name}: points={x_spec.shape[0]} dim={x_spec.shape[1]}")
            xy_spec = _fit_umap(
                x=x_spec,
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
                metric=args.umap_metric,
                deterministic=args.deterministic,
                seed=args.seed,
            )
            out_base = umap_dir / spec_rep_name
            _scatter_umap(xy=xy_spec, labels=y_spec, title=f"{args.species} | {spec_rep_name}", out_base=out_base)
            rep_meta = {
                "points": int(x_spec.shape[0]),
                "dims": int(x_spec.shape[1]),
                "png": str(out_base.with_suffix(".png")),
                "pdf": str(out_base.with_suffix(".pdf")),
            }
            if args.syllable_plot and s_spec.shape[0] == x_spec.shape[0]:
                syll_base = umap_dir / f"{spec_rep_name}_syllable"
                _scatter_umap_syllables(
                    xy=xy_spec,
                    labels=s_spec,
                    bird_labels=y_spec,
                    title=f"{args.species} | {spec_rep_name} | syllable",
                    out_base=syll_base,
                )
                rep_meta["syllable_png"] = str(syll_base.with_suffix(".png"))
                rep_meta["syllable_pdf"] = str(syll_base.with_suffix(".pdf"))
            generated[spec_rep_name] = rep_meta
        else:
            print(f"[skip] {spec_rep_name}: not enough points ({x_spec.shape[0]})")

    if not generated:
        raise SystemExit("No UMAP outputs were generated (all representations were skipped).")

    birds_used = sorted(
        set(per_bird_emb.keys())
        | set(per_bird_spec.keys())
    )

    summary = {
        "species": args.species,
        "annotation_json": str(annotation_json),
        "spec_dir": str(spec_dir),
        "run_dir": str(run_dir),
        "run_cache_tag": run_tag,
        "songs_per_bird": int(args.songs_per_bird),
        "seed": int(args.seed),
        "birds_requested": bird_ids,
        "birds_used": birds_used,
        "skipped_birds": skipped_birds,
        "sampled_recordings": sampled_recordings,
        "extraction_meta": extraction_meta,
        "pool_windows": pool_windows,
        "pool_mode": args.pool_mode,
        "songmae_embedding_key": args.songmae_embedding_key,
        "pool_hop_ratio": float(args.pool_hop_ratio),
        "syllable_plot": bool(args.syllable_plot),
        "spec_baseline": bool(not args.no_spec_baseline),
        "spec_window_bins_requested": int(args.spec_window_bins),
        "spec_window_hop_bins_requested": int(args.spec_window_hop_bins),
        "spec_scale_by_patch_width": bool(args.spec_scale_by_patch_width),
        "spec_patch_width_scale": int(spec_patch_width_scale),
        "spec_window_bins": int(effective_spec_window_bins),
        "spec_window_hop_bins": int(effective_spec_hop_bins),
        "spec_feature_mode": args.spec_feature_mode,
        "generated_umaps": generated,
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
