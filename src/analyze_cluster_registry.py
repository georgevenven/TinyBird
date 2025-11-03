#!/usr/bin/env python3
"""Explore statistics stored in the global cluster registry."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOG = logging.getLogger("cluster_registry_analysis")

FILENAME_PATTERN = re.compile(r"^(?P<timestamp>[^._]+)_(?P<bird0>[^._]+)_(?P<bird1>[^._]+)$")


class CoveragePoint(NamedTuple):
    fraction: float
    clusters_needed: int
    blocks_covered: int


@dataclass
class AnalysisArtifacts:
    output_dir: Path
    coverage_plot: Path
    bird_histogram: Path
    length_density_plot: Path
    summary_path: Path
    cluster_stats_path: Path
    noise_candidates_path: Path
    top_cluster_size_plot: Path
    top_cluster_length_plot: Path
    top_cluster_bird_plot: Path
    bird_balance_plot: Path
    bird_balance_ratio_plot: Path
    top_clusters_stats_path: Path
    top_cluster_birds_path: Path
    bird_balance_table_path: Path
    top_cluster_distance_heatmap: Path
    top_cluster_distance_matrix: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("cluster_registry.sqlite"),
        help="Path to the SQLite cluster registry.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "cluster_registry_analysis",
        help="Directory where plots and reports will be written.",
    )
    parser.add_argument(
        "--bird-top-n",
        type=int,
        default=30,
        help="Number of individual birds to show explicitly in the histogram (others grouped together).",
    )
    parser.add_argument(
        "--noise-max-members",
        type=int,
        default=5,
        help="Treat clusters with at most this many members as potential noise when scoring candidates.",
    )
    parser.add_argument(
        "--noise-distance-quantile",
        type=float,
        default=0.75,
        help="Noise clusters must have a 90th percentile distance above this quantile of the global distribution.",
    )
    parser.add_argument(
        "--min-density-length",
        type=int,
        default=1,
        help="Clamp median length to at least this value when computing density metrics.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_tables(registry_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    with sqlite3.connect(registry_path) as connection:
        clusters_df = pd.read_sql_query(
            """
            SELECT id, dims, window, exemplar, member_count, created_at, updated_at
            FROM clusters
            """,
            connection,
        )
        members_df = pd.read_sql_query(
            """
            SELECT cluster_id,
                   file_path,
                   channel_index,
                   block_index,
                   start_col,
                   end_col,
                   distance,
                   split,
                   source_pickle
            FROM members
            """,
            connection,
        )
    return clusters_df, members_df


def parse_birds_from_filename(path: str) -> tuple[str | None, str | None]:
    name = Path(path).name
    base = name.split(".", 1)[0]
    match = FILENAME_PATTERN.match(base)
    if not match:
        return None, None
    return match.group("bird0"), match.group("bird1")


def assign_bird_labels(members_df: pd.DataFrame) -> pd.DataFrame:
    birds0: list[str | None] = []
    birds1: list[str | None] = []
    birds_channel: list[str | None] = []
    for file_path, channel in zip(members_df["file_path"], members_df["channel_index"], strict=True):
        bird0, bird1 = parse_birds_from_filename(file_path)
        birds0.append(bird0)
        birds1.append(bird1)
        bird: str | None
        if channel == 0:
            bird = bird0
        elif channel == 1:
            bird = bird1
        else:
            bird = bird0 if bird0 == bird1 else None
        birds_channel.append(bird)
    members_df = members_df.copy()
    members_df["bird0"] = birds0
    members_df["bird1"] = birds1
    members_df["bird"] = birds_channel
    members_df["bird_pair"] = [
        f"{b0}+{b1}" if b0 and b1 else None for b0, b1 in zip(birds0, birds1, strict=True)
    ]
    return members_df


def enrich_members(members_df: pd.DataFrame) -> pd.DataFrame:
    members_df = members_df.copy()
    for column in ("start_col", "end_col", "channel_index", "cluster_id", "block_index"):
        members_df[column] = members_df[column].fillna(0).astype(np.int64)
    members_df["distance"] = pd.to_numeric(members_df["distance"], errors="coerce")
    lengths = members_df["end_col"] - members_df["start_col"]
    members_df["block_length"] = lengths.clip(lower=0)
    members_df["has_distance"] = members_df["distance"].notna()
    members_df["block_key"] = list(
        zip(
            members_df["file_path"],
            members_df["channel_index"],
            members_df["start_col"],
            members_df["end_col"],
        )
    )
    return members_df


def compute_cluster_stats(
    members_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    min_density_length: int,
) -> pd.DataFrame:
    def percentile(series: pd.Series, value: float) -> float:
        data = series.dropna().to_numpy()
        if data.size == 0:
            return float("nan")
        return float(np.percentile(data, value))

    grouped = members_df.groupby("cluster_id")
    stats = grouped.agg(
        samples=("cluster_id", "size"),
        min_length=("block_length", "min"),
        median_length=("block_length", "median"),
        mean_length=("block_length", "mean"),
        max_length=("block_length", "max"),
        median_distance=("distance", "median"),
        mean_distance=("distance", "mean"),
        p90_distance=("distance", lambda series: percentile(series, 90.0)),
        has_distance=("has_distance", "any"),
        unique_files=("file_path", pd.Series.nunique),
        unique_blocks=("block_key", pd.Series.nunique),
    )
    stats.index.name = "cluster_id"
    clusters_meta = clusters_df.rename(columns={"id": "cluster_id", "member_count": "registry_member_count"})
    stats = stats.merge(
        clusters_meta[["cluster_id", "registry_member_count", "window"]],
        on="cluster_id",
        how="left",
    )
    stats["member_count"] = stats["samples"]
    median_lengths = stats["median_length"].fillna(0)
    stats["median_length_clamped"] = median_lengths.clip(lower=min_density_length)
    stats["density"] = stats["member_count"] / stats["median_length_clamped"]
    stats["log_member_count"] = np.log1p(stats["member_count"])
    stats["log_median_length"] = np.log1p(stats["median_length_clamped"])
    stats["has_distance"] = stats["has_distance"].astype(bool)
    return stats


def plot_cumulative_coverage(cluster_stats: pd.DataFrame, output_path: Path) -> list[CoveragePoint]:
    counts = cluster_stats["member_count"].to_numpy(dtype=np.float64)
    if counts.size == 0 or counts.sum() == 0:
        LOG.warning("No cluster members available to plot cumulative coverage.")
        return []

    sorted_counts = np.sort(counts)[::-1]
    cumulative = np.cumsum(sorted_counts)
    total = cumulative[-1]
    fraction = cumulative / total

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(1, sorted_counts.size + 1), fraction * 100.0, linewidth=2.0)
    ax.set_xlabel("Clusters (sorted by population)")
    ax.set_ylabel("Cumulative block coverage (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    markers: list[CoveragePoint] = []
    for target in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        idx = int(np.searchsorted(fraction, target, side="left"))
        if idx >= sorted_counts.size:
            idx = sorted_counts.size - 1
        point = CoveragePoint(
            fraction=target,
            clusters_needed=idx + 1,
            blocks_covered=int(cumulative[idx]),
        )
        markers.append(point)
        ax.axhline(target * 100, color="gray", linestyle=":", linewidth=0.8)
        ax.axvline(point.clusters_needed, color="gray", linestyle=":", linewidth=0.8)
        ax.annotate(
            f"{int(target * 100)}% @ {point.clusters_needed} clusters",
            xy=(point.clusters_needed, target * 100),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="black",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    LOG.info("Saved cumulative coverage plot to %s", output_path)
    return markers


def plot_bird_histogram(members_df: pd.DataFrame, output_path: Path, top_n: int) -> dict[str, int]:
    bird_counts = members_df["bird"].dropna().value_counts()
    if bird_counts.empty:
        LOG.warning("No bird identifiers detected; skipping histogram.")
        return {}
    counts = bird_counts.head(top_n).copy()
    other_total = int(bird_counts.iloc[top_n:].sum()) if bird_counts.size > top_n else 0
    other_count = bird_counts.size - min(top_n, bird_counts.size)
    if other_total > 0:
        counts.loc[f"Other ({other_count} birds)"] = other_total

    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.4), 6))
    counts.plot(kind="bar", ax=ax, color="#1f77b4")
    ax.set_ylabel("Blocks observed")
    ax.set_xlabel("Bird")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    LOG.info("Saved bird histogram to %s", output_path)
    return counts.to_dict()


def plot_length_vs_density(cluster_stats: pd.DataFrame, output_path: Path) -> None:
    mask = (cluster_stats["median_length_clamped"] > 0) & (cluster_stats["member_count"] > 0)
    if not mask.any():
        LOG.warning("Insufficient data to plot length vs density.")
        return

    subset = cluster_stats.loc[mask]
    x = subset["median_length_clamped"].to_numpy(dtype=float)
    y = subset["member_count"].to_numpy(dtype=float)
    c = subset["p90_distance"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        x,
        y,
        c=c,
        cmap="viridis",
        alpha=0.7,
        edgecolors="none",
        s=20,
    )
    ax.set_xlabel("Median block length (frames)")
    ax.set_ylabel("Members per cluster")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("90th percentile distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    LOG.info("Saved length vs density scatter to %s", output_path)


def detect_noise_clusters(
    cluster_stats: pd.DataFrame,
    max_members: int,
    distance_quantile: float,
) -> pd.DataFrame:
    if cluster_stats.empty:
        return pd.DataFrame()

    candidates = cluster_stats[
        (cluster_stats["member_count"] <= max_members) & cluster_stats["has_distance"]
    ].copy()
    if candidates.empty:
        LOG.info("No clusters met the preliminary noise filter (<= %d members).", max_members)
        return candidates

    reference = cluster_stats.loc[cluster_stats["p90_distance"].notna(), "p90_distance"]
    if reference.empty:
        LOG.info("Distance values missing; cannot score noise clusters.")
        return pd.DataFrame()

    threshold = reference.quantile(distance_quantile)
    LOG.debug(
        "Noise detection threshold (quantile %.2f) -> %.4f",
        distance_quantile,
        threshold,
    )

    candidates = candidates[candidates["p90_distance"] >= threshold]
    candidates = candidates.assign(
        noise_score=candidates["p90_distance"] / np.log1p(candidates["member_count"])
    )
    return candidates.sort_values("noise_score", ascending=False)


def compute_correlations(cluster_stats: pd.DataFrame) -> dict[str, float]:
    mask = (
        cluster_stats["median_length_clamped"].notna()
        & cluster_stats["member_count"].notna()
        & (cluster_stats["median_length_clamped"] > 0)
        & (cluster_stats["member_count"] > 0)
    )
    if not mask.any():
        return {}
    subset = cluster_stats.loc[mask]
    length = subset["median_length_clamped"].to_numpy(dtype=float)
    members = subset["member_count"].to_numpy(dtype=float)
    log_length = np.log1p(length)
    log_members = np.log1p(members)

    pearson = float(np.corrcoef(length, members)[0, 1])
    pearson_log = float(np.corrcoef(log_length, log_members)[0, 1])
    return {
        "pearson_length_member": pearson,
        "pearson_log_length_member": pearson_log,
    }


def select_top_clusters(
    cluster_stats: pd.DataFrame,
    coverage_fraction: float,
    max_clusters: int,
) -> tuple[pd.DataFrame, float, int]:
    if cluster_stats.empty:
        return cluster_stats.head(0), 0.0, 0

    sorted_stats = cluster_stats.sort_values("member_count", ascending=False)
    counts = sorted_stats["member_count"].to_numpy(dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return sorted_stats.head(0), 0.0, 0

    cumulative = np.cumsum(counts)
    target = coverage_fraction * total
    threshold_idx = int(np.searchsorted(cumulative, target, side="left")) if counts.size else 0
    clusters_needed = min(threshold_idx + 1, counts.size)

    top_n = min(max_clusters, counts.size)
    top_stats = sorted_stats.head(top_n).copy()
    coverage_top_n = float(cumulative[top_n - 1] / total) if top_n else 0.0
    return top_stats, coverage_top_n, clusters_needed


def summarize_top_cluster_birds(
    top_stats: pd.DataFrame,
    members_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if top_stats.empty:
        empty = top_stats.head(0)
        return empty, pd.DataFrame(columns=["cluster_id", "bird", "block_count"]), members_df.head(0)

    cluster_ids = top_stats.index.to_numpy()
    top_members = members_df[members_df["cluster_id"].isin(cluster_ids)].copy()
    top_members["bird"] = top_members["bird"].fillna("unknown")

    bird_counts = (
        top_members.groupby(["cluster_id", "bird"])
        .size()
        .rename("block_count")
        .reset_index()
        .sort_values(["cluster_id", "block_count"], ascending=[True, False])
    )

    summary_records: list[dict] = []
    for cluster_id, sub in bird_counts.groupby("cluster_id"):
        member_count = int(top_stats.loc[cluster_id, "member_count"])
        if member_count <= 0:
            continue
        top_entry = sub.iloc[0]
        dominant_bird = str(top_entry["bird"])
        dominant_fraction = float(top_entry["block_count"] / member_count)
        unique_birds = int(sub["bird"].nunique())
        summary_records.append(
            {
                "cluster_id": int(cluster_id),
                "member_count": member_count,
                "dominant_bird": dominant_bird,
                "dominant_fraction": dominant_fraction,
                "unique_birds": unique_birds,
            }
        )

    summary_df = pd.DataFrame(summary_records).set_index("cluster_id")
    summary_df = summary_df.reindex(top_stats.index).fillna(
        {
            "member_count": top_stats["member_count"],
            "dominant_bird": "unknown",
            "dominant_fraction": 0.0,
            "unique_birds": 0,
        }
    )
    summary_df["member_count"] = top_stats["member_count"]
    summary_df["median_length"] = top_stats["median_length"]
    summary_df["min_length"] = top_stats["min_length"]
    summary_df["max_length"] = top_stats["max_length"]
    summary_df["p90_distance"] = top_stats["p90_distance"]
    summary_df["window"] = top_stats["window"]
    if "unique_files" in top_stats:
        summary_df["unique_files"] = top_stats["unique_files"]
    if "unique_blocks" in top_stats:
        summary_df["unique_blocks"] = top_stats["unique_blocks"]
    return summary_df, bird_counts, top_members


def plot_top_cluster_sizes(top_stats: pd.DataFrame, output_path: Path) -> None:
    if top_stats.empty:
        LOG.warning("Top cluster set is empty; skipping size plot.")
        return

    member_counts = top_stats["member_count"].to_numpy(dtype=np.float64)
    cluster_labels = [str(int(idx)) for idx in top_stats.index]
    fig, ax = plt.subplots(figsize=(min(18, max(10, len(cluster_labels) * 0.18)), 6))
    ax.bar(np.arange(len(member_counts)), member_counts, color="#1f77b4")
    ax.set_ylabel("Blocks per cluster")
    ax.set_xlabel("Cluster id (sorted by size)")
    ax.set_xticks(np.arange(len(cluster_labels)))
    step = max(1, len(cluster_labels) // 20)
    ax.set_xticklabels(cluster_labels, rotation=45, ha="right")
    for idx, label in enumerate(ax.xaxis.get_ticklabels()):
        if isinstance(label, matplotlib.text.Text) and (idx % step != 0):
            label.set_visible(False)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    LOG.info("Saved top cluster size plot to %s", output_path)


def plot_top_cluster_lengths(top_stats: pd.DataFrame, output_path: Path) -> None:
    if top_stats.empty:
        LOG.warning("Top cluster set is empty; skipping length range plot.")
        return
    med = top_stats["median_length"].to_numpy(dtype=np.float64)
    min_len = top_stats["min_length"].fillna(0).to_numpy(dtype=np.float64)
    max_len = top_stats["max_length"].fillna(0).to_numpy(dtype=np.float64)
    med = np.clip(med, a_min=1e-3, a_max=None)
    min_len = np.clip(min_len, a_min=1e-3, a_max=None)
    max_len = np.clip(max_len, a_min=1e-3, a_max=None)
    lower_err = np.maximum(0, med - min_len)
    upper_err = np.maximum(0, max_len - med)
    cluster_labels = [str(int(idx)) for idx in top_stats.index]
    positions = np.arange(len(cluster_labels))
    fig, ax = plt.subplots(figsize=(min(18, max(10, len(cluster_labels) * 0.18)), 6))
    ax.errorbar(
        positions,
        med,
        yerr=[lower_err, upper_err],
        fmt="o",
        ecolor="#2ca02c",
        color="#ff7f0e",
        elinewidth=1.0,
        capsize=3,
        alpha=0.8,
    )
    ax.set_ylabel("Block length (frames)")
    ax.set_xlabel("Cluster id (sorted by size)")
    ax.set_yscale("log")
    ax.set_xticks(positions)
    step = max(1, len(cluster_labels) // 20)
    ax.set_xticklabels(cluster_labels, rotation=45, ha="right")
    for idx, label in enumerate(ax.xaxis.get_ticklabels()):
        if isinstance(label, matplotlib.text.Text) and (idx % step != 0):
            label.set_visible(False)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    LOG.info("Saved top cluster length range plot to %s", output_path)


def plot_top_cluster_bird_specificity(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        LOG.warning("No top cluster summary available; skipping bird specificity plot.")
        return
    counts = summary_df["member_count"].to_numpy(dtype=np.float64)
    counts = np.clip(counts, a_min=1e-3, a_max=None)
    dominant_fraction = summary_df["dominant_fraction"].to_numpy(dtype=np.float64)
    birds = summary_df["dominant_bird"].astype(str).tolist()
    unique_birds = sorted(set(birds))
    cmap = plt.get_cmap("tab20", max(1, len(unique_birds)))
    color_map = {bird: cmap(idx % cmap.N) for idx, bird in enumerate(unique_birds)}
    colors = [color_map[bird] for bird in birds]

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(counts, dominant_fraction * 100.0, c=colors, s=40, alpha=0.8, edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("Cluster size (blocks, log scale)")
    ax.set_ylabel("Dominant bird share (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    legend_handles = [
        matplotlib.lines.Line2D([0], [0], marker="o", color="w", label=bird, markerfacecolor=color_map[bird], markersize=6)
        for bird in unique_birds[:15]
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, title="Dominant bird", loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    LOG.info("Saved bird specificity plot to %s", output_path)


def compute_bird_balance(
    members_df: pd.DataFrame,
    top_members: pd.DataFrame,
    top_limit: int = 20,
) -> pd.DataFrame:
    overall = members_df["bird"].dropna()
    subset = top_members["bird"].dropna()
    if overall.empty or subset.empty:
        return pd.DataFrame(
            columns=[
                "bird",
                "top_blocks",
                "overall_blocks",
                "top_share",
                "overall_share",
                "representation_ratio",
                "balanced_weighted_share",
            ]
        )

    combined = (
        pd.concat(
            [
                subset.value_counts().rename("top_blocks"),
                overall.value_counts().rename("overall_blocks"),
            ],
            axis=1,
        )
        .fillna(0)
        .sort_values("top_blocks", ascending=False)
    )

    tail_birds: list[str] = []
    if combined.shape[0] > top_limit:
        head = combined.iloc[:top_limit]
        tail = combined.iloc[top_limit:]
        tail_birds = tail.index.tolist()
        aggregated = pd.DataFrame(
            {
                "top_blocks": [tail["top_blocks"].sum()],
                "overall_blocks": [tail["overall_blocks"].sum()],
            },
            index=["Other"],
        )
        combined = pd.concat([head, aggregated], axis=0)

    total_top = combined["top_blocks"].sum()
    total_overall = combined["overall_blocks"].sum()
    combined["top_share"] = (combined["top_blocks"] / total_top) if total_top else 0.0
    combined["overall_share"] = (combined["overall_blocks"] / total_overall) if total_overall else 0.0
    combined["representation_ratio"] = combined["top_share"] / combined["overall_share"].replace({0: np.nan})

    weights = (1.0 / overall.value_counts()).to_dict()
    weighted_series = subset.map(weights).fillna(0.0)
    balanced_counts = weighted_series.groupby(subset).sum()
    combined_index = combined.index.tolist()
    balanced_counts = balanced_counts.reindex([idx for idx in combined_index if idx != "Other"], fill_value=0.0)
    if "Other" in combined_index:
        other_weight = weighted_series[top_members["bird"].isin(tail_birds)].sum() if tail_birds else 0.0
        balanced_counts = pd.concat([balanced_counts, pd.Series({"Other": other_weight})])
    total_weight = balanced_counts.sum()
    if total_weight > 0:
        balanced_share = (balanced_counts / total_weight)
    else:
        balanced_share = balanced_counts
    combined["balanced_weighted_share"] = balanced_share.reindex(combined.index).fillna(0.0)

    combined.reset_index(inplace=True)
    combined.rename(columns={"index": "bird"}, inplace=True)
    return combined


def _deserialize_exemplar(blob: bytes) -> np.ndarray:
    return pickle.loads(blob)


def _normalize_vector(array: np.ndarray) -> np.ndarray:
    vec = np.asarray(array, dtype=np.float64).ravel()
    if vec.size == 0:
        return vec
    mean = np.mean(vec)
    std = np.std(vec)
    if std < 1e-8:
        return vec - mean
    return (vec - mean) / std


def load_prototypes_for_clusters(clusters_df: pd.DataFrame, cluster_ids: Iterable[int]) -> dict[int, dict]:
    subset = clusters_df[clusters_df["id"].isin(list(cluster_ids))]
    prototypes: dict[int, dict] = {}
    for _, row in subset.iterrows():
        cluster_id = int(row["id"])
        dims = json.loads(row["dims"]) if row["dims"] else []
        dims_arr = np.asarray(dims, dtype=np.int32)
        exemplar_blob = row["exemplar"]
        exemplar = _deserialize_exemplar(exemplar_blob)
        if dims_arr.size == 0 and exemplar.ndim == 2:
            dims_arr = np.arange(exemplar.shape[0], dtype=np.int32)
        prototypes[cluster_id] = {
            "dims": dims_arr,
            "window": int(row["window"]),
            "exemplar": np.asarray(exemplar, dtype=np.float64),
        }
    return prototypes


def build_distance_matrix(
    prototypes: dict[int, dict],
    ordered_ids: list[int],
) -> np.ndarray:
    n = len(ordered_ids)
    matrix = np.full((n, n), np.nan, dtype=np.float64)
    vectors: dict[int, np.ndarray] = {}
    windows: dict[int, int] = {}
    for cluster_id in ordered_ids:
        proto = prototypes.get(cluster_id)
        if not proto:
            continue
        exemplar = proto["exemplar"]
        dims = proto["dims"]
        if exemplar.ndim == 2 and dims.size and dims.size == exemplar.shape[0]:
            selected = exemplar
        else:
            selected = exemplar
            if exemplar.ndim == 3:
                selected = exemplar.reshape(exemplar.shape[0], -1)
        vectors[cluster_id] = _normalize_vector(selected)
        windows[cluster_id] = proto["window"]

    for i in range(n):
        cid_i = ordered_ids[i]
        vec_i = vectors.get(cid_i)
        win_i = windows.get(cid_i)
        if vec_i is None:
            continue
        for j in range(i + 1, n):
            cid_j = ordered_ids[j]
            vec_j = vectors.get(cid_j)
            win_j = windows.get(cid_j)
            if vec_j is None or win_i != win_j or vec_i.size != vec_j.size:
                continue
            dist = float(np.linalg.norm(vec_i - vec_j))
            matrix[i, j] = dist
            matrix[j, i] = dist
    np.fill_diagonal(matrix, 0.0)
    return matrix


def order_distance_matrix(distance_matrix: np.ndarray) -> list[int]:
    n = distance_matrix.shape[0]
    if n == 0:
        return []
    order = [0]
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    for _ in range(1, n):
        last = order[-1]
        row = distance_matrix[last].copy()
        row[np.isnan(row)] = np.inf
        row[visited] = np.inf
        if np.all(np.isinf(row)):
            next_idx_candidates = np.where(~visited)[0]
            if next_idx_candidates.size == 0:
                break
            next_idx = int(next_idx_candidates[0])
        else:
            next_idx = int(np.argmin(row))
        order.append(next_idx)
        visited[next_idx] = True
    return order


def plot_distance_heatmap(
    distance_matrix: np.ndarray,
    cluster_ids: list[int],
    output_path: Path,
) -> None:
    if distance_matrix.size == 0:
        LOG.warning("Distance matrix empty; skipping heatmap.")
        return
    finite = distance_matrix[np.isfinite(distance_matrix)]
    fill_value = float(np.nanmax(finite)) if finite.size else 0.0
    remapped = np.where(np.isnan(distance_matrix), fill_value, distance_matrix)
    mask = np.isnan(distance_matrix)
    data = np.ma.array(remapped, mask=mask)
    n = distance_matrix.shape[0]
    fig_size = max(6, min(18, n * 0.35))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    cmap = plt.cm.get_cmap("magma").copy()
    cmap.set_bad(color="lightgrey")
    im = ax.imshow(data, cmap=cmap, interpolation="nearest")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(cluster_ids, rotation=90, fontsize="small")
    ax.set_yticklabels(cluster_ids, fontsize="small")
    ax.set_xlabel("Cluster id")
    ax.set_ylabel("Cluster id")
    ax.set_title("Top coverage clusters similarity (Euclidean distance)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    LOG.info("Saved cluster distance heatmap to %s", output_path)


def write_distance_matrix_csv(
    distance_matrix: np.ndarray,
    cluster_ids: list[int],
    output_path: Path,
) -> None:
    if distance_matrix.size == 0:
        return
    df = pd.DataFrame(distance_matrix, index=cluster_ids, columns=cluster_ids)
    df.to_csv(output_path)
    LOG.info("Wrote cluster distance matrix to %s", output_path)


def plot_bird_balance(
    balance_df: pd.DataFrame,
    share_plot_path: Path,
    ratio_plot_path: Path,
) -> None:
    if balance_df.empty:
        LOG.warning("Bird balance table empty; skipping balance plots.")
        return

    birds = balance_df["bird"].tolist()
    x = np.arange(len(birds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(min(16, max(10, len(birds) * 0.5)), 6))
    ax.bar(x - width / 2, balance_df["overall_share"], width, label="Overall share", color="#1f77b4")
    ax.bar(x + width / 2, balance_df["top_share"], width, label="Top clusters share", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(birds, rotation=45, ha="right")
    ax.set_ylabel("Share of blocks")
    ax.set_xlabel("Bird")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(share_plot_path, dpi=200)
    plt.close(fig)
    LOG.info("Saved bird share comparison plot to %s", share_plot_path)

    fig, ax = plt.subplots(figsize=(min(16, max(10, len(birds) * 0.5)), 5))
    ax.bar(x, balance_df["representation_ratio"], color="#2ca02c")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(birds, rotation=45, ha="right")
    ax.set_ylabel("Representation ratio (top / overall)")
    ax.set_xlabel("Bird")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(ratio_plot_path, dpi=200)
    plt.close(fig)
    LOG.info("Saved bird representation ratio plot to %s", ratio_plot_path)

def write_summary(
    output_path: Path,
    total_clusters: int,
    populated_clusters: int,
    total_blocks: int,
    coverage_points: Iterable[CoveragePoint],
    bird_counts: dict[str, int],
    noise_candidates: pd.DataFrame,
    correlations: dict[str, float],
    top_stats: pd.DataFrame,
    top_coverage: float,
    clusters_for_half: int,
    balance_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("Cluster Registry Analysis")
    lines.append("========================\n")

    lines.append(f"Registry clusters: {total_clusters:,}")
    lines.append(f"Clusters with members: {populated_clusters:,}")
    lines.append(f"Total registered blocks: {total_blocks:,}\n")

    if coverage_points:
        lines.append("Block coverage by cluster count:")
        for point in coverage_points:
            pct = int(point.fraction * 100)
            lines.append(
                f"  {pct:>2}% coverage -> {point.clusters_needed:,} clusters "
                f"({point.blocks_covered:,} blocks)"
            )
        lines.append("")

    if not top_stats.empty:
        lines.append("Top clusters focus:")
        lines.append(
            f"  Analysed clusters: {len(top_stats):,} (cover {top_coverage * 100:.2f}% of blocks)"
        )
        lines.append(
            f"  Clusters needed for 50%% coverage: {clusters_for_half:,}"
        )
        median_dominant = top_stats["dominant_fraction"].median() * 100 if "dominant_fraction" in top_stats else float("nan")
        lines.append(
            f"  Median dominant bird share: {median_dominant:.1f}%"
        )
        lines.append("")

    if bird_counts:
        lines.append("Top birds by block count:")
        for bird, count in bird_counts.items():
            lines.append(f"  {bird}: {count:,}")
        lines.append("")

    if correlations:
        lines.append("Cluster length vs population correlation:")
        for name, value in correlations.items():
            lines.append(f"  {name}: {value:.4f}")
        lines.append("")

    if not noise_candidates.empty:
        lines.append("Most suspicious noise clusters (top 20):")
        for _, row in noise_candidates.head(20).iterrows():
            cid_value = row.get("cluster_id", row.name)
            cid = int(cid_value) if pd.notna(cid_value) else int(row.name)
            lines.append(
                f"  cluster {cid}: members={int(row['member_count'])}, "
                f"median_length={row['median_length']:.1f}, "
                f"p90_distance={row['p90_distance']:.4f}, "
                f"noise_score={row['noise_score']:.4f}"
            )
        lines.append("")
    else:
        lines.append("No candidate noise clusters met the scoring criteria.\n")

    if not balance_df.empty:
        lines.append("Bird representation in top clusters (top entries):")
        for _, row in balance_df.head(10).iterrows():
            bird = row["bird"]
            top_share = row["top_share"] * 100
            overall_share = row["overall_share"] * 100
            ratio = row["representation_ratio"]
            lines.append(
                f"  {bird}: top share {top_share:.2f}% vs overall {overall_share:.2f}% (ratio {ratio:.2f}x)"
            )
        lines.append("")

    output_path.write_text("\n".join(lines))
    LOG.info("Wrote summary to %s", output_path)


def write_tables(
    cluster_stats: pd.DataFrame,
    noise_candidates: pd.DataFrame,
    top_summary: pd.DataFrame,
    top_bird_counts: pd.DataFrame,
    balance_df: pd.DataFrame,
    artifacts: AnalysisArtifacts,
) -> None:
    cluster_stats.to_csv(artifacts.cluster_stats_path, index=True)
    LOG.info("Wrote cluster statistics to %s", artifacts.cluster_stats_path)
    if not noise_candidates.empty:
        noise_candidates.to_csv(artifacts.noise_candidates_path, index=True)
        LOG.info("Wrote noise candidate table to %s", artifacts.noise_candidates_path)
    if not top_summary.empty:
        top_summary.to_csv(artifacts.top_clusters_stats_path, index=True)
        LOG.info("Wrote top cluster summary to %s", artifacts.top_clusters_stats_path)
    if not top_bird_counts.empty:
        top_bird_counts.to_csv(artifacts.top_cluster_birds_path, index=False)
        LOG.info("Wrote top cluster bird distribution to %s", artifacts.top_cluster_birds_path)
    if not balance_df.empty:
        balance_df.to_csv(artifacts.bird_balance_table_path, index=False)
        LOG.info("Wrote bird balance table to %s", artifacts.bird_balance_table_path)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    output_dir = ensure_output_dir(args.output_dir)
    clusters_df, members_df = load_tables(args.registry)
    LOG.info(
        "Loaded %d clusters and %d members from %s",
        len(clusters_df),
        len(members_df),
        args.registry,
    )

    members_df = assign_bird_labels(enrich_members(members_df))
    cluster_stats = compute_cluster_stats(members_df, clusters_df, args.min_density_length)

    coverage_plot = output_dir / "cumulative_block_coverage.png"
    bird_histogram = output_dir / "blocks_per_bird.png"
    length_density_plot = output_dir / "cluster_length_vs_population.png"
    summary_path = output_dir / "summary.txt"
    cluster_stats_path = output_dir / "cluster_stats.csv"
    noise_candidates_path = output_dir / "noise_clusters.csv"
    top_cluster_size_plot = output_dir / "top_clusters_block_counts.png"
    top_cluster_length_plot = output_dir / "top_clusters_length_ranges.png"
    top_cluster_bird_plot = output_dir / "top_clusters_dominant_birds.png"
    bird_balance_plot = output_dir / "bird_share_comparison.png"
    bird_balance_ratio_plot = output_dir / "bird_representation_ratio.png"
    top_cluster_distance_heatmap = output_dir / "top_clusters_distance_heatmap.png"
    top_cluster_distance_matrix = output_dir / "top_clusters_distance_matrix.csv"
    top_clusters_stats_path = output_dir / "top_clusters_summary.csv"
    top_cluster_birds_path = output_dir / "top_clusters_bird_distribution.csv"
    bird_balance_table_path = output_dir / "bird_balance_table.csv"

    coverage_points = plot_cumulative_coverage(cluster_stats, coverage_plot)
    bird_counts = plot_bird_histogram(members_df, bird_histogram, args.bird_top_n)
    plot_length_vs_density(cluster_stats, length_density_plot)
    correlations = compute_correlations(cluster_stats)

    top_stats_raw, top_coverage, clusters_for_half = select_top_clusters(
        cluster_stats, coverage_fraction=0.5, max_clusters=129
    )
    top_summary, top_bird_counts, top_members = summarize_top_cluster_birds(top_stats_raw, members_df)
    plot_top_cluster_sizes(top_summary, top_cluster_size_plot)
    plot_top_cluster_lengths(top_summary, top_cluster_length_plot)
    plot_top_cluster_bird_specificity(top_summary, top_cluster_bird_plot)
    balance_df = compute_bird_balance(members_df, top_members)
    plot_bird_balance(balance_df, bird_balance_plot, bird_balance_ratio_plot)

    distance_order_ids = list(top_summary.index)
    distance_matrix = np.empty((0, 0))
    if distance_order_ids:
        prototypes = load_prototypes_for_clusters(clusters_df, distance_order_ids)
        distance_matrix = build_distance_matrix(prototypes, distance_order_ids)
        if distance_matrix.size:
            order_indices = order_distance_matrix(distance_matrix)
            ordered_ids = [distance_order_ids[idx] for idx in order_indices]
            reordered_matrix = distance_matrix[np.ix_(order_indices, order_indices)]
            plot_distance_heatmap(reordered_matrix, ordered_ids, top_cluster_distance_heatmap)
            write_distance_matrix_csv(reordered_matrix, ordered_ids, top_cluster_distance_matrix)
        else:
            distance_matrix = np.empty((0, 0))
            LOG.info("Distance matrix unavailable; heatmap skipped.")

    noise_candidates = detect_noise_clusters(
        cluster_stats,
        max_members=args.noise_max_members,
        distance_quantile=args.noise_distance_quantile,
    )

    artifacts = AnalysisArtifacts(
        output_dir=output_dir,
        coverage_plot=coverage_plot,
        bird_histogram=bird_histogram,
        length_density_plot=length_density_plot,
        summary_path=summary_path,
        cluster_stats_path=cluster_stats_path,
        noise_candidates_path=noise_candidates_path,
        top_cluster_size_plot=top_cluster_size_plot,
        top_cluster_length_plot=top_cluster_length_plot,
        top_cluster_bird_plot=top_cluster_bird_plot,
        bird_balance_plot=bird_balance_plot,
        bird_balance_ratio_plot=bird_balance_ratio_plot,
        top_clusters_stats_path=top_clusters_stats_path,
        top_cluster_birds_path=top_cluster_birds_path,
        bird_balance_table_path=bird_balance_table_path,
        top_cluster_distance_heatmap=top_cluster_distance_heatmap,
        top_cluster_distance_matrix=top_cluster_distance_matrix,
    )

    write_summary(
        summary_path,
        total_clusters=len(clusters_df),
        populated_clusters=cluster_stats["member_count"].astype(bool).sum(),
        total_blocks=int(cluster_stats["member_count"].sum()),
        coverage_points=coverage_points,
        bird_counts=bird_counts,
        noise_candidates=noise_candidates,
        correlations=correlations,
        top_stats=top_summary,
        top_coverage=top_coverage,
        clusters_for_half=clusters_for_half,
        balance_df=balance_df,
    )

    write_tables(
        cluster_stats,
        noise_candidates,
        top_summary,
        top_bird_counts,
        balance_df,
        artifacts,
    )

    metadata = {
        "registry": str(args.registry),
        "artifacts": {
            "coverage_plot": str(coverage_plot),
            "bird_histogram": str(bird_histogram),
            "length_vs_population_plot": str(length_density_plot),
            "top_cluster_size_plot": str(top_cluster_size_plot),
            "top_cluster_length_plot": str(top_cluster_length_plot),
            "top_cluster_bird_plot": str(top_cluster_bird_plot),
            "bird_balance_plot": str(bird_balance_plot),
            "bird_balance_ratio_plot": str(bird_balance_ratio_plot),
            "top_cluster_distance_heatmap": str(top_cluster_distance_heatmap),
            "top_cluster_distance_matrix": str(top_cluster_distance_matrix),
            "summary": str(summary_path),
            "cluster_stats": str(cluster_stats_path),
            "noise_candidates": str(noise_candidates_path),
            "top_clusters_summary": str(top_clusters_stats_path),
            "top_cluster_birds": str(top_cluster_birds_path),
            "bird_balance_table": str(bird_balance_table_path),
        },
        "parameters": {
            "bird_top_n": args.bird_top_n,
            "noise_max_members": args.noise_max_members,
            "noise_distance_quantile": args.noise_distance_quantile,
            "min_density_length": args.min_density_length,
        },
    }
    metadata_path = output_dir / "analysis_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    LOG.info("Wrote metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
