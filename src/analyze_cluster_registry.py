#!/usr/bin/env python3
"""Explore statistics stored in the global cluster registry."""

from __future__ import annotations

import argparse
import json
import logging
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
        median_length=("block_length", "median"),
        mean_length=("block_length", "mean"),
        max_length=("block_length", "max"),
        median_distance=("distance", "median"),
        mean_distance=("distance", "mean"),
        p90_distance=("distance", lambda series: percentile(series, 90.0)),
        has_distance=("has_distance", "any"),
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


def write_summary(
    output_path: Path,
    total_clusters: int,
    populated_clusters: int,
    total_blocks: int,
    coverage_points: Iterable[CoveragePoint],
    bird_counts: dict[str, int],
    noise_candidates: pd.DataFrame,
    correlations: dict[str, float],
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

    output_path.write_text("\n".join(lines))
    LOG.info("Wrote summary to %s", output_path)


def write_tables(
    cluster_stats: pd.DataFrame,
    noise_candidates: pd.DataFrame,
    artifacts: AnalysisArtifacts,
) -> None:
    cluster_stats.to_csv(artifacts.cluster_stats_path, index=True)
    LOG.info("Wrote cluster statistics to %s", artifacts.cluster_stats_path)
    if not noise_candidates.empty:
        noise_candidates.to_csv(artifacts.noise_candidates_path, index=True)
        LOG.info("Wrote noise candidate table to %s", artifacts.noise_candidates_path)


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

    coverage_points = plot_cumulative_coverage(cluster_stats, coverage_plot)
    bird_counts = plot_bird_histogram(members_df, bird_histogram, args.bird_top_n)
    plot_length_vs_density(cluster_stats, length_density_plot)
    correlations = compute_correlations(cluster_stats)

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
    )

    write_tables(cluster_stats, noise_candidates, artifacts)

    metadata = {
        "registry": str(args.registry),
        "artifacts": {
            "coverage_plot": str(coverage_plot),
            "bird_histogram": str(bird_histogram),
            "length_vs_population_plot": str(length_density_plot),
            "summary": str(summary_path),
            "cluster_stats": str(cluster_stats_path),
            "noise_candidates": str(noise_candidates_path),
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
