#!/usr/bin/env python3
"""Inspect the cluster registry focusing on unique block assignments."""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG = logging.getLogger("cluster_registry_inspector")
DEFAULT_OUTPUT = Path("results") / "cluster_registry_snapshot"

FILENAME_PATTERN = re.compile(r"^(?P<timestamp>[^._]+)_(?P<bird0>[^._]+)_(?P<bird1>[^._]+)")


# -----------------------------------------------------------------------------
# Argument parsing / setup
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("cluster_registry.sqlite"),
        help="Path to the cluster registry SQLite database.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory where plots and tables will be written.",
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


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------


def load_member_table(registry_path: Path) -> pd.DataFrame:
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    with sqlite3.connect(registry_path) as connection:
        df = pd.read_sql_query(
            """
            SELECT cluster_id,
                   file_path,
                   channel_index,
                   block_index,
                   start_col,
                   end_col,
                   split
            FROM members
            """,
            connection,
        )
    if df.empty:
        raise ValueError("No records found in members table.")

    df["channel_index"] = pd.to_numeric(df["channel_index"], errors="coerce").fillna(0).astype(np.int64)
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").fillna(-1).astype(np.int64)
    df["start_col"] = pd.to_numeric(df["start_col"], errors="coerce").astype(float)
    df["end_col"] = pd.to_numeric(df["end_col"], errors="coerce").astype(float)
    df["split"] = df["split"].fillna("").astype(str)
    return df


def parse_bird_from_path(path_str: str) -> str | None:
    name = Path(path_str).name
    base = name.split(".", 1)[0]
    match = FILENAME_PATTERN.match(base)
    if not match:
        return None
    bird0 = match.group("bird0")
    bird1 = match.group("bird1")
    return bird0 if bird0 == bird1 else f"{bird0}+{bird1}"


def prepare_unique_blocks(members_df: pd.DataFrame) -> pd.DataFrame:
    members_df = members_df.copy()
    members_df["bird"] = members_df["file_path"].map(parse_bird_from_path)
    key_cols = ["file_path", "channel_index", "start_col", "end_col"]
    unique_blocks = members_df.drop_duplicates(subset=key_cols)
    unique_blocks["block_key"] = list(
        zip(
            unique_blocks["file_path"],
            unique_blocks["channel_index"],
            unique_blocks["start_col"],
            unique_blocks["end_col"],
        )
    )
    return unique_blocks


# -----------------------------------------------------------------------------
# Validations
# -----------------------------------------------------------------------------


def ensure_no_overlap(blocks_df: pd.DataFrame) -> None:
    grouped = blocks_df.groupby(["file_path", "channel_index"], sort=False)
    violations: list[str] = []
    for (file_path, channel_idx), group in grouped:
        ordered = group.sort_values("start_col").reset_index(drop=True)
        previous_end = None
        for _, row in ordered.iterrows():
            start = float(row["start_col"])
            end = float(row["end_col"])
            if previous_end is not None and start < previous_end:
                violations.append(
                    f"{file_path} ch={channel_idx}: ({start}, {end}) overlaps previous end {previous_end}"
                )
                break
            previous_end = end

    if violations:
        examples = "\n  ".join(violations[:10])
        message = "Overlapping blocks detected:\n  " + examples
        if len(violations) > 10:
            message += f"\n  ... ({len(violations) - 10} more)"
        raise ValueError(message)


def ensure_split_coverage(unique_blocks: pd.DataFrame, total_count: int) -> None:
    split_series = unique_blocks["split"].str.lower()
    train_aliases = {"train", "training"}
    val_aliases = {"val", "validation", "valid"}

    train_mask = split_series.isin(train_aliases)
    val_mask = split_series.isin(val_aliases)

    other_mask = ~(train_mask | val_mask)
    other_blocks = unique_blocks[other_mask]
    if not other_blocks.empty:
        unexpected = other_blocks["split"].unique().tolist()
        raise ValueError(
            "Found blocks outside training/validation splits. "
            f"Unexpected splits: {unexpected}"
        )

    train_total = int(train_mask.sum())
    val_total = int(val_mask.sum())
    combined = train_total + val_total

    if combined != total_count:
        raise ValueError(
            "Mismatch between summed unique block assignments and train+validation totals: "
            f"{combined} (train+validation) vs {total_count} (all unique assignments)."
        )

    print(
        f"Split totals check passed: train={train_total:,}, "
        f"validation={val_total:,}, combined={combined:,}."
    )


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def build_cluster_summary(unique_blocks: pd.DataFrame) -> pd.DataFrame:
    grouped = unique_blocks.groupby("cluster_id")
    summary = pd.DataFrame(
        {
            "unique_block_assignments": grouped.size(),
            "unique_birds": grouped["bird"].nunique(dropna=True),
            "unique_files": grouped["file_path"].nunique(),
        }
    )
    summary.index.name = "cluster_id"
    summary = summary.sort_values("unique_block_assignments", ascending=False)
    return summary


def plot_cumulative_clusters(summary_df: pd.DataFrame, output_path: Path) -> None:
    counts = summary_df["unique_block_assignments"].to_numpy(dtype=np.float64)
    if counts.size == 0:
        LOG.warning("No cluster data available for cumulative plot.")
        return

    cumulative = np.cumsum(counts)
    total = cumulative[-1]
    percentages = cumulative / total * 100.0

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(1, counts.size + 1)
    ax.plot(x, cumulative, linewidth=2.0, color="#1f77b4")
    ax.set_xlabel("Clusters (sorted by unique block count, descending)")
    ax.set_ylabel("Cumulative unique block assignments")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for pct in percentiles:
        threshold = pct / 100.0
        idx = np.searchsorted(percentages, pct, side="left")
        if idx >= counts.size:
            idx = counts.size - 1
        ax.axhline(total * threshold, color="gray", linestyle=":", linewidth=0.8)
        ax.axvline(x[idx], color="gray", linestyle=":", linewidth=0.8)
        ax.annotate(
            f"{pct}% @ {x[idx]} clusters",
            xy=(x[idx], total * threshold),
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=8,
            color="black",
        )

    ax.set_xlim(1, counts.size)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    LOG.info("Saved cumulative cluster plot to %s", output_path)


def write_summary_csv(summary_df: pd.DataFrame, output_path: Path) -> None:
    summary_df.to_csv(output_path, index=True)
    LOG.info("Wrote cluster summary to %s", output_path)


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    output_dir = ensure_output_dir(args.output_dir)

    members_df = load_member_table(args.registry)
    unique_blocks = prepare_unique_blocks(members_df)
    ensure_no_overlap(unique_blocks)

    summary_df = build_cluster_summary(unique_blocks)
    write_summary_csv(summary_df, output_dir / "cluster_summary.csv")
    plot_cumulative_clusters(summary_df, output_dir / "cumulative_cluster_sizes.png")

    total_unique = int(summary_df["unique_block_assignments"].sum())
    print(f"Total unique block assignments across all clusters: {total_unique:,}")
    ensure_split_coverage(unique_blocks, total_unique)

    print("Analysis completed successfully.")


if __name__ == "__main__":
    main()
