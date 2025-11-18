#!/usr/bin/env python3
"""Export motif registry analytics to CSV files.

This tool inspects the SQLite registry produced by build_motif_registry.py and
generates CSV summaries that can be loaded into Google Sheets, Excel, or any
other data tool for charting. It produces:

* motif_summary.csv – one row per motif with aggregate coverage stats.
* motif_<motif_id>_overview.csv / motif_<motif_id>_matches.csv – optional motif
  drill-downs when --motif_id is supplied.
* file_summary_<stem>.csv / file_matches_<stem>.csv – optional per-channel file
  analytics when --file_path is supplied.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sqlite3
import statistics
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze motif registry and export analytics as CSV files.")
    parser.add_argument(
        "--registry", type=Path, default=Path("../motif_registry.sqlite"), help="Path to the motif registry DB."
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("../results/motif_analytics"), help="Directory for generated CSVs."
    )
    parser.add_argument("--motif_id", type=int, help="Optional motif ID for detailed analytics CSVs.")
    parser.add_argument("--file_path", type=Path, help="Optional .pt file path for per-channel analytics CSVs.")
    return parser.parse_args(argv)


def load_registry_data(db_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute("SELECT id, pattern_dataset_id, dim_index, length FROM motifs ORDER BY id")
    motifs: list[dict[str, Any]] = []
    for motif_id, dataset_id, dim_index, length in cursor.fetchall():
        motifs.append(
            {
                "motif_id": int(motif_id),
                "pattern_dataset_id": dataset_id,
                "dim_index": int(dim_index),
                "length": int(length),
            }
        )

    cursor.execute("SELECT motif_id, dataset_id, seed_index, seed_distance, starts FROM motif_matches")
    matches: list[dict[str, Any]] = []
    for motif_id, dataset_id, seed_index, seed_distance, starts_blob in cursor.fetchall():
        try:
            starts = list(pickle.loads(starts_blob))
        except Exception:
            starts = []
        matches.append(
            {
                "motif_id": int(motif_id),
                "dataset_id": dataset_id,
                "seed_index": int(seed_index),
                "seed_distance": float(seed_distance),
                "starts": starts,
            }
        )

    cursor.execute("SELECT dataset_id, file_path, channel_index, split, bird, column_map FROM channels")
    channels: list[dict[str, Any]] = []
    for dataset_id, file_path, channel_index, split, bird, column_blob in cursor.fetchall():
        try:
            column_map = np.asarray(pickle.loads(column_blob), dtype=np.int64)
        except Exception:
            column_map = np.asarray([], dtype=np.int64)
        channels.append(
            {
                "dataset_id": dataset_id,
                "file_path": file_path,
                "channel_index": int(channel_index),
                "split": split,
                "bird": bird,
                "column_map": column_map,
            }
        )

    connection.close()
    return motifs, matches, channels


def _format_starts(starts: list[int], max_items: int = 25) -> str:
    if not starts:
        return ""
    if len(starts) > max_items:
        head = ", ".join(str(s) for s in starts[:max_items])
        return f"{head}, ... (total={len(starts)})"
    return ", ".join(str(s) for s in starts)


def build_motif_summary(
    motifs: list[dict[str, Any]], matches: list[dict[str, Any]], channels: list[dict[str, Any]]
) -> tuple[list[list[Any]], dict[int, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    dataset_meta = {row["dataset_id"]: row for row in channels}
    matches_by_motif: dict[int, list[dict[str, Any]]] = {}
    for match in matches:
        matches_by_motif.setdefault(match["motif_id"], []).append(match)

    rows: list[list[Any]] = [
        [
            "motif_id",
            "pattern_dataset_id",
            "dim_index",
            "length",
            "total_occurrences",
            "datasets_with_match",
            "birds_with_match",
        ]
    ]
    for motif in motifs:
        motif_id = motif["motif_id"]
        motif_matches = matches_by_motif.get(motif_id, [])
        dataset_ids = {m["dataset_id"] for m in motif_matches}
        birds = {dataset_meta.get(ds, {}).get("bird") for ds in dataset_ids if dataset_meta.get(ds, {}).get("bird")}
        rows.append(
            [
                motif_id,
                motif["pattern_dataset_id"],
                motif["dim_index"],
                motif["length"],
                sum(len(m["starts"]) for m in motif_matches),
                len(dataset_ids),
                len(birds),
            ]
        )
    return rows, matches_by_motif, dataset_meta


def build_motif_detail(
    motif_id: int,
    motifs: list[dict[str, Any]],
    matches_by_motif: dict[int, list[dict[str, Any]]],
    dataset_meta: dict[str, dict[str, Any]],
) -> tuple[list[list[Any]], list[list[Any]]]:
    motif_lookup = {m["motif_id"]: m for m in motifs}
    motif = motif_lookup.get(motif_id)
    if motif is None:
        raise ValueError(f"Motif {motif_id} not found in registry.")

    motif_matches = matches_by_motif.get(motif_id, [])
    dataset_ids = {m["dataset_id"] for m in motif_matches}
    birds = {dataset_meta.get(ds, {}).get("bird") for ds in dataset_ids if dataset_meta.get(ds, {}).get("bird")}
    overview_rows = [
        ["metric", "value"],
        ["motif_id", motif_id],
        ["pattern_dataset_id", motif["pattern_dataset_id"]],
        ["dim_index", motif["dim_index"]],
        ["length", motif["length"]],
        ["datasets_with_matches", len(dataset_ids)],
        ["birds_with_matches", len(birds)],
        ["total_occurrences", sum(len(m["starts"]) for m in motif_matches)],
    ]

    detail_rows = [["dataset_id", "bird", "seed_index", "seed_distance", "occurrences", "start_positions"]]
    for match in sorted(motif_matches, key=lambda m: (m["dataset_id"], m["seed_distance"])):
        meta = dataset_meta.get(match["dataset_id"], {})
        detail_rows.append(
            [
                match["dataset_id"],
                meta.get("bird", ""),
                match["seed_index"],
                round(match["seed_distance"], 3),
                len(match["starts"]),
                _format_starts(match["starts"]),
            ]
        )
    return overview_rows, detail_rows


def _count_segments(column_map: np.ndarray) -> int:
    if column_map.size == 0:
        return 0
    segments = 0
    in_segment = False
    for value in column_map:
        if value >= 0 and not in_segment:
            segments += 1
            in_segment = True
        elif value < 0 and in_segment:
            in_segment = False
    return segments


def build_file_overview(
    file_path: Path, channels: list[dict[str, Any]], matches: list[dict[str, Any]]
) -> tuple[list[list[Any]], list[list[Any]]]:
    resolved = str(file_path.resolve())
    matches_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for match in matches:
        matches_by_dataset.setdefault(match["dataset_id"], []).append(match)

    channel_rows = [
        row
        for row in channels
        if row["file_path"] == str(file_path)
        or str(Path(row["file_path"]).resolve()) == resolved
        or Path(row["file_path"]).name == file_path.name
    ]
    if not channel_rows:
        raise ValueError(f"No channel metadata found for {file_path}")

    summary_rows = [
        [
            "dataset_id",
            "channel_index",
            "bird",
            "split",
            "chirp_segments",
            "columns",
            "unique_motifs",
            "total_occurrences",
            "avg_seed_distance",
        ]
    ]
    detail_rows = [["dataset_id", "motif_id", "seed_index", "seed_distance", "occurrences", "start_positions"]]

    for row in sorted(channel_rows, key=lambda r: r["channel_index"]):
        dataset_id = row["dataset_id"]
        dataset_matches = matches_by_dataset.get(dataset_id, [])
        column_map = row["column_map"]
        segments = _count_segments(column_map)
        columns = int(np.count_nonzero(column_map >= 0))
        total_occurrences = sum(len(match["starts"]) for match in dataset_matches)
        unique_motifs = {match["motif_id"] for match in dataset_matches}
        avg_seed = statistics.mean(match["seed_distance"] for match in dataset_matches) if dataset_matches else None

        summary_rows.append(
            [
                dataset_id,
                row["channel_index"],
                row.get("bird", ""),
                row.get("split", ""),
                segments,
                columns,
                len(unique_motifs),
                total_occurrences,
                round(avg_seed, 3) if avg_seed is not None else "",
            ]
        )

        for match in sorted(dataset_matches, key=lambda m: m["motif_id"]):
            detail_rows.append(
                [
                    dataset_id,
                    match["motif_id"],
                    match["seed_index"],
                    round(match["seed_distance"], 3),
                    len(match["starts"]),
                    _format_starts(match["starts"]),
                ]
            )

    return summary_rows, detail_rows


def write_csv(path: Path, rows: list[list[Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    motifs, matches, channels = load_registry_data(args.registry)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows, matches_by_motif, dataset_meta = build_motif_summary(motifs, matches, channels)
    summary_path = output_dir / "motif_summary.csv"
    write_csv(summary_path, summary_rows)
    print(f"Wrote motif summary: {summary_path}")

    if args.motif_id is not None:
        overview_rows, detail_rows = build_motif_detail(args.motif_id, motifs, matches_by_motif, dataset_meta)
        overview_path = output_dir / f"motif_{args.motif_id}_overview.csv"
        matches_path = output_dir / f"motif_{args.motif_id}_matches.csv"
        write_csv(overview_path, overview_rows)
        write_csv(matches_path, detail_rows)
        print(f"Wrote motif detail CSVs: {overview_path}, {matches_path}")

    if args.file_path is not None:
        summary_rows_file, detail_rows_file = build_file_overview(args.file_path, channels, matches)
        stem = args.file_path.stem.replace(" ", "_")
        summary_path_file = output_dir / f"file_summary_{stem}.csv"
        detail_path_file = output_dir / f"file_matches_{stem}.csv"
        write_csv(summary_path_file, summary_rows_file)
        write_csv(detail_path_file, detail_rows_file)
        print(f"Wrote file analytics CSVs: {summary_path_file}, {detail_path_file}")


if __name__ == "__main__":
    main()
