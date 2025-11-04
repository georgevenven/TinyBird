#!/usr/bin/env python3
"""
Ensure each audio block is assigned to exactly one cluster.

The script removes duplicate block assignments from the `members` table and
enforces a unique constraint so future inserts cannot violate the rule.
"""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, NamedTuple


class BlockRow(NamedTuple):
    rowid: int
    cluster_id: int
    file_path: str
    channel_index: int
    start_col: float
    end_col: float
    distance: float | None


class FixSummary(NamedTuple):
    removed_rows: int
    removed_duplicates: int
    removed_incomplete: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("cluster_registry.sqlite"),
        help="Path to the cluster registry SQLite database.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Tolerance used when comparing start/end column values.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("results") / "cluster_registry_fixes" / "fixed_duplicate_blocks.csv",
        help="CSV file capturing all corrective actions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect the database and report conflicts without applying changes.",
    )
    return parser.parse_args()


def normalize_value(value: float | None, tolerance: float) -> float | None:
    if value is None or math.isnan(value):
        return None
    return round(float(value) / tolerance) * tolerance


def fetch_members(connection: sqlite3.Connection) -> list[BlockRow]:
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT rowid,
               cluster_id,
               file_path,
               channel_index,
               start_col,
               end_col,
               distance
        FROM members
        """
    ).fetchall()
    result: list[BlockRow] = []
    for row in rows:
        cluster_id = row["cluster_id"]
        file_path = row["file_path"]
        channel_index = row["channel_index"]
        start_col = row["start_col"]
        end_col = row["end_col"]
        distance = row["distance"]
        result.append(
            BlockRow(
                rowid=int(row["rowid"]),
                cluster_id=int(cluster_id) if cluster_id is not None else -1,
                file_path=str(file_path) if file_path is not None else "",
                channel_index=int(channel_index) if channel_index is not None else -1,
                start_col=float(start_col) if start_col is not None else math.nan,
                end_col=float(end_col) if end_col is not None else math.nan,
                distance=float(distance) if distance is not None else None,
            )
        )
    return result


def fetch_cluster_sizes(connection: sqlite3.Connection) -> Mapping[int, int]:
    connection.row_factory = sqlite3.Row
    rows = connection.execute("SELECT id, member_count FROM clusters").fetchall()
    return {int(row["id"]): int(row["member_count"] or 0) for row in rows}


def classify_rows(
    rows: Iterable[BlockRow],
    tolerance: float,
) -> tuple[dict[tuple, list[BlockRow]], list[BlockRow]]:
    duplicates: dict[tuple, list[BlockRow]] = defaultdict(list)
    incomplete: list[BlockRow] = []
    for row in rows:
        if (
            row.cluster_id < 0
            or not row.file_path
            or row.channel_index < 0
            or math.isnan(row.start_col)
            or math.isnan(row.end_col)
        ):
            incomplete.append(row)
            continue
        key = (
            row.file_path,
            row.channel_index,
            normalize_value(row.start_col, tolerance),
            normalize_value(row.end_col, tolerance),
        )
        duplicates[key].append(row)
    return duplicates, incomplete


def choose_primary(
    rows: list[BlockRow],
    cluster_sizes: Mapping[int, int],
) -> BlockRow:
    def sort_key(item: BlockRow) -> tuple:
        distance = item.distance
        distance_missing = distance is None or math.isnan(distance)
        distance_value = float("inf") if distance_missing else float(distance)
        cluster_size = -cluster_sizes.get(item.cluster_id, 0)
        return distance_missing, distance_value, cluster_size, item.cluster_id, item.rowid

    rows_sorted = sorted(rows, key=sort_key)
    return rows_sorted[0]


def apply_corrections(
    connection: sqlite3.Connection,
    duplicates: Mapping[tuple, list[BlockRow]],
    incomplete: list[BlockRow],
    cluster_sizes: Mapping[int, int],
    log_path: Path,
    dry_run: bool,
) -> FixSummary:
    duplicate_rows_to_remove: list[BlockRow] = []
    log_entries: list[dict[str, object]] = []

    for key, items in duplicates.items():
        if len(items) == 1:
            continue
        keeper = choose_primary(items, cluster_sizes)
        for row in items:
            if row.rowid == keeper.rowid:
                continue
            duplicate_rows_to_remove.append(row)
            log_entries.append(
                {
                    "action": "remove_duplicate",
                    "file_path": row.file_path,
                    "channel_index": row.channel_index,
                    "start_col": row.start_col,
                    "end_col": row.end_col,
                    "cluster_id_kept": keeper.cluster_id,
                    "cluster_id_removed": row.cluster_id,
                    "distance_kept": keeper.distance,
                    "distance_removed": row.distance,
                    "rowid_removed": row.rowid,
                }
            )

    for row in incomplete:
        log_entries.append(
            {
                "action": "remove_incomplete",
                "file_path": row.file_path,
                "channel_index": row.channel_index,
                "start_col": row.start_col,
                "end_col": row.end_col,
                "cluster_id_removed": row.cluster_id,
                "rowid_removed": row.rowid,
            }
        )

    removed_duplicates = len(duplicate_rows_to_remove)
    removed_incomplete = len(incomplete)
    removed_rows = removed_duplicates + removed_incomplete

    if dry_run:
        print(f"[dry-run] Duplicate blocks: {removed_duplicates}")
        print(f"[dry-run] Incomplete blocks: {removed_incomplete}")
        return FixSummary(removed_rows, removed_duplicates, removed_incomplete)

    if removed_rows:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "action",
                    "file_path",
                    "channel_index",
                    "start_col",
                    "end_col",
                    "cluster_id_kept",
                    "cluster_id_removed",
                    "distance_kept",
                    "distance_removed",
                    "rowid_removed",
                ],
            )
            writer.writeheader()
            writer.writerows(log_entries)

    with connection:
        if removed_duplicates:
            ids = tuple(row.rowid for row in duplicate_rows_to_remove)
            connection.execute(
                f"DELETE FROM members WHERE rowid IN ({','.join('?' for _ in ids)})",
                ids,
            )
        if removed_incomplete:
            ids = tuple(row.rowid for row in incomplete)
            connection.execute(
                f"DELETE FROM members WHERE rowid IN ({','.join('?' for _ in ids)})",
                ids,
            )

        connection.execute(
            """
            WITH counts AS (
                SELECT cluster_id, COUNT(*) AS total
                FROM members
                GROUP BY cluster_id
            )
            UPDATE clusters
            SET member_count = COALESCE(
                (SELECT total FROM counts WHERE counts.cluster_id = clusters.id),
                0
            )
            """
        )
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_members_unique_block "
            "ON members(file_path, channel_index, start_col, end_col)"
        )

    return FixSummary(removed_rows, removed_duplicates, removed_incomplete)


def main() -> None:
    args = parse_args()
    if not args.registry.exists():
        raise FileNotFoundError(f"Registry not found: {args.registry}")

    with sqlite3.connect(args.registry) as connection:
        members = fetch_members(connection)
        duplicates, incomplete = classify_rows(members, args.tolerance)
        cluster_sizes = fetch_cluster_sizes(connection)
        summary = apply_corrections(
            connection,
            duplicates,
            incomplete,
            cluster_sizes,
            args.log_path,
            args.dry_run,
        )

    print(f"Removed rows: {summary.removed_rows}")
    print(f"  duplicates: {summary.removed_duplicates}")
    print(f"  incomplete: {summary.removed_incomplete}")
    if args.dry_run:
        print("No changes were written (dry-run).")
    else:
        print(f"Logged details to: {args.log_path}")


if __name__ == "__main__":
    main()
