"""Global cluster registry management CLI."""
from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# SQLite helpers
# ──────────────────────────────────────────────────────────────────────────────


def _serialize_array(array: np.ndarray) -> bytes:
    return pickle.dumps(np.asarray(array), protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize_array(blob: bytes) -> np.ndarray:
    return pickle.loads(blob)


def _ensure_schema(connection: sqlite3.Connection) -> None:
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dims TEXT NOT NULL,
            window INTEGER NOT NULL,
            exemplar BLOB NOT NULL,
            member_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS channel_clusters (
            file_path TEXT NOT NULL,
            channel_index INTEGER NOT NULL,
            local_cluster_id INTEGER NOT NULL,
            global_cluster_id INTEGER NOT NULL,
            split TEXT NOT NULL,
            distance REAL NOT NULL,
            dims TEXT NOT NULL,
            window INTEGER NOT NULL,
            PRIMARY KEY (file_path, channel_index, local_cluster_id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            channel_index INTEGER NOT NULL,
            block_index INTEGER NOT NULL,
            start_col INTEGER NOT NULL,
            end_col INTEGER NOT NULL,
            split TEXT NOT NULL,
            distance REAL NOT NULL,
            source_pickle TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(file_path, channel_index, block_index),
            FOREIGN KEY(cluster_id) REFERENCES clusters(id)
        )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_members_cluster ON members(cluster_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_channel_clusters_global ON channel_clusters(global_cluster_id)")
    connection.commit()


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class LocalCluster:
    file_path: Path
    channel_index: int
    local_id: int
    dims: np.ndarray
    window: int
    exemplar: np.ndarray
    member_blocks: List[int]
    intervals: np.ndarray
    split: str
    pickle_path: Path


@dataclass
class GlobalCluster:
    cluster_id: int
    dims: np.ndarray
    window: int
    exemplar: np.ndarray


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────


def _z_normalize(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-8:
        return arr - mean
    return (arr - mean) / std


def _flatten_subsequence(snippet: np.ndarray) -> np.ndarray:
    return _z_normalize(snippet.ravel())


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(np.linalg.norm(a - b) / np.sqrt(a.size))


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _iter_cluster_pickles(clusters_dir: Path) -> Iterator[Path]:
    for path in sorted(clusters_dir.glob("**/*.pkl")):
        if path.is_file():
            yield path


def _gather_local_clusters(pickle_path: Path, payload: dict, split: str) -> List[LocalCluster]:
    meta = payload.get("meta", {})
    channel_index = int(meta.get("channel_index", 0))
    intervals = np.asarray(payload.get("chirp_intervals"), dtype=np.int64)
    cluster_exemplars = payload.get("cluster_exemplars", [])
    block_cluster_ids = np.asarray(payload.get("block_cluster_ids", []), dtype=np.int32)

    clusters: List[LocalCluster] = []
    for exemplar in cluster_exemplars:
        local_id = int(exemplar["id"])
        dims = np.asarray(exemplar.get("dims", []), dtype=np.int32)
        if dims.size == 0:
            dims = np.arange(exemplar["features"].shape[0], dtype=np.int32)
        window = int(exemplar["features"].shape[1])
        exemplar_arr = np.asarray(exemplar["features"], dtype=np.float64)
        members = np.where(block_cluster_ids == local_id)[0].tolist()
        clusters.append(
            LocalCluster(
                file_path=Path(meta.get("file_path", meta.get("source_path", "")) or pickle_path),
                channel_index=channel_index,
                local_id=local_id,
                dims=dims,
                window=window,
                exemplar=exemplar_arr,
                member_blocks=members,
                intervals=intervals,
                split=split,
                pickle_path=pickle_path,
            )
        )
    return clusters


def _load_global_clusters(connection: sqlite3.Connection) -> List[GlobalCluster]:
    cursor = connection.cursor()
    cursor.execute("SELECT id, dims, window, exemplar FROM clusters")
    clusters: List[GlobalCluster] = []
    for cluster_id, dims_json, window, exemplar_blob in cursor.fetchall():
        dims = np.asarray(json.loads(dims_json), dtype=np.int32)
        exemplar = _deserialize_array(exemplar_blob)
        clusters.append(GlobalCluster(cluster_id=cluster_id, dims=dims, window=int(window), exemplar=exemplar))
    return clusters


def _match_global_cluster(
    global_clusters: List[GlobalCluster],
    local_cluster: LocalCluster,
    threshold: float,
) -> Optional[tuple[int, float]]:
    candidate_vector = _flatten_subsequence(local_cluster.exemplar)
    for global_cluster in global_clusters:
        if global_cluster.window != local_cluster.window:
            continue
        if not np.array_equal(global_cluster.dims, local_cluster.dims):
            continue
        global_vector = _flatten_subsequence(global_cluster.exemplar)
        distance = _euclidean_distance(candidate_vector, global_vector)
        if distance <= threshold:
            return global_cluster.cluster_id, distance
    return None


def _create_global_cluster(
    connection: sqlite3.Connection,
    local_cluster: LocalCluster,
) -> int:
    cursor = connection.cursor()
    dims_json = json.dumps(local_cluster.dims.tolist())
    exemplar_blob = _serialize_array(local_cluster.exemplar)
    cursor.execute(
        """
        INSERT INTO clusters (dims, window, exemplar, member_count)
        VALUES (?, ?, ?, 0)
        """,
        (dims_json, local_cluster.window, exemplar_blob),
    )
    cluster_id = int(cursor.lastrowid)
    connection.commit()
    return cluster_id


def _upsert_channel_cluster(
    connection: sqlite3.Connection,
    cluster_id: int,
    local_cluster: LocalCluster,
    distance: float,
) -> None:
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO channel_clusters
        (file_path, channel_index, local_cluster_id, global_cluster_id, split, distance, dims, window)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(local_cluster.file_path),
            local_cluster.channel_index,
            local_cluster.local_id,
            cluster_id,
            local_cluster.split,
            float(distance),
            json.dumps(local_cluster.dims.tolist()),
            local_cluster.window,
        ),
    )
    connection.commit()


def _upsert_member_records(
    connection: sqlite3.Connection,
    cluster_id: int,
    local_cluster: LocalCluster,
    payload: dict,
) -> None:
    block_cluster_ids = np.asarray(payload.get("block_cluster_ids", []), dtype=np.int32)
    global_assignments = payload.get("global_assignments")
    if global_assignments is None or global_assignments.shape[0] != block_cluster_ids.shape[0]:
        global_assignments = np.full(block_cluster_ids.shape, -1, dtype=np.int32)

    cursor = connection.cursor()
    members_inserted = 0
    for block_index in local_cluster.member_blocks:
        start, end = map(int, local_cluster.intervals[block_index])
        cursor.execute(
            """
            INSERT OR IGNORE INTO members
            (cluster_id, file_path, channel_index, block_index, start_col, end_col, split, distance, source_pickle)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cluster_id,
                str(local_cluster.file_path),
                local_cluster.channel_index,
                int(block_index),
                start,
                end,
                local_cluster.split,
                0.0,
                str(local_cluster.pickle_path),
            ),
        )
        if cursor.rowcount:
            members_inserted += 1
        global_assignments[block_index] = cluster_id

    if members_inserted:
        cursor.execute(
            """
            UPDATE clusters
            SET member_count = member_count + ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (members_inserted, cluster_id),
        )
    payload["global_assignments"] = global_assignments.astype(np.int32)
    connection.commit()


def _save_pickle(path: Path, payload: dict) -> None:
    with path.open("wb") as fh:
        pickle.dump(payload, fh)


# ──────────────────────────────────────────────────────────────────────────────
# CLI actions
# ──────────────────────────────────────────────────────────────────────────────


def command_sync(args: argparse.Namespace) -> None:
    registry_path = Path(args.registry)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(registry_path)
    _ensure_schema(connection)

    clusters_dir = Path(args.clusters_dir)
    if not clusters_dir.exists():
        raise SystemExit(f"[ERROR] clusters_dir not found: {clusters_dir}")

    if args.file_list:
        wanted = {Path(line.strip()).stem for line in Path(args.file_list).read_text().splitlines() if line.strip()}
    else:
        wanted = None

    global_clusters_cache = _load_global_clusters(connection)

    processed_files = 0
    for pickle_path in _iter_cluster_pickles(clusters_dir):
        if wanted and pickle_path.stem.split("_ch")[0] not in wanted:
            continue

        payload = _load_pickle(pickle_path)
        local_clusters = _gather_local_clusters(pickle_path, payload, args.split)
        if not local_clusters:
            continue

        modified = False
        for local_cluster in local_clusters:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT global_cluster_id FROM channel_clusters
                WHERE file_path = ? AND channel_index = ? AND local_cluster_id = ?
                """,
                (str(local_cluster.file_path), local_cluster.channel_index, local_cluster.local_id),
            )
            row = cursor.fetchone()
            if row and not args.rematch:
                cluster_id = int(row[0])
                distance = float(0.0)
            else:
                match = _match_global_cluster(global_clusters_cache, local_cluster, args.match_threshold)
                if match is None:
                    cluster_id = _create_global_cluster(connection, local_cluster)
                    global_clusters_cache = _load_global_clusters(connection)
                    distance = 0.0
                else:
                    cluster_id, distance = match
                _upsert_channel_cluster(connection, cluster_id, local_cluster, distance)

            _upsert_member_records(connection, cluster_id, local_cluster, payload)
            modified = True

        if modified:
            meta = payload.get("meta", {})
            meta.setdefault("split", args.split)
            payload["meta"] = meta
            _save_pickle(pickle_path, payload)
            processed_files += 1

    print(f"Synchronized clusters for {processed_files} channel pickle(s). Registry: {registry_path}")


def _load_registry_clusters_for_reclassify(connection: sqlite3.Connection) -> List[GlobalCluster]:
    return _load_global_clusters(connection)


def _slide_distance(snippet: np.ndarray, exemplar: np.ndarray) -> float:
    dims, total_cols = snippet.shape
    _, window = exemplar.shape
    if total_cols < window:
        return float("inf")
    best = float("inf")
    exemplar_vec = _flatten_subsequence(exemplar)
    for start in range(0, total_cols - window + 1):
        window_slice = snippet[:, start : start + window]
        candidate_vec = _flatten_subsequence(window_slice)
        dist = _euclidean_distance(candidate_vec, exemplar_vec)
        if dist < best:
            best = dist
    return best


def command_reclassify(args: argparse.Namespace) -> None:
    registry_path = Path(args.registry)
    if not registry_path.exists():
        raise SystemExit(f"[ERROR] registry not found: {registry_path}")
    connection = sqlite3.connect(registry_path)
    clusters = _load_registry_clusters_for_reclassify(connection)
    if not clusters:
        print("[WARN] registry is empty; nothing to reclassify")
        return

    clusters_dir = Path(args.clusters_dir)
    threshold = args.match_threshold

    processed = 0
    for pickle_path in _iter_cluster_pickles(clusters_dir):
        payload = _load_pickle(pickle_path)
        features = np.asarray(payload.get("features", []), dtype=np.float64)
        intervals = np.asarray(payload.get("chirp_intervals", []), dtype=np.int64)
        block_cluster_ids = np.asarray(payload.get("block_cluster_ids", []), dtype=np.int32)
        global_assignments = payload.get("global_assignments")
        if global_assignments is None or global_assignments.shape[0] != block_cluster_ids.shape[0]:
            global_assignments = np.full(block_cluster_ids.shape, -1, dtype=np.int32)

        meta = payload.get("meta", {})
        file_path = Path(meta.get("file_path", meta.get("source_path", pickle_path)))
        channel_index = int(meta.get("channel_index", 0))
        split = meta.get("split", "unspecified")

        changed = False
        new_members: List[tuple[int, int, int, int]] = []  # (cluster_id, block_index, start, end)
        for block_index, (cluster_id, global_id) in enumerate(zip(block_cluster_ids, global_assignments)):
            if cluster_id >= 0 or global_id >= 0:
                continue
            start, end = intervals[block_index]
            snippet = features[:, start:end]
            if snippet.size == 0:
                continue
            best_match = None
            best_distance = float("inf")
            for cluster in clusters:
                dims = cluster.dims
                window = cluster.window
                candidate = snippet[np.asarray(dims, dtype=int), :]
                dist = _slide_distance(candidate, cluster.exemplar)
                if dist < best_distance and dist <= threshold:
                    best_distance = dist
                    best_match = cluster.cluster_id
            if best_match is not None:
                global_assignments[block_index] = best_match
                new_members.append((best_match, block_index, int(start), int(end)))
                changed = True
        if changed:
            payload["global_assignments"] = global_assignments.astype(np.int32)
            _save_pickle(pickle_path, payload)
            cursor = connection.cursor()
            for cluster_id, block_index, start, end in new_members:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO members
                    (cluster_id, file_path, channel_index, block_index, start_col, end_col, split, distance, source_pickle)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cluster_id,
                        str(file_path),
                        channel_index,
                        block_index,
                        start,
                        end,
                        split,
                        0.0,
                        str(pickle_path),
                    ),
                )
                if cursor.rowcount:
                    cursor.execute(
                        "UPDATE clusters SET member_count = member_count + 1, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (cluster_id,),
                    )
            connection.commit()
            processed += 1

    print(f"Reclassified noise blocks in {processed} channel pickle(s). Registry: {registry_path}")


def command_info(args: argparse.Namespace) -> None:
    registry_path = Path(args.registry)
    if not registry_path.exists():
        raise SystemExit(f"[ERROR] registry not found: {registry_path}")
    connection = sqlite3.connect(registry_path)
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM clusters")
    num_clusters = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM members")
    num_members = cursor.fetchone()[0]
    cursor.execute("SELECT split, COUNT(*) FROM members GROUP BY split ORDER BY split")
    splits = cursor.fetchall()

    print(f"Registry: {registry_path}")
    print(f"  clusters : {num_clusters}")
    print(f"  members  : {num_members}")
    if splits:
        print("  splits:")
        for split, count in splits:
            print(f"    {split or 'unspecified'}: {count}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI parser
# ──────────────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Global cluster registry management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync", help="Sync per-channel clusters into the registry")
    sync_parser.add_argument("--registry", type=str, default="cluster_registry.sqlite")
    sync_parser.add_argument("--clusters_dir", type=str, default="cluster")
    sync_parser.add_argument("--file_list", type=str, default=None)
    sync_parser.add_argument("--split", type=str, default="unspecified")
    sync_parser.add_argument("--match-threshold", type=float, default=0.5)
    sync_parser.add_argument("--rematch", action="store_true", help="Force rematching even if mapping exists")
    sync_parser.set_defaults(func=command_sync)

    recl_parser = subparsers.add_parser("reclassify", help="Attempt to assign noise blocks")
    recl_parser.add_argument("--registry", type=str, default="cluster_registry.sqlite")
    recl_parser.add_argument("--clusters_dir", type=str, default="cluster")
    recl_parser.add_argument("--match-threshold", type=float, default=0.5)
    recl_parser.set_defaults(func=command_reclassify)

    info_parser = subparsers.add_parser("info", help="Show registry summary")
    info_parser.add_argument("--registry", type=str, default="cluster_registry.sqlite")
    info_parser.set_defaults(func=command_info)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.func(args)


if __name__ == "__main__":
    main()
