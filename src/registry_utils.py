"""Utilities for reading the global cluster registry and channel cluster files."""

from __future__ import annotations

import json
import pickle
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np


@dataclass(frozen=True)
class GlobalPrototype:
    """Prototype for a global cluster."""

    id: int
    m: int
    dims: np.ndarray
    exemplar: np.ndarray


@dataclass(frozen=True)
class BlockRef:
    """Reference to a block belonging to a global cluster."""

    file_path: Path
    channel_index: int
    start: int
    end: int
    global_id: int
    pickle_path: Path
    split: str


def _deserialize_array(blob: bytes) -> np.ndarray:
    return pickle.loads(blob)


def load_registry(registry_path: str) -> Dict[int, List[GlobalPrototype]]:
    """Return dictionary mapping window length (m) to prototypes."""

    registry = Path(registry_path)
    if not registry.exists():
        raise FileNotFoundError(f"registry not found: {registry}")

    connection = sqlite3.connect(registry)
    cursor = connection.cursor()
    cursor.execute("SELECT id, dims, window, exemplar FROM clusters")

    clusters_by_m: Dict[int, List[GlobalPrototype]] = {}
    for cluster_id, dims_json, window, exemplar_blob in cursor.fetchall():
        dims = np.asarray(json.loads(dims_json), dtype=np.int32)
        exemplar = _deserialize_array(exemplar_blob)
        if dims.size == 0:
            dims = np.arange(exemplar.shape[0], dtype=np.int32)
        proto = GlobalPrototype(
            id=int(cluster_id),
            m=int(window),
            dims=dims,
            exemplar=np.asarray(exemplar, dtype=np.float32),
        )
        clusters_by_m.setdefault(proto.m, []).append(proto)

    connection.close()
    return clusters_by_m


def iter_members(registry_path: str) -> Iterator[BlockRef]:
    """Yield block references registered in the global registry."""

    registry = Path(registry_path)
    if not registry.exists():
        raise FileNotFoundError(f"registry not found: {registry}")

    connection = sqlite3.connect(registry)
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT m.cluster_id,
               m.file_path,
               m.channel_index,
               m.block_index,
               m.start_col,
               m.end_col,
               m.source_pickle,
               m.split
        FROM members AS m
        ORDER BY m.cluster_id
        """
    )

    seen = set()
    for (
        cluster_id,
        file_path,
        channel_index,
        block_index,
        start_col,
        end_col,
        pickle_path,
        split,
    ) in cursor.fetchall():
        key = (file_path, channel_index, block_index)
        if key in seen:
            continue
        seen.add(key)
        yield BlockRef(
            file_path=Path(file_path),
            channel_index=int(channel_index),
            start=int(start_col),
            end=int(end_col),
            global_id=int(cluster_id),
            pickle_path=Path(pickle_path),
            split=split or "unspecified",
        )

    connection.close()


@lru_cache(maxsize=512)
def load_cluster_pickle(pickle_path: Path) -> dict:
    """Load and memoise a per-channel cluster pickle."""

    with pickle_path.open("rb") as fh:
        payload = pickle.load(fh)
    return payload

