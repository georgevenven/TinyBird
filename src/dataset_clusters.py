"""Dataset utilities for training on clustered audio blocks."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from registry_utils import BlockRef, GlobalPrototype, iter_members, load_cluster_pickle, load_registry


@dataclass
class Sample:
    x: np.ndarray
    mask: np.ndarray
    cluster_id: int
    dims: np.ndarray
    window: int
    info: dict


class ClusterBalancedDataset(Dataset):
    """Sample blocks uniformly across global clusters."""

    def __init__(
        self,
        registry_path: str,
        clusters_dir: str,
        min_members_per_cluster: int = 1,
        split_filter: Iterable[str] | None = None,
        feature_cache_size: int = 256,
        lambda_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.registry_path = registry_path
        self.clusters_dir = clusters_dir
        self.min_members_per_cluster = min_members_per_cluster
        self.lambda_normalize = lambda_normalize

        self.prototypes_by_m: Dict[int, List[GlobalPrototype]] = load_registry(registry_path)
        self.prototype_by_id: Dict[int, GlobalPrototype] = {}
        for group in self.prototypes_by_m.values():
            for proto in group:
                self.prototype_by_id[proto.id] = proto

        allowed_splits = set(split_filter) if split_filter else None
        clusters: Dict[int, List[BlockRef]] = {}
        for block in iter_members(registry_path):
            if allowed_splits and block.split not in allowed_splits:
                continue
            clusters.setdefault(block.global_id, []).append(block)

        # Filter clusters that have enough members and for which we have prototypes.
        self.clusters: Dict[int, List[BlockRef]] = {}
        for gid, members in clusters.items():
            if gid not in self.prototype_by_id:
                continue
            uniq = {(ref.file_path, ref.channel_index, ref.start, ref.end): ref for ref in members}
            members = list(uniq.values())
            if len(members) >= self.min_members_per_cluster:
                self.clusters[gid] = members

        if not self.clusters:
            raise RuntimeError("No clusters available after filtering.")

        self.cluster_ids: List[int] = sorted(self.clusters.keys())
        self.feature_cache_size = feature_cache_size
        self._payload_cache: Dict[Tuple[str, int], dict] = {}

        # Determine feature dimensionality from first sample.
        first_cluster = self.cluster_ids[0]
        first_block = self.clusters[first_cluster][0]
        payload = self._load_payload(first_block)
        features = np.asarray(payload["features"], dtype=np.float32)
        self.feature_dim = features.shape[0]

    def __len__(self) -> int:
        return sum(len(members) for members in self.clusters.values())

    def __getitem__(self, index: int) -> Sample:
        cluster_id = random.choice(self.cluster_ids)
        members = self.clusters[cluster_id]
        block = random.choice(members)
        payload = self._load_payload(block)
        features = np.asarray(payload["features"], dtype=np.float32)
        start = block.start
        end = block.end
        snippet = features[:, start:end]
        snippet = np.nan_to_num(snippet, nan=0.0)

        if self.lambda_normalize:
            snippet = self._normalize(snippet)

        mask = np.ones(snippet.shape[1], dtype=np.float32)

        proto = self.prototype_by_id[cluster_id]
        info = {
            "file_path": str(block.file_path),
            "channel_index": block.channel_index,
            "start": start,
            "end": end,
            "pickle_path": str(block.pickle_path),
        }

        return Sample(
            x=snippet,
            mask=mask,
            cluster_id=cluster_id,
            dims=np.asarray(proto.dims, dtype=np.int32),
            window=int(proto.m),
            info=info,
        )

    def _load_payload(self, block: BlockRef) -> dict:
        key = (str(block.file_path), block.channel_index)
        if key in self._payload_cache:
            return self._payload_cache[key]
        payload = load_cluster_pickle(block.pickle_path)
        if len(self._payload_cache) >= self.feature_cache_size:
            self._payload_cache.pop(next(iter(self._payload_cache)))
        self._payload_cache[key] = payload
        return payload

    @staticmethod
    def _normalize(features: np.ndarray) -> np.ndarray:
        valid = np.isfinite(features)
        if not np.any(valid):
            return features
        mean = np.mean(features[valid])
        std = np.std(features[valid])
        if std < 1e-6:
            return features - mean
        return (features - mean) / std


def collate_blocks(batch: List[Sample]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[np.ndarray], List[dict]]:
    """Pad variable-length blocks and return tensors ready for training."""

    if not batch:
        raise ValueError("empty batch")

    feature_dim = batch[0].x.shape[0]
    lengths = [sample.x.shape[1] for sample in batch]
    max_len = max(lengths)

    xs = torch.zeros(len(batch), 1, feature_dim, max_len, dtype=torch.float32)
    masks = torch.zeros(len(batch), 1, 1, max_len, dtype=torch.float32)
    labels = torch.zeros(len(batch), dtype=torch.long)

    dims_masks: List[np.ndarray] = []
    infos: List[dict] = []

    for idx, sample in enumerate(batch):
        T = sample.x.shape[1]
        xs[idx, 0, :, :T] = torch.from_numpy(sample.x)
        masks[idx, 0, 0, :T] = torch.from_numpy(sample.mask)
        labels[idx] = int(sample.cluster_id)
        dims_masks.append(sample.dims)
        infos.append(sample.info)

    return xs, masks, labels, dims_masks, infos
