"""Weighted LeRobot dataset mixtures shared by classic policy trainers."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class DatasetSource:
    name: str
    repo_id: str
    root: str
    weight: float


def load_dataset_sources(env_name: str = "KUAVO_DATASET_MIX_JSON") -> tuple[DatasetSource, ...]:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return ()
    payload = json.loads(raw)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{env_name} must be a non-empty JSON list")
    weights = [float(item["weight"]) for item in payload]
    if any(weight <= 0 for weight in weights):
        raise ValueError("Dataset mixture weights must be positive")
    total = sum(weights)
    sources = tuple(
        DatasetSource(
            name=str(item.get("name") or f"source_{index:02d}"),
            repo_id=str(item["repo_id"]),
            root=str(Path(item["root"]).expanduser()),
            weight=weight / total,
        )
        for index, (item, weight) in enumerate(zip(payload, weights, strict=True), 1)
    )
    if len({source.name for source in sources}) != len(sources):
        raise ValueError("Dataset mixture source names must be unique")
    return sources


def validate_metadata_compatibility(metadata: Sequence[Any]) -> None:
    if not metadata:
        raise ValueError("At least one dataset metadata object is required")
    reference = metadata[0]
    reference_features = {
        key: (tuple(value.get("shape", ())), value.get("dtype"), value.get("names"))
        for key, value in reference.features.items()
        if key in {"observation.state", "action"} or key.startswith("observation.images.")
    }
    for current in metadata[1:]:
        current_features = {
            key: (tuple(value.get("shape", ())), value.get("dtype"), value.get("names"))
            for key, value in current.features.items()
            if key in {"observation.state", "action"} or key.startswith("observation.images.")
        }
        if current_features != reference_features:
            raise ValueError("Dataset mixture feature schema mismatch")
        if current.fps != reference.fps:
            raise ValueError(
                f"Dataset mixture FPS mismatch: {reference.fps} != {current.fps}"
            )
        if set(current.camera_keys) != set(reference.camera_keys):
            raise ValueError("Dataset mixture camera keys mismatch")
        if current.info.get("robot_type") != reference.info.get("robot_type"):
            raise ValueError("Dataset mixture robot_type mismatch")


def combine_dataset_stats(stats: Sequence[dict], weights: Sequence[float]) -> dict:
    """Combine source stats for the configured sampling distribution."""
    if not stats or len(stats) != len(weights):
        raise ValueError("Stats and weights must be non-empty and aligned")
    normalized = np.asarray(weights, dtype=np.float64)
    if np.any(normalized <= 0):
        raise ValueError("Stats weights must be positive")
    normalized /= normalized.sum()
    common_keys = set(stats[0])
    for source in stats[1:]:
        common_keys &= set(source)
    combined: dict[str, dict[str, np.ndarray]] = {}
    for key in common_keys:
        fields = stats[0][key]
        if not isinstance(fields, dict) or "mean" not in fields:
            continue
        means = np.stack([np.asarray(source[key]["mean"]) for source in stats])
        stds = np.stack([np.asarray(source[key]["std"]) for source in stats])
        mean = np.tensordot(normalized, means, axes=1)
        second_moment = np.tensordot(
            normalized, np.square(stds) + np.square(means), axes=1
        )
        result = {
            "mean": mean,
            "std": np.sqrt(np.maximum(second_moment - np.square(mean), 0.0)),
            "min": np.min(
                np.stack([np.asarray(source[key]["min"]) for source in stats]), axis=0
            ),
            "max": np.max(
                np.stack([np.asarray(source[key]["max"]) for source in stats]), axis=0
            ),
            "count": np.sum(
                np.stack([np.asarray(source[key]["count"]) for source in stats]), axis=0
            ),
        }
        for quantile in ("q01", "q10", "q50", "q90", "q99"):
            if all(quantile in source[key] for source in stats):
                values = np.stack(
                    [np.asarray(source[key][quantile]) for source in stats]
                )
                result[quantile] = np.tensordot(normalized, values, axes=1)
        combined[key] = result
    return combined


def _trim_episode_tails(dataset: Dataset, drop_n_last_frames: int) -> Dataset:
    if drop_n_last_frames <= 0:
        return dataset
    episodes = dataset.meta.episodes
    indices: list[int] = []
    for start, end in zip(
        episodes["dataset_from_index"],
        episodes["dataset_to_index"],
        strict=True,
    ):
        stop = max(int(start), int(end) - drop_n_last_frames)
        indices.extend(range(int(start), stop))
    return Subset(dataset, indices)


class VirtualWeightedDataset(Dataset):
    """Uniform virtual index space whose source slot counts follow configured weights."""

    def __init__(
        self,
        datasets: Sequence[Dataset],
        sources: Sequence[DatasetSource],
        *,
        metadata: Any,
    ):
        if not datasets or len(datasets) != len(sources):
            raise ValueError("One non-empty dataset is required per mixture source")
        if any(len(dataset) == 0 for dataset in datasets):
            raise ValueError("Dataset mixture sources must be non-empty")
        counts = virtual_source_counts(
            [len(dataset) for dataset in datasets],
            sources,
        )
        self.datasets = tuple(datasets)
        self.sources = tuple(sources)
        self.counts = tuple(counts)
        self.cumulative_sizes = np.cumsum(counts).tolist()
        self.meta = metadata

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def source_index(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return bisect.bisect_right(self.cumulative_sizes, index)

    def __getitem__(self, index: int):
        source_index = self.source_index(index)
        start = 0 if source_index == 0 else self.cumulative_sizes[source_index - 1]
        local_index = (index - start) % len(self.datasets[source_index])
        return self.datasets[source_index][local_index]

    @property
    def realized_weights(self) -> tuple[float, ...]:
        total = len(self)
        return tuple(count / total for count in self.counts)


def build_virtual_mixture(
    datasets: Sequence[Dataset],
    sources: Sequence[DatasetSource],
    *,
    metadata: Any,
    drop_n_last_frames: int = 0,
) -> VirtualWeightedDataset:
    trimmed = [
        _trim_episode_tails(dataset, drop_n_last_frames) for dataset in datasets
    ]
    return VirtualWeightedDataset(trimmed, sources, metadata=metadata)


def virtual_source_counts(
    lengths: Sequence[int], sources: Sequence[DatasetSource]
) -> tuple[int, ...]:
    """Return source slot counts; each source is traversed at least once per epoch."""
    if not lengths or len(lengths) != len(sources):
        raise ValueError("One positive length is required per mixture source")
    if any(length <= 0 for length in lengths):
        raise ValueError("Dataset mixture lengths must be positive")
    virtual_size = max(
        math.ceil(length / source.weight)
        for length, source in zip(lengths, sources, strict=True)
    )
    return tuple(max(1, round(virtual_size * source.weight)) for source in sources)
