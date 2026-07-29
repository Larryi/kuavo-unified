#!/usr/bin/env python3
"""Compute LingBot-v1 normalization statistics for one or more weighted datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from kuavo_train.dataset_mixture import VirtualWeightedDataset, load_dataset_sources
from lingbotvla.data.vla_data.base_dataset import VLADataset
from lingbotvla.utils.arguments import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    parse_args,
)
import lingbotvla.utils.normalize as normalize


@dataclass
class NormTrainingArguments(TrainingArguments):
    """Accept Kuavo trainer-only options while computing dataset statistics."""

    use_ema: bool = False
    ignore_depth: bool = False
    keep_last_checkpoints: int = 0

    def __post_init__(self) -> None:
        # The upstream LingBot arguments assume torchrun always populated
        # these variables. Norm computation is intentionally a single plain
        # Python process, so declare that topology before upstream validation.
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
        # The source YAML describes the later multi-GPU training job. This
        # helper is one process and only consumes micro_batch_size/chunk_size,
        # so reconcile global_batch_size solely for upstream validation.
        self.global_batch_size = (
            self.micro_batch_size * self.gradient_accumulation_steps
        )
        super().__post_init__()


@dataclass
class Arguments:
    model: ModelArguments = field(default_factory=ModelArguments)
    data: DataArguments = field(default_factory=DataArguments)
    train: NormTrainingArguments = field(default_factory=NormTrainingArguments)


def main() -> int:
    args = parse_args(Arguments)
    sources = load_dataset_sources()
    roots = [source.root for source in sources] if sources else [args.data.train_path]
    datasets = [
        VLADataset(
            repo_id=root,
            data_name=args.data.data_name,
            robot_config_root=args.data.robot_config_root,
            config=None,
            data_config=args.data,
            do_nomalize=False,
        )
        for root in roots
    ]
    dataset = (
        VirtualWeightedDataset(datasets, sources, metadata=None)
        if sources
        else datasets[0]
    )
    reference = datasets[0].feature_transform
    state_keys = reference.states
    action_keys = reference.actions
    delta_norm = reference.action_subtract_state
    for current in datasets[1:]:
        transform = current.feature_transform
        if (
            transform.states != state_keys
            or transform.actions != action_keys
            or transform.action_subtract_state != delta_norm
        ):
            raise ValueError("LingBot dataset mixture feature transforms do not match")

    stats = {
        key: normalize.RunningStats() for key in action_keys + state_keys
    }
    loader = DataLoader(
        dataset,
        batch_size=args.train.micro_batch_size,
        num_workers=args.data.num_workers,
        shuffle=False,
        drop_last=False,
    )
    for batch in tqdm(loader, desc="Computing weighted LingBot normalization stats"):
        for key in state_keys:
            values = np.asarray(batch[key])
            stats[key].update(values.reshape(-1, values.shape[-1]))
        for key in action_keys:
            values = (
                np.asarray(batch[key].reshape(batch[key].shape[0], -1))
                if delta_norm[key]
                else np.asarray(batch[key][:, 0])
            )
            stats[key].update(values.reshape(-1, values.shape[-1]))

    norm_stats = {
        key: running.get_statistics(
            chunk_size=args.train.chunk_size if delta_norm.get(key, False) else None
        )
        for key, running in stats.items()
    }
    counts = {running._count for running in stats.values()}
    if len(counts) != 1:
        raise ValueError(f"Normalization feature counts do not match: {sorted(counts)}")
    output = Path(args.data.norm_stats_file)
    normalize.save(output, norm_stats, counts.pop())
    if sources:
        print(
            "Mixture weights: "
            + json.dumps(
                {
                    source.repo_id: weight
                    for source, weight in zip(
                        sources, dataset.realized_weights, strict=True
                    )
                },
                sort_keys=True,
            )
        )
    print(f"Wrote normalization stats: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
