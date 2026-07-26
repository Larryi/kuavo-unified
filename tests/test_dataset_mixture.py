from types import SimpleNamespace

import numpy as np
import pytest
from torch.utils.data import Dataset

from kuavo_train.dataset_mixture import (
    DatasetSource,
    VirtualWeightedDataset,
    combine_dataset_stats,
    validate_metadata_compatibility,
    virtual_source_counts,
)


class RangeDataset(Dataset):
    def __init__(self, prefix: str, size: int):
        self.prefix = prefix
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.prefix, index


def sources(*weights):
    return tuple(
        DatasetSource(f"s{i}", f"owner/d{i}", f"/d{i}", weight)
        for i, weight in enumerate(weights)
    )


def test_virtual_mixture_matches_weights_and_covers_every_source():
    mix = VirtualWeightedDataset(
        [RangeDataset("a", 100), RangeDataset("b", 20)],
        sources(0.75, 0.25),
        metadata=None,
    )

    assert mix.counts == (100, 34)
    assert mix.realized_weights == pytest.approx((0.75, 0.25), abs=0.004)
    assert {mix[index] for index in range(len(mix)) if mix[index][0] == "a"} == {
        ("a", index) for index in range(100)
    }
    assert {mix[index] for index in range(len(mix)) if mix[index][0] == "b"} == {
        ("b", index) for index in range(20)
    }


def test_virtual_counts_drive_epoch_and_scheduler_length():
    assert virtual_source_counts([100, 20], sources(0.75, 0.25)) == (100, 34)


def test_combines_mean_and_variance_by_sampling_weight():
    stats = [
        {
            "observation.state": {
                "mean": np.array([0.0]),
                "std": np.array([1.0]),
                "min": np.array([-2.0]),
                "max": np.array([2.0]),
                "count": np.array([100]),
            }
        },
        {
            "observation.state": {
                "mean": np.array([10.0]),
                "std": np.array([2.0]),
                "min": np.array([5.0]),
                "max": np.array([15.0]),
                "count": np.array([20]),
            }
        },
    ]

    result = combine_dataset_stats(stats, [0.75, 0.25])["observation.state"]

    assert result["mean"] == pytest.approx([2.5])
    assert result["std"] == pytest.approx([4.52769257])
    assert result["min"] == pytest.approx([-2.0])
    assert result["max"] == pytest.approx([15.0])


def test_metadata_schema_mismatch_is_rejected():
    common = {
        "fps": 10,
        "camera_keys": ["observation.images.head_cam_h"],
        "info": {"robot_type": "kuavo"},
    }
    first = SimpleNamespace(
        **common,
        features={
            "observation.state": {"shape": (8,), "dtype": "float32"},
            "action": {"shape": (8,), "dtype": "float32"},
        },
    )
    second = SimpleNamespace(
        **common,
        features={
            "observation.state": {"shape": (16,), "dtype": "float32"},
            "action": {"shape": (16,), "dtype": "float32"},
        },
    )

    with pytest.raises(ValueError, match="schema mismatch"):
        validate_metadata_compatibility([first, second])
