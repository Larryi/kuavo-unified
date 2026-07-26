from pathlib import Path

import pytest

from kuavo_train.checkpoint_retention import prune_periodic_epoch_checkpoints


def test_prunes_only_old_periodic_epoch_snapshots(tmp_path: Path):
    for name in ("epoch10", "epoch20", "epoch30", "epochbest", "epochlatest"):
        (tmp_path / name).mkdir()

    removed = prune_periodic_epoch_checkpoints(tmp_path, keep_last=1)

    assert [path.name for path in removed] == ["epoch10", "epoch20"]
    assert (tmp_path / "epoch30").is_dir()
    assert (tmp_path / "epochbest").is_dir()
    assert (tmp_path / "epochlatest").is_dir()


def test_zero_removes_all_periodic_snapshots(tmp_path: Path):
    (tmp_path / "epoch10").mkdir()
    (tmp_path / "epochbest").mkdir()

    prune_periodic_epoch_checkpoints(tmp_path, keep_last=0)

    assert not (tmp_path / "epoch10").exists()
    assert (tmp_path / "epochbest").is_dir()


def test_negative_retention_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="non-negative"):
        prune_periodic_epoch_checkpoints(tmp_path, keep_last=-1)
