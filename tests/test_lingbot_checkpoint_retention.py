from pathlib import Path

from kuavo_train.lingbot.checkpoint_retention import (
    checkpoint_is_complete,
    prepare_checkpoint_slot,
    prune_old_checkpoints,
)


def _make_checkpoint(root: Path, step: int, world_size: int = 1) -> Path:
    checkpoint = root / f"global_step_{step}"
    (checkpoint / "model").mkdir(parents=True)
    (checkpoint / "optimizer").mkdir()
    (checkpoint / "extra_state").mkdir()
    (checkpoint / "model" / ".metadata").touch()
    (checkpoint / "optimizer" / ".metadata").touch()
    for rank in range(world_size):
        (checkpoint / "extra_state" / f"extra_state_rank_{rank}.pt").touch()
    return checkpoint


def test_keep_one_frees_loaded_checkpoint_before_replacement(tmp_path: Path) -> None:
    old = _make_checkpoint(tmp_path, 500)
    partial = tmp_path / "global_step_1000"
    (partial / "model").mkdir(parents=True)

    removed = prepare_checkpoint_slot(tmp_path, 1000, keep=1, world_size=1)

    assert removed == [partial, old]
    assert not partial.exists()
    assert not old.exists()


def test_keep_two_retains_one_rollback_checkpoint_while_saving(
    tmp_path: Path,
) -> None:
    old_500 = _make_checkpoint(tmp_path, 500)
    old_1000 = _make_checkpoint(tmp_path, 1000)

    removed = prepare_checkpoint_slot(tmp_path, 1500, keep=2, world_size=1)

    assert removed == [old_500]
    assert not old_500.exists()
    assert checkpoint_is_complete(old_1000, world_size=1)


def test_post_save_pruning_keeps_newest_complete_checkpoint(
    tmp_path: Path,
) -> None:
    old = _make_checkpoint(tmp_path, 500)
    latest = _make_checkpoint(tmp_path, 1000)

    removed = prune_old_checkpoints(tmp_path, keep=1, world_size=1)

    assert removed == [old]
    assert not old.exists()
    assert checkpoint_is_complete(latest, world_size=1)
