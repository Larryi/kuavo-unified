"""Disk-bounded retention helpers for LingBot distributed checkpoints."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"global_step_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def checkpoint_is_complete(path: Path, world_size: int) -> bool:
    return (
        (path / "model" / ".metadata").is_file()
        and (path / "optimizer" / ".metadata").is_file()
        and all(
            (path / "extra_state" / f"extra_state_rank_{rank}.pt").is_file()
            for rank in range(world_size)
        )
    )


def prepare_checkpoint_slot(
    checkpoint_root: str | Path,
    upcoming_step: int,
    keep: int,
    world_size: int,
) -> list[Path]:
    """Free space before saving while preserving the requested final count."""

    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    removed: list[Path] = []
    target = root / f"global_step_{upcoming_step}"

    # A failed attempt at this same step cannot be resumed and may occupy most
    # of the volume. A retry must start from a clean DCP directory.
    if target.exists() and not checkpoint_is_complete(target, world_size):
        shutil.rmtree(target)
        removed.append(target)

    if keep <= 0:
        return removed

    complete = sorted(
        (
            path
            for path in root.glob("global_step_*")
            if path != target and checkpoint_is_complete(path, world_size)
        ),
        key=checkpoint_step,
    )
    retain_before_save = max(keep - 1, 0)
    stale = complete if retain_before_save == 0 else complete[:-retain_before_save]
    for path in stale:
        shutil.rmtree(path)
        removed.append(path)
    return removed


def prune_old_checkpoints(
    checkpoint_root: str | Path,
    keep: int,
    world_size: int,
) -> list[Path]:
    if keep <= 0:
        return []
    root = Path(checkpoint_root)
    complete = sorted(
        (
            path
            for path in root.glob("global_step_*")
            if checkpoint_is_complete(path, world_size)
        ),
        key=checkpoint_step,
    )
    removed = complete[:-keep]
    for path in removed:
        shutil.rmtree(path)
    return removed
