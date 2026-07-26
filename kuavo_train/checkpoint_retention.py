"""Disk-bounded checkpoint retention helpers."""

from __future__ import annotations

from pathlib import Path
import re
import shutil


_PERIODIC_EPOCH = re.compile(r"^epoch([0-9]+)$")


def prune_periodic_epoch_checkpoints(output_directory: Path, keep_last: int) -> list[Path]:
    """Delete old ``epochN`` snapshots while preserving best/latest resume state."""
    if keep_last < 0:
        raise ValueError("keep_last must be non-negative")
    checkpoints = []
    for path in output_directory.iterdir():
        match = _PERIODIC_EPOCH.fullmatch(path.name)
        if path.is_dir() and match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()
    remove = checkpoints if keep_last == 0 else checkpoints[:-keep_last]
    removed = []
    for _, path in remove:
        shutil.rmtree(path)
        removed.append(path)
    return removed
