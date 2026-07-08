"""Compatibility helpers for locally saved Kuavo policy configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

import draccus
from huggingface_hub.constants import CONFIG_NAME


T = TypeVar("T")


def decode_local_policy_config(
    config_cls: type[T], pretrained_path: str | Path, expected_type: str
) -> T | None:
    """Decode old/new local configs and recover from empty epoch configs."""

    model_path = Path(pretrained_path).expanduser()
    if not model_path.is_dir():
        return None

    candidates = [model_path / CONFIG_NAME]
    candidates.extend(parent / CONFIG_NAME for parent in list(model_path.parents)[:4])
    for config_path in candidates:
        if not config_path.is_file():
            continue
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict) or not raw.get("input_features") or not raw.get("output_features"):
            continue

        declared_type = raw.pop("type", None)
        if declared_type not in (None, expected_type):
            continue
        return draccus.decode(config_cls, raw)

    raise FileNotFoundError(
        f"No usable {expected_type!r} {CONFIG_NAME} found in {model_path} or its run directory."
    )
