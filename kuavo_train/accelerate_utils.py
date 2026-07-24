"""Configuration helpers for deterministic Accelerate trainer startup."""

from __future__ import annotations

from typing import Any


SUPPORTED_MIXED_PRECISION = {"no", "fp16", "bf16", "fp8"}


def resolve_accelerate_options(training_cfg: Any, policy_cfg: Any) -> dict[str, Any]:
    mixed_precision = str(
        training_cfg.get(
            "mixed_precision",
            "fp16" if policy_cfg.get("use_amp", False) else "no",
        )
    )
    if mixed_precision not in SUPPORTED_MIXED_PRECISION:
        raise ValueError(f"Unsupported mixed precision mode: {mixed_precision!r}")
    return {
        "mixed_precision": mixed_precision,
        "ddp_find_unused_parameters": bool(
            training_cfg.get("ddp_find_unused_parameters", True)
        ),
        "allow_tf32": bool(training_cfg.get("allow_tf32", True)),
    }


def dataloader_worker_options(training_cfg: Any) -> dict[str, Any]:
    workers = int(training_cfg.get("num_workers", 0))
    return {
        "prefetch_factor": int(training_cfg.get("prefetch_factor", 2)) if workers > 0 else None,
        "persistent_workers": (
            bool(training_cfg.get("persistent_workers", True)) if workers > 0 else False
        ),
    }
