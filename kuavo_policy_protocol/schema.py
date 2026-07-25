"""Validation for the Kuavo observation/action wire contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


HEAD_CAMERA_KEY = "observation.images.head_cam_h"
WRIST_CAMERA_KEYS = (
    "observation.images.wrist_cam_l",
    "observation.images.wrist_cam_r",
)
STATE_KEY = "observation.state"
PROMPT_KEY = "prompt"


@dataclass(frozen=True)
class ObservationSchema:
    state_dim: int | None = None
    require_head_camera: bool = True
    require_wrist_camera: bool = True
    require_prompt: bool = True


@dataclass(frozen=True)
class PolicyMetadata:
    backend: str = "unknown"
    checkpoint_format: str = "unknown"
    action_semantics: str = "unknown"
    action_dim: int | None = None
    action_horizon: int | None = None
    camera_keys: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "PolicyMetadata":
        raw = dict(value or {})
        known = {
            "backend",
            "checkpoint_format",
            "action_semantics",
            "action_dim",
            "action_horizon",
            "camera_keys",
        }
        return cls(
            backend=str(raw.get("backend", "unknown")),
            checkpoint_format=str(raw.get("checkpoint_format", "unknown")),
            action_semantics=str(raw.get("action_semantics", "unknown")),
            action_dim=_optional_positive_int(raw.get("action_dim"), "action_dim"),
            action_horizon=_optional_positive_int(raw.get("action_horizon"), "action_horizon"),
            camera_keys=tuple(str(key) for key in raw.get("camera_keys", ())),
            extra={key: value for key, value in raw.items() if key not in known},
        )


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _validate_image(value: Any, key: str) -> None:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"{key} must be an unbatched RGB image, got shape {image.shape}")
    if image.shape[0] != 3 and image.shape[-1] != 3:
        raise ValueError(f"{key} must be CHW or HWC RGB, got shape {image.shape}")
    if image.dtype.kind not in ("u", "i", "f"):
        raise ValueError(f"{key} must have a numeric dtype, got {image.dtype}")
    if not np.all(np.isfinite(image)):
        raise ValueError(f"{key} contains non-finite values")


def validate_observation(
    observation: Mapping[str, Any],
    schema: ObservationSchema | None = None,
) -> None:
    schema = schema or ObservationSchema()
    if schema.require_head_camera:
        if HEAD_CAMERA_KEY not in observation:
            raise KeyError(f"Missing required observation key: {HEAD_CAMERA_KEY}")
        _validate_image(observation[HEAD_CAMERA_KEY], HEAD_CAMERA_KEY)
    for key in WRIST_CAMERA_KEYS:
        if key in observation:
            _validate_image(observation[key], key)
    if schema.require_wrist_camera and not any(key in observation for key in WRIST_CAMERA_KEYS):
        raise KeyError(f"At least one wrist camera is required: {WRIST_CAMERA_KEYS}")
    if STATE_KEY not in observation:
        raise KeyError(f"Missing required observation key: {STATE_KEY}")
    state = np.asarray(observation[STATE_KEY])
    if state.ndim != 1:
        raise ValueError(f"{STATE_KEY} must have shape [D], got {state.shape}")
    if schema.state_dim is not None and state.shape != (schema.state_dim,):
        raise ValueError(f"{STATE_KEY} must have shape [{schema.state_dim}], got {state.shape}")
    if state.dtype.kind not in ("u", "i", "f") or not np.all(np.isfinite(state)):
        raise ValueError(f"{STATE_KEY} must contain finite numeric values")
    if schema.require_prompt:
        prompt = observation.get(PROMPT_KEY)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")


def validate_action_response(
    response: Mapping[str, Any],
    *,
    action_dim: int | None = None,
) -> np.ndarray:
    if "actions" not in response:
        raise KeyError(f"Policy response lacks 'actions': {sorted(response)}")
    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"actions must have shape [T,D], got {actions.shape}")
    if actions.shape[0] <= 0 or actions.shape[1] <= 0:
        raise ValueError(f"actions must be non-empty, got {actions.shape}")
    if action_dim is not None and actions.shape[1] != action_dim:
        raise ValueError(f"actions must have dimension {action_dim}, got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions contain non-finite values")
    return actions
