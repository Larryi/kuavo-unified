"""Open-loop adapter for an isolated OpenPI WebSocket policy server."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from kuavo_policy_protocol import (
    ObservationSchema,
    WebSocketPolicyClient,
    validate_action_response,
    validate_observation,
)


DEFAULT_CAMERA_KEYS = (
    "observation.images.head_cam_h",
    "observation.images.wrist_cam_r",
)


def _wire_value(key: str, value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if key.startswith("observation.images.") and array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if key == "observation.state" and array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    return array


class OpenPIRemotePolicy:
    """Expose OpenPI's unbatched wire protocol as a LeRobot-like policy."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        task_prompt: str,
        state_dim: int = 8,
        action_dim: int | None = None,
        action_horizon: int = 50,
        connect_timeout_s: float = 60.0,
        request_timeout_s: float = 60.0,
    ) -> None:
        self._client = WebSocketPolicyClient(
            host=host,
            port=port,
            connect_timeout_s=connect_timeout_s,
            request_timeout_s=request_timeout_s,
        )
        metadata = self._client.metadata
        # Physical Intelligence's native WebsocketPolicyServer may only expose
        # generic metadata. Kuavo's current policies use matching state/action
        # widths (Task1=8, Task2=16), so the configured state schema is a safe
        # final fallback. The first response is still validated against it.
        resolved_action_dim = action_dim or metadata.action_dim or state_dim
        camera_keys = tuple(metadata.camera_keys) or DEFAULT_CAMERA_KEYS
        horizon = metadata.action_horizon or action_horizon
        self.config = SimpleNamespace(
            type="openpi",
            input_features={
                **{key: SimpleNamespace(shape=None) for key in camera_keys},
                "observation.state": SimpleNamespace(shape=(state_dim,)),
            },
            output_features={
                "action": SimpleNamespace(shape=(int(resolved_action_dim),)),
            },
            chunk_size=int(horizon),
            horizon=int(horizon),
        )
        self._task_prompt = task_prompt
        self._schema = ObservationSchema(state_dim=state_dim)
        self._action_dim = int(resolved_action_dim)
        self._actions: deque[np.ndarray] = deque()

    def _request(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        request = {
            key: _wire_value(key, value)
            for key, value in observation.items()
            if key.startswith("observation.images.") or key == "observation.state"
        }
        request["prompt"] = str(
            observation.get("prompt") or observation.get("task") or self._task_prompt
        )
        validate_observation(request, self._schema)
        return request

    def predict_action_chunk(self, observation: Mapping[str, Any]) -> torch.Tensor:
        actions = validate_action_response(
            self._client.infer(self._request(observation)),
            action_dim=self._action_dim,
        )
        return torch.from_numpy(actions.copy()).unsqueeze(0)

    def select_action(self, observation: Mapping[str, Any]) -> torch.Tensor:
        if not self._actions:
            self._actions.extend(self.predict_action_chunk(observation)[0].numpy())
        return torch.from_numpy(self._actions.popleft().copy()).unsqueeze(0)

    def reset(self) -> None:
        self._actions.clear()
        self._client.reset()

    def eval(self) -> "OpenPIRemotePolicy":
        return self

    def to(self, _device) -> "OpenPIRemotePolicy":
        return self


def identity_processor(value):
    return value


def load_openpi_remote_policy(
    endpoint: str,
    *,
    task_prompt: str,
    state_dim: int = 8,
    action_dim: int | None = None,
    action_horizon: int = 50,
) -> tuple[OpenPIRemotePolicy, Any, Any]:
    host, separator, raw_port = endpoint.rpartition(":")
    if not separator:
        host, raw_port = endpoint, "8000"
    policy = OpenPIRemotePolicy(
        host=host or "127.0.0.1",
        port=int(raw_port),
        task_prompt=task_prompt,
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
    )
    return policy, identity_processor, identity_processor
