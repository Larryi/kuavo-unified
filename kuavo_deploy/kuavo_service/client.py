"""ROS-facing remote policy adapter.

The transport is dependency-light msgpack over WebSocket and is compatible
with OpenPI's ``WebsocketPolicyServer``. Model-framework dependencies remain in
the worker process.
"""

from __future__ import annotations

from collections import deque
import os
from typing import Any, Mapping

import numpy as np


def _to_numpy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_numpy(item) for item in value)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


class PolicyClient:
    """Expose a remote action-chunk policy through ``select_action``."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        task_prompt: str = "",
        api_key: str | None = None,
        api_key_env: str = "KUAVO_POLICY_API_KEY",
        connect_timeout_s: float = 30.0,
        request_timeout_s: float = 30.0,
        action_dim: int | None = None,
        state_dim: int | None = None,
        execute_steps: int = 1,
        validate_schema: bool = True,
    ) -> None:
        from kuavo_policy_protocol import (
            ObservationSchema,
            WebSocketPolicyClient,
            validate_action_response,
            validate_observation,
        )

        resolved_api_key = api_key or (os.getenv(api_key_env) if api_key_env else None)
        self._client = WebSocketPolicyClient(
            host=host,
            port=port,
            api_key=resolved_api_key,
            connect_timeout_s=connect_timeout_s,
            request_timeout_s=request_timeout_s,
        )
        self._task_prompt = task_prompt
        self._configured_action_dim = action_dim
        self._schema = ObservationSchema(state_dim=state_dim)
        self._execute_steps = int(execute_steps)
        if self._execute_steps <= 0:
            raise ValueError(f"execute_steps must be positive, got {execute_steps}")
        self._validate_schema = validate_schema
        self._validate_action_response = validate_action_response
        self._validate_observation = validate_observation
        self._actions: deque[np.ndarray] = deque()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._client.get_server_metadata()

    def health(self) -> bool:
        return self._client.health()

    def select_action(self, observation: Mapping[str, Any]):
        if not self._actions:
            request = {
                key: _to_numpy(value)
                for key, value in observation.items()
                if key.startswith("observation.images.") or key == "observation.state"
            }
            request["prompt"] = str(
                observation.get("prompt") or observation.get("task") or self._task_prompt
            )
            if self._validate_schema:
                self._validate_observation(request, self._schema)
            metadata_dim = self._client.metadata.action_dim
            action_dim = self._configured_action_dim or metadata_dim
            actions = self._validate_action_response(
                self._client.infer(request),
                action_dim=action_dim,
            )
            self._actions.extend(actions[: self._execute_steps])
        action = self._actions.popleft()
        try:
            import torch
        except ImportError:
            return action[np.newaxis, :]
        return torch.from_numpy(action.copy()).unsqueeze(0)

    def reset(self) -> None:
        self._actions.clear()
        self._client.reset()

    def close(self) -> None:
        self._actions.clear()
        self._client.close()

    def eval(self) -> "PolicyClient":
        return self

    def to(self, _device) -> "PolicyClient":
        return self
