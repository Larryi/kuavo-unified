"""ROS-facing remote policy adapter.

The transport is dependency-light msgpack over WebSocket and is compatible
with OpenPI's ``WebsocketPolicyServer``. Model-framework dependencies remain in
the worker process.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import os
import queue
import threading
import time
from typing import Any, Mapping

import numpy as np


logger = logging.getLogger(__name__)


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


def _to_wire_observation(key: str, value: Any) -> Any:
    """Remove LeRobot's single-sample batch for the OpenPI wire protocol."""
    value = _to_numpy(value)
    array = np.asarray(value)
    if key.startswith("observation.images.") and array.ndim == 4 and array.shape[0] == 1:
        return array[0]
    if key == "observation.state" and array.ndim == 2 and array.shape[0] == 1:
        return array[0]
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be 0/1 or false/true, got {raw!r}")


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= minimum:
        raise ValueError(f"{name} must be > {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class _AsyncInferenceResult:
    response: Mapping[str, Any] | None
    latency_s: float
    error: BaseException | None = None


class _AsyncPolicyWorker:
    """Own a separate WebSocket connection for one in-flight RTC request."""

    def __init__(self, client_type, client_kwargs: Mapping[str, Any]) -> None:
        self._client_type = client_type
        self._client_kwargs = dict(client_kwargs)
        self._requests: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[_AsyncInferenceResult] = queue.Queue(maxsize=1)
        self._busy = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="kuavo-openpi-rtc",
            daemon=True,
        )

    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    def start(self) -> None:
        self._thread.start()

    def submit(self, request: dict[str, Any]) -> bool:
        if self.busy:
            return False
        self._busy.set()
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            self._busy.clear()
            return False
        return True

    def poll(self) -> _AsyncInferenceResult | None:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        client = None
        try:
            while True:
                request = self._requests.get()
                if request is None:
                    return
                started = time.perf_counter()
                try:
                    if client is None:
                        client = self._client_type(**self._client_kwargs)
                    response = client.infer(request)
                    result = _AsyncInferenceResult(
                        response=response,
                        latency_s=time.perf_counter() - started,
                    )
                except BaseException as error:
                    result = _AsyncInferenceResult(
                        response=None,
                        latency_s=time.perf_counter() - started,
                        error=error,
                    )
                self._results.put(result)
                self._busy.clear()
        finally:
            if client is not None:
                client.close()


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
        self._client_type = WebSocketPolicyClient
        self._client_kwargs = {
            "host": host,
            "port": port,
            "api_key": resolved_api_key,
            "connect_timeout_s": connect_timeout_s,
            "request_timeout_s": request_timeout_s,
        }
        self._client = self._client_type(
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
        self._rtc_enabled = _env_bool("KUAVO_OPENPI_RTC_ENABLED", False)
        self._rtc_worker: _AsyncPolicyWorker | None = None
        self._rtc_request_sent_at: int | None = None
        self._rtc_action_count = 0
        self._rtc_estimated_delay = 0
        self._rtc_warmed_up = False

        if self._rtc_enabled:
            metadata_horizon = self._client.metadata.action_horizon
            default_horizon = metadata_horizon or 50
            self._rtc_open_loop_horizon = _env_int(
                "KUAVO_OPENPI_RTC_OPEN_LOOP_HORIZON",
                default_horizon,
            )
            self._rtc_queue_threshold = _env_int(
                "KUAVO_OPENPI_RTC_QUEUE_THRESHOLD",
                min(20, self._rtc_open_loop_horizon),
            )
            self._rtc_execution_horizon = _env_int(
                "KUAVO_OPENPI_RTC_EXECUTION_HORIZON",
                min(10, self._rtc_queue_threshold),
            )
            self._rtc_max_guidance_weight = _env_float(
                "KUAVO_OPENPI_RTC_MAX_GUIDANCE_WEIGHT",
                10.0,
            )
            self._rtc_result_timeout_s = _env_float(
                "KUAVO_OPENPI_RTC_RESULT_TIMEOUT_S",
                10.0,
            )
            self._rtc_control_hz = _env_float(
                "KUAVO_OPENPI_CONTROL_HZ",
                10.0,
            )
            self._rtc_warmup = _env_bool("KUAVO_OPENPI_RTC_WARMUP", True)
            if metadata_horizon and self._rtc_open_loop_horizon > metadata_horizon:
                raise ValueError(
                    "KUAVO_OPENPI_RTC_OPEN_LOOP_HORIZON cannot exceed server "
                    f"action_horizon={metadata_horizon}"
                )
            if not (
                1
                <= self._rtc_execution_horizon
                <= self._rtc_queue_threshold
                <= self._rtc_open_loop_horizon
            ):
                raise ValueError(
                    "RTC horizons must satisfy 1 <= execution_horizon <= "
                    "queue_threshold <= open_loop_horizon"
                )
            logger.warning(
                "OpenPI RTC ENABLED: open_loop_horizon=%d queue_threshold=%d "
                "execution_horizon=%d control_hz=%.3g warmup=%s "
                "max_guidance_weight=%.3g; execute_steps=%d is ignored",
                self._rtc_open_loop_horizon,
                self._rtc_queue_threshold,
                self._rtc_execution_horizon,
                self._rtc_control_hz,
                self._rtc_warmup,
                self._rtc_max_guidance_weight,
                self._execute_steps,
            )
        else:
            logger.info(
                "OpenPI RTC disabled; using synchronous action chunks with execute_steps=%d",
                self._execute_steps,
            )

    @property
    def metadata(self) -> dict[str, Any]:
        return self._client.get_server_metadata()

    def health(self) -> bool:
        return self._client.health()

    def _make_request(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        request = {
            key: _to_wire_observation(key, value)
            for key, value in observation.items()
            if key.startswith("observation.images.") or key == "observation.state"
        }
        request["prompt"] = str(
            observation.get("prompt") or observation.get("task") or self._task_prompt
        )
        if self._validate_schema:
            self._validate_observation(request, self._schema)
        return request

    def _action_dim(self) -> int | None:
        return self._configured_action_dim or self._client.metadata.action_dim

    def _validate_response(self, response: Mapping[str, Any]) -> np.ndarray:
        return self._validate_action_response(response, action_dim=self._action_dim())

    def _make_rtc_request(
        self,
        request: Mapping[str, Any],
        prefix: np.ndarray,
    ) -> dict[str, Any]:
        rtc_request = dict(request)
        rtc_request["rtc"] = {
            "prev_actions": np.asarray(prefix, dtype=np.float32),
            "inference_delay": min(
                self._rtc_estimated_delay,
                self._rtc_open_loop_horizon,
            ),
            "execution_horizon": min(self._rtc_execution_horizon, len(prefix)),
            "max_guidance_weight": self._rtc_max_guidance_weight,
        }
        return rtc_request

    def _start_rtc_worker(self) -> None:
        if self._rtc_worker is None:
            self._rtc_worker = _AsyncPolicyWorker(
                self._client_type,
                self._client_kwargs,
            )
            self._rtc_worker.start()

    def _warmup_rtc(self, request: Mapping[str, Any]) -> None:
        if self._rtc_warmed_up:
            return
        prefix = np.asarray(list(self._actions), dtype=np.float32)[
            : self._rtc_execution_horizon
        ]
        warmup_request = self._make_rtc_request(request, prefix)
        warmup_request["rtc"]["inference_delay"] = 0
        if self._rtc_warmup:
            logger.warning("OpenPI RTC: compiling JAX RTC graph before motion")
            started = time.perf_counter()
            self._validate_response(self._client.infer(warmup_request))
            logger.warning(
                "OpenPI RTC: JAX compile warmup complete in %.3fs",
                time.perf_counter() - started,
            )
            started = time.perf_counter()
            self._validate_response(self._client.infer(warmup_request))
            hot_latency_s = time.perf_counter() - started
            self._rtc_estimated_delay = int(np.ceil(hot_latency_s * self._rtc_control_hz))
            logger.warning(
                "OpenPI RTC: hot warmup complete in %.3fs; estimated_delay=%d actions",
                hot_latency_s,
                self._rtc_estimated_delay,
            )
        self._rtc_warmed_up = True
        self._start_rtc_worker()

    def _consume_rtc_result(self, completed: _AsyncInferenceResult) -> None:
        if completed.error is not None:
            raise RuntimeError("Background OpenPI RTC inference failed") from completed.error
        if completed.response is None or self._rtc_request_sent_at is None:
            raise RuntimeError("OpenPI RTC result has no matching request")
        actual_delay = self._rtc_action_count - self._rtc_request_sent_at
        new_actions = self._validate_response(completed.response)[
            : self._rtc_open_loop_horizon
        ]
        if actual_delay >= len(new_actions):
            raise RuntimeError(
                f"OpenPI RTC inference consumed {actual_delay} action periods, "
                f"but returned only {len(new_actions)} actions"
            )
        retained = new_actions[actual_delay:]
        self._actions.clear()
        self._actions.extend(retained)
        self._rtc_request_sent_at = None
        self._rtc_estimated_delay = int(
            np.ceil(completed.latency_s * self._rtc_control_hz)
        )
        logger.info(
            "OpenPI RTC chunk merged: latency=%.3fs estimated_delay=%d "
            "actual_delay=%d retained=%d",
            completed.latency_s,
            self._rtc_estimated_delay,
            actual_delay,
            len(retained),
        )
        if self._rtc_estimated_delay >= self._rtc_execution_horizon:
            logger.warning(
                "OpenPI RTC estimated delay %d reaches/exceeds execution horizon %d",
                self._rtc_estimated_delay,
                self._rtc_execution_horizon,
            )

    def _poll_rtc_result(self) -> bool:
        if self._rtc_worker is None:
            return False
        completed = self._rtc_worker.poll()
        if completed is None:
            return False
        self._consume_rtc_result(completed)
        return True

    def _submit_rtc_if_needed(self, request: Mapping[str, Any]) -> None:
        if self._rtc_worker is None or self._rtc_request_sent_at is not None:
            return
        remaining = len(self._actions)
        if not 0 < remaining <= self._rtc_queue_threshold:
            return
        prefix = np.asarray(list(self._actions), dtype=np.float32)[
            : self._rtc_execution_horizon
        ]
        if self._rtc_worker.submit(self._make_rtc_request(request, prefix)):
            self._rtc_request_sent_at = self._rtc_action_count
            logger.info(
                "OpenPI RTC inference started: remaining=%d estimated_delay=%d prefix=%d",
                remaining,
                self._rtc_estimated_delay,
                len(prefix),
            )

    def _wait_for_rtc_result(self) -> None:
        logger.warning(
            "OpenPI RTC queue underrun; holding the last command while inference finishes"
        )
        deadline = time.monotonic() + self._rtc_result_timeout_s
        while time.monotonic() < deadline:
            if self._poll_rtc_result():
                return
            time.sleep(0.01)
        raise TimeoutError(
            f"Timed out after {self._rtc_result_timeout_s:.3g}s waiting for OpenPI RTC"
        )

    def select_action(self, observation: Mapping[str, Any]):
        request = self._make_request(observation)
        if self._rtc_enabled:
            self._poll_rtc_result()
        if not self._actions:
            if self._rtc_enabled and self._rtc_request_sent_at is not None:
                self._wait_for_rtc_result()
            else:
                actions = self._validate_response(self._client.infer(request))
                keep = (
                    self._rtc_open_loop_horizon
                    if self._rtc_enabled
                    else self._execute_steps
                )
                self._actions.extend(actions[:keep])
                if self._rtc_enabled:
                    self._warmup_rtc(request)
        if self._rtc_enabled:
            self._submit_rtc_if_needed(request)
        action = self._actions.popleft()
        if self._rtc_enabled:
            self._rtc_action_count += 1
        try:
            import torch
        except ImportError:
            return action[np.newaxis, :]
        return torch.from_numpy(action.copy()).unsqueeze(0)

    def reset(self) -> None:
        if self._rtc_worker is not None:
            self._rtc_worker.close()
            self._rtc_worker = None
        self._actions.clear()
        self._rtc_request_sent_at = None
        self._rtc_action_count = 0
        self._rtc_estimated_delay = 0
        self._rtc_warmed_up = False
        self._client.reset()

    def close(self) -> None:
        if self._rtc_worker is not None:
            self._rtc_worker.close()
            self._rtc_worker = None
        self._actions.clear()
        self._client.close()

    def eval(self) -> "PolicyClient":
        return self

    def to(self, _device) -> "PolicyClient":
        return self
