"""Synchronous msgpack WebSocket client compatible with OpenPI."""

from __future__ import annotations

import logging
import threading
import time
from types import TracebackType
from typing import Any, Mapping
import urllib.error
import urllib.request

from . import msgpack_numpy
from .schema import PolicyMetadata


class PolicyClientError(RuntimeError):
    """Base error raised by the remote policy client."""


class PolicyTimeoutError(PolicyClientError):
    """A bounded connect or inference operation timed out."""


class WebSocketPolicyClient:
    """OpenPI-compatible synchronous policy client.

    The OpenPI wire format sends metadata as the first binary frame, then
    accepts raw observation dictionaries and returns action dictionaries.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        api_key: str | None = None,
        connect_timeout_s: float = 30.0,
        request_timeout_s: float = 30.0,
        retry_interval_s: float = 0.25,
    ) -> None:
        self._uri = _websocket_uri(host, port)
        self._health_url = _health_url(host, port)
        self._api_key = api_key
        self._connect_timeout_s = _positive_timeout(connect_timeout_s, "connect_timeout_s")
        self._request_timeout_s = _positive_timeout(request_timeout_s, "request_timeout_s")
        self._retry_interval_s = _positive_timeout(retry_interval_s, "retry_interval_s")
        self._packer = msgpack_numpy.Packer()
        self._lock = threading.Lock()
        self._ws = None
        self._metadata_raw: dict[str, Any] = {}
        self._connect()

    @property
    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata.from_mapping(self._metadata_raw)

    def get_server_metadata(self) -> dict[str, Any]:
        return dict(self._metadata_raw)

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._ws is None:
                self._connect()
            assert self._ws is not None
            try:
                self._ws.send(self._packer.pack(dict(observation)))
                response = self._ws.recv(timeout=self._request_timeout_s)
            except TimeoutError as exc:
                self.close()
                raise PolicyTimeoutError(
                    f"Policy inference timed out after {self._request_timeout_s:.3g}s"
                ) from exc
            except Exception as exc:
                self.close()
                raise PolicyClientError(f"Policy inference transport failed: {type(exc).__name__}") from exc
            if isinstance(response, str):
                self.close()
                raise PolicyClientError(f"Policy server returned an error: {response}")
            decoded = msgpack_numpy.unpackb(response)
            if not isinstance(decoded, dict):
                raise PolicyClientError(f"Policy response must be a mapping, got {type(decoded).__name__}")
            return decoded

    def reset(self) -> None:
        """Reset client state and reconnect.

        OpenPI policies currently define reset as a no-op. Reconnecting clears
        transport state and refreshes metadata while remaining wire-compatible.
        """

        with self._lock:
            self.close()
            self._connect()

    def health(self) -> bool:
        request = urllib.request.Request(self._health_url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self._request_timeout_s) as response:
                return response.status == 200 and response.read().strip() == b"OK"
        except (OSError, urllib.error.URLError, TimeoutError):
            return False

    def close(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def _connect(self) -> None:
        try:
            import websockets.sync.client
        except ImportError as exc:
            raise PolicyClientError(
                "WebSocket policy support requires websockets>=11 and msgpack; "
                "install requirements_policy_client.txt"
            ) from exc

        deadline = time.monotonic() + self._connect_timeout_s
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        last_error: Exception | None = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PolicyTimeoutError(
                    f"Policy server did not become ready at {self._uri} "
                    f"within {self._connect_timeout_s:.3g}s"
                ) from last_error
            try:
                kwargs = {
                    "compression": None,
                    "max_size": None,
                    "open_timeout": min(remaining, self._request_timeout_s),
                    "close_timeout": self._request_timeout_s,
                }
                if headers:
                    kwargs["additional_headers"] = headers
                try:
                    ws = websockets.sync.client.connect(self._uri, **kwargs)
                except TypeError:
                    if "additional_headers" not in kwargs:
                        raise
                    kwargs["extra_headers"] = kwargs.pop("additional_headers")
                    ws = websockets.sync.client.connect(self._uri, **kwargs)
                metadata_frame = ws.recv(timeout=min(remaining, self._request_timeout_s))
                if isinstance(metadata_frame, str):
                    ws.close()
                    raise PolicyClientError(f"Policy metadata handshake failed: {metadata_frame}")
                metadata = msgpack_numpy.unpackb(metadata_frame)
                if not isinstance(metadata, dict):
                    ws.close()
                    raise PolicyClientError(
                        f"Policy metadata must be a mapping, got {type(metadata).__name__}"
                    )
                self._ws = ws
                self._metadata_raw = metadata
                return
            except PolicyClientError:
                raise
            except Exception as exc:
                last_error = exc
                logging.debug("Policy server not ready at %s: %s", self._uri, type(exc).__name__)
                time.sleep(min(self._retry_interval_s, max(0.0, deadline - time.monotonic())))

    def __enter__(self) -> "WebSocketPolicyClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _positive_timeout(value: float, name: str) -> float:
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _websocket_uri(host: str, port: int) -> str:
    host = host.rstrip("/")
    if host.startswith(("ws://", "wss://")):
        return host if _has_explicit_port(host) else f"{host}:{int(port)}"
    return f"ws://{host}:{int(port)}"


def _health_url(host: str, port: int) -> str:
    host = host.rstrip("/")
    if host.startswith("wss://"):
        host = "https://" + host[len("wss://") :]
    elif host.startswith("ws://"):
        host = "http://" + host[len("ws://") :]
    elif not host.startswith(("http://", "https://")):
        host = "http://" + host
    if not _has_explicit_port(host):
        host = f"{host}:{int(port)}"
    return f"{host}/healthz"


def _has_explicit_port(uri: str) -> bool:
    authority = uri.split("://", 1)[-1].split("/", 1)[0]
    if authority.startswith("["):
        return "]:" in authority
    return ":" in authority
