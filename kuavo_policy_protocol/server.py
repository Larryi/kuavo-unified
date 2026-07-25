"""Framework-independent policy worker server."""

from __future__ import annotations

import hmac
import http
import logging
import threading
import time
import traceback
from typing import Any, Mapping, Protocol

from . import msgpack_numpy


class InferencePolicy(Protocol):
    def infer(self, observation: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def reset(self) -> None: ...


class WebSocketPolicyServer:
    """Serve a local policy with the OpenPI-compatible wire format."""

    def __init__(
        self,
        policy: InferencePolicy,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: Mapping[str, Any] | None = None,
        api_key: str | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = int(port)
        self._metadata = dict(metadata or {})
        self._api_key = api_key
        self._policy_lock = threading.Lock()
        self._packer = msgpack_numpy.Packer()

    def serve_forever(self) -> None:
        try:
            from websockets.exceptions import ConnectionClosed
            from websockets.sync.server import serve
        except ImportError as exc:
            raise RuntimeError(
                "Policy worker server requires websockets>=11 and msgpack; "
                "install requirements_policy_client.txt"
            ) from exc

        def process_request(connection, request):
            if not self._authorized(request.headers.get("Authorization")):
                return connection.respond(http.HTTPStatus.UNAUTHORIZED, "Unauthorized\n")
            if request.path == "/healthz":
                return connection.respond(http.HTTPStatus.OK, "OK\n")
            return None

        def handler(websocket):
            logging.info("Policy client connected: %s", websocket.remote_address)
            with self._policy_lock:
                self._policy.reset()
            websocket.send(self._packer.pack(self._metadata))
            try:
                while True:
                    started = time.monotonic()
                    payload = websocket.recv()
                    if isinstance(payload, str):
                        raise TypeError("Policy requests must be binary MessagePack frames")
                    observation = msgpack_numpy.unpackb(payload)
                    if not isinstance(observation, dict):
                        raise TypeError(
                            f"Policy observation must be a mapping, got {type(observation).__name__}"
                        )
                    with self._policy_lock:
                        result = dict(self._policy.infer(observation))
                    result.setdefault("server_timing", {})["infer_ms"] = (
                        time.monotonic() - started
                    ) * 1000
                    websocket.send(self._packer.pack(result))
            except ConnectionClosed:
                logging.info("Policy client disconnected: %s", websocket.remote_address)
            except Exception:
                # Match OpenPI: send a text error frame and close the connection.
                websocket.send(traceback.format_exc())
                websocket.close(code=1011, reason="Policy inference failed")
                raise

        with serve(
            handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=process_request,
        ) as server:
            address = server.socket.getsockname()
            logging.info("Policy worker ready at ws://%s:%s", address[0], address[1])
            server.serve_forever()

    def _authorized(self, authorization: str | None) -> bool:
        if self._api_key is None:
            return True
        expected = f"Api-Key {self._api_key}"
        return authorization is not None and hmac.compare_digest(authorization, expected)
