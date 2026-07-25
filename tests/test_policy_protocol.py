from __future__ import annotations

from types import SimpleNamespace
import http
import threading
import urllib.error

import numpy as np
import pytest

from kuavo_policy_protocol import msgpack_numpy
from kuavo_policy_protocol.client import (
    PolicyClientError,
    PolicyTimeoutError,
    WebSocketPolicyClient,
)
from kuavo_policy_protocol.schema import (
    ObservationSchema,
    PolicyMetadata,
    validate_action_response,
    validate_observation,
)


def valid_observation() -> dict:
    return {
        "observation.images.head_cam_h": np.zeros((3, 8, 10), dtype=np.uint8),
        "observation.images.wrist_cam_r": np.zeros((8, 10, 3), dtype=np.uint8),
        "observation.state": np.arange(8, dtype=np.float32),
        "prompt": "pick up the object",
    }


def test_msgpack_numpy_round_trip_and_rejects_object_arrays() -> None:
    value = {
        "image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
        "state": np.arange(8, dtype=np.float32),
        "step": np.int64(3),
    }
    restored = msgpack_numpy.unpackb(msgpack_numpy.packb(value))

    np.testing.assert_array_equal(restored["image"], value["image"])
    np.testing.assert_array_equal(restored["state"], value["state"])
    assert restored["step"] == 3
    with pytest.raises(ValueError, match="Unsupported dtype"):
        msgpack_numpy.packb({"unsafe": np.asarray([object()], dtype=object)})


def test_msgpack_numpy_is_wire_compatible_with_openpi() -> None:
    from openpi_client import msgpack_numpy as openpi_msgpack

    value = {
        "image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
        "state": np.arange(8, dtype=np.float32),
    }
    decoded_by_openpi = openpi_msgpack.unpackb(msgpack_numpy.packb(value))
    decoded_by_kuavo = msgpack_numpy.unpackb(openpi_msgpack.packb(value))
    np.testing.assert_array_equal(decoded_by_openpi["image"], value["image"])
    np.testing.assert_array_equal(decoded_by_kuavo["state"], value["state"])


def test_schema_validates_observation_metadata_and_actions() -> None:
    observation = valid_observation()
    validate_observation(observation, ObservationSchema(state_dim=8))
    metadata = PolicyMetadata.from_mapping(
        {
            "backend": "openpi",
            "action_dim": 8,
            "action_horizon": 50,
            "custom": "preserved",
        }
    )
    assert metadata.backend == "openpi"
    assert metadata.extra == {"custom": "preserved"}
    actions = validate_action_response(
        {"actions": np.zeros((50, 8), dtype=np.float32)},
        action_dim=8,
    )
    assert actions.shape == (50, 8)

    with pytest.raises(KeyError, match="wrist camera"):
        validate_observation(
            {key: value for key, value in observation.items() if "wrist" not in key}
        )
    with pytest.raises(ValueError, match="dimension 8"):
        validate_action_response({"actions": np.zeros((2, 7))}, action_dim=8)


class FakeWebSocket:
    def __init__(self, *, response: bytes | str | Exception | None = None) -> None:
        self.frames = [
            msgpack_numpy.packb(
                {
                    "backend": "openpi",
                    "action_dim": 8,
                    "action_horizon": 2,
                }
            )
        ]
        self.response = response
        self.sent: list[dict] = []
        self.closed = False

    def send(self, payload: bytes) -> None:
        self.sent.append(msgpack_numpy.unpackb(payload))

    def recv(self, timeout: float | None = None):
        del timeout
        if self.frames:
            return self.frames.pop(0)
        if isinstance(self.response, Exception):
            raise self.response
        if self.response is not None:
            return self.response
        return msgpack_numpy.packb({"actions": np.zeros((2, 8), dtype=np.float32)})

    def close(self) -> None:
        self.closed = True


def test_websocket_client_matches_openpi_handshake_infer_and_reset(monkeypatch) -> None:
    sockets: list[FakeWebSocket] = []

    def connect(*_args, **_kwargs):
        socket = FakeWebSocket()
        sockets.append(socket)
        return socket

    monkeypatch.setattr("websockets.sync.client.connect", connect)
    client = WebSocketPolicyClient(
        "127.0.0.1",
        8000,
        connect_timeout_s=0.1,
        request_timeout_s=0.1,
        retry_interval_s=0.001,
    )
    assert client.metadata.backend == "openpi"
    response = client.infer(valid_observation())
    assert np.asarray(response["actions"]).shape == (2, 8)
    assert sockets[0].sent[0]["prompt"] == "pick up the object"

    client.reset()
    assert sockets[0].closed
    assert len(sockets) == 2
    client.close()
    assert sockets[1].closed


def test_websocket_client_bounds_inference_timeout(monkeypatch) -> None:
    socket = FakeWebSocket(response=TimeoutError())
    monkeypatch.setattr("websockets.sync.client.connect", lambda *_args, **_kwargs: socket)
    client = WebSocketPolicyClient(
        connect_timeout_s=0.1,
        request_timeout_s=0.1,
        retry_interval_s=0.001,
    )
    with pytest.raises(PolicyTimeoutError, match="timed out"):
        client.infer(valid_observation())
    assert socket.closed


def test_websocket_client_surfaces_server_error_without_traceback_execution(monkeypatch) -> None:
    socket = FakeWebSocket(response="remote traceback text")
    monkeypatch.setattr("websockets.sync.client.connect", lambda *_args, **_kwargs: socket)
    client = WebSocketPolicyClient(
        connect_timeout_s=0.1,
        request_timeout_s=0.1,
        retry_interval_s=0.001,
    )
    with pytest.raises(PolicyClientError, match="remote traceback text"):
        client.infer(valid_observation())


def test_websocket_client_health_uses_openpi_healthz(monkeypatch) -> None:
    socket = FakeWebSocket()
    monkeypatch.setattr("websockets.sync.client.connect", lambda *_args, **_kwargs: socket)

    class Response:
        status = 200

        def read(self):
            return b"OK\n"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    requested_urls: list[str] = []

    def urlopen(request, timeout):
        assert timeout == 0.1
        requested_urls.append(request.full_url)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    client = WebSocketPolicyClient(
        "ws://policy-worker",
        9000,
        connect_timeout_s=0.1,
        request_timeout_s=0.1,
        retry_interval_s=0.001,
    )
    assert client.health()
    assert requested_urls == ["http://policy-worker:9000/healthz"]

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(urllib.error.URLError("down")),
    )
    assert not client.health()


def test_ros_policy_client_consumes_action_chunk(monkeypatch) -> None:
    class FakeProtocolClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.metadata = SimpleNamespace(action_dim=8)
            self.infer_calls = 0
            self.reset_calls = 0

        def get_server_metadata(self):
            return {"backend": "openpi", "action_dim": 8}

        def infer(self, request):
            self.infer_calls += 1
            assert request["prompt"] == "task prompt"
            return {
                "actions": np.stack(
                    [
                        np.arange(8, dtype=np.float32),
                        np.arange(8, dtype=np.float32) + 10,
                    ]
                )
            }

        def health(self):
            return True

        def reset(self):
            self.reset_calls += 1

        def close(self):
            pass

    monkeypatch.setattr("kuavo_policy_protocol.WebSocketPolicyClient", FakeProtocolClient)
    from kuavo_deploy.kuavo_service.client import PolicyClient

    client = PolicyClient(
        task_prompt="task prompt",
        action_dim=8,
        state_dim=8,
        execute_steps=2,
    )
    observation = valid_observation()
    observation.pop("prompt")
    first = np.asarray(client.select_action(observation))
    second = np.asarray(client.select_action(observation))
    np.testing.assert_array_equal(first, np.arange(8, dtype=np.float32)[None, :])
    np.testing.assert_array_equal(second, (np.arange(8, dtype=np.float32) + 10)[None, :])
    assert client._client.infer_calls == 1

    client.reset()
    assert client._client.reset_calls == 1
    client.select_action(observation)
    assert client._client.infer_calls == 2


def test_real_websocket_server_openpi_flow() -> None:
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.server import serve

    requests: list[dict] = []

    def process_request(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    def handler(websocket):
        websocket.send(
            msgpack_numpy.packb(
                {
                    "backend": "openpi",
                    "action_dim": 8,
                    "action_horizon": 2,
                }
            )
        )
        try:
            while True:
                request = msgpack_numpy.unpackb(websocket.recv())
                requests.append(request)
                websocket.send(
                    msgpack_numpy.packb(
                        {"actions": np.ones((2, 8), dtype=np.float32)}
                    )
                )
        except ConnectionClosed:
            return

    try:
        server = serve(
            handler,
            "127.0.0.1",
            0,
            compression=None,
            max_size=None,
            process_request=process_request,
        )
    except PermissionError:
        pytest.skip("Local socket creation is disabled in this sandbox")
    port = server.socket.getsockname()[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = WebSocketPolicyClient(
            "127.0.0.1",
            port,
            connect_timeout_s=1,
            request_timeout_s=1,
            retry_interval_s=0.01,
        )
        assert client.health()
        response = client.infer(valid_observation())
        assert np.asarray(response["actions"]).shape == (2, 8)
        assert requests[0]["prompt"] == "pick up the object"
        client.reset()
        assert client.metadata.backend == "openpi"
        client.close()
    finally:
        server.shutdown()
        thread.join(timeout=2)
