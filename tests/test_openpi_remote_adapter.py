from types import SimpleNamespace

import numpy as np


def test_openpi_remote_adapter_unbatches_observation_and_returns_chunk(monkeypatch) -> None:
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.metadata = SimpleNamespace(
                action_dim=8,
                action_horizon=4,
                camera_keys=(
                    "observation.images.head_cam_h",
                    "observation.images.wrist_cam_r",
                ),
            )

        def infer(self, request):
            captured["request"] = request
            return {"actions": np.ones((4, 8), dtype=np.float32)}

        def reset(self):
            captured["reset"] = True

    monkeypatch.setattr(
        "kuavo_deploy.utils.openpi_remote_adapter.WebSocketPolicyClient",
        FakeClient,
    )
    from kuavo_deploy.utils.openpi_remote_adapter import load_openpi_remote_policy

    policy, preprocessor, postprocessor = load_openpi_remote_policy(
        "policy-host:9000",
        task_prompt="pick",
    )
    observation = {
        "observation.images.head_cam_h": np.zeros((1, 3, 10, 12), dtype=np.float32),
        "observation.images.wrist_cam_r": np.zeros((1, 3, 10, 12), dtype=np.float32),
        "observation.state": np.zeros((1, 8), dtype=np.float32),
    }
    chunk = policy.predict_action_chunk(preprocessor(observation))

    assert captured["kwargs"]["host"] == "policy-host"
    assert captured["kwargs"]["port"] == 9000
    assert captured["request"]["observation.images.head_cam_h"].shape == (3, 10, 12)
    assert captured["request"]["observation.state"].shape == (8,)
    assert captured["request"]["prompt"] == "pick"
    assert chunk.shape == (1, 4, 8)
    assert postprocessor(chunk) is chunk
    assert policy.config.chunk_size == 4


def test_openpi_remote_adapter_uses_reviewed_horizon_when_metadata_is_sparse(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **_kwargs):
            self.metadata = SimpleNamespace(
                action_dim=None,
                action_horizon=None,
                camera_keys=(),
            )

    monkeypatch.setattr(
        "kuavo_deploy.utils.openpi_remote_adapter.WebSocketPolicyClient",
        FakeClient,
    )
    from kuavo_deploy.utils.openpi_remote_adapter import load_openpi_remote_policy

    policy, _, _ = load_openpi_remote_policy(
        "127.0.0.1:8000",
        task_prompt="pick",
        action_dim=8,
        action_horizon=50,
    )
    assert policy.config.output_features["action"].shape == (8,)
    assert policy.config.chunk_size == 50
