from types import SimpleNamespace

import numpy as np
import torch

from kuavo_deploy.utils.lingbot_adapter import (
    LingbotDeployPolicy,
    _ActionDimNormalizer,
    _robot_feature_defaults,
    _repair_policy_transform,
    _split_raw_norm_stats_for_robot,
)


def make_adapter(actions: np.ndarray):
    captured = {}

    def infer(payload):
        captured.update(payload)
        return {"action": actions}

    adapter = LingbotDeployPolicy.__new__(LingbotDeployPolicy)
    adapter.policy = SimpleNamespace(infer=infer)
    adapter.task_prompt = "pick the connector"
    adapter.execute_raw_action = False
    adapter._action_queue = []
    return adapter, captured


def test_lingbot_payload_mirrors_right_wrist_and_converts_images():
    adapter, captured = make_adapter(np.zeros((3, 8), dtype=np.float32))
    head = torch.full((3, 4, 5), 0.5)
    right = torch.ones((1, 3, 4, 5))
    observation = {
        "observation.images.head_cam_h": head,
        "observation.images.wrist_cam_r": right,
        "observation.state": torch.arange(8, dtype=torch.float32).unsqueeze(0),
    }

    chunk = adapter.predict_action_chunk(observation)

    assert chunk.shape == (3, 8)
    assert captured["observation.images.head_cam_h"].shape == (4, 5, 3)
    assert captured["observation.images.head_cam_h"].dtype == np.uint8
    np.testing.assert_array_equal(
        captured["observation.images.wrist_cam_l"],
        captured["observation.images.wrist_cam_r"],
    )
    np.testing.assert_array_equal(captured["observation.state"], np.arange(8, dtype=np.float32))
    assert captured["task"] == "pick the connector"


def test_select_action_preserves_single_action_contract():
    actions = np.arange(24, dtype=np.float32).reshape(3, 8)
    adapter, _ = make_adapter(actions)
    observation = {
        "observation.images.head_cam_h": np.zeros((4, 5, 3), dtype=np.uint8),
        "observation.images.wrist_cam_r": np.zeros((4, 5, 3), dtype=np.uint8),
        "observation.state": np.zeros(8, dtype=np.float32),
    }

    action = adapter.select_action(observation)

    assert action.shape == (1, 8)
    torch.testing.assert_close(action, torch.from_numpy(actions[:1]))


def test_select_action_reuses_upstream_chunk_before_replanning():
    actions = np.arange(24, dtype=np.float32).reshape(3, 8)
    adapter, captured = make_adapter(actions)
    observation = {
        "observation.images.head_cam_h": np.zeros((4, 5, 3), dtype=np.uint8),
        "observation.images.wrist_cam_r": np.zeros((4, 5, 3), dtype=np.uint8),
        "observation.state": np.zeros(8, dtype=np.float32),
    }

    first = adapter.select_action(observation)
    captured.clear()
    second = adapter.select_action(observation)

    torch.testing.assert_close(first, torch.from_numpy(actions[:1]))
    torch.testing.assert_close(second, torch.from_numpy(actions[1:2]))
    assert captured == {}


def test_action_normalizer_crops_upstream_14d_actions():
    class CaptureNormalizer:
        def unnormalize(self, data):
            return data

        def normalize(self, data):
            return data

    normalizer = _ActionDimNormalizer(CaptureNormalizer(), action_dim=8)
    result = normalizer.unnormalize({"action": torch.zeros(50, 14), "state": torch.zeros(8)})

    assert result["action"].shape == (50, 8)
    assert result["state"].shape == (8,)


def test_v1_robot_config_declares_eight_raw_dimensions():
    assert (
        LingbotDeployPolicy._raw_dim_from_robot_config(
            "kuavo_v1_right_arm",
            "states",
            "observation.state",
        )
        == 8
    )


def test_v1_robot_config_recovers_missing_training_features():
    joints, cameras = _robot_feature_defaults("kuavo_v1_right_arm")

    assert joints == ["{'arm.position': 7}", "{'effector.position': 1}"]
    assert cameras == ["camera_top", "camera_wrist_right"]


def test_v1_raw_norm_stats_are_split_to_robot_features():
    raw = {
        "observation.state": {
            "mean": list(range(8)),
            "std": [1] * 8,
            "q01": [-1] * 8,
            "q99": [1] * 8,
        },
        "action": {
            "mean": list(range(10, 18)),
            "std": [2] * 8,
            "q01": [-2] * 8,
            "q99": [2] * 8,
        },
    }

    mapped = _split_raw_norm_stats_for_robot(raw, "kuavo_v1_right_arm")

    np.testing.assert_array_equal(
        mapped["observation.state.arm.position"]["mean"],
        np.arange(7),
    )
    np.testing.assert_array_equal(
        mapped["observation.state.effector.position"]["mean"],
        np.array([7]),
    )
    np.testing.assert_array_equal(
        mapped["action.arm.position"]["mean"],
        np.arange(10, 17),
    )
    np.testing.assert_array_equal(
        mapped["action.effector.position"]["mean"],
        np.array([17]),
    )


def test_v1_transform_repair_survives_each_policy_reset():
    raw = {
        "observation.state": {"mean": list(range(8))},
        "action": {"mean": list(range(10, 18))},
    }
    normalizer = SimpleNamespace(
        norm_stats=raw,
        norm_type={"action.arm.position": "bounds_99_woclip"},
    )
    policy = SimpleNamespace(
        vla=SimpleNamespace(
            feature_transform=SimpleNamespace(normalizer=normalizer),
        )
    )

    _repair_policy_transform(policy, "kuavo_v1_right_arm")

    np.testing.assert_array_equal(
        normalizer.norm_stats["observation.state.arm.position"]["mean"],
        np.arange(7),
    )
    assert normalizer.norm_type["action.arm.position"] == "bounds_99"
    assert (
        LingbotDeployPolicy._raw_dim_from_robot_config(
            "kuavo_v1_right_arm",
            "actions",
            "action",
        )
        == 8
    )
