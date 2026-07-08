from pathlib import Path
from types import SimpleNamespace

import torch

from kuavo_train.lingbot_v2.lora import merge_lora_state_dict


def test_merge_lora_state_dict_restores_original_linear_key():
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    state = {
        "layer.base_layer.weight": base,
        "layer.lora_A.default.weight": torch.ones(2, 4),
        "layer.lora_B.default.weight": torch.ones(3, 2),
    }

    merged = merge_lora_state_dict(state, alpha=4, rank=2)

    assert set(merged) == {"layer.weight"}
    assert torch.equal(merged["layer.weight"], base + 4)


def test_kuavo_v2_mapping_uses_relative_arm_and_absolute_gripper():
    from lingbotvla.data.vla_data.utils import FeatureTransform

    repo_root = Path(__file__).resolve().parents[1]
    config = SimpleNamespace(
        joints=["{'arm.position': 14}", "{'end.position': 14}", "{'effector.position': 2}"],
        cameras=["camera_top", "camera_wrist_left", "camera_wrist_right"],
        norm_type=[],
    )
    transform = FeatureTransform(
        repo_root / "configs/robot_configs/kuavo_v2.yaml",
        config,
        None,
        None,
        disabled_image_features=True,
        do_nomalize=False,
        chunk_size=50,
        return_item_befor_padding=True,
    )
    state = torch.arange(16, dtype=torch.float32)
    action = state.repeat(50, 1) + 0.25

    converted = transform.apply(
        {
            "observation.state": state,
            "action": action,
            "action_is_pad": torch.zeros(50, dtype=torch.bool),
            "task": "move",
        }
    )

    assert converted["action.arm.position"].shape == (50, 7)
    assert torch.allclose(converted["action.arm.position"], torch.full((50, 7), 0.25))
    assert torch.allclose(converted["action.effector.position"], action[:, [7]])


def test_lingbot_v2_deploy_payload_preserves_kuavo_slots():
    from kuavo_deploy.utils.lingbot_v2_adapter import LingbotV2DeployPolicy

    policy = LingbotV2DeployPolicy.__new__(LingbotV2DeployPolicy)
    policy.task_prompt = "move"
    policy.action_dim = 16
    image = torch.zeros(3, 8, 8)
    state = torch.arange(16, dtype=torch.float32)
    payload = policy._payload(
        {
            "observation.images.head_cam_h": image,
            "observation.images.wrist_cam_l": image,
            "observation.images.wrist_cam_r": image,
            "observation.state": state,
        }
    )

    assert payload["observation.state"].shape == (16,)
    assert payload["observation.state"][7] == 7
    assert payload["observation.state"][15] == 15
    assert payload["observation.images.head_cam_h"].shape == (8, 8, 3)
