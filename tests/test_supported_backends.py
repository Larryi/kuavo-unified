import contextlib
import io
import sys
from pathlib import Path
from unittest.mock import patch

from kuavo_deploy.config import ConfigInference, load_kuavo_config
from tools.open_loop_eval import parse_args


def test_deploy_config_accepts_only_delivery_backends():
    for policy_type in ("act", "diffusion", "lingbot", "lingbot_v2", "client"):
        ConfigInference(policy_type=policy_type).validate()

    try:
        ConfigInference(policy_type="smolvla").validate()
    except ValueError:
        pass
    else:
        raise AssertionError("SmolVLA must not be accepted as a delivery backend")


def test_open_loop_cli_accepts_lingbot_v2():
    argv = [
        "open_loop_eval.py",
        "--policy-type",
        "lingbot_v2",
        "--policy-path",
        "/tmp/model",
    ]
    with patch.object(sys, "argv", argv):
        args = parse_args()

    assert args.policy_type == "lingbot_v2"
    assert args.policy_path == Path("/tmp/model")


def test_openpi_client_deploy_config_is_complete():
    cfg = load_kuavo_config("configs/deploy/kuavo_env.openpi_client.yaml")
    assert cfg.inference.policy_type == "client"
    assert cfg.inference.client_protocol == "msgpack_websocket"
    assert cfg.inference.client_action_dim == 8
    assert cfg.inference.client_state_dim == 8
    assert cfg.inference.client_execute_steps == 1
    assert cfg.env.obs_key_map["head_cam_h"]["handle"]["params"]["resize_wh"] == [848, 480]
    assert cfg.env.obs_key_map["wrist_cam_r"]["handle"]["params"]["resize_wh"] == [848, 480]
    assert "gripper" in cfg.env.obs_key_map


def test_open_loop_cli_rejects_smolvla():
    argv = [
        "open_loop_eval.py",
        "--policy-type",
        "smolvla",
        "--policy-path",
        "/tmp/model",
    ]
    with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
        try:
            parse_args()
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("SmolVLA must not be accepted by the open-loop CLI")
