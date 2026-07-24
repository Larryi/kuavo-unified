import contextlib
import io
import sys
from pathlib import Path
from unittest.mock import patch

from kuavo_deploy.config import ConfigInference
from tools.open_loop_eval import parse_args


def test_deploy_config_accepts_only_delivery_backends():
    for policy_type in ("act", "diffusion", "lingbot", "lingbot_v2"):
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
