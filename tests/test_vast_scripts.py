from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from kuavo_train.train_lingbot_v2 import _optional_asset_args


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "vast" / "launch.sh"
REMOTE_RUNNER = ROOT / "scripts" / "vast" / "run_backend.sh"


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("dp", "configs/policy/dp_r1.yaml"),
        ("act", "configs/policy/act_config.yaml"),
        ("openpi", "pi05_base/params"),
        ("lingbot-v1", "Qwen/Qwen2.5-VL-3B-Instruct"),
        ("lingbot-v2", "Ruicheng/moge-2-vitb-normal"),
    ],
)
def test_launcher_dry_run_is_offline_and_model_aware(backend: str, expected: str) -> None:
    env = os.environ.copy()
    env.update(
        {
            "MODEL_BACKEND": backend,
            "DRY_RUN": "1",
            "HF_TOKEN": "do-not-print-hf",
            "WANDB_API_KEY": "do-not-print-wandb",
            "SERVERCHAN_SENDKEY": "do-not-print-serverchan",
            "VAST_API_KEY": "do-not-print-vast",
        }
    )
    result = subprocess.run(
        [str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    assert "SSH, rsync, secret upload, and remote execution were skipped" in result.stdout
    for secret in (
        "do-not-print-hf",
        "do-not-print-wandb",
        "do-not-print-serverchan",
        "do-not-print-vast",
    ):
        assert secret not in result.stdout
        assert secret not in result.stderr


def test_remote_runner_rejects_unknown_backend() -> None:
    result = subprocess.run(
        [str(REMOTE_RUNNER)],
        cwd=ROOT,
        env={**os.environ, "MODEL_BACKEND": "smolvla", "DRY_RUN": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Unsupported MODEL_BACKEND=smolvla" in result.stderr


def test_launcher_rejects_unsafe_remote_path_before_network() -> None:
    result = subprocess.run(
        [str(LAUNCHER)],
        cwd=ROOT,
        env={
            **os.environ,
            "MODEL_BACKEND": "dp",
            "DRY_RUN": "0",
            "REMOTE_ROOT": "/workspace/unsafe path",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Unsafe remote path" in result.stderr


def test_launcher_uses_scp_port_flag_and_keeps_env_separate(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "calls.log"
    for command in ("ssh", "rsync", "scp"):
        executable = fake_bin / command
        executable.write_text(
            "#!/bin/sh\n"
            f"printf '%s' '{command}' >> \"$CALL_LOG\"\n"
            "printf ' <%s>' \"$@\" >> \"$CALL_LOG\"\n"
            "printf '\\n' >> \"$CALL_LOG\"\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)

    private_env = tmp_path / "private.env"
    private_env.write_text('export HF_TOKEN="not-in-log"\n', encoding="utf-8")
    private_env.chmod(0o600)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CALL_LOG": str(call_log),
        "MODEL_BACKEND": "dp",
        "VAST_SSH_HOST": "example.invalid",
        "VAST_SSH_PORT": "2222",
        "VAST_ENV_FILE": str(private_env),
        "SYNC_ONLY": "1",
    }
    result = subprocess.run(
        [str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = call_log.read_text(encoding="utf-8")
    scp_call = next(line for line in calls.splitlines() if line.startswith("scp "))
    assert "<-P> <2222>" in scp_call
    assert "not-in-log" not in calls


def test_launcher_requires_private_env_permissions(tmp_path: Path) -> None:
    private_env = tmp_path / "private.env"
    private_env.write_text('export HF_TOKEN="hf_private_value_123"\n', encoding="utf-8")
    private_env.chmod(0o644)
    result = subprocess.run(
        [str(LAUNCHER)],
        cwd=ROOT,
        env={
            **os.environ,
            "MODEL_BACKEND": "act",
            "VAST_SSH_HOST": "example.invalid",
            "VAST_SSH_PORT": "2222",
            "VAST_ENV_FILE": str(private_env),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 3
    assert "mode 600 or 400" in result.stderr
    assert "hf_private_value_123" not in result.stdout
    assert "hf_private_value_123" not in result.stderr


def test_lingbot_v2_cloud_assets_override_developer_paths() -> None:
    args = _optional_asset_args(
        {
            "LINGBOT_V2_MOGE_PATH": "/workspace/models/moge/model.pt",
            "LINGBOT_V2_DEPTH_PATH": "/workspace/models/v2/depth/model.pt",
            "LINGBOT_V2_DINO_CKPT": "/workspace/models/v2/dino/teacher.pth",
            "LINGBOT_V2_DINO_CONFIG": "/workspace/models/v2/dino/config.yaml",
        }
    )

    assert args == [
        "--train.align_params.depth.moge_path",
        "/workspace/models/moge/model.pt",
        "--train.align_params.depth.morgbd_path",
        "/workspace/models/v2/depth/model.pt",
        "--train.align_params.video.ckpt_path",
        "/workspace/models/v2/dino/teacher.pth",
        "--train.align_params.video.config_path",
        "/workspace/models/v2/dino/config.yaml",
    ]
