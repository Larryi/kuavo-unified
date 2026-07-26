from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from kuavo_train.train_lingbot_v2 import _optional_asset_args
from tools.vast_bootstrap import (
    BootstrapError,
    options_without_forwarding,
    parse_ssh_command,
)
from tools.vast_job_wizard import (
    normalize_dataset_mix,
    resume_repo_has_training_state,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "vast" / "launch.sh"
REMOTE_RUNNER = ROOT / "scripts" / "vast" / "run_backend.sh"
JOB_LAUNCHER = ROOT / "scripts" / "vast" / "launch_job.sh"
STATUS = ROOT / "scripts" / "vast" / "status.sh"
BOOTSTRAP = ROOT / "scripts" / "vast" / "bootstrap_from_ssh"
RESTORE = ROOT / "scripts" / "vast" / "restore_and_launch.sh"


@pytest.mark.parametrize(
    ("backend", "task", "expected"),
    [
        ("dp", "task2", "configs/policy/dp_r2_h100.yaml"),
        ("act", "task3", "configs/policy/act_config.yaml"),
        ("openpi", "task1", "pi05_base/params"),
        ("lingbot-v1", "task1", "Qwen/Qwen2.5-VL-3B-Instruct"),
    ],
)
def test_launcher_dry_run_is_offline_and_model_aware(
    backend: str, task: str, expected: str,
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "MODEL_BACKEND": backend,
            "TRAINING_TASK": task,
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
        env={
            **os.environ,
            "MODEL_BACKEND": "smolvla",
            "TRAINING_TASK": "task1",
            "DRY_RUN": "1",
        },
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
            "TRAINING_TASK": "task2",
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
        "TRAINING_TASK": "task2",
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
            "TRAINING_TASK": "task3",
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


def test_launcher_rejects_mismatched_task_backend_before_network() -> None:
    result = subprocess.run(
        [str(LAUNCHER)],
        cwd=ROOT,
        env={
            **os.environ,
            "MODEL_BACKEND": "dp",
            "TRAINING_TASK": "task3",
            "DRY_RUN": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Unsupported VastAI task/backend pair" in result.stderr


def test_lingbot_v2_cloud_job_is_paused() -> None:
    result = subprocess.run(
        [str(LAUNCHER)],
        cwd=ROOT,
        env={
            **os.environ,
            "MODEL_BACKEND": "lingbot-v2",
            "TRAINING_TASK": "task2",
            "DRY_RUN": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "cloud training integration is paused" in result.stderr


def test_job_launcher_dry_run_describes_resume_without_network() -> None:
    result = subprocess.run(
        [
            str(JOB_LAUNCHER),
            "--task", "task2",
            "--algorithm", "dp",
            "--resume-repo", "owner/full-state",
            "--resume-run-id", "run_20260711_011552",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Resume: hf run=run_20260711_011552" in result.stdout
    assert "SSH, rsync, secret upload, and remote execution were skipped" in result.stdout


def test_status_reader_is_bounded_and_reports_gpu() -> None:
    text = STATUS.read_text(encoding="utf-8")
    assert 'TAIL_LINES:=30' in text
    assert 'tail -n "${tail_lines}"' in text
    assert "nvidia-smi --query-gpu=" in text
    assert "cat \"${status_file}\"" in text


def test_ssh_parser_accepts_forwarding_after_target() -> None:
    options, target = parse_ssh_command(
        "ssh -p 12345 root@1.2.3.4 -L 8080:localhost:8080"
    )
    assert target == "root@1.2.3.4"
    assert options == ["-p", "12345", "-L", "8080:localhost:8080"]
    assert options_without_forwarding(options) == ["-p", "12345"]


def test_ssh_parser_rejects_remote_command() -> None:
    with pytest.raises(BootstrapError, match="Remote shell commands"):
        parse_ssh_command("ssh root@example.invalid rm -rf /tmp/example")


def test_bootstrap_dry_run_uploads_only_remote_git_script() -> None:
    result = subprocess.run(
        [
            str(BOOTSTRAP),
            "--ssh-command",
            "ssh -p 12345 root@1.2.3.4 -L 8080:localhost:8080",
            "--dry-run",
            "--no-launch",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "remote_clone_and_restore.sh" in result.stdout
    assert "--repo-url https://github.com/Larryi/kuavo-unified.git" in result.stdout
    assert "--git-ref codex/unified-stack" in result.stdout
    assert "Git publication preflight skipped in dry-run mode" in result.stdout
    assert "--exclude .git" not in result.stdout
    assert f"{ROOT}/ root@1.2.3.4:" not in result.stdout
    assert "all pinned submodules" in result.stdout


def test_bootstrap_worktree_sync_is_explicit_fallback() -> None:
    result = subprocess.run(
        [
            str(BOOTSTRAP),
            "--ssh-command",
            "ssh -p 12345 root@1.2.3.4",
            "--sync-working-tree",
            "--dry-run",
            "--no-launch",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Explicit fallback enabled" in result.stdout
    assert "--exclude .git" in result.stdout
    assert ".vast_sync_manifest.json" in result.stdout


def test_dataset_mix_normalizes_weights() -> None:
    mixture = normalize_dataset_mix(
        ["owner/task1-sz", "owner/task1-bj"],
        [25, 75],
    )
    assert [item.weight for item in mixture] == [0.25, 0.75]
    assert [item.repo_id for item in mixture] == [
        "owner/task1-sz",
        "owner/task1-bj",
    ]


@pytest.mark.parametrize(
    ("algorithm", "files", "expected"),
    [
        ("openpi", ["10000/params/_METADATA"], True),
        ("lingbot-v1", ["checkpoints/global_step_15000/.metadata"], True),
        ("dp", ["learning_state.pth"], True),
        ("act", ["model.safetensors"], False),
    ],
)
def test_resume_repository_state_detection(
    algorithm: str,
    files: list[str],
    expected: bool,
) -> None:
    assert resume_repo_has_training_state(algorithm, files) is expected


def test_remote_runner_rejects_weighted_mix_for_classic_before_network() -> None:
    mixture = json.dumps(
        [
            {"repo_id": "owner/a", "weight": 0.5},
            {"repo_id": "owner/b", "weight": 0.5},
        ]
    )
    result = subprocess.run(
        [str(REMOTE_RUNNER)],
        cwd=ROOT,
        env={
            **os.environ,
            "MODEL_BACKEND": "dp",
            "TRAINING_TASK": "task2",
            "DATASET_MIX_JSON": mixture,
            "DRY_RUN": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "weighted virtual mixtures are supported by OpenPI" in result.stderr


def test_remote_runner_accepts_weighted_mix_for_openpi_dry_run() -> None:
    mixture = json.dumps(
        [
            {"repo_id": "owner/task1-sz", "weight": 25},
            {"repo_id": "owner/task1-bj", "weight": 75},
        ]
    )
    result = subprocess.run(
        [str(REMOTE_RUNNER)],
        cwd=ROOT,
        env={
            **os.environ,
            "MODEL_BACKEND": "openpi",
            "TRAINING_TASK": "task1",
            "DATASET_MIX_JSON": mixture,
            "DRY_RUN": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Datasets: 2" in result.stdout
    assert "no network" in result.stdout


def test_restore_uses_official_pypi_and_private_env() -> None:
    text = RESTORE.read_text(encoding="utf-8")
    assert "https://pypi.org/simple" in text
    assert "mirrors.bfsu.edu.cn" not in text
    assert "umask 077" in text
    assert "vast_job_wizard.py" in text


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
