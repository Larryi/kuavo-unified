from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("script", "context"),
    [
        ("build_classic.sh", "classic_env=/path/to/classic-env-context"),
        ("build_lingbot.sh", "lingbot_env=/path/to/lingbot-v1-env-context"),
        ("build_lingbot_v2.sh", "lingbot_v2_env=/path/to/lingbot-v2-env-context"),
    ],
)
def test_packed_environment_builders_have_offline_dry_run(script: str, context: str) -> None:
    result = subprocess.run(
        [str(ROOT / "docker" / script)],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert context in result.stdout
    assert "--secret" not in result.stdout


def test_openpi_builder_uses_pinned_submodule_dockerfile() -> None:
    result = subprocess.run(
        [str(ROOT / "docker/build_openpi.sh")],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Dockerfile.openpi" in result.stdout
    assert str(ROOT) in result.stdout


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("act", "--backend act"),
        ("diffusion", "--backend diffusion"),
        ("lingbot", "--backend lingbot"),
        ("lingbot_v2", "--backend lingbot_v2"),
    ],
)
def test_policy_worker_runner_dry_run(backend: str, expected: str) -> None:
    result = subprocess.run(
        [str(ROOT / "docker/run_policy_worker.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "BACKEND": backend,
            "MODEL_DIR": "/host/models",
            "DRY_RUN": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    assert "/host/models:/models:ro" in result.stdout
    assert "docker rm" not in result.stdout
    assert "docker rmi" not in result.stdout


def test_openpi_runner_requires_server_args_without_calling_docker() -> None:
    result = subprocess.run(
        [str(ROOT / "docker/run_policy_worker.sh")],
        cwd=ROOT,
        env={**os.environ, "BACKEND": "openpi", "DRY_RUN": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "SERVER_ARGS" in result.stderr


def test_lingbot_image_does_not_persist_credentials_or_ros_addresses() -> None:
    dockerfile = (ROOT / "Dockerfile.lingbot").read_text(encoding="utf-8")
    assert "kuavo_diag_env" not in dockerfile
    assert "/opt/kuavo_secrets" not in dockerfile
    assert "192.168." not in dockerfile


def test_conda_pack_wrapper_dry_run_is_non_overwriting() -> None:
    result = subprocess.run(
        [str(ROOT / "docker/package_conda_envs.sh"), "classic"],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1", "DOCKER_ENV_ROOT": "/safe/envs"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "TMPDIR=/safe/envs/.tmp" in result.stdout
    assert "conda-pack -n kdc_dev --ignore-editable-packages" in result.stdout
    assert "/safe/envs/classic/myenv.tar.gz" in result.stdout


def test_lingbot_v2_env_dry_run_resolves_wheel_after_torch() -> None:
    result = subprocess.run(
        [str(ROOT / "docker/create_lingbot_v2_env.sh")],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.index("torch==2.8.0") < result.stdout.index("flash_attn_wheel.py")
    assert "--resume --flash-attn-wheel" in result.stdout
