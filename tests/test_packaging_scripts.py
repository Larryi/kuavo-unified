from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def make_dp_run(path: Path) -> None:
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "policy_preprocessor.json").write_text("{}", encoding="utf-8")
    (path / "policy_postprocessor.json").write_text("{}", encoding="utf-8")
    epoch = path / "epochbest"
    epoch.mkdir()
    (epoch / "model.safetensors").write_bytes(b"weights")


def test_base_image_export_is_opt_in(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            str(ROOT / "scripts/kuavo_base_image"),
            "--backend", "dp",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Base image: kuavo-classic:latest" in result.stdout
    assert "TAR export skipped" in result.stdout
    assert "docker save" not in result.stdout


def test_base_image_can_optionally_export_tar(tmp_path: Path) -> None:
    output = tmp_path / "classic.tar"
    result = subprocess.run(
        [
            str(ROOT / "scripts/kuavo_base_image"),
            "--backend", "act",
            "--save-tar", str(output),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"docker save -o {output}" in result.stdout
    assert not output.exists()


def test_base_image_distinguishes_docker_access_from_missing_image(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'permission denied while trying to connect to the Docker daemon' >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    result = subprocess.run(
        [str(ROOT / "scripts/kuavo_base_image"), "--backend", "dp"],
        cwd=ROOT,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 4
    assert "Docker daemon is not accessible" in result.stderr
    assert "Base image is missing" not in result.stderr


def test_inference_packager_routes_task_and_skips_tar_by_default(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run"
    make_dp_run(checkpoint)
    result = subprocess.run(
        [
            str(ROOT / "scripts/package_inference_image"),
            "--task", "task2",
            "--algorithm", "dp",
            "--checkpoint", str(checkpoint),
            "--checkpoint-subpath", "epochbest",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "任务路由: task2-dp -> dp" in result.stdout
    assert "TAR export skipped" in result.stdout
    assert not any(
        line.startswith("+ docker save") for line in result.stdout.splitlines()
    )


def test_inference_packager_rejects_wrong_pair_before_build(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            str(ROOT / "scripts/package_inference_image"),
            "--task", "task3",
            "--algorithm", "dp",
            "--checkpoint", str(tmp_path),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Unsupported task/algorithm pair" in result.stderr
