from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from tools.kuavo_docker import (
    BACKENDS,
    UsageError,
    render_config,
    reject_sensitive_files,
    validate_checkpoint,
    validate_qwen_bundle,
)


ROOT = Path(__file__).resolve().parents[1]


def make_classic_checkpoint(path: Path) -> None:
    path.mkdir()
    (path / "policy_preprocessor.json").write_text("{}", encoding="utf-8")
    (path / "policy_postprocessor.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")


def make_qwen_processor(path: Path, with_weights: bool = False) -> None:
    path.mkdir()
    for name in ("config.json", "tokenizer.json", "preprocessor_config.json"):
        (path / name).write_text("{}", encoding="utf-8")
    if with_weights:
        (path / "model-00001-of-00001.safetensors").write_bytes(b"weights")


def test_backend_routes_share_classic_without_merging_runtimes() -> None:
    assert BACKENDS["act"].image == BACKENDS["dp"].image == "kuavo-classic:latest"
    assert BACKENDS["openpi"].image == "kuavo-openpi:latest"
    assert BACKENDS["openpi"].tokenizer_required
    assert BACKENDS["lingbot-v1"].qwen_required
    assert not BACKENDS["lingbot-v2"].ros_ready


def test_qwen_processor_bundle_does_not_require_base_weights(tmp_path: Path) -> None:
    qwen = tmp_path / "qwen"
    make_qwen_processor(qwen)
    assert validate_qwen_bundle(qwen) == []


def test_qwen_weight_files_are_warned_but_not_deleted(tmp_path: Path) -> None:
    qwen = tmp_path / "qwen"
    make_qwen_processor(qwen, with_weights=True)
    warnings = validate_qwen_bundle(qwen)
    assert "权重文件" in warnings[0]
    assert (qwen / "model-00001-of-00001.safetensors").is_file()


def test_lingbot_incremental_checkpoint_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(UsageError, match="完整 HF checkpoint"):
        validate_checkpoint(BACKENDS["lingbot-v1"], checkpoint)


def test_release_assets_with_private_key_are_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "id_rsa").write_text("private", encoding="utf-8")
    with pytest.raises(UsageError, match="疑似凭据"):
        reject_sensitive_files([checkpoint])


def test_config_placeholder_is_rendered_to_container_path(tmp_path: Path) -> None:
    destination = tmp_path / "rendered.yaml"
    render_config(
        ROOT / "configs/deploy/kuavo_env.act.yaml",
        destination,
        pretrained_path="/models/checkpoint/epochlast",
    )
    text = destination.read_text(encoding="utf-8")
    assert "__PRETRAINED_PATH__" not in text
    assert 'pretrained_path: "/models/checkpoint/epochlast"' in text


def test_shell_dry_run_mounts_checkpoint_and_never_runs_auto_test(tmp_path: Path) -> None:
    checkpoint = tmp_path / "classic"
    make_classic_checkpoint(checkpoint)
    result = subprocess.run(
        [
            str(ROOT / "scripts/kuavo_docker"),
            "shell",
            "--backend", "act",
            "--checkpoint", str(checkpoint),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"{checkpoint.resolve()}:/models/checkpoint:ro" in result.stdout
    assert (
        "/root/kuavo_data_challenge/configs/deploy/kuavo_env.yaml:ro"
        in result.stdout
    )
    assert "--entrypoint bash kuavo-classic:latest" in result.stdout
    assert "容器内人工推理命令" in result.stdout
    assert "--config configs/deploy/kuavo_env.yaml" in result.stdout
    command_line = next(line for line in result.stdout.splitlines() if line.startswith("+ docker run"))
    assert "script_auto_test.py" not in command_line


def test_release_dry_run_builds_image_without_saving_tar(tmp_path: Path) -> None:
    checkpoint = tmp_path / "classic"
    make_classic_checkpoint(checkpoint)
    result = subprocess.run(
        [
            str(ROOT / "scripts/kuavo_docker"),
            "release",
            "--backend", "dp",
            "--checkpoint", str(checkpoint),
            "--tag", "kuavo-dp-release:test",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Dockerfile.release" in result.stdout
    assert "kuavo-dp-release:test" in result.stdout
    command_line = next(line for line in result.stdout.splitlines() if line.startswith("+ docker"))
    assert "docker save" not in command_line


def test_lingbot_v2_release_is_blocked_until_ros_image_exists() -> None:
    result = subprocess.run(
        [
            str(ROOT / "scripts/kuavo_docker"),
            "release",
            "--backend", "lingbot-v2",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "尚未包含 ROS Noetic/KuavoBaseEnv" in result.stderr


def test_export_is_the_only_command_that_calls_docker_save(tmp_path: Path) -> None:
    output = tmp_path / "release.tar"
    result = subprocess.run(
        [
            str(ROOT / "scripts/kuavo_docker"),
            "export",
            "--image", "kuavo-act-release:test",
            "--output", str(output),
            "--yes",
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


def test_release_dockerfile_keeps_manual_entrypoint_and_no_secrets() -> None:
    dockerfile = (ROOT / "Dockerfile.release").read_text(encoding="utf-8")
    assert "ENTRYPOINT []" in dockerfile
    assert 'CMD ["bash"]' in dockerfile
    assert (
        "COPY --from=deploy_config /kuavo_env.yaml "
        "/root/kuavo_data_challenge/configs/deploy/kuavo_env.yaml"
        in dockerfile
    )
    assert "TOKEN" not in dockerfile
    assert "PASSWORD" not in dockerfile
