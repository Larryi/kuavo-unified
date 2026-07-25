from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from tools.kuavo_docker import (
    BACKENDS,
    TASKS,
    UsageError,
    render_config,
    reject_sensitive_files,
    stage_classic_checkpoint,
    stage_qwen_processor,
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
    assert BACKENDS["lingbot-v2"].delivery_paused
    assert BACKENDS["lingbot-v2"].image == "kuavo-lingbot-v2:latest"
    assert TASKS["task2-dp"].config.endswith("kuavo_env.dp.task2.yaml")
    assert TASKS["task3-act"].config.endswith("kuavo_env.act.task3.yaml")


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


def test_release_qwen_bundle_excludes_base_weights(tmp_path: Path) -> None:
    qwen = tmp_path / "qwen"
    make_qwen_processor(qwen, with_weights=True)
    staged = tmp_path / "staged"
    stage_qwen_processor(qwen, staged)
    assert (staged / "config.json").is_file()
    assert (staged / "tokenizer.json").is_file()
    assert not list(staged.glob("*.safetensors"))


def test_release_classic_bundle_keeps_only_selected_epoch_and_processors(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    make_classic_checkpoint(run)
    epochbest = run / "epochbest"
    epochbest.mkdir()
    (epochbest / "model.safetensors").write_bytes(b"best")
    epochold = run / "epochold"
    epochold.mkdir()
    (epochold / "model.safetensors").write_bytes(b"old")
    (run / "learning_state.pth").write_bytes(b"optimizer")
    staged = tmp_path / "staged"
    stage_classic_checkpoint(run, staged, "epochbest")
    assert (staged / "epochbest/model.safetensors").read_bytes() == b"best"
    assert (staged / "policy_preprocessor.json").is_file()
    assert not (staged / "epochold").exists()
    assert not (staged / "learning_state.pth").exists()


def test_lingbot_incremental_checkpoint_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(UsageError, match="完整 HF checkpoint"):
        validate_checkpoint(BACKENDS["lingbot-v1"], checkpoint)


def test_classic_nonportable_epoch_processor_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "epochbest"
    make_classic_checkpoint(checkpoint)
    (checkpoint / "policy_preprocessor.json").write_text(
        """
        {
          "steps": [
            {"class": "__main__.AugmentationProcessorStep", "config": {}}
          ]
        }
        """,
        encoding="utf-8",
    )
    with pytest.raises(UsageError, match="run 根目录"):
        validate_checkpoint(BACKENDS["act"], checkpoint)


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


def test_openpi_shell_prints_structured_server_arguments(tmp_path: Path) -> None:
    checkpoint = tmp_path / "openpi" / "45000"
    params = checkpoint / "params"
    params.mkdir(parents=True)
    (params / "_METADATA").write_text("{}", encoding="utf-8")
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_text("tokenizer", encoding="utf-8")

    result = subprocess.run(
        [
            str(ROOT / "scripts/kuavo_docker"),
            "shell",
            "--backend", "openpi",
            "--checkpoint", str(params),
            "--tokenizer", str(tokenizer),
            "--openpi-config", "pi05_kuavo",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "自动改用 checkpoint step" in result.stdout
    assert f"{checkpoint.resolve()}:/models/checkpoint:ro" in result.stdout
    assert "OPENPI_POLICY_CONFIG=pi05_kuavo" in result.stdout
    assert "OPENPI_POLICY_DIR=/models/checkpoint" in result.stdout
    assert "SERVER_ARGS=" not in result.stdout


def test_release_dry_run_builds_image_without_saving_tar(tmp_path: Path) -> None:
    checkpoint = tmp_path / "classic"
    make_classic_checkpoint(checkpoint)
    result = subprocess.run(
        [
            str(ROOT / "scripts/kuavo_docker"),
            "release",
            "--backend", "dp",
            "--task", "task2-dp",
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
    assert "任务路由: task2-dp -> dp" in result.stdout
    assert "--build-arg KUAVO_TASK=task2-dp" in result.stdout
    command_line = next(line for line in result.stdout.splitlines() if line.startswith("+ docker"))
    assert "docker save" not in command_line


def test_release_uses_bounded_build_log(tmp_path: Path) -> None:
    checkpoint = tmp_path / "classic"
    make_classic_checkpoint(checkpoint)
    build_log = tmp_path / "release.log"
    result = subprocess.run(
        [
            str(ROOT / "scripts/kuavo_docker"),
            "release",
            "--backend", "dp",
            "--task", "task2-dp",
            "--checkpoint", str(checkpoint),
            "--build-log", str(build_log),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not build_log.exists()


def test_lingbot_v2_delivery_is_blocked_while_paused(tmp_path: Path) -> None:
    checkpoint = tmp_path / "hf_ckpt"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "lingbotvla_cli.yaml").write_text("model: {}", encoding="utf-8")
    qwen = tmp_path / "qwen"
    make_qwen_processor(qwen)
    norm = tmp_path / "norm_stats.json"
    norm.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [
            str(ROOT / "scripts/kuavo_docker"),
            "shell",
            "--backend", "lingbot-v2",
            "--checkpoint", str(checkpoint),
            "--qwen", str(qwen),
            "--norm-stats", str(norm),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "交付适配已按操作者决定暂缓" in result.stderr


def test_backend_task_mismatch_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "classic"
    make_classic_checkpoint(checkpoint)
    result = subprocess.run(
        [
            str(ROOT / "scripts/kuavo_docker"),
            "release",
            "--backend", "act",
            "--task", "task2-dp",
            "--checkpoint", str(checkpoint),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "绑定 backend=dp" in result.stderr


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
    assert (
        "COPY --from=release_manifest /release_manifest.json "
        "/etc/kuavo/release_manifest.json"
        in dockerfile
    )
    assert "TOKEN" not in dockerfile
    assert "PASSWORD" not in dockerfile
