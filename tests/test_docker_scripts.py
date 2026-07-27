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


def test_classic_image_preloads_official_resnet18_checkpoint() -> None:
    result = subprocess.run(
        [str(ROOT / "docker/build_classic.sh")],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "torch_checkpoints=/path/to/torch-checkpoint-context" in result.stdout

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY --from=torch_checkpoints /resnet18-f37072fd.pth" in dockerfile
    assert (
        "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
        in dockerfile
    )


def test_openpi_builder_uses_classic_base_and_pinned_source_context() -> None:
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
    assert "CLASSIC_BASE_IMAGE=kuavo-classic:latest" in result.stdout
    assert "openpi_src=" in result.stdout
    assert str(ROOT) in result.stdout


def test_openpi_image_uses_bfsu_for_uv_packages() -> None:
    dockerfile = (ROOT / "Dockerfile.openpi").read_text(encoding="utf-8")
    assert "UV_DEFAULT_INDEX=https://mirrors.bfsu.edu.cn/pypi/web/simple" in dockerfile
    assert "conda create -p /opt/openpi-python" in dockerfile
    assert "mirrors.bfsu.edu.cn/anaconda/cloud/conda-forge" in dockerfile
    assert "python=3.11.9 pip -y" in dockerfile
    assert "uv venv --python /opt/openpi-python/bin/python" in dockerfile


def test_openpi_delivery_image_contains_ros_and_is_not_ubuntu_2204_worker() -> None:
    dockerfile = (ROOT / "Dockerfile.openpi").read_text(encoding="utf-8")
    assert "FROM ${CLASSIC_BASE_IMAGE}" in dockerfile
    assert "import rospy" in dockerfile
    assert "PolicyClient" in dockerfile
    assert "ubuntu22.04" not in dockerfile
    assert "chmod 755 /root" in dockerfile
    assert "jax_plugins.xla_cuda12" in dockerfile
    assert "jax.devices()" not in dockerfile
    assert "chmod -R a+rX /root/kuavo_data_challenge" not in dockerfile


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


def test_openpi_runner_uses_single_ros_capable_image_entrypoint() -> None:
    result = subprocess.run(
        [str(ROOT / "docker/run_policy_worker.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "BACKEND": "openpi",
            "SERVER_ARGS": "policy:checkpoint --policy.config=pi05_kuavo",
            "DRY_RUN": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "kuavo-openpi:latest" in result.stdout
    assert "start_openpi_ros.sh" in result.stdout


def test_openpi_stack_launcher_does_not_eval_server_arguments() -> None:
    launcher = (ROOT / "docker/start_openpi_ros.sh").read_text(encoding="utf-8")
    assert "shlex.split" in launcher
    assert "OPENPI_POLICY_CONFIG" in launcher
    assert "OPENPI_POLICY_DIR" in launcher
    assert 'if [[ "${OPENPI_PORT}" != "8000" ]]' in launcher
    assert 'server_argv=("--port=${OPENPI_PORT}" "${server_argv[@]}")' in launcher
    assert '"--policy.config=${OPENPI_POLICY_CONFIG}"' in launcher
    assert '"--policy.dir=${OPENPI_POLICY_DIR}"' in launcher
    assert '"$@" &' not in launcher
    assert '\n"$@"\n' in launcher
    assert 'bash -lc "exec scripts/kuavo_openpi' not in launcher


def test_lingbot_v2_launcher_keeps_ros_client_in_foreground() -> None:
    launcher = (ROOT / "docker/start_lingbot_v2_ros.sh").read_text(encoding="utf-8")
    assert "/opt/kuavo-env/bin/python tools/policy_worker.py" in launcher
    assert "--backend lingbot_v2" in launcher
    assert "source /opt/ros/noetic/setup.bash" in launcher
    assert 'source "${REPO_ROOT}/myenv/bin/activate"' in launcher
    assert '\n"$@"\n' in launcher


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
    assert "--resume [--flash-attn-wheel" in result.stdout
    assert "source build otherwise" in result.stdout
    assert "FLASH_ATTN_INSTALL_MODE=defer" in result.stdout


def test_lingbot_v2_env_streams_long_torch_install() -> None:
    text = (ROOT / "docker/create_lingbot_v2_env.sh").read_text(encoding="utf-8")
    assert 'conda run --no-capture-output -n "${ENV_NAME}" python -m pip install' in text
    assert "Installing PyTorch 2.8.0 CUDA stack; progress follows live." in text


def test_lingbot_v2_uses_moge_pinned_utils3d_not_pypi_namesake() -> None:
    text = (
        ROOT / "third_party" / "lingbot-vla-v2" / "tools" / "create_train_env.sh"
    ).read_text(encoding="utf-8")
    assert "EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183" in text
    assert 'version("utils3d") == "1.3"' in text
    assert "morgbd_clean/3rd/utils3d" not in text


def test_lingbot_v2_image_builds_flash_attention_on_focal() -> None:
    dockerfile = (ROOT / "Dockerfile.lingbot_v2").read_text(encoding="utf-8")
    build_script = (ROOT / "docker/build_lingbot_v2.sh").read_text(encoding="utf-8")
    assert "nvidia/cuda:12.8.1-devel-ubuntu20.04" in dockerfile
    assert "FLASH_ATTENTION_FORCE_BUILD=TRUE" in dockerfile
    assert "python -m pip wheel" in dockerfile
    assert "assert_glibc_compatible" in dockerfile
    assert "COPY --from=lingbot-v2-flash-builder /opt/kuavo-env" in dockerfile
    assert "COPY --from=lingbot_v2_source . ./third_party/lingbot-vla-v2" in dockerfile
    assert "--build-context \"lingbot_v2_source=" in build_script


def test_lingbot_training_compile_defaults_are_enabled() -> None:
    v1_pipeline = (ROOT / "scripts/run_lingbot_v1_full_pipeline.sh").read_text()
    v1_config = (ROOT / "configs/policy/lingbot/task1_345_full.yaml").read_text()
    v2_config = (ROOT / "configs/policy/lingbot_v2/kuavo_lora.yaml").read_text()
    v2_task2 = (
        ROOT / "configs/policy/lingbot_v2/kuavo_lora_task2_bimanual.yaml"
    ).read_text()
    assert ': "${USE_COMPILE:=true}"' in v1_pipeline
    assert "use_compile: true" in v1_config
    assert "use_compile: true" in v2_config
    assert "use_compile: true" in v2_task2
    assert "save_hf_weights: true" in v2_task2
    assert "keep_last_checkpoints: 1" in v2_task2
