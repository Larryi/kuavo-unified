from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from tools.vast_bootstrap import (
    BootstrapError,
    options_without_forwarding,
    parse_ssh_command,
)
from tools.vast_job_wizard import (
    JOB_MATRIX,
    latest_openpi_checkpoint_step,
    normalize_dataset_mix,
    openpi_checkpoint_steps,
    resume_repo_has_training_state,
    select_training_datasets,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "vast" / "launch.sh"
REMOTE_RUNNER = ROOT / "scripts" / "vast" / "run_backend.sh"
JOB_LAUNCHER = ROOT / "scripts" / "vast" / "launch_job.sh"
STATUS = ROOT / "scripts" / "vast" / "status.sh"
BOOTSTRAP = ROOT / "scripts" / "vast" / "bootstrap_from_ssh"
RESTORE = ROOT / "scripts" / "vast" / "restore_and_launch.sh"
OPENPI_PIPELINE = (
    ROOT
    / "third_party"
    / "openpi-kuavo"
    / "scripts"
    / "vast"
    / "run_pi05_pipeline.sh"
)


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


def test_lingbot_v2_cloud_job_dry_run_is_available() -> None:
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
    assert result.returncode == 0, result.stderr
    assert "Backend: lingbot-v2" in result.stdout
    assert "Task: task2" in result.stdout


def test_vla_algorithms_are_routed_to_task1_and_task2() -> None:
    for algorithm in ("openpi", "lingbot-v1", "lingbot-v2"):
        assert JOB_MATRIX[algorithm] == ("task1", "task2")


def test_openpi_pipeline_has_no_default_gpu_model_lock() -> None:
    text = OPENPI_PIPELINE.read_text(encoding="utf-8")
    assert ': "${REQUIRE_GPU_NAME:=}"' in text
    assert ': "${REQUIRE_GPU_NAME:=A100}"' not in text
    assert ': "${CUDA_NVCC_VERSION:=auto}"' in text
    assert ': "${NORM_NUM_WORKERS:=0}"' in text
    assert ': "${NUM_WORKERS:=8}"' in text
    assert ': "${TRAIN_VIDEO_BACKEND:=torchcodec}"' in text
    assert ': "${TORCH_VERSION:=2.7.1}"' in text
    assert ': "${TORCHCODEC_VERSION:=0.5}"' in text
    assert "torch==${TORCH_VERSION}" in text
    assert "torch==${TORCH_VERSION}+cu128" not in text
    assert "TRAIN_GLOBAL_BATCH_SIZE=16" in text
    assert "Parallel norm-stat loading failed" in text
    assert "--state-action-only" in text
    assert "Reused OpenPI norm cache" in text

    assert "api.dataset_info(repo_id).sha" in text
    norm_script = (
        ROOT / "third_party" / "openpi-kuavo" / "scripts" / "compute_norm_stats.py"
    ).read_text(encoding="utf-8")
    assert "Vectorized norm stats:" in norm_script
    assert "episode_end_by_frame" in norm_script
    config_text = (
        ROOT
        / "third_party"
        / "openpi-kuavo"
        / "src"
        / "openpi"
        / "training"
        / "config.py"
    ).read_text(encoding="utf-8")
    assert "episodes: tuple[int, ...] | None = None" in config_text
    assert '"nvidia-cuda-nvcc-cu12==${CUDA_NVCC_VERSION}"' in text
    assert 'if [[ "${CUDA_NVCC_VERSION}" != "auto" ]]' in text
    assert '--lr-schedule.peak-lr "${PEAK_LR}"' in text
    assert '--lr-schedule.decay-steps "${LR_DECAY_STEPS}"' in text


def test_classic_environment_resolves_absolute_python_for_uv() -> None:
    text = REMOTE_RUNNER.read_text(encoding="utf-8")
    assert 'PYTHON_BIN="$(command -v "${PYTHON_BIN}")"' in text
    assert "print(sys.executable)" in text
    assert 'uv pip install --python "${PYTHON_BIN}"' in text
    assert 'CLASSIC_TORCH_VERSION:=2.7.1' in text
    assert 'CLASSIC_TORCHVISION_VERSION:=0.22.1' in text
    assert 'CLASSIC_PYTORCH_INDEX_URL:=https://download.pytorch.org/whl/cu128' in text
    assert "Blackwell {capability} requires a CUDA 12.8+ PyTorch wheel" in text
    assert '"sm_120" in torch.cuda.get_arch_list()' in text
    assert "memory_mb >= 30000" in text
    assert "TRAIN_BATCH_SIZE=40" in text


def test_restore_passes_persistent_credentials_file() -> None:
    text = RESTORE.read_text(encoding="utf-8")
    assert ".secrets/vast-credentials.json" in text
    assert '--credentials-file "${CREDENTIALS_FILE}"' in text


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


def test_resume_run_directory_does_not_double_prefix_run_id() -> None:
    text = REMOTE_RUNNER.read_text(encoding="utf-8")
    assert "run_directory_name()" in text
    assert 'run_dir="${output_base}/$(run_directory_name "${RESUME_RUN_ID}")"' in text
    assert 'run_dir="${output_base}/run_${RESUME_RUN_ID}"' not in text


def test_openpi_resume_uses_checkpoint_wandb_id_and_lr_tail() -> None:
    text = (ROOT / "tools" / "vast_job_wizard.py").read_text(encoding="utf-8")
    assert "W&B run ID 将从 checkpoint 的 wandb_id.txt 自动恢复" in text
    assert "留空则创建新 run" in text
    assert 'values["WANDB_RUN_ID"] = openpi_wandb_run_id' in text
    assert 'openpi_warmup_steps = 0' in text
    assert '"LR_TAIL_START_STEP": str(openpi_tail_start_step)' in text
    assert '"LR_TAIL_DECAY_STEPS": str(openpi_tail_decay_steps)' in text
    assert '"LR_TAIL_DECAY_LR": str(openpi_tail_decay_lr)' in text
    assert "选择 OpenPI checkpoint step（最新排在最前）" in text
    assert 'openpi_resume_state_mode = "full" if restore_learning_state else "weights_only"' in text
    assert '"OPENPI_RESUME_STATE_MODE": openpi_resume_state_mode' in text

    train_text = OPENPI_PIPELINE.parent.parent.joinpath("train.py").read_text(
        encoding="utf-8"
    )
    assert 'os.environ.get("WANDB_RUN_ID", "").strip()' in train_text
    pipeline_text = OPENPI_PIPELINE.read_text(encoding="utf-8")
    assert 'ignore_patterns=[".cache/**"]' in pipeline_text
    assert "shutil.rmtree(upload_cache)" in pipeline_text
    assert "Primary training failure (last 80 log lines)" in pipeline_text
    assert "skipping fallback upload" in pipeline_text
    assert 'resume_staging_dir="${resume_download_dir}.hf-download"' in pipeline_text
    assert ".kuavo_hf_resume_complete" in pipeline_text
    assert "staging.replace(download_dir)" in pipeline_text
    assert 'RESUME_STATE_MODE}" == "weights_only"' in pipeline_text
    assert 'allow_patterns=allow_patterns' in pipeline_text
    assert 'allow_patterns = [f"{selected_step}/params/**"]' in pipeline_text
    assert "HF_HUB_DISABLE_XET=1" in pipeline_text
    assert 'RESUME_HF_DOWNLOAD_WORKERS:=4' in pipeline_text
    assert 'retry "${RESUME_DOWNLOAD_RETRIES}" download_resume_checkpoint' in pipeline_text
    assert ': "${MODEL_REPO_PRIVATE:=0}"' in pipeline_text


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


def test_openpi_dataset_selection_adds_sources_one_by_one(monkeypatch) -> None:
    answers = iter(["1", "y", "2", "n"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    selected = select_training_datasets(
        ["owner/task1-sz", "owner/task1-bj"],
        allow_multiple=True,
    )
    assert selected == ["owner/task1-sz", "owner/task1-bj"]


@pytest.mark.parametrize(
    ("algorithm", "files", "expected"),
    [
        ("openpi", ["10000/_CHECKPOINT_METADATA"], True),
        ("openpi", ["10000/params/_METADATA"], False),
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


def test_latest_openpi_checkpoint_step_uses_finalized_step_marker() -> None:
    files = [
        "checkpoints/pi05_kuavo/run_a/9000/_CHECKPOINT_METADATA",
        "checkpoints/pi05_kuavo/run_a/10000/_CHECKPOINT_METADATA",
        "checkpoints/pi05_kuavo/run_a/11000/params/_METADATA",
    ]
    assert openpi_checkpoint_steps(files) == [9000, 10000]
    assert latest_openpi_checkpoint_step(files) == 10000


@pytest.mark.parametrize(
    ("backend", "task"),
    [
        ("dp", "task2"),
        ("act", "task3"),
        ("lingbot-v1", "task1"),
        ("lingbot-v2", "task2"),
    ],
)
def test_remote_runner_accepts_weighted_mix_for_supported_backends(
    backend: str, task: str
) -> None:
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
            "MODEL_BACKEND": backend,
            "TRAINING_TASK": task,
            "DATASET_MIX_JSON": mixture,
            "DRY_RUN": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Datasets: 2" in result.stdout


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
    assert 'VAST_PYPI_INDEX:=https://pypi.org/simple' in text
    runner = REMOTE_RUNNER.read_text(encoding="utf-8")
    assert 'PIP_INDEX_URL="${VAST_PYPI_INDEX}"' in runner
    assert "unset PIP_EXTRA_INDEX_URL UV_INDEX_URL UV_EXTRA_INDEX_URL" in runner
    pipeline = OPENPI_PIPELINE.read_text(encoding="utf-8")
    assert "mirrors.bfsu.edu.cn" not in pipeline
    assert ': "${VAST_PYPI_INDEX:=https://pypi.org/simple}"' in pipeline
    openpi_lock = (
        ROOT / "third_party" / "openpi-kuavo" / "uv.lock"
    ).read_text(encoding="utf-8")
    assert "mirrors.bfsu.edu.cn" not in openpi_lock


def test_secret_reader_echoes_asterisks() -> None:
    text = (ROOT / "tools" / "vast_job_wizard.py").read_text(encoding="utf-8")
    assert "def masked_input" in text
    assert 'sys.stdout.write("*")' in text
    assert "termios.tcsetattr" in text


def test_lingbot_v2_cloud_assets_are_forwarded_as_environment() -> None:
    launcher = (
        ROOT / "kuavo_train" / "train_lingbot_v2.py"
    ).read_text(encoding="utf-8")
    trainer = (
        ROOT / "kuavo_train" / "lingbot_v2" / "train_lingbotvla_lora.py"
    ).read_text(encoding="utf-8")
    for name in (
        "LINGBOT_V2_MOGE_PATH",
        "LINGBOT_V2_DEPTH_PATH",
        "LINGBOT_V2_DINO_CKPT",
        "LINGBOT_V2_DINO_CONFIG",
    ):
        assert name in launcher
        assert name in trainer
    assert "--train.align_params.depth.moge_path" not in launcher


def test_lingbot_v2_save_frequency_can_be_overridden() -> None:
    launcher = (
        ROOT / "kuavo_train" / "train_lingbot_v2.py"
    ).read_text(encoding="utf-8")
    assert 'os.getenv("TRAIN_SAVE_STEPS"' in launcher
    assert '"--train.save_steps"' in launcher
    assert "TRAIN_SAVE_STEPS must be a positive integer" in launcher


def test_lingbot_cloud_downloads_qwen_processor_without_base_weights() -> None:
    runner = REMOTE_RUNNER.read_text(encoding="utf-8")
    v1_pipeline = (
        ROOT / "scripts/run_lingbot_v1_full_pipeline.sh"
    ).read_text(encoding="utf-8")
    assert "download_hf_processor" in runner
    assert '"*processor_config.json"' in runner
    assert '"model-*.safetensors"' not in runner
    assert '--include \\\n        "config.json"' in v1_pipeline


def test_lingbot_v1_cloud_uses_blackwell_compatible_torch() -> None:
    pipeline = (
        ROOT / "scripts" / "run_lingbot_v1_full_pipeline.sh"
    ).read_text(encoding="utf-8")
    requirements = (
        ROOT / "requirements_lingbot_cloud.txt"
    ).read_text(encoding="utf-8")
    assert "--index-url https://download.pytorch.org/whl/cu128" in pipeline
    assert "--reinstall-package torch" in pipeline
    assert "--reinstall-package torchvision" in pipeline
    assert "torch==2.7.1+cu128" in pipeline
    assert "torchvision==0.22.1+cu128" in pipeline
    assert 'assert torch.version.cuda == "12.8"' in pipeline
    assert "capability in torch.cuda.get_arch_list()" in pipeline
    assert 'torch.ones(1, device="cuda").item()' in pipeline
    assert "official cu128" in requirements
    assert "cu126" not in requirements


def test_lingbot_v1_norm_accepts_trainer_only_config_fields() -> None:
    norm_script = (
        ROOT / "kuavo_train" / "lingbot" / "compute_mixture_norm.py"
    ).read_text(encoding="utf-8")
    assert "class NormTrainingArguments(TrainingArguments):" in norm_script
    for field_name in ("use_ema", "ignore_depth", "keep_last_checkpoints"):
        assert f"    {field_name}:" in norm_script
    assert (
        "train: NormTrainingArguments = "
        "field(default_factory=NormTrainingArguments)"
    ) in norm_script
    for variable, value in (
        ("LOCAL_RANK", "0"),
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_WORLD_SIZE", "1"),
    ):
        assert f'os.environ.setdefault("{variable}", "{value}")' in norm_script
    assert (
        "self.micro_batch_size * self.gradient_accumulation_steps"
        in norm_script
    )
    assert "super().__post_init__()" in norm_script


def test_lingbot_v1_norm_batch_is_independent_from_training_batch() -> None:
    pipeline = (
        ROOT / "scripts" / "run_lingbot_v1_full_pipeline.sh"
    ).read_text(encoding="utf-8")
    assert ': "${NORM_BATCH_SIZE:=128}"' in pipeline
    assert ': "${NORM_NUM_WORKERS:=8}"' in pipeline
    assert '--data.num_workers "${NORM_NUM_WORKERS}"' in pipeline
    assert '--train.micro_batch_size "${NORM_BATCH_SIZE}"' in pipeline
    assert "images=disabled" in pipeline
