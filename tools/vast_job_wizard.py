#!/usr/bin/env python3
"""Interactive, secret-safe VastAI training job configurator."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from getpass import getpass
import json
import os
from pathlib import Path
import re
import shlex
import sys
import termios
import tty
from typing import Iterable


JOB_MATRIX = {
    "openpi": "task1",
    "lingbot-v1": "task1",
    "dp": "task2",
    "act": "task3",
}
RESUME_MARKERS = {
    "openpi": ("params/_METADATA",),
    "lingbot-v1": ("checkpoints/global_step_",),
    "dp": ("learning_state.pth", "epochlatest/", "training_latest_state.pth"),
    "act": ("learning_state.pth", "epochlatest/", "training_latest_state.pth"),
}
REPO_ID = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class DatasetSelection:
    name: str
    repo_id: str
    weight: float


def normalize_dataset_mix(repo_ids: list[str], weights: list[float]) -> list[DatasetSelection]:
    if not repo_ids or len(repo_ids) != len(weights):
        raise ValueError("Dataset repositories and weights must be non-empty and aligned")
    if len(set(repo_ids)) != len(repo_ids):
        raise ValueError("The same dataset repository cannot be selected twice")
    if any(not REPO_ID.fullmatch(repo_id) for repo_id in repo_ids):
        raise ValueError("Every dataset must use owner/repository form")
    if any(weight <= 0 for weight in weights):
        raise ValueError("Dataset weights must be positive")
    total = sum(weights)
    return [
        DatasetSelection(
            name=f"source_{index + 1:02d}_{repo_id.rsplit('/', 1)[-1]}",
            repo_id=repo_id,
            weight=weight / total,
        )
        for index, (repo_id, weight) in enumerate(zip(repo_ids, weights, strict=True))
    ]


def resume_repo_has_training_state(algorithm: str, files: Iterable[str]) -> bool:
    markers = RESUME_MARKERS[algorithm]
    return any(any(marker in path for marker in markers) for path in files)


def choose(prompt: str, values: list[str]) -> str:
    for index, value in enumerate(values, 1):
        print(f"  {index}. {value}")
    while True:
        raw = input(f"{prompt} [1-{len(values)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(values):
            return values[int(raw) - 1]
        print("请输入有效编号。")


def yes_no(prompt: str, *, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suffix}: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


def masked_input(prompt: str) -> str:
    """Read an ASCII secret while echoing one asterisk per entered character."""
    if not sys.stdin.isatty():
        return getpass(prompt)
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    secret = bytearray()
    sys.stdout.write(prompt)
    sys.stdout.flush()
    try:
        tty.setraw(fd)
        while True:
            char = os.read(fd, 1)
            if char in {b"\r", b"\n"}:
                break
            if char == b"\x03":
                raise KeyboardInterrupt
            if char in {b"\x08", b"\x7f"}:
                if secret:
                    secret.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if char == b"\x15":
                while secret:
                    secret.pop()
                    sys.stdout.write("\b \b")
                sys.stdout.flush()
                continue
            if 32 <= char[0] <= 126:
                secret.extend(char)
                sys.stdout.write("*")
                sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
        sys.stdout.write("\n")
        sys.stdout.flush()
    return secret.decode("ascii")


def select_repositories(
    prompt: str,
    available: list[str],
    *,
    multiple: bool,
) -> list[str]:
    if available:
        for index, repo_id in enumerate(available, 1):
            print(f"  {index}. {repo_id}")
    hint = "编号或 owner/repo，多个用逗号分隔" if multiple else "编号或 owner/repo"
    while True:
        raw = input(f"{prompt}（{hint}）: ").strip()
        tokens = [part.strip() for part in raw.split(",") if part.strip()]
        selected: list[str] = []
        try:
            for token in tokens:
                repo_id = available[int(token) - 1] if token.isdigit() else token
                if not REPO_ID.fullmatch(repo_id):
                    raise ValueError(repo_id)
                selected.append(repo_id)
            if selected and (multiple or len(selected) == 1):
                return selected
        except (IndexError, ValueError):
            pass
        print("仓库选择无效。")


def select_training_datasets(
    available: list[str],
    *,
    allow_multiple: bool,
) -> list[str]:
    selected: list[str] = []
    if allow_multiple:
        print("OpenPI 支持多个同构数据集的虚拟加权混合。请逐个添加。")
    else:
        print("当前算法只支持一个训练数据集。")
    while True:
        repo_id = select_repositories(
            "选择或输入训练数据集",
            available,
            multiple=False,
        )[0]
        if repo_id in selected:
            print("该数据集已经添加，请选择其他仓库。")
            continue
        selected.append(repo_id)
        print(f"已添加 {len(selected)} 个数据集：{repo_id}")
        if not allow_multiple or not yes_no("继续添加另一个数据集？"):
            return selected


def list_owned_repositories(api, identity: dict, *, repo_type: str) -> list[str]:
    authors = [identity["name"]]
    authors.extend(
        org["name"] if isinstance(org, dict) else str(org)
        for org in identity.get("orgs", [])
    )
    found: set[str] = set()
    for author in authors:
        iterator = (
            api.list_datasets(author=author, limit=100)
            if repo_type == "dataset"
            else api.list_models(author=author, limit=100)
        )
        found.update(item.id for item in iterator)
    return sorted(found)


def shell_exports(values: dict[str, str]) -> str:
    lines = [
        "# Generated by tools/vast_job_wizard.py; contains secrets.",
        "umask 077",
    ]
    lines.extend(f"export {key}={shlex.quote(value)}" for key, value in values.items())
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-env", required=True)
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required; run restore_and_launch.sh") from exc

    algorithm = choose("选择算法", list(JOB_MATRIX))
    task = JOB_MATRIX[algorithm]
    print(f"任务路由：{task} + {algorithm}")

    token = masked_input("HF token（以 * 回显）: ").strip()
    if not token:
        raise SystemExit("HF token 不能为空")
    api = HfApi(token=token)
    identity = api.whoami()
    print(f"HF 身份验证成功：{identity['name']}")

    datasets = list_owned_repositories(api, identity, repo_type="dataset")
    allow_multiple = algorithm == "openpi"
    selected = select_training_datasets(
        datasets,
        allow_multiple=allow_multiple,
    )
    if len(selected) > 1 and not allow_multiple:
        raise SystemExit(f"{algorithm} 当前只支持一个数据集")
    weights = [1.0]
    if len(selected) > 1:
        while True:
            raw = input(
                f"依次输入 {len(selected)} 个正数配比（逗号分隔，如 55,20,25）: "
            )
            try:
                weights = [float(part.strip()) for part in raw.split(",")]
                mixture = normalize_dataset_mix(selected, weights)
                break
            except ValueError as exc:
                print(f"配比无效：{exc}")
    else:
        mixture = normalize_dataset_mix(selected, weights)

    models = list_owned_repositories(api, identity, repo_type="model")
    output_repo = select_repositories(
        "选择或输入训练输出模型仓库",
        models,
        multiple=False,
    )[0]
    resume_mode = "none"
    resume_repo = ""
    resume_run_id = ""
    if yes_no("是否从 HF 完整训练状态继续训练？"):
        resume_repo = select_repositories(
            "选择或输入 resume 模型仓库",
            models,
            multiple=False,
        )[0]
        files = api.list_repo_files(resume_repo, repo_type="model")
        if not resume_repo_has_training_state(algorithm, files):
            markers = ", ".join(RESUME_MARKERS[algorithm])
            raise SystemExit(
                f"仓库 {resume_repo} 未找到 {algorithm} 完整训练状态标记：{markers}"
            )
        resume_run_id = input("输入原训练 RUN_ID（用于恢复到同一输出目录）: ").strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]+", resume_run_id):
            raise SystemExit("RUN_ID 只能包含字母、数字、点、下划线和连字符")
        resume_mode = "hf"

    wandb_key = masked_input("W&B API key（可留空，以 * 回显）: ").strip()
    wandb_project = input("W&B project [kuavo-training]: ").strip() or "kuavo-training"
    serverchan = masked_input("ServerChan SendKey（可留空，以 * 回显）: ").strip()
    vast_instance = input("VastAI instance ID（自动关机时必填，可留空）: ").strip()
    vast_key = masked_input("VastAI API key（自动关机时必填，以 * 回显）: ").strip()
    auto_stop = "1" if vast_instance and vast_key and yes_no("成功后自动关闭 VastAI 实例？", default=True) else "0"
    gpu_ids = input("CUDA GPU IDs [0]: ").strip() or "0"
    if not re.fullmatch(r"\d+(,\d+)*", gpu_ids):
        raise SystemExit("GPU IDs 格式应为 0 或 0,1,2")
    gpu_count = str(len(gpu_ids.split(",")))
    run_id = (
        resume_run_id
        if resume_mode == "hf"
        else input("新 RUN_ID（留空自动生成）: ").strip()
    )

    mix_payload = [asdict(item) for item in mixture]
    values = {
        "MODEL_BACKEND": algorithm,
        "TRAINING_TASK": task,
        "HF_TOKEN": token,
        "DATASET_REPO": mixture[0].repo_id,
        "DATASET_MIX_JSON": json.dumps(mix_payload, separators=(",", ":")),
        "MODEL_REPO": output_repo,
        "MODEL_REPO_PRIVATE": "1",
        "RESUME_MODE": resume_mode,
        "RESUME_REPO": resume_repo,
        "RESUME_RUN_ID": resume_run_id,
        "WANDB_API_KEY": wandb_key,
        "WANDB_PROJECT": wandb_project,
        "SERVERCHAN_SENDKEY": serverchan,
        "VAST_INSTANCE_ID": vast_instance,
        "VAST_API_KEY": vast_key,
        "AUTO_STOP_INSTANCE": auto_stop,
        "AUTO_STOP_ON_FAILURE": "0",
        "AUTO_STOP_ON_UPLOAD_FAILURE": "0",
        "GPU_IDS": gpu_ids,
        "GPU_COUNT": gpu_count,
        "RUN_ID": run_id,
        "PREPARE_ENV": "1",
        "UV_DEFAULT_INDEX": "https://pypi.org/simple",
        "PIP_INDEX_URL": "https://pypi.org/simple",
    }
    if algorithm == "openpi":
        values.update({"PIPELINE_MODE": "train", "CONFIRM_FULL_TRAIN": "YES"})

    env_path = Path(args.output_env)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(shell_exports(values), encoding="utf-8")
    env_path.chmod(0o600)

    manifest = {
        "schema_version": 1,
        "task": task,
        "algorithm": algorithm,
        "datasets": mix_payload,
        "output_model_repo": output_repo,
        "resume": {
            "mode": resume_mode,
            "repo": resume_repo or None,
            "run_id": resume_run_id or None,
        },
        "gpu_ids": gpu_ids,
        "auto_stop": auto_stop == "1",
        "credentials": {
            "hf": "configured",
            "wandb": "configured" if wandb_key else "disabled",
            "serverchan": "configured" if serverchan else "disabled",
            "vast": "configured" if vast_key else "disabled",
        },
    }
    manifest_path = Path(args.output_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"私有环境文件：{env_path}（0600）")
    print(f"无凭据任务清单：{manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
