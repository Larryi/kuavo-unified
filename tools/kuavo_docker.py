#!/usr/bin/env python3
"""Interactive Docker build/test/release router for Kuavo policy backends."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
SESSION_ROOT = REPO_ROOT / "outputs" / "docker_sessions"
WEIGHT_SUFFIXES = {".bin", ".pt", ".pth", ".safetensors", ".ckpt"}
SENSITIVE_NAMES = {
    ".env", "credentials.json", "secrets.json", "id_rsa", "id_ed25519",
}


@dataclass(frozen=True)
class BackendSpec:
    key: str
    policy_type: str
    image: str
    build_script: str
    archive_env: str | None
    config: str
    qwen_required: bool = False
    norm_required: bool = False
    tokenizer_required: bool = False
    ros_ready: bool = True
    delivery_paused: bool = False


@dataclass(frozen=True)
class TaskSpec:
    key: str
    backend: str
    config: str
    description: str


BACKENDS = {
    "act": BackendSpec(
        "act", "act", "kuavo-classic:latest", "docker/build_classic.sh",
        "CLASSIC_ENV_ARCHIVE", "configs/deploy/kuavo_env.act.yaml",
    ),
    "dp": BackendSpec(
        "dp", "diffusion", "kuavo-classic:latest", "docker/build_classic.sh",
        "CLASSIC_ENV_ARCHIVE", "configs/deploy/kuavo_env.dp.yaml",
    ),
    "lingbot-v1": BackendSpec(
        "lingbot-v1", "lingbot", "kdc_real_task1_lingbot:latest",
        "docker/build_lingbot.sh", "LINGBOT_ENV_ARCHIVE",
        "configs/deploy/kuavo_env.lingbot.yaml", True, True,
    ),
    "lingbot-v2": BackendSpec(
        "lingbot-v2", "lingbot_v2", "kuavo-lingbot-v2:latest",
        "docker/build_lingbot_v2.sh", "LINGBOT_V2_ENV_ARCHIVE",
        "configs/deploy/kuavo_env.lingbot_v2.yaml",
        qwen_required=True, norm_required=True, delivery_paused=True,
    ),
    "openpi": BackendSpec(
        "openpi", "client", "kuavo-openpi:latest", "docker/build_openpi.sh",
        None, "configs/deploy/kuavo_env.openpi_client.yaml",
        tokenizer_required=True,
    ),
}

TASKS = {
    "task1-openpi": TaskSpec(
        "task1-openpi", "openpi", "configs/deploy/kuavo_env.openpi_client.yaml",
        "Task1 right arm + Leju claw, OpenPI Pi0.5",
    ),
    "task1-lingbot-v1": TaskSpec(
        "task1-lingbot-v1", "lingbot-v1", "configs/deploy/kuavo_env.lingbot.yaml",
        "Task1 right arm + Leju claw, legacy absolute-action LingBot-v1",
    ),
    "task2-dp": TaskSpec(
        "task2-dp", "dp", "configs/deploy/kuavo_env.dp.task2.yaml",
        "Task2 bimanual + dual Leju claws, Diffusion Policy",
    ),
    "task3-act": TaskSpec(
        "task3-act", "act", "configs/deploy/kuavo_env.act.task3.yaml",
        "Task3 right arm + Qiangnao end effector, ACT",
    ),
}

DEFAULT_TASK_BY_BACKEND = {
    task.backend: task.key for task in TASKS.values()
}


class UsageError(RuntimeError):
    pass


def quote_command(command: list[str]) -> str:
    return shlex.join(command)


def run(command: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    print(f"+ {quote_command(command)}")
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def run_bounded_build(
    command: list[str],
    *,
    dry_run: bool,
    log_path: Path,
) -> None:
    print(f"+ {quote_command(command)}")
    if dry_run:
        return
    print(f"Docker 完整构建日志: {log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        print("Docker 构建失败，末尾 80 行：", file=sys.stderr)
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        print("\n".join(lines[-80:]), file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, command)
    print("Docker release 镜像构建完成。")


def prompt_choice(title: str, choices: list[str]) -> str:
    if not sys.stdin.isatty():
        raise UsageError(f"{title} is required in non-interactive mode")
    print(title)
    for index, choice in enumerate(choices, 1):
        print(f"  {index}) {choice}")
    while True:
        answer = input("> ").strip()
        if answer in choices:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        print("请输入编号或完整名称。")


def prompt_path(title: str, current: str | None = None) -> str:
    if current:
        return current
    if not sys.stdin.isatty():
        raise UsageError(f"{title} is required in non-interactive mode")
    return input(f"{title}: ").strip()


def confirm_action(message: str, *, yes: bool, dry_run: bool) -> None:
    if dry_run or yes:
        return
    if not sys.stdin.isatty():
        raise UsageError(f"非交互执行必须显式传入 --yes：{message}")
    answer = input(f"{message} [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        raise UsageError("操作者已取消")


def resolve_existing(path: str, *, kind: str, directory: bool = False) -> Path:
    candidate = Path(path).expanduser().resolve()
    valid = candidate.is_dir() if directory else candidate.exists()
    if not valid:
        raise UsageError(f"{kind}不存在或类型不正确: {candidate}")
    return candidate


def nonempty_directory(path: Path, label: str) -> None:
    if not any(path.iterdir()):
        raise UsageError(f"{label}目录为空: {path}")


def validate_qwen_bundle(path: Path) -> list[str]:
    errors: list[str] = []
    if not (path / "config.json").is_file():
        errors.append("缺少 config.json")
    if not any((path / name).is_file() for name in (
        "tokenizer.json", "tokenizer.model", "vocab.json",
    )):
        errors.append("缺少 tokenizer.json/tokenizer.model/vocab.json")
    if not any((path / name).is_file() for name in (
        "preprocessor_config.json", "processor_config.json",
        "image_processor_config.json",
    )):
        errors.append("缺少图像 processor 配置")
    if errors:
        raise UsageError(f"Qwen processor bundle 无效 ({path}): " + "；".join(errors))

    warnings: list[str] = []
    weights = [
        item for item in path.rglob("*")
        if item.is_file() and (
            item.suffix.lower() in WEIGHT_SUFFIXES
            or item.name.startswith("model-") and item.name.endswith(".safetensors")
        )
    ]
    if weights:
        size = sum(item.stat().st_size for item in weights)
        warnings.append(
            f"Qwen 目录含 {len(weights)} 个权重文件（约 {size / 2**30:.1f} GiB）。"
            "完整 LingBot HF checkpoint 推理通常只需 config/tokenizer/processor；"
            "源目录不会被修改，release 镜像会自动排除这些基模权重。"
        )
    return warnings


def validate_checkpoint(spec: BackendSpec, path: Path) -> list[str]:
    nonempty_directory(path, "checkpoint")
    warnings: list[str] = []
    files = [item for item in path.rglob("*") if item.is_file()]
    if spec.key in {"lingbot-v1", "lingbot-v2"}:
        if not any(item.suffix == ".safetensors" for item in files):
            raise UsageError(
                "LingBot 部署 checkpoint 未发现 safetensors；"
                "LoRA/DCP 必须先合并导出为完整 HF checkpoint。"
            )
        if spec.key == "lingbot-v1" and not (path / "lingbotvla_cli.yaml").is_file():
            raise UsageError("LingBot-v1 hf_ckpt 缺少 lingbotvla_cli.yaml")
        if spec.key == "lingbot-v2":
            ancestors = (path, *path.parents[:4])
            if not any((parent / "lingbotvla_cli.yaml").is_file() for parent in ancestors):
                raise UsageError(
                    "LingBot-v2 checkpoint 或其上级目录缺少 lingbotvla_cli.yaml；"
                    "打包前需要保留训练配置。"
                )
    elif spec.key in {"act", "dp"}:
        processor_names = {"policy_preprocessor.json", "policy_postprocessor.json"}
        visible = {item.name for item in files}
        missing = processor_names - visible
        if missing:
            warnings.append(
                "checkpoint bundle 中未发现 "
                + ", ".join(sorted(missing))
                + "；请把包含 processor 文件的训练目录作为 --checkpoint，"
                  "并用 --checkpoint-subpath 指向实际权重目录。"
            )
        for item in files:
            if item.parent != path or item.name not in processor_names:
                continue
            try:
                processor_config = json.loads(item.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if any(
                step.get("class", "").startswith("__main__.")
                for step in processor_config.get("steps", [])
            ):
                raise UsageError(
                    f"{item} 含不可移植的训练期 processor（__main__.*）。"
                    "请把含干净 processor 的训练 run 根目录作为 --checkpoint，"
                    "并用 --checkpoint-subpath 指向 epoch 权重目录。"
                )
    elif spec.key == "openpi":
        metadata = path / "params" / "_METADATA"
        if not metadata.is_file():
            raise UsageError(
                "OpenPI checkpoint 必须是包含 params/_METADATA 的训练 step 目录；"
                "不要把 params 目录本身挂载为 /models/checkpoint。"
            )
    return warnings


def normalize_checkpoint_bundle(spec: BackendSpec, path: Path) -> Path:
    """Normalize user-facing checkpoint paths to the runtime bundle root."""
    if spec.key != "openpi" or not (path / "_METADATA").is_file():
        return path
    if path.name != "params":
        raise UsageError(
            "检测到 OpenPI Orbax metadata，但目录名不是 params；"
            "请传入包含 params/_METADATA 的训练 step 目录。"
        )
    step_dir = path.parent
    if not (step_dir / "params" / "_METADATA").is_file():
        raise UsageError(f"无法确定 OpenPI checkpoint step 目录: {path}")
    print(f"OpenPI 检测到 params 目录，自动改用 checkpoint step: {step_dir}")
    return step_dir


def reject_sensitive_files(paths: list[Path | None]) -> None:
    matches: list[Path] = []
    for root in paths:
        if root is None:
            continue
        candidates = [root] if root.is_file() else root.rglob("*")
        for item in candidates:
            if not item.is_file():
                continue
            lowered_parts = {part.lower() for part in item.parts}
            if (
                item.name.lower() in SENSITIVE_NAMES
                or item.suffix.lower() in {".pem", ".key"}
                or lowered_parts.intersection({".ssh", "secrets"})
            ):
                matches.append(item)
    if matches:
        shown = ", ".join(str(path) for path in matches[:5])
        raise UsageError(
            f"release 资产中发现疑似凭据/私钥文件，已拒绝构建: {shown}"
        )


def ensure_ros_ready(spec: BackendSpec, operation: str) -> None:
    if spec.delivery_paused and operation in {"shell", "release"}:
        raise UsageError(
            "LingBot-v2 交付适配已按操作者决定暂缓；"
            f"已阻止 {operation}，恢复前不得生成或标记最终交付镜像。"
        )
    if not spec.ros_ready and operation in {"shell", "release"}:
        raise UsageError(
            "LingBot-v2 当前仍是纯 Worker 镜像，尚未包含 ROS Noetic/KuavoBaseEnv；"
            f"已阻止 {operation}，避免误认为可最终交付。"
        )


def resolve_task(args: argparse.Namespace, spec: BackendSpec) -> TaskSpec:
    task_key = args.task or DEFAULT_TASK_BY_BACKEND.get(spec.key)
    if not task_key:
        raise UsageError(f"--backend {spec.key} 没有默认任务，必须显式传入 --task")
    task = TASKS[task_key]
    if task.backend != spec.key:
        raise UsageError(
            f"任务 {task.key} 绑定 backend={task.backend}，不能用于 {spec.key}"
        )
    return task


def deploy_config_path(args: argparse.Namespace, spec: BackendSpec) -> Path:
    task = resolve_task(args, spec)
    return resolve_existing(
        args.config or str(REPO_ROOT / task.config),
        kind="deploy config",
    )


def checkpoint_container_path(subpath: str) -> str:
    cleaned = subpath.strip().strip("/")
    if ".." in Path(cleaned).parts:
        raise UsageError("--checkpoint-subpath 不能包含 '..'")
    return "/models/checkpoint" + (f"/{cleaned}" if cleaned else "")


def render_config(
    source: Path,
    destination: Path,
    *,
    pretrained_path: str,
) -> None:
    text = source.read_text(encoding="utf-8")
    if "__PRETRAINED_PATH__" in text:
        text = text.replace("__PRETRAINED_PATH__", pretrained_path)
    elif "pretrained_path:" in text and pretrained_path not in text:
        raise UsageError(
            f"自定义配置 {source} 的 pretrained_path 未指向 {pretrained_path}，"
            "请使用 __PRETRAINED_PATH__ 占位符或容器内路径。"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def collect_assets(
    args: argparse.Namespace,
    spec: BackendSpec,
) -> tuple[Path, Path | None, Path | None, Path | None, Path | None]:
    checkpoint = resolve_existing(
        prompt_path("Checkpoint bundle 路径", args.checkpoint),
        kind="checkpoint", directory=True,
    )
    checkpoint = normalize_checkpoint_bundle(spec, checkpoint)
    warnings = validate_checkpoint(spec, checkpoint)

    qwen: Path | None = None
    if spec.qwen_required:
        qwen = resolve_existing(
            prompt_path("Qwen processor bundle 路径", args.qwen),
            kind="Qwen processor", directory=True,
        )
        warnings.extend(validate_qwen_bundle(qwen))

    norm: Path | None = None
    if spec.norm_required:
        norm = resolve_existing(
            prompt_path("norm_stats.json 路径", args.norm_stats),
            kind="norm stats",
        )
        if not norm.is_file():
            raise UsageError(f"norm stats 必须是文件: {norm}")

    runtime_assets: Path | None = None
    if args.runtime_assets:
        runtime_assets = resolve_existing(
            args.runtime_assets, kind="runtime assets", directory=True,
        )

    tokenizer: Path | None = None
    if spec.tokenizer_required:
        tokenizer = resolve_existing(
            prompt_path("PaliGemma tokenizer.model 路径", args.tokenizer),
            kind="OpenPI tokenizer",
        )
        if not tokenizer.is_file():
            raise UsageError(f"OpenPI tokenizer 必须是文件: {tokenizer}")

    for warning in warnings:
        print(f"警告: {warning}", file=sys.stderr)
    return checkpoint, qwen, norm, runtime_assets, tokenizer


def build_command(args: argparse.Namespace, spec: BackendSpec) -> None:
    environment = os.environ.copy()
    if spec.archive_env:
        archive_value = prompt_path(
            f"{spec.archive_env} (myenv.tar.gz)",
            args.env_archive or environment.get(spec.archive_env),
        )
        archive = resolve_existing(archive_value, kind=spec.archive_env)
        if archive.name != "myenv.tar.gz":
            raise UsageError(f"{spec.archive_env} 文件名必须是 myenv.tar.gz")
        environment[spec.archive_env] = str(archive)
    run([str(REPO_ROOT / spec.build_script)], dry_run=args.dry_run, env=environment)


def prepare_session_config(
    args: argparse.Namespace,
    spec: BackendSpec,
    pretrained_path: str,
) -> Path:
    source = deploy_config_path(args, spec)
    if args.dry_run:
        return Path("/planned/kuavo_env.yaml")
    session = SESSION_ROOT / f"{spec.key}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    destination = session / "kuavo_env.yaml"
    render_config(source, destination, pretrained_path=pretrained_path)
    return destination


def shell_command(args: argparse.Namespace, spec: BackendSpec) -> None:
    ensure_ros_ready(spec, "shell")
    checkpoint, qwen, norm, runtime_assets, tokenizer = collect_assets(args, spec)
    policy_path = checkpoint_container_path(args.checkpoint_subpath)
    config = prepare_session_config(args, spec, policy_path)

    command = [
        "docker", "run", "--rm", "-it", "--init",
        "--name", args.container_name or f"kuavo-{spec.key}-test",
        "--network", "host", "--gpus", args.gpus,
        "-v", f"{checkpoint}:/models/checkpoint:ro",
        "-v", f"{config}:/run/kuavo/kuavo_env.yaml:ro",
        "-v",
        f"{config}:/root/kuavo_data_challenge/configs/deploy/kuavo_env.yaml:ro",
    ]
    if qwen:
        command += ["-v", f"{qwen}:/assets/qwen:ro"]
    if norm:
        command += ["-v", f"{norm}:/assets/norm_stats/norm_stats.json:ro"]
    if runtime_assets:
        command += ["-v", f"{runtime_assets}:/assets/runtime:ro"]
    if tokenizer:
        command += [
            "-v",
            f"{tokenizer}:/mnt/pqssd/pretrained/google/paligemma-3b-pt-224/tokenizer.model:ro",
        ]
    command += ["--entrypoint", "bash", args.image or spec.image]

    print(f"待人工检查 YAML: {config}")
    if not args.dry_run:
        print("----- kuavo_env.yaml -----")
        print(config.read_text(encoding="utf-8").rstrip())
        print("--------------------------")
    print("容器内人工推理命令:")
    if spec.key == "openpi":
        print(
            f"  OPENPI_POLICY_CONFIG={args.openpi_config} "
            f"OPENPI_POLICY_DIR={policy_path} "
            "docker/start_openpi_ros.sh bash"
        )
    elif spec.key == "lingbot-v2":
        print(
            f"  LINGBOT_V2_POLICY_DIR={policy_path} "
            "LINGBOT_V2_QWEN_DIR=/assets/qwen "
            "LINGBOT_V2_NORM_STATS=/assets/norm_stats/norm_stats.json "
            "docker/start_lingbot_v2_ros.sh bash"
        )
    print(
        "  python kuavo_deploy/src/scripts/script_auto_test.py "
        "--task auto_test --config configs/deploy/kuavo_env.yaml"
    )
    confirm_action(
        "确认已检查 YAML，启动测试容器？",
        yes=args.yes,
        dry_run=args.dry_run,
    )
    run(command, dry_run=args.dry_run)


def context_or_empty(stack: contextlib.ExitStack, path: Path | None, name: str) -> Path:
    if path is not None:
        return path
    directory = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix=f"kuavo-{name}-")))
    (directory / ".keep").write_text("", encoding="utf-8")
    return directory


def stage_qwen_processor(source: Path, destination: Path) -> None:
    """Copy Qwen runtime metadata while intentionally excluding base weights."""
    destination.mkdir()
    for item in source.iterdir():
        if not item.is_file():
            continue
        is_weight = (
            item.suffix.lower() in WEIGHT_SUFFIXES
            or item.name.startswith("model-") and item.name.endswith(".safetensors")
        )
        if not is_weight:
            shutil.copy2(item, destination / item.name)
    validate_qwen_bundle(destination)


def stage_classic_checkpoint(
    source: Path,
    destination: Path,
    checkpoint_subpath: str,
) -> Path:
    """Keep only deploy processors and the selected ACT/DP checkpoint."""
    selected = checkpoint_container_path(checkpoint_subpath)
    cleaned = checkpoint_subpath.strip().strip("/")
    if not cleaned:
        shutil.copytree(source, destination)
        return destination

    selected_source = source / cleaned
    if not selected_source.is_dir():
        raise UsageError(
            f"--checkpoint-subpath 在 checkpoint bundle 中不存在: {selected_source}"
        )
    destination.mkdir()
    selected_destination = destination / cleaned
    selected_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(selected_source, selected_destination)
    for item in source.iterdir():
        if not item.is_file():
            continue
        if (
            item.name == "config.json"
            or item.name.startswith("policy_preprocessor")
            or item.name.startswith("policy_postprocessor")
        ):
            shutil.copy2(item, destination / item.name)
    # The return value documents the unchanged in-container policy path.
    assert selected == checkpoint_container_path(checkpoint_subpath)
    return destination


def release_command(args: argparse.Namespace, spec: BackendSpec) -> None:
    ensure_ros_ready(spec, "release")
    task = resolve_task(args, spec)
    checkpoint, qwen, norm, runtime_assets, tokenizer = collect_assets(args, spec)
    reject_sensitive_files([checkpoint, qwen, norm, runtime_assets, tokenizer])
    policy_path = checkpoint_container_path(args.checkpoint_subpath)
    source_config = deploy_config_path(args, spec)
    tag = args.tag or f"kuavo-{task.key}-release:latest"

    with contextlib.ExitStack() as stack:
        staging = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="kuavo-release-")))
        config_dir = staging / "config"
        config_dir.mkdir()
        render_config(
            source_config, config_dir / "kuavo_env.yaml",
            pretrained_path=policy_path,
        )
        norm_dir: Path | None = None
        if norm:
            norm_dir = staging / "norm"
            norm_dir.mkdir()
            shutil.copy2(norm, norm_dir / "norm_stats.json")
        tokenizer_dir: Path | None = None
        if tokenizer:
            tokenizer_dir = staging / "tokenizer"
            tokenizer_dir.mkdir()
            shutil.copy2(tokenizer, tokenizer_dir / "tokenizer.model")
        qwen_dir: Path | None = None
        if qwen:
            qwen_dir = staging / "qwen"
            stage_qwen_processor(qwen, qwen_dir)
        checkpoint_dir = checkpoint
        if spec.key in {"act", "dp"}:
            checkpoint_dir = staging / "checkpoint"
            stage_classic_checkpoint(
                checkpoint, checkpoint_dir, args.checkpoint_subpath
            )
        manifest_dir = staging / "manifest"
        manifest_dir.mkdir()
        (manifest_dir / "release_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "backend": spec.key,
                    "policy_type": spec.policy_type,
                    "task": task.key,
                    "task_description": task.description,
                    "checkpoint_path": policy_path,
                    "base_image": args.image or spec.image,
                    "manual_start_required": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        contexts = {
            "checkpoint": checkpoint_dir,
            "qwen": context_or_empty(stack, qwen_dir, "qwen"),
            "norm_stats": context_or_empty(stack, norm_dir, "norm"),
            "runtime_assets": context_or_empty(stack, runtime_assets, "runtime"),
            "tokenizer": context_or_empty(stack, tokenizer_dir, "tokenizer"),
            "deploy_config": config_dir,
            "release_manifest": manifest_dir,
        }
        command = [
            "docker", "buildx", "build", "--load", "--progress=plain",
            "--build-arg", f"BASE_IMAGE={args.image or spec.image}",
            "--build-arg", f"KUAVO_BACKEND={spec.key}",
            "--build-arg", f"KUAVO_TASK={task.key}",
            "--build-arg", f"SOURCE_IMAGE={args.image or spec.image}",
        ]
        for name, path in contexts.items():
            command += ["--build-context", f"{name}={path}"]
        command += ["-f", str(REPO_ROOT / "Dockerfile.release"), "-t", tag, str(REPO_ROOT)]
        print(f"最终镜像标签: {tag}")
        print(f"任务路由: {task.key} -> {spec.key} ({task.description})")
        print("不会执行 docker save；需要 TAR 时单独运行 export。")
        confirm_action(
            "确认将上述 checkpoint/运行资产固化进 release 镜像？",
            yes=args.yes,
            dry_run=args.dry_run,
        )
        default_log_name = tag.replace("/", "_").replace(":", "_") + ".build.log"
        run_bounded_build(
            command,
            dry_run=args.dry_run,
            log_path=Path(args.build_log or f"/tmp/{default_log_name}").expanduser(),
        )


def export_command(args: argparse.Namespace) -> None:
    image = args.image
    if not image:
        image = prompt_path("要导出的镜像标签")
    output_value = prompt_path("TAR 输出路径", args.output)
    output = Path(output_value).expanduser().resolve()
    if output.exists():
        raise UsageError(f"拒绝覆盖已有文件: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.yes:
        if not sys.stdin.isatty():
            raise UsageError("非交互 export 必须显式传入 --yes")
        answer = input(f"将创建可能很大的 TAR：{output}。继续？[y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("已取消。")
            return
    run(["docker", "save", "-o", str(output), image], dry_run=args.dry_run)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kuavo Docker 交互构建、测试挂载、release 和显式 TAR 导出",
    )
    parser.add_argument("command", nargs="?", choices=["build", "shell", "release", "export"])
    parser.add_argument("--backend", choices=sorted(BACKENDS))
    parser.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-subpath", default="")
    parser.add_argument("--qwen")
    parser.add_argument("--norm-stats")
    parser.add_argument("--runtime-assets")
    parser.add_argument("--tokenizer")
    parser.add_argument("--config")
    parser.add_argument("--env-archive")
    parser.add_argument("--image")
    parser.add_argument("--tag")
    parser.add_argument("--output")
    parser.add_argument("--build-log")
    parser.add_argument("--container-name")
    parser.add_argument("--gpus", default="all")
    parser.add_argument("--openpi-config", default="pi05_kuavo")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    try:
        command = args.command or prompt_choice(
            "选择操作", ["build", "shell", "release", "export"],
        )
        if command == "export":
            export_command(args)
            return 0
        backend = args.backend or prompt_choice("选择模型类型", list(BACKENDS))
        spec = BACKENDS[backend]
        if command == "build":
            build_command(args, spec)
        elif command == "shell":
            shell_command(args, spec)
        elif command == "release":
            release_command(args, spec)
        return 0
    except UsageError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        return exc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
