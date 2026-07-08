#!/usr/bin/env python3
"""Lightweight Hydra launcher for LingBot-VLA v2 training."""

from __future__ import annotations

from pathlib import Path
import os
import socket

import hydra
from omegaconf import DictConfig

from kuavo_train.wrapper.policy.lingbot import CustomLingbotConfigWrapper, CustomLingbotPolicyWrapper


def _available_port(preferred: int) -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("0.0.0.0", preferred))
            return preferred
    except PermissionError:
        return preferred
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("0.0.0.0", 0))
            return int(sock.getsockname()[1])


@hydra.main(config_path="../configs/policy", config_name="lingbot_v2_config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.policy_name != "lingbot_v2":
        raise ValueError(f"Expected policy_name=lingbot_v2, got {cfg.policy_name!r}")

    repo_root = Path(__file__).resolve().parents[1]
    policy_cfg = cfg.policy
    env = {str(k): str(v) for k, v in dict(policy_cfg.get("env", {})).items()}
    # JSON-valued LoRA settings are awkward to pass through Hydra's override grammar.
    # Prefer an explicit shell environment variable when provided.
    for key in ("KUAVO_LINGBOT_V2_LORA", "CUDA_VISIBLE_DEVICES", "LINGBOT_V2_ROOT"):
        if os.getenv(key):
            env[key] = os.environ[key]
    cuda_devices = [item for item in env.get("CUDA_VISIBLE_DEVICES", "0").split(",") if item.strip()]
    world_size = max(1, len(cuda_devices))
    wrapper_cfg = CustomLingbotConfigWrapper(
        lingbot_root=str(policy_cfg.lingbot_root),
        lerobot_root=str(policy_cfg.get("lerobot_root", "")),
        train_entry=str(policy_cfg.train_entry),
        config_path=str(policy_cfg.config_path),
        nnodes=1,
        node_rank=0,
        master_addr="0.0.0.0",
        master_port=_available_port(int(os.getenv("MASTER_PORT", "62500"))),
        dry_run=bool(policy_cfg.get("dry_run", False)),
        env=env,
    )

    output_dir = Path(cfg.training.output_directory)
    resume_timestamp = str(cfg.training.get("resume_timestamp", "") or "").strip()
    if bool(cfg.training.get("resume", False)) and resume_timestamp:
        run_name = resume_timestamp if resume_timestamp.startswith("run_") else f"run_{resume_timestamp}"
    else:
        run_name = f"run_{cfg.timestamp}"
    output_dir = output_dir / run_name

    micro_batch = int(cfg.training.batch_size)
    accumulation = int(cfg.training.accumulation_steps)
    global_batch = micro_batch * accumulation * world_size
    extra_args = [
        "--data.train_path",
        str(Path(cfg.root).expanduser()),
        "--train.output_dir",
        str(output_dir),
        "--train.micro_batch_size",
        str(micro_batch),
        "--train.global_batch_size",
        str(global_batch),
        "--train.max_steps",
        str(int(cfg.training.max_training_step)),
        "--train.num_train_epochs",
        str(int(cfg.training.max_epoch)),
        "--model.model_path",
        str(policy_cfg.model_path),
        "--model.tokenizer_path",
        str(policy_cfg.tokenizer_path),
    ]
    extra_args.extend(str(item) for item in policy_cfg.get("extra_args", []))

    code = CustomLingbotPolicyWrapper(wrapper_cfg).launch(repo_root=repo_root, extra_args=extra_args)
    if code != 0:
        raise RuntimeError(f"LingBot-VLA v2 training failed with exit code {code}")


if __name__ == "__main__":
    main()
