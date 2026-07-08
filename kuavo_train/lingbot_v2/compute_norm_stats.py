#!/usr/bin/env python3
"""Run LingBot-VLA v2 norm statistics without importing CUDA MoE kernels."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import os
import runpy
import sys
import types

from lingbotvla.utils.arguments import DataArguments, TrainingArguments


@dataclass
class MyTrainingArguments(TrainingArguments):
    freeze_vision_encoder: bool = False
    tokenizer_max_length: int = 72
    action_dim: int = 55
    max_action_dim: int = 55
    max_state_dim: int = 55
    chunk_size: int = 50
    vlm_causal: bool = True
    loss_type: str = "fm"
    align_params: Optional[dict[str, Any]] = field(default_factory=dict)
    attention_implementation: str = "flex_cached"
    precompute_grid_thw: bool = True
    use_moe: bool = True
    token_moe_layers: Optional[list[int]] = None
    token_num_experts: int = 32
    token_top_k: int = 4
    token_moe_intermediate_size: int = 512
    token_shared_intermediate_size: int = 704
    bias_update_speed: float = 0.0
    sequence_wise_mode: str = "per_sequence"
    sequence_wise_loss_coeff: float = 0.0
    router_z_loss_coeff: float = 0.0
    router_activation: str = "sigmoid"
    routed_scaling_factor: float = 4.0
    use_shared_expert_gate: bool = False
    use_moe_expert_lr: bool = False
    vlm_fsdp: bool = False


@dataclass
class MyDataArguments(DataArguments):
    source_name: str | None = None
    robot_config_root: str | None = None
    joints: Optional[list[str]] = None
    cameras: Optional[list[str]] = None
    norm_type: Optional[list[str]] = None
    img_size: int = 256
    norm_stats_file: str | None = None
    prompt_type: str = "global"
    use_future_image: bool = False


def main() -> None:
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    root = Path(os.environ.get("LINGBOT_V2_ROOT", "/home/larry/lingbot-vla-v2")).expanduser()
    script = root / "scripts/compute_norm_stats.py"
    if not script.is_file():
        raise FileNotFoundError(f"LingBot-VLA v2 norm script not found: {script}")

    fake_module = types.ModuleType("tasks.vla.train_lingbotvla")
    fake_module.MyTrainingArguments = MyTrainingArguments
    fake_module.MyDataArguments = MyDataArguments
    sys.modules["tasks.vla.train_lingbotvla"] = fake_module
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
