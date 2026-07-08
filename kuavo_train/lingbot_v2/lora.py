"""LoRA injection and checkpoint merging for LingBot-VLA v2."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
from peft import LoraConfig, inject_adapter_in_model


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "qkv",
    "linear_fc1",
    "linear_fc2",
)

DEFAULT_FULL_TRAIN_PATTERNS = (
    "state_proj",
    "action_in_proj",
    "action_out_proj",
    "action_time_mlp",
    "depth_align",
    "video_align",
    "shared_task_proj",
    ".mlp.gate.",
    "e_score_correction_bias",
)


@dataclass(frozen=True)
class LingbotV2LoraSettings:
    enabled: bool = True
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    full_train_patterns: tuple[str, ...] = DEFAULT_FULL_TRAIN_PATTERNS

    @classmethod
    def from_env(cls) -> "LingbotV2LoraSettings":
        raw = os.getenv("KUAVO_LINGBOT_V2_LORA", "").strip()
        if not raw:
            return cls()
        data = json.loads(raw)
        return cls(
            enabled=bool(data.get("enabled", True)),
            rank=int(data.get("rank", 8)),
            alpha=int(data.get("alpha", 16)),
            dropout=float(data.get("dropout", 0.0)),
            target_modules=tuple(data.get("target_modules", DEFAULT_TARGET_MODULES)),
            full_train_patterns=tuple(data.get("full_train_patterns", DEFAULT_FULL_TRAIN_PATTERNS)),
        )


def apply_lora(model: torch.nn.Module, settings: LingbotV2LoraSettings) -> torch.nn.Module:
    if not settings.enabled:
        return model

    model.requires_grad_(False)
    config = LoraConfig(
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        bias="none",
        target_modules=list(settings.target_modules),
    )
    model = inject_adapter_in_model(config, model)

    for name, parameter in model.named_parameters():
        if "lora_" in name or any(pattern in name for pattern in settings.full_train_patterns):
            parameter.requires_grad_(True)
        if "lora_" in name:
            parameter.data = parameter.data.float()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0:
        raise RuntimeError("LingBot-VLA v2 LoRA injection produced no trainable parameters")
    print(f"[Kuavo LoRA] trainable={trainable:,}/{total:,} ({100 * trainable / total:.3f}%)")
    return model


def merge_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    alpha: float,
    rank: int,
) -> dict[str, torch.Tensor]:
    """Merge PEFT Linear adapters and restore the original model key names."""

    merged = dict(state_dict)
    suffix = ".lora_A.default.weight"
    for a_key in [key for key in merged if key.endswith(suffix)]:
        prefix = a_key[: -len(suffix)]
        b_key = f"{prefix}.lora_B.default.weight"
        base_key = f"{prefix}.base_layer.weight"
        if b_key not in merged or base_key not in merged:
            raise KeyError(f"Incomplete LoRA state for {prefix}")
        a = merged[a_key]
        b = merged[b_key]
        base = merged[base_key]
        if a.ndim != 2 or b.ndim != 2 or base.ndim != 2:
            raise ValueError(f"Only Linear LoRA merge is supported for {prefix}")
        delta = torch.matmul(b.float(), a.float()).mul_(float(alpha) / float(rank))
        if delta.shape != base.shape:
            raise ValueError(
                f"LoRA shape mismatch for {prefix}: base={tuple(base.shape)}, delta={tuple(delta.shape)}"
            )
        merged[f"{prefix}.weight"] = (base.float() + delta).to(base.dtype)

    for key in list(merged):
        if ".lora_" in key:
            del merged[key]
        elif ".base_layer." in key:
            original_key = key.replace(".base_layer.", ".")
            merged.setdefault(original_key, merged[key])
            del merged[key]
    return merged
