#!/usr/bin/env python3
"""Export a LingBot distributed LoRA checkpoint as merged HF weights."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
import yaml


LORA_A_SUFFIX = ".lora_A.default.weight"
LORA_B_SUFFIX = ".lora_B.default.weight"
BASE_LAYER_TOKEN = ".base_layer."


def find_run_dir(checkpoint: Path) -> Path:
    for parent in (checkpoint, *checkpoint.parents):
        if (parent / "lingbotvla_cli.yaml").is_file():
            return parent
    raise FileNotFoundError(
        f"Could not find lingbotvla_cli.yaml above checkpoint: {checkpoint}"
    )


def load_lora_config(run_dir: Path) -> tuple[float, int, str]:
    with (run_dir / "lingbotvla_cli.yaml").open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    train = config.get("train", {})
    if not train.get("use_lora", False):
        raise ValueError("Checkpoint config does not declare train.use_lora=true")
    return float(train["lora_alpha"]), int(train["lora_rank"]), str(
        train.get("ckpt_manager", "dcp")
    )


def _lora_delta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError(f"Invalid LoRA shapes: A={tuple(a.shape)}, B={tuple(b.shape)}")
    if b.shape[1] != a.shape[0] or any(size != 1 for size in b.shape[2:]):
        raise ValueError(f"Unsupported LoRA shapes: A={tuple(a.shape)}, B={tuple(b.shape)}")
    # Handles Linear A=[r,in], B=[out,r] and ConvNd A=[r,in,...],
    # B=[out,r,1,...] without flattening the convolution kernel.
    return torch.einsum("or,ri...->oi...", b.reshape(b.shape[0], b.shape[1]), a)


def merge_lora_state_dict(
    state_dict: dict[str, torch.Tensor], lora_alpha: float, lora_rank: int
) -> tuple[dict[str, torch.Tensor], int]:
    merged: dict[str, torch.Tensor] = {}
    lora_prefixes = {key[: -len(LORA_A_SUFFIX)] for key in state_dict if key.endswith(LORA_A_SUFFIX)}
    scale = lora_alpha / lora_rank

    for key, tensor in state_dict.items():
        if key.endswith((LORA_A_SUFFIX, LORA_B_SUFFIX)):
            continue
        output_key = key.replace(BASE_LAYER_TOKEN, ".")
        merged[output_key] = tensor

    for prefix in sorted(lora_prefixes):
        a_key = prefix + LORA_A_SUFFIX
        b_key = prefix + LORA_B_SUFFIX
        base_key = prefix + ".base_layer.weight"
        missing = [key for key in (a_key, b_key, base_key) if key not in state_dict]
        if missing:
            raise KeyError(f"Incomplete LoRA module {prefix}: missing {missing}")

        base = state_dict[base_key]
        delta = _lora_delta(state_dict[a_key].float(), state_dict[b_key].float())
        if tuple(delta.shape) != tuple(base.shape):
            raise ValueError(
                f"LoRA/base shape mismatch for {prefix}: base={tuple(base.shape)}, "
                f"delta={tuple(delta.shape)}"
            )
        merged[prefix + ".weight"] = (base.float() + delta.mul_(scale)).to(base.dtype)

    leftover = [key for key in merged if ".lora_" in key or BASE_LAYER_TOKEN in key]
    if leftover:
        raise RuntimeError(f"Unmerged LoRA keys remain, first key: {leftover[0]}")
    return merged, len(lora_prefixes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="global_step_* distributed checkpoint")
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Default: CHECKPOINT/hf_ckpt")
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory")
    parser.add_argument("--save-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    lingbot_root = args.lingbot_root.resolve()
    if not (checkpoint / "model").is_dir():
        raise FileNotFoundError(f"DCP model directory not found: {checkpoint / 'model'}")
    if not lingbot_root.is_dir():
        raise FileNotFoundError(f"LingBot repository not found: {lingbot_root}")

    output = (args.output or checkpoint / "hf_ckpt").resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {output}. Pass --force to replace it.")
        shutil.rmtree(output)

    sys.path.insert(0, str(lingbot_root))
    from lingbotvla.checkpoint.format_utils import ckpt_to_state_dict
    from lingbotvla.utils.dit_utils import save_model_weights

    run_dir = find_run_dir(checkpoint)
    lora_alpha, lora_rank, ckpt_manager = load_lora_config(run_dir)
    print(f"Loading {ckpt_manager} checkpoint: {checkpoint}")
    state_dict = ckpt_to_state_dict(checkpoint, run_dir, ckpt_manager=ckpt_manager)
    merged, module_count = merge_lora_state_dict(state_dict, lora_alpha, lora_rank)
    print(f"Merged {module_count} LoRA modules (alpha={lora_alpha:g}, rank={lora_rank})")

    output.mkdir(parents=True)
    assets_dir = run_dir / "model_assets"
    if not assets_dir.is_dir():
        raise FileNotFoundError(f"Model assets directory not found: {assets_dir}")
    for source in assets_dir.iterdir():
        destination = output / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)

    save_model_weights(output, merged, save_dtype=args.save_dtype)
    print(f"Export complete: {output}")


if __name__ == "__main__":
    main()
