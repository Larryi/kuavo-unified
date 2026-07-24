#!/usr/bin/env python3
"""Export a complete LingBot DCP checkpoint as deployable HF weights."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml


def find_run_dir(checkpoint: Path) -> Path:
    for parent in (checkpoint, *checkpoint.parents):
        if (parent / "lingbotvla_cli.yaml").is_file():
            return parent
    raise FileNotFoundError(f"Could not find lingbotvla_cli.yaml above {checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate checkpoint, run metadata, source, and output safety without writing.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    lingbot_root = args.lingbot_root.resolve()
    output = args.output.resolve()
    if not (checkpoint / "model" / ".metadata").is_file():
        raise FileNotFoundError(f"Incomplete DCP model checkpoint: {checkpoint}")
    if not lingbot_root.is_dir():
        raise FileNotFoundError(f"LingBot repository not found: {lingbot_root}")
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output exists: {output}; pass --force to replace it")

    run_dir = find_run_dir(checkpoint)
    config = yaml.safe_load((run_dir / "lingbotvla_cli.yaml").read_text(encoding="utf-8"))
    if config.get("train", {}).get("use_lora", False):
        raise ValueError("This exporter is for full fine-tuning checkpoints, not LoRA checkpoints")
    ckpt_manager = str(config.get("train", {}).get("ckpt_manager", "dcp"))
    assets_dir = run_dir / "model_assets"
    if not assets_dir.is_dir():
        raise FileNotFoundError(f"Model assets directory not found: {assets_dir}")

    if args.dry_run:
        print(
            "LingBot checkpoint export dry-run passed: "
            f"checkpoint={checkpoint}, manager={ckpt_manager}, output={output}"
        )
        return

    if output.exists():
        shutil.rmtree(output)

    sys.path.insert(0, str(lingbot_root))
    from lingbotvla.checkpoint.format_utils import ckpt_to_state_dict
    from lingbotvla.models.module_utils import save_model_weights

    state_dict = ckpt_to_state_dict(checkpoint, run_dir, ckpt_manager=ckpt_manager)
    output.mkdir(parents=True)
    for source in assets_dir.iterdir():
        destination = output / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    save_model_weights(output, state_dict, save_dtype=args.save_dtype)

    config_file = output / "config.json"
    index_file = output / "model.safetensors.index.json"
    if not config_file.is_file() or not index_file.is_file():
        raise RuntimeError("Export is incomplete: config.json or safetensors index is missing")
    index = json.loads(index_file.read_text(encoding="utf-8"))
    missing = sorted({name for name in index["weight_map"].values() if not (output / name).is_file()})
    if missing:
        raise RuntimeError(f"Export is missing safetensor shards: {missing}")
    print(f"Full LingBot HF checkpoint exported and verified: {output}")


if __name__ == "__main__":
    main()
