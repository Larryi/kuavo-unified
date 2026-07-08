#!/usr/bin/env python3
"""Run the upstream LingBot-VLA v2 trainer with Kuavo LoRA hooks."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import lingbotvla.utils.async_hf_checkpoint as async_hf_checkpoint

from kuavo_train.lingbot_v2.lora import (
    LingbotV2LoraSettings,
    apply_lora,
    merge_lora_state_dict,
)


def _load_upstream_trainer():
    root = Path(os.environ.get("LINGBOT_V2_ROOT", "")).expanduser()
    trainer_path = root / "tasks/vla/train_lingbotvla.py"
    if not trainer_path.is_file():
        raise FileNotFoundError(f"LingBot-VLA v2 trainer not found: {trainer_path}")
    spec = importlib.util.spec_from_file_location("lingbot_v2_upstream_trainer", trainer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load LingBot-VLA v2 trainer: {trainer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    settings = LingbotV2LoraSettings.from_env()
    trainer = _load_upstream_trainer()

    original_build = trainer.build_foundation_model

    def build_with_lora(*args, **kwargs):
        model = original_build(*args, **kwargs)
        return apply_lora(model, settings)

    trainer.build_foundation_model = build_with_lora

    saver_cls = trainer.AsyncHFCheckpointSaver
    original_save = saver_cls._save_one_hf_checkpoint

    def save_merged_lora(self, *args, **kwargs):
        original_converter = async_hf_checkpoint.ckpt_to_state_dict

        def merged_converter(*converter_args, **converter_kwargs):
            state = original_converter(*converter_args, **converter_kwargs)
            return merge_lora_state_dict(state, settings.alpha, settings.rank)

        async_hf_checkpoint.ckpt_to_state_dict = merged_converter
        try:
            return original_save(self, *args, **kwargs)
        finally:
            async_hf_checkpoint.ckpt_to_state_dict = original_converter

    saver_cls._save_one_hf_checkpoint = save_merged_lora
    trainer.main()


if __name__ == "__main__":
    main()
