#!/usr/bin/env python3
"""Run the upstream LingBot-VLA v2 trainer with Kuavo LoRA hooks."""

from __future__ import annotations

import importlib.util
from copy import deepcopy
import os
from pathlib import Path
import sys

import lingbotvla.utils.async_hf_checkpoint as async_hf_checkpoint

from kuavo_train.lingbot_v2.lora import (
    LingbotV2LoraSettings,
    apply_lora,
    merge_lora_state_dict,
)
from kuavo_train.lingbot_v2.attention_compat import (
    patch_v2_attention_constructors,
    prepare_attention_imports,
)
from kuavo_train.lingbot_v2.dependency_checks import validate_utils3d
from kuavo_train.dataset_mixture import VirtualWeightedDataset, load_dataset_sources


_ASSET_PATHS = {
    "LINGBOT_V2_MOGE_PATH": ("depth", "moge_path"),
    "LINGBOT_V2_DEPTH_PATH": ("depth", "morgbd_path"),
    "LINGBOT_V2_DINO_CKPT": ("video", "ckpt_path"),
    "LINGBOT_V2_DINO_CONFIG": ("video", "config_path"),
}


def _apply_asset_paths(arguments, environment=None) -> None:
    """Apply cloud asset locations after the upstream YAML has been parsed."""
    environment = os.environ if environment is None else environment
    align_params = arguments.train.align_params
    if not isinstance(align_params, dict):
        raise TypeError("train.align_params must be a dictionary")
    for env_name, (section, key) in _ASSET_PATHS.items():
        value = str(environment.get(env_name, "")).strip()
        if value:
            align_params.setdefault(section, {})[key] = value


def _load_upstream_trainer():
    root = Path(os.environ.get("LINGBOT_V2_ROOT", "")).expanduser()
    trainer_path = root / "tasks/vla/train_lingbotvla.py"
    if not trainer_path.is_file():
        raise FileNotFoundError(f"LingBot-VLA v2 trainer not found: {trainer_path}")
    spec = importlib.util.spec_from_file_location("lingbot_v2_upstream_trainer", trainer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load LingBot-VLA v2 trainer: {trainer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def main() -> None:
    validate_utils3d()
    settings = LingbotV2LoraSettings.from_env()
    attention_backend = prepare_attention_imports()
    trainer = _load_upstream_trainer()
    patch_v2_attention_constructors(attention_backend)

    original_parse_args = trainer.parse_args

    def parse_args_with_cloud_assets(*args, **kwargs):
        arguments = original_parse_args(*args, **kwargs)
        _apply_asset_paths(arguments)
        return arguments

    trainer.parse_args = parse_args_with_cloud_assets

    original_build = trainer.build_foundation_model

    def build_with_lora(*args, **kwargs):
        model = original_build(*args, **kwargs)
        return apply_lora(model, settings)

    trainer.build_foundation_model = build_with_lora

    original_build_dataset = trainer.build_vla_dataset

    def build_weighted_dataset(*args, **kwargs):
        sources = load_dataset_sources()
        if not sources:
            return original_build_dataset(*args, **kwargs)
        dataset_config = kwargs.get("dataset_config")
        if dataset_config is None:
            raise ValueError("LingBot-v2 mixture requires dataset_config")
        configured_data_name = os.environ.get("LINGBOT_V2_DATA_NAME", "").strip()
        current_data_name = str(
            getattr(dataset_config, "data_name", "") or ""
        ).strip()
        data_name = configured_data_name or current_data_name
        if not data_name or data_name == "multi":
            raise ValueError(
                "LingBot-v2 dataset mixture needs a concrete robot data name; "
                "set LINGBOT_V2_DATA_NAME"
            )
        datasets = []
        for source in sources:
            source_config = deepcopy(dataset_config)
            source_config.train_path = source.root
            source_config.data_name = data_name
            source_kwargs = dict(kwargs)
            source_kwargs["dataset_config"] = source_config
            datasets.append(original_build_dataset(*args, **source_kwargs))
        mixture = VirtualWeightedDataset(datasets, sources, metadata=None)
        print(
            "LingBot-v2 dataset mixture:",
            [
                (source.repo_id, weight, source.root)
                for source, weight in zip(
                    sources, mixture.realized_weights, strict=True
                )
            ],
        )
        return mixture

    trainer.build_vla_dataset = build_weighted_dataset

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
