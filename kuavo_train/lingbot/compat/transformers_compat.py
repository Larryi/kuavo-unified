"""Runtime shims for LingBot on newer `transformers` versions."""

from __future__ import annotations

from typing import TypedDict
import torch


def _patch_legacy_flash_attention_arg(model_cls) -> None:
    """Translate LingBot's legacy `_from_config` flag for new Transformers."""

    marker = "_kuavo_legacy_flash_arg_patched"
    # Check the concrete class, not an inherited attribute. LingBot defines
    # several Qwen subclasses and their import order must not affect patching.
    if model_cls.__dict__.get(marker, False):
        return

    original_from_config = model_cls._from_config

    def compatible_from_config(cls, config, **kwargs):
        use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)
        if use_flash_attention_2:
            config._attn_implementation = "flash_attention_2"
        return original_from_config(config, **kwargs)

    model_cls._from_config = classmethod(compatible_from_config)
    setattr(model_cls, marker, True)


def patch_transformers_for_lingbot() -> None:
    """Patch `transformers` APIs removed or relocated in newer releases.

    LingBot's current source expects:
    - `transformers.modeling_flash_attention_utils.apply_rotary_emb`
    - `transformers.utils.LossKwargs`

    `transformers==4.57.x` no longer exposes these symbols in the same places.
    Inject lightweight shims before importing `lingbotvla`.
    """

    import sys
    import types
    import transformers.modeling_flash_attention_utils as flash_utils
    import transformers.utils as transformers_utils
    from transformers import Qwen2_5_VLForConditionalGeneration

    _patch_legacy_flash_attention_arg(Qwen2_5_VLForConditionalGeneration)
    if not hasattr(flash_utils, "apply_rotary_emb"):
        from flash_attn.layers.rotary import apply_rotary_emb as flash_apply_rotary_emb

        def apply_rotary_emb(x, cos, sin):
            if getattr(cos, "ndim", 0) == 3 and cos.shape[1] == 1:
                cos = cos.squeeze(1)
                sin = sin.squeeze(1)
            return flash_apply_rotary_emb(x, cos, sin)

        flash_utils.apply_rotary_emb = apply_rotary_emb

    if not hasattr(flash_utils, "flash_attn_varlen_func"):
        from flash_attn import flash_attn_varlen_func

        flash_utils.flash_attn_varlen_func = flash_attn_varlen_func

    if not hasattr(transformers_utils, "LossKwargs"):
        class LossKwargs(TypedDict, total=False):
            pass

        transformers_utils.LossKwargs = LossKwargs

    if "transformers.modeling_layers" not in sys.modules:
        modeling_layers = types.ModuleType("transformers.modeling_layers")

        class GradientCheckpointingLayer(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.gradient_checkpointing = False

        modeling_layers.GradientCheckpointingLayer = GradientCheckpointingLayer
        sys.modules["transformers.modeling_layers"] = modeling_layers


def patch_pi0_config_for_lingbot() -> None:
    """Backfill PI0Config fields expected by LingBot but missing in older LeRobot builds."""

    from lerobot.policies.pi0.configuration_pi0 import PI0Config

    compat_defaults = {
        "use_peft": False,
        "freeze_vision_encoder": False,
        "train_expert_only": False,
        "resize_imgs_with_padding": (224, 224),
        "adapt_to_pi_aloha": False,
        "use_delta_joint_actions_aloha": False,
        "proj_width": None,
        "num_steps": 10,
        "use_cache": True,
        "attention_implementation": "eager",
        "train_state_proj": False,
    }

    for field_name, default_value in compat_defaults.items():
        if not hasattr(PI0Config, field_name):
            setattr(PI0Config, field_name, default_value)


def patch_lingbot_model_loader() -> None:
    """Stabilize LingBot model resolution on newer dependency stacks.

    Dynamic registry discovery in the upstream loader can fail under newer
    `transformers`/typing combinations even after symbol-level shims are in
    place. For the training path used here we know exactly which config/model
    pair we need, so bind it explicitly.
    """

    from abc import update_abstractmethods

    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    import lingbotvla.models.auto as auto_mod
    import lingbotvla.models.loader as loader_mod
    from lingbotvla.models.loader import CustomizedModelingLoader
    from lingbotvla.models.vla.pi0.modeling_lingbot_vla import (
        LingbotVlaPolicy,
        Qwen2ForCausalLM as LingbotQwen2ForCausalLM,
        Qwen2_5_VLForConditionalGeneration as LingbotQwen2_5_VLForConditionalGeneration,
    )
    from lingbotvla.models.vla.pi0.modeling_pi0 import PI0Policy
    from lingbotvla.models.vla.pi0.qwenvl_in_vla import Qwen2_5_VisionTransformerPretrainedModel

    for model_cls in (
        LingbotQwen2_5_VLForConditionalGeneration,
        LingbotQwen2ForCausalLM,
        Qwen2_5_VisionTransformerPretrainedModel,
    ):
        _patch_legacy_flash_attention_arg(model_cls)

    def predict_action_chunk(self, batch, **_kwargs):
        return self.select_action(batch)

    def refresh_abstract_subclasses(policy_cls):
        for subclass in policy_cls.__subclasses__():
            update_abstractmethods(subclass)
            refresh_abstract_subclasses(subclass)

    # LeRobot 0.4.2 added predict_action_chunk to the PreTrainedPolicy ABC,
    # while the current LingBot source predates that interface. The inference
    # subclasses provide select_action through PolicyPreprocessMixin, so this
    # compatibility method safely delegates to the concrete implementation.
    for policy_cls in (LingbotVlaPolicy, PI0Policy):
        if "predict_action_chunk" in getattr(policy_cls, "__abstractmethods__", ()):
            policy_cls.predict_action_chunk = predict_action_chunk
            update_abstractmethods(policy_cls)
        refresh_abstract_subclasses(policy_cls)

    if getattr(loader_mod.get_loader, "_kuavo_patched", False):
        return

    original_get_loader = loader_mod.get_loader

    def patched_get_loader(model_config, force_use_huggingface):
        if force_use_huggingface:
            return original_get_loader(model_config, force_use_huggingface)

        if isinstance(model_config, PI0Config):
            tokenizer_path = str(getattr(model_config, "tokenizer_path", "")).lower()
            if "qwen" in tokenizer_path:
                return CustomizedModelingLoader(model_cls=LingbotVlaPolicy)
            return CustomizedModelingLoader(model_cls=PI0Policy)

        return original_get_loader(model_config, force_use_huggingface)

    patched_get_loader._kuavo_patched = True
    loader_mod.get_loader = patched_get_loader
    auto_mod.get_loader = patched_get_loader
