"""Attention backend compatibility for LingBot-VLA v2."""

from __future__ import annotations

import os
import warnings


def prepare_attention_imports() -> str:
    """Disable broken FlashAttention discovery before V2 modules are imported."""

    requested = os.getenv("KUAVO_LINGBOT_V2_ATTENTION_BACKEND", "auto").strip().lower()
    if requested not in {"auto", "flash_attention_2", "sdpa", "eager"}:
        raise ValueError(f"Unsupported LingBot-VLA v2 attention backend: {requested}")

    backend = requested
    if requested in {"auto", "flash_attention_2"}:
        try:
            import flash_attn  # noqa: F401

            backend = "flash_attention_2"
        except (ImportError, OSError) as exc:
            if requested == "flash_attention_2":
                raise RuntimeError("FlashAttention was explicitly requested but cannot be imported") from exc
            backend = "sdpa"
            warnings.warn(
                f"FlashAttention is unavailable ({exc}); falling back to SDPA.",
                RuntimeWarning,
                stacklevel=2,
            )

    os.environ["KUAVO_LINGBOT_V2_EFFECTIVE_ATTENTION"] = backend
    if backend != "flash_attention_2":
        import transformers.modeling_flash_attention_utils as flash_utils
        import transformers.utils.import_utils as import_utils

        import_utils.is_flash_attn_2_available = lambda: False
        import_utils.is_flash_attn_3_available = lambda: False
        flash_utils.is_flash_attn_available = lambda: False
    return backend


def patch_v2_attention_constructors(backend: str) -> None:
    """Override upstream hardcoded FlashAttention configs without changing weights."""

    if backend == "flash_attention_2":
        return

    from lingbotvla.models.vla.lingbot_vla import modeling_lingbot_vla_v2 as modeling

    def patch_from_config(model_cls):
        original = model_cls._from_config.__func__

        def from_config(cls, config, **kwargs):
            config._attn_implementation = backend
            for child_name in ("text_config", "vision_config"):
                child = getattr(config, child_name, None)
                if child is not None:
                    child._attn_implementation = backend
            return original(cls, config, **kwargs)

        model_cls._from_config = classmethod(from_config)

    patch_from_config(modeling.Qwen3VLForConditionalGeneration)
    patch_from_config(modeling.Qwen2ForCausalLM)
