"""Optional, component-level ``torch.compile`` support for diffusion training."""

from __future__ import annotations

from typing import Any

import torch


def _torch_compile_kwargs(compile_cfg: Any) -> dict[str, Any]:
    kwargs = {
        "mode": compile_cfg.get("mode", "default"),
        "fullgraph": compile_cfg.get("fullgraph", False),
        "dynamic": compile_cfg.get("dynamic", False),
    }
    backend = compile_cfg.get("backend")
    if backend:
        kwargs["backend"] = backend
    return kwargs


def _compile_callable(
    owner: Any,
    attr_name: str,
    compile_kwargs: dict[str, Any],
    label: str,
    log=print,
) -> bool:
    fn = getattr(owner, attr_name, None)
    if not callable(fn):
        log(f"[torch.compile] skip missing callable: {label}")
        return False
    try:
        compiled = torch.compile(fn, **compile_kwargs)
    except Exception as exc:
        log(f"[torch.compile] failed {label}: {exc}; using eager mode.")
        return False
    setattr(owner, attr_name, compiled)
    log(f"[torch.compile] compiled callable: {label}")
    return True


def _compile_module_forward(
    module: Any,
    compile_kwargs: dict[str, Any],
    label: str,
    log=print,
) -> int:
    if module is None:
        log(f"[torch.compile] skip missing module: {label}")
        return 0
    if isinstance(module, torch.nn.ModuleList):
        return sum(
            _compile_module_forward(child, compile_kwargs, f"{label}.{index}", log)
            for index, child in enumerate(module)
        )
    if isinstance(module, torch.nn.ModuleDict):
        return sum(
            _compile_module_forward(child, compile_kwargs, f"{label}.{key}", log)
            for key, child in module.items()
        )
    if not isinstance(module, torch.nn.Module):
        log(f"[torch.compile] skip non-module: {label}")
        return 0
    try:
        compiled = torch.compile(module.forward, **compile_kwargs)
    except Exception as exc:
        log(f"[torch.compile] failed {label}: {exc}; using eager mode.")
        return 0
    module.forward = compiled
    log(f"[torch.compile] compiled module.forward: {label}")
    return 1


def maybe_compile_policy(policy: torch.nn.Module, cfg: Any, log=print) -> int:
    """Compile selected diffusion components and return the number compiled."""

    training_cfg = cfg.get("training", {})
    compile_cfg = training_cfg.get("torch_compile")
    if not compile_cfg or not compile_cfg.get("enabled", False):
        return 0
    if not hasattr(torch, "compile"):
        log("[torch.compile] unavailable in this PyTorch version; using eager mode.")
        return 0
    if cfg.get("policy_name") != "diffusion":
        log(f"[torch.compile] policy_name={cfg.get('policy_name')!r}; using eager mode.")
        return 0
    diffusion = getattr(policy, "diffusion", None)
    if diffusion is None:
        log("[torch.compile] policy has no diffusion module; using eager mode.")
        return 0

    compile_kwargs = _torch_compile_kwargs(compile_cfg)
    components = list(compile_cfg.get("components", ["compute_loss"]))
    compiled_count = 0
    module_components = {
        "unet": "unet",
        "rgb_encoder": "rgb_encoder",
        "rgb": "rgb_encoder",
        "depth_encoder": "depth_encoder",
        "depth": "depth_encoder",
        "state_encoder": "state_encoder",
        "state": "state_encoder",
        "rgb_attn": "rgb_attn_layer",
        "rgb_attention": "rgb_attn_layer",
        "depth_attn": "depth_attn_layer",
        "depth_attention": "depth_attn_layer",
        "multimodal_fuse": "multimodalfuse",
        "multimodalfuse": "multimodalfuse",
        "state_guided": "state_guided",
        "state_fuse": "state_guided",
    }
    for component in components:
        name = str(component).strip().lower()
        if name == "compute_loss":
            compiled_count += int(
                _compile_callable(
                    diffusion,
                    "compute_loss",
                    compile_kwargs,
                    "policy.diffusion.compute_loss",
                    log,
                )
            )
        elif name == "core_modules":
            for attr_name in ("unet", "rgb_encoder", "depth_encoder", "state_encoder"):
                compiled_count += _compile_module_forward(
                    getattr(diffusion, attr_name, None),
                    compile_kwargs,
                    f"policy.diffusion.{attr_name}",
                    log,
                )
        elif name in module_components:
            attr_name = module_components[name]
            compiled_count += _compile_module_forward(
                getattr(diffusion, attr_name, None),
                compile_kwargs,
                f"policy.diffusion.{attr_name}",
                log,
            )
        else:
            log(f"[torch.compile] unknown component {component!r}; skipped.")
    log(f"[torch.compile] kwargs={compile_kwargs}, compiled_count={compiled_count}")
    return compiled_count
