import json
from pathlib import Path
from typing import Any

import torch

import lerobot_patches.custom_patches  # noqa: F401
from kuavo_deploy.utils.logging_utils import setup_logger
from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper
from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from lerobot.policies.factory import make_pre_post_processors


log_model = setup_logger("model")
REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSOR_CONFIG_FILES = (
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)


def _has_nonportable_processor_steps(processor_root: Path) -> bool:
    """Reject training snapshots that serialized locally defined processor classes."""
    for name in PROCESSOR_CONFIG_FILES:
        config_path = processor_root / name
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for step in config.get("steps", []):
            class_name = step.get("class", "")
            if class_name.startswith("__main__."):
                return True
    return False


def resolve_policy_path(cfg: Any) -> Path:
    """Resolve old epoch-style paths and new LeRobot checkpoint paths."""
    policy_path = getattr(cfg, "policy_path", None) or getattr(cfg, "pretrained_path", None)
    if policy_path:
        path = Path(policy_path).expanduser()
        if path.is_absolute() or path.exists():
            return path
        return REPO_ROOT / path

    return REPO_ROOT / f"outputs/train/{cfg.task}/{cfg.method}/{cfg.timestamp}/epoch{cfg.epoch}"


def resolve_eval_output_dir(cfg: Any, pretrained_path: Path) -> Path:
    if getattr(cfg, "policy_path", None) or getattr(cfg, "pretrained_path", None):
        return Path(f"outputs/eval/{cfg.task}/{cfg.method}/{cfg.timestamp}/epoch{cfg.epoch}")
    return Path(f"outputs/eval/{cfg.task}/{cfg.method}/{cfg.timestamp}/epoch{cfg.epoch}")


def resolve_processor_root(pretrained_path: str | Path) -> Path:
    """Find the run directory that owns the saved pre/post processors."""
    policy_path = Path(pretrained_path).expanduser().resolve()
    candidates = (policy_path, *policy_path.parents)
    rejected: list[Path] = []
    for candidate in candidates:
        if all((candidate / name).is_file() for name in PROCESSOR_CONFIG_FILES):
            if _has_nonportable_processor_steps(candidate):
                rejected.append(candidate)
                log_model.warning(
                    "Skipping non-portable training processor snapshot: %s",
                    candidate,
                )
                continue
            return candidate
    if rejected:
        rejected_text = ", ".join(str(path) for path in rejected)
        raise FileNotFoundError(
            "Only non-portable training processor snapshots were found at or above "
            f"checkpoint {policy_path}: {rejected_text}. "
            "Use the run bundle containing the inference-safe root processors."
        )
    raise FileNotFoundError(
        f"Could not find {PROCESSOR_CONFIG_FILES} at or above checkpoint {policy_path}"
    )


def load_policy_and_processors(
    pretrained_path,
    policy_type: str,
    device: torch.device,
    policy_kwargs: dict[str, Any] | None = None,
):
    pretrained_path = Path(pretrained_path).expanduser()
    policy_kwargs = policy_kwargs or {}

    if device.type == "cpu":
        log_model.warning("Warning: Using CPU for inference, this may be slow.")

    if policy_type == "diffusion":
        policy = CustomDiffusionPolicyWrapper.from_pretrained(pretrained_path, strict=True)
        processor_root = resolve_processor_root(pretrained_path)
        preprocessor, postprocessor = make_pre_post_processors(
            None,
            processor_root,
            preprocessor_overrides={
                "device_processor": {"device": str(device)},
            },
        )
    elif policy_type == "act":
        policy = CustomACTPolicyWrapper.from_pretrained(pretrained_path, strict=True)
        processor_root = resolve_processor_root(pretrained_path)
        preprocessor, postprocessor = make_pre_post_processors(
            None,
            processor_root,
            preprocessor_overrides={
                "device_processor": {"device": str(device)},
            },
        )
    elif policy_type == "lingbot":
        if device.type != "cuda":
            raise ValueError("LingBot inference currently requires a CUDA device.")
        from kuavo_deploy.utils.lingbot_adapter import LingbotDeployPolicy

        policy = LingbotDeployPolicy(model_path=pretrained_path, **policy_kwargs)
        preprocessor, postprocessor = (lambda obs: obs), (lambda action: action)
    elif policy_type == "lingbot_v2":
        if device.type != "cuda":
            raise ValueError("LingBot-VLA v2 inference requires a CUDA device.")
        from kuavo_deploy.utils.lingbot_v2_adapter import LingbotV2DeployPolicy

        policy = LingbotV2DeployPolicy(model_path=pretrained_path, **policy_kwargs)
        preprocessor, postprocessor = (lambda obs: obs), (lambda action: action)
    else:
        raise ValueError(f"Unsupported local policy type: {policy_type}")

    policy.eval()
    policy.to(device)
    policy.reset()

    log_model.info(f"Model loaded from {pretrained_path}")
    log_model.info(f"Model type: {policy_type}")
    if hasattr(policy.config, "n_obs_steps"):
        log_model.info(f"Model n_obs_steps: {policy.config.n_obs_steps}")
    if hasattr(policy.config, "n_action_steps"):
        log_model.info(f"Model n_action_steps: {policy.config.n_action_steps}")
    log_model.info(f"Model device: {device}")
    return policy, preprocessor, postprocessor
