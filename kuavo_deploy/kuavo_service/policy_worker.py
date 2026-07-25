"""Load a Kuavo backend and expose it as a msgpack WebSocket worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _postprocess_action(postprocessor, action):
    try:
        return postprocessor(action)
    except Exception:
        original_shape = action.shape
        flattened = action.reshape(-1, original_shape[-1])
        output = postprocessor(flattened)
        return output.reshape(*original_shape)


@dataclass
class LocalPolicyWorker:
    backend: str
    policy: Any
    preprocessor: Any
    postprocessor: Any

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        import torch

        model_observation = {
            key: value
            for key, value in observation.items()
            if key.startswith("observation.") or key in {"prompt", "task"}
        }
        if self.backend in {"act", "diffusion"}:
            # Classic LeRobot processors don't consume language keys.
            model_observation.pop("prompt", None)
            model_observation.pop("task", None)
        batch = self.preprocessor(model_observation)
        with torch.inference_mode():
            if self.backend == "diffusion":
                action = self.policy.select_action(batch)
            elif hasattr(self.policy, "predict_action_chunk"):
                action = self.policy.predict_action_chunk(batch)
            else:
                action = self.policy.select_action(batch)
        action = _postprocess_action(self.postprocessor, action)
        actions = _to_numpy(action).astype(np.float32, copy=False)
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise ValueError(f"Worker expects batch size 1, got actions {actions.shape}")
            actions = actions[0]
        elif actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2:
            raise ValueError(f"Worker actions must have shape [T,D], got {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("Worker actions contain non-finite values")
        return {"actions": actions}

    @property
    def metadata(self) -> dict[str, Any]:
        config = getattr(self.policy, "config", None)
        action_dim = None
        output_features = getattr(config, "output_features", {}) if config is not None else {}
        if "action" in output_features:
            shape = getattr(output_features["action"], "shape", None)
            if shape:
                action_dim = int(shape[-1])
        action_dim = action_dim or getattr(self.policy, "action_dim", None)
        horizon = 1 if self.backend == "diffusion" else (
            getattr(config, "chunk_size", None)
            or getattr(config, "n_action_steps", None)
            or getattr(config, "horizon", None)
        )
        camera_keys = tuple(
            key
            for key in getattr(config, "input_features", {})
            if key.startswith("observation.images.")
        )
        return {
            "backend": self.backend,
            "checkpoint_format": "lerobot" if self.backend in {"act", "diffusion"} else "hf",
            "action_semantics": "absolute_joint_target",
            "action_dim": int(action_dim) if action_dim else None,
            "action_horizon": int(horizon) if horizon else None,
            "camera_keys": camera_keys,
        }


def load_local_worker(
    *,
    backend: str,
    policy_path: str | Path,
    device: str = "cuda",
    lingbot_root: str = "",
    qwen_path: str = "",
    robot_name: str = "",
    norm_stats_file: str = "",
    task_prompt: str = "",
    use_compile: bool = False,
) -> LocalPolicyWorker:
    import torch

    from kuavo_deploy.utils.policy_loader import load_policy_and_processors

    if backend not in {"act", "diffusion", "lingbot", "lingbot_v2"}:
        raise ValueError(f"Unsupported worker backend: {backend}")
    policy_kwargs: dict[str, Any] = {}
    if backend == "lingbot":
        policy_kwargs = {
            "lingbot_root": lingbot_root,
            "qwen25_path": qwen_path,
            "robot_name": robot_name or "kuavo_v1_right_arm",
            "norm_stats_file": norm_stats_file,
            "task_prompt": task_prompt,
            "use_compile": use_compile,
        }
    elif backend == "lingbot_v2":
        policy_kwargs = {
            "lingbot_v2_root": lingbot_root,
            "qwen3vl_path": qwen_path,
            "robot_name": robot_name or "kuavo_v2_right_arm",
            "norm_stats_file": norm_stats_file,
            "task_prompt": task_prompt,
            "use_compile": use_compile,
        }
    policy, preprocessor, postprocessor = load_policy_and_processors(
        policy_path,
        backend,
        torch.device(device),
        policy_kwargs=policy_kwargs,
    )
    return LocalPolicyWorker(
        backend=backend,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
