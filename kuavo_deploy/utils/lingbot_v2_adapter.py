"""Kuavo deployment adapter for LingBot-VLA v2."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml

from kuavo_deploy.utils.lingbot_adapter import _to_hwc_uint8


def _resolve_root(root: str) -> Path:
    candidates = [
        Path(root).expanduser() if root else None,
        Path(os.getenv("LINGBOT_V2_ROOT", "")).expanduser() if os.getenv("LINGBOT_V2_ROOT") else None,
        Path(__file__).resolve().parents[3] / "lingbot-vla-v2",
        Path(__file__).resolve().parents[2] / "third_party/lingbot-vla-v2",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "deploy/lingbot_vla_v2_policy.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError("LingBot-VLA v2 repo not found; set inference.lingbot_v2_root")


class LingbotV2DeployPolicy:
    def __init__(
        self,
        model_path: str | Path,
        *,
        lingbot_v2_root: str = "",
        qwen3vl_path: str = "",
        robot_name: str = "kuavo_v2",
        task_prompt: str = "robot manipulation",
        use_length: int = 5,
        chunk_ret: bool = True,
        norm_stats_file: str = "",
        use_compile: bool = False,
    ) -> None:
        root = _resolve_root(lingbot_v2_root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        os.environ["LINGBOT_V2_ROOT"] = str(root)
        if qwen3vl_path:
            qwen_path = Path(qwen3vl_path).expanduser().resolve()
            if not qwen_path.is_dir():
                raise FileNotFoundError(f"Qwen3-VL path not found: {qwen_path}")
            os.environ["QWEN3VL_PATH"] = str(qwen_path)

        deploy_pkg = sys.modules.get("deploy")
        if deploy_pkg is not None:
            package_paths = [Path(item).resolve() for item in getattr(deploy_pkg, "__path__", [])]
            if root / "deploy" not in package_paths:
                raise RuntimeError(
                    "A different LingBot deploy package is already imported. "
                    "Run LingBot-VLA v1 and v2 in separate processes."
                )
        module = importlib.import_module("deploy.lingbot_vla_v2_policy")

        self.model_path = str(Path(model_path).expanduser().resolve())
        self.robot_name = robot_name
        self.task_prompt = task_prompt or "robot manipulation"
        self.action_dim = self._raw_dim_from_robot_config(robot_name, "actions", "action")
        self.state_dim = self._raw_dim_from_robot_config(robot_name, "states", "observation.state")
        self._validate_norm_stats(norm_stats_file)
        self.policy = module.LingbotVLAv2Server(
            path_to_pi_model=self.model_path,
            robot_norm_path=norm_stats_file or None,
            use_length=use_length,
            chunk_ret=chunk_ret,
            use_bf16=True,
            use_fp32=False,
            use_compile=use_compile,
        )
        self.policy.reset(robo_name=self.robot_name)

        action_dim = self.action_dim
        state_dim = self.state_dim
        chunk_size = int(getattr(self.policy.config, "chunk_size", 50))
        self.config = SimpleNamespace(
            type="lingbot_v2",
            input_features={
                "observation.images.head_cam_h": SimpleNamespace(shape=(3, 480, 848)),
                "observation.images.wrist_cam_l": SimpleNamespace(shape=(3, 480, 848)),
                "observation.images.wrist_cam_r": SimpleNamespace(shape=(3, 480, 848)),
                "observation.state": SimpleNamespace(shape=(state_dim,)),
            },
            output_features={"action": SimpleNamespace(shape=(action_dim,))},
            image_features={},
            depth_features={},
            chunk_size=chunk_size,
            n_action_steps=use_length if use_length > 0 else chunk_size,
            action_delta_indices=list(range(chunk_size)),
            observation_delta_indices=None,
            reward_delta_indices=None,
        )

    def eval(self):
        return self

    def to(self, _device):
        return self

    def reset(self):
        self.policy.reset(robo_name=self.robot_name)
        return self

    @staticmethod
    def _validate_norm_stats(norm_stats_file: str) -> None:
        if not norm_stats_file:
            raise ValueError("LingBot-VLA v2 deployment requires lingbot_norm_stats_file")
        with open(norm_stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f).get("norm_stats", {})
        for key in ("action.arm.position", "action.effector.position"):
            if stats.get(key, {}).get("mean") is None:
                raise KeyError(f"Missing {key}.mean in {norm_stats_file}")

    @staticmethod
    def _raw_dim_from_robot_config(robot_name: str, category: str, origin_key: str) -> int:
        robot_config = Path("configs/robot_configs") / f"{robot_name}.yaml"
        if not robot_config.is_file():
            raise FileNotFoundError(f"LingBot-VLA v2 robot config not found: {robot_config}")
        with open(robot_config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        max_end = 0
        for feature_info in config.get(category, []):
            if not isinstance(feature_info, dict):
                continue
            spec = next(iter(feature_info.values()))
            origins = spec.get("origin_keys")
            if isinstance(origins, str):
                if origins == origin_key:
                    return -1
                continue
            if not isinstance(origins, list):
                continue
            for item in origins:
                for key, span in item.items():
                    if key == origin_key and "end" in span:
                        max_end = max(max_end, int(span["end"]))
        if max_end <= 0:
            raise ValueError(f"Cannot infer {origin_key} dim from {robot_config}")
        return max_end

    def _state(self, observation: dict[str, Any]) -> np.ndarray:
        for key in ("observation.state", "state", "state.state"):
            if key in observation:
                value = observation[key]
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                state = np.asarray(value, dtype=np.float32).reshape(-1)
                if state.shape[0] != self.state_dim:
                    raise ValueError(
                        f"LingBot-VLA v2 Kuavo state must be {self.state_dim}-D, got {state.shape}"
                    )
                return state
        raise KeyError("Missing observation.state")

    def _payload(self, observation: dict[str, Any]) -> dict[str, Any]:
        def image(*keys):
            for key in keys:
                if key in observation:
                    return observation[key]
            return None

        head = image("observation.images.head_cam_h", "observation.images.camera_top")
        left = image("observation.images.wrist_cam_l", "observation.images.camera_wrist_left")
        right = image("observation.images.wrist_cam_r", "observation.images.camera_wrist_right")
        if head is None or (left is None and right is None):
            raise KeyError("LingBot-VLA v2 requires head RGB and at least one wrist RGB image")
        left = right if left is None else left
        right = left if right is None else right
        return {
            "observation.images.head_cam_h": _to_hwc_uint8(head),
            "observation.images.wrist_cam_l": _to_hwc_uint8(left),
            "observation.images.wrist_cam_r": _to_hwc_uint8(right),
            "observation.state": self._state(observation),
            "task": self.task_prompt,
        }

    def _infer_actions(self, observation: dict[str, Any]) -> torch.Tensor:
        output = self.policy.infer(self._payload(observation))
        if "action" not in output:
            raise KeyError(f"LingBot-VLA v2 output lacks 'action': {sorted(output)}")
        action = np.asarray(output["action"], dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected LingBot-VLA v2 action [T,{self.action_dim}], got {action.shape}"
            )
        return torch.from_numpy(action)

    def predict_action_chunk(self, observation: dict[str, Any]) -> torch.Tensor:
        return self._infer_actions(observation)

    def select_action(self, observation: dict[str, Any]) -> torch.Tensor:
        return self._infer_actions(observation)[:1]
