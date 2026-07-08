from __future__ import annotations

import os
import sys
import json
import yaml
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import torch
from kuavo_train.lingbot.compat import (
    patch_lingbot_model_loader,
    patch_pi0_config_for_lingbot,
    patch_transformers_for_lingbot,
)


def _resolve_lingbot_root(lingbot_root: str | None = None) -> Path:
    if lingbot_root:
        p = Path(lingbot_root).expanduser().resolve()
        if p.is_dir():
            return p

    env_root = os.getenv("LINGBOT_ROOT", "")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if p.is_dir():
            return p

    candidates = [
        Path("/home/yunxi/lmy/VLA/lingbot-vla"),
        (Path(__file__).resolve().parents[3] / "lingbot-vla").resolve(),
        (Path(__file__).resolve().parents[4] / "lingbot-vla").resolve(),
    ]
    for p in candidates:
        if p.is_dir():
            return p

    raise FileNotFoundError(
        "LingBot-VLA repo not found. Set `inference.lingbot_root` or env `LINGBOT_ROOT`."
    )


def _resolve_qwen25_path(model_path: str | Path, lingbot_root: Path, qwen25_path: str = "") -> str | None:
    if qwen25_path:
        candidate = Path(qwen25_path).expanduser().resolve()
        if candidate.is_dir():
            return str(candidate)

    env_path = os.getenv("QWEN25_PATH", "")
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if candidate.is_dir():
            return str(candidate)

    training_config_path = Path(model_path).expanduser().resolve().parent.parent.parent / "lingbotvla_cli.yaml"
    configured_path = ""
    if training_config_path.is_file():
        with open(training_config_path, "r", encoding="utf-8") as f:
            training_config = yaml.safe_load(f) or {}
        configured_path = str(training_config.get("model", {}).get("tokenizer_path", "") or "")
        if configured_path:
            candidate = Path(configured_path).expanduser().resolve()
            if candidate.is_dir():
                return str(candidate)

    configured_name = Path(configured_path).name if configured_path else "Qwen2.5-VL-3B-Instruct"
    candidates = [
        lingbot_root.parent / configured_name,
        Path(__file__).resolve().parents[3] / configured_name,
        Path.cwd() / configured_name,
    ]
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.is_dir():
            return str(candidate)

    return None


def _to_hwc_uint8(image: Any) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)

    # Expected from Kuavo env: (1, C, H, W) or (C, H, W).
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim != 3:
        raise ValueError(f"Expected image ndim=3 after squeeze/permute, got shape={arr.shape}")

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if arr.dtype != np.uint8:
        # Kuavo observations are usually float in [0, 1].
        arr = np.clip(arr, 0.0, 1.0) * 255.0 if arr.dtype.kind == "f" else np.clip(arr, 0, 255)
        arr = arr.astype(np.uint8)
    return arr


class _ActionDimNormalizer:
    """Keep upstream LingBot's hardcoded 14-D output aligned with Kuavo stats."""

    def __init__(self, normalizer, action_dim: int):
        self.normalizer = normalizer
        self.action_dim = action_dim

    def normalize(self, data):
        return self.normalizer.normalize(data)

    def unnormalize(self, data):
        action = data.get("action")
        if action is not None and action.shape[-1] < self.action_dim:
            raise ValueError(
                f"LingBot predicted {action.shape[-1]} action dimensions, but the configured "
                f"normalization statistics require {self.action_dim}."
            )
        if action is not None and action.shape[-1] > self.action_dim:
            data = dict(data)
            data["action"] = action[..., : self.action_dim]
        return self.normalizer.unnormalize(data)

    def __getattr__(self, name):
        return getattr(self.normalizer, name)


def _install_dynamic_action_selector(policy) -> None:
    """Replace upstream's hardcoded 14-D inference crop with checkpoint action_dim."""

    @torch.no_grad()
    def select_action(self, observation, use_bf16=False, vlm_causal=False, noise=None):
        self.eval()
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        if len(observation["images"].shape) == 4:
            observation["images"] = observation["images"].unsqueeze(0)
            observation["img_masks"] = observation["img_masks"].unsqueeze(0)

        sample_kwargs = {
            "vlm_causal": vlm_causal,
        }
        if "expert_imgs" in observation:
            sample_kwargs["expert_imgs"] = observation["expert_imgs"].to(dtype=dtype, device="cuda")
        actions = self.model.sample_actions(
            observation["images"].to(dtype=dtype, device="cuda"),
            observation["img_masks"].to(device="cuda"),
            observation["lang_tokens"].unsqueeze(0).to(device="cuda"),
            observation["lang_masks"].unsqueeze(0).to(device="cuda"),
            observation["state"].unsqueeze(0).to(dtype=dtype, device="cuda"),
            **sample_kwargs,
        )
        action_dim = int(getattr(self, "action_dim", actions.shape[-1]))
        observation["action"] = actions.squeeze(0)[..., :action_dim].float().cpu()
        if use_bf16:
            observation["state"] = observation["state"].float()
        return self.normalizer.unnormalize(observation)

    policy.select_action = MethodType(select_action, policy)


def _install_chunk_step_compat(server) -> None:
    """Recover the cached 1-D action from LingBot's broken chunk-step branch."""

    original_infer = server.infer

    def infer(self, observation, *args, **kwargs):
        try:
            return original_infer(observation, *args, **kwargs)
        except IndexError as exc:
            known_shape_bug = "too many indices" in str(exc) and "1-dimensional" in str(exc)
            if (
                self.chunk_ret
                or self.use_length <= 0
                or self.last_action_chunk is None
                or not known_shape_bug
            ):
                raise

            step_in_chunk = self.global_step % self.use_length
            action = self.last_action_chunk[step_in_chunk, : self.action_dim]
            self.global_step += 1
            return {"action": action}

    server.infer = MethodType(infer, server)


class LingbotDeployPolicy:
    """Adapter that exposes LingBot-VLA inference as `select_action(obs)`."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        lingbot_root: str = "",
        qwen25_path: str = "",
        task_prompt: str = "",
        use_length: int = 1,
        chunk_ret: bool = True,
        norm_stats_file: str = "",
        data_type: str = "robotwin",
        execute_raw_action: bool = False,
    ):
        root = _resolve_lingbot_root(lingbot_root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        patch_transformers_for_lingbot()
        patch_pi0_config_for_lingbot()
        patch_lingbot_model_loader()

        qwen25_path = _resolve_qwen25_path(model_path, root, qwen25_path)
        if qwen25_path:
            os.environ["QWEN25_PATH"] = qwen25_path

        import deploy.lingbot_robotwin_policy as lingbot_policy_module  # type: ignore
        from lingbotvla.data.vla_data.transform import Normalizer  # type: ignore

        self.model_path = str(model_path)
        self.task_prompt = task_prompt or "robot manipulation"
        self.execute_raw_action = execute_raw_action

        # QwenPiServer hardcodes data_type="robotwin" during its initial
        # Normalizer construction. Kuavo checkpoints use flat LeRobot stats,
        # so inject the configured type before model construction rather than
        # waiting for the post-load normalizer override below.
        upstream_normalizer = lingbot_policy_module.Normalizer
        upstream_normalizer_init = upstream_normalizer.__init__

        def compatible_normalizer_init(instance, *args, **kwargs):
            norm_stats = kwargs.get("norm_stats", args[0] if args else {})
            # Kuavo LeRobot stats are already flattened. Upstream LingBot
            # otherwise forces RobotWin's split arm/effector schema here.
            if "observation.state" in norm_stats and "action" in norm_stats:
                kwargs["data_type"] = "customized"
            else:
                kwargs["data_type"] = data_type
            upstream_normalizer_init(instance, *args, **kwargs)

        upstream_normalizer.__init__ = compatible_normalizer_init
        try:
            self.policy = lingbot_policy_module.QwenPiServer(
                path_to_pi_model=self.model_path,
                use_length=use_length,
                chunk_ret=chunk_ret,
                use_bf16=True,
                use_fp32=False,
            )
        finally:
            upstream_normalizer.__init__ = upstream_normalizer_init

        _install_dynamic_action_selector(self.policy.vla)
        _install_chunk_step_compat(self.policy)

        if norm_stats_file:
            with open(norm_stats_file, "r", encoding="utf-8") as f:
                norm_stats = json.load(f)
            self.policy.norm_stats_file = norm_stats_file
            action_dim = int(getattr(self.policy, "action_dim", 0))
            action_stats = norm_stats.get("norm_stats", {}).get("action", {})
            for key in ("q01", "q99", "mean", "std", "min", "max"):
                if key in action_stats:
                    action_dim = len(action_stats[key])
                    self.policy.action_dim = action_dim
                    self.policy.vla.action_dim = action_dim
                    break
            self.policy.vla.normalizer = _ActionDimNormalizer(
                Normalizer(
                    norm_stats=norm_stats["norm_stats"],
                    from_file=True,
                    data_type=data_type,
                    norm_type={
                        "observation.images.cam_high": "identity",
                        "observation.images.cam_left_wrist": "identity",
                        "observation.images.cam_right_wrist": "identity",
                        "observation.state": getattr(
                            self.policy.data_config, "norm_type", "bounds_99_woclip"
                        ),
                        "action": getattr(
                            self.policy.data_config, "norm_type", "bounds_99_woclip"
                        ),
                    },
                ),
                action_dim=action_dim,
            )

        action_dim = int(getattr(self.policy, "action_dim", 0))
        chunk_size = int(getattr(self.policy, "chunk_size", 1))
        self.config = SimpleNamespace(
            type="lingbot",
            input_features={
                "observation.images.head_cam_h": SimpleNamespace(shape=(3, 480, 848)),
                "observation.images.wrist_cam_r": SimpleNamespace(shape=(3, 480, 848)),
                "observation.state": SimpleNamespace(shape=(action_dim,)),
            },
            output_features={"action": SimpleNamespace(shape=(action_dim,))},
            image_features={
                "observation.images.head_cam_h": SimpleNamespace(shape=(3, 480, 848)),
                "observation.images.wrist_cam_r": SimpleNamespace(shape=(3, 480, 848)),
            },
            depth_features={},
            chunk_size=chunk_size,
            n_action_steps=int(use_length if use_length > 0 else chunk_size),
            reward_delta_indices=None,
            action_delta_indices=list(range(chunk_size)),
            observation_delta_indices=None,
        )

    def eval(self):
        return self

    def to(self, _device):
        return self

    def reset(self):
        # Keep a stable robot name for LingBot reset semantics.
        self.policy.reset(robo_name="kuavo")
        return self

    def _extract_state(self, obs: dict[str, Any]) -> np.ndarray:
        for key in ("observation.state", "state", "state.state"):
            if key in obs:
                v = obs[key]
                if isinstance(v, torch.Tensor):
                    arr = v.detach().cpu().numpy()
                else:
                    arr = np.asarray(v)
                if arr.ndim > 1:
                    arr = arr.reshape(-1)
                return arr.astype(np.float32)
        raise KeyError("No state key found in observation. Tried: observation.state/state/state.state")

    def _build_observation_payload(self, observation: dict[str, Any]) -> dict[str, Any]:
        def _get_first(keys: list[str]):
            for k in keys:
                if k in observation:
                    return observation[k]
            return None

        head = _get_first(["observation.images.head_cam_h", "observation.images.cam_high"])
        left = _get_first(["observation.images.wrist_cam_l", "observation.images.cam_left_wrist"])
        right = _get_first(["observation.images.wrist_cam_r", "observation.images.cam_right_wrist"])

        if head is None:
            raise KeyError(
                "LingBot requires a head RGB key: observation.images.head_cam_h or observation.images.cam_high"
            )
        if left is None and right is None:
            raise KeyError(
                "LingBot requires at least one wrist RGB key: observation.images.wrist_cam_l or observation.images.wrist_cam_r"
            )

        # Training supports missing left/right wrist cameras. Keep deployment consistent by
        # mirroring the available wrist image when only one side exists.
        if left is None:
            left = right
        if right is None:
            right = left

        return {
            "observation.images.cam_high": _to_hwc_uint8(head),
            "observation.images.cam_left_wrist": _to_hwc_uint8(left),
            "observation.images.cam_right_wrist": _to_hwc_uint8(right),
            "observation.state": self._extract_state(observation),
            "task": self.task_prompt,
        }

    def _infer_actions(self, observation: dict[str, Any]) -> torch.Tensor:
        out = self.policy.infer(self._build_observation_payload(observation))
        action_key = "raw_action" if self.execute_raw_action else "action"
        if action_key not in out:
            raise KeyError(f"LingBot inference output does not contain {action_key!r}: {sorted(out)}")
        action_np = np.asarray(out[action_key], dtype=np.float32)

        if action_np.ndim == 1:
            action_np = action_np[None, :]
        if action_np.ndim != 2:
            raise ValueError(f"Expected LingBot actions with shape [T, D], got {action_np.shape}")
        return torch.from_numpy(action_np)

    def predict_action_chunk(self, observation: dict[str, Any]) -> torch.Tensor:
        """Return every action produced by LingBot for open-loop diagnostics."""
        return self._infer_actions(observation)

    def select_action(self, observation: dict[str, Any]) -> torch.Tensor:
        """Return one action, preserving the existing real-robot deployment contract."""
        return self._infer_actions(observation)[:1]
