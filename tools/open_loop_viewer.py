#!/usr/bin/env python
"""Streamlit viewer for LeRobot policy open-loop action checks."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

os.environ["HF_DATASETS_CACHE"] = os.environ.get(
    "OPEN_LOOP_HF_DATASETS_CACHE", str(Path.cwd() / ".cache" / "huggingface" / "datasets")
)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageEnhance

import lerobot_patches.custom_patches  # noqa: F401
from kuavo_deploy.utils.action_postprocessing import (
    CausalActionPostprocessor,
    CausalChunkBoundaryBlender,
    CausalJointRateLimiter,
    ChunkBoundaryBlendConfig,
    JointRateLimitConfig,
)
from kuavo_deploy.utils.gripper_latch import (
    GripperIntentLatch,
    GripperLatchConfig,
    parse_action_indices,
)
from kuavo_deploy.utils.policy_loader import load_policy_and_processors
from lerobot.datasets.factory import resolve_delta_timestamps
import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


DEFAULT_DATASET_ROOT = "/mnt/pqssd/Real_PQ_3.0/TASK1_SZ/lerobot_trimmed"
DEFAULT_POLICY_PATH = ""
DEFAULT_LINGBOT_POLICY_PATH = (
    "/mnt/pqssd/lingbot_weights/clean_meanstd_fm_L2V2_mb16_gb16_8k_20260623_175250/"
    "checkpoints/global_step_8000/hf_ckpt"
)
DEFAULT_LINGBOT_V2_POLICY_PATH = ""
DEFAULT_LINGBOT_ROOT = "/home/larry/lingbot-vla"
DEFAULT_QWEN25_PATH = "/home/larry/Qwen2.5_VL"
DEFAULT_LINGBOT_V2_ROOT = "/home/larry/lingbot-vla-v2"
DEFAULT_QWEN3VL_PATH = "/mnt/pqssd/pretrained/Qwen3-VL-4B-Instruct"
DEFAULT_NORM_STATS = "assets/norm_stats/lerobot_trimmed.json"
DEFAULT_LINGBOT_V2_NORM_STATS = "assets/norm_stats/kuavo_v2_right_arm_meanstd.json"
DEFAULT_LINGBOT_V2_TASK2_NORM_STATS = "assets/norm_stats/kuavo_v2_bimanual_task2_meanstd.json"
DEFAULT_LINGBOT_V2_ROBOT_NAME = "kuavo_v2_right_arm"
DEFAULT_TASK = "Pick and Place the safety belt, cable and pin connector"


@contextmanager
def local_dataset_only(dataset_root: Path):
    """Prevent a local viewer session from silently contacting Hugging Face."""

    original_get_safe_version = lerobot_dataset_module.get_safe_version
    original_metadata_pull = LeRobotDatasetMetadata.pull_from_repo
    original_dataset_pull = LeRobotDataset.pull_from_repo
    original_download = LeRobotDataset.download

    def local_revision(_repo_id, revision):
        return revision

    def block_metadata_pull(_self, *_args, **_kwargs):
        raise FileNotFoundError(f"Local LeRobot metadata is incomplete under {dataset_root / 'meta'}")

    def block_data_pull(_self, *_args, **_kwargs):
        raise FileNotFoundError(f"Local LeRobot episode data is incomplete under {dataset_root / 'data'}")

    lerobot_dataset_module.get_safe_version = local_revision
    LeRobotDatasetMetadata.pull_from_repo = block_metadata_pull
    LeRobotDataset.pull_from_repo = block_data_pull
    LeRobotDataset.download = block_data_pull
    try:
        yield
    finally:
        lerobot_dataset_module.get_safe_version = original_get_safe_version
        LeRobotDatasetMetadata.pull_from_repo = original_metadata_pull
        LeRobotDataset.pull_from_repo = original_dataset_pull
        LeRobotDataset.download = original_download


def fix_episode_subset_delta_indices(dataset) -> None:
    """Use absolute frame indices when querying chunks from an episode subset."""

    if getattr(dataset, "_absolute_to_relative_idx", None) is None:
        return
    relative_to_absolute = [scalar_int(index) for index in dataset.hf_dataset["index"]]
    original_get_query_indices = dataset._get_query_indices

    def get_query_indices(_self, relative_idx: int, episode_idx: int):
        return original_get_query_indices(relative_to_absolute[relative_idx], episode_idx)

    dataset._get_query_indices = MethodType(get_query_indices, dataset)


def scalar_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu()
    if image.ndim == 4:
        image = image[-1]
    if image.shape[0] in (1, 3):
        array = image.permute(1, 2, 0).numpy()
    else:
        array = image.numpy()
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array)


def pil_to_tensor(image: Image.Image, like: torch.Tensor, force_depth: bool = False) -> torch.Tensor:
    channels = 1 if force_depth or is_depth_tensor(like) else (like.shape[-3] if like.ndim >= 3 else 1)
    if channels == 1:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
    else:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor.to(dtype=like.dtype)


def is_depth_key(key: str) -> bool:
    return key.startswith("observation.depth")


def get_depth_keys(policy) -> set[str]:
    depth_keys = set(getattr(policy.config, "depth_features", {}) or {})
    for key, feature in getattr(policy.config, "input_features", {}).items():
        feature_type = str(getattr(feature, "type", "")).upper()
        if "DEPTH" in feature_type:
            depth_keys.add(key)
    return depth_keys


def is_depth_tensor(tensor: torch.Tensor) -> bool:
    if tensor.ndim < 3:
        return True
    if tensor.shape[-3] == 1:
        return True
    return False


def normalize_depth_tensor(depth: torch.Tensor) -> torch.Tensor:
    """Convert depth observations to [..., 1, H, W] before policy preprocessing."""
    if depth.ndim == 2:
        return depth.unsqueeze(0)
    if depth.ndim == 3:
        if depth.shape[0] in (1, 3):
            chw = depth
        elif depth.shape[-1] in (1, 3):
            chw = depth.permute(2, 0, 1)
        else:
            return depth.unsqueeze(0)
        return chw.mean(dim=0, keepdim=True) if chw.shape[0] != 1 else chw
    if depth.ndim == 4:
        if depth.shape[1] in (1, 3):
            tchw = depth
        elif depth.shape[-1] in (1, 3):
            tchw = depth.permute(0, 3, 1, 2)
        else:
            return depth
        return tchw.mean(dim=1, keepdim=True) if tchw.shape[1] != 1 else tchw
    return depth


def crop_pil(image: Image.Image, crop_mode: str, center_ratio: float, manual_box: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    if crop_mode == "Manual":
        left_r, top_r, right_r, bottom_r = manual_box
        left = int(width * left_r)
        top = int(height * top_r)
        right = int(width * right_r)
        bottom = int(height * bottom_r)
        if right <= left or bottom <= top:
            return image
    elif center_ratio < 1.0:
        new_w = max(1, int(width * center_ratio))
        new_h = max(1, int(height * center_ratio))
        left = (width - new_w) // 2
        top = (height - new_h) // 2
        right = left + new_w
        bottom = top + new_h
    else:
        return image
    return image.crop((left, top, right, bottom)).resize((width, height), Image.BILINEAR)


def augment_single_visual(
    image: torch.Tensor,
    brightness: float,
    contrast: float,
    color: float,
    crop_mode: str,
    center_ratio: float,
    manual_box: tuple[float, float, float, float],
    force_depth: bool = False,
) -> torch.Tensor:
    pil = tensor_to_pil(image)
    pil = crop_pil(pil, crop_mode, center_ratio, manual_box)
    pil = ImageEnhance.Brightness(pil).enhance(brightness)
    pil = ImageEnhance.Contrast(pil).enhance(contrast)
    if not force_depth and image.shape[-3] != 1:
        pil = ImageEnhance.Color(pil).enhance(color)
    return pil_to_tensor(pil, image, force_depth=force_depth)


def augment_visual(
    image: torch.Tensor,
    brightness: float,
    contrast: float,
    color: float,
    crop_mode: str,
    center_ratio: float,
    manual_box: tuple[float, float, float, float],
    force_depth: bool = False,
) -> torch.Tensor:
    if image.ndim == 4:
        frames = [
            augment_single_visual(
                frame, brightness, contrast, color, crop_mode, center_ratio, manual_box, force_depth
            )
            for frame in image
        ]
        return torch.stack(frames, dim=0)
    return augment_single_visual(image, brightness, contrast, color, crop_mode, center_ratio, manual_box, force_depth)


def is_visual_key(key: str) -> bool:
    return key.startswith("observation.images.") or key.startswith("observation.depth")


def missing_required_keys(policy, sample: dict) -> list[str]:
    if str(getattr(policy.config, "type", "")) == "lingbot_v2":
        missing = []
        if "observation.state" not in sample:
            missing.append("observation.state")
        if "observation.images.head_cam_h" not in sample and "observation.images.camera_top" not in sample:
            missing.append("observation.images.head_cam_h|observation.images.camera_top")
        has_wrist = any(
            key in sample
            for key in (
                "observation.images.wrist_cam_l",
                "observation.images.wrist_cam_r",
                "observation.images.camera_wrist_left",
                "observation.images.camera_wrist_right",
            )
        )
        if not has_wrist:
            missing.append("observation.images.wrist_cam_l|observation.images.wrist_cam_r")
        return missing

    required = set(policy.config.input_features)
    required.update(getattr(policy.config, "image_features", {}) or {})
    required.update(getattr(policy.config, "depth_features", {}) or {})
    return sorted(key for key in required if key not in sample)


def visual_keys_for_policy(policy, sample: dict) -> list[str]:
    if str(getattr(policy.config, "type", "")) == "lingbot_v2":
        candidates = (
            "observation.images.head_cam_h",
            "observation.images.camera_top",
            "observation.images.wrist_cam_l",
            "observation.images.wrist_cam_r",
            "observation.images.camera_wrist_left",
            "observation.images.camera_wrist_right",
        )
        return [key for key in candidates if key in sample]

    depth_keys = get_depth_keys(policy)
    return [
        key for key in policy.config.input_features if is_visual_key(key) or key in depth_keys
    ]


def observation_for_policy(policy, sample: dict, task: str) -> dict:
    if str(getattr(policy.config, "type", "")) == "lingbot_v2":
        candidates = (
            "observation.state",
            "observation.images.head_cam_h",
            "observation.images.camera_top",
            "observation.images.wrist_cam_l",
            "observation.images.wrist_cam_r",
            "observation.images.camera_wrist_left",
            "observation.images.camera_wrist_right",
        )
        observation = {key: sample[key] for key in candidates if key in sample}
        observation["task"] = sample.get("task") or task
        return observation

    return {key: sample[key] for key in policy.config.input_features if key in sample}


def prepare_depth_batch(policy, batch: dict) -> None:
    """Force legacy Kuavo depth policies to receive observation.depth as 1-channel tensors."""
    depth_keys = [key for key in get_depth_keys(policy) if key in batch and isinstance(batch[key], torch.Tensor)]
    for key in depth_keys:
        batch[key] = normalize_depth_tensor(batch[key])
    if depth_keys:
        batch["observation.depth"] = [batch[key] for key in depth_keys]


def ensure_bchw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim == 5 and tensor.shape[1] == 1:
        return tensor[:, 0]
    return tensor


def predict_act_direct(policy, batch: dict) -> torch.Tensor:
    """Bypass legacy ACT wrapper depth reshaping and feed the model exactly what it expects."""
    model_batch = dict(batch)
    image_keys = list(getattr(policy.config, "image_features", {}) or {})
    depth_keys = [key for key in get_depth_keys(policy) if key in model_batch]

    if image_keys:
        model_batch["observation.images"] = [ensure_bchw(model_batch[key]) for key in image_keys]
    if depth_keys:
        model_batch["observation.depth"] = [
            ensure_bchw(normalize_depth_tensor(model_batch[key])) for key in depth_keys
        ]

    return policy.model(model_batch)[0]


def normalize_pred_actions(action: torch.Tensor) -> torch.Tensor:
    action = action.detach().cpu().float()
    if action.ndim == 1:
        return action.unsqueeze(0)
    if action.ndim == 2:
        return action
    if action.ndim == 3 and action.shape[0] == 1:
        return action.squeeze(0)
    if action.ndim == 3 and action.shape[1] == 1:
        return action.squeeze(1)
    return action.reshape(-1, action.shape[-1])


def normalize_gt_actions(action: torch.Tensor, action_dim: int) -> torch.Tensor:
    action = action.detach().cpu().float()
    if action.ndim == 1:
        return action.unsqueeze(0)
    if action.shape[-1] == action_dim:
        return action.reshape(-1, action_dim)
    if action.shape[0] == action_dim:
        return action.transpose(0, 1).reshape(-1, action_dim)
    return action.reshape(action.shape[0], -1)


def current_action_state(sample: dict, action_dim: int) -> torch.Tensor:
    """Read the latest action-shaped state vector used to seed causal filters."""

    value = torch.as_tensor(sample["observation.state"]).detach().cpu().float()
    if value.ndim == 1 and value.numel() == action_dim:
        return value
    if value.ndim >= 2 and value.shape[-1] == action_dim:
        return value.reshape(-1, action_dim)[-1]
    raise ValueError(
        f"observation.state shape {tuple(value.shape)} cannot seed action_dim={action_dim}."
    )


@st.cache_data(show_spinner="Computing dataset action-rate limits...")
def compute_dataset_action_limits(
    _dataset,
    action_indices: tuple[int, ...],
    percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(_dataset.hf_dataset["action"], dtype=np.float64)
    episodes = np.asarray(_dataset.hf_dataset["episode_index"], dtype=np.int64)
    if actions.ndim != 2:
        raise ValueError(f"Expected dataset actions [frames, dim], got {actions.shape}.")
    if any(index < 0 or index >= actions.shape[-1] for index in action_indices):
        raise ValueError(f"Joint indices {action_indices} exceed action_dim={actions.shape[-1]}.")
    first_valid = episodes[1:] == episodes[:-1]
    first_delta = np.diff(actions, axis=0)[first_valid][:, action_indices]
    second_valid = (episodes[2:] == episodes[1:-1]) & (episodes[1:-1] == episodes[:-2])
    second_delta = (actions[2:] - 2.0 * actions[1:-1] + actions[:-2])[second_valid][
        :, action_indices
    ]
    if len(first_delta) == 0 or len(second_delta) == 0:
        raise ValueError("The dataset needs at least three consecutive frames in one episode.")
    return (
        np.percentile(np.abs(first_delta), percentile, axis=0),
        np.percentile(np.abs(second_delta), percentile, axis=0),
    )


def make_action_postprocessor(
    initial_action: torch.Tensor,
    joint_indices: tuple[int, ...],
    max_delta: np.ndarray | None,
    max_second_delta: np.ndarray | None,
    blend_steps: int,
    gripper_latch_config: GripperLatchConfig | None,
) -> CausalActionPostprocessor:
    blender = None
    if blend_steps:
        blender = CausalChunkBoundaryBlender(
            ChunkBoundaryBlendConfig(joint_indices, blend_steps=blend_steps)
        )
    limiter = None
    if max_delta is not None:
        limiter = CausalJointRateLimiter(
            JointRateLimitConfig(
                joint_indices,
                tuple(float(value) for value in max_delta),
                None
                if max_second_delta is None
                else tuple(float(value) for value in max_second_delta),
            )
        )
    latch = GripperIntentLatch(gripper_latch_config) if gripper_latch_config else None
    return CausalActionPostprocessor(
        initial_action,
        boundary_blender=blender,
        rate_limiter=limiter,
        gripper_latch=latch,
    )


def read_raw_gt_chunk(dataset, relative_frame: int, horizon: int) -> torch.Tensor:
    """Read GT actions directly from parquet rows, bypassing delta-query padding."""

    row = dataset.hf_dataset[relative_frame]
    episode_index = scalar_int(row["episode_index"])
    absolute_index = scalar_int(row["index"])
    episode = dataset.meta.episodes[episode_index]
    episode_end = int(episode["dataset_to_index"])

    absolute_to_relative = getattr(dataset, "_absolute_to_relative_idx", None)
    if absolute_to_relative is None:
        absolute_to_relative = {i: i for i in range(len(dataset.hf_dataset))}

    actions = []
    for offset in range(max(1, horizon)):
        target_absolute = min(absolute_index + offset, episode_end - 1)
        target_relative = absolute_to_relative[target_absolute]
        action = dataset.hf_dataset[target_relative]["action"]
        actions.append(torch.as_tensor(action).detach().cpu().float().reshape(-1))
    return torch.stack(actions)


@st.cache_resource(show_spinner="Loading policy checkpoint...")
def load_model(
    policy_path: str,
    policy_type: str,
    device_name: str,
    lingbot_root: str,
    qwen25_path: str,
    norm_stats_file: str,
    robot_name: str,
    use_compile: bool,
    task: str,
):
    device = torch.device(device_name if torch.cuda.is_available() or not device_name.startswith("cuda") else "cpu")
    policy_kwargs = None
    if policy_type == "lingbot":
        policy_kwargs = {
            "lingbot_root": lingbot_root,
            "qwen25_path": qwen25_path,
            "task_prompt": task,
            "use_length": 50,
            "chunk_ret": True,
            "norm_stats_file": norm_stats_file,
            "data_type": "customized",
            "execute_raw_action": False,
        }
    elif policy_type == "lingbot_v2":
        policy_kwargs = {
            "lingbot_v2_root": lingbot_root,
            "qwen3vl_path": qwen25_path,
            "robot_name": robot_name,
            "task_prompt": task,
            "use_length": 50,
            "chunk_ret": True,
            "norm_stats_file": norm_stats_file,
            "use_compile": use_compile,
        }
    return load_policy_and_processors(
        Path(policy_path), policy_type, device, policy_kwargs=policy_kwargs
    )


@st.cache_resource(show_spinner="Loading LeRobot dataset...")
def load_dataset(
    repo_id: str,
    dataset_root: str,
    policy_path: str,
    policy_type: str,
    device_name: str,
    episodes: tuple[int, ...],
    lingbot_root: str,
    qwen25_path: str,
    norm_stats_file: str,
    robot_name: str,
    use_compile: bool,
    task: str,
    video_backend: str,
):
    policy, _, _ = load_model(
        policy_path,
        policy_type,
        device_name,
        lingbot_root,
        qwen25_path,
        norm_stats_file,
        robot_name,
        use_compile,
        task,
    )
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    with local_dataset_only(dataset_root_path):
        ds_meta = LeRobotDatasetMetadata(repo_id, root=dataset_root_path)
        delta_timestamps = resolve_delta_timestamps(policy.config, ds_meta)
        dataset = LeRobotDataset(
            repo_id,
            root=dataset_root_path,
            episodes=list(episodes),
            delta_timestamps=delta_timestamps,
            video_backend=video_backend or None,
        )
    fix_episode_subset_delta_indices(dataset)
    return dataset


def predict_actions(policy, preprocessor, postprocessor, sample: dict, task: str, mode: str, policy_type: str) -> torch.Tensor:
    observation = observation_for_policy(policy, sample, task)
    batch = preprocessor(observation)
    for key in list(batch):
        if is_depth_key(key) and isinstance(batch[key], torch.Tensor):
            batch[key] = normalize_depth_tensor(batch[key])
    prepare_depth_batch(policy, batch)
    with torch.inference_mode():
        policy.reset()
        if mode == "chunk" and policy_type == "diffusion":
            first_action = policy.select_action(batch)
            queued_actions = list(getattr(policy, "_queues", {}).get("action", []))
            action = torch.stack([first_action, *queued_actions], dim=1)
        elif policy_type == "act" and mode == "chunk":
            action = predict_act_direct(policy, batch)
        elif mode == "chunk" and hasattr(policy, "predict_action_chunk"):
            action = policy.predict_action_chunk(batch)
        else:
            action = policy.select_action(batch)
        action = postprocessor(action)
    return normalize_pred_actions(action)


def build_episode_timeline(
    dataset,
    policy,
    preprocessor,
    postprocessor,
    task: str,
    policy_type: str,
    frame_limit: int,
    inference_stride: int,
    aggregation: str,
    augment_kwargs: dict,
    joint_indices: tuple[int, ...] = (),
    max_delta: np.ndarray | None = None,
    max_second_delta: np.ndarray | None = None,
    blend_steps: int = 0,
    gripper_latch_config: GripperLatchConfig | None = None,
) -> dict:
    """Run independent chunk predictions over ground-truth episode observations."""
    frame_count = min(frame_limit, len(dataset))
    if frame_count <= 0:
        raise ValueError("The selected episode contains no frames.")

    first_gt = read_raw_gt_chunk(dataset, 0, 1)
    action_dim = first_gt.shape[-1]
    gt = np.full((frame_count, action_dim), np.nan, dtype=np.float32)
    pred_sum = np.zeros((frame_count, action_dim), dtype=np.float64)
    raw_pred_sum = np.zeros((frame_count, action_dim), dtype=np.float64)
    pred_count = np.zeros(frame_count, dtype=np.int32)
    inference_frames: list[int] = []
    inference_ms: list[float] = []
    depth_keys = get_depth_keys(policy)
    postprocessing_requested = bool(
        max_delta is not None or blend_steps or gripper_latch_config is not None
    )
    persistent_pipeline = make_action_postprocessor(
        current_action_state(dataset[0], action_dim)
        if postprocessing_requested
        else torch.zeros(action_dim),
        joint_indices,
        max_delta,
        max_second_delta,
        blend_steps,
        gripper_latch_config,
    )

    for frame in range(frame_count):
        gt[frame] = read_raw_gt_chunk(dataset, frame, 1)[0].numpy()

    for frame in range(0, frame_count, inference_stride):
        sample = dict(dataset[frame])
        visual_keys = visual_keys_for_policy(policy, sample)
        for key in visual_keys:
            if key not in sample:
                continue
            sample[key] = augment_visual(
                sample[key],
                force_depth=is_depth_key(key) or key in depth_keys,
                **augment_kwargs,
            )
            if is_depth_key(key) or key in depth_keys:
                sample[key] = normalize_depth_tensor(sample[key])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        chunk = predict_actions(
            policy, preprocessor, postprocessor, sample, task, "chunk", policy_type
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_ms.append((time.perf_counter() - started) * 1000.0)
        inference_frames.append(frame)

        if chunk.shape[-1] != action_dim:
            raise ValueError(
                f"Predicted action dimension {chunk.shape[-1]} does not match GT {action_dim}."
            )
        raw_chunk = chunk.detach().cpu().float()
        usable = min(raw_chunk.shape[0], frame_count - frame)
        if aggregation == "Execute prefix":
            usable = min(usable, inference_stride)
            pipeline = persistent_pipeline
        else:
            pipeline = make_action_postprocessor(
                current_action_state(sample, action_dim)
                if postprocessing_requested
                else torch.zeros(action_dim),
                joint_indices,
                max_delta,
                max_second_delta,
                blend_steps,
                gripper_latch_config,
            )
        processed_chunk = pipeline.process_chunk(raw_chunk[:usable])
        pred_sum[frame : frame + usable] += processed_chunk.numpy()
        raw_pred_sum[frame : frame + usable] += raw_chunk[:usable].numpy()
        pred_count[frame : frame + usable] += 1

    pred = np.full((frame_count, action_dim), np.nan, dtype=np.float32)
    raw_pred = np.full((frame_count, action_dim), np.nan, dtype=np.float32)
    covered = pred_count > 0
    pred[covered] = (pred_sum[covered] / pred_count[covered, None]).astype(np.float32)
    raw_pred[covered] = (raw_pred_sum[covered] / pred_count[covered, None]).astype(np.float32)
    return {
        "gt": gt,
        "pred": pred,
        "raw_pred": raw_pred,
        "postprocessing_enabled": persistent_pipeline.enabled,
        "joint_indices": np.asarray(joint_indices, dtype=np.int64),
        "error": pred - gt,
        "covered": covered,
        "inference_frames": np.asarray(inference_frames),
        "inference_ms": np.asarray(inference_ms),
    }


def render_episode_timeline(result: dict) -> None:
    import altair as alt

    alt.data_transformers.disable_max_rows()
    action_dim = result["gt"].shape[-1]
    dim = st.selectbox("Timeline action dimension", list(range(action_dim)), index=0)
    frames = np.arange(result["gt"].shape[0])
    series = [("GT", result["gt"][:, dim])]
    if result.get("postprocessing_enabled"):
        series.extend(
            [
                ("Raw Pred", result["raw_pred"][:, dim]),
                ("Postprocessed", result["pred"][:, dim]),
            ]
        )
    else:
        series.append(("Pred", result["pred"][:, dim]))
    values = pd.concat(
        [
            pd.DataFrame({"frame": frames, "series": name, "value": values})
            for name, values in series
        ],
        ignore_index=True,
    )
    lines = (
        alt.Chart(values)
        .mark_line()
        .encode(
            x=alt.X("frame:Q", title="Episode frame"),
            y=alt.Y("value:Q", title=f"Action dimension {dim}"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=["GT", "Pred", "Raw Pred", "Postprocessed"],
                    range=["#6b7280", "#16a34a", "#dc2626", "#2563eb"],
                ),
            ),
            tooltip=["frame:Q", "series:N", alt.Tooltip("value:Q", format=".5f")],
        )
    )
    rules = (
        alt.Chart(pd.DataFrame({"frame": result["inference_frames"]}))
        .mark_rule(color="#dc2626", opacity=0.18)
        .encode(x="frame:Q")
    )
    st.altair_chart((lines + rules).properties(height=360), width="stretch")
    st.line_chart(
        pd.DataFrame({"frame": frames, "error": result["error"][:, dim]}).set_index("frame"),
        height=220,
    )

    valid_error = result["error"][result["covered"]]
    latency = result["inference_ms"]
    cols = st.columns(6 if result.get("postprocessing_enabled") else 5)
    cols[0].metric("Covered", f"{result['covered'].mean() * 100:.1f}%")
    metric_offset = 1
    if result.get("postprocessing_enabled"):
        raw_error = (result["raw_pred"] - result["gt"])[result["covered"]]
        cols[1].metric("Raw MAE", f"{np.nanmean(np.abs(raw_error)):.6f}")
        metric_offset = 2
    cols[metric_offset].metric("Timeline MAE", f"{np.nanmean(np.abs(valid_error)):.6f}")
    cols[metric_offset + 1].metric("Timeline RMSE", f"{np.sqrt(np.nanmean(valid_error ** 2)):.6f}")
    cols[metric_offset + 2].metric("Mean inference", f"{latency.mean():.1f} ms")
    cols[metric_offset + 3].metric("P95 inference", f"{np.percentile(latency, 95):.1f} ms")


def main() -> None:
    st.set_page_config(page_title="LeRobot Open-loop", layout="wide")
    st.title("LeRobot Open-loop")

    with st.sidebar:
        dataset_root = st.text_input("Dataset root", DEFAULT_DATASET_ROOT)
        repo_id = st.text_input("Repo ID", "kuavo/task1_sz")
        policy_type = st.selectbox(
            "Policy type", ["act", "diffusion", "lingbot", "lingbot_v2"], index=0
        )
        if policy_type == "lingbot":
            default_policy_path = DEFAULT_LINGBOT_POLICY_PATH
        elif policy_type == "lingbot_v2":
            default_policy_path = DEFAULT_LINGBOT_V2_POLICY_PATH
        else:
            default_policy_path = DEFAULT_POLICY_PATH
        policy_path = st.text_input(
            "Policy path", default_policy_path, key=f"policy_path_{policy_type}"
        )
        task = st.text_area("Task", DEFAULT_TASK)
        device_options = ["cuda"] if policy_type in {"lingbot", "lingbot_v2"} else ["cuda", "cpu"]
        device = st.selectbox("Device", device_options, index=0)
        video_backend = st.selectbox("Video backend", ["pyav", "torchcodec"], index=0)
        if policy_type in {"lingbot", "lingbot_v2"}:
            is_v2 = policy_type == "lingbot_v2"
            lingbot_root = st.text_input(
                "LingBot root", DEFAULT_LINGBOT_V2_ROOT if is_v2 else DEFAULT_LINGBOT_ROOT
            )
            qwen25_path = st.text_input(
                "Qwen processor path", DEFAULT_QWEN3VL_PATH if is_v2 else DEFAULT_QWEN25_PATH
            )
            if is_v2:
                v2_preset = st.selectbox("LingBot V2 preset", ["Task1 right arm", "Task2 bimanual", "Custom"], index=0)
                if v2_preset == "Task2 bimanual":
                    default_robot_name = "kuavo_v2_bimanual"
                    default_norm_stats = DEFAULT_LINGBOT_V2_TASK2_NORM_STATS
                else:
                    default_robot_name = DEFAULT_LINGBOT_V2_ROBOT_NAME
                    default_norm_stats = DEFAULT_LINGBOT_V2_NORM_STATS
                robot_name = st.text_input("LingBot V2 robot name", default_robot_name)
                norm_stats_file = st.text_input("Norm stats file", default_norm_stats)
                use_compile = st.checkbox("Use torch.compile", value=False)
            else:
                robot_name = ""
                norm_stats_file = st.text_input("Norm stats file", DEFAULT_NORM_STATS)
                use_compile = False
        else:
            lingbot_root = ""
            qwen25_path = ""
            norm_stats_file = ""
            robot_name = ""
            use_compile = False
        episode = st.number_input("Episode", min_value=0, value=0, step=1)
        predict_mode = st.selectbox("Predict mode", ["chunk", "single"], index=0)
        if policy_type == "diffusion" and predict_mode == "chunk":
            st.caption("Diffusion chunk mode is shown through select_action to populate observation queues safely.")
        st.divider()
        brightness = st.slider("Brightness", 0.4, 1.8, 1.0, 0.05)
        contrast = st.slider("Contrast", 0.4, 1.8, 1.0, 0.05)
        color = st.slider("Color", 0.0, 1.8, 1.0, 0.05)
        crop_mode = st.radio("Crop mode", ["Center", "Manual"], horizontal=True)
        center_crop = st.slider("Center crop ratio", 0.5, 1.0, 1.0, 0.01, disabled=crop_mode != "Center")
        left_crop = st.slider("Manual left", 0.0, 0.95, 0.0, 0.01, disabled=crop_mode != "Manual")
        top_crop = st.slider("Manual top", 0.0, 0.95, 0.0, 0.01, disabled=crop_mode != "Manual")
        right_crop = st.slider("Manual right", 0.05, 1.0, 1.0, 0.01, disabled=crop_mode != "Manual")
        bottom_crop = st.slider("Manual bottom", 0.05, 1.0, 1.0, 0.01, disabled=crop_mode != "Manual")
        if st.button("Load / run inference", type="primary"):
            st.session_state["open_loop_enabled"] = True

    if not st.session_state.get("open_loop_enabled", False):
        st.info("Configure the policy and dataset in the sidebar, then load the model.")
        st.stop()

    policy, preprocessor, postprocessor = load_model(
        policy_path,
        policy_type,
        device,
        lingbot_root,
        qwen25_path,
        norm_stats_file,
        robot_name,
        use_compile,
        task,
    )
    dataset = load_dataset(
        repo_id,
        dataset_root,
        policy_path,
        policy_type,
        device,
        (int(episode),),
        lingbot_root,
        qwen25_path,
        norm_stats_file,
        robot_name,
        use_compile,
        task,
        video_backend,
    )
    max_frame = max(0, len(dataset) - 1)
    action_dim = int(read_raw_gt_chunk(dataset, 0, 1).shape[-1])
    if action_dim == 16:
        default_gripper_indices = "7,15"
        default_joint_indices = ",".join(str(index) for index in range(16) if index not in (7, 15))
    elif action_dim == 8:
        default_gripper_indices = "7"
        default_joint_indices = ",".join(str(index) for index in range(7))
    else:
        default_gripper_indices = ""
        default_joint_indices = ",".join(str(index) for index in range(action_dim))
    with st.sidebar:
        frame = st.slider("Frame in selected episode", min_value=0, max_value=max_frame, value=0, step=1)
        max_compare_h = int(getattr(policy.config, "chunk_size", getattr(policy.config, "horizon", 1)))
        horizon = st.slider("Compare horizon", min_value=1, max_value=max(1, max_compare_h), value=max(1, max_compare_h))
        st.divider()
        timeline_frames = st.number_input(
            "Timeline frames",
            min_value=1,
            max_value=max(1, len(dataset)),
            value=min(200, max(1, len(dataset))),
        )
        inference_stride = st.number_input(
            "Chunk inference stride", min_value=1, max_value=max(1, max_compare_h), value=1
        )
        timeline_aggregation = st.selectbox(
            "Chunk timeline mode", ["Execute prefix", "Overlap mean"], index=0
        )
        st.divider()
        st.subheader("Causal action postprocessing")
        joint_indices_text = st.text_input("Arm/joint action dimensions", default_joint_indices)
        enable_rate_limit = st.checkbox("Dataset-derived joint rate limit", value=False)
        rate_percentile = st.slider(
            "Dataset |Δaction| percentile",
            95.0,
            99.9,
            99.5,
            0.1,
            disabled=not enable_rate_limit,
        )
        enable_acceleration_limit = st.checkbox(
            "Also limit second difference",
            value=False,
            disabled=not enable_rate_limit,
        )
        enable_boundary_blend = st.checkbox("Blend new chunk prefix", value=False)
        boundary_blend_steps = st.number_input(
            "Chunk prefix blend steps",
            min_value=1,
            max_value=max(1, max_compare_h),
            value=min(4, max(1, max_compare_h)),
            disabled=not enable_boundary_blend,
        )
        enable_gripper_latch = st.checkbox("Latch gripper intent", value=False)
        gripper_indices_text = st.text_input(
            "Gripper action dimensions",
            default_gripper_indices,
            disabled=not enable_gripper_latch,
        )
        intent_steps = st.number_input(
            "Consecutive intent steps",
            min_value=1,
            value=5,
            disabled=not enable_gripper_latch,
        )
        min_close_steps = st.number_input(
            "Minimum closed hold steps",
            min_value=0,
            value=10,
            disabled=not enable_gripper_latch,
        )
        min_open_steps = st.number_input(
            "Minimum open hold steps",
            min_value=0,
            value=4,
            disabled=not enable_gripper_latch,
        )

    joint_indices: tuple[int, ...] = ()
    if enable_rate_limit or enable_boundary_blend:
        try:
            joint_indices = parse_action_indices(joint_indices_text)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
    max_delta = max_second_delta = None
    if enable_rate_limit:
        try:
            max_delta, dataset_second_delta = compute_dataset_action_limits(
                dataset, joint_indices, float(rate_percentile)
            )
            if enable_acceleration_limit:
                max_second_delta = dataset_second_delta
        except (ValueError, IndexError) as exc:
            st.error(str(exc))
            st.stop()
    gripper_latch_config = None
    if enable_gripper_latch:
        try:
            gripper_latch_config = GripperLatchConfig(
                action_indices=parse_action_indices(gripper_indices_text),
                intent_steps=int(intent_steps),
                min_close_steps=int(min_close_steps),
                min_open_steps=int(min_open_steps),
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    idx = min(int(frame), len(dataset) - 1)
    sample = dataset[idx]
    missing_keys = missing_required_keys(policy, sample)
    if missing_keys:
        st.error(
            "The selected policy expects observation keys that are not present in this dataset sample. "
            "Use the matching LeRobot dataset/checkpoint pair, or pick a policy trained without these modalities."
        )
        st.code("\n".join(missing_keys))
        st.stop()

    depth_keys = get_depth_keys(policy)
    image_keys = visual_keys_for_policy(policy, sample)
    preview_sample = dict(sample)
    manual_box = (left_crop, top_crop, right_crop, bottom_crop)
    for key in image_keys:
        preview_sample[key] = augment_visual(
            sample[key],
            brightness,
            contrast,
            color,
            crop_mode,
            center_crop,
            manual_box,
            force_depth=is_depth_key(key) or key in depth_keys,
        )
        if is_depth_key(key) or key in depth_keys:
            preview_sample[key] = normalize_depth_tensor(preview_sample[key])

    raw_pred = predict_actions(
        policy, preprocessor, postprocessor, preview_sample, task, predict_mode, policy_type
    )
    postprocessing_requested = bool(
        max_delta is not None
        or enable_boundary_blend
        or gripper_latch_config is not None
    )
    try:
        action_postprocessor = make_action_postprocessor(
            current_action_state(sample, raw_pred.shape[-1])
            if postprocessing_requested
            else torch.zeros(raw_pred.shape[-1]),
            joint_indices,
            max_delta,
            max_second_delta,
            int(boundary_blend_steps) if enable_boundary_blend else 0,
            gripper_latch_config,
        )
        pred = action_postprocessor.process_chunk(raw_pred)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    postprocessing_enabled = action_postprocessor.enabled
    gt = read_raw_gt_chunk(dataset, idx, pred.shape[0])
    if pred.shape[-1] != gt.shape[-1]:
        st.error("Predicted action dimension does not match dataset action dimension.")
        st.write({"pred_shape": list(pred.shape), "gt_shape": list(gt.shape)})
        st.stop()
    compare_h = min(int(horizon), pred.shape[0], gt.shape[0])
    err = pred[:compare_h] - gt[:compare_h]
    horizon_mae = err.abs().mean(dim=1).numpy()
    dim_mae = err.abs().mean(dim=0).numpy()

    progress = 0.0 if max_frame == 0 else idx / max_frame
    st.progress(progress, text=f"Episode {scalar_int(sample['episode_index'])}, frame {idx}/{max_frame}")

    meta_cols = st.columns(6)
    meta_cols[0].metric("Episode", scalar_int(sample["episode_index"]))
    meta_cols[1].metric("Frame", scalar_int(sample["frame_index"]))
    meta_cols[2].metric("Policy", policy_type)
    meta_cols[3].metric("Chunk", getattr(policy.config, "chunk_size", getattr(policy.config, "horizon", pred.shape[0])))
    meta_cols[4].metric("n_action_steps", getattr(policy.config, "n_action_steps", pred.shape[0]))
    meta_cols[5].metric("First MAE", f"{horizon_mae[0]:.6f}")

    if image_keys:
        cols = st.columns(len(image_keys))
        for col, key in zip(cols, image_keys, strict=False):
            col.image(tensor_to_pil(preview_sample[key]), caption=key, width="stretch")

    first = pd.DataFrame(
        {
            "dim": list(range(pred.shape[-1])),
            **({"raw_pred": raw_pred[0].numpy()} if postprocessing_enabled else {}),
            "postprocessed" if postprocessing_enabled else "pred": pred[0].numpy(),
            "gt": gt[0].numpy(),
            "err": (pred[0] - gt[0]).numpy(),
            "abs_err": (pred[0] - gt[0]).abs().numpy(),
        }
    )
    st.subheader("First Action")
    st.dataframe(first, width="stretch", hide_index=True)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.subheader("Horizon MAE")
        st.line_chart(pd.DataFrame({"mae": horizon_mae}))
    with chart_cols[1]:
        st.subheader("Dimension MAE")
        st.bar_chart(pd.DataFrame({"mae": dim_mae}))

    st.subheader("Pred vs GT by Action Dimension")
    dim = st.selectbox("Action dimension", list(range(pred.shape[-1])), index=0)
    dim_df = pd.DataFrame(
        {
            "horizon": np.arange(compare_h),
            "pred": pred[:compare_h, dim].numpy(),
            "gt": gt[:compare_h, dim].numpy(),
            "error": (pred[:compare_h, dim] - gt[:compare_h, dim]).numpy(),
        }
    ).set_index("horizon")
    line_cols = st.columns(2)
    line_cols[0].line_chart(dim_df[["pred", "gt"]])
    line_cols[1].line_chart(dim_df[["error"]])

    st.subheader("Episode Chunk Timeline")
    st.caption(
        "GT and Pred are action values. Red vertical rules mark model inference frames; "
        "the lower chart is signed prediction error."
    )
    timeline_key = (
        dataset_root,
        int(episode),
        policy_path,
        policy_type,
        robot_name,
        norm_stats_file,
        bool(use_compile),
        int(timeline_frames),
        int(inference_stride),
        timeline_aggregation,
        tuple(joint_indices),
        None if max_delta is None else tuple(np.round(max_delta, 8)),
        None if max_second_delta is None else tuple(np.round(max_second_delta, 8)),
        int(boundary_blend_steps) if enable_boundary_blend else 0,
        repr(gripper_latch_config),
    )
    if st.button("Run episode timeline", type="primary"):
        with st.spinner("Running chunk inference over the selected episode..."):
            st.session_state["episode_timeline"] = {
                "key": timeline_key,
                "result": build_episode_timeline(
                    dataset=dataset,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    task=task,
                    policy_type=policy_type,
                    frame_limit=int(timeline_frames),
                    inference_stride=int(inference_stride),
                    aggregation=timeline_aggregation,
                    augment_kwargs={
                        "brightness": brightness,
                        "contrast": contrast,
                        "color": color,
                        "crop_mode": crop_mode,
                        "center_ratio": center_crop,
                        "manual_box": manual_box,
                    },
                    joint_indices=joint_indices,
                    max_delta=max_delta,
                    max_second_delta=max_second_delta,
                    blend_steps=int(boundary_blend_steps) if enable_boundary_blend else 0,
                    gripper_latch_config=gripper_latch_config,
                ),
            }
    cached_timeline = st.session_state.get("episode_timeline")
    if cached_timeline and cached_timeline.get("key") == timeline_key:
        render_episode_timeline(cached_timeline["result"])

    with st.expander("State / Chunk Table", expanded=False):
        state = sample.get("observation.state")
        if isinstance(state, torch.Tensor):
            st.write({"observation.state": state.detach().cpu().float().tolist()})
        chunk_df = pd.DataFrame(
            {
                "horizon": np.arange(compare_h),
                **{f"pred_{i}": pred[:compare_h, i].numpy() for i in range(pred.shape[-1])},
                **{f"gt_{i}": gt[:compare_h, i].numpy() for i in range(gt.shape[-1])},
            }
        )
        st.dataframe(chunk_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
