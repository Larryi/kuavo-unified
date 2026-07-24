#!/usr/bin/env python3
"""Export audited Beijing ROS bags as schema-compatible LeRobot v3 datasets.

Alignment rules:
* head RGB defines the 30 Hz master clock; drop 10 frames at each end, stride 3
* wrist RGB uses nearest Header timestamp
* continuous robot/hand State uses linear Header-time interpolation
* arm/hand Action uses causal Header-time zero-order hold

The final feature names, dimensions, RGB modalities, FPS, resolution, and task
description are copied from the previously repaired TASK1/2/3 datasets.
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rosbag
from lerobot.datasets import lerobot_dataset as lerobot_dataset_module
from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass(frozen=True)
class TaskConfig:
    description: str
    reference_root: Path
    cameras: dict[str, str]
    arm_slice: tuple[int, int]
    action_slice: tuple[int, int]
    hand_state_topic: str
    hand_action_topic: str
    state_hand_index: int
    action_hand_index: int
    hand_scale: float


TASKS = {
    "task1": TaskConfig(
        description="Pick and Place the safety belt, cable and pin connector",
        reference_root=Path(
            "/mnt/pqssd/Real_PQ_3.0/TASK1_SZ_Repaired/lerobot_task1_345"
        ),
        cameras={
            "observation.images.head_cam_h": "/cam_h/color/image_raw/compressed",
            "observation.images.wrist_cam_r": "/cam_r/color/image_raw/compressed",
        },
        arm_slice=(19, 26),
        action_slice=(7, 14),
        hand_state_topic="/leju_claw_state",
        hand_action_topic="/leju_claw_command",
        state_hand_index=1,
        action_hand_index=1,
        hand_scale=100.0,
    ),
    "task2": TaskConfig(
        description="Sorting sleeves by weight",
        reference_root=Path(
            "/mnt/pqssd/Real_PQ_3.0/TASK2_SZ_Repaired/lerobot_task2_264"
        ),
        cameras={
            "observation.images.head_cam_h": "/cam_h/color/image_raw/compressed",
            "observation.images.wrist_cam_l": "/cam_l/color/image_raw/compressed",
            "observation.images.wrist_cam_r": "/cam_r/color/image_raw/compressed",
        },
        arm_slice=(12, 26),
        action_slice=(0, 14),
        hand_state_topic="/leju_claw_state",
        hand_action_topic="/leju_claw_command",
        state_hand_index=-1,
        action_hand_index=-1,
        hand_scale=100.0,
    ),
    "task3": TaskConfig(
        description="Grab the car body panel and place it in the area",
        reference_root=Path(
            "/mnt/pqssd/Real_PQ_3.0/TASK3_SZ_Repaired/lerobot_task3_165"
        ),
        cameras={
            "observation.images.head_cam_h": "/cam_h/color/image_raw/compressed",
            "observation.images.wrist_cam_r": "/cam_r/color/image_raw/compressed",
        },
        arm_slice=(19, 26),
        action_slice=(7, 14),
        hand_state_topic="/dexhand/state",
        hand_action_topic="/control_robot_hand_position",
        state_hand_index=6,
        action_hand_index=0,
        hand_scale=100.0,
    ),
}

ARM_STATE_TOPIC = "/sensors_data_raw"
ARM_ACTION_TOPIC = "/kuavo_arm_traj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--whitelist", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--sample-drop", type=int, default=10)
    parser.add_argument("--camera-stride", type=int, default=3)
    parser.add_argument(
        "--trim-idle", action=argparse.BooleanOptionalAction, default=True,
        help="Trim only static command prefixes/suffixes; retain all internal pauses.",
    )
    parser.add_argument("--trim-pre-frames", type=int, default=10)
    parser.add_argument("--trim-post-frames", type=int, default=15)
    parser.add_argument("--trim-joint-delta", type=float, default=0.001)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--video-crf", type=int, default=23)
    parser.add_argument("--video-preset", default="veryfast")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate alignment/schema and print episode trims without writing LeRobot.",
    )
    return parser.parse_args()


def header_time(msg: Any) -> float:
    value = float(msg.header.stamp.to_sec())
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"invalid Header timestamp: {value}")
    return value


def deduplicate(
    times: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    keep = np.r_[np.diff(times) > 0, True]
    return times[keep], values[keep]


def interpolate(
    times: np.ndarray, values: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    times, values = deduplicate(times, values)
    if targets[0] < times[0]:
        raise ValueError("State has no sample before the first retained image")
    return np.column_stack(
        [np.interp(targets, times, values[:, dim]) for dim in range(values.shape[1])]
    )


def causal_hold(
    times: np.ndarray, values: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    times, values = deduplicate(times, values)
    indices = np.searchsorted(times, targets, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("Action has no causal sample for a retained image")
    return values[indices]


def nearest_indices(times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    order = np.argsort(times, kind="stable")
    times = times[order]
    right = np.searchsorted(times, targets, side="left")
    right = np.clip(right, 0, times.size - 1)
    left = np.clip(right - 1, 0, times.size - 1)
    use_right = np.abs(times[right] - targets) < np.abs(times[left] - targets)
    return order[np.where(use_right, right, left)]


def command_trim_range(
    action: np.ndarray,
    joint_dimensions: list[int],
    joint_delta: float,
    pre_frames: int,
    post_frames: int,
) -> tuple[int, int]:
    if action.shape[0] < 2:
        return 0, action.shape[0]
    delta = np.abs(np.diff(action, axis=0))
    joint_active = np.max(delta[:, joint_dimensions], axis=1) > joint_delta
    hand_dimensions = [
        index for index in range(action.shape[1]) if index not in joint_dimensions
    ]
    hand_active = np.max(delta[:, hand_dimensions], axis=1) > 0.1
    active = joint_active | hand_active
    if active.size >= 3:
        active[1:-1] |= active[:-2] & active[2:]
    indices = np.flatnonzero(active)
    if not indices.size:
        return 0, action.shape[0]
    start = max(0, int(indices[0]) + 1 - max(0, pre_frames))
    end = min(action.shape[0], int(indices[-1]) + 2 + max(0, post_frames))
    return start, end


def decode_rgb(data: bytes, expected_shape: tuple[int, int]) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV failed to decode compressed RGB image")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    expected_h, expected_w = expected_shape
    if image.shape[:2] != (expected_h, expected_w):
        image = cv2.resize(image, (expected_w, expected_h), interpolation=cv2.INTER_AREA)
    return image


def read_episode(
    bag_path: Path,
    config: TaskConfig,
    sample_drop: int,
    stride: int,
    trim_idle: bool,
    trim_pre: int,
    trim_post: int,
    trim_joint_delta: float,
) -> dict[str, Any]:
    camera_times = {key: [] for key in config.cameras}
    camera_data = {key: [] for key in config.cameras}
    lowdim_times = {
        "arm_state": [], "arm_action": [], "hand_state": [], "hand_action": [],
    }
    lowdim_values = {key: [] for key in lowdim_times}
    camera_by_topic = {topic: key for key, topic in config.cameras.items()}
    lowdim_by_topic = {
        ARM_STATE_TOPIC: "arm_state",
        ARM_ACTION_TOPIC: "arm_action",
        config.hand_state_topic: "hand_state",
        config.hand_action_topic: "hand_action",
    }
    topics = [*camera_by_topic, *lowdim_by_topic]
    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, msg, _ in bag.read_messages(topics=topics):
            if topic in camera_by_topic:
                key = camera_by_topic[topic]
                camera_times[key].append(header_time(msg))
                camera_data[key].append(bytes(msg.data))
                continue
            key = lowdim_by_topic[topic]
            lowdim_times[key].append(header_time(msg))
            if key == "arm_state":
                values = np.asarray(msg.joint_data.joint_q, dtype=np.float64)
            elif key == "arm_action":
                values = np.deg2rad(np.asarray(msg.position, dtype=np.float64))
            elif config.hand_state_topic == "/dexhand/state" and key == "hand_state":
                values = np.asarray(msg.position, dtype=np.float64)
            elif (
                config.hand_action_topic == "/control_robot_hand_position"
                and key == "hand_action"
            ):
                values = np.asarray(list(msg.right_hand_position), dtype=np.float64)
            else:
                values = np.asarray(msg.data.position, dtype=np.float64)
            lowdim_values[key].append(values)

    missing = [
        key for key, values in {**camera_times, **lowdim_times}.items() if not values
    ]
    if missing:
        raise ValueError(f"missing topics: {missing}")
    head_key = "observation.images.head_cam_h"
    head_times = np.asarray(camera_times[head_key], dtype=np.float64)
    retained_head_indices = np.arange(head_times.size)[sample_drop:-sample_drop:stride]
    targets = head_times[retained_head_indices]
    if targets.size < 2:
        raise ValueError("not enough retained head-camera frames")

    arrays_t = {
        key: np.asarray(values, dtype=np.float64)
        for key, values in lowdim_times.items()
    }
    arrays_v = {
        key: np.stack(values).astype(np.float64)
        for key, values in lowdim_values.items()
    }
    arm_state = interpolate(
        arrays_t["arm_state"], arrays_v["arm_state"], targets
    )[:, config.arm_slice[0]:config.arm_slice[1]]
    arm_action = causal_hold(
        arrays_t["arm_action"], arrays_v["arm_action"], targets
    )[:, config.action_slice[0]:config.action_slice[1]]
    hand_state_all = interpolate(
        arrays_t["hand_state"], arrays_v["hand_state"], targets
    )
    hand_action_all = causal_hold(
        arrays_t["hand_action"], arrays_v["hand_action"], targets
    )

    if config.state_hand_index < 0:
        hand_state = hand_state_all[:, :2] / config.hand_scale
        hand_action = hand_action_all[:, :2] / config.hand_scale
        state = np.column_stack([
            arm_state[:, :7], hand_state[:, 0],
            arm_state[:, 7:14], hand_state[:, 1],
        ])
        action = np.column_stack([
            arm_action[:, :7], hand_action[:, 0],
            arm_action[:, 7:14], hand_action[:, 1],
        ])
        joint_dimensions = list(range(7)) + list(range(8, 15))
    else:
        hand_state = (
            hand_state_all[:, config.state_hand_index:config.state_hand_index + 1]
            / config.hand_scale
        )
        hand_action = (
            hand_action_all[:, config.action_hand_index:config.action_hand_index + 1]
            / config.hand_scale
        )
        state = np.column_stack([arm_state, hand_state])
        action = np.column_stack([arm_action, hand_action])
        joint_dimensions = list(range(7))
    state = state.astype(np.float32)
    action = action.astype(np.float32)
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("non-finite reconstructed State/Action")
    if np.nanmin(state[:, -1]) < -0.05 or np.nanmax(state[:, -1]) > 1.05:
        raise ValueError(
            f"hand State scale mismatch: {state[:, -1].min()}..{state[:, -1].max()}"
        )
    if trim_idle:
        trim_start, trim_end = command_trim_range(
            action, joint_dimensions, trim_joint_delta, trim_pre, trim_post
        )
    else:
        trim_start, trim_end = 0, targets.size
    if trim_end - trim_start < 2:
        raise ValueError(f"trim produced only {trim_end - trim_start} frames")

    selected_images: dict[str, list[bytes]] = {}
    for key in config.cameras:
        if key == head_key:
            indices = retained_head_indices
        else:
            indices = nearest_indices(
                np.asarray(camera_times[key], dtype=np.float64), targets
            )
        selected_images[key] = [
            camera_data[key][int(index)] for index in indices[trim_start:trim_end]
        ]
    return {
        "state": state[trim_start:trim_end],
        "action": action[trim_start:trim_end],
        "target_times": targets[trim_start:trim_end],
        "images": selected_images,
        "untrimmed_frames": int(targets.size),
        "trim_start": int(trim_start),
        "trim_end": int(trim_end),
    }


def encode_h264_frames(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    *,
    crf: int = 23,
    preset: str = "veryfast",
    **_: Any,
) -> None:
    imgs_dir, video_path = Path(imgs_dir), Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "/usr/bin/ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(int(fps)),
            "-i", str(imgs_dir / "frame-%06d.png"),
            "-c:v", "libx264", "-preset", preset,
            "-crf", str(int(crf)), "-g", "10", "-pix_fmt", "yuv420p",
            str(video_path),
        ],
        check=True,
    )


def reference_features(config: TaskConfig) -> tuple[dict[str, Any], tuple[int, int]]:
    info = json.loads(
        (config.reference_root / "meta/info.json").read_text(encoding="utf-8")
    )
    selected = {"observation.state", "action", *config.cameras}
    features = {
        key: value for key, value in info["features"].items() if key in selected
    }
    # LeRobot's runtime validator expects tuple shapes even though info.json
    # serializes them as JSON lists.
    for feature in features.values():
        feature["shape"] = tuple(feature["shape"])
    missing = selected - features.keys()
    if missing:
        raise ValueError(f"reference dataset lacks expected features: {sorted(missing)}")
    head_shape = features["observation.images.head_cam_h"]["shape"]
    return features, (int(head_shape[1]), int(head_shape[2]))


def validate_schema(output_root: Path, config: TaskConfig) -> None:
    reference = json.loads(
        (config.reference_root / "meta/info.json").read_text(encoding="utf-8")
    )
    output = json.loads((output_root / "meta/info.json").read_text(encoding="utf-8"))
    keys = {"observation.state", "action", *config.cameras}
    errors = []
    for key in keys:
        for field in ("dtype", "shape", "names"):
            if output["features"][key].get(field) != reference["features"][key].get(field):
                errors.append(f"{key}.{field}")
    if output["fps"] != reference["fps"]:
        errors.append("fps")
    if errors:
        raise ValueError(f"output schema differs from reference: {errors}")


def main() -> None:
    args = parse_args()
    config = TASKS[args.task]
    paths = [
        Path(line.strip()) for line in args.whitelist.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.max_episodes:
        paths = paths[:args.max_episodes]
    if not paths:
        raise ValueError("whitelist is empty")
    features, image_shape = reference_features(config)
    expected_dimension = 16 if args.task == "task2" else 8
    for key in ("observation.state", "action"):
        if tuple(features[key]["shape"]) != (expected_dimension,):
            raise ValueError(f"reference {key} shape is {features[key]['shape']}")
    print(
        f"task={args.task} episodes={len(paths)} dimension={expected_dimension} "
        f"cameras={list(config.cameras)} task_description={config.description!r}",
        flush=True,
    )

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and not args.dry_run:
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    destination = None
    if not args.dry_run:
        lerobot_dataset_module.encode_video_frames = partial(
            encode_h264_frames, crf=args.video_crf, preset=args.video_preset
        )
        kwargs = {
            "repo_id": args.repo_id,
            "root": output_root,
            "fps": 10,
            "robot_type": "kuavo4pro",
            "features": features,
            "use_videos": True,
            "image_writer_processes": 0,
            "image_writer_threads": 0,
            "video_backend": "pyav",
            "batch_encoding_size": 1,
        }
        supported = inspect.signature(LeRobotDataset.create).parameters
        destination = LeRobotDataset.create(
            **{key: value for key, value in kwargs.items() if key in supported}
        )

    manifest = []
    for output_episode, bag_path in enumerate(paths):
        episode = read_episode(
            bag_path, config, args.sample_drop, args.camera_stride,
            args.trim_idle, args.trim_pre_frames, args.trim_post_frames,
            args.trim_joint_delta,
        )
        frame_count = int(episode["state"].shape[0])
        if destination is not None:
            for frame_index in range(frame_count):
                frame = {
                    "observation.state": episode["state"][frame_index],
                    "action": episode["action"][frame_index],
                    "task": config.description,
                }
                for key in config.cameras:
                    frame[key] = decode_rgb(
                        episode["images"][key][frame_index], image_shape
                    )
                destination.add_frame(frame)
            destination.save_episode(parallel_encoding=True)
        manifest.append({
            "output_episode": output_episode,
            "bag_path": str(bag_path),
            "untrimmed_frames": episode["untrimmed_frames"],
            "trim_start": episode["trim_start"],
            "trim_end": episode["trim_end"],
            "frames": frame_count,
            "header_start": float(episode["target_times"][0]),
            "header_end": float(episode["target_times"][-1]),
        })
        if (
            (output_episode + 1) % max(1, args.progress_every) == 0
            or output_episode + 1 == len(paths)
        ):
            print(
                f"[{output_episode + 1}/{len(paths)}] {bag_path.name}: "
                f"{episode['untrimmed_frames']} -> {frame_count} frames",
                flush=True,
            )
    if args.dry_run:
        print("dry-run passed", flush=True)
        return
    if hasattr(destination, "finalize"):
        destination.finalize()
    elif hasattr(destination, "consolidate"):
        destination.consolidate()
    info_path = output_root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for key in config.cameras:
        info["features"][key].setdefault("info", {})["video.codec"] = "h264"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=4), encoding="utf-8")
    validate_schema(output_root, config)
    (output_root / "header_export_manifest.json").write_text(
        json.dumps({
            "source_whitelist": str(args.whitelist.resolve()),
            "task": args.task,
            "task_description": config.description,
            "alignment": {
                "clock": "ROS message Header",
                "state": "linear interpolation",
                "action": "causal zero-order hold",
                "wrist_rgb": "nearest Header timestamp",
                "head_sampling": f"drop {args.sample_drop}, stride {args.camera_stride}",
            },
            "trim_idle": args.trim_idle,
            "schema_reference": str(config.reference_root),
            "episodes": manifest,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"completed: {output_root}", flush=True)


if __name__ == "__main__":
    main()
