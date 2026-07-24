#!/usr/bin/env python3
"""Build an RGB-only TASK2 LeRobot dataset with Header-aligned low-dimensional data.

The source LeRobot dataset supplies the already-trimmed RGB frames.  ROS bags are
read only for the head-camera timeline and four low-dimensional streams.  State
and action are reconstructed causally using message Header timestamps, avoiding
the rosbag recorder backlog that contaminated the original conversion.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import rosbag
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets import lerobot_dataset as lerobot_dataset_module
from lerobot.datasets.video_utils import encode_video_frames


CAMERA_TOPIC = "/cam_h/color/image_raw/compressed"
TOPICS = {
    "joint_state": "/sensors_data_raw",
    "arm_action": "/kuavo_arm_traj",
    "claw_state": "/leju_claw_state",
    "claw_action": "/leju_claw_command",
}
RGB_PREFIX = "observation.images."
AUTO_KEYS = {
    "index", "episode_index", "frame_index", "timestamp", "next.done", "task_index"
}
REBUILD_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("/mnt/pqssd/Real_PQ_3.0/TASK2_SZ_Depth/lerobot"),
    )
    parser.add_argument("--source-repo-id", default="task2_sz_depth_raw1000")
    parser.add_argument(
        "--bag-dir", type=Path, default=Path("/mnt/pqssd/task2_chengzhong")
    )
    parser.add_argument(
        "--whitelist", type=Path,
        default=Path(
            "outputs/audit/task2_lowdim_quality_all/"
            "recommended_repairable_single_episodes.txt"
        ),
    )
    parser.add_argument(
        "--audit-csv", type=Path,
        default=Path("outputs/audit/task2_lowdim_quality_all/all_lowdim_alignment.csv"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path(
            "/mnt/pqssd/Real_PQ_3.0/TASK2_SZ_Repaired/lerobot_rgb_264"
        ),
    )
    parser.add_argument("--output-repo-id", default="task2_sz_repaired_rgb_264")
    parser.add_argument(
        "--arm-mode", choices=("both", "right"), default="both",
        help="both=TASK2 16D; right=TASK1 right-arm-only 8D.",
    )
    parser.add_argument("--sample-drop", type=int, default=10)
    parser.add_argument("--camera-stride", type=int, default=3)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--image-writer-processes", type=int, default=0,
        help="Keep 0: LeRobot 0.4.3 can deadlock while draining its image-writer process pool.",
    )
    parser.add_argument(
        "--video-codec", choices=("h264", "libsvtav1"), default="h264",
        help="H.264 is dramatically faster for this one-time filtered export.",
    )
    parser.add_argument("--video-crf", type=int, default=23)
    parser.add_argument("--video-preset", default="veryfast")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/task2_header_rebuild"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument(
        "--trim-idle", action="store_true",
        help="Trim only leading/trailing static commands; internal pauses are retained.",
    )
    parser.add_argument("--trim-pre-frames", type=int, default=10)
    parser.add_argument("--trim-post-frames", type=int, default=15)
    parser.add_argument(
        "--trim-joint-delta", type=float, default=0.001,
        help="Per-frame joint command delta threshold in radians.",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Optional prefix limit for smoke tests.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Extract and validate repaired low-dimensional arrays without encoding video.",
    )
    return parser.parse_args()


def header_time(msg: Any) -> float:
    return float(msg.header.stamp.to_sec())


def causal_indices(source_times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    order = np.argsort(source_times, kind="stable")
    sorted_times = source_times[order]
    indices = np.searchsorted(sorted_times, targets, side="right") - 1
    if np.any(indices < 0):
        raise ValueError(f"{int(np.count_nonzero(indices < 0))} targets lack causal history")
    return order[indices]


def read_ids(path: Path) -> list[int]:
    return sorted({int(line.strip()) for line in path.read_text().splitlines() if line.strip()})


def read_audit_bag_names(path: Path) -> dict[int, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["raw_episode"]): row["bag_name"] for row in csv.DictReader(handle)}


def source_episode_layout(source_root: Path) -> tuple[dict[int, tuple[int, int]], int]:
    files = sorted((source_root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet data under {source_root / 'data'}")
    layout: dict[int, tuple[int, int]] = {}
    global_offset = 0
    for path in files:
        table = pq.read_table(path, columns=["episode_index"])
        episodes = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        for episode in np.unique(episodes):
            local = np.flatnonzero(episodes == episode)
            if int(episode) in layout:
                raise ValueError(f"Episode {episode} spans multiple parquet files; unsupported")
            layout[int(episode)] = (global_offset + int(local[0]), len(local))
        global_offset += len(episodes)
    return layout, global_offset


def command_trim_range(
    action: np.ndarray, joint_dims: list[int], joint_delta: float,
    pre_frames: int, post_frames: int,
) -> tuple[int, int]:
    """Return a conservative [start, end) range, preserving every internal pause."""
    if len(action) < 2:
        return 0, len(action)
    delta = np.abs(np.diff(action, axis=0))
    joint_active = np.max(delta[:, joint_dims], axis=1) > joint_delta
    claw_dims = [index for index in range(action.shape[1]) if index not in joint_dims]
    claw_active = (
        np.max(delta[:, claw_dims], axis=1) > 0.1
        if claw_dims else np.zeros(len(delta), dtype=bool)
    )
    active = joint_active | claw_active
    # A one-frame bridge prevents command quantization from fragmenting activity.
    if len(active) >= 3:
        active[1:-1] |= active[:-2] & active[2:]
    indices = np.flatnonzero(active)
    if not len(indices):
        return 0, len(action)
    start = max(0, int(indices[0]) + 1 - max(0, pre_frames))
    end = min(len(action), int(indices[-1]) + 2 + max(0, post_frames))
    return start, end


def extract_one(
    job: tuple[int, str, int, int, int, str, bool, str, bool, int, int, float]
) -> dict[str, Any]:
    (
        episode, bag_text, expected_frames, sample_drop, stride, cache_text, force,
        arm_mode, trim_idle, trim_pre, trim_post, trim_joint_delta,
    ) = job
    bag_path = Path(bag_text)
    cache_path = Path(cache_text)
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as cached:
            if (
                int(cached["version"]) == REBUILD_VERSION
                and int(cached["episode"]) == episode
                and int(cached["bag_size"]) == bag_path.stat().st_size
                and int(cached["expected_frames"]) == expected_frames
                and str(cached["arm_mode"]) == arm_mode
                and bool(cached["trim_idle"]) == trim_idle
                and int(cached["trim_pre"]) == trim_pre
                and int(cached["trim_post"]) == trim_post
                and float(cached["trim_joint_delta"]) == trim_joint_delta
            ):
                return {
                    "episode": episode,
                    "state": cached["state"],
                    "action": cached["action"],
                    "target_times": cached["target_times"],
                    "trim_start": int(cached["trim_start"]),
                    "trim_end": int(cached["trim_end"]),
                    "bag_path": str(bag_path),
                    "cached": True,
                }

    times: dict[str, list[float]] = {"camera": []}
    times.update({key: [] for key in TOPICS})
    values: dict[str, list[np.ndarray]] = {key: [] for key in TOPICS}
    topic_to_key = {topic: key for key, topic in TOPICS.items()}
    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, msg, _ in bag.read_messages(topics=[CAMERA_TOPIC, *TOPICS.values()]):
            if topic == CAMERA_TOPIC:
                times["camera"].append(header_time(msg))
                continue
            key = topic_to_key[topic]
            times[key].append(header_time(msg))
            if key == "joint_state":
                value = np.asarray(msg.joint_data.joint_q, dtype=np.float64)
                if value.size < 26:
                    raise ValueError(f"joint_state has {value.size} values")
            elif key == "arm_action":
                value = np.deg2rad(np.asarray(msg.position, dtype=np.float64))
                if value.size < 14:
                    raise ValueError(f"arm_action has {value.size} values")
            else:
                value = np.asarray(msg.data.position, dtype=np.float64)
                if value.size < 2:
                    raise ValueError(f"{key} has {value.size} values")
            values[key].append(value)

    camera = np.asarray(times["camera"], dtype=np.float64)
    if sample_drop:
        targets = camera[sample_drop:-sample_drop:stride]
    else:
        targets = camera[::stride]
    if len(targets) != expected_frames:
        raise ValueError(
            f"episode {episode}: LeRobot={expected_frames} frames, "
            f"bag master timeline={len(targets)}"
        )

    aligned: dict[str, np.ndarray] = {}
    for key in TOPICS:
        source_times = np.asarray(times[key], dtype=np.float64)
        source_values = np.stack(values[key])
        aligned[key] = source_values[causal_indices(source_times, targets)]

    if arm_mode == "right":
        # TASK1: right 7 arm joints + right claw.
        state = np.concatenate([
            aligned["joint_state"][:, 19:26],
            aligned["claw_state"][:, 1:2] / 100.0,
        ], axis=1).astype(np.float32)
        action = np.concatenate([
            aligned["arm_action"][:, 7:14],
            aligned["claw_action"][:, 1:2] / 100.0,
        ], axis=1).astype(np.float32)
        joint_dims = list(range(7))
    else:
        # TASK2: left 7 arm joints + left claw + right 7 arm joints + right claw.
        state = np.concatenate([
            aligned["joint_state"][:, 12:19],
            aligned["claw_state"][:, 0:1] / 100.0,
            aligned["joint_state"][:, 19:26],
            aligned["claw_state"][:, 1:2] / 100.0,
        ], axis=1).astype(np.float32)
        action = np.concatenate([
            aligned["arm_action"][:, 0:7],
            aligned["claw_action"][:, 0:1] / 100.0,
            aligned["arm_action"][:, 7:14],
            aligned["claw_action"][:, 1:2] / 100.0,
        ], axis=1).astype(np.float32)
        joint_dims = list(range(7)) + list(range(8, 15))
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"episode {episode}: non-finite reconstructed values")

    trim_start, trim_end = (
        command_trim_range(
            action, joint_dims, trim_joint_delta, trim_pre, trim_post
        ) if trim_idle else (0, len(action))
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npz")
    stat = bag_path.stat()
    np.savez_compressed(
        temporary, version=REBUILD_VERSION, episode=episode,
        bag_size=stat.st_size, expected_frames=expected_frames,
        arm_mode=arm_mode, trim_idle=trim_idle, trim_pre=trim_pre,
        trim_post=trim_post, trim_joint_delta=trim_joint_delta,
        trim_start=trim_start, trim_end=trim_end,
        state=state, action=action, target_times=targets,
    )
    temporary.replace(cache_path)
    return {
        "episode": episode, "state": state, "action": action,
        "target_times": targets, "bag_path": str(bag_path), "cached": False,
        "trim_start": trim_start, "trim_end": trim_end,
    }


def task_map(source_root: Path) -> dict[int, str]:
    path = source_root / "meta/tasks.parquet"
    table = pq.read_table(path)
    task_column = "task" if "task" in table.column_names else "__index_level_0__"
    return {
        int(index): str(task)
        for index, task in zip(
            table["task_index"].to_pylist(), table[task_column].to_pylist()
        )
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
    """Quiet, fast H.264 encoder compatible with LeRobot's worker API."""
    imgs_dir = Path(imgs_dir)
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(int(fps)),
        "-i", str(imgs_dir / "frame-%06d.png"),
        "-c:v", "libx264", "-preset", str(preset),
        "-crf", str(int(crf)), "-g", "10", "-pix_fmt", "yuv420p",
        str(video_path),
    ]
    subprocess.run(command, check=True)


def clean_source_frame(
    frame: dict[str, Any], state: np.ndarray, action: np.ndarray,
    tasks: dict[int, str], rgb_keys: set[str],
) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    task_index = int(frame.get("task_index", 0))
    for key, value in frame.items():
        if key in AUTO_KEYS or key.startswith("next."):
            continue
        if key.startswith("observation.depth"):
            continue
        if key.startswith("observation.images.") and key not in rgb_keys:
            continue
        if key in ("observation.state", "action"):
            continue
        if key in rgb_keys and isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if key in rgb_keys:
            array = np.asarray(value)
            if array.ndim == 3 and array.shape[0] in (1, 3, 4):
                array = np.moveaxis(array, 0, -1)
            if np.issubdtype(array.dtype, np.floating):
                if array.size and float(np.nanmax(array)) <= 1.5:
                    array = array * 255.0
                array = np.clip(array, 0, 255).astype(np.uint8)
            cleaned[key] = array
        else:
            cleaned[key] = value
    cleaned["observation.state"] = torch.from_numpy(state)
    cleaned["action"] = torch.from_numpy(action)
    cleaned["task"] = tasks.get(task_index, next(iter(tasks.values())))
    return cleaned


def create_destination(
    source: LeRobotDataset, source_root: Path, output_root: Path,
    output_repo_id: str, image_writer_processes: int,
) -> tuple[LeRobotDataset, set[str]]:
    features = dict(source.features)
    rgb_keys = {
        key for key, feature in features.items()
        if key.startswith(RGB_PREFIX) and not key.startswith("observation.depth")
        and feature.get("dtype") in ("video", "image")
    }
    for key in list(features):
        if key in AUTO_KEYS or key.startswith("next.") or key not in (
            {"observation.state", "action"} | rgb_keys
        ):
            features.pop(key, None)
    kwargs = {
        "repo_id": output_repo_id,
        "root": output_root,
        "fps": int(source.fps),
        "features": features,
        "robot_type": getattr(source.meta, "robot_type", None),
        "use_videos": True,
        "image_writer_processes": int(image_writer_processes),
        "image_writer_threads": 0,
        "batch_encoding_size": 1,
    }
    supported = inspect.signature(LeRobotDataset.create).parameters
    return LeRobotDataset.create(**{k: v for k, v in kwargs.items() if k in supported}), rgb_keys


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    whitelist = read_ids(args.whitelist)
    if args.max_episodes is not None:
        whitelist = whitelist[:max(0, int(args.max_episodes))]
    layout, _ = source_episode_layout(source_root)
    missing = sorted(set(whitelist) - set(layout))
    if missing:
        raise ValueError(f"Whitelist episodes absent from source: {missing[:20]}")

    audit_names = read_audit_bag_names(args.audit_csv)
    bag_files = sorted(args.bag_dir.expanduser().resolve().glob("*.bag"))
    bag_by_name = {path.name: path for path in bag_files}
    jobs = []
    for episode in whitelist:
        bag_name = audit_names[episode]
        if bag_name not in bag_by_name:
            raise FileNotFoundError(f"Missing bag for episode {episode}: {bag_name}")
        _, frame_count = layout[episode]
        cache = args.cache_dir.expanduser().resolve() / f"episode_{episode:04d}.npz"
        jobs.append((
            episode, str(bag_by_name[bag_name]), frame_count,
            args.sample_drop, args.camera_stride, str(cache), args.force_rebuild_cache,
            args.arm_mode, args.trim_idle, args.trim_pre_frames,
            args.trim_post_frames, args.trim_joint_delta,
        ))

    print(f"Extracting {len(jobs)} episodes with {args.workers} workers", flush=True)
    repaired: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(extract_one, job): job[0] for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            repaired[int(result["episode"])] = result
            if completed % 10 == 0 or completed == len(futures):
                print(f"lowdim {completed}/{len(futures)}", flush=True)

    if args.dry_run:
        print("Dry-run complete: all reconstructed arrays passed validation.")
        return
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_root}; pass --overwrite")
        shutil.rmtree(output_root)

    # LeRobot 0.4.3 hard-codes libsvtav1 in its internal worker. Override the
    # inherited worker global before it forks; H.264 turns a multi-minute AV1
    # episode encode into a practical batch export while remaining supported by
    # LeRobot's video loader.
    if args.video_codec == "h264":
        lerobot_dataset_module.encode_video_frames = partial(
            encode_h264_frames,
            crf=int(args.video_crf),
            preset=str(args.video_preset),
        )
    else:
        lerobot_dataset_module.encode_video_frames = partial(
            encode_video_frames,
            vcodec=args.video_codec,
            crf=int(args.video_crf),
            g=10,
            fast_decode=1,
        )

    source_kwargs = {"repo_id": args.source_repo_id, "root": source_root}
    if "return_uint8" in inspect.signature(LeRobotDataset).parameters:
        source_kwargs["return_uint8"] = True
    source = LeRobotDataset(**source_kwargs)
    destination, rgb_keys = create_destination(
        source, source_root, output_root, args.output_repo_id,
        args.image_writer_processes,
    )
    tasks = task_map(source_root)
    manifest = []
    total = sum(
        repaired[episode]["trim_end"] - repaired[episode]["trim_start"]
        for episode in whitelist
    )
    done = 0
    for output_episode, source_episode in enumerate(whitelist):
        global_start, frame_count = layout[source_episode]
        repair = repaired[source_episode]
        trim_start = int(repair["trim_start"])
        trim_end = int(repair["trim_end"])
        for local_index in range(trim_start, trim_end):
            frame = source[global_start + local_index]
            destination.add_frame(clean_source_frame(
                frame, repair["state"][local_index], repair["action"][local_index],
                tasks, rgb_keys,
            ))
            done += 1
            if done % 100 == 0 or done == total:
                print(f"frames {done}/{total}", flush=True)
        # Three camera encoders run concurrently.  libx264 uses about seven
        # threads per camera here, which fills a 24-thread workstation well.
        destination.save_episode(parallel_encoding=True)
        manifest.append({
            "output_episode": output_episode,
            "source_episode": source_episode,
            "bag_path": repair["bag_path"],
            "source_frames": frame_count,
            "trim_start": trim_start,
            "trim_end": trim_end,
            "frames": trim_end - trim_start,
            "header_start": float(repair["target_times"][trim_start]),
            "header_end": float(repair["target_times"][trim_end - 1]),
        })
        print(
            f"saved output episode {output_episode}/{len(whitelist)-1} "
            f"from source {source_episode}", flush=True,
        )

    if hasattr(destination, "finalize"):
        destination.finalize()
    elif hasattr(destination, "consolidate"):
        destination.consolidate()
    info_path = output_root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    canonical_codec = "h264" if args.video_codec == "h264" else "av1"
    for key in rgb_keys:
        feature_info = info["features"][key].setdefault("info", {})
        feature_info["video.codec"] = canonical_codec
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=4), encoding="utf-8")
    (output_root / "header_rebuild_manifest.json").write_text(
        json.dumps({
            "source_root": str(source_root),
            "whitelist": str(args.whitelist.resolve()),
            "sample_drop": args.sample_drop,
            "camera_stride": args.camera_stride,
            "arm_mode": args.arm_mode,
            "trim_idle": args.trim_idle,
            "trim_pre_frames": args.trim_pre_frames,
            "trim_post_frames": args.trim_post_frames,
            "trim_joint_delta": args.trim_joint_delta,
            "video_codec": args.video_codec,
            "video_crf": args.video_crf,
            "video_preset": args.video_preset,
            "rgb_keys": sorted(rgb_keys),
            "episodes": manifest,
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Complete: {output_root}")


if __name__ == "__main__":
    main()
