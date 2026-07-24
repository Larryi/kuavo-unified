#!/usr/bin/env python3
"""Rebuild the qualified TASK3-200 episodes with Header-aligned low-dimensional data.

The RGB videos and episode trims are preserved byte-for-byte.  Right-arm and
right-hand state/action are reconstructed from the original bag selected by the
two trim manifests.  State is linearly interpolated at each retained head-camera
Header timestamp; action uses causal zero-order hold (latest command at or before
the image timestamp).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rosbag
from lerobot.datasets.compute_stats import aggregate_feature_stats


CAMERA_TOPIC = "/cam_h/color/image_raw/compressed"
TOPICS = {
    "arm_state": "/sensors_data_raw",
    "arm_action": "/kuavo_arm_traj",
    "hand_state": "/dexhand/state",
    "hand_action": "/control_robot_hand_position",
}
CAMERAS = (
    "observation.images.head_cam_h",
    "observation.images.wrist_cam_r",
)
STAT_NAMES = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-dir", type=Path, default=Path("/mnt/pqssd/task3_dajian"))
    parser.add_argument("--source-root", type=Path, default=Path(
        "/mnt/pqssd/Real_PQ_3.0/TASK3_SZ/task3_supertrimmed_200"
    ))
    parser.add_argument("--trim-config", type=Path, default=Path(
        "/mnt/pqssd/Real_PQ_3.0/TASK3_SZ/lerobot/trim_config.json"
    ))
    parser.add_argument("--supertrim-manifest", type=Path, default=Path(
        "/mnt/pqssd/Real_PQ_3.0/TASK3_SZ/task3_supertrimmed_200/trim_manifest.json"
    ))
    parser.add_argument("--whitelist", type=Path, default=Path(
        "/mnt/pqssd/task3_lowdim_quality/usable_task3_supertrimmed_200_episodes.txt"
    ))
    parser.add_argument("--output-root", type=Path, default=Path(
        "/mnt/pqssd/Real_PQ_3.0/TASK3_SZ_Repaired/lerobot_task3_165"
    ))
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/task3_header_rebuild_cache"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--video-mode", choices=("hardlink", "copy", "symlink"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stamp(msg: Any) -> float:
    return float(msg.header.stamp.to_sec())


def payload(key: str, msg: Any) -> np.ndarray:
    if key == "arm_state":
        return np.asarray(msg.joint_data.joint_q, dtype=np.float64)[19:26]
    if key == "arm_action":
        return np.deg2rad(np.asarray(msg.position, dtype=np.float64)[7:14])
    if key == "hand_state":
        return np.asarray(msg.position, dtype=np.float64)[6:12]
    return np.asarray(list(msg.right_hand_position), dtype=np.float64)[:6]


def deduplicate(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    keep = np.r_[np.diff(times) > 0, True]  # retain the newest value at a duplicate timestamp
    return times[keep], values[keep]


def interpolate(times: np.ndarray, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    times, values = deduplicate(times, values)
    if targets[0] < times[0]:
        raise ValueError("state has no causal sample for the first camera timestamp")
    # np.interp performs a causal last-value hold if the final camera frame is
    # slightly newer than the last state sample.  Qualified episodes already
    # bound that sample age; this avoids inventing a future observation.
    return np.column_stack([
        np.interp(targets, times, values[:, dim]) for dim in range(values.shape[1])
    ])


def causal_hold(times: np.ndarray, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    times, values = deduplicate(times, values)
    indices = np.searchsorted(times, targets, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("action has no causal sample for an image timestamp")
    return values[indices]


def extract_one(job: tuple[int, int, str, int, int, int, str]) -> dict[str, Any]:
    source_episode, raw_episode, bag_text, start, end, expected, cache_text = job
    cache = Path(cache_text)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if (int(data["source_episode"]) == source_episode
                    and int(data["raw_episode"]) == raw_episode
                    and int(data["start"]) == start and int(data["end"]) == end):
                return {
                    "source_episode": source_episode, "raw_episode": raw_episode,
                    "bag_path": bag_text, "start": start, "end": end,
                    "target_times": data["target_times"].copy(),
                    "state": data["state"].copy(), "action": data["action"].copy(),
                }
    topic_to_key = {topic: key for key, topic in TOPICS.items()}
    times = {"camera": []} | {key: [] for key in TOPICS}
    values = {key: [] for key in TOPICS}
    with rosbag.Bag(bag_text, "r") as bag:
        for topic, msg, _ in bag.read_messages(topics=[CAMERA_TOPIC, *TOPICS.values()]):
            key = "camera" if topic == CAMERA_TOPIC else topic_to_key[topic]
            times[key].append(stamp(msg))
            if key != "camera":
                values[key].append(payload(key, msg))
    camera = np.asarray(times["camera"], dtype=np.float64)[10:-10:3]
    if not 0 <= start < end <= len(camera):
        raise ValueError(f"episode {source_episode}: trim {start}:{end} outside {len(camera)}")
    target = camera[start:end]
    if len(target) != expected:
        raise ValueError(f"episode {source_episode}: bag gives {len(target)} frames, source has {expected}")
    arrays_t = {key: np.asarray(times[key], dtype=np.float64) for key in TOPICS}
    arrays_v = {key: np.asarray(values[key], dtype=np.float64) for key in TOPICS}
    state = np.column_stack([
        interpolate(arrays_t["arm_state"], arrays_v["arm_state"], target),
        interpolate(arrays_t["hand_state"], arrays_v["hand_state"], target)[:, 0] / 100.0,
    ]).astype(np.float32)
    action = np.column_stack([
        causal_hold(arrays_t["arm_action"], arrays_v["arm_action"], target),
        causal_hold(arrays_t["hand_action"], arrays_v["hand_action"], target)[:, 0] / 100.0,
    ]).astype(np.float32)
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"episode {source_episode}: non-finite reconstructed values")
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary, source_episode=source_episode, raw_episode=raw_episode,
        start=start, end=end, target_times=target, state=state, action=action,
    )
    os.replace(temporary, cache)
    return {
        "source_episode": source_episode, "raw_episode": raw_episode,
        "bag_path": bag_text, "start": start, "end": end,
        "target_times": target, "state": state, "action": action,
    }


def vector_stats(array: np.ndarray) -> dict[str, list]:
    array = np.asarray(array)
    return {
        "min": np.min(array, axis=0).tolist(), "max": np.max(array, axis=0).tolist(),
        "mean": np.mean(array, axis=0, dtype=np.float64).tolist(),
        "std": np.std(array, axis=0, dtype=np.float64).tolist(),
        "count": [int(len(array))],
        "q01": np.quantile(array, .01, axis=0).tolist(),
        "q10": np.quantile(array, .10, axis=0).tolist(),
        "q50": np.quantile(array, .50, axis=0).tolist(),
        "q90": np.quantile(array, .90, axis=0).tolist(),
        "q99": np.quantile(array, .99, axis=0).tolist(),
    }


def replace_stats(row: dict, feature: str, stats: dict[str, list]) -> None:
    for name, value in stats.items():
        row[f"stats/{feature}/{name}"] = value


def row_feature_stats(row: dict, feature: str) -> dict[str, np.ndarray]:
    return {name: np.asarray(row[f"stats/{feature}/{name}"]) for name in STAT_NAMES}


def source_video(root: Path, camera: str, row: dict) -> Path:
    prefix = f"videos/{camera}"
    return root / "videos" / camera / f"chunk-{int(row[f'{prefix}/chunk_index']):03d}" / f"file-{int(row[f'{prefix}/file_index']):03d}.mp4"


def load_mapping(trim_config: Path, manifest: Path, bags: list[Path]) -> dict[int, dict[str, Any]]:
    trims_raw = json.loads(trim_config.read_text(encoding="utf-8"))["trims"]
    trims = {int(k): v for k, v in trims_raw.items()}
    kept_raw = [i for i in range(len(bags)) if not bool(trims[i].get("drop", False))]
    exported = json.loads(manifest.read_text(encoding="utf-8"))["exported"]
    mapping = {}
    for item in exported:
        output = int(item["output_episode"])
        segments = item["source_segments"]
        if len(segments) != 1:
            raise ValueError(f"source episode {output} contains {len(segments)} segments")
        segment = segments[0]
        raw = kept_raw[int(segment["source_episode"])]
        mapping[output] = {
            "raw_episode": raw, "bag_path": bags[raw],
            "start": int(trims[raw]["start"]) + int(segment["start"]),
            "end": int(trims[raw]["start"]) + int(segment["end"]),
        }
    return mapping


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    bags = sorted(args.bag_dir.resolve().glob("*.bag"), key=lambda path: str(path))
    if len(bags) != 1000:
        raise ValueError(f"expected 1000 bags, found {len(bags)}")
    selected = sorted({int(x) for x in args.whitelist.read_text().split()})
    if args.max_episodes:
        selected = selected[:args.max_episodes]
    mapping = load_mapping(args.trim_config, args.supertrim_manifest, bags)
    source_episode_table = pq.read_table(source_root / "meta/episodes/chunk-000/file-000.parquet")
    source_rows = {int(row["episode_index"]): row for row in source_episode_table.to_pylist()}
    source_data = pq.read_table(source_root / "data/chunk-000/file-000.parquet")
    old_state = np.asarray(source_data["observation.state"].to_pylist(), dtype=np.float32)
    old_action = np.asarray(source_data["action"].to_pylist(), dtype=np.float32)

    jobs = []
    for source_episode in selected:
        row, item = source_rows[source_episode], mapping[source_episode]
        cache = args.cache_dir / f"episode_{source_episode:03d}_raw_{item['raw_episode']:04d}.npz"
        jobs.append((source_episode, item["raw_episode"], str(item["bag_path"]),
                     item["start"], item["end"], int(row["length"]), str(cache)))
    rebuilt_by_id = {}
    print(f"Extracting {len(jobs)} episodes with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(extract_one, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            item = future.result()
            rebuilt_by_id[item["source_episode"]] = item
            if completed % 10 == 0 or completed == len(futures):
                print(f"lowdim {completed}/{len(futures)}", flush=True)
    rebuilt = [rebuilt_by_id[index] for index in selected]

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "data/chunk-000").mkdir(parents=True)
    (output_root / "meta/episodes/chunk-000").mkdir(parents=True)

    states = np.concatenate([item["state"] for item in rebuilt])
    actions = np.concatenate([item["action"] for item in rebuilt])
    lengths = [len(item["state"]) for item in rebuilt]
    timestamps = np.concatenate([np.arange(n, dtype=np.float32) / 10 for n in lengths])
    frame_indices = np.concatenate([np.arange(n, dtype=np.int64) for n in lengths])
    episode_indices = np.concatenate([np.full(n, i, dtype=np.int64) for i, n in enumerate(lengths)])
    indices = np.arange(sum(lengths), dtype=np.int64)
    task_indices = np.zeros(sum(lengths), dtype=np.int64)
    source_schema = source_data.schema
    columns = {
        "observation.state": pa.array(states.tolist(), type=source_schema.field("observation.state").type),
        "action": pa.array(actions.tolist(), type=source_schema.field("action").type),
        "timestamp": pa.array(timestamps, type=source_schema.field("timestamp").type),
        "frame_index": pa.array(frame_indices, type=source_schema.field("frame_index").type),
        "episode_index": pa.array(episode_indices, type=source_schema.field("episode_index").type),
        "index": pa.array(indices, type=source_schema.field("index").type),
        "task_index": pa.array(task_indices, type=source_schema.field("task_index").type),
    }
    pq.write_table(pa.Table.from_arrays(
        [columns[field.name] for field in source_schema], schema=source_schema
    ), output_root / "data/chunk-000/file-000.parquet")

    episode_rows, manifest_rows = [], []
    cursor = 0
    for output_episode, item in enumerate(rebuilt):
        source_episode = int(item["source_episode"])
        source_row = source_rows[source_episode]
        n = len(item["state"])
        row = dict(source_row)
        row.update({
            "episode_index": output_episode, "length": n,
            "data/chunk_index": 0, "data/file_index": 0,
            "dataset_from_index": cursor, "dataset_to_index": cursor + n,
            "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
        })
        replace_stats(row, "observation.state", vector_stats(item["state"]))
        replace_stats(row, "action", vector_stats(item["action"]))
        scalars = {
            "timestamp": np.arange(n, dtype=np.float32) / 10,
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, output_episode, dtype=np.int64),
            "index": np.arange(cursor, cursor + n, dtype=np.int64),
            "task_index": np.zeros(n, dtype=np.int64),
        }
        for name, array in scalars.items():
            replace_stats(row, name, vector_stats(array.reshape(-1, 1)))
        for camera in CAMERAS:
            prefix = f"videos/{camera}"
            source = source_video(source_root, camera, source_row)
            duration = (float(source_row[f"{prefix}/to_timestamp"])
                        - float(source_row[f"{prefix}/from_timestamp"]))
            if abs(duration - n / 10.0) > 1e-5:
                raise ValueError(f"{source}: video window {duration}s does not match {n} frames")
            # LeRobot packs many episodes consecutively in each AV1 file.  Keep
            # the original file index and timestamp window, and materialize each
            # referenced container only once in the filtered dataset.
            output = (output_root / "videos" / camera
                      / f"chunk-{int(source_row[f'{prefix}/chunk_index']):03d}"
                      / f"file-{int(source_row[f'{prefix}/file_index']):03d}.mp4")
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.exists():
                if args.video_mode == "hardlink":
                    try:
                        os.link(source, output)
                    except OSError as exc:
                        if exc.errno != 18:  # EXDEV: fall back across filesystems
                            raise
                        shutil.copy2(source, output)
                elif args.video_mode == "copy":
                    shutil.copy2(source, output)
                else:
                    output.symlink_to(source)
        start, end = int(source_row["dataset_from_index"]), int(source_row["dataset_to_index"])
        state_delta = np.abs(item["state"] - old_state[start:end])
        action_delta = np.abs(item["action"] - old_action[start:end])
        manifest_rows.append({
            "output_episode": output_episode, "source_episode": source_episode,
            "raw_episode": int(item["raw_episode"]), "bag_path": item["bag_path"],
            "master_trim_start": int(item["start"]), "master_trim_end": int(item["end"]),
            "frames": n, "header_start": float(item["target_times"][0]),
            "header_end": float(item["target_times"][-1]),
            "old_state_mae": float(state_delta.mean()), "old_state_max": float(state_delta.max()),
            "old_action_mae": float(action_delta.mean()), "old_action_max": float(action_delta.max()),
        })
        episode_rows.append(row)
        cursor += n
    pq.write_table(pa.Table.from_pylist(episode_rows, schema=source_episode_table.schema),
                   output_root / "meta/episodes/chunk-000/file-000.parquet")
    shutil.copy2(source_root / "meta/tasks.parquet", output_root / "meta/tasks.parquet")

    info = copy.deepcopy(json.loads((source_root / "meta/info.json").read_text()))
    info.update({"total_episodes": len(rebuilt), "total_frames": sum(lengths),
                 "splits": {"train": f"0:{len(rebuilt)}"}})
    (output_root / "meta/info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    stats = copy.deepcopy(json.loads((source_root / "meta/stats.json").read_text()))
    arrays = {"observation.state": states, "action": actions, "timestamp": timestamps,
              "frame_index": frame_indices, "episode_index": episode_indices,
              "index": indices, "task_index": task_indices}
    for name, array in arrays.items():
        stats[name] = vector_stats(array if array.ndim == 2 else array.reshape(-1, 1))
    for camera in CAMERAS:
        aggregated = aggregate_feature_stats([
            row_feature_stats(source_rows[index], camera) for index in selected
        ])
        stats[camera] = {name: np.asarray(value).tolist() for name, value in aggregated.items()}
    (output_root / "meta/stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (output_root / "header_rebuild_manifest.json").write_text(json.dumps({
        "source_root": str(source_root), "bag_dir": str(args.bag_dir.resolve()),
        "alignment": {"clock": "ROS message Header", "state": "linear interpolation",
                      "action": "causal zero-order hold", "camera_master": CAMERA_TOPIC,
                      "camera_sampling": "drop 10 at each end, then stride 3"},
        "schema": "7 right-arm joints + right-hand first scalar / 100",
        "video_mode": args.video_mode, "episodes": manifest_rows,
    }, indent=2), encoding="utf-8")

    # Structural read-back validation.
    check = pq.read_table(output_root / "data/chunk-000/file-000.parquet")
    check_eps = pq.read_table(output_root / "meta/episodes/chunk-000/file-000.parquet")
    assert check.num_rows == sum(lengths) and check_eps.num_rows == len(rebuilt)
    assert np.array_equal(np.asarray(check["index"]), np.arange(sum(lengths)))
    assert np.isfinite(np.asarray(check["observation.state"].to_pylist())).all()
    assert np.isfinite(np.asarray(check["action"].to_pylist())).all()
    print(f"Complete: {output_root} ({len(rebuilt)} episodes, {sum(lengths)} frames)", flush=True)


if __name__ == "__main__":
    main()
