#!/usr/bin/env python3
"""Audit Beijing TASK1/2/3 ROS bags before LeRobot conversion.

The tool treats every ``task*/(main|slave)_p4-*`` directory as an independent
collection batch. Images are never decoded. It compares the old converter's
bag-time nearest-neighbour alignment with causal Header-time alignment, checks
payloads, arm command/state tracking, and claw/dex-hand response events.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import rosbag


AUDIT_VERSION = 2
CAMERA_TOPICS = {
    "camera_h": "/cam_h/color/image_raw/compressed",
    "camera_l": "/cam_l/color/image_raw/compressed",
    "camera_r": "/cam_r/color/image_raw/compressed",
}

TASKS = {
    "task1": {
        "topics": {
            "arm_state": "/sensors_data_raw",
            "arm_action": "/kuavo_arm_traj",
            "hand_state": "/leju_claw_state",
            "hand_action": "/leju_claw_command",
        },
        "arm_slice": (19, 26),
        "hand_indices": (1,),
        "hand_scale": 1.0,
        "min_events": 6,
    },
    "task2": {
        "topics": {
            "arm_state": "/sensors_data_raw",
            "arm_action": "/kuavo_arm_traj",
            "hand_state": "/leju_claw_state",
            "hand_action": "/leju_claw_command",
        },
        "arm_slice": (12, 26),
        "hand_indices": (0, 1),
        "hand_scale": 1.0,
        "min_events": 4,
    },
    "task3": {
        "topics": {
            "arm_state": "/sensors_data_raw",
            "arm_action": "/kuavo_arm_traj",
            "hand_state": "/dexhand/state",
            "hand_action": "/control_robot_hand_position",
        },
        "arm_slice": (19, 26),
        "hand_indices": (0, 2, 3, 4, 5),
        "hand_scale": 100.0,
        "min_events": 2,
    },
}

LIMITS = {
    "arm_state": (20.0, 100.0),
    "arm_action": (50.0, 150.0),
    "hand_state": (20.0, 100.0),
    "hand_action": (50.0, 150.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("/mnt/pqssd/Real_Beijing_Bag/beijing"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/tmp/beijing_rosbag_audit"),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit-per-group", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--sample-drop", type=int, default=10)
    parser.add_argument("--camera-stride", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--relax-event-count", action="store_true",
        help="Report event counts but do not reject episodes with fewer than expected.",
    )
    return parser.parse_args()


def stamp(msg: Any) -> float:
    try:
        value = float(msg.header.stamp.to_sec())
        return value if value > 0 else float("nan")
    except Exception:
        return float("nan")


def percentile(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else float("nan")


def payload(task_key: str, key: str, msg: Any) -> np.ndarray:
    config = TASKS[task_key]
    begin, end = config["arm_slice"]
    if key == "arm_state":
        values = np.asarray(msg.joint_data.joint_q, dtype=np.float64)
        if values.size < end:
            raise ValueError(f"joint_q length={values.size}, expected>={end}")
        return values[begin:end]
    if key == "arm_action":
        values = np.asarray(msg.position, dtype=np.float64)
        if values.size < end - 12:
            raise ValueError(f"arm action length={values.size}")
        action_begin = 0 if task_key == "task2" else 7
        action_end = 14 if task_key == "task2" else 14
        return np.deg2rad(values[action_begin:action_end])
    if task_key == "task3":
        if key == "hand_state":
            values = np.asarray(msg.position, dtype=np.float64)
            if values.size < 12:
                raise ValueError(f"dex state length={values.size}")
            return values[6:12]
        # ROS ``uint8[]`` is exposed as ``bytes`` by rospy on Python 3.
        values = np.asarray(list(msg.right_hand_position), dtype=np.float64)
        if values.size < 6:
            raise ValueError(f"dex action length={values.size}")
        return values[:6]
    values = np.asarray(msg.data.position, dtype=np.float64)
    if values.size < 2:
        raise ValueError(f"claw length={values.size}")
    return values[:2]


def causal_indices(source: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(source, targets, side="right") - 1
    valid = indices >= 0
    safe = np.maximum(indices, 0)
    ages = np.full(targets.shape, np.nan)
    ages[valid] = (targets[valid] - source[safe[valid]]) * 1000.0
    return safe, ages


def alignment_stats(
    source_header: np.ndarray,
    source_bag: np.ndarray,
    target_header: np.ndarray,
    target_bag: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    indices, ages = causal_indices(source_header, target_header)
    result["indices"] = indices
    result["missing_count"] = int(np.count_nonzero(~np.isfinite(ages)))
    for name, q in (("p50_ms", 50), ("p95_ms", 95), ("p99_ms", 99), ("max_ms", 100)):
        result[name] = percentile(ages, q)
    next_indices = np.searchsorted(source_header, target_header, side="left")
    bracket_valid = (indices >= 0) & (next_indices < source_header.size)
    bracket = np.full(target_header.shape, np.nan)
    nearest = np.full(target_header.shape, np.nan)
    bracket[bracket_valid] = (
        source_header[next_indices[bracket_valid]]
        - source_header[indices[bracket_valid]]
    ) * 1000.0
    nearest[bracket_valid] = np.minimum(
        target_header[bracket_valid] - source_header[indices[bracket_valid]],
        source_header[next_indices[bracket_valid]] - target_header[bracket_valid],
    ) * 1000.0
    result["bracket_missing_count"] = int(np.count_nonzero(~bracket_valid))
    result["bracket_p99_ms"] = percentile(bracket, 99)
    result["bracket_max_ms"] = percentile(bracket, 100)
    result["nearest_p99_ms"] = percentile(nearest, 99)

    right = np.searchsorted(source_bag, target_bag, side="left")
    right = np.clip(right, 0, source_bag.size - 1)
    left = np.clip(right - 1, 0, source_bag.size - 1)
    use_right = np.abs(source_bag[right] - target_bag) < np.abs(
        source_bag[left] - target_bag
    )
    direct_indices = np.where(use_right, right, left)
    direct = np.abs(source_header[direct_indices] - target_header) * 1000.0
    for name, q in (
        ("direct_p50_ms", 50), ("direct_p95_ms", 95),
        ("direct_p99_ms", 99), ("direct_max_ms", 100),
    ):
        result[name] = percentile(direct, q)
    return result


def stable_edges(times: np.ndarray, closed: np.ndarray) -> tuple[list[int], int]:
    raw = np.flatnonzero(closed[1:] != closed[:-1]) + 1
    kept = []
    for ordinal, edge in enumerate(raw):
        end = times[raw[ordinal + 1]] if ordinal + 1 < raw.size else times[-1]
        if end - times[edge] >= 0.20:
            kept.append(int(edge))
    return kept, int(raw.size)


def response_events(
    task_key: str,
    action_t: np.ndarray,
    action: np.ndarray,
    state_t: np.ndarray,
    state: np.ndarray,
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    config = TASKS[task_key]
    dimensions = tuple(config["hand_indices"])
    events: list[dict[str, Any]] = []
    channels = [dimensions] if task_key == "task3" else [(dimension,) for dimension in dimensions]
    for channel in channels:
        command = np.median(action[:, channel], axis=1) / float(config["hand_scale"])
        measured = np.median(state[:, channel], axis=1)
        if task_key == "task3":
            measured = measured / 100.0
        dimension: int | str = "main_fingers" if len(channel) > 1 else channel[0]
        closed = command > 0.5
        edges, raw_count = stable_edges(action_t, closed)
        events.append({
            "dimension": dimension,
            "kind": "diagnostic",
            "raw_edge_count": raw_count,
            "stable_edge_count": len(edges),
            "valid": True,
        })
        for ordinal, edge in enumerate(edges):
            event_time = float(action_t[edge])
            if not start <= event_time <= end:
                continue
            event_end = min(
                event_time + 1.5,
                float(action_t[edges[ordinal + 1]]) - 1e-6
                if ordinal + 1 < len(edges) else end,
            )
            direction = 1.0 if closed[edge] else -1.0
            before = measured[(state_t >= event_time - 0.10) & (state_t < event_time)]
            tail = measured[(state_t >= event_end - 0.15) & (state_t <= event_end)]
            valid = before.size > 0 and tail.size > 0 and event_end - event_time >= 0.20
            delta = (
                float(np.median(tail) - np.median(before))
                if valid else float("nan")
            )
            valid = bool(valid and direction * delta >= 0.10)
            events.append({
                "dimension": dimension,
                "kind": "close" if direction > 0 else "open",
                "time": event_time,
                "delta": delta,
                "valid": valid,
                "reason": "ok" if valid else "insufficient_response",
            })
    return events


def cache_path(cache_dir: Path, group: str, index: int, bag: Path) -> Path:
    digest = hashlib.sha1(str(bag).encode()).hexdigest()[:10]
    return cache_dir / group / f"{index:04d}_{digest}.json"


def audit_one(job: tuple[str, str, str, int, int, int]) -> dict[str, Any]:
    task_key, group, path_text, episode, sample_drop, stride = job
    path = Path(path_text)
    config = TASKS[task_key]
    result: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "task": task_key,
        "group": group,
        "group_episode": episode,
        "bag_name": path.name,
        "bag_path": str(path),
        "error": "",
    }
    try:
        stat = path.stat()
        result.update(bag_size_bytes=stat.st_size, bag_mtime_ns=stat.st_mtime_ns)
        keys = [*CAMERA_TOPICS, *config["topics"]]
        headers = {key: [] for key in keys}
        bag_times = {key: [] for key in keys}
        values = {key: [] for key in config["topics"]}
        invalid = {key: 0 for key in config["topics"]}
        topic_to_key = {topic: key for key, topic in config["topics"].items()}
        camera_topic_to_key = {topic: key for key, topic in CAMERA_TOPICS.items()}
        empty_camera_payload = {key: 0 for key in CAMERA_TOPICS}
        with rosbag.Bag(str(path), "r") as bag:
            for topic, msg, bag_time in bag.read_messages(
                topics=[*CAMERA_TOPICS.values(), *config["topics"].values()]
            ):
                key = (
                    camera_topic_to_key[topic]
                    if topic in camera_topic_to_key else topic_to_key[topic]
                )
                headers[key].append(stamp(msg))
                bag_times[key].append(float(bag_time.to_sec()))
                if key in CAMERA_TOPICS:
                    empty_camera_payload[key] += int(not bool(getattr(msg, "data", b"")))
                else:
                    try:
                        item = payload(task_key, key, msg)
                        invalid[key] += int(not np.isfinite(item).all())
                    except Exception:
                        width = 14 if task_key == "task2" and key.startswith("arm") else (
                            7 if key.startswith("arm") else (6 if task_key == "task3" else 2)
                        )
                        item = np.full(width, np.nan)
                        invalid[key] += 1
                    values[key].append(item)
        missing = [key for key in keys if not headers[key]]
        result.update(
            missing_topics=missing,
            invalid_payload=invalid,
            empty_camera_payload=empty_camera_payload,
        )
        if missing:
            result["error"] = "missing_topics"
            return result
        ht = {key: np.asarray(value, dtype=np.float64) for key, value in headers.items()}
        bt = {key: np.asarray(value, dtype=np.float64) for key, value in bag_times.items()}
        arrays = {key: np.asarray(value, dtype=np.float64) for key, value in values.items()}
        camera_h = ht["camera_h"][sample_drop:-sample_drop:stride]
        camera_b = bt["camera_h"][sample_drop:-sample_drop:stride]
        if camera_h.size < 2:
            result["error"] = "insufficient_camera_frames"
            return result
        result["retained_frame_count"] = int(camera_h.size)
        result["duration_s"] = float(camera_h[-1] - camera_h[0])
        for camera_key in CAMERA_TOPICS:
            result[f"{camera_key}_count"] = int(ht[camera_key].size)
            result[f"{camera_key}_nonmonotonic"] = int(
                np.count_nonzero(np.diff(ht[camera_key]) <= 0)
            )
        for camera_key in ("camera_l", "camera_r"):
            camera_stats = alignment_stats(
                ht[camera_key], bt[camera_key], camera_h, camera_b
            )
            camera_stats.pop("indices")
            for name, value in camera_stats.items():
                result[f"{camera_key}_{name}"] = value
        aligned = {}
        for key in config["topics"]:
            result[f"{key}_count"] = int(ht[key].size)
            result[f"{key}_nonmonotonic"] = int(np.count_nonzero(np.diff(ht[key]) <= 0))
            stats = alignment_stats(ht[key], bt[key], camera_h, camera_b)
            aligned[key] = arrays[key][stats.pop("indices")]
            for name, value in stats.items():
                result[f"{key}_{name}"] = value
            offset = (bt[key] - ht[key]) * 1000.0
            result[f"{key}_bag_header_offset_p50_ms"] = percentile(offset, 50)
            result[f"{key}_bag_header_offset_p99_ms"] = percentile(offset, 99)
            result[f"{key}_bag_header_offset_min_ms"] = percentile(offset, 0)
            result[f"{key}_bag_header_offset_max_ms"] = percentile(offset, 100)
        arm_error = np.abs(aligned["arm_action"] - aligned["arm_state"])
        result["arm_tracking_mae_rad"] = float(np.nanmean(arm_error))
        result["arm_tracking_p95_rad"] = percentile(np.nanmax(arm_error, axis=1), 95)
        result["arm_tracking_max_rad"] = percentile(np.nanmax(arm_error, axis=1), 100)
        events = response_events(
            task_key,
            ht["hand_action"], arrays["hand_action"],
            ht["hand_state"], arrays["hand_state"],
            float(camera_h[0]), float(camera_h[-1]),
        )
        physical = [event for event in events if event["kind"] != "diagnostic"]
        result["hand_event_count"] = len(physical)
        result["hand_valid_event_count"] = sum(bool(event["valid"]) for event in physical)
        result["hand_close_event_count"] = sum(event["kind"] == "close" for event in physical)
        result["hand_open_event_count"] = sum(event["kind"] == "open" for event in physical)
        result["hand_events"] = events
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def classify(
    row: dict[str, Any],
    relax_events: bool,
    include_cameras: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    direct: list[str] = []
    causal: list[str] = []
    interpolated: list[str] = []
    if row.get("error"):
        direct.append(str(row["error"]))
        causal.append(str(row["error"]))
        interpolated.append(str(row["error"]))
    if include_cameras:
        empty_camera = row.get("empty_camera_payload", {})
        for camera_key in CAMERA_TOPICS:
            if int(empty_camera.get(camera_key, 1)):
                reason = f"{camera_key}_empty_payload"
                direct.append(reason)
                causal.append(reason)
                interpolated.append(reason)
            if int(row.get(f"{camera_key}_nonmonotonic", 1)):
                reason = f"{camera_key}_nonmonotonic"
                direct.append(reason)
                causal.append(reason)
                interpolated.append(reason)
        for camera_key in ("camera_l", "camera_r"):
            nearest_p99 = float(row.get(f"{camera_key}_nearest_p99_ms", math.inf))
            bracket_p99 = float(row.get(f"{camera_key}_bracket_p99_ms", math.inf))
            bracket_max = float(row.get(f"{camera_key}_bracket_max_ms", math.inf))
            if (
                int(row.get(f"{camera_key}_bracket_missing_count", 1))
                or nearest_p99 > 50.0 or bracket_p99 > 100.0 or bracket_max > 250.0
            ):
                reason = (
                    f"{camera_key}_sync_nearest_p99_{nearest_p99:.1f}"
                    f"_bracket_p99_{bracket_p99:.1f}_max_{bracket_max:.1f}ms"
                )
                direct.append(reason)
                causal.append(reason)
                interpolated.append(reason)
    invalid = row.get("invalid_payload", {})
    for key, (p99_limit, max_limit) in LIMITS.items():
        if int(invalid.get(key, 1)):
            for reasons in (direct, causal, interpolated):
                reasons.append(f"{key}_invalid_payload")
        if int(row.get(f"{key}_nonmonotonic", 1)):
            for reasons in (direct, causal, interpolated):
                reasons.append(f"{key}_nonmonotonic")
        direct_p99 = float(row.get(f"{key}_direct_p99_ms", math.inf))
        direct_max = float(row.get(f"{key}_direct_max_ms", math.inf))
        if direct_p99 > p99_limit or direct_max > max_limit:
            direct.append(f"{key}_direct_p99_{direct_p99:.1f}_max_{direct_max:.1f}ms")
        p99 = float(row.get(f"{key}_p99_ms", math.inf))
        maximum = float(row.get(f"{key}_max_ms", math.inf))
        if int(row.get(f"{key}_missing_count", 1)) or p99 > p99_limit or maximum > max_limit:
            causal.append(f"{key}_p99_{p99:.1f}_max_{maximum:.1f}ms")
        if key.endswith("state"):
            bracket_p99 = float(row.get(f"{key}_bracket_p99_ms", math.inf))
            bracket_max = float(row.get(f"{key}_bracket_max_ms", math.inf))
            nearest_p99 = float(row.get(f"{key}_nearest_p99_ms", math.inf))
            if (
                int(row.get(f"{key}_bracket_missing_count", 1))
                or bracket_p99 > 100.0 or bracket_max > 250.0 or nearest_p99 > 50.0
            ):
                interpolated.append(
                    f"{key}_bracket_p99_{bracket_p99:.1f}_max_{bracket_max:.1f}ms"
                )
        elif int(row.get(f"{key}_missing_count", 1)) or p99 > p99_limit or maximum > max_limit:
            interpolated.append(f"{key}_p99_{p99:.1f}_max_{maximum:.1f}ms")
    task = str(row.get("task"))
    expected = int(TASKS.get(task, {}).get("min_events", 1))
    event_count = int(row.get("hand_event_count", 0))
    valid_count = int(row.get("hand_valid_event_count", 0))
    if not relax_events and (event_count < expected or valid_count != event_count):
        reason = f"hand_events_{valid_count}_of_{event_count}_expected>={expected}"
        direct.append(reason)
        causal.append(reason)
        interpolated.append(reason)
    arm_p95 = float(row.get("arm_tracking_p95_rad", math.inf))
    if arm_p95 > 0.35:
        reason = f"arm_tracking_p95_{arm_p95:.3f}rad"
        direct.append(reason)
        causal.append(reason)
        interpolated.append(reason)
    return direct, causal, interpolated


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    cache_dir = output / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    group_bags: dict[str, list[Path]] = {}
    for task_dir in sorted(root.glob("task*")):
        task_key = task_dir.name.split("_", 1)[0]
        if task_key not in TASKS:
            continue
        for group_dir in sorted(task_dir.glob("*")):
            if not group_dir.is_dir():
                continue
            group = f"{task_key}/{group_dir.name}"
            bags = sorted(group_dir.glob("*.bag"), key=lambda path: path.name)
            if args.limit_per_group:
                bags = bags[:args.limit_per_group]
            group_bags[group] = bags
            for episode, bag in enumerate(bags):
                jobs.append((task_key, group, str(bag), episode, args.sample_drop, args.camera_stride))
    if not jobs:
        raise SystemExit(f"No bags found under {root}")

    results: dict[tuple[str, int], dict[str, Any]] = {}
    pending = []
    for job in jobs:
        task_key, group, path_text, episode, _, _ = job
        bag = Path(path_text)
        cache = cache_path(cache_dir, group.replace("/", "__"), episode, bag)
        if cache.exists() and not args.force:
            row = json.loads(cache.read_text(encoding="utf-8"))
            stat = bag.stat()
            if (
                row.get("audit_version") == AUDIT_VERSION
                and row.get("bag_size_bytes") == stat.st_size
                and row.get("bag_mtime_ns") == stat.st_mtime_ns
            ):
                results[(group, episode)] = row
                continue
        pending.append(job)
    print(
        f"groups={len(group_bags)} bags={len(jobs)} cached={len(results)} "
        f"pending={len(pending)} workers={args.workers}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(audit_one, job): job for job in pending}
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            key = (str(row["group"]), int(row["group_episode"]))
            results[key] = row
            cache = cache_path(
                cache_dir, str(row["group"]).replace("/", "__"),
                int(row["group_episode"]), Path(row["bag_path"]),
            )
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(".tmp")
            temporary.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, cache)
            if completed % max(1, args.progress_every) == 0 or completed == len(pending):
                print(f"completed={completed}/{len(pending)}", flush=True)

    rows = []
    summary: dict[str, Any] = {
        "root": str(root),
        "audit_version": AUDIT_VERSION,
        "limits": LIMITS,
        "groups": {},
    }
    task_whitelists: dict[str, list[str]] = {task: [] for task in TASKS}
    task_lowdim_whitelists: dict[str, list[str]] = {task: [] for task in TASKS}
    for group, bags in group_bags.items():
        counters = {
            "total": len(bags),
            "direct": 0,
            "lowdim_causal": 0,
            "causal": 0,
            "interpolatable": 0,
        }
        reason_counts: dict[str, int] = {}
        group_whitelist = []
        group_lowdim_whitelist = []
        for episode, _ in enumerate(bags):
            row = results[(group, episode)]
            direct, causal, interpolated = classify(row, args.relax_event_count)
            _, lowdim_causal, _ = classify(
                row, args.relax_event_count, include_cameras=False
            )
            row.update(
                direct=not direct,
                lowdim_causal_header_usable=not lowdim_causal,
                causal_header_usable=not causal,
                interpolatable=not interpolated,
                direct_reasons=direct,
                causal_reasons=causal,
                interpolatable_reasons=interpolated,
            )
            counters["direct"] += int(not direct)
            counters["lowdim_causal"] += int(not lowdim_causal)
            counters["causal"] += int(not causal)
            counters["interpolatable"] += int(not interpolated)
            for reason in causal:
                reason_counts[reason.split("_p99_", 1)[0]] = (
                    reason_counts.get(reason.split("_p99_", 1)[0], 0) + 1
                )
            if not causal:
                group_whitelist.append(row["bag_path"])
                task_whitelists[str(row["task"])].append(row["bag_path"])
            if not lowdim_causal:
                group_lowdim_whitelist.append(row["bag_path"])
                task_lowdim_whitelists[str(row["task"])].append(row["bag_path"])
            flat = dict(row)
            for key in (
                "invalid_payload", "empty_camera_payload", "hand_events", "direct_reasons",
                "causal_reasons", "interpolatable_reasons",
            ):
                flat[key] = json.dumps(row.get(key), ensure_ascii=False)
            rows.append(flat)
        group_key = group.replace("/", "__")
        (output / f"{group_key}_causal_whitelist.txt").write_text(
            "\n".join(group_whitelist) + ("\n" if group_whitelist else ""),
            encoding="utf-8",
        )
        (output / f"{group_key}_lowdim_causal_whitelist.txt").write_text(
            "\n".join(group_lowdim_whitelist)
            + ("\n" if group_lowdim_whitelist else ""),
            encoding="utf-8",
        )
        summary["groups"][group] = {
            **counters,
            "causal_rejection_reason_counts": dict(
                sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
        }
    for task, paths in task_whitelists.items():
        (output / f"{task}_causal_whitelist.txt").write_text(
            "\n".join(paths) + ("\n" if paths else ""), encoding="utf-8"
        )
    for task, paths in task_lowdim_whitelists.items():
        (output / f"{task}_lowdim_causal_whitelist.txt").write_text(
            "\n".join(paths) + ("\n" if paths else ""), encoding="utf-8"
        )
    write_csv(output / "beijing_rosbag_audit.csv", rows)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
