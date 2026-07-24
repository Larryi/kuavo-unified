#!/usr/bin/env python3
"""Audit TASK3 right-arm and six-DoF dexterous-hand alignment.

The raw bag ordering is reproduced by lexicographically sorting ``*.bag``.
Only low-bandwidth topics and compressed-image headers are read; images are
never decoded.  Each retained 10 Hz camera frame is checked against:

* right arm state: ``/sensors_data_raw.joint_data.joint_q[19:26]``
* right arm action: ``/kuavo_arm_traj.position[7:14]`` (degrees -> radians)
* right hand state: ``/dexhand/state.position[6:12]``
* right hand action: ``/control_robot_hand_position.right_hand_position[0:6]``

The report distinguishes the old bag-time nearest-neighbour conversion from a
repairable causal Header alignment and also scores command/state tracking plus
open/close response events.  Existing TASK3 supertrimmed episodes are traced
back through ``trim_config.json`` and ``trim_manifest.json``.
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


CAMERA_TOPIC = "/cam_h/color/image_raw/compressed"
TOPICS = {
    "arm_state": "/sensors_data_raw",
    "arm_action": "/kuavo_arm_traj",
    "hand_state": "/dexhand/state",
    "hand_action": "/control_robot_hand_position",
}
AUDIT_VERSION = 4
MAIN_HAND_DIMS = np.asarray([0, 2, 3, 4, 5], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-dir", type=Path, default=Path("/mnt/pqssd/task3_dajian"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/task3_lowdim_quality"))
    parser.add_argument("--trim-config", type=Path, default=Path(
        "/mnt/pqssd/Real_PQ_3.0/TASK3_SZ/lerobot/trim_config.json"
    ))
    parser.add_argument("--supertrim-manifest", type=Path, default=Path(
        "/mnt/pqssd/Real_PQ_3.0/TASK3_SZ/task3_supertrimmed_200/trim_manifest.json"
    ))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--expected-bags", type=int, default=1000)
    parser.add_argument("--allow-count-mismatch", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Audit only the first N bags")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sample-drop", type=int, default=10)
    parser.add_argument("--camera-stride", type=int, default=3)
    parser.add_argument("--event-window-seconds", type=float, default=1.5)
    parser.add_argument("--arm-state-p99-ms", type=float, default=20.0)
    parser.add_argument("--arm-state-max-ms", type=float, default=100.0)
    parser.add_argument("--arm-action-p99-ms", type=float, default=50.0)
    parser.add_argument("--arm-action-max-ms", type=float, default=150.0)
    parser.add_argument("--hand-state-p99-ms", type=float, default=20.0)
    parser.add_argument("--hand-state-max-ms", type=float, default=100.0)
    parser.add_argument("--hand-action-p99-ms", type=float, default=50.0)
    parser.add_argument("--hand-action-max-ms", type=float, default=150.0)
    parser.add_argument("--arm-tracking-p95-rad", type=float, default=0.35)
    parser.add_argument(
        "--hand-tracking-p95", type=float, default=60.0,
        help="Advisory only; physical hand range/lag is not used to reject an episode",
    )
    return parser.parse_args()


def header_stamp(msg: Any) -> float:
    try:
        value = float(msg.header.stamp.to_sec())
        return value if value > 0 else float("nan")
    except Exception:
        return float("nan")


def payload(key: str, msg: Any) -> np.ndarray:
    if key == "arm_state":
        values = np.asarray(msg.joint_data.joint_q, dtype=np.float64)
        if values.size < 26:
            raise ValueError(f"arm_state has {values.size} values")
        return values[19:26]
    if key == "arm_action":
        values = np.asarray(msg.position, dtype=np.float64)
        if values.size < 14:
            raise ValueError(f"arm_action has {values.size} values")
        return np.deg2rad(values[7:14])
    if key == "hand_state":
        values = np.asarray(msg.position, dtype=np.float64)
        if values.size < 12:
            raise ValueError(f"hand_state has {values.size} values")
        return values[6:12]
    values = np.asarray(list(msg.right_hand_position), dtype=np.float64)
    if values.size < 6:
        raise ValueError(f"hand_action has {values.size} values")
    return values[:6]


def percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, q)) if finite.size else float("nan")


def monotonic_count(values: np.ndarray) -> int:
    return int(np.count_nonzero(np.diff(values) <= 0))


def causal_indices(source: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(source, targets, side="right") - 1
    valid = indices >= 0
    safe = np.maximum(indices, 0)
    ages = np.full(targets.shape, np.nan, dtype=np.float64)
    ages[valid] = (targets[valid] - source[safe[valid]]) * 1000.0
    return safe, ages


def direct_header_errors(
    source_bag: np.ndarray,
    source_header: np.ndarray,
    target_bag: np.ndarray,
    target_header: np.ndarray,
) -> np.ndarray:
    right = np.searchsorted(source_bag, target_bag, side="left")
    right = np.clip(right, 0, len(source_bag) - 1)
    left = np.clip(right - 1, 0, len(source_bag) - 1)
    choose_right = np.abs(source_bag[right] - target_bag) < np.abs(
        source_bag[left] - target_bag
    )
    indices = np.where(choose_right, right, left)
    return np.abs(source_header[indices] - target_header) * 1000.0


def first_crossing(
    times: np.ndarray,
    signal: np.ndarray,
    start: float,
    end: float,
    threshold: float,
    direction: int,
) -> float | None:
    ids = np.flatnonzero((times >= start) & (times <= end))
    ids = ids[signal[ids] >= threshold] if direction > 0 else ids[signal[ids] <= threshold]
    return None if not ids.size else float(times[ids[0]])


def hand_events(
    command_t: np.ndarray,
    command: np.ndarray,
    state_t: np.ndarray,
    state: np.ndarray,
    window_seconds: float,
    interval_start: float,
    interval_end: float,
) -> tuple[list[dict[str, Any]], int, int]:
    command_signal = np.median(command[:, MAIN_HAND_DIMS], axis=1)
    state_signal = np.median(state[:, MAIN_HAND_DIMS], axis=1)
    closed = command_signal > 40.0
    all_edges = np.flatnonzero(closed[1:] != closed[:-1]) + 1
    raw_edges = all_edges[
        (command_t[all_edges] >= interval_start)
        & (command_t[all_edges] <= interval_end)
    ]
    # A smooth 100 Hz command ramp can cross back and forth around 40 for a few
    # samples.  Treat only a new intent held for >=200 ms as an event, while
    # retaining the discarded edge count as a command-chatter diagnostic.
    edges = []
    for ordinal, edge in enumerate(raw_edges):
        next_edge = raw_edges[ordinal + 1] if ordinal + 1 < len(raw_edges) else None
        hold_end = command_t[next_edge] if next_edge is not None else interval_end
        hold_end = min(float(hold_end), interval_end)
        if float(hold_end - command_t[edge]) >= 0.20:
            edges.append(int(edge))
    events: list[dict[str, Any]] = []
    for ordinal, edge in enumerate(edges):
        start = float(command_t[edge])
        next_edge = float(command_t[edges[ordinal + 1]]) if ordinal + 1 < len(edges) else interval_end
        end = min(start + window_seconds, next_edge - 1e-6, interval_end)
        direction = 1 if bool(closed[edge]) else -1
        event: dict[str, Any] = {
            "kind": "close" if direction > 0 else "open",
            "command_header_time": start,
            "window_end": end,
            "valid": False,
            "reason": "",
        }
        before_mask = (state_t >= start - 0.10) & (state_t < start)
        tail_mask = (state_t >= end - 0.15) & (state_t <= end)
        if end - start < 0.20 or not before_mask.any() or not tail_mask.any():
            event["reason"] = "insufficient_response_window"
            events.append(event)
            continue
        before = np.median(state[before_mask], axis=0)
        final = np.median(state[tail_mask], axis=0)
        delta = final - before
        responsive = direction * delta[MAIN_HAND_DIMS] >= 5.0
        baseline_signal = float(np.median(before[MAIN_HAND_DIMS]))
        final_signal = float(np.median(final[MAIN_HAND_DIMS]))
        signal_delta = final_signal - baseline_signal
        event.update({
            "baseline": baseline_signal,
            "final": final_signal,
            "delta": signal_delta,
            "responsive_main_dims": int(responsive.sum()),
            "per_dim_delta": delta.tolist(),
        })
        if direction * signal_delta < 10.0 or responsive.sum() < 4:
            event["reason"] = "insufficient_hand_response"
            events.append(event)
            continue
        for fraction in (0.05, 0.50, 0.95):
            threshold = baseline_signal + fraction * signal_delta
            hit = first_crossing(state_t, state_signal, start, end, threshold, direction)
            event[f"delay_{int(fraction * 100):02d}_ms"] = (
                None if hit is None else (hit - start) * 1000.0
            )
        delay95 = event["delay_95_ms"]
        if delay95 is None or not 0.0 <= float(delay95) <= 1200.0:
            event["reason"] = "response_delay_out_of_range"
            events.append(event)
            continue
        event["valid"] = True
        event["reason"] = "ok"
        events.append(event)
    return events, int(raw_edges.size), int(raw_edges.size - len(edges))


def cache_path(cache_dir: Path, episode: int, bag_path: Path) -> Path:
    digest = hashlib.sha1(str(bag_path).encode()).hexdigest()[:10]
    return cache_dir / f"episode_{episode:04d}_{digest}.json"


def audit_one(job: tuple[int, str, int, int, int, int, float]) -> dict[str, Any]:
    episode, path_text, trim_start, trim_end, sample_drop, stride, event_window = job
    path = Path(path_text)
    result: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "raw_episode": episode,
        "bag_path": str(path),
        "bag_name": path.name,
        "trim_start": trim_start,
        "trim_end": trim_end,
        "sample_drop": sample_drop,
        "camera_stride": stride,
        "event_window_seconds": event_window,
        "error": "",
    }
    try:
        stat = path.stat()
        result["bag_size_bytes"] = stat.st_size
        result["bag_mtime_ns"] = stat.st_mtime_ns
        headers = {"camera": []} | {key: [] for key in TOPICS}
        bag_times = {"camera": []} | {key: [] for key in TOPICS}
        values = {key: [] for key in TOPICS}
        invalid_payload = {key: 0 for key in TOPICS}
        topic_to_key = {value: key for key, value in TOPICS.items()}
        with rosbag.Bag(str(path), "r") as bag:
            for topic, msg, bag_time in bag.read_messages(topics=[CAMERA_TOPIC, *TOPICS.values()]):
                key = "camera" if topic == CAMERA_TOPIC else topic_to_key[topic]
                headers[key].append(header_stamp(msg))
                bag_times[key].append(float(bag_time.to_sec()))
                if key != "camera":
                    try:
                        item = payload(key, msg)
                        invalid_payload[key] += int(not np.isfinite(item).all())
                    except Exception:
                        item = np.full(7 if key.startswith("arm") else 6, np.nan)
                        invalid_payload[key] += 1
                    values[key].append(item)
        result["invalid_payload"] = invalid_payload
        missing = [key for key, sequence in headers.items() if not sequence]
        result["missing_topics"] = missing
        if missing:
            result["error"] = "missing_topics"
            return result
        ht = {key: np.asarray(sequence, dtype=np.float64) for key, sequence in headers.items()}
        bt = {key: np.asarray(sequence, dtype=np.float64) for key, sequence in bag_times.items()}
        arrays = {key: np.asarray(sequence, dtype=np.float64) for key, sequence in values.items()}
        camera_h = ht["camera"][sample_drop:-sample_drop:stride]
        camera_b = bt["camera"][sample_drop:-sample_drop:stride]
        result["master_frame_count"] = int(camera_h.size)
        if not 0 <= trim_start < trim_end <= camera_h.size:
            result["error"] = f"trim_{trim_start}_{trim_end}_outside_master_{camera_h.size}"
            return result
        target_h = camera_h[trim_start:trim_end]
        target_b = camera_b[trim_start:trim_end]
        result["retained_frame_count"] = int(target_h.size)
        aligned: dict[str, np.ndarray] = {}
        for key in TOPICS:
            result[f"{key}_count"] = int(ht[key].size)
            result[f"{key}_nonmonotonic"] = monotonic_count(ht[key])
            indices, ages = causal_indices(ht[key], target_h)
            aligned[key] = arrays[key][indices]
            result[f"{key}_missing_count"] = int(np.count_nonzero(~np.isfinite(ages)))
            result[f"{key}_p50_ms"] = percentile(ages, 50)
            result[f"{key}_p95_ms"] = percentile(ages, 95)
            result[f"{key}_p99_ms"] = percentile(ages, 99)
            result[f"{key}_max_ms"] = percentile(ages, 100)
            direct = direct_header_errors(bt[key], ht[key], target_b, target_h)
            result[f"{key}_direct_p50_ms"] = percentile(direct, 50)
            result[f"{key}_direct_p95_ms"] = percentile(direct, 95)
            result[f"{key}_direct_p99_ms"] = percentile(direct, 99)
            result[f"{key}_direct_max_ms"] = percentile(direct, 100)
        arm_error = np.abs(aligned["arm_action"] - aligned["arm_state"])
        hand_error = np.abs(aligned["hand_action"] - aligned["hand_state"])
        result["arm_tracking_mae_rad"] = float(np.nanmean(arm_error))
        result["arm_tracking_p95_rad"] = percentile(np.max(arm_error, axis=1), 95)
        result["arm_tracking_max_rad"] = percentile(np.max(arm_error, axis=1), 100)
        result["hand_tracking_mae"] = float(np.nanmean(hand_error[:, MAIN_HAND_DIMS]))
        result["hand_tracking_p95"] = percentile(
            np.max(hand_error[:, MAIN_HAND_DIMS], axis=1), 95
        )
        result["hand_tracking_max"] = percentile(
            np.max(hand_error[:, MAIN_HAND_DIMS], axis=1), 100
        )
        result["hand_aux_tracking_mae"] = float(np.nanmean(hand_error[:, 1]))
        result["hand_action_min"] = np.nanmin(arrays["hand_action"], axis=0).tolist()
        result["hand_action_max"] = np.nanmax(arrays["hand_action"], axis=0).tolist()
        result["hand_state_min"] = np.nanmin(arrays["hand_state"], axis=0).tolist()
        result["hand_state_max"] = np.nanmax(arrays["hand_state"], axis=0).tolist()
        events, raw_edge_count, short_edge_count = hand_events(
            ht["hand_action"], arrays["hand_action"],
            ht["hand_state"], arrays["hand_state"], event_window,
            float(target_h[0]), float(target_h[-1]),
        )
        result["hand_raw_threshold_edge_count"] = raw_edge_count
        result["hand_short_threshold_edge_count"] = short_edge_count
        result["hand_event_count"] = len(events)
        result["hand_valid_event_count"] = sum(bool(event["valid"]) for event in events)
        delays = np.asarray([
            event["delay_95_ms"] for event in events
            if event.get("valid") and event.get("delay_95_ms") is not None
        ], dtype=np.float64)
        result["hand_delay95_median_ms"] = percentile(delays, 50)
        result["hand_delay95_max_ms"] = percentile(delays, 100)
        delay50 = np.asarray([
            event["delay_50_ms"] for event in events
            if event.get("valid") and event.get("delay_50_ms") is not None
        ], dtype=np.float64)
        result["hand_delay50_median_ms"] = percentile(delay50, 50)
        if delay50.size:
            lag_seconds = float(np.median(delay50) / 1000.0)
            lag_indices, lag_ages = causal_indices(
                ht["hand_action"], target_h - lag_seconds
            )
            lag_action = arrays["hand_action"][lag_indices]
            lag_error = np.abs(lag_action - aligned["hand_state"])
            lag_error[~np.isfinite(lag_ages)] = np.nan
            lag_main = lag_error[:, MAIN_HAND_DIMS]
            valid_rows = np.isfinite(lag_main).any(axis=1)
            per_frame_max = (
                np.nanmax(lag_main[valid_rows], axis=1)
                if valid_rows.any() else np.asarray([], dtype=np.float64)
            )
            result["hand_lag_compensated_mae"] = float(
                np.nanmean(lag_main)
            )
            result["hand_lag_compensated_p95"] = percentile(
                per_frame_max, 95
            )
            result["hand_lag_compensated_max"] = percentile(
                per_frame_max, 100
            )
        else:
            result["hand_lag_compensated_mae"] = float("nan")
            result["hand_lag_compensated_p95"] = float("nan")
            result["hand_lag_compensated_max"] = float("nan")
        result["hand_events"] = events
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def load_trim_config(path: Path) -> dict[int, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))["trims"]
    return {
        int(key): {
            "start": int(value["start"]),
            "end": int(value["end"]),
            "drop": bool(value.get("drop", False)),
        }
        for key, value in data.items()
    }


def load_manifest(path: Path) -> dict[int, list[int]]:
    exported = json.loads(path.read_text(encoding="utf-8"))["exported"]
    return {
        int(entry["output_episode"]): [
            int(segment["source_episode"]) for segment in entry["source_segments"]
        ]
        for entry in exported
    }


def load_manifest_segments(path: Path) -> dict[int, list[dict[str, int]]]:
    exported = json.loads(path.read_text(encoding="utf-8"))["exported"]
    return {
        int(entry["output_episode"]): [
            {
                "source_episode": int(segment["source_episode"]),
                "start": int(segment["start"]),
                "end": int(segment["end"]),
            }
            for segment in entry["source_segments"]
        ]
        for entry in exported
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def qualify(
    row: dict[str, Any],
    limits: dict[str, tuple[float, float]],
    arm_tracking_p95_rad: float,
) -> tuple[list[str], list[str]]:
    direct_reasons: list[str] = []
    usable_reasons: list[str] = []
    if row.get("error"):
        direct_reasons.append(str(row["error"]))
        usable_reasons.append(str(row["error"]))
    for key, (p99_limit, max_limit) in limits.items():
        if int(row.get("invalid_payload", {}).get(key, 1)):
            direct_reasons.append(f"{key}_invalid_payload")
            usable_reasons.append(f"{key}_invalid_payload")
        if int(row.get(f"{key}_nonmonotonic", 1)):
            direct_reasons.append(f"{key}_nonmonotonic")
            usable_reasons.append(f"{key}_nonmonotonic")
        direct_p99 = float(row.get(f"{key}_direct_p99_ms", math.inf))
        direct_max = float(row.get(f"{key}_direct_max_ms", math.inf))
        if direct_p99 > p99_limit or direct_max > max_limit:
            direct_reasons.append(
                f"{key}_direct_p99_{direct_p99:.1f}_max_{direct_max:.1f}ms"
            )
        p99 = float(row.get(f"{key}_p99_ms", math.inf))
        maximum = float(row.get(f"{key}_max_ms", math.inf))
        if int(row.get(f"{key}_missing_count", 1)) or p99 > p99_limit or maximum > max_limit:
            usable_reasons.append(f"{key}_p99_{p99:.1f}_max_{maximum:.1f}ms")
    event_count = int(row.get("hand_event_count", 0))
    valid_events = int(row.get("hand_valid_event_count", 0))
    if event_count < 2 or valid_events != event_count:
        reason = f"hand_events_{valid_events}_of_{event_count}"
        direct_reasons.append(reason)
        usable_reasons.append(reason)
    arm_p95 = float(row.get("arm_tracking_p95_rad", math.inf))
    if arm_p95 > arm_tracking_p95_rad:
        direct_reasons.append(f"arm_tracking_p95_{arm_p95:.3f}rad")
        usable_reasons.append(f"arm_tracking_p95_{arm_p95:.3f}rad")
    return direct_reasons, usable_reasons


def main() -> None:
    args = parse_args()
    bags = sorted(args.bag_dir.expanduser().resolve().glob("*.bag"), key=lambda path: str(path))
    if len(bags) != args.expected_bags and not args.allow_count_mismatch:
        raise SystemExit(f"Expected {args.expected_bags} bags, found {len(bags)}")
    trims = load_trim_config(args.trim_config)
    if len(trims) != len(bags) and not args.allow_count_mismatch:
        raise SystemExit(f"Trim config has {len(trims)} episodes, bags={len(bags)}")
    selected = list(range(min(len(bags), args.limit))) if args.limit else list(range(len(bags)))
    output_dir = args.output_dir.expanduser().resolve()
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[int, dict[str, Any]] = {}
    jobs = []
    for episode in selected:
        bag = bags[episode]
        trim = trims[episode]
        cache = cache_path(cache_dir, episode, bag)
        if cache.exists() and not args.force:
            item = json.loads(cache.read_text(encoding="utf-8"))
            stat = bag.stat()
            current = (
                item.get("audit_version") == AUDIT_VERSION
                and item.get("bag_size_bytes") == stat.st_size
                and item.get("bag_mtime_ns") == stat.st_mtime_ns
                and item.get("trim_start") == trim["start"]
                and item.get("trim_end") == trim["end"]
                and item.get("sample_drop") == args.sample_drop
                and item.get("camera_stride") == args.camera_stride
                and item.get("event_window_seconds") == args.event_window_seconds
            )
            if current:
                results[episode] = item
                continue
        jobs.append((
            episode, str(bag), trim["start"], trim["end"],
            args.sample_drop, args.camera_stride, args.event_window_seconds,
        ))
    print(
        f"bags={len(bags)} selected={len(selected)} cached={len(results)} "
        f"pending={len(jobs)} workers={args.workers}", flush=True,
    )
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(audit_one, job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            episode = int(row["raw_episode"])
            results[episode] = row
            cache = cache_path(cache_dir, episode, bags[episode])
            temporary = cache.with_suffix(".tmp")
            temporary.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, cache)
            if completed % max(1, args.progress_every) == 0 or completed == len(jobs):
                print(f"completed={completed}/{len(jobs)}", flush=True)

    limits = {
        "arm_state": (args.arm_state_p99_ms, args.arm_state_max_ms),
        "arm_action": (args.arm_action_p99_ms, args.arm_action_max_ms),
        "hand_state": (args.hand_state_p99_ms, args.hand_state_max_ms),
        "hand_action": (args.hand_action_p99_ms, args.hand_action_max_ms),
    }
    direct_raw: set[int] = set()
    usable_raw: set[int] = set()
    rows = []
    for episode in selected:
        row = results[episode]
        direct_reasons, usable_reasons = qualify(
            row, limits, args.arm_tracking_p95_rad
        )
        row["previously_dropped"] = bool(trims[episode]["drop"])
        row["direct"] = not direct_reasons
        row["direct_reasons"] = direct_reasons
        row["usable"] = not usable_reasons
        row["usable_reasons"] = usable_reasons
        if row["direct"]:
            direct_raw.add(episode)
        if row["usable"]:
            usable_raw.add(episode)
        flat = dict(row)
        flat["invalid_payload"] = json.dumps(row.get("invalid_payload", {}))
        flat["direct_reasons"] = ";".join(direct_reasons)
        flat["usable_reasons"] = ";".join(usable_reasons)
        flat["hand_events"] = json.dumps(row.get("hand_events", []))
        rows.append(flat)

    fields = [
        "raw_episode", "bag_name", "bag_path", "trim_start", "trim_end",
        "retained_frame_count", "previously_dropped", "direct", "direct_reasons",
        "usable", "usable_reasons", "error", "invalid_payload",
        "arm_tracking_mae_rad", "arm_tracking_p95_rad", "arm_tracking_max_rad",
        "hand_tracking_mae", "hand_tracking_p95", "hand_tracking_max",
        "hand_lag_compensated_mae", "hand_lag_compensated_p95",
        "hand_lag_compensated_max", "hand_delay50_median_ms",
        "hand_aux_tracking_mae", "hand_event_count", "hand_valid_event_count",
        "hand_raw_threshold_edge_count", "hand_short_threshold_edge_count",
        "hand_delay95_median_ms", "hand_delay95_max_ms", "hand_events",
    ]
    for key in TOPICS:
        fields.extend([
            f"{key}_count", f"{key}_missing_count", f"{key}_p50_ms",
            f"{key}_p95_ms", f"{key}_p99_ms", f"{key}_max_ms",
            f"{key}_direct_p50_ms", f"{key}_direct_p95_ms",
            f"{key}_direct_p99_ms", f"{key}_direct_max_ms",
            f"{key}_nonmonotonic",
        ])
    write_csv(output_dir / "task3_lowdim_alignment.csv", rows, fields)
    for name, values in (("direct_raw_episodes.txt", direct_raw), ("usable_raw_episodes.txt", usable_raw)):
        ordered = sorted(values)
        (output_dir / name).write_text(
            "\n".join(map(str, ordered)) + ("\n" if ordered else ""), encoding="utf-8"
        )

    mapping_report: dict[str, Any] = {}
    if not args.limit and len(selected) == len(bags):
        kept_raw = [episode for episode in range(len(bags)) if not trims[episode]["drop"]]
        trimmed_to_raw = {index: raw for index, raw in enumerate(kept_raw)}
        super_manifest = load_manifest_segments(args.supertrim_manifest)
        super_cache_dir = output_dir / "supertrimmed_cache"
        super_cache_dir.mkdir(exist_ok=True)
        super_results: dict[int, dict[str, Any]] = {}
        super_jobs = []
        super_to_raw: dict[int, set[int]] = {}
        for output, segments in super_manifest.items():
            if len(segments) != 1:
                raise ValueError(
                    f"TASK3 output {output} has {len(segments)} source segments; "
                    "the exact audit expects merge_group_size=1"
                )
            segment = segments[0]
            raw = trimmed_to_raw[segment["source_episode"]]
            super_to_raw[output] = {raw}
            start = trims[raw]["start"] + segment["start"]
            end = trims[raw]["start"] + segment["end"]
            bag = bags[raw]
            cache = cache_path(super_cache_dir, output, bag)
            if cache.exists() and not args.force:
                item = json.loads(cache.read_text(encoding="utf-8"))
                stat = bag.stat()
                current = (
                    item.get("audit_version") == AUDIT_VERSION
                    and item.get("bag_size_bytes") == stat.st_size
                    and item.get("bag_mtime_ns") == stat.st_mtime_ns
                    and item.get("trim_start") == start
                    and item.get("trim_end") == end
                    and item.get("sample_drop") == args.sample_drop
                    and item.get("camera_stride") == args.camera_stride
                    and item.get("event_window_seconds") == args.event_window_seconds
                )
                if current:
                    super_results[output] = item
                    continue
            super_jobs.append((
                output, str(bag), start, end, args.sample_drop,
                args.camera_stride, args.event_window_seconds,
            ))
        print(
            f"exact supertrimmed audit: cached={len(super_results)} "
            f"pending={len(super_jobs)}", flush=True,
        )
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(audit_one, job): job for job in super_jobs}
            for completed, future in enumerate(as_completed(futures), 1):
                item = future.result()
                output = int(item["raw_episode"])
                super_results[output] = item
                raw = next(iter(super_to_raw[output]))
                cache = cache_path(super_cache_dir, output, bags[raw])
                temporary = cache.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(item, ensure_ascii=False), encoding="utf-8"
                )
                os.replace(temporary, cache)
                if completed % max(1, args.progress_every) == 0 or completed == len(super_jobs):
                    print(f"exact supertrimmed completed={completed}/{len(super_jobs)}", flush=True)
        direct_super: list[int] = []
        usable_super: list[int] = []
        super_reason_map: dict[int, list[str]] = {}
        for output in sorted(super_results):
            direct_reasons, usable_reasons = qualify(
                super_results[output], limits, args.arm_tracking_p95_rad
            )
            super_results[output]["direct_reasons"] = direct_reasons
            super_results[output]["usable_reasons"] = usable_reasons
            super_reason_map[output] = usable_reasons
            if not direct_reasons:
                direct_super.append(output)
            if not usable_reasons:
                usable_super.append(output)
        rejected_super = sorted(set(super_to_raw) - set(usable_super))
        for filename, values in (
            ("direct_task3_supertrimmed_200_episodes.txt", direct_super),
            ("usable_task3_supertrimmed_200_episodes.txt", usable_super),
            ("rejected_task3_supertrimmed_200_episodes.txt", rejected_super),
        ):
            (output_dir / filename).write_text(
                "\n".join(map(str, values)) + ("\n" if values else ""), encoding="utf-8"
            )
        trace_rows = []
        for output in sorted(super_to_raw):
            raw_sources = sorted(super_to_raw[output])
            trace_rows.append({
                "output_episode": output,
                "raw_sources": ",".join(map(str, raw_sources)),
                "usable": output in usable_super,
                "raw_bag_names": "|".join(bags[raw].name for raw in raw_sources),
                "reasons": ";".join(super_reason_map[output]),
            })
        write_csv(
            output_dir / "task3_supertrimmed_200_traceability.csv",
            trace_rows,
            ["output_episode", "raw_sources", "usable", "raw_bag_names", "reasons"],
        )
        mapping_report = {
            "trim_config_kept_count": len(kept_raw),
            "supertrimmed_total": len(super_to_raw),
            "direct_supertrimmed_count": len(direct_super),
            "usable_supertrimmed_count": len(usable_super),
            "rejected_supertrimmed_count": len(rejected_super),
            "supertrimmed_qualification_scope": "exact nested trim ranges",
        }
    summary = {
        "bag_dir": str(args.bag_dir),
        "selected_count": len(selected),
        "direct_raw_count": len(direct_raw),
        "usable_raw_count": len(usable_raw),
        "limits": {
            key: {"p99_ms": value[0], "max_ms": value[1]}
            for key, value in limits.items()
        },
        "tracking_limits": {
            "arm_p95_rad": args.arm_tracking_p95_rad,
            "hand_lag_compensated_p95_advisory": args.hand_tracking_p95,
        },
        **mapping_report,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
