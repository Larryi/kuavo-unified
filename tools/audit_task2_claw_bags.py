#!/usr/bin/env python3
"""Audit TASK2 claw timing and map reliable bags to existing LeRobot episodes.

The original converter assigns raw episode indices by lexicographically sorting
``*.bag`` in one directory.  This tool reproduces that ordering, audits the
source timestamps of the claw command/state streams, and follows the existing
trim manifests without creating another dataset.

Only these small topics are deserialized; RGB/depth payloads are never decoded:

* /leju_claw_command
* /leju_claw_state
* /cam_h/color/camera_info

The scan is resumable.  One compact JSON cache file is written per bag, then
CSV/JSON/text reports are rebuilt from the cache after every complete run.
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


COMMAND_TOPIC = "/leju_claw_command"
STATE_TOPIC = "/leju_claw_state"
CAMERA_INFO_TOPIC = "/cam_h/color/camera_info"
AUDIT_VERSION = 2

DEFAULT_BAG_DIR = Path("/mnt/pqssd/task2_chengzhong")
DEFAULT_OUTPUT_DIR = Path("/mnt/pqssd/task2_claw_quality")
DEFAULT_BASE_MANIFEST = Path(
    "/mnt/pqssd/Real_PQ_3.0/TASK2_SZ/lerobot_trimmed/trim_manifest.json"
)
DEFAULT_DEPTH200_MANIFEST = Path(
    "/mnt/pqssd/Real_PQ_3.0/TASK2_SZ_Depth/"
    "lerobot_supertrimmed_200/trim_manifest.json"
)
DEFAULT_GOOD89_MANIFEST = Path(
    "/mnt/pqssd/Real_PQ_3.0/TASK2_SZ_Good/"
    "lerobot_supertrimmed_89/trim_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-dir", type=Path, default=DEFAULT_BAG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Parallel bag readers (16 is a balanced default for a 24-thread CPU and local SSD)",
    )
    parser.add_argument("--expected-bags", type=int, default=1000)
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Allow a partial/test directory. Mapping reports are disabled.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--force", action="store_true", help="Ignore cached audits")
    parser.add_argument(
        "--hands", choices=("both", "right", "left"), default="both",
        help="Only score command/state events for the controlled hand(s).",
    )
    parser.add_argument(
        "--skip-mapping", action="store_true",
        help="Audit raw bags only; do not apply the TASK2-specific manifest chain.",
    )
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--depth200-manifest", type=Path, default=DEFAULT_DEPTH200_MANIFEST)
    parser.add_argument("--good89-manifest", type=Path, default=DEFAULT_GOOD89_MANIFEST)
    parser.add_argument("--a-coverage", type=float, default=0.98)
    parser.add_argument("--a-age-p95-ms", type=float, default=100.0)
    parser.add_argument("--a-final-backlog-ms", type=float, default=200.0)
    parser.add_argument("--b-coverage", type=float, default=0.95)
    parser.add_argument("--b-age-p95-ms", type=float, default=300.0)
    parser.add_argument("--b-final-backlog-ms", type=float, default=500.0)
    parser.add_argument(
        "--event-window-seconds",
        type=float,
        default=1.2,
        help="State response window after each command edge",
    )
    return parser.parse_args()


def message_stamp(msg: Any) -> float:
    if not hasattr(msg, "header") or not hasattr(msg.header, "stamp"):
        return float("nan")
    value = float(msg.header.stamp.to_sec())
    return value if value > 0 else float("nan")


def percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, q)) if len(finite) else float("nan")


def safe_rate(count: int, span: float) -> float:
    return float(count / span) if count > 1 and span > 0 else float("nan")


def first_crossing(
    times: np.ndarray,
    values: np.ndarray,
    start: float,
    end: float,
    threshold: float,
    direction: int,
) -> float | None:
    ids = np.flatnonzero((times >= start) & (times <= end))
    if direction > 0:
        ids = ids[values[ids] >= threshold]
    else:
        ids = ids[values[ids] <= threshold]
    return None if not len(ids) else float(times[ids[0]])


def command_events(
    command_header: np.ndarray,
    command_values: list[np.ndarray],
    state_header: np.ndarray,
    state_values: list[np.ndarray],
    camera_end: float,
    event_window: float,
    hands: str = "both",
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    hand_pairs = [(0, "left"), (1, "right")]
    if hands != "both":
        hand_pairs = [pair for pair in hand_pairs if pair[1] == hands]
    for hand_index, hand in hand_pairs:
        command = command_values[hand_index]
        closed = command > 50.0
        edge_ids = np.flatnonzero(closed[1:] != closed[:-1]) + 1
        for ordinal, edge_index in enumerate(edge_ids):
            start = float(command_header[edge_index])
            next_edge = (
                float(command_header[edge_ids[ordinal + 1]])
                if ordinal + 1 < len(edge_ids)
                else float(camera_end)
            )
            end = min(start + event_window, next_edge - 1e-6)
            direction = 1 if bool(closed[edge_index]) else -1
            event: dict[str, Any] = {
                "hand": hand,
                "kind": "close" if direction > 0 else "open",
                "command_header_time": start,
                "window_end": end,
                "valid": False,
                "reason": "",
            }
            if not np.isfinite(start) or end - start < 0.15:
                event["reason"] = "command_hold_too_short"
                events.append(event)
                continue
            if not len(state_header) or state_header[0] > start - 0.05:
                event["reason"] = "missing_pre_state"
                events.append(event)
                continue
            if state_header[-1] < end - 0.05:
                event["reason"] = "state_header_does_not_cover_response_window"
                events.append(event)
                continue
            state = state_values[hand_index]
            before = state[(state_header >= start - 0.08) & (state_header < start)]
            tail = state[(state_header >= end - 0.10) & (state_header <= end)]
            if not len(before) or not len(tail):
                event["reason"] = "insufficient_state_samples"
                events.append(event)
                continue
            baseline = float(np.median(before))
            final = float(np.median(tail))
            delta = final - baseline
            event.update({"baseline": baseline, "final": final, "delta": delta})
            if direction * delta < 10.0:
                event["reason"] = "state_did_not_move_in_command_direction"
                events.append(event)
                continue
            delays: dict[str, float | None] = {}
            for fraction in (0.05, 0.50, 0.95):
                threshold = baseline + fraction * delta
                hit = first_crossing(
                    state_header, state, start, end, threshold, direction
                )
                delays[f"delay_{int(fraction * 100):02d}_ms"] = (
                    None if hit is None else (hit - start) * 1000.0
                )
            delay95 = delays["delay_95_ms"]
            max_delay_ms = 800.0 if direction > 0 else 1000.0
            if delay95 is None or not 0.0 <= delay95 <= max_delay_ms:
                event["reason"] = "response_delay_out_of_range"
                event.update(delays)
                events.append(event)
                continue
            event.update(delays)
            event["valid"] = True
            event["reason"] = "ok"
            events.append(event)
    events.sort(key=lambda item: item["command_header_time"])
    return events


def reliable_prefix_seconds(
    state_bag: np.ndarray,
    state_header: np.ndarray,
    camera_bag_start: float,
    camera_bag_end: float,
    max_age_seconds: float,
) -> float:
    """Return the prefix before two consecutive 250 ms stale bins."""
    if not len(state_bag) or camera_bag_end <= camera_bag_start:
        return 0.0
    ages = state_bag - state_header
    bin_width = 0.25
    bin_ids = np.floor((state_bag - camera_bag_start) / bin_width).astype(int)
    last_bin = max(0, int(math.ceil((camera_bag_end - camera_bag_start) / bin_width)))
    medians = np.full(last_bin + 1, np.nan, dtype=np.float64)
    for bin_id in np.unique(bin_ids):
        if 0 <= bin_id < len(medians):
            medians[bin_id] = np.median(ages[bin_ids == bin_id])
    bad = np.isfinite(medians) & (medians > max_age_seconds)
    for index in range(len(bad) - 1):
        if bad[index] and bad[index + 1]:
            return max(0.0, index * bin_width)
    return max(0.0, camera_bag_end - camera_bag_start)


def classify(result: dict[str, Any], thresholds: dict[str, float]) -> tuple[str, str]:
    if result.get("error"):
        return "D", result["error"]
    if result["missing_topics"]:
        return "D", "missing_topics"
    if result["state_header_nonmonotonic"] > 0:
        return "D", "nonmonotonic_state_header"
    if result["command_header_nonmonotonic"] > 0:
        return "D", "nonmonotonic_command_header"
    total = int(result["event_count"])
    valid = int(result["valid_event_count"])
    events_ok = total >= 2 and valid == total
    coverage = float(result["state_camera_header_coverage"])
    age_p95 = float(result["state_age_p95_ms"])
    final_backlog = float(result["state_final_backlog_ms"])
    if (
        events_ok
        and coverage >= thresholds["a_coverage"]
        and age_p95 <= thresholds["a_age_p95_ms"]
        and final_backlog <= thresholds["a_final_backlog_ms"]
    ):
        return "A", "all_strict_checks_passed"
    if (
        events_ok
        and coverage >= thresholds["b_coverage"]
        and age_p95 <= thresholds["b_age_p95_ms"]
        and final_backlog <= thresholds["b_final_backlog_ms"]
    ):
        return "B", "all_usable_checks_passed"
    if result["reliable_prefix_seconds"] >= 2.0 and valid > 0:
        reasons = []
        if coverage < thresholds["b_coverage"]:
            reasons.append("incomplete_header_coverage")
        if age_p95 > thresholds["b_age_p95_ms"]:
            reasons.append("state_backlog")
        if valid < total:
            reasons.append("unverified_command_events")
        return "C", "+".join(reasons) or "only_prefix_is_reliable"
    return "D", "no_reliable_complete_task_interval"


def audit_bag(job: tuple[int, str, dict[str, float], float, str]) -> dict[str, Any]:
    raw_episode, path_text, thresholds, event_window, hands = job
    path = Path(path_text)
    result: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "event_window_seconds": event_window,
        "hands": hands,
        "raw_episode": raw_episode,
        "bag_path": str(path),
        "bag_name": path.name,
        "error": "",
    }
    try:
        stat = path.stat()
        result["bag_size_bytes"] = stat.st_size
        result["bag_mtime_ns"] = stat.st_mtime_ns
        bag_times: dict[str, list[float]] = {
            "command": [], "state": [], "camera": []
        }
        header_times: dict[str, list[float]] = {
            "command": [], "state": [], "camera": []
        }
        command_values = [[], []]
        state_values = [[], []]
        with rosbag.Bag(str(path), "r") as bag:
            result["bag_start"] = float(bag.get_start_time())
            result["bag_end"] = float(bag.get_end_time())
            for topic, msg, bag_time in bag.read_messages(
                topics=[COMMAND_TOPIC, STATE_TOPIC, CAMERA_INFO_TOPIC]
            ):
                if topic == COMMAND_TOPIC:
                    key = "command"
                elif topic == STATE_TOPIC:
                    key = "state"
                else:
                    key = "camera"
                bag_times[key].append(float(bag_time.to_sec()))
                header_times[key].append(message_stamp(msg))
                if key in ("command", "state"):
                    position = list(msg.data.position)
                    target = command_values if key == "command" else state_values
                    for hand_index in range(2):
                        target[hand_index].append(
                            float(position[hand_index])
                            if hand_index < len(position)
                            else float("nan")
                        )
        missing = [
            topic
            for key, topic in (
                ("command", COMMAND_TOPIC),
                ("state", STATE_TOPIC),
                ("camera", CAMERA_INFO_TOPIC),
            )
            if not bag_times[key]
        ]
        result["missing_topics"] = missing
        if missing:
            result["grade"], result["grade_reason"] = classify(result, thresholds)
            result["events"] = []
            return result

        bt = {key: np.asarray(value, dtype=np.float64) for key, value in bag_times.items()}
        ht = {key: np.asarray(value, dtype=np.float64) for key, value in header_times.items()}
        command_np = [np.asarray(value, dtype=np.float64) for value in command_values]
        state_np = [np.asarray(value, dtype=np.float64) for value in state_values]
        finite_headers = all(np.isfinite(ht[key]).all() for key in ht)
        if not finite_headers:
            result["error"] = "missing_or_invalid_header_stamp"
            result["events"] = []
            result["grade"], result["grade_reason"] = classify(result, thresholds)
            return result

        camera_span = float(ht["camera"][-1] - ht["camera"][0])
        overlap = max(
            0.0,
            min(float(ht["state"][-1]), float(ht["camera"][-1]))
            - max(float(ht["state"][0]), float(ht["camera"][0])),
        )
        state_age = bt["state"] - ht["state"]
        command_age = bt["command"] - ht["command"]
        events = command_events(
            ht["command"], command_np, ht["state"], state_np,
            float(ht["camera"][-1]), event_window, hands,
        )
        result.update(
            {
                "bag_duration": float(result["bag_end"] - result["bag_start"]),
                "command_count": int(len(bt["command"])),
                "state_count": int(len(bt["state"])),
                "camera_info_count": int(len(bt["camera"])),
                "command_header_span": float(ht["command"][-1] - ht["command"][0]),
                "state_header_span": float(ht["state"][-1] - ht["state"][0]),
                "camera_header_span": camera_span,
                "command_receive_hz": safe_rate(
                    len(bt["command"]), float(bt["command"][-1] - bt["command"][0])
                ),
                "state_receive_hz": safe_rate(
                    len(bt["state"]), float(bt["state"][-1] - bt["state"][0])
                ),
                "command_source_hz": safe_rate(
                    len(ht["command"]), float(ht["command"][-1] - ht["command"][0])
                ),
                "state_source_hz": safe_rate(
                    len(ht["state"]), float(ht["state"][-1] - ht["state"][0])
                ),
                "state_camera_header_coverage": overlap / camera_span if camera_span > 0 else 0.0,
                "state_age_p50_ms": percentile(state_age, 50) * 1000.0,
                "state_age_p95_ms": percentile(state_age, 95) * 1000.0,
                "state_age_max_ms": percentile(state_age, 100) * 1000.0,
                "state_final_backlog_ms": float(state_age[-1] * 1000.0),
                "command_age_p95_ms": percentile(command_age, 95) * 1000.0,
                "command_final_backlog_ms": float(command_age[-1] * 1000.0),
                "state_header_nonmonotonic": int(np.count_nonzero(np.diff(ht["state"]) <= 0)),
                "command_header_nonmonotonic": int(np.count_nonzero(np.diff(ht["command"]) <= 0)),
                "event_count": len(events),
                "valid_event_count": sum(bool(event["valid"]) for event in events),
                "reliable_prefix_seconds": reliable_prefix_seconds(
                    bt["state"], ht["state"], float(bt["camera"][0]),
                    float(bt["camera"][-1]), thresholds["b_age_p95_ms"] / 1000.0,
                ),
                "events": events,
            }
        )
        result["grade"], result["grade_reason"] = classify(result, thresholds)
        return result
    except Exception as exc:  # keep a 1000-bag batch running after one bad file
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["missing_topics"] = []
        result["events"] = []
        result["grade"] = "D"
        result["grade_reason"] = "audit_exception"
        return result


def cache_path(cache_dir: Path, raw_episode: int, bag_path: Path) -> Path:
    digest = hashlib.sha1(str(bag_path).encode("utf-8")).hexdigest()[:10]
    return cache_dir / f"episode_{raw_episode:04d}_{digest}.json"


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "exported" not in data:
        raise ValueError(f"Manifest has no exported list: {path}")
    return data


def sources_by_output(manifest: dict[str, Any]) -> dict[int, list[int]]:
    return {
        int(entry["output_episode"]): [
            int(segment["source_episode"])
            for segment in entry["source_segments"]
        ]
        for entry in manifest["exported"]
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def trace_raw_sources(
    output_episode: int,
    mapping: dict[int, list[int]],
    parent_sources: dict[int, set[int]],
) -> set[int]:
    result: set[int] = set()
    for source in mapping[output_episode]:
        result.update(parent_sources[source])
    return result


def emit_mapping_reports(
    output_dir: Path,
    results: list[dict[str, Any]],
    base_manifest_path: Path,
    depth200_manifest_path: Path,
    good89_manifest_path: Path,
) -> dict[str, Any]:
    base = sources_by_output(load_manifest(base_manifest_path))
    depth200 = sources_by_output(load_manifest(depth200_manifest_path))
    good89 = sources_by_output(load_manifest(good89_manifest_path))
    raw_sources = {int(row["raw_episode"]): {int(row["raw_episode"])} for row in results}
    base_sources = {
        output: trace_raw_sources(output, base, raw_sources) for output in base
    }
    depth200_sources = {
        output: trace_raw_sources(output, depth200, base_sources) for output in depth200
    }
    good89_sources = {
        output: trace_raw_sources(output, good89, depth200_sources) for output in good89
    }
    strict_raw = {
        int(row["raw_episode"]) for row in results if row["grade"] in ("A", "B")
    }

    dataset_sources = {
        "/mnt/pqssd/Real_PQ_3.0/TASK2_SZ_Depth/lerobot": raw_sources,
        "/mnt/pqssd/Real_PQ_3.0/TASK2_SZ/lerobot_trimmed": base_sources,
        "/mnt/pqssd/Real_PQ_3.0/TASK2_SZ_Depth/lerobot_supertrimmed_200": depth200_sources,
        "/mnt/pqssd/Real_PQ_3.0/TASK2_SZ_Good/lerobot_supertrimmed_89": good89_sources,
    }
    report: dict[str, Any] = {
        "qualification_rule": (
            "A/B only; every raw source bag of a merged episode must pass"
        ),
        "datasets": {},
    }
    for dataset_path, source_map in dataset_sources.items():
        qualified = sorted(
            episode
            for episode, sources in source_map.items()
            if sources and sources.issubset(strict_raw)
        )
        key = Path(dataset_path).name
        if key == "lerobot":
            key = "TASK2_SZ_Depth_lerobot"
        elif key == "lerobot_trimmed":
            key = "TASK2_SZ_lerobot_trimmed"
        elif key == "lerobot_supertrimmed_200":
            key = "TASK2_SZ_Depth_lerobot_supertrimmed_200"
        else:
            key = "TASK2_SZ_Good_lerobot_supertrimmed_89"
        text_path = output_dir / f"qualified_{key}.txt"
        text_path.write_text(
            "\n".join(str(episode) for episode in qualified) + ("\n" if qualified else ""),
            encoding="utf-8",
        )
        report["datasets"][key] = {
            "path": dataset_path,
            "total_episodes": len(source_map),
            "qualified_count": len(qualified),
            "qualified_episodes": qualified,
            "text_file": str(text_path),
        }

    raw_to_base = {
        source: output for output, sources in base_sources.items() for source in sources
    }
    base_to_depth200 = {
        source: output for output, sources in depth200.items() for source in sources
    }
    depth200_to_good89 = {
        source: output for output, sources in good89.items() for source in sources
    }
    mapping_rows = []
    for row in results:
        raw = int(row["raw_episode"])
        base_episode = raw_to_base.get(raw)
        depth_episode = (
            base_to_depth200.get(base_episode) if base_episode is not None else None
        )
        good_episode = (
            depth200_to_good89.get(depth_episode) if depth_episode is not None else None
        )
        mapping_rows.append(
            {
                "raw_episode": raw,
                "bag_name": row["bag_name"],
                "grade": row["grade"],
                "strict_pass": row["grade"] in ("A", "B"),
                "TASK2_SZ_episode": base_episode,
                "TASK2_SZ_Depth_200_episode": depth_episode,
                "TASK2_SZ_Good_89_episode": good_episode,
            }
        )
    write_csv(
        output_dir / "episode_traceability.csv",
        mapping_rows,
        [
            "raw_episode", "bag_name", "grade", "strict_pass",
            "TASK2_SZ_episode", "TASK2_SZ_Depth_200_episode",
            "TASK2_SZ_Good_89_episode",
        ],
    )
    (output_dir / "qualified_existing_episodes.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    bag_dir = args.bag_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    bag_paths = sorted(bag_dir.glob("*.bag"), key=lambda path: str(path))
    if len(bag_paths) != args.expected_bags and not args.allow_count_mismatch:
        raise SystemExit(
            f"Refusing unsafe episode mapping: found {len(bag_paths)} bags, "
            f"expected {args.expected_bags}. Wait for copying to finish or use "
            "--allow-count-mismatch for a scan-only test."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    thresholds = {
        "a_coverage": args.a_coverage,
        "a_age_p95_ms": args.a_age_p95_ms,
        "a_final_backlog_ms": args.a_final_backlog_ms,
        "b_coverage": args.b_coverage,
        "b_age_p95_ms": args.b_age_p95_ms,
        "b_final_backlog_ms": args.b_final_backlog_ms,
    }

    results_by_episode: dict[int, dict[str, Any]] = {}
    pending = []
    for raw_episode, bag_path in enumerate(bag_paths):
        path = cache_path(cache_dir, raw_episode, bag_path)
        if path.exists() and not args.force:
            cached = json.loads(path.read_text(encoding="utf-8"))
            stat = bag_path.stat()
            cache_is_current = (
                cached.get("audit_version") == AUDIT_VERSION
                and cached.get("bag_path") == str(bag_path)
                and cached.get("bag_size_bytes") == stat.st_size
                and cached.get("bag_mtime_ns") == stat.st_mtime_ns
                and cached.get("event_window_seconds") == args.event_window_seconds
                and cached.get("hands") == args.hands
            )
            if cache_is_current:
                # Threshold-only changes do not require reading the bag again.
                cached["grade"], cached["grade_reason"] = classify(
                    cached, thresholds
                )
                results_by_episode[raw_episode] = cached
                continue
        pending.append((
            raw_episode, str(bag_path), thresholds,
            args.event_window_seconds, args.hands,
        ))

    print(
        f"bags={len(bag_paths)} cached={len(results_by_episode)} "
        f"pending={len(pending)} workers={args.workers}",
        flush=True,
    )
    if pending:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(audit_bag, job): job for job in pending}
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                raw_episode = int(result["raw_episode"])
                results_by_episode[raw_episode] = result
                path = cache_path(cache_dir, raw_episode, Path(result["bag_path"]))
                temporary = path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(result, ensure_ascii=False), encoding="utf-8"
                )
                os.replace(temporary, path)
                if completed % max(1, args.progress_every) == 0 or completed == len(pending):
                    grades = {grade: 0 for grade in "ABCD"}
                    for item in results_by_episode.values():
                        grades[item.get("grade", "D")] += 1
                    print(
                        f"completed={completed}/{len(pending)} total_cached={len(results_by_episode)} "
                        f"grades={grades}",
                        flush=True,
                    )

    results = [results_by_episode[index] for index in sorted(results_by_episode)]
    summary_fields = [
        "raw_episode", "bag_name", "bag_path", "grade", "grade_reason", "error",
        "bag_duration", "command_count", "state_count", "camera_info_count",
        "state_header_span", "camera_header_span", "state_camera_header_coverage",
        "command_receive_hz", "state_receive_hz", "command_source_hz", "state_source_hz",
        "state_age_p50_ms", "state_age_p95_ms", "state_age_max_ms",
        "state_final_backlog_ms", "command_age_p95_ms", "command_final_backlog_ms",
        "state_header_nonmonotonic", "command_header_nonmonotonic",
        "event_count", "valid_event_count", "reliable_prefix_seconds",
    ]
    write_csv(output_dir / "bag_quality.csv", results, summary_fields)
    with (output_dir / "bag_events.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps({
                "raw_episode": result["raw_episode"],
                "bag_name": result["bag_name"],
                "grade": result["grade"],
                "events": result.get("events", []),
            }, ensure_ascii=False) + "\n")

    grade_counts = {grade: 0 for grade in "ABCD"}
    for result in results:
        grade_counts[result.get("grade", "D")] += 1
    final_report: dict[str, Any] = {
        "bag_dir": str(bag_dir),
        "bag_count": len(bag_paths),
        "hands": args.hands,
        "thresholds": thresholds,
        "grade_counts": grade_counts,
        "strict_qualified_raw_count": grade_counts["A"] + grade_counts["B"],
    }
    if len(bag_paths) == args.expected_bags and not args.skip_mapping:
        final_report["existing_datasets"] = emit_mapping_reports(
            output_dir, results, args.base_manifest, args.depth200_manifest,
            args.good89_manifest,
        )["datasets"]
    else:
        final_report["mapping_skipped"] = (
            "explicitly disabled" if args.skip_mapping else "bag count mismatch"
        )
    (output_dir / "summary.json").write_text(
        json.dumps(final_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(final_report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
