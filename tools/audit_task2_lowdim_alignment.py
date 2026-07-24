#!/usr/bin/env python3
"""Verify Header-based repairability of TASK2 low-dimensional DP data.

This audit can target all raw single-sleeve episodes or only the sources that
feed the current Good89 dataset. It reproduces the 10 Hz head-camera timeline,
applies each source segment's recorded trim range, then asks whether every retained frame
can causally obtain fresh data from all four DP streams:

* observation arm state: /sensors_data_raw
* arm action: /kuavo_arm_traj
* gripper state: /leju_claw_state
* gripper action: /leju_claw_command

RGB payloads are read only to obtain the exact master-frame Header used by the
converter; images are never decoded. Results are resumable and no dataset is
modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import rosbag


CAMERA_TOPIC = "/cam_h/color/image_raw/compressed"
TOPICS = {
    "joint_state": "/sensors_data_raw",
    "arm_action": "/kuavo_arm_traj",
    "claw_state": "/leju_claw_state",
    "claw_action": "/leju_claw_command",
}
AUDIT_VERSION = 3

DEFAULT_BAG_QUALITY = Path("/mnt/pqssd/task2_claw_quality/bag_quality.csv")
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
DEFAULT_TRIM_CONFIG = Path(
    "/mnt/pqssd/Real_PQ_3.0/TASK2_SZ_Depth/lerobot/trim_config.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-quality", type=Path, default=DEFAULT_BAG_QUALITY)
    parser.add_argument("--source-pool", choices=("all", "good89"), default="all")
    parser.add_argument("--trim-config", type=Path, default=DEFAULT_TRIM_CONFIG)
    parser.add_argument(
        "--source-root", type=Path, default=None,
        help="Raw LeRobot root used for [0,length) ranges when trim-config is absent.",
    )
    parser.add_argument(
        "--kept-manifest", type=Path, default=None,
        help="Optional first-stage manifest; raw episodes absent from it count as previously dropped.",
    )
    parser.add_argument(
        "--manifest-chain", type=Path, nargs="*", default=None,
        help="Selection manifests from raw outward, used to report qualified existing episodes.",
    )
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--depth200-manifest", type=Path, default=DEFAULT_DEPTH200_MANIFEST)
    parser.add_argument("--good89-manifest", type=Path, default=DEFAULT_GOOD89_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/pqssd/task2_lowdim_quality"))
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Parallel bag readers (16 is a balanced default for a 24-thread CPU and local SSD)",
    )
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-episodes", default="0,1,2",
        help="Comma-separated raw episodes excluded before auditing",
    )
    parser.add_argument("--sample-drop", type=int, default=10)
    parser.add_argument("--camera-stride", type=int, default=3)
    parser.add_argument("--joint-state-p99-ms", type=float, default=20.0)
    parser.add_argument("--joint-state-max-ms", type=float, default=100.0)
    parser.add_argument("--arm-action-p99-ms", type=float, default=50.0)
    parser.add_argument("--arm-action-max-ms", type=float, default=150.0)
    parser.add_argument("--claw-state-p99-ms", type=float, default=20.0)
    parser.add_argument("--claw-state-max-ms", type=float, default=100.0)
    parser.add_argument("--claw-action-p99-ms", type=float, default=50.0)
    parser.add_argument("--claw-action-max-ms", type=float, default=150.0)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[int, list[dict[str, int]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(entry["output_episode"]): [
            {
                "source_episode": int(segment["source_episode"]),
                "start": int(segment["start"]),
                "end": int(segment["end"]),
            }
            for segment in entry["source_segments"]
        ]
        for entry in data["exported"]
    }


def good89_source_segments(
    base_path: Path, depth200_path: Path, good89_path: Path
) -> tuple[dict[int, dict[str, int]], dict[int, list[int]]]:
    base = load_manifest(base_path)
    depth200 = load_manifest(depth200_path)
    good89 = load_manifest(good89_path)
    raw_segments: dict[int, dict[str, int]] = {}
    good_sources: dict[int, list[int]] = {}
    for good_episode, depth_segments in good89.items():
        raw: list[int] = []
        for depth_segment in depth_segments:
            depth_episode = depth_segment["source_episode"]
            for base_segment_ref in depth200[depth_episode]:
                base_episode = base_segment_ref["source_episode"]
                for segment in base[base_episode]:
                    source = segment["source_episode"]
                    raw_segments[source] = {
                        "start": segment["start"], "end": segment["end"]
                    }
                    raw.append(source)
        good_sources[good_episode] = raw
    return raw_segments, good_sources


def all_source_segments(trim_config_path: Path) -> dict[int, dict[str, Any]]:
    data = json.loads(trim_config_path.read_text(encoding="utf-8"))
    return {
        int(episode): {
            "start": int(config["start"]),
            "end": int(config["end"]),
            "previously_dropped": bool(config.get("drop", False)),
        }
        for episode, config in data["trims"].items()
    }


def source_segments_from_lerobot(source_root: Path) -> dict[int, dict[str, Any]]:
    episode_files = sorted((source_root / "meta/episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No episode metadata under {source_root}")
    segments: dict[int, dict[str, Any]] = {}
    for path in episode_files:
        table = pq.read_table(path, columns=["episode_index", "length"])
        for episode, length in zip(
            table["episode_index"].to_pylist(), table["length"].to_pylist()
        ):
            segments[int(episode)] = {
                "start": 0, "end": int(length), "previously_dropped": False,
            }
    return segments


def manifest_source_episodes(path: Path) -> set[int]:
    manifest = load_manifest(path)
    return {
        int(segment["source_episode"])
        for segments in manifest.values()
        for segment in segments
    }


def trace_manifest_chain(
    paths: list[Path], qualified_raw: set[int]
) -> list[dict[str, Any]]:
    previous_sources: dict[int, set[int]] = {
        episode: {episode} for episode in qualified_raw
    }
    reports: list[dict[str, Any]] = []
    for path in paths:
        manifest = load_manifest(path)
        output_sources: dict[int, set[int]] = {}
        all_output_ids: set[int] = set()
        for output_episode, segments in manifest.items():
            all_output_ids.add(int(output_episode))
            raw: set[int] = set()
            valid = True
            for segment in segments:
                source = int(segment["source_episode"])
                if source not in previous_sources:
                    valid = False
                    break
                raw.update(previous_sources[source])
            if valid and raw:
                output_sources[int(output_episode)] = raw
        reports.append({
            "name": path.parent.name,
            "path": str(path),
            "total": len(all_output_ids),
            "qualified": sorted(output_sources),
            "qualified_raw_sources": sorted(
                set().union(*output_sources.values()) if output_sources else set()
            ),
        })
        previous_sources = output_sources
    return reports


def header_stamp(msg: Any) -> float:
    if not hasattr(msg, "header") or not hasattr(msg.header, "stamp"):
        return float("nan")
    return float(msg.header.stamp.to_sec())


def payload_is_valid(key: str, msg: Any) -> bool:
    try:
        if key == "joint_state":
            values = list(msg.joint_data.joint_q)
            minimum = 26
        elif key == "arm_action":
            values = list(msg.position)
            minimum = 14
        else:
            values = list(msg.data.position)
            minimum = 2
        return len(values) >= minimum and bool(np.isfinite(values).all())
    except Exception:
        return False


def causal_stats(source_header: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    finite = source_header[np.isfinite(source_header)]
    if not len(finite) or not len(targets):
        return {
            "target_count": int(len(targets)), "missing_count": int(len(targets)),
            "p50_ms": float("nan"), "p95_ms": float("nan"),
            "p99_ms": float("nan"), "max_ms": float("nan"),
            "nearest_p99_ms": float("nan"), "nearest_max_ms": float("nan"),
            "bracket_p99_ms": float("nan"), "bracket_max_ms": float("nan"),
            "bracket_missing_count": int(len(targets)),
        }
    source = np.unique(np.sort(finite))
    indices = np.searchsorted(source, targets, side="right") - 1
    valid = indices >= 0
    stale = np.full(len(targets), np.nan, dtype=np.float64)
    stale[valid] = targets[valid] - source[indices[valid]]
    # A future/reset clock produces negative staleness after sorting and is invalid.
    valid &= stale >= 0
    values = stale[valid] * 1000.0
    result = {
        "target_count": int(len(targets)),
        "missing_count": int(len(targets) - np.count_nonzero(valid)),
    }
    for name, q in (("p50_ms", 50), ("p95_ms", 95), ("p99_ms", 99), ("max_ms", 100)):
        result[name] = float(np.percentile(values, q)) if len(values) else float("nan")
    next_indices = np.searchsorted(source, targets, side="left")
    bracket_valid = (indices >= 0) & (next_indices < len(source))
    result["bracket_missing_count"] = int(len(targets) - np.count_nonzero(bracket_valid))
    nearest = np.full(len(targets), np.nan, dtype=np.float64)
    bracket = np.full(len(targets), np.nan, dtype=np.float64)
    nearest[bracket_valid] = np.minimum(
        targets[bracket_valid] - source[indices[bracket_valid]],
        source[next_indices[bracket_valid]] - targets[bracket_valid],
    )
    bracket[bracket_valid] = (
        source[next_indices[bracket_valid]] - source[indices[bracket_valid]]
    )
    nearest_values = nearest[np.isfinite(nearest)] * 1000.0
    bracket_values = bracket[np.isfinite(bracket)] * 1000.0
    for prefix, series in (("nearest", nearest_values), ("bracket", bracket_values)):
        result[f"{prefix}_p99_ms"] = (
            float(np.percentile(series, 99)) if len(series) else float("nan")
        )
        result[f"{prefix}_max_ms"] = (
            float(np.max(series)) if len(series) else float("nan")
        )
    return result


def direct_alignment_stats(
    source_bag: np.ndarray,
    source_header: np.ndarray,
    target_bag: np.ndarray,
    target_header: np.ndarray,
) -> dict[str, Any]:
    """Measure the Header error produced by the converter's Bag-time nearest neighbor."""
    finite = np.isfinite(source_bag) & np.isfinite(source_header)
    source_bag = source_bag[finite]
    source_header = source_header[finite]
    if not len(source_bag) or not len(target_bag):
        return {
            "direct_missing_count": int(len(target_bag)),
            "direct_p50_ms": float("nan"), "direct_p95_ms": float("nan"),
            "direct_p99_ms": float("nan"), "direct_max_ms": float("nan"),
        }
    right = np.searchsorted(source_bag, target_bag, side="left")
    right = np.clip(right, 0, len(source_bag) - 1)
    left = np.clip(right - 1, 0, len(source_bag) - 1)
    choose_right = np.abs(source_bag[right] - target_bag) < np.abs(
        source_bag[left] - target_bag
    )
    indices = np.where(choose_right, right, left)
    error_ms = np.abs(source_header[indices] - target_header) * 1000.0
    result = {"direct_missing_count": 0}
    for name, q in (
        ("direct_p50_ms", 50), ("direct_p95_ms", 95),
        ("direct_p99_ms", 99), ("direct_max_ms", 100),
    ):
        result[name] = float(np.percentile(error_ms, q))
    return result


def cache_file(cache_dir: Path, raw_episode: int, bag_path: Path) -> Path:
    digest = hashlib.sha1(str(bag_path).encode("utf-8")).hexdigest()[:10]
    return cache_dir / f"episode_{raw_episode:04d}_{digest}.json"


def audit_one(job: tuple[int, str, int, int, int, int]) -> dict[str, Any]:
    raw_episode, path_text, trim_start, trim_end, sample_drop, stride = job
    path = Path(path_text)
    result: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "raw_episode": raw_episode,
        "bag_path": str(path),
        "bag_name": path.name,
        "trim_start": trim_start,
        "trim_end": trim_end,
        "sample_drop": sample_drop,
        "camera_stride": stride,
        "error": "",
    }
    try:
        stat = path.stat()
        result["bag_size_bytes"] = stat.st_size
        result["bag_mtime_ns"] = stat.st_mtime_ns
        headers: dict[str, list[float]] = {"camera": []}
        headers.update({key: [] for key in TOPICS})
        bag_times: dict[str, list[float]] = {"camera": []}
        bag_times.update({key: [] for key in TOPICS})
        invalid_payload = {key: 0 for key in TOPICS}
        topic_to_key = {topic: key for key, topic in TOPICS.items()}
        with rosbag.Bag(str(path), "r") as bag:
            for topic, msg, bag_time in bag.read_messages(
                topics=[CAMERA_TOPIC, *TOPICS.values()]
            ):
                if topic == CAMERA_TOPIC:
                    headers["camera"].append(header_stamp(msg))
                    bag_times["camera"].append(float(bag_time.to_sec()))
                else:
                    key = topic_to_key[topic]
                    headers[key].append(header_stamp(msg))
                    bag_times[key].append(float(bag_time.to_sec()))
                    if not payload_is_valid(key, msg):
                        invalid_payload[key] += 1
        missing_topics = [key for key, values in headers.items() if not values]
        result["missing_topics"] = missing_topics
        result["invalid_payload"] = invalid_payload
        if missing_topics:
            result["error"] = "missing_topics"
            return result
        camera = np.asarray(headers["camera"], dtype=np.float64)
        camera_bag = np.asarray(bag_times["camera"], dtype=np.float64)
        if sample_drop > 0:
            camera = camera[sample_drop:-sample_drop:stride]
            camera_bag = camera_bag[sample_drop:-sample_drop:stride]
        else:
            camera = camera[::stride]
            camera_bag = camera_bag[::stride]
        result["master_frame_count"] = int(len(camera))
        if not 0 <= trim_start < trim_end <= len(camera):
            result["error"] = (
                f"trim_range_{trim_start}_{trim_end}_outside_master_{len(camera)}"
            )
            return result
        targets = camera[trim_start:trim_end]
        target_bag = camera_bag[trim_start:trim_end]
        result["retained_frame_count"] = int(len(targets))
        result["target_start_header"] = float(targets[0])
        result["target_end_header"] = float(targets[-1])
        for key in TOPICS:
            source = np.asarray(headers[key], dtype=np.float64)
            source_bag = np.asarray(bag_times[key], dtype=np.float64)
            result[f"{key}_count"] = int(len(source))
            result[f"{key}_nonmonotonic"] = int(np.count_nonzero(np.diff(source) <= 0))
            for metric, value in causal_stats(source, targets).items():
                result[f"{key}_{metric}"] = value
            for metric, value in direct_alignment_stats(
                source_bag, source, target_bag, targets
            ).items():
                result[f"{key}_{metric}"] = value
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def determine_usable(
    row: dict[str, Any], bag_quality: dict[str, str], limits: dict[str, dict[str, float]]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if row.get("error"):
        reasons.append(str(row["error"]))
    event_count = int(bag_quality.get("event_count", 0))
    valid_event_count = int(bag_quality.get("valid_event_count", 0))
    if event_count < 2 or valid_event_count != event_count:
        reasons.append(f"claw_events_{valid_event_count}_of_{event_count}")
    for key, threshold in limits.items():
        if int(row.get(f"{key}_missing_count", 1)) != 0:
            reasons.append(f"{key}_missing_aligned_frames")
        if int(row.get("invalid_payload", {}).get(key, 1)) != 0:
            reasons.append(f"{key}_invalid_payload")
        p99 = float(row.get(f"{key}_p99_ms", float("inf")))
        maximum = float(row.get(f"{key}_max_ms", float("inf")))
        if not np.isfinite(p99) or p99 > threshold["p99_ms"]:
            reasons.append(f"{key}_p99_stale_{p99:.1f}ms")
        if not np.isfinite(maximum) or maximum > threshold["max_ms"]:
            reasons.append(f"{key}_max_stale_{maximum:.1f}ms")
    return not reasons, reasons


def determine_direct(
    row: dict[str, Any], bag_quality: dict[str, str], limits: dict[str, dict[str, float]]
) -> tuple[bool, list[str]]:
    """Whether the already-converted LeRobot frame is trustworthy unchanged."""
    reasons: list[str] = []
    if row.get("error"):
        reasons.append(str(row["error"]))
    event_count = int(bag_quality.get("event_count", 0))
    valid_event_count = int(bag_quality.get("valid_event_count", 0))
    if event_count < 2 or valid_event_count != event_count:
        reasons.append(f"claw_events_{valid_event_count}_of_{event_count}")
    for key, threshold in limits.items():
        if int(row.get(f"{key}_direct_missing_count", 1)) != 0:
            reasons.append(f"{key}_direct_missing_frames")
        if int(row.get("invalid_payload", {}).get(key, 1)) != 0:
            reasons.append(f"{key}_invalid_payload")
        p99 = float(row.get(f"{key}_direct_p99_ms", float("inf")))
        maximum = float(row.get(f"{key}_direct_max_ms", float("inf")))
        if not np.isfinite(p99) or p99 > threshold["p99_ms"]:
            reasons.append(f"{key}_direct_p99_error_{p99:.1f}ms")
        if not np.isfinite(maximum) or maximum > threshold["max_ms"]:
            reasons.append(f"{key}_direct_max_error_{maximum:.1f}ms")
    return not reasons, reasons


def determine_interpolatable(
    row: dict[str, Any], bag_quality: dict[str, str]
) -> tuple[bool, list[str]]:
    """A conservative second tier allowing interpolation of continuous State."""
    reasons: list[str] = []
    if row.get("error"):
        reasons.append(str(row["error"]))
    event_count = int(bag_quality.get("event_count", 0))
    valid_event_count = int(bag_quality.get("valid_event_count", 0))
    if event_count < 2 or valid_event_count != event_count:
        reasons.append(f"claw_events_{valid_event_count}_of_{event_count}")
    for key in TOPICS:
        if int(row.get(f"{key}_missing_count", 1)) != 0:
            reasons.append(f"{key}_missing_aligned_frames")
        if int(row.get("invalid_payload", {}).get(key, 1)) != 0:
            reasons.append(f"{key}_invalid_payload")
    # Joint and claw State are continuous measurements. Do not interpolate
    # across more than 250 ms, and require 99% of brackets to stay <=100 ms.
    for key in ("joint_state", "claw_state"):
        if int(row.get(f"{key}_bracket_missing_count", 1)) != 0:
            reasons.append(f"{key}_missing_interpolation_brackets")
        bracket_p99 = float(row.get(f"{key}_bracket_p99_ms", float("inf")))
        bracket_max = float(row.get(f"{key}_bracket_max_ms", float("inf")))
        nearest_p99 = float(row.get(f"{key}_nearest_p99_ms", float("inf")))
        if bracket_p99 > 100.0 or bracket_max > 250.0 or nearest_p99 > 50.0:
            reasons.append(
                f"{key}_interpolation_gap_p99_{bracket_p99:.1f}_max_{bracket_max:.1f}ms"
            )
    # Actions retain causal zero-order hold semantics.
    for key, p99_limit, max_limit in (
        ("arm_action", 50.0, 150.0), ("claw_action", 50.0, 150.0)
    ):
        p99 = float(row.get(f"{key}_p99_ms", float("inf")))
        maximum = float(row.get(f"{key}_max_ms", float("inf")))
        if p99 > p99_limit or maximum > max_limit:
            reasons.append(f"{key}_causal_gap_p99_{p99:.1f}_max_{maximum:.1f}ms")
    return not reasons, reasons


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    if args.source_pool == "good89":
        raw_segments, good_sources = good89_source_segments(
            args.base_manifest, args.depth200_manifest, args.good89_manifest
        )
        for segment in raw_segments.values():
            segment["previously_dropped"] = False
    else:
        if args.trim_config.exists():
            raw_segments = all_source_segments(args.trim_config)
        elif args.source_root is not None:
            raw_segments = source_segments_from_lerobot(args.source_root)
        else:
            raise FileNotFoundError(
                f"Trim config absent ({args.trim_config}); provide --source-root"
            )
        if args.kept_manifest is not None:
            kept = manifest_source_episodes(args.kept_manifest)
            for episode, segment in raw_segments.items():
                segment["previously_dropped"] = episode not in kept
        good_sources = {}
    skipped = {
        int(value.strip())
        for value in args.skip_episodes.split(",")
        if value.strip()
    }
    with args.bag_quality.open(newline="", encoding="utf-8") as handle:
        quality_rows = list(csv.DictReader(handle))
    quality = {int(row["raw_episode"]): row for row in quality_rows}
    selected = sorted(set(raw_segments) - skipped)
    limits = {
        "joint_state": {"p99_ms": args.joint_state_p99_ms, "max_ms": args.joint_state_max_ms},
        "arm_action": {"p99_ms": args.arm_action_p99_ms, "max_ms": args.arm_action_max_ms},
        "claw_state": {"p99_ms": args.claw_state_p99_ms, "max_ms": args.claw_state_max_ms},
        "claw_action": {"p99_ms": args.claw_action_p99_ms, "max_ms": args.claw_action_max_ms},
    }
    results: dict[int, dict[str, Any]] = {}
    jobs = []
    for raw_episode in selected:
        bag_path = Path(quality[raw_episode]["bag_path"])
        segment = raw_segments[raw_episode]
        cache = cache_file(cache_dir, raw_episode, bag_path)
        if cache.exists() and not args.force:
            value = json.loads(cache.read_text(encoding="utf-8"))
            stat = bag_path.stat()
            current = (
                value.get("audit_version") == AUDIT_VERSION
                and value.get("bag_size_bytes") == stat.st_size
                and value.get("bag_mtime_ns") == stat.st_mtime_ns
                and value.get("trim_start") == segment["start"]
                and value.get("trim_end") == segment["end"]
                and value.get("sample_drop") == args.sample_drop
                and value.get("camera_stride") == args.camera_stride
            )
            if current:
                results[raw_episode] = value
                continue
        jobs.append((
            raw_episode, str(bag_path), segment["start"], segment["end"],
            args.sample_drop, args.camera_stride,
        ))
    print(
        f"source_pool={args.source_pool} raw segments={len(selected)} "
        f"skipped={sorted(skipped)} cached={len(results)} "
        f"pending={len(jobs)} workers={args.workers}", flush=True,
    )
    if jobs:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(audit_one, job): job for job in jobs}
            for completed, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                raw_episode = int(row["raw_episode"])
                results[raw_episode] = row
                cache = cache_file(cache_dir, raw_episode, Path(row["bag_path"]))
                temporary = cache.with_suffix(".tmp")
                temporary.write_text(json.dumps(row), encoding="utf-8")
                os.replace(temporary, cache)
                if completed % max(1, args.progress_every) == 0 or completed == len(jobs):
                    print(f"completed={completed}/{len(jobs)}", flush=True)

    flat_rows = []
    direct_raw: set[int] = set()
    usable_raw: set[int] = set()
    interpolatable_raw: set[int] = set()
    for raw_episode in selected:
        row = results[raw_episode]
        direct, direct_reasons = determine_direct(row, quality[raw_episode], limits)
        usable, reasons = determine_usable(row, quality[raw_episode], limits)
        interpolatable_by_bracketing, interpolation_reasons = determine_interpolatable(
            row, quality[raw_episode]
        )
        # Exact causal alignment is always at least as trustworthy as an
        # interpolation candidate, including at the final frame where a future
        # bracket is intentionally unavailable.
        interpolatable = usable or interpolatable_by_bracketing
        if usable:
            interpolation_reasons = []
        row["direct"] = direct
        row["direct_reasons"] = direct_reasons
        row["usable"] = usable
        row["reasons"] = reasons
        row["interpolatable"] = interpolatable
        row["interpolation_reasons"] = interpolation_reasons
        flat = dict(row)
        flat["previously_dropped"] = bool(
            raw_segments[raw_episode].get("previously_dropped", False)
        )
        flat["invalid_payload"] = json.dumps(row.get("invalid_payload", {}))
        flat["direct_reasons"] = ";".join(direct_reasons)
        flat["reasons"] = ";".join(reasons)
        flat["interpolation_reasons"] = ";".join(interpolation_reasons)
        flat_rows.append(flat)
        if direct:
            direct_raw.add(raw_episode)
        if usable:
            usable_raw.add(raw_episode)
        if interpolatable:
            interpolatable_raw.add(raw_episode)

    good_usable = sorted(
        episode
        for episode, sources in good_sources.items()
        if sources and all(source in usable_raw for source in sources)
    )
    good_interpolatable = sorted(
        episode
        for episode, sources in good_sources.items()
        if sources and all(source in interpolatable_raw for source in sources)
    )
    source_counts = {
        episode: sum(source in usable_raw for source in sources)
        for episode, sources in good_sources.items()
    }
    fields = [
        "raw_episode", "bag_name", "trim_start", "trim_end", "retained_frame_count",
        "previously_dropped", "direct", "direct_reasons", "usable", "reasons",
        "interpolatable", "interpolation_reasons",
        "error", "invalid_payload",
    ]
    for key in TOPICS:
        fields.extend([
            f"{key}_count", f"{key}_missing_count", f"{key}_p50_ms",
            f"{key}_p95_ms", f"{key}_p99_ms", f"{key}_max_ms",
            f"{key}_nearest_p99_ms", f"{key}_nearest_max_ms",
            f"{key}_bracket_p99_ms", f"{key}_bracket_max_ms",
            f"{key}_bracket_missing_count",
            f"{key}_direct_p50_ms", f"{key}_direct_p95_ms",
            f"{key}_direct_p99_ms", f"{key}_direct_max_ms",
            f"{key}_nonmonotonic",
        ])
    report_stem = "good89" if args.source_pool == "good89" else "all"
    csv_path = output_dir / f"{report_stem}_lowdim_alignment.csv"
    write_csv(csv_path, flat_rows, fields)
    (output_dir / "direct_raw_episodes.txt").write_text(
        "\n".join(map(str, sorted(direct_raw))) + ("\n" if direct_raw else ""),
        encoding="utf-8",
    )
    (output_dir / "usable_raw_episodes.txt").write_text(
        "\n".join(map(str, sorted(usable_raw))) + ("\n" if usable_raw else ""),
        encoding="utf-8",
    )
    (output_dir / "usable_existing_good89_episodes.txt").write_text(
        "\n".join(map(str, good_usable)) + ("\n" if good_usable else ""),
        encoding="utf-8",
    )
    (output_dir / "interpolatable_raw_episodes.txt").write_text(
        "\n".join(map(str, sorted(interpolatable_raw)))
        + ("\n" if interpolatable_raw else ""), encoding="utf-8",
    )
    previously_kept = {
        episode
        for episode in selected
        if not raw_segments[episode].get("previously_dropped", False)
    }
    for filename, episodes in (
        ("recommended_direct_single_episodes.txt", direct_raw & previously_kept),
        ("recommended_repairable_single_episodes.txt", usable_raw & previously_kept),
        (
            "recommended_interpolatable_single_episodes.txt",
            interpolatable_raw & previously_kept,
        ),
    ):
        ordered = sorted(episodes)
        (output_dir / filename).write_text(
            "\n".join(map(str, ordered)) + ("\n" if ordered else ""),
            encoding="utf-8",
        )
    mapped_usable = trace_manifest_chain(
        list(args.manifest_chain or []), usable_raw
    )
    mapped_interpolatable = trace_manifest_chain(
        list(args.manifest_chain or []), interpolatable_raw
    )
    for prefix, reports in (
        ("usable", mapped_usable),
        ("interpolatable", mapped_interpolatable),
    ):
        for report in reports:
            safe_name = report["name"].replace("/", "_")
            values = report["qualified"]
            (output_dir / f"{prefix}_{safe_name}_episodes.txt").write_text(
                "\n".join(map(str, values)) + ("\n" if values else ""),
                encoding="utf-8",
            )
            raw_values = report["qualified_raw_sources"]
            (output_dir / f"{prefix}_{safe_name}_raw_sources.txt").write_text(
                "\n".join(map(str, raw_values)) + ("\n" if raw_values else ""),
                encoding="utf-8",
            )
    summary = {
        "source_pool": args.source_pool,
        "explicitly_skipped_raw_episodes": sorted(skipped),
        "raw_segment_count": len(selected),
        "direct_raw_count": len(direct_raw),
        "direct_and_previously_kept_count": len(direct_raw & previously_kept),
        "usable_raw_count": len(usable_raw),
        "usable_and_previously_kept_count": len(usable_raw & previously_kept),
        "theoretical_free_triplets": len(usable_raw) // 3,
        "interpolatable_raw_count": len(interpolatable_raw),
        "interpolatable_and_previously_kept_count": len(
            interpolatable_raw & previously_kept
        ),
        "interpolatable_theoretical_free_triplets": len(interpolatable_raw) // 3,
        "mapped_usable": [
            {
                k: (len(v) if k in ("qualified", "qualified_raw_sources") else v)
                for k, v in report.items()
            }
            for report in mapped_usable
        ],
        "mapped_interpolatable": [
            {
                k: (len(v) if k in ("qualified", "qualified_raw_sources") else v)
                for k, v in report.items()
            }
            for report in mapped_interpolatable
        ],
        "limits": limits,
        "output_csv": str(csv_path),
    }
    if good_sources:
        summary.update({
            "existing_good89_fully_usable_count": len(good_usable),
            "existing_good89_fully_usable_episodes": good_usable,
            "existing_good89_fully_interpolatable_count": len(good_interpolatable),
            "existing_good89_fully_interpolatable_episodes": good_interpolatable,
            "existing_good89_usable_source_count_histogram": {
                str(count): sum(value == count for value in source_counts.values())
                for count in range(4)
            },
        })
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
