#!/usr/bin/env python3
"""Count debounced gripper close/release cycles in a LeRobot dataset.

This is a trajectory-structure audit, not a task-success classifier.  Extra
cycles are useful candidates for manual review as possible failed-grasp/recovery
episodes, but they may also be intentional re-grasps or annotation artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pads


PRESETS = {
    "task1": {"indices": [7], "expected": [3], "names": ["right_gripper"]},
    "task2": {"indices": [7, 15], "expected": [1, 1], "names": ["left_gripper", "right_gripper"]},
    "task3": {"indices": [7], "expected": [1], "names": ["right_dexhand"]},
}


def csv_ints(text: str | None) -> list[int] | None:
    return None if text is None else [int(value.strip()) for value in text.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(PRESETS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-indices", help="Override preset, e.g. 7,15")
    parser.add_argument("--expected-cycles", help="Override preset, e.g. 1,1")
    parser.add_argument("--close-threshold", type=float, default=0.7)
    parser.add_argument("--open-threshold", type=float, default=0.3)
    parser.add_argument(
        "--debounce-frames", type=int, default=2,
        help="A new open/closed phase must persist for this many frames",
    )
    parser.add_argument(
        "--min-cycle-frames", type=int, default=3,
        help="Flag cycles shorter than this many frames as implausibly short",
    )
    return parser.parse_args()


def stable_state(value: float, open_threshold: float, close_threshold: float) -> int | None:
    if value <= open_threshold:
        return 0
    if value >= close_threshold:
        return 1
    return None


def debounce_events(
    signal: np.ndarray,
    open_threshold: float,
    close_threshold: float,
    debounce_frames: int,
) -> tuple[int | None, int | None, list[dict[str, Any]]]:
    """Return initial/final stable state and confirmed transitions.

    Event frame is the first frame of the run that ultimately passed debounce.
    Values inside the hysteresis band do not cause a transition.
    """
    state: int | None = None
    initial: int | None = None
    candidate: int | None = None
    candidate_start = -1
    candidate_count = 0
    events: list[dict[str, Any]] = []
    for frame, value in enumerate(signal):
        observed = stable_state(float(value), open_threshold, close_threshold)
        if observed is None or observed == state:
            candidate = None
            candidate_count = 0
            continue
        if observed != candidate:
            candidate = observed
            candidate_start = frame
            candidate_count = 1
        else:
            candidate_count += 1
        if candidate_count < debounce_frames:
            continue
        previous = state
        state = observed
        if initial is None:
            initial = state
        else:
            events.append({
                "kind": "close" if state else "open",
                "frame": int(candidate_start),
                "from": "closed" if previous else "open",
                "to": "closed" if state else "open",
                "value": float(signal[candidate_start]),
            })
        candidate = None
        candidate_count = 0
    return initial, state, events


def pair_cycles(events: list[dict[str, Any]]) -> tuple[list[dict[str, int]], int, int]:
    pending_close: int | None = None
    cycles: list[dict[str, int]] = []
    orphan_opens = 0
    for event in events:
        if event["kind"] == "close":
            pending_close = int(event["frame"])
        elif pending_close is None:
            orphan_opens += 1
        else:
            opened = int(event["frame"])
            cycles.append({
                "close_frame": pending_close,
                "open_frame": opened,
                "closed_frames": opened - pending_close,
            })
            pending_close = None
    return cycles, orphan_opens, int(pending_close is not None)


def analyze_signal(
    signal: np.ndarray,
    expected: int,
    fps: float,
    open_threshold: float,
    close_threshold: float,
    debounce_frames: int,
    min_cycle_frames: int,
) -> dict[str, Any]:
    initial, final, events = debounce_events(
        signal, open_threshold, close_threshold, debounce_frames
    )
    cycles, orphan_opens, unclosed = pair_cycles(events)
    closes = sum(event["kind"] == "close" for event in events)
    opens = sum(event["kind"] == "open" for event in events)
    short = sum(cycle["closed_frames"] < min_cycle_frames for cycle in cycles)
    complete = len(cycles)
    if initial is None:
        status = "no_stable_gripper_state"
    elif complete > expected or closes > expected:
        status = "extra_cycle_recovery_candidate"
    elif complete < expected:
        status = "insufficient_cycles"
    elif unclosed or final != 0:
        status = "terminally_closed"
    elif short:
        status = "short_cycle_anomaly"
    else:
        status = "nominal_cycle_count"
    for event in events:
        event["time_s"] = event["frame"] / fps
    for cycle in cycles:
        cycle["closed_seconds"] = cycle["closed_frames"] / fps
    return {
        "expected_cycles": expected,
        "initial_state": None if initial is None else ("closed" if initial else "open"),
        "final_state": None if final is None else ("closed" if final else "open"),
        "close_transitions": closes,
        "open_transitions": opens,
        "complete_cycles": complete,
        "extra_cycles": max(0, complete - expected),
        "orphan_opens": orphan_opens,
        "unclosed_close": unclosed,
        "short_cycles": short,
        "status": status,
        "events": events,
        "cycles": cycles,
        "signal_min": float(np.min(signal)),
        "signal_max": float(np.max(signal)),
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.open_threshold < args.close_threshold <= 1:
        raise ValueError("require 0 <= open_threshold < close_threshold <= 1")
    preset = PRESETS[args.task]
    indices = csv_ints(args.action_indices) or preset["indices"]
    expected = csv_ints(args.expected_cycles) or preset["expected"]
    names = preset["names"] if indices == preset["indices"] else [f"action_{i}" for i in indices]
    if len(indices) != len(expected):
        raise ValueError("action indices and expected cycle counts must have equal length")

    root = args.dataset_root.expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    action_dim = int(info["features"]["action"]["shape"][0])
    if any(index < 0 or index >= action_dim for index in indices):
        raise ValueError(f"indices {indices} outside action dimension {action_dim}")
    table = pads.dataset(root / "data", format="parquet").to_table(
        columns=["episode_index", "frame_index", "index", "action"]
    )
    order = np.argsort(np.asarray(table["index"]), kind="stable")
    episode_ids = np.asarray(table["episode_index"], dtype=np.int64)[order]
    frame_ids = np.asarray(table["frame_index"], dtype=np.int64)[order]
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)[order]

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for episode in sorted(np.unique(episode_ids).tolist()):
        mask = episode_ids == episode
        episode_order = np.argsort(frame_ids[mask], kind="stable")
        episode_actions = actions[mask][episode_order]
        hand_results = []
        for name, index, target in zip(names, indices, expected):
            result = analyze_signal(
                episode_actions[:, index], target, fps,
                args.open_threshold, args.close_threshold,
                args.debounce_frames, args.min_cycle_frames,
            )
            result.update({"name": name, "action_index": index})
            hand_results.append(result)
        statuses = [item["status"] for item in hand_results]
        if any(status == "extra_cycle_recovery_candidate" for status in statuses):
            episode_status = "recovery_candidate"
        elif all(status == "nominal_cycle_count" for status in statuses):
            episode_status = "nominal"
        else:
            episode_status = "anomaly"
        flat: dict[str, Any] = {
            "episode_index": episode,
            "frames": len(episode_actions),
            "duration_s": len(episode_actions) / fps,
            "episode_status": episode_status,
        }
        for item in hand_results:
            prefix = item["name"]
            for key in (
                "action_index", "expected_cycles", "initial_state", "final_state",
                "close_transitions", "open_transitions", "complete_cycles",
                "extra_cycles", "unclosed_close", "short_cycles", "status",
                "signal_min", "signal_max",
            ):
                flat[f"{prefix}/{key}"] = item[key]
            flat[f"{prefix}/event_frames"] = json.dumps(
                [{"kind": e["kind"], "frame": e["frame"]} for e in item["events"]]
            )
        rows.append(flat)
        details.append({
            "episode_index": episode, "frames": len(episode_actions),
            "duration_s": len(episode_actions) / fps,
            "episode_status": episode_status, "hands": hand_results,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "gripper_cycle_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["episode_status"] for row in rows)
    hand_distributions = {}
    for name in names:
        hand_distributions[name] = {
            "complete_cycles": dict(sorted(Counter(
                int(row[f"{name}/complete_cycles"]) for row in rows
            ).items())),
            "close_transitions": dict(sorted(Counter(
                int(row[f"{name}/close_transitions"]) for row in rows
            ).items())),
            "statuses": dict(Counter(str(row[f"{name}/status"]) for row in rows)),
        }
    summary = {
        "dataset_root": str(root), "task": args.task, "fps": fps,
        "total_episodes": len(rows), "action_dimension": action_dim,
        "action_indices": indices, "expected_cycles": expected,
        "thresholds": {"open": args.open_threshold, "close": args.close_threshold,
                       "debounce_frames": args.debounce_frames,
                       "min_cycle_frames": args.min_cycle_frames},
        "episode_status_counts": dict(counts),
        "hand_distributions": hand_distributions,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "episode_details.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for status, filename in (
        ("recovery_candidate", "recovery_candidate_episodes.txt"),
        ("anomaly", "anomaly_episodes.txt"),
        ("nominal", "nominal_episodes.txt"),
    ):
        values = [str(row["episode_index"]) for row in rows if row["episode_status"] == status]
        (args.output_dir / filename).write_text(
            "\n".join(values) + ("\n" if values else ""), encoding="utf-8"
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
