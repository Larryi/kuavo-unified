#!/usr/bin/env python3
"""Download and validate a weighted LeRobot mixture from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from huggingface_hub import snapshot_download


TASK_SCHEMA = {
    "task1": {
        "dim": 8,
        "cameras": {
            "observation.images.head_cam_h",
            "observation.images.wrist_cam_r",
        },
    },
    "task2": {
        "dim": 16,
        "cameras": {
            "observation.images.head_cam_h",
            "observation.images.wrist_cam_l",
            "observation.images.wrist_cam_r",
        },
    },
    "task3": {
        "dim": 8,
        "cameras": {
            "observation.images.head_cam_h",
            "observation.images.wrist_cam_r",
        },
    },
}


def validate_info(info: dict, *, task: str, reference: dict | None) -> None:
    schema = TASK_SCHEMA[task]
    if info["codebase_version"] != "v3.0":
        raise ValueError(f"Expected LeRobot v3.0, got {info['codebase_version']}")
    if info["features"]["observation.state"]["shape"] != [schema["dim"]]:
        raise ValueError("Unexpected observation.state dimension")
    if info["features"]["action"]["shape"] != [schema["dim"]]:
        raise ValueError("Unexpected action dimension")
    cameras = {
        key for key in info["features"] if key.startswith("observation.images.")
    }
    if cameras != schema["cameras"]:
        raise ValueError(f"Unexpected camera schema: {sorted(cameras)}")
    if any("depth" in key for key in info["features"]):
        raise ValueError("Depth features are not supported by this training profile")
    if reference is not None:
        for key in ("fps", "robot_type"):
            if info[key] != reference[key]:
                raise ValueError(
                    f"Dataset mixture {key} mismatch: {reference[key]} != {info[key]}"
                )
        reference_features = {
            key: value
            for key, value in reference["features"].items()
            if key in {"observation.state", "action"}
            or key.startswith("observation.images.")
        }
        current_features = {
            key: value
            for key, value in info["features"].items()
            if key in {"observation.state", "action"}
            or key.startswith("observation.images.")
        }
        if current_features != reference_features:
            raise ValueError("Dataset mixture feature metadata mismatch")


def resolve_mixture(payload: list[dict], *, task: str, output_root: Path) -> list[dict]:
    if not payload:
        raise ValueError("Dataset mixture cannot be empty")
    weights = [float(source["weight"]) for source in payload]
    if any(weight <= 0 for weight in weights):
        raise ValueError("Dataset mixture weights must be positive")
    total = sum(weights)
    resolved = []
    reference = None
    for index, (source, weight) in enumerate(zip(payload, weights, strict=True), 1):
        repo_id = str(source["repo_id"])
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", repo_id)
        root = output_root / f"{index:02d}_{safe_name}"
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=root,
            max_workers=int(os.environ.get("HF_DOWNLOAD_WORKERS", "16")),
            token=os.environ["HF_TOKEN"],
        )
        info = json.loads((root / "meta/info.json").read_text())
        validate_info(info, task=task, reference=reference)
        reference = reference or info
        resolved.append(
            {
                "name": str(source.get("name") or f"source_{index:02d}"),
                "repo_id": repo_id,
                "root": str(root),
                "weight": weight / total,
            }
        )
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_SCHEMA), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resolved-output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(os.environ["DATASET_MIX_JSON"])
    resolved = resolve_mixture(payload, task=args.task, output_root=args.output_root)
    args.resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.resolved_output.with_suffix(".tmp")
    temporary.write_text(json.dumps(resolved, indent=2) + "\n")
    temporary.replace(args.resolved_output)
    for source in resolved:
        print(
            f"Dataset ready: {source['repo_id']} weight={source['weight']:.6f} "
            f"root={source['root']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
