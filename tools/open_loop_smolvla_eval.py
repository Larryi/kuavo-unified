#!/usr/bin/env python
"""Open-loop diagnostics for local LeRobot and LingBot policies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

os.environ["HF_DATASETS_CACHE"] = os.environ.get(
    "OPEN_LOOP_HF_DATASETS_CACHE", str(Path.cwd() / ".cache" / "huggingface" / "datasets")
)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from tqdm import tqdm

import lerobot_patches.custom_patches  # noqa: F401
from kuavo_deploy.utils.policy_loader import load_policy_and_processors
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


DEFAULT_DATASET_ROOT = Path("/mnt/pqssd/Real_PQ_3.0/TASK1_SZ/lerobot_trimmed")
DEFAULT_POLICY_PATH = Path(
    "outputs/train/r1/smolvla/run_20260625_212157_from10k_lrfix/checkpoints/040000/pretrained_model"
)
DEFAULT_TASK = "Pick and Place the safety belt, cable and pin connector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare LeRobot policy actions with dataset actions without ROS, sim, or robot."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default="kuavo/task1_sz")
    parser.add_argument(
        "--policy-type", choices=["smolvla", "act", "diffusion", "lingbot"], default="smolvla"
    )
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--task-description", default=DEFAULT_TASK)
    parser.add_argument("--lingbot-root", default="")
    parser.add_argument("--qwen25-path", default="")
    parser.add_argument("--norm-stats-file", default="")
    parser.add_argument("--lingbot-data-type", default="customized")
    parser.add_argument("--lingbot-use-length", type=int, default=5)
    parser.add_argument("--episodes", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-frames-per-episode", type=int, default=80)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-horizon", type=int, default=50)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    parser.add_argument("--mode", choices=["chunk", "queue"], default="chunk")
    parser.add_argument("--video-backend", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def scalar_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def as_cpu_float_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32)


def make_observation(sample: dict, input_keys: Iterable[str], task: str, policy_type: str) -> dict:
    observation = {key: sample[key] for key in input_keys if key in sample}
    if policy_type == "smolvla":
        observation["task"] = sample.get("task") or task
    return observation


def missing_required_keys(policy, sample: dict) -> list[str]:
    required = set(policy.config.input_features)
    required.update(getattr(policy.config, "image_features", {}) or {})
    required.update(getattr(policy.config, "depth_features", {}) or {})
    return sorted(key for key in required if key not in sample)


def postprocess_action(postprocessor, action: torch.Tensor) -> torch.Tensor:
    """Run the LeRobot postprocessor, flattening chunks if an older processor expects 2-D."""
    try:
        return postprocessor(action)
    except Exception:
        original_shape = action.shape
        flat = action.reshape(-1, original_shape[-1])
        out = postprocessor(flat)
        return out.reshape(*original_shape)


def prediction_from_sample(
    policy,
    preprocessor,
    postprocessor,
    sample: dict,
    task: str,
    mode: str,
    policy_type: str,
) -> torch.Tensor:
    observation = make_observation(sample, policy.config.input_features.keys(), task, policy_type)
    batch = preprocessor(observation)
    with torch.inference_mode():
        if mode == "chunk" and hasattr(policy, "predict_action_chunk"):
            policy.reset()
            action = policy.predict_action_chunk(batch)
        else:
            action = policy.select_action(batch)
    action = postprocess_action(postprocessor, action)
    return action.detach().cpu().float()


def action_range_stats(pred: torch.Tensor, action_min: torch.Tensor | None, action_max: torch.Tensor | None) -> tuple[int, int]:
    if action_min is None or action_max is None:
        return 0, 0
    pred_2d = pred.reshape(-1, pred.shape[-1])
    lo = action_min.to(pred_2d)
    hi = action_max.to(pred_2d)
    violations = ((pred_2d < lo) | (pred_2d > hi)).sum().item()
    return int(violations), int(pred_2d.numel())


def get_action_bounds(ds_meta: LeRobotDatasetMetadata) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    stats = ds_meta.stats.get("action") if ds_meta.stats is not None else None
    if not stats or "min" not in stats or "max" not in stats:
        return None, None
    return as_cpu_float_tensor(stats["min"]), as_cpu_float_tensor(stats["max"])


def init_sums(horizon: int, action_dim: int) -> dict[str, torch.Tensor | int]:
    return {
        "sum_abs": torch.zeros(horizon, action_dim),
        "sum_sq": torch.zeros(horizon, action_dim),
        "count_h": torch.zeros(horizon, 1),
        "count": 0,
        "range_violations": 0,
        "range_values": 0,
    }


def write_summary(output_dir: Path, summary: dict, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if rows:
        with (output_dir / "first_action_errors.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def maybe_write_plots(output_dir: Path, summary: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return

    horizons = np.arange(len(summary["horizon_mae"]))
    plt.figure(figsize=(8, 4))
    plt.plot(horizons, summary["horizon_mae"])
    plt.xlabel("horizon step")
    plt.ylabel("MAE")
    plt.title("Open-loop horizon MAE")
    plt.tight_layout()
    plt.savefig(output_dir / "horizon_mae.png", dpi=160)
    plt.close()

    dims = np.arange(len(summary["dim_mae"]))
    plt.figure(figsize=(8, 4))
    plt.bar(dims, summary["dim_mae"])
    plt.xlabel("action dimension")
    plt.ylabel("MAE")
    plt.title("Open-loop per-dimension MAE")
    plt.tight_layout()
    plt.savefig(output_dir / "dim_mae.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    policy_path = resolve_path(args.policy_path)
    output_dir = resolve_path(
        args.output_dir or Path(f"outputs/open_loop/r1/{args.policy_type}")
    )
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    policy_kwargs = None
    if args.policy_type == "lingbot":
        if args.mode == "queue" and args.stride != 1:
            raise ValueError("LingBot queue mode requires --stride=1 so cached actions stay frame-aligned.")
        policy_kwargs = {
            "lingbot_root": args.lingbot_root,
            "qwen25_path": args.qwen25_path,
            "task_prompt": args.task_description,
            "use_length": args.max_horizon if args.mode == "chunk" else args.lingbot_use_length,
            "chunk_ret": args.mode == "chunk",
            "norm_stats_file": args.norm_stats_file,
            "data_type": args.lingbot_data_type,
            "execute_raw_action": False,
        }
    policy, preprocessor, postprocessor = load_policy_and_processors(
        policy_path, args.policy_type, device, policy_kwargs=policy_kwargs
    )
    ds_meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy.config, ds_meta)
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=args.episodes or None,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
    )

    action_min, action_max = get_action_bounds(ds_meta)
    policy_chunk_size = int(getattr(policy.config, "chunk_size", getattr(policy.config, "horizon", args.max_horizon)))
    target_horizon = min(args.max_horizon, policy_chunk_size)
    action_dim = policy.config.output_features["action"].shape[0]
    sums = init_sums(target_horizon, action_dim)
    first_rows: list[dict] = []
    sampled_per_episode = defaultdict(int)
    current_episode = None
    target_episodes = set(args.episodes or [])
    inference_times: list[float] = []
    nonfinite_values = 0
    predicted_values = 0

    progress = tqdm(range(len(dataset)), desc=f"open-loop {args.mode}")
    for idx in progress:
        sample = dataset[idx]
        episode_index = scalar_int(sample["episode_index"])
        frame_index = scalar_int(sample["frame_index"])

        if args.stride > 1 and frame_index % args.stride != 0:
            continue
        if sampled_per_episode[episode_index] >= args.max_frames_per_episode:
            continue

        if args.mode == "queue" and episode_index != current_episode:
            policy.reset()
            current_episode = episode_index

        missing_keys = missing_required_keys(policy, sample)
        if missing_keys:
            raise KeyError(
                "The selected policy expects observation keys that are not present in this dataset sample: "
                f"{missing_keys}. Use the matching LeRobot dataset/checkpoint pair."
            )

        inference_start = time.perf_counter()
        pred = prediction_from_sample(
            policy, preprocessor, postprocessor, sample, args.task_description, args.mode, args.policy_type
        )
        inference_times.append(time.perf_counter() - inference_start)
        gt = as_cpu_float_tensor(sample["action"])

        if args.mode == "queue":
            pred = pred.reshape(1, -1)
            gt = gt[:1] if gt.ndim == 2 else gt.reshape(1, -1)
        else:
            pred = pred.reshape(-1, action_dim)
            gt = gt.reshape(-1, action_dim)

        pad_key = "action_is_pad"
        valid_mask = None
        if pad_key in sample and args.mode == "chunk":
            valid_mask = ~sample[pad_key].detach().cpu().bool()

        horizon = min(target_horizon, pred.shape[0], gt.shape[0])
        if horizon <= 0:
            continue

        pred_h = pred[:horizon]
        gt_h = gt[:horizon]
        if valid_mask is not None:
            valid = valid_mask[:horizon]
            pred_h = pred_h[valid]
            gt_h = gt_h[valid]
            horizon = pred_h.shape[0]
            if horizon <= 0:
                continue

        nonfinite_values += int((~torch.isfinite(pred_h)).sum().item())
        predicted_values += int(pred_h.numel())
        if not torch.isfinite(pred_h).all():
            continue

        err = pred_h - gt_h
        sums["sum_abs"][:horizon] += err.abs()
        sums["sum_sq"][:horizon] += err.square()
        sums["count_h"][:horizon] += 1
        sums["count"] += 1

        violations, values = action_range_stats(pred_h, action_min, action_max)
        sums["range_violations"] += violations
        sums["range_values"] += values

        first_err = err[0]
        row = {
            "dataset_index": idx,
            "episode_index": episode_index,
            "frame_index": frame_index,
            "first_action_mae": float(first_err.abs().mean().item()),
            "first_action_rmse": float(math.sqrt(first_err.square().mean().item())),
        }
        for dim in range(action_dim):
            row[f"pred_{dim}"] = float(pred_h[0, dim].item())
            row[f"gt_{dim}"] = float(gt_h[0, dim].item())
            row[f"err_{dim}"] = float(first_err[dim].item())
        first_rows.append(row)
        sampled_per_episode[episode_index] += 1
        progress.set_postfix(samples=int(sums["count"]))
        if target_episodes and all(
            sampled_per_episode[ep] >= args.max_frames_per_episode for ep in target_episodes
        ):
            break

    count = int(sums["count"])
    if count == 0:
        raise RuntimeError("No samples were evaluated. Check episodes, stride, and max-frames-per-episode.")

    count_h = sums["count_h"].clamp_min(1)
    mae = sums["sum_abs"] / count_h
    mse = sums["sum_sq"] / count_h
    horizon_mae = mae.mean(dim=1)
    dim_mae = mae.mean(dim=0)
    inference_array = np.asarray(inference_times, dtype=np.float64)
    summary = {
        "dataset_root": str(args.dataset_root),
        "repo_id": args.repo_id,
        "policy_type": args.policy_type,
        "policy_path": str(policy_path),
        "mode": args.mode,
        "episodes": args.episodes,
        "stride": args.stride,
        "max_frames_per_episode": args.max_frames_per_episode,
        "samples": count,
        "chunk_size": policy_chunk_size,
        "n_action_steps": int(getattr(policy.config, "n_action_steps", policy_chunk_size)),
        "evaluated_horizon": target_horizon,
        "action_dim": action_dim,
        "overall_mae": float(mae.mean().item()),
        "overall_rmse": float(torch.sqrt(mse.mean()).item()),
        "first_action_mae": float(horizon_mae[0].item()),
        "first_action_rmse": float(torch.sqrt(mse[0].mean()).item()),
        "horizon_mae": [float(v) for v in horizon_mae.tolist()],
        "dim_mae": [float(v) for v in dim_mae.tolist()],
        "joint_mae": float(mae[:, : min(7, action_dim)].mean().item()),
        "gripper_mae": float(mae[:, -1].mean().item()),
        "nonfinite_rate": float(nonfinite_values / predicted_values) if predicted_values else None,
        "inference_seconds_mean": float(inference_array.mean()),
        "inference_seconds_p95": float(np.percentile(inference_array, 95)),
        "range_violation_rate": (
            float(sums["range_violations"] / sums["range_values"]) if sums["range_values"] else None
        ),
        "samples_per_episode": {str(k): v for k, v in sorted(sampled_per_episode.items())},
    }
    write_summary(output_dir, summary, first_rows)
    if not args.no_plots:
        maybe_write_plots(output_dir, summary)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote open-loop diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
