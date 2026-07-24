#!/usr/bin/env python3
"""Fast TASK1 LeRobot rebuild from repaired lowdim caches and trimmed source videos."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import aggregate_feature_stats


CAMERAS = (
    "observation.images.head_cam_h",
    "observation.images.wrist_cam_r",
)
STAT_NAMES = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--whitelist", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def vector_stats(array: np.ndarray) -> dict[str, list]:
    array = np.asarray(array)
    axis = 0
    return {
        "min": np.min(array, axis=axis).tolist(),
        "max": np.max(array, axis=axis).tolist(),
        "mean": np.mean(array, axis=axis, dtype=np.float64).tolist(),
        "std": np.std(array, axis=axis, dtype=np.float64).tolist(),
        "count": [int(len(array))],
        "q01": np.quantile(array, 0.01, axis=axis).tolist(),
        "q10": np.quantile(array, 0.10, axis=axis).tolist(),
        "q50": np.quantile(array, 0.50, axis=axis).tolist(),
        "q90": np.quantile(array, 0.90, axis=axis).tolist(),
        "q99": np.quantile(array, 0.99, axis=axis).tolist(),
    }


def source_video(root: Path, camera: str, chunk: int, file: int) -> Path:
    return root / "videos" / camera / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"


def encode_one(job: tuple[Path, Path, float, int, int, str]) -> tuple[str, int]:
    source, output, start_s, frames, crf, preset = job
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "/usr/bin/ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start_s:.9f}",
        "-i", str(source), "-map", "0:v:0", "-frames:v", str(frames),
        "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-r", "10", "-vsync", "cfr", "-g", "10", "-pix_fmt", "yuv420p",
        str(output),
    ]
    subprocess.run(command, check=True)
    probe = subprocess.run([
        "/usr/bin/ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames", "-of", "default=nw=1:nk=1",
        str(output),
    ], check=True, capture_output=True, text=True)
    actual = int(probe.stdout.strip())
    if actual != frames:
        raise ValueError(f"{output}: expected {frames} frames, encoded {actual}")
    return str(output), actual


def replace_stats(row: dict, feature: str, stats: dict[str, list]) -> None:
    for name, value in stats.items():
        row[f"stats/{feature}/{name}"] = value


def row_feature_stats(row: dict, feature: str) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(row[f"stats/{feature}/{name}"])
        for name in STAT_NAMES
    }


def scalar_array_stats(array: np.ndarray) -> dict[str, list]:
    result = vector_stats(np.asarray(array).reshape(-1, 1))
    return result


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    ids = sorted({int(x) for x in args.whitelist.read_text().split()})
    source_info = json.loads((source_root / "meta/info.json").read_text())
    source_stats = json.loads((source_root / "meta/stats.json").read_text())
    episode_path = source_root / "meta/episodes/chunk-000/file-000.parquet"
    source_episodes = pq.read_table(episode_path)
    source_rows = {int(row["episode_index"]): row for row in source_episodes.to_pylist()}

    rebuilt: list[dict] = []
    for source_episode in ids:
        cache_path = args.cache_dir / f"episode_{source_episode:04d}.npz"
        with np.load(cache_path, allow_pickle=False) as cache:
            start, end = int(cache["trim_start"]), int(cache["trim_end"])
            rebuilt.append({
                "source_episode": source_episode,
                "source_row": source_rows[source_episode],
                "trim_start": start,
                "trim_end": end,
                "state": cache["state"][start:end].copy(),
                "action": cache["action"][start:end].copy(),
                "header_start": float(cache["target_times"][start]),
                "header_end": float(cache["target_times"][end - 1]),
            })
    total_frames = sum(len(item["state"]) for item in rebuilt)
    print(f"Preparing {len(rebuilt)} episodes, {total_frames} frames", flush=True)

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_root}; pass --overwrite")
        shutil.rmtree(output_root)
    (output_root / "data/chunk-000").mkdir(parents=True)
    (output_root / "meta/episodes/chunk-000").mkdir(parents=True)

    states = np.concatenate([item["state"] for item in rebuilt])
    actions = np.concatenate([item["action"] for item in rebuilt])
    timestamps = np.concatenate([
        np.arange(len(item["state"]), dtype=np.float32) / 10.0 for item in rebuilt
    ])
    frame_indices = np.concatenate([
        np.arange(len(item["state"]), dtype=np.int64) for item in rebuilt
    ])
    episode_indices = np.concatenate([
        np.full(len(item["state"]), index, dtype=np.int64)
        for index, item in enumerate(rebuilt)
    ])
    indices = np.arange(total_frames, dtype=np.int64)
    task_indices = np.zeros(total_frames, dtype=np.int64)
    source_schema = pq.read_schema(source_root / "data/chunk-000/file-000.parquet")
    columns = {
        "observation.state": pa.array(states.tolist(), type=source_schema.field("observation.state").type),
        "action": pa.array(actions.tolist(), type=source_schema.field("action").type),
        "timestamp": pa.array(timestamps, type=pa.float32()),
        "frame_index": pa.array(frame_indices, type=pa.int64()),
        "episode_index": pa.array(episode_indices, type=pa.int64()),
        "index": pa.array(indices, type=pa.int64()),
        "task_index": pa.array(task_indices, type=pa.int64()),
    }
    data_table = pa.Table.from_arrays(
        [columns[field.name] for field in source_schema], schema=source_schema
    )
    pq.write_table(data_table, output_root / "data/chunk-000/file-000.parquet")

    episode_rows = []
    manifest = []
    cursor = 0
    video_jobs = []
    for output_episode, item in enumerate(rebuilt):
        n = len(item["state"])
        row = dict(item["source_row"])
        row.update({
            "episode_index": output_episode,
            "length": n,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": cursor,
            "dataset_to_index": cursor + n,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        })
        episode_scalars = {
            "timestamp": np.arange(n, dtype=np.float32) / 10.0,
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, output_episode, dtype=np.int64),
            "index": np.arange(cursor, cursor + n, dtype=np.int64),
            "task_index": np.zeros(n, dtype=np.int64),
        }
        replace_stats(row, "observation.state", vector_stats(item["state"]))
        replace_stats(row, "action", vector_stats(item["action"]))
        for key, array in episode_scalars.items():
            replace_stats(row, key, scalar_array_stats(array))
        for camera in CAMERAS:
            prefix = f"videos/{camera}"
            chunk = int(item["source_row"][f"{prefix}/chunk_index"])
            file = int(item["source_row"][f"{prefix}/file_index"])
            source_from = float(item["source_row"][f"{prefix}/from_timestamp"])
            row[f"{prefix}/chunk_index"] = 0
            row[f"{prefix}/file_index"] = output_episode
            row[f"{prefix}/from_timestamp"] = 0.0
            row[f"{prefix}/to_timestamp"] = n / 10.0
            source = source_video(source_root, camera, chunk, file)
            output = output_root / "videos" / camera / "chunk-000" / f"file-{output_episode:03d}.mp4"
            start_s = source_from + item["trim_start"] / 10.0
            video_jobs.append((source, output, start_s, n, args.crf, args.preset))
        episode_rows.append(row)
        manifest.append({
            "output_episode": output_episode,
            "source_episode": item["source_episode"],
            "trim_start": item["trim_start"],
            "trim_end": item["trim_end"],
            "frames": n,
            "header_start": item["header_start"],
            "header_end": item["header_end"],
        })
        cursor += n
    episode_table = pa.Table.from_pylist(episode_rows, schema=source_episodes.schema)
    pq.write_table(episode_table, output_root / "meta/episodes/chunk-000/file-000.parquet")

    shutil.copy2(source_root / "meta/tasks.parquet", output_root / "meta/tasks.parquet")
    info = source_info
    info.update({
        "total_episodes": len(rebuilt), "total_frames": total_frames,
        "total_tasks": 1, "splits": {"train": f"0:{len(rebuilt)}"},
    })
    for camera in CAMERAS:
        info["features"][camera]["info"]["video.codec"] = "h264"
    (output_root / "meta/info.json").write_text(json.dumps(info, indent=2))
    global_stats = source_stats
    global_arrays = {
        "observation.state": states, "action": actions,
        "timestamp": timestamps, "frame_index": frame_indices,
        "episode_index": episode_indices, "index": indices, "task_index": task_indices,
    }
    for key, array in global_arrays.items():
        global_stats[key] = vector_stats(array if array.ndim == 2 else array.reshape(-1, 1))
    for camera in CAMERAS:
        aggregated = aggregate_feature_stats([
            row_feature_stats(item["source_row"], camera) for item in rebuilt
        ])
        global_stats[camera] = {
            name: np.asarray(value).tolist() for name, value in aggregated.items()
        }
    (output_root / "meta/stats.json").write_text(json.dumps(global_stats, indent=2))
    (output_root / "header_rebuild_manifest.json").write_text(json.dumps({
        "source_root": str(source_root), "whitelist": str(args.whitelist.resolve()),
        "fps": 10, "arm_mode": "right", "rgb_keys": list(CAMERAS),
        "episodes": manifest,
    }, indent=2))

    print(f"Encoding {len(video_jobs)} videos with {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(encode_one, job) for job in video_jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 20 == 0 or completed == len(futures):
                print(f"videos {completed}/{len(futures)}", flush=True)
    print(f"Complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
