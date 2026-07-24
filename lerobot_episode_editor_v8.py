"""
LeRobot Episode Editor / Trimmer

A lightweight local Streamlit tool for visually inspecting LeRobot v3 datasets,
marking per-episode trim ranges, and exporting a new trimmed dataset.

Usage:
    pip install streamlit matplotlib pandas pyarrow
    streamlit run lerobot_episode_editor_v8.py

Recommended example:
    Source repo id: so101_test
    Source root:    C:\so101_test
    Output repo id: so101_test_trimmed
    Output root:   C:\so101_test_trimmed
"""

from __future__ import annotations

import inspect
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

from lerobot.datasets.lerobot_dataset import LeRobotDataset


AUTO_GENERATED_KEYS = {
    "index",
    "episode_index",
    "frame_index",
    "timestamp",
    "next.done",
    "task_index",
}
AUTO_PREFIXES = ("next.",)


def as_python_scalar(x: Any) -> Any:
    if torch is not None and isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.detach().cpu().item()
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray) and x.size == 1:
        return x.item()
    return x


def to_numpy(x: Any) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "__array__"):
        return np.asarray(x)
    return np.asarray(x)


def image_to_numpy(img: Any) -> np.ndarray:
    """Convert PIL / torch / numpy image into HWC uint8 for Streamlit."""
    arr = to_numpy(img)

    # CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    # grayscale -> HWC
    if arr.ndim == 2:
        arr = arr[..., None]

    # float image
    if np.issubdtype(arr.dtype, np.floating):
        finite = arr[np.isfinite(arr)]
        max_val = float(finite.max()) if finite.size else 1.0
        if max_val <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def image_to_lerobot_hwc(img: Any) -> np.ndarray:
    """Convert decoded dataset image tensors back to HWC uint8 before add_frame().

    LeRobotDataset.__getitem__ commonly returns image tensors as CHW for training
    / visualization, while LeRobotDataset.add_frame() validates raw image inputs
    against HWC shapes such as (H, W, 3). This converter prevents shape mismatch
    during export.
    """
    arr = to_numpy(img)

    # Common PyTorch format: C,H,W -> H,W,C
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    # Grayscale is uncommon here, but keep it valid.
    if arr.ndim == 2:
        arr = arr[..., None]

    # Convert floats in [0, 1] to uint8 image.
    if np.issubdtype(arr.dtype, np.floating):
        finite = arr[np.isfinite(arr)]
        max_val = float(finite.max()) if finite.size else 1.0
        if max_val <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(arr)


def read_info_json(root: Path) -> dict[str, Any]:
    path = root / "meta" / "info.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_tasks(root: Path) -> dict[int, str]:
    """Read task_index -> task string mapping from tasks.parquet if available."""
    path = root / "meta" / "tasks.parquet"
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path)
    except Exception:
        return {}

    if df.empty:
        return {}

    # Common v3 fields are often task_index + task.
    cols = list(df.columns)
    idx_col = None
    task_col = None

    for c in ["task_index", "index", "id"]:
        if c in cols:
            idx_col = c
            break
    for c in ["task", "task_name", "name"]:
        if c in cols:
            task_col = c
            break

    if idx_col is None or task_col is None:
        return {}

    out: dict[int, str] = {}
    for _, row in df.iterrows():
        try:
            out[int(row[idx_col])] = str(row[task_col])
        except Exception:
            pass
    return out


@st.cache_resource(show_spinner=False)
def load_dataset(repo_id: str, root_str: str) -> LeRobotDataset:
    return LeRobotDataset(repo_id=repo_id, root=Path(root_str))


@st.cache_data(show_spinner=False, max_entries=5000)
def load_episode_first_frame_image(repo_id: str, root_str: str, global_index: int, image_key: str) -> np.ndarray:
    """Load one episode boundary thumbnail.

    The contact-sheet view may request many first frames. Caching here makes the
    wall responsive after the first pass while keeping the main dataset object out
    of Streamlit's data-cache hash.
    """
    local_ds = LeRobotDataset(repo_id=repo_id, root=Path(root_str))
    sample = local_ds[int(global_index)]
    if image_key not in sample:
        raise KeyError(f"Image key not found in sample: {image_key}")
    return image_to_numpy(sample[image_key])


def maybe_bordered_container():
    """Return a Streamlit container, using borders when the installed version supports it."""
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


@st.cache_data(show_spinner=False)
def build_episode_table(root_str: str) -> pd.DataFrame:
    """
    Build episode table by reading only small index columns from data/*.parquet.
    This is intentionally independent of private LeRobot metadata APIs.
    """
    root = Path(root_str)
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    parts = []
    cumulative = 0

    for f in files:
        if pq is not None:
            schema_cols = pq.ParquetFile(f).schema.names
            wanted = [c for c in ["index", "episode_index", "frame_index", "timestamp", "task_index"] if c in schema_cols]
            df = pd.read_parquet(f, columns=wanted)
        else:
            df = pd.read_parquet(f)
            keep = [c for c in ["index", "episode_index", "frame_index", "timestamp", "task_index"] if c in df.columns]
            df = df[keep]

        if "index" in df.columns:
            df["_global_index"] = df["index"].astype(int)
        else:
            df["_global_index"] = np.arange(cumulative, cumulative + len(df), dtype=int)

        cumulative += len(df)
        parts.append(df)

    full = pd.concat(parts, ignore_index=True)

    if "episode_index" not in full.columns:
        raise KeyError("The parquet files do not contain an episode_index column.")

    rows = []
    for ep_idx, g in full.groupby("episode_index", sort=True):
        global_start = int(g["_global_index"].min())
        global_end = int(g["_global_index"].max()) + 1

        row = {
            "episode_index": int(ep_idx),
            "global_start": global_start,
            "global_end": global_end,
            "num_frames": int(len(g)),
        }

        if "frame_index" in g.columns:
            row["frame_min"] = int(g["frame_index"].min())
            row["frame_max"] = int(g["frame_index"].max())
        if "timestamp" in g.columns:
            try:
                row["duration_s"] = float(g["timestamp"].max() - g["timestamp"].min())
            except Exception:
                row["duration_s"] = np.nan
        if "task_index" in g.columns:
            try:
                row["task_index"] = int(g["task_index"].iloc[0])
            except Exception:
                pass

        rows.append(row)

    return pd.DataFrame(rows).sort_values("episode_index").reset_index(drop=True)


def get_feature_keys(ds: LeRobotDataset, sample: dict[str, Any] | None = None) -> tuple[list[str], list[str]]:
    features = getattr(ds, "features", {}) or {}
    keys = list(features.keys())
    if sample:
        keys = sorted(set(keys) | set(sample.keys()))

    image_keys = [k for k in keys if k.startswith("observation.images")]
    lowdim_keys = [k for k in keys if k in ("action", "observation.state") or k.startswith("observation.") and not k.startswith("observation.images")]
    return image_keys, lowdim_keys


def get_dataset_fps(ds: LeRobotDataset, root: Path, fallback: float = 30.0) -> float:
    """Resolve dataset FPS from LeRobotDataset first, then meta/info.json."""
    for value in [getattr(ds, "fps", None), read_info_json(root).get("fps")]:
        try:
            fps = float(value)
            if np.isfinite(fps) and fps > 0:
                return fps
        except Exception:
            pass
    return fallback


def parse_episode_id_spec(spec: str) -> list[int]:
    """Parse strings like "0,1,2,10-12" into sorted unique episode ids."""
    ids: set[int] = set()
    for part in str(spec or "").replace("；", ",").replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a.strip()), int(b.strip())
            lo, hi = min(a, b), max(a, b)
            ids.update(range(lo, hi + 1))
        else:
            ids.add(int(part))
    return sorted(ids)


def make_episode_groups(
    episode_table: pd.DataFrame,
    group_size: int,
    trim_config: dict[str, dict[str, Any]] | None = None,
    regroup_after_drops: bool = False,
) -> list[list[int]]:
    """Group episode IDs into consecutive chunks.

    If regroup_after_drops=True, episodes already marked as drop are removed
    before regrouping. This is useful when isolated / bad episodes appear in the
    raw sequence: mark them as drop first, then group the remaining valid
    episodes as [valid0, valid1, valid2], [valid3, valid4, valid5], ...
    """
    eps = episode_table["episode_index"].astype(int).tolist()
    if regroup_after_drops and trim_config is not None:
        eps = [ep for ep in eps if not bool(trim_config.get(str(int(ep)), {}).get("drop", False))]
    group_size = max(1, int(group_size))
    return [eps[i : i + group_size] for i in range(0, len(eps), group_size)]


def build_length_table(
    episode_table: pd.DataFrame,
    fps: float,
    trim_config: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Add raw/trimmed duration columns for length diagnostics and sampling."""
    df = episode_table.copy()
    fps = max(1e-6, float(fps))

    if "duration_s" in df.columns:
        raw_duration = pd.to_numeric(df["duration_s"], errors="coerce")
    else:
        raw_duration = pd.Series(np.nan, index=df.index)

    fallback_duration = pd.to_numeric(df["num_frames"], errors="coerce") / fps
    raw_duration = raw_duration.where(np.isfinite(raw_duration) & (raw_duration > 0), fallback_duration)
    df["raw_duration_s"] = raw_duration.astype(float)
    df["raw_num_frames"] = pd.to_numeric(df["num_frames"], errors="coerce").fillna(0).astype(int)

    kept_frames = []
    kept_duration = []
    dropped = []
    for _, row in df.iterrows():
        ep = str(int(row["episode_index"]))
        n = int(row["num_frames"])
        cfg = (trim_config or {}).get(ep, {})
        drop = bool(cfg.get("drop", False))
        start = max(0, min(n, int(cfg.get("start", 0))))
        end = max(start, min(n, int(cfg.get("end", n))))
        k = 0 if drop else max(0, end - start)
        kept_frames.append(k)
        kept_duration.append(k / fps)
        dropped.append(drop)

    df["kept_frames"] = kept_frames
    df["kept_duration_s"] = kept_duration
    df["drop"] = dropped
    df["length_outlier_iqr"] = False

    valid = df["raw_duration_s"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) >= 4:
        q1, q3 = valid.quantile([0.25, 0.75])
        iqr = q3 - q1
        if np.isfinite(iqr) and iqr > 0:
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            df["length_outlier_iqr"] = (df["raw_duration_s"] < lo) | (df["raw_duration_s"] > hi)

    return df



def build_group_length_table(
    episode_table: pd.DataFrame,
    length_df: pd.DataFrame,
    group_size: int,
    trim_config: dict[str, dict[str, Any]] | None = None,
    regroup_after_drops: bool = False,
) -> pd.DataFrame:
    """Aggregate episode lengths into consecutive long-task groups.

    A group is defined by the current sidebar setting "Episodes per long task
    group". If group_size=3, sorted source episodes are grouped as [0,1,2],
    [3,4,5], [6,7,8], etc. Both diagnostics and export can then share the
    same unit of analysis.
    """
    groups = make_episode_groups(episode_table, int(group_size), trim_config, regroup_after_drops)
    length_by_ep = {int(r["episode_index"]): r for _, r in length_df.iterrows()}
    rows = []

    for group_idx, eps in enumerate(groups):
        eps = [int(e) for e in eps]
        ep_rows = [length_by_ep[e] for e in eps if e in length_by_ep]
        if not ep_rows:
            continue

        raw_frames = int(sum(int(r.get("raw_num_frames", r.get("num_frames", 0))) for r in ep_rows))
        kept_frames = int(sum(int(r.get("kept_frames", 0)) for r in ep_rows))
        raw_duration = float(sum(float(r.get("raw_duration_s", 0.0)) for r in ep_rows))
        kept_duration = float(sum(float(r.get("kept_duration_s", 0.0)) for r in ep_rows))
        drop_count = int(sum(bool(r.get("drop", False)) for r in ep_rows))
        outlier_count = int(sum(bool(r.get("length_outlier_iqr", False)) for r in ep_rows))

        rows.append(
            {
                "group_index": int(group_idx),
                "source_episodes": ", ".join(map(str, eps)),
                "start_episode": int(eps[0]),
                "end_episode": int(eps[-1]),
                "group_num_episodes": int(len(eps)),
                "complete_group": bool(len(eps) == max(1, int(group_size))),
                "raw_num_frames": raw_frames,
                "raw_duration_s": raw_duration,
                "kept_frames": kept_frames,
                "kept_duration_s": kept_duration,
                "drop_count": drop_count,
                "kept_episode_count": int(len(eps) - drop_count),
                "group_drop": bool(drop_count == len(eps)),
                "episode_outlier_count": outlier_count,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["group_length_outlier_iqr"] = False
    valid = df["raw_duration_s"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) >= 4:
        q1, q3 = valid.quantile([0.25, 0.75])
        iqr = q3 - q1
        if np.isfinite(iqr) and iqr > 0:
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            df["group_length_outlier_iqr"] = (df["raw_duration_s"] < lo) | (df["raw_duration_s"] > hi)
    return df


def plot_group_length_distribution(
    group_df: pd.DataFrame,
    duration_col: str,
    min_s: float | None = None,
    max_s: float | None = None,
    sampled_groups: list[int] | None = None,
):
    """Render histogram and group-index scatter for long-task group lengths."""
    if plt is None:
        st.error(f"matplotlib import failed: {_MATPLOTLIB_IMPORT_ERROR}")
        return

    df = group_df.copy()
    vals = pd.to_numeric(df[duration_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        st.warning("No valid group durations found.")
        return

    sampled_groups = set(map(int, sampled_groups or []))

    fig, ax = plt.subplots(figsize=(9, 3.0), dpi=130)
    bins = min(60, max(10, int(np.sqrt(len(vals))) * 2))
    ax.hist(vals.to_numpy(), bins=bins, alpha=0.75)
    ax.axvline(vals.median(), linestyle="--", linewidth=1.2, label=f"median={vals.median():.2f}s")
    if min_s is not None:
        ax.axvline(float(min_s), linestyle=":", linewidth=1.2, label=f"min={float(min_s):.2f}s")
    if max_s is not None:
        ax.axvline(float(max_s), linestyle=":", linewidth=1.2, label=f"max={float(max_s):.2f}s")
    ax.set_xlabel("Group duration (s)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    st.pyplot(fig, clear_figure=True)

    fig2, ax2 = plt.subplots(figsize=(9, 3.0), dpi=130)
    x = df["group_index"].astype(int).to_numpy()
    y = pd.to_numeric(df[duration_col], errors="coerce").to_numpy()
    ax2.scatter(x, y, s=14, alpha=0.65, label="groups")
    if min_s is not None:
        ax2.axhline(float(min_s), linestyle=":", linewidth=1.2)
    if max_s is not None:
        ax2.axhline(float(max_s), linestyle=":", linewidth=1.2)
    if sampled_groups:
        mask = df["group_index"].astype(int).isin(sampled_groups).to_numpy()
        ax2.scatter(x[mask], y[mask], s=32, marker="x", label="sampled groups")
    ax2.set_xlabel("Group index")
    ax2.set_ylabel("Group duration (s)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", fontsize=8)
    st.pyplot(fig2, clear_figure=True)


def plot_episode_length_distribution(
    length_df: pd.DataFrame,
    duration_col: str,
    min_s: float | None = None,
    max_s: float | None = None,
    sampled_eps: list[int] | None = None,
):
    """Render histogram and episode-index scatter for episode lengths."""
    if plt is None:
        st.error(f"matplotlib import failed: {_MATPLOTLIB_IMPORT_ERROR}")
        return

    df = length_df.copy()
    vals = pd.to_numeric(df[duration_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        st.warning("No valid episode durations found.")
        return

    sampled_eps = set(map(int, sampled_eps or []))

    fig, ax = plt.subplots(figsize=(9, 3.0), dpi=130)
    bins = min(60, max(10, int(np.sqrt(len(vals))) * 2))
    ax.hist(vals.to_numpy(), bins=bins, alpha=0.75)
    ax.axvline(vals.median(), linestyle="--", linewidth=1.2, label=f"median={vals.median():.2f}s")
    if min_s is not None:
        ax.axvline(float(min_s), linestyle=":", linewidth=1.2, label=f"min={float(min_s):.2f}s")
    if max_s is not None:
        ax.axvline(float(max_s), linestyle=":", linewidth=1.2, label=f"max={float(max_s):.2f}s")
    ax.set_xlabel("Episode duration (s)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    st.pyplot(fig, clear_figure=True)

    fig2, ax2 = plt.subplots(figsize=(9, 3.0), dpi=130)
    x = df["episode_index"].astype(int).to_numpy()
    y = pd.to_numeric(df[duration_col], errors="coerce").to_numpy()
    ax2.scatter(x, y, s=12, alpha=0.65, label="episodes")
    if min_s is not None:
        ax2.axhline(float(min_s), linestyle=":", linewidth=1.2)
    if max_s is not None:
        ax2.axhline(float(max_s), linestyle=":", linewidth=1.2)
    if sampled_eps:
        mask = df["episode_index"].astype(int).isin(sampled_eps).to_numpy()
        ax2.scatter(x[mask], y[mask], s=28, marker="x", label="sampled")
    ax2.set_xlabel("Episode index")
    ax2.set_ylabel("Episode duration (s)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", fontsize=8)
    st.pyplot(fig2, clear_figure=True)


def sample_episodes_uniform_by_duration(
    length_df: pd.DataFrame,
    duration_col: str,
    min_s: float,
    max_s: float,
    count: int,
    seed: int = 42,
    exclude_dropped: bool = True,
    strategy: str = "uniform_targets",
) -> list[int]:
    """Sample episodes inside a duration interval.

    strategy='uniform_targets' places evenly spaced targets along the duration axis
    and picks the nearest unused candidate episode for each target, so the chosen
    subset covers the requested duration range more evenly than plain random
    sampling when the length distribution is skewed.
    """
    df = length_df.copy()
    duration = pd.to_numeric(df[duration_col], errors="coerce")
    mask = duration.notna() & np.isfinite(duration) & (duration >= float(min_s)) & (duration <= float(max_s))
    if exclude_dropped and "drop" in df.columns:
        mask &= ~df["drop"].astype(bool)
    candidates = df.loc[mask, ["episode_index", duration_col]].copy()
    candidates[duration_col] = pd.to_numeric(candidates[duration_col], errors="coerce")
    candidates = candidates.dropna().sort_values([duration_col, "episode_index"]).reset_index(drop=True)

    if candidates.empty:
        return []

    count = max(0, int(count))
    if count <= 0 or count >= len(candidates):
        return sorted(candidates["episode_index"].astype(int).tolist())

    rng = np.random.default_rng(int(seed))

    if strategy == "random":
        selected = rng.choice(candidates["episode_index"].astype(int).to_numpy(), size=count, replace=False)
        return sorted(map(int, selected.tolist()))

    durations = candidates[duration_col].astype(float).to_numpy()
    eps = candidates["episode_index"].astype(int).to_numpy()
    targets = np.linspace(float(min_s), float(max_s), count)
    selected_indices: list[int] = []
    used: set[int] = set()

    # Randomize target order very slightly for tie-breaking while keeping even coverage.
    for target in targets:
        order = np.argsort(np.abs(durations - target))
        pick = None
        for idx in order:
            idx = int(idx)
            if idx not in used:
                pick = idx
                break
        if pick is None:
            break
        used.add(pick)
        selected_indices.append(pick)

    selected_eps = eps[selected_indices].astype(int).tolist()
    return sorted(map(int, selected_eps))


def apply_episode_subset_to_trim_config(
    trim_config: dict[str, dict[str, Any]],
    episode_table: pd.DataFrame,
    selected_eps: list[int],
    *,
    drop_unselected: bool = True,
):
    """Mark sampled episodes as keep and, optionally, all others as drop."""
    selected_set = set(map(int, selected_eps))
    for _, row in episode_table.iterrows():
        ep = int(row["episode_index"])
        key = str(ep)
        n = int(row["num_frames"])
        if key not in trim_config:
            trim_config[key] = {"start": 0, "end": n, "drop": False}
        if drop_unselected:
            trim_config[key]["drop"] = ep not in selected_set
        elif ep in selected_set:
            trim_config[key]["drop"] = False



def sample_groups_uniform_by_duration(
    group_df: pd.DataFrame,
    duration_col: str,
    min_s: float,
    max_s: float,
    count: int,
    seed: int = 42,
    exclude_dropped: bool = True,
    only_complete_groups: bool = True,
    strategy: str = "uniform_targets",
) -> list[int]:
    """Sample long-task groups inside a duration interval."""
    df = group_df.copy()
    duration = pd.to_numeric(df[duration_col], errors="coerce")
    mask = duration.notna() & np.isfinite(duration) & (duration >= float(min_s)) & (duration <= float(max_s))
    if exclude_dropped and "group_drop" in df.columns:
        mask &= ~df["group_drop"].astype(bool)
    if only_complete_groups and "complete_group" in df.columns:
        mask &= df["complete_group"].astype(bool)

    candidates = df.loc[mask, ["group_index", duration_col]].copy()
    candidates[duration_col] = pd.to_numeric(candidates[duration_col], errors="coerce")
    candidates = candidates.dropna().sort_values([duration_col, "group_index"]).reset_index(drop=True)

    if candidates.empty:
        return []

    count = max(0, int(count))
    if count <= 0 or count >= len(candidates):
        return sorted(candidates["group_index"].astype(int).tolist())

    rng = np.random.default_rng(int(seed))
    if strategy == "random":
        selected = rng.choice(candidates["group_index"].astype(int).to_numpy(), size=count, replace=False)
        return sorted(map(int, selected.tolist()))

    durations = candidates[duration_col].astype(float).to_numpy()
    groups = candidates["group_index"].astype(int).to_numpy()
    targets = np.linspace(float(min_s), float(max_s), count)
    selected_indices: list[int] = []
    used: set[int] = set()
    for target in targets:
        order = np.argsort(np.abs(durations - target))
        pick = None
        for idx in order:
            idx = int(idx)
            if idx not in used:
                pick = idx
                break
        if pick is None:
            break
        used.add(pick)
        selected_indices.append(pick)
    selected_groups = groups[selected_indices].astype(int).tolist()
    return sorted(map(int, selected_groups))


def apply_group_subset_to_trim_config(
    trim_config: dict[str, dict[str, Any]],
    episode_table: pd.DataFrame,
    selected_group_indices: list[int],
    group_size: int,
    *,
    drop_unselected: bool = True,
    regroup_after_drops: bool = False,
):
    """Mark selected long-task groups as keep and, optionally, all others as drop."""
    groups = make_episode_groups(episode_table, int(group_size), trim_config, regroup_after_drops)
    selected_group_set = set(map(int, selected_group_indices))
    selected_eps: list[int] = []
    for group_idx, eps in enumerate(groups):
        if int(group_idx) in selected_group_set:
            selected_eps.extend(map(int, eps))
    apply_episode_subset_to_trim_config(
        trim_config,
        episode_table,
        selected_eps,
        drop_unselected=drop_unselected,
    )


def playback_sequence(
    episode_table: pd.DataFrame,
    trim_config: dict[str, dict[str, Any]],
    episodes: list[int],
    use_trimmed_range: bool,
) -> list[dict[str, int]]:
    """Build a flat frame sequence across one or more episodes."""
    rows_by_ep = {int(r["episode_index"]): r for _, r in episode_table.iterrows()}
    seq: list[dict[str, int]] = []

    for ep in episodes:
        if ep not in rows_by_ep:
            continue
        row = rows_by_ep[ep]
        global_start = int(row["global_start"])
        n = int(row["num_frames"])
        cfg = trim_config.get(str(ep), {})

        if cfg.get("drop", False):
            continue

        if use_trimmed_range:
            local_start = int(cfg.get("start", 0))
            local_end = int(cfg.get("end", n))
        else:
            local_start, local_end = 0, n

        local_start = max(0, min(n, local_start))
        local_end = max(local_start, min(n, local_end))

        for local_i in range(local_start, local_end):
            seq.append(
                {
                    "episode": int(ep),
                    "local_frame": int(local_i),
                    "global_index": int(global_start + local_i),
                    "episode_num_frames": int(n),
                }
            )

    return seq


def render_sample_panel(
    sample: dict[str, Any],
    ds: LeRobotDataset,
    title: str | None = None,
    show_action: bool = True,
    show_lowdim: bool = False,
):
    """Render camera images and optional low-dimensional values."""
    if title:
        st.caption(title)

    image_keys, lowdim_keys = get_feature_keys(ds, sample)

    if image_keys:
        img_cols = st.columns(len(image_keys))
        for col, key in zip(img_cols, image_keys):
            with col:
                st.caption(key)
                try:
                    st.image(image_to_numpy(sample[key]), use_container_width=True)
                except Exception as exc:
                    st.error(f"Failed to show {key}: {exc}")
    else:
        st.warning("No observation.images.* keys found.")

    if show_action:
        action = to_numpy(sample.get("action", np.array([]))).reshape(-1)
        st.caption("Current action")
        st.code(np.array2string(action, precision=4, suppress_small=True), language="text")

    if show_lowdim:
        for key in lowdim_keys:
            if key in sample:
                st.caption(key)
                arr = to_numpy(sample[key]).reshape(-1)
                st.code(np.array2string(arr, precision=4, suppress_small=True), language="text")


def play_frames(
    ds: LeRobotDataset,
    sequence: list[dict[str, int]],
    fps: float,
    speed: float,
    show_action: bool,
    show_lowdim: bool,
):
    """Blocking Streamlit playback loop with a progress bar and synced frame display."""
    if not sequence:
        st.warning("Playback sequence is empty. Check trim ranges or dropped episodes.")
        return

    fps = max(1e-6, float(fps))
    speed = max(1e-6, float(speed))
    target_dt = 1.0 / (fps * speed)

    progress = st.progress(0.0, text="Preparing playback...")
    meta_box = st.empty()
    frame_box = st.empty()

    total = len(sequence)
    wall_start = time.perf_counter()

    for i, item in enumerate(sequence):
        tic = time.perf_counter()
        sample = ds[int(item["global_index"])]

        progress.progress(
            min(1.0, (i + 1) / total),
            text=(
                f"Playing {i + 1}/{total} | "
                f"episode {item['episode']} | "
                f"frame {item['local_frame']}/{max(0, item['episode_num_frames'] - 1)} | "
                f"{speed:g}x @ {fps:g} FPS"
            ),
        )
        elapsed_wall = time.perf_counter() - wall_start
        meta_box.caption(
            f"Episode {item['episode']} · local frame {item['local_frame']} · "
            f"global index {item['global_index']} · playback wall time {elapsed_wall:.2f}s"
        )

        with frame_box.container():
            render_sample_panel(
                sample,
                ds,
                title=None,
                show_action=show_action,
                show_lowdim=show_lowdim,
            )

        spent = time.perf_counter() - tic
        sleep_s = max(0.0, target_dt - spent)
        if sleep_s > 0:
            time.sleep(sleep_s)


@st.cache_data(show_spinner=False)
def compute_action_metrics(repo_id: str, root_str: str, global_start: int, global_end: int) -> pd.DataFrame:
    ds = LeRobotDataset(repo_id=repo_id, root=Path(root_str))
    rows = []
    prev_action = None
    for local_i, global_i in enumerate(range(global_start, global_end)):
        frame = ds[global_i]
        action = to_numpy(frame.get("action", np.array([]))).astype(float).reshape(-1)
        if action.size == 0:
            l2 = np.nan
            mean_abs = np.nan
            delta_l2 = np.nan
        else:
            l2 = float(np.linalg.norm(action))
            mean_abs = float(np.mean(np.abs(action)))
            if prev_action is None:
                delta_l2 = 0.0
            else:
                delta_l2 = float(np.linalg.norm(action - prev_action))
            prev_action = action
        rows.append(
            {
                "local_frame": local_i,
                "global_index": global_i,
                "action_l2": l2,
                "action_mean_abs": mean_abs,
                "action_delta_l2": delta_l2,
            }
        )
    return pd.DataFrame(rows)


def suggest_trim(metrics: pd.DataFrame, column: str, threshold: float, margin: int) -> tuple[int, int]:
    if metrics.empty or column not in metrics.columns:
        return 0, 0

    vals = metrics[column].to_numpy()
    active = np.isfinite(vals) & (vals > threshold)

    if not active.any():
        return 0, len(metrics)

    active_indices = np.flatnonzero(active)
    start = max(0, int(active_indices[0]) - margin)
    end = min(len(metrics), int(active_indices[-1]) + 1 + margin)
    return start, end


def plot_metrics(metrics: pd.DataFrame, trim_start: int, trim_end: int, current_frame: int, y_col: str):
    if plt is None:
        st.error(f"matplotlib import failed: {_MATPLOTLIB_IMPORT_ERROR}")
        return

    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=130)
    x = metrics["local_frame"].to_numpy()
    y = metrics[y_col].to_numpy()

    ax.plot(x, y, linewidth=1.2, label=y_col)
    ax.axvspan(trim_start, max(trim_start, trim_end - 1), alpha=0.15, label="kept range")
    ax.axvline(current_frame, linestyle="--", linewidth=1.0, label="current frame")
    ax.set_xlabel("Local frame")
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    st.pyplot(fig, clear_figure=True)


def clean_frame_for_add(frame: dict[str, Any], task_map: dict[int, str]) -> dict[str, Any]:
    new_frame = dict(frame)

    # LeRobotDataset.__getitem__ returns decoded images in training-friendly CHW,
    # but add_frame() expects raw HWC images. Convert image fields before validation.
    for key in list(new_frame.keys()):
        if key.startswith("observation.images"):
            new_frame[key] = image_to_lerobot_hwc(new_frame[key])

    # Try to preserve task as a string if LeRobot's add_frame expects it.
    if "task" not in new_frame:
        task_index = new_frame.get("task_index", None)
        task_index = as_python_scalar(task_index)
        try:
            task_index_int = int(task_index)
        except Exception:
            task_index_int = None
        if task_index_int is not None and task_index_int in task_map:
            new_frame["task"] = task_map[task_index_int]
        elif task_map:
            new_frame["task"] = next(iter(task_map.values()))

    for key in list(new_frame.keys()):
        if key in AUTO_GENERATED_KEYS or any(key.startswith(p) for p in AUTO_PREFIXES):
            new_frame.pop(key, None)

    return new_frame


def cleaned_features_for_create(ds: LeRobotDataset) -> dict[str, Any]:
    features = dict(getattr(ds, "features", {}) or {})
    for key in list(features.keys()):
        if key in AUTO_GENERATED_KEYS or any(key.startswith(p) for p in AUTO_PREFIXES):
            features.pop(key, None)
    return features


def call_create_dataset(
    src: LeRobotDataset,
    dst_repo_id: str,
    dst_root: Path,
    src_root: Path,
    *,
    streaming_encoding: bool = True,
    encoder_threads: int | None = None,
    image_writer_threads: int = 0,
    image_writer_processes: int = 0,
    batch_encoding_size: int = 1,
):
    info = read_info_json(src_root)
    robot_type = getattr(getattr(src, "meta", None), "robot_type", None) or info.get("robot_type")
    fps = getattr(src, "fps", None) or info.get("fps")
    features = cleaned_features_for_create(src)

    # LeRobot's create() signature changes across versions. We pass all useful
    # acceleration knobs opportunistically and keep only parameters supported by
    # the installed local version.
    candidates = {
        "repo_id": dst_repo_id,
        "root": dst_root,
        "fps": fps,
        "features": features,
        "robot_type": robot_type,
        "use_videos": True,
        # Fast path: encode video while add_frame() streams frames into the writer.
        # This avoids writing thousands of temporary PNGs and then encoding them.
        "streaming_encoding": streaming_encoding,
        "encoder_threads": None if encoder_threads is None or encoder_threads <= 0 else int(encoder_threads),
        # Fallback acceleration if streaming_encoding is not available in this version.
        "image_writer_threads": int(image_writer_threads),
        "image_writer_processes": int(image_writer_processes),
        "batch_encoding_size": max(1, int(batch_encoding_size)),
    }

    sig = inspect.signature(LeRobotDataset.create)
    kwargs = {k: v for k, v in candidates.items() if k in sig.parameters and v is not None}
    return LeRobotDataset.create(**kwargs)


def finalize_dataset(ds: LeRobotDataset):
    if hasattr(ds, "finalize"):
        return ds.finalize()
    if hasattr(ds, "consolidate"):
        return ds.consolidate()
    return None


def export_trimmed_dataset(
    src_repo_id: str,
    src_root: Path,
    dst_repo_id: str,
    dst_root: Path,
    episode_table: pd.DataFrame,
    trim_config: dict[str, dict[str, Any]],
    overwrite: bool,
    merge_group_size: int = 1,
    *,
    streaming_encoding: bool = True,
    encoder_threads: int | None = None,
    image_writer_threads: int = 0,
    image_writer_processes: int = 0,
    batch_encoding_size: int = 1,
    regroup_after_drops: bool = False,
    progress_cb=None,
):
    """Export a trimmed dataset.

    If merge_group_size <= 1, each kept source episode becomes one output episode.
    If merge_group_size > 1, consecutive source episodes are concatenated in groups
    of merge_group_size and each group is saved as one output episode. This is useful
    when one logical long-horizon task was recorded as multiple short episodes.
    """
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"{dst_root} already exists. Enable overwrite to replace it.")
        shutil.rmtree(dst_root)

    # return_uint8=True avoids an unnecessary float image path on newer LeRobot versions.
    src_kwargs = {"repo_id": src_repo_id, "root": src_root}
    if "return_uint8" in inspect.signature(LeRobotDataset).parameters:
        src_kwargs["return_uint8"] = True
    src = LeRobotDataset(**src_kwargs)
    dst = call_create_dataset(
        src,
        dst_repo_id,
        dst_root,
        src_root,
        streaming_encoding=streaming_encoding,
        encoder_threads=encoder_threads,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
        batch_encoding_size=batch_encoding_size,
    )
    task_map = read_tasks(src_root)

    exported = []
    merge_group_size = max(1, int(merge_group_size))

    rows = [row for _, row in episode_table.sort_values("episode_index").iterrows()]

    if merge_group_size <= 1:
        groups = [[row] for row in rows]
    elif regroup_after_drops:
        active_rows = [row for row in rows if not bool(trim_config.get(str(int(row["episode_index"])), {}).get("drop", False))]
        groups = [active_rows[i : i + merge_group_size] for i in range(0, len(active_rows), merge_group_size)]
    else:
        groups = [rows[i : i + merge_group_size] for i in range(0, len(rows), merge_group_size)]

    total_export_frames = 0
    for row in rows:
        ep = int(row["episode_index"])
        cfg = trim_config.get(str(ep), {})
        if cfg.get("drop", False):
            continue
        n = int(row["num_frames"])
        local_start = max(0, min(n, int(cfg.get("start", 0))))
        local_end = max(local_start, min(n, int(cfg.get("end", n))))
        total_export_frames += max(0, local_end - local_start)

    processed_frames = 0
    if progress_cb is not None:
        progress_cb(0, max(1, total_export_frames), "starting export")

    for group_idx, group_rows in enumerate(groups):
        group_exported = []
        group_frame_count = 0

        for row in group_rows:
            ep = int(row["episode_index"])
            cfg = trim_config.get(str(ep), {})
            if cfg.get("drop", False):
                continue

            global_start = int(row["global_start"])
            n = int(row["num_frames"])

            local_start = int(cfg.get("start", 0))
            local_end = int(cfg.get("end", n))

            local_start = max(0, min(n, local_start))
            local_end = max(local_start, min(n, local_end))

            if local_end <= local_start:
                continue

            for global_i in range(global_start + local_start, global_start + local_end):
                frame = src[global_i]
                dst.add_frame(clean_frame_for_add(frame, task_map))
                processed_frames += 1
                if progress_cb is not None and (processed_frames % 30 == 0 or processed_frames == total_export_frames):
                    progress_cb(
                        processed_frames,
                        max(1, total_export_frames),
                        f"group {group_idx}, source episode {ep}, global frame {global_i}",
                    )

            frames = local_end - local_start
            group_frame_count += frames
            group_exported.append({"source_episode": ep, "start": local_start, "end": local_end, "frames": frames})

            # In non-merge mode, each group contains one source episode, so this still
            # saves exactly one output episode per source episode.

        if group_frame_count > 0:
            dst.save_episode()
            exported.append(
                {
                    "output_episode": len(exported),
                    "merge_group_index": group_idx,
                    "source_segments": group_exported,
                    "frames": group_frame_count,
                }
            )

    finalize_dataset(dst)

    manifest = {
        "source_repo_id": src_repo_id,
        "source_root": str(src_root),
        "output_repo_id": dst_repo_id,
        "output_root": str(dst_root),
        "merge_group_size": merge_group_size,
        "regroup_after_drops": bool(regroup_after_drops),
        "exported": exported,
    }
    (dst_root / "trim_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


st.set_page_config(page_title="LeRobot Episode Editor", layout="wide")

st.title("LeRobot Episode Editor / Trimmer v8")
st.caption("Visual inspect episodes, mark trim ranges, and export a new trimmed LeRobot dataset. The source dataset is never modified.")

with st.sidebar:
    st.header("Dataset")
    src_repo_id = st.text_input("Source repo id", value="so101_test")
    src_root_str = st.text_input("Source root", value=r"C:\so101_test")
    src_root = Path(src_root_str)

    dst_repo_id = st.text_input("Output repo id", value=f"{src_repo_id}_trimmed")
    dst_root_str = st.text_input("Output root", value=str(src_root.parent / f"{src_root.name}_trimmed"))
    dst_root = Path(dst_root_str)

    st.divider()
    st.header("Viewer")
    metric_col = st.selectbox("Activity metric", ["action_delta_l2", "action_mean_abs", "action_l2"], index=0)
    threshold = st.number_input("Auto-trim threshold", min_value=0.0, value=0.005, step=0.001, format="%.6f")
    margin = st.number_input("Margin frames", min_value=0, value=3, step=1)
    show_lowdim = st.checkbox("Show low-dimensional vectors", value=False)

    st.divider()
    st.header("Playback")
    playback_speed_label = st.selectbox("Playback speed", ["1x", "2x", "5x"], index=0)
    playback_speed = {"1x": 1.0, "2x": 2.0, "5x": 5.0}[playback_speed_label]
    playback_use_trimmed = st.checkbox("Play trimmed ranges only", value=True)
    playback_show_action = st.checkbox("Show action during playback", value=False)
    group_size = st.number_input("Episodes per long task group", min_value=1, max_value=10000, value=3, step=1)
    regroup_after_drops = st.checkbox(
        "Regroup after dropped/ignored episodes",
        value=True,
        help=(
            "When enabled, episodes marked as drop are removed first, then the remaining valid "
            "episodes are re-numbered into groups. For group_size=3, valid episodes become "
            "[v0,v1,v2], [v3,v4,v5], ... even if raw episode ids contain isolated bad episodes."
        ),
    )

if not src_root.exists():
    st.error(f"Source root does not exist: {src_root}")
    st.stop()

try:
    ds = load_dataset(src_repo_id, str(src_root))
    episode_table = build_episode_table(str(src_root))
except Exception as exc:
    st.exception(exc)
    st.stop()

if "trim_config" not in st.session_state:
    st.session_state.trim_config = {}

if "sampled_episode_ids" not in st.session_state:
    st.session_state.sampled_episode_ids = []

if "sampled_group_indices" not in st.session_state:
    st.session_state.sampled_group_indices = []

# Auto-load a saved config from the source root once per app session.
# This lets you update/fix this editor script without losing work, as long as
# you saved trim_config.json before replacing/rerunning the script.
if not st.session_state.get("_auto_loaded_trim_config", False):
    saved_config_path = src_root / "trim_config.json"
    if saved_config_path.exists():
        try:
            payload = json.loads(saved_config_path.read_text(encoding="utf-8"))
            loaded = payload.get("trims", payload)
            if isinstance(loaded, dict):
                st.session_state.trim_config.update(loaded)
                st.toast(f"Loaded saved trim config from {saved_config_path}")
        except Exception as exc:
            st.warning(f"Found trim_config.json but failed to load it: {exc}")
    st.session_state["_auto_loaded_trim_config"] = True

# Ensure default full-range config exists for every episode.
for _, row in episode_table.iterrows():
    ep = str(int(row["episode_index"]))
    if ep not in st.session_state.trim_config:
        st.session_state.trim_config[ep] = {
            "start": 0,
            "end": int(row["num_frames"]),
            "drop": False,
        }

with st.expander("Ignored / orphan episodes before grouping", expanded=False):
    st.caption(
        "Use this when raw episode ids are not perfectly aligned to the long-task rhythm. "
        "Mark isolated or bad episodes as drop first; if regroup-after-drops is enabled, "
        "groups are rebuilt from the remaining valid sequence."
    )
    ignore_spec = st.text_input(
        "Episode ids or ranges to ignore",
        value="",
        placeholder="e.g. 3, 17, 28-30",
        help="Comma-separated ids/ranges. These episodes will be marked drop, then excluded from regrouping/export when regroup-after-drops is enabled.",
    )
    ig1, ig2, ig3 = st.columns(3)
    parsed_ignore_ids: list[int] = []
    if ignore_spec.strip():
        try:
            parsed_ignore_ids = [ep for ep in parse_episode_id_spec(ignore_spec) if ep in set(episode_table["episode_index"].astype(int))]
            st.info(f"Parsed {len(parsed_ignore_ids)} valid episode ids: {parsed_ignore_ids[:30]}" + (" ..." if len(parsed_ignore_ids) > 30 else ""))
        except Exception as exc:
            st.error(f"Failed to parse episode ids: {exc}")
    with ig1:
        if st.button("Mark listed episodes as drop", use_container_width=True, disabled=not bool(parsed_ignore_ids)):
            for ep in parsed_ignore_ids:
                key = str(int(ep))
                if key in st.session_state.trim_config:
                    st.session_state.trim_config[key]["drop"] = True
            st.success(f"Marked {len(parsed_ignore_ids)} episodes as drop.")
            st.rerun()
    with ig2:
        if st.button("Un-drop listed episodes", use_container_width=True, disabled=not bool(parsed_ignore_ids)):
            for ep in parsed_ignore_ids:
                key = str(int(ep))
                if key in st.session_state.trim_config:
                    st.session_state.trim_config[key]["drop"] = False
            st.success(f"Restored {len(parsed_ignore_ids)} episodes.")
            st.rerun()
    with ig3:
        dropped_now = [int(ep) for ep, cfg in st.session_state.trim_config.items() if bool(cfg.get("drop", False))]
        st.metric("Currently dropped", len(dropped_now))

dataset_fps = get_dataset_fps(ds, src_root)
length_table = build_length_table(episode_table, dataset_fps, st.session_state.trim_config)
episode_groups = make_episode_groups(episode_table, int(group_size), st.session_state.trim_config, bool(regroup_after_drops))

with st.expander("Group contact sheet / orphan episode review", expanded=True):
    st.caption(
        "Use this wall to inspect long-task boundaries visually. Each box is one current group; "
        "each thumbnail is the first frame of one source episode. Drop an isolated episode under its image, "
        "then the wall is rebuilt immediately. When regroup-after-drops is enabled, the remaining episodes are "
        "re-grouped as a new valid sequence."
    )

    # Resolve image keys from the first sample. This avoids relying on feature metadata only.
    try:
        first_sample_for_wall = ds[int(episode_table.iloc[0]["global_start"])]
        wall_image_keys, _ = get_feature_keys(ds, first_sample_for_wall)
    except Exception:
        wall_image_keys, _ = get_feature_keys(ds, None)

    if not wall_image_keys:
        st.warning("No observation.images.* keys found for the contact sheet.")
    else:
        wall_top_1, wall_top_2, wall_top_3, wall_top_4 = st.columns([1.2, 1.0, 1.0, 1.4])
        with wall_top_1:
            wall_camera_key = st.selectbox(
                "Thumbnail camera",
                wall_image_keys,
                index=0,
                key="contact_sheet_camera_key",
            )
        with wall_top_2:
            groups_per_row = st.number_input("Groups per row", min_value=1, max_value=6, value=3, step=1)
        with wall_top_3:
            groups_per_page = st.number_input("Groups per page", min_value=3, max_value=300, value=30, step=3)
        with wall_top_4:
            st.caption(
                f"Current group size: {int(group_size)} episode(s). "
                f"Current regroup mode: {'on' if regroup_after_drops else 'off'}."
            )

        total_groups_for_wall = len(episode_groups)
        max_page = max(0, int(np.ceil(total_groups_for_wall / max(1, int(groups_per_page)))) - 1)
        wall_page = st.number_input(
            "Contact sheet page",
            min_value=0,
            max_value=max_page,
            value=min(int(st.session_state.get("contact_sheet_page", 0)), max_page),
            step=1,
            help="Use paging for large datasets; rendering every episode thumbnail at once can be slow.",
        )
        st.session_state.contact_sheet_page = int(wall_page)

        start_group = int(wall_page) * int(groups_per_page)
        end_group = min(total_groups_for_wall, start_group + int(groups_per_page))

        ep_rows_by_id = {int(r["episode_index"]): r for _, r in episode_table.iterrows()}
        st.info(
            f"Showing groups {start_group}–{max(start_group, end_group) - 1} of {total_groups_for_wall}. "
            "Buttons update drop flags and rerun the page; if regrouping is enabled, group membership changes immediately."
        )

        dropped_ids = [int(ep) for ep, cfg in st.session_state.trim_config.items() if bool(cfg.get("drop", False))]
        if dropped_ids:
            with st.expander(f"Dropped episodes not shown in regrouped wall ({len(dropped_ids)})", expanded=False):
                st.write(", ".join(map(str, sorted(dropped_ids)[:300])) + (" ..." if len(dropped_ids) > 300 else ""))
                restore_spec = st.text_input(
                    "Restore dropped episode ids/ranges",
                    value="",
                    placeholder="e.g. 3,17,28-30",
                    key="contact_sheet_restore_spec",
                )
                if st.button("Restore listed dropped episodes", disabled=not restore_spec.strip()):
                    try:
                        restore_ids = parse_episode_id_spec(restore_spec)
                        for ep in restore_ids:
                            key = str(int(ep))
                            if key in st.session_state.trim_config:
                                st.session_state.trim_config[key]["drop"] = False
                        st.success(f"Restored {len(restore_ids)} episode(s).")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to restore: {exc}")

        visible_group_indices = list(range(start_group, end_group))
        for row_start in range(0, len(visible_group_indices), int(groups_per_row)):
            cols = st.columns(int(groups_per_row))
            for col, group_idx in zip(cols, visible_group_indices[row_start : row_start + int(groups_per_row)]):
                eps = [int(e) for e in episode_groups[group_idx]]
                with col:
                    with maybe_bordered_container():
                        complete_flag = "complete" if len(eps) == int(group_size) else "incomplete"
                        st.markdown(f"**Group {group_idx}** · {complete_flag} · ep {', '.join(map(str, eps))}")

                        if st.button(
                            "Drop entire group",
                            key=f"contact_drop_group_{group_idx}_{'_'.join(map(str, eps))}",
                            use_container_width=True,
                        ):
                            for ep in eps:
                                key = str(int(ep))
                                if key in st.session_state.trim_config:
                                    st.session_state.trim_config[key]["drop"] = True
                            st.rerun()

                        thumb_cols = st.columns(max(1, len(eps)))
                        for thumb_col, ep in zip(thumb_cols, eps):
                            row = ep_rows_by_id.get(int(ep))
                            if row is None:
                                continue
                            global_start = int(row["global_start"])
                            n_frames = int(row["num_frames"])
                            duration_s = n_frames / max(1e-6, float(dataset_fps))
                            is_dropped = bool(st.session_state.trim_config.get(str(int(ep)), {}).get("drop", False))
                            with thumb_col:
                                st.caption(f"Ep {ep} · {duration_s:.1f}s")
                                try:
                                    img = load_episode_first_frame_image(src_repo_id, str(src_root), global_start, wall_camera_key)
                                    st.image(img, use_container_width=True)
                                except Exception as exc:
                                    st.warning(f"Image load failed: {exc}")

                                if is_dropped:
                                    if st.button(
                                        "Restore",
                                        key=f"contact_restore_ep_{ep}_{group_idx}",
                                        use_container_width=True,
                                    ):
                                        st.session_state.trim_config[str(int(ep))]["drop"] = False
                                        st.rerun()
                                else:
                                    if st.button(
                                        "Drop",
                                        key=f"contact_drop_ep_{ep}_{group_idx}",
                                        use_container_width=True,
                                    ):
                                        st.session_state.trim_config[str(int(ep))]["drop"] = True
                                        st.rerun()

with st.expander("Episode / group length diagnostics & sampling", expanded=False):
    st.caption(
        "Use this panel to find abnormal durations, filter by duration, sample a balanced subset, "
        "and mark unselected episodes or long-task groups as dropped before export. "
        "When group mode is selected, the same group size used for playback/export is used for diagnostics. "
        "If regroup-after-drops is enabled, dropped/ignored episodes are removed before grouping."
    )

    diagnostic_unit = st.radio(
        "Diagnostic unit",
        ["Single episodes", f"Groups of {int(group_size)} episodes"],
        horizontal=True,
        help="Choose group mode when each long task spans multiple consecutive source episodes. For group_size=3, groups are ep 0–2, 3–5, 6–8, etc.",
    )
    use_group_diagnostics = diagnostic_unit.startswith("Groups")

    duration_basis_label = st.radio(
        "Duration basis",
        ["Raw length", "Trimmed kept length"],
        horizontal=True,
        help="Raw length ignores current trim ranges. Trimmed kept length uses the current [start,end) and drop flags.",
    )
    duration_col = "raw_duration_s" if duration_basis_label == "Raw length" else "kept_duration_s"

    if use_group_diagnostics:
        group_length_table = build_group_length_table(
            episode_table,
            length_table,
            int(group_size),
            st.session_state.trim_config,
            bool(regroup_after_drops),
        )
        active_table = group_length_table
        unit_name = "groups"
        id_col = "group_index"
        if active_table.empty:
            st.warning("No groups available for diagnostics.")
            valid_durations = pd.Series(dtype=float)
        else:
            valid_durations = pd.to_numeric(active_table[duration_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    else:
        active_table = length_table
        unit_name = "episodes"
        id_col = "episode_index"
        valid_durations = pd.to_numeric(active_table[duration_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    if valid_durations.empty:
        st.warning(f"No valid durations available for {unit_name} sampling.")
    else:
        q = valid_durations.quantile([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(unit_name.capitalize(), len(active_table))
        m2.metric("Median", f"{q.loc[0.5]:.2f}s")
        m3.metric("P05–P95", f"{q.loc[0.05]:.2f}–{q.loc[0.95]:.2f}s")
        m4.metric("Min–Max", f"{q.loc[0.0]:.2f}–{q.loc[1.0]:.2f}s")
        if use_group_diagnostics:
            incomplete = int((~active_table.get("complete_group", pd.Series([], dtype=bool))).sum()) if len(active_table) else 0
            m5.metric("Incomplete groups", incomplete)
        else:
            m5.metric("IQR outliers", int(active_table["length_outlier_iqr"].sum()))

        default_min = float(q.loc[0.05])
        default_max = float(q.loc[0.95])
        min_allowed = float(q.loc[0.0])
        max_allowed = float(q.loc[1.0])
        if default_min >= default_max:
            default_min, default_max = min_allowed, max_allowed

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sample_min_s = st.number_input(
                "Min duration (s)",
                min_value=0.0,
                value=max(0.0, default_min),
                step=0.5,
                format="%.3f",
                key="length_sample_min_s_group" if use_group_diagnostics else "length_sample_min_s_episode",
            )
        with c2:
            sample_max_s = st.number_input(
                "Max duration (s)",
                min_value=0.0,
                value=max(default_max, default_min),
                step=0.5,
                format="%.3f",
                key="length_sample_max_s_group" if use_group_diagnostics else "length_sample_max_s_episode",
            )
        with c3:
            max_count = len(active_table)
            default_count = min(100, max_count)
            sample_count = st.number_input(
                f"Target {unit_name}",
                min_value=0,
                max_value=max_count,
                value=default_count,
                step=1,
                key="length_sample_count_group" if use_group_diagnostics else "length_sample_count_episode",
            )
        with c4:
            sample_seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=42, step=1)

        c5, c6, c7, c8 = st.columns([1.25, 1.05, 1.05, 1.65])
        with c5:
            sample_strategy_label = st.selectbox(
                "Sampling strategy",
                ["Uniform over duration range", "Random among candidates"],
                index=0,
                key="length_sample_strategy_group" if use_group_diagnostics else "length_sample_strategy_episode",
            )
        with c6:
            exclude_dropped = st.checkbox("Exclude dropped", value=True)
        with c7:
            if use_group_diagnostics:
                only_complete_groups = st.checkbox("Only complete groups", value=True)
            else:
                only_complete_groups = False
        with c8:
            st.caption("Uniform strategy places evenly spaced targets on the duration axis and picks nearest items.")

        strategy = "uniform_targets" if sample_strategy_label == "Uniform over duration range" else "random"

        duration_series = pd.to_numeric(active_table[duration_col], errors="coerce")
        candidates_mask = duration_series.notna() & (duration_series >= float(sample_min_s)) & (duration_series <= float(sample_max_s))
        if use_group_diagnostics:
            if exclude_dropped and "group_drop" in active_table.columns:
                candidates_mask &= ~active_table["group_drop"].astype(bool)
            if only_complete_groups and "complete_group" in active_table.columns:
                candidates_mask &= active_table["complete_group"].astype(bool)
        else:
            if exclude_dropped:
                candidates_mask &= ~active_table["drop"].astype(bool)
        candidate_count = int(candidates_mask.sum())
        st.info(f"Candidate {unit_name} in range: {candidate_count}. Requested sample: {int(sample_count)}.")

        if use_group_diagnostics:
            if st.button("Sample groups from duration range"):
                sampled_groups = sample_groups_uniform_by_duration(
                    group_length_table,
                    duration_col=duration_col,
                    min_s=float(sample_min_s),
                    max_s=float(sample_max_s),
                    count=int(sample_count),
                    seed=int(sample_seed),
                    exclude_dropped=bool(exclude_dropped),
                    only_complete_groups=bool(only_complete_groups),
                    strategy=strategy,
                )
                st.session_state.sampled_group_indices = sampled_groups
                st.success(f"Sampled {len(sampled_groups)} groups.")

            sampled_group_indices = list(map(int, st.session_state.get("sampled_group_indices", [])))
            plot_group_length_distribution(
                group_length_table,
                duration_col=duration_col,
                min_s=float(sample_min_s),
                max_s=float(sample_max_s),
                sampled_groups=sampled_group_indices,
            )

            preview_cols = [
                "group_index", "source_episodes", "group_num_episodes", "complete_group",
                "raw_num_frames", "raw_duration_s", "kept_frames", "kept_duration_s",
                "drop_count", "group_drop", "episode_outlier_count", "group_length_outlier_iqr",
            ]
            if sampled_group_indices:
                sampled_df = group_length_table[group_length_table["group_index"].astype(int).isin(sampled_group_indices)].copy()
                st.subheader("Sampled groups preview")
                st.dataframe(sampled_df[preview_cols].sort_values("group_index"), use_container_width=True, hide_index=True, height=240)

                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("Apply sampled groups: drop all other groups", use_container_width=True):
                        apply_group_subset_to_trim_config(
                            st.session_state.trim_config,
                            episode_table,
                            sampled_group_indices,
                            int(group_size),
                            drop_unselected=True,
                            regroup_after_drops=bool(regroup_after_drops),
                        )
                        st.success("Applied group subset. Episodes inside sampled groups are kept; all other episodes are marked drop.")
                        st.rerun()
                with b2:
                    if st.button("Keep sampled groups without dropping others", use_container_width=True):
                        apply_group_subset_to_trim_config(
                            st.session_state.trim_config,
                            episode_table,
                            sampled_group_indices,
                            int(group_size),
                            drop_unselected=False,
                            regroup_after_drops=bool(regroup_after_drops),
                        )
                        st.success("Sampled groups marked as keep. Other groups were not changed.")
                        st.rerun()
                with b3:
                    if st.button("Clear sampled groups", use_container_width=True):
                        st.session_state.sampled_group_indices = []
                        st.rerun()

            st.subheader("Groups sorted by length")
            st.dataframe(group_length_table[preview_cols].sort_values(duration_col, ascending=False), use_container_width=True, hide_index=True, height=320)
        else:
            if st.button("Sample episodes from duration range"):
                sampled = sample_episodes_uniform_by_duration(
                    length_table,
                    duration_col=duration_col,
                    min_s=float(sample_min_s),
                    max_s=float(sample_max_s),
                    count=int(sample_count),
                    seed=int(sample_seed),
                    exclude_dropped=bool(exclude_dropped),
                    strategy=strategy,
                )
                st.session_state.sampled_episode_ids = sampled
                st.success(f"Sampled {len(sampled)} episodes.")

            sampled_episode_ids = list(map(int, st.session_state.get("sampled_episode_ids", [])))
            plot_episode_length_distribution(
                length_table,
                duration_col=duration_col,
                min_s=float(sample_min_s),
                max_s=float(sample_max_s),
                sampled_eps=sampled_episode_ids,
            )

            preview_cols = ["episode_index", "raw_num_frames", "raw_duration_s", "kept_frames", "kept_duration_s", "drop", "length_outlier_iqr"]
            if sampled_episode_ids:
                sampled_df = length_table[length_table["episode_index"].astype(int).isin(sampled_episode_ids)].copy()
                st.subheader("Sampled episodes preview")
                st.dataframe(sampled_df[preview_cols].sort_values("episode_index"), use_container_width=True, hide_index=True, height=240)

                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("Apply sampled subset: drop all others", use_container_width=True):
                        apply_episode_subset_to_trim_config(
                            st.session_state.trim_config,
                            episode_table,
                            sampled_episode_ids,
                            drop_unselected=True,
                        )
                        st.success("Applied subset. Existing trim ranges are preserved for sampled episodes; all unselected episodes are marked drop.")
                        st.rerun()
                with b2:
                    if st.button("Keep sampled without dropping others", use_container_width=True):
                        apply_episode_subset_to_trim_config(
                            st.session_state.trim_config,
                            episode_table,
                            sampled_episode_ids,
                            drop_unselected=False,
                        )
                        st.success("Sampled episodes marked as keep. Other episodes were not changed.")
                        st.rerun()
                with b3:
                    if st.button("Clear sampled list", use_container_width=True):
                        st.session_state.sampled_episode_ids = []
                        st.rerun()

            st.subheader("Episodes sorted by length")
            st.dataframe(length_table[preview_cols].sort_values(duration_col, ascending=False), use_container_width=True, hide_index=True, height=320)

left, right = st.columns([1, 2.2], gap="large")

with left:
    st.subheader("Episodes")

    display_table = episode_table.copy()
    display_table["keep_start"] = display_table["episode_index"].map(lambda e: st.session_state.trim_config[str(int(e))]["start"])
    display_table["keep_end"] = display_table["episode_index"].map(lambda e: st.session_state.trim_config[str(int(e))]["end"])
    display_table["kept_frames"] = display_table["keep_end"] - display_table["keep_start"]
    display_table["drop"] = display_table["episode_index"].map(lambda e: st.session_state.trim_config[str(int(e))].get("drop", False))
    display_table["raw_duration_s"] = display_table["episode_index"].map(
        lambda e: float(length_table.loc[length_table["episode_index"].astype(int) == int(e), "raw_duration_s"].iloc[0])
    )
    display_table["sampled"] = display_table["episode_index"].astype(int).isin(st.session_state.get("sampled_episode_ids", []))

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
        height=280,
    )

    ep_options = episode_table["episode_index"].astype(int).tolist()
    selected_ep = st.selectbox("Select episode", ep_options, index=0)

    row = episode_table.loc[episode_table["episode_index"] == selected_ep].iloc[0]
    ep_key = str(int(selected_ep))
    cfg = st.session_state.trim_config[ep_key]

    cfg["drop"] = st.checkbox("Drop this episode", value=bool(cfg.get("drop", False)))

    trim_range = st.slider(
        "Keep range [start, end]",
        min_value=0,
        max_value=int(row["num_frames"]),
        value=(int(cfg["start"]), int(cfg["end"])),
        step=1,
        key=f"trim_slider_{ep_key}",
    )
    cfg["start"], cfg["end"] = int(trim_range[0]), int(trim_range[1])

    current_frame = st.slider(
        "Current frame",
        min_value=0,
        max_value=max(0, int(row["num_frames"]) - 1),
        value=min(int(cfg["start"]), max(0, int(row["num_frames"]) - 1)),
        step=1,
        key=f"frame_slider_{ep_key}",
    )

    st.divider()
    st.subheader("Long-task groups")
    if episode_groups:
        selected_group_idx = st.selectbox(
            "Select group",
            options=list(range(len(episode_groups))),
            format_func=lambda i: (
                f"Group {i}: ep {episode_groups[i][0]}–{episode_groups[i][-1]} "
                f"({len(episode_groups[i])} episodes)"
            ),
        )
        selected_group_eps = episode_groups[int(selected_group_idx)]
        st.caption("Selected group episodes: " + ", ".join(map(str, selected_group_eps)))
        st.caption("Grouping mode: " + ("filtered valid episodes first" if regroup_after_drops else "raw source episode order"))
    else:
        selected_group_idx = 0
        selected_group_eps = []
        st.warning("No episode groups available.")

    if st.button("Save trim_config.json", use_container_width=True):
        path = src_root / "trim_config.json"
        payload = {
            "source_repo_id": src_repo_id,
            "source_root": str(src_root),
            "trims": st.session_state.trim_config,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        st.success(f"Saved: {path}")

    uploaded = st.file_uploader("Load trim_config.json", type=["json"])
    if uploaded is not None:
        payload = json.loads(uploaded.read().decode("utf-8"))
        loaded = payload.get("trims", payload)
        if isinstance(loaded, dict):
            st.session_state.trim_config.update(loaded)
            st.success("Loaded trim config. Refresh or switch episode to see changes.")

with right:
    st.subheader(f"Episode {selected_ep}")

    global_start = int(row["global_start"])
    global_end = int(row["global_end"])
    n_frames = int(row["num_frames"])

    metrics = compute_action_metrics(src_repo_id, str(src_root), global_start, global_end)
    sugg_start, sugg_end = suggest_trim(metrics, metric_col, float(threshold), int(margin))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames", n_frames)
    c2.metric("Suggested start", sugg_start)
    c3.metric("Suggested end", sugg_end)
    c4.metric("Kept frames", max(0, int(cfg["end"]) - int(cfg["start"])))

    if st.button("Apply suggested trim to this episode"):
        cfg["start"], cfg["end"] = int(sugg_start), int(sugg_end)
        st.rerun()

    plot_metrics(metrics, int(cfg["start"]), int(cfg["end"]), int(current_frame), metric_col)

    sample = ds[global_start + current_frame]
    render_sample_panel(
        sample,
        ds,
        title=f"Static preview · FPS={dataset_fps:g}",
        show_action=True,
        show_lowdim=show_lowdim,
    )

    st.divider()
    st.subheader("Group boundary check")
    try:
        check_sample_for_keys = ds[int(episode_table.iloc[0]["global_start"])] if len(episode_table) else sample
        _img_keys, _ = get_feature_keys(ds, check_sample_for_keys)
    except Exception:
        _img_keys = []
    if selected_group_eps and _img_keys:
        check_cam = st.selectbox("Camera for first-frame check", options=_img_keys, index=0)
        first_ep = int(selected_group_eps[0])
        rows_by_ep_for_check = {int(r["episode_index"]): r for _, r in episode_table.iterrows()}
        if first_ep in rows_by_ep_for_check:
            first_row = rows_by_ep_for_check[first_ep]
            first_global = int(first_row["global_start"])
            first_sample = ds[first_global]
            st.caption(f"First frame of first episode in selected group · group {selected_group_idx}, episode {first_ep}, global index {first_global}")
            try:
                st.image(image_to_numpy(first_sample[check_cam]), caption=f"{check_cam} · ep {first_ep} first frame", use_container_width=True)
            except Exception as exc:
                st.error(f"Failed to show first-frame preview for {check_cam}: {exc}")
        else:
            st.warning("The first episode of the selected group is not found in the episode table.")
    else:
        st.caption("No group/camera available for first-frame boundary check.")

    st.divider()
    st.subheader("Playback")
    play_col1, play_col2 = st.columns(2)

    with play_col1:
        if st.button("▶ Play current episode", use_container_width=True):
            seq = playback_sequence(
                episode_table,
                st.session_state.trim_config,
                [int(selected_ep)],
                bool(playback_use_trimmed),
            )
            play_frames(
                ds,
                seq,
                fps=dataset_fps,
                speed=float(playback_speed),
                show_action=bool(playback_show_action),
                show_lowdim=bool(show_lowdim),
            )

    with play_col2:
        if st.button("▶ Play selected group", use_container_width=True):
            seq = playback_sequence(
                episode_table,
                st.session_state.trim_config,
                list(map(int, selected_group_eps)),
                bool(playback_use_trimmed),
            )
            play_frames(
                ds,
                seq,
                fps=dataset_fps,
                speed=float(playback_speed),
                show_action=bool(playback_show_action),
                show_lowdim=bool(show_lowdim),
            )

st.divider()
st.subheader("Export trimmed dataset")

col_a, col_b, col_c = st.columns([1, 1, 2])
with col_a:
    overwrite = st.checkbox("Overwrite output root", value=False)
with col_b:
    merge_on_export = st.checkbox("Merge groups on export", value=False)
    export_merge_group_size = int(group_size) if merge_on_export else 1
with col_c:
    dry_summary = st.button("Preview export summary")
    do_export = st.button("Export trimmed dataset", type="primary", use_container_width=True)

st.caption(
    "Export mode: "
    + (
        f"merge every {int(group_size)} " + ("valid non-dropped" if regroup_after_drops else "source") + " episodes into one output episode"
        if merge_on_export
        else "keep one output episode per source episode"
    )
    + f" · sampled list size: {len(st.session_state.get('sampled_episode_ids', []))}"
)

with st.expander("Export performance settings", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        export_streaming_encoding = st.checkbox(
            "Use streaming video encoding",
            value=True,
            help="Usually much faster: frames are encoded directly instead of being written as temporary PNGs and encoded at save_episode().",
        )
    with c2:
        export_encoder_threads = st.number_input(
            "FFmpeg encoder threads",
            min_value=0,
            max_value=64,
            value=0,
            step=1,
            help="0 lets the codec decide. Try 4 or 8 if CPU utilization is low.",
        )
    with c3:
        export_image_writer_threads = st.number_input(
            "Image writer threads",
            min_value=0,
            max_value=64,
            value=0,
            step=1,
            help="Fallback for non-streaming export. Usually leave 0 when streaming encoding is enabled.",
        )
    with c4:
        export_batch_encoding_size = st.number_input(
            "Batch encoding size",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
            help="Experimental. Keep 1 unless you know your LeRobot version handles batched video finalization correctly.",
        )

def summarize_export():
    rows = []
    if not merge_on_export:
        for _, r in episode_table.iterrows():
            ep = int(r["episode_index"])
            cfg = st.session_state.trim_config[str(ep)]
            if cfg.get("drop", False):
                rows.append({"output_episode": None, "source_episode": ep, "status": "drop", "start": None, "end": None, "frames": 0})
            else:
                start = int(cfg.get("start", 0))
                end = int(cfg.get("end", int(r["num_frames"])))
                rows.append({"output_episode": ep, "source_episode": ep, "status": "keep", "start": start, "end": end, "frames": max(0, end - start)})
        return pd.DataFrame(rows)

    groups = make_episode_groups(episode_table, int(group_size), st.session_state.trim_config, bool(regroup_after_drops))
    rows_by_ep = {int(r["episode_index"]): r for _, r in episode_table.iterrows()}
    for group_idx, eps in enumerate(groups):
        total = 0
        segments = []
        for ep in eps:
            r = rows_by_ep[int(ep)]
            cfg = st.session_state.trim_config[str(int(ep))]
            if cfg.get("drop", False):
                segments.append(f"{ep}:drop")
                continue
            start = int(cfg.get("start", 0))
            end = int(cfg.get("end", int(r["num_frames"])))
            frames = max(0, end - start)
            total += frames
            segments.append(f"{ep}:[{start},{end})")
        rows.append({"output_episode": group_idx, "source_episodes": ", ".join(segments), "frames": total})
    return pd.DataFrame(rows)


if dry_summary:
    st.dataframe(summarize_export(), use_container_width=True, hide_index=True)

if do_export:
    try:
        export_progress = st.progress(0.0, text="preparing export")
        export_status = st.empty()

        def _progress_cb(done: int, total: int, message: str):
            ratio = min(1.0, max(0.0, done / max(1, total)))
            export_progress.progress(ratio, text=f"{done}/{total} frames · {message}")
            export_status.caption(f"Export progress: {done}/{total} frames ({ratio:.1%})")

        with st.spinner("Exporting trimmed dataset. Streaming encoding should be faster, but this still decodes and re-encodes video frames..."):
            manifest = export_trimmed_dataset(
                src_repo_id=src_repo_id,
                src_root=src_root,
                dst_repo_id=dst_repo_id,
                dst_root=dst_root,
                episode_table=episode_table,
                trim_config=st.session_state.trim_config,
                overwrite=overwrite,
                merge_group_size=export_merge_group_size,
                streaming_encoding=export_streaming_encoding,
                encoder_threads=None if int(export_encoder_threads) <= 0 else int(export_encoder_threads),
                image_writer_threads=int(export_image_writer_threads),
                image_writer_processes=0,
                batch_encoding_size=int(export_batch_encoding_size),
                regroup_after_drops=bool(regroup_after_drops),
                progress_cb=_progress_cb,
            )
        export_progress.progress(1.0, text="export complete")
        st.success(f"Exported to: {dst_root}")
        st.json(manifest)
    except Exception as exc:
        st.exception(exc)
        st.error(
            "Export failed. Your source data was not modified. "
            "If this is caused by LeRobot API changes, run: "
            "python -c \"from lerobot.datasets.lerobot_dataset import LeRobotDataset; import inspect; "
            "print(inspect.signature(LeRobotDataset.create)); print(inspect.signature(LeRobotDataset.add_frame)); "
            "print(inspect.signature(LeRobotDataset.save_episode))\""
        )
