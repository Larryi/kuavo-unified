#!/usr/bin/env python3
"""Interactive Streamlit viewer for asynchronous STATE/action log lines.

Run:
  conda run -n kdc_dev streamlit run tools/state_action_log_viewer.py -- \
    --log /path/to/state_action.log
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


DEFAULT_LOG = Path("state_action.log")
LINE_RE = re.compile(r"^\s*(STATE|clip\s+action|action)\s*:\s*\[([^\]]*)\]", re.IGNORECASE)


def cli_log_path() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args, _ = parser.parse_known_args()
    return args.log


@st.cache_data(show_spinner=False)
def parse_log(path_str: str, mtime_ns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    del mtime_ns  # cache invalidation input
    state_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    clip_action_rows: list[np.ndarray] = []
    skipped = 0
    malformed = 0
    for line in Path(path_str).read_text(encoding="utf-8", errors="replace").splitlines():
        match = LINE_RE.match(line)
        if not match:
            skipped += 1
            continue
        values = np.fromstring(match.group(2).replace(",", " "), sep=" ")
        if values.size == 0:
            malformed += 1
            continue
        prefix = " ".join(match.group(1).lower().split())
        if prefix == "state":
            state_rows.append(values)
        elif prefix == "clip action":
            clip_action_rows.append(values)
        else:
            action_rows.append(values)

    def stack_consistent(rows: list[np.ndarray], name: str) -> np.ndarray:
        if not rows:
            return np.empty((0, 0))
        widths = pd.Series([row.size for row in rows]).value_counts()
        width = int(widths.index[0])
        dropped = sum(row.size != width for row in rows)
        if dropped:
            st.warning(f"{name}: ignored {dropped} rows whose dimension differs from the common width {width}.")
        return np.stack([row for row in rows if row.size == width])

    return (stack_consistent(state_rows, "STATE"), stack_consistent(action_rows, "action"),
            stack_consistent(clip_action_rows, "clip action"), {
        "skipped": skipped, "malformed": malformed,
    })


def plot_signal(name: str, values: np.ndarray, dims: list[int], frame_range: tuple[int, int], separate: bool) -> None:
    start, end = frame_range
    x = np.arange(start, end + 1)
    selected = values[start:end + 1]
    if separate:
        fig, axes = plt.subplots(len(dims), 1, figsize=(11, max(2.2, 2.0 * len(dims))), dpi=120,
                                 sharex=True, squeeze=False)
        for ax, dim in zip(axes[:, 0], dims):
            ax.plot(x, selected[:, dim], linewidth=1.1, color=f"C{dim % 10}")
            ax.set_ylabel(f"{name}[{dim}]")
            ax.grid(True, alpha=0.25)
        axes[-1, 0].set_xlabel(f"{name} frame index")
    else:
        fig, ax = plt.subplots(figsize=(11, 4.2), dpi=120)
        for dim in dims:
            ax.plot(x, selected[:, dim], linewidth=1.1, label=f"{name}[{dim}]")
        ax.set_xlabel(f"{name} frame index")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=min(4, len(dims)))
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def signal_panel(name: str, values: np.ndarray, key_suffix: str = "") -> None:
    widget_key = f"{name}_{key_suffix}" if key_suffix else name
    st.subheader(name)
    if values.size == 0:
        st.warning(f"No {name} lines found.")
        return
    frames, width = values.shape
    c1, c2, c3 = st.columns(3)
    c1.metric("Frames", frames)
    c2.metric("Dimensions", width)
    c3.metric("NaN values", int(np.isnan(values).sum()))

    default_dims = list(range(min(width, 8)))
    dims = st.multiselect(
        f"{name} dimensions", range(width), default=default_dims,
        format_func=lambda i: f"{name}[{i}]", key=f"dims_{widget_key}",
    )
    frame_range = st.slider(
        f"{name} frame range", 0, frames - 1, (0, frames - 1), key=f"range_{widget_key}",
    )
    separate = st.checkbox("One subplot per dimension", value=False, key=f"separate_{widget_key}")
    if dims:
        plot_signal(name, values, dims, frame_range, separate)
        stats = pd.DataFrame(values[frame_range[0]:frame_range[1] + 1, dims], columns=[f"{name}[{d}]" for d in dims])
        with st.expander(f"{name} statistics for selected range"):
            st.dataframe(stats.describe().T, use_container_width=True)
    else:
        st.info("Select at least one dimension.")


def action_comparison_panel(action: np.ndarray, clip_action: np.ndarray, key_suffix: str = "") -> None:
    st.subheader("action vs clip action")
    if action.size == 0 or clip_action.size == 0:
        st.warning("Both action and clip action are required for comparison.")
        signal_panel("action", action, key_suffix)
        return
    width = min(action.shape[1], clip_action.shape[1])
    c1, c2, c3 = st.columns(3)
    c1.metric("action frames", action.shape[0])
    c2.metric("clip action frames", clip_action.shape[0])
    c3.metric("Comparable dimensions", width)
    dims = st.multiselect(
        "Action dimensions", range(width), default=list(range(min(width, 8))),
        format_func=lambda i: f"action[{i}]", key=f"compare_dims_{key_suffix}",
    )
    max_frame = max(action.shape[0], clip_action.shape[0]) - 1
    start, end = st.slider(
        "Action frame range", 0, max_frame, (0, max_frame), key=f"compare_range_{key_suffix}",
        help="Each sequence uses its own occurrence index; the shorter sequence simply ends earlier.",
    )
    if not dims:
        st.info("Select at least one dimension.")
        return
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=120)
    for dim in dims:
        color = f"C{dim % 10}"
        a_end = min(end + 1, action.shape[0])
        c_end = min(end + 1, clip_action.shape[0])
        if start < a_end:
            ax.plot(np.arange(start, a_end), action[start:a_end, dim], color=color,
                    linewidth=1.1, label=f"action[{dim}]")
        if start < c_end:
            ax.plot(np.arange(start, c_end), clip_action[start:c_end, dim], color=color,
                    linewidth=1.1, linestyle="--", label=f"clip action[{dim}]")
    ax.set_xlabel("Independent occurrence frame index")
    ax.set_ylabel("Action value")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=min(4, max(1, len(dims))))
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


st.set_page_config(page_title="STATE / action Log Viewer", layout="wide")
st.title("STATE / action Log Viewer")
st.caption("STATE and action are numbered independently in their own occurrence order; no synchronization is assumed.")

default_path = cli_log_path()
path = Path(st.text_input("Log path", str(default_path)))
if not path.is_file():
    st.error(f"Log file does not exist: {path}")
    st.stop()

state, action, clip_action, parse_info = parse_log(str(path), path.stat().st_mtime_ns)
st.caption(
    f"Parsed {len(state)} STATE rows, {len(action)} action rows, and {len(clip_action)} clip action rows "
    f"· ignored unrelated lines: {parse_info['skipped']}"
)

state_tab, action_tab, clip_tab, side_tab = st.tabs(["STATE", "action comparison", "clip action", "Side by side"])
with state_tab:
    signal_panel("STATE", state)
with action_tab:
    action_comparison_panel(action, clip_action, "main")
with clip_tab:
    signal_panel("clip action", clip_action)
with side_tab:
    left, right = st.columns(2)
    with left:
        signal_panel("STATE", state, "side")
    with right:
        action_comparison_panel(action, clip_action, "side")
