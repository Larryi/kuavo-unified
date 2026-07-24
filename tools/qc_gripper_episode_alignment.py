#!/usr/bin/env python3
"""
Interactive QC for LeRobot gripper state/action episode alignment.

Shows observation.state and action curves for selected gripper dimensions and
lets the user mark episodes as OK/BAD. Outputs a bad episode list and a
trim-config JSON compatible with the episode editor's drop field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def parse_indices(text: str) -> List[int]:
    out: List[int] = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("At least one dimension index is required")
    return out


def load_table(dataset: Path) -> pd.DataFrame:
    files = sorted((dataset / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {dataset / 'data'}")

    parts = []
    for file in files:
        wanted = [
            "episode_index",
            "frame_index",
            "timestamp",
            "index",
            "observation.state",
            "action",
        ]
        try:
            parts.append(pq.read_table(file, columns=wanted).to_pandas())
        except Exception:
            fallback = ["episode_index", "frame_index", "observation.state", "action"]
            parts.append(pq.read_table(file, columns=fallback).to_pandas())
    df = pd.concat(parts, ignore_index=True).sort_values(["episode_index", "frame_index"])
    if "timestamp" not in df.columns:
        df["timestamp"] = df.groupby("episode_index").cumcount().astype(float)
    return df


def load_existing_bad(path: Path) -> set[int]:
    if not path.exists():
        return set()
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text())
        if isinstance(obj, dict) and "bad_episodes" in obj:
            return {int(x) for x in obj["bad_episodes"]}
        if isinstance(obj, list):
            return {int(x) for x in obj}
        return {
            int(k)
            for k, v in obj.items()
            if isinstance(v, dict) and bool(v.get("drop", False))
        }
    vals = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vals.append(int(line.split()[0].strip(",")))
    return set(vals)


def save_outputs(
    output_dir: Path,
    bad_episodes: set[int],
    decisions: Dict[int, str],
    episode_lengths: Dict[int, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bad_sorted = sorted(int(x) for x in bad_episodes)
    (output_dir / "bad_episodes.txt").write_text(
        "\n".join(map(str, bad_sorted)) + ("\n" if bad_sorted else "")
    )
    (output_dir / "qc_decisions.json").write_text(
        json.dumps(
            {
                "bad_episodes": bad_sorted,
                "decisions": {str(k): v for k, v in sorted(decisions.items())},
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    trim_config = {}
    for ep, n in sorted(episode_lengths.items()):
        if ep in bad_episodes:
            trim_config[str(ep)] = {"start": 0, "end": int(n), "drop": True}
    (output_dir / "qc_trim_config.json").write_text(
        json.dumps(trim_config, indent=2, ensure_ascii=False)
    )


def plot_episode(
    ep_df: pd.DataFrame,
    episode: int,
    state_dims: List[int],
    action_dims: List[int],
    title_suffix: str = "",
):
    t = ep_df["timestamp"].to_numpy(dtype=float)
    if len(t) and abs(float(t[0])) > 1e-6:
        t = t - float(t[0])
    states = np.stack(ep_df["observation.state"].to_numpy())
    actions = np.stack(ep_df["action"].to_numpy())

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for dim in state_dims:
        axes[0].plot(t, states[:, dim], label=f"observation.state[{dim}]")
    for dim in action_dims:
        axes[1].plot(t, actions[:, dim], label=f"action[{dim}]")

    axes[0].set_title(f"Episode {episode} gripper state/action alignment{title_suffix}")
    axes[0].set_ylabel("observation.state")
    axes[1].set_ylabel("action")
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--state-dims", default="7,15")
    parser.add_argument("--action-dims", default="7,15")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/lerobot_gripper_qc"))
    parser.add_argument("--start-episode", type=int, default=None)
    parser.add_argument("--episodes", default=None, help="Comma list/ranges, e.g. 0,2,10-20")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-only", action="store_true", help="Save all plots without interactive marking")
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def expand_episode_spec(spec: str, all_eps: List[int]) -> List[int]:
    if not spec:
        return all_eps
    selected: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            selected.update(range(int(a), int(b) + 1))
        else:
            selected.add(int(part))
    return [ep for ep in all_eps if ep in selected]


def main() -> None:
    args = parse_args()
    state_dims = parse_indices(args.state_dims)
    action_dims = parse_indices(args.action_dims)
    df = load_table(args.dataset)

    groups = {int(ep): g.copy() for ep, g in df.groupby("episode_index", sort=True)}
    all_eps = sorted(groups)
    if args.start_episode is not None:
        all_eps = [ep for ep in all_eps if ep >= args.start_episode]
    all_eps = expand_episode_spec(args.episodes, all_eps)
    episode_lengths = {ep: len(g) for ep, g in groups.items()}

    bad_episodes = load_existing_bad(args.resume) if args.resume else set()
    decisions: Dict[int, str] = {}

    print(f"Dataset: {args.dataset}")
    print(f"Episodes to review: {len(all_eps)}")
    print(f"Output dir: {args.output_dir}")
    print("Keys: Enter/space/o=OK, b=BAD, s=skip, p=previous, q=quit")

    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if args.save_only:
        for ep in all_eps:
            fig = plot_episode(groups[ep], ep, state_dims, action_dims)
            out = plot_dir / f"episode_{ep:06d}_gripper_qc.png"
            fig.savefig(out, dpi=args.dpi)
            plt.close(fig)
            print(out)
        return

    idx = 0
    while 0 <= idx < len(all_eps):
        ep = all_eps[idx]
        status = "BAD" if ep in bad_episodes else "unmarked"
        fig = plot_episode(
            groups[ep],
            ep,
            state_dims,
            action_dims,
            title_suffix=f"  [{idx + 1}/{len(all_eps)}; {status}]",
        )
        fig.canvas.manager.set_window_title(f"Episode {ep} QC")
        plt.show(block=False)
        print(f"\nEpisode {ep} [{idx + 1}/{len(all_eps)}], current={status}")
        cmd = input("[Enter/o OK, b BAD, s skip, p prev, q quit] > ").strip().lower()
        plt.close(fig)

        if cmd in ("", "o", "ok", " "):
            bad_episodes.discard(ep)
            decisions[ep] = "ok"
            idx += 1
        elif cmd in ("b", "bad"):
            bad_episodes.add(ep)
            decisions[ep] = "bad"
            idx += 1
        elif cmd in ("s", "skip"):
            decisions.setdefault(ep, "skip")
            idx += 1
        elif cmd in ("p", "prev", "previous"):
            idx = max(0, idx - 1)
        elif cmd in ("q", "quit", "exit"):
            break
        else:
            print(f"Unknown command: {cmd!r}")

        save_outputs(args.output_dir, bad_episodes, decisions, episode_lengths)

    save_outputs(args.output_dir, bad_episodes, decisions, episode_lengths)
    print(f"\nSaved bad episodes: {args.output_dir / 'bad_episodes.txt'}")
    print(f"Saved trim config:  {args.output_dir / 'qc_trim_config.json'}")


if __name__ == "__main__":
    main()
