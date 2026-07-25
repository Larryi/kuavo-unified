#!/usr/bin/env python3
"""Start an isolated ACT, DP, or LingBot policy worker."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kuavo_deploy.kuavo_service.policy_worker import load_local_worker
from kuavo_policy_protocol import WebSocketPolicyServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        required=True,
        choices=("act", "diffusion", "lingbot", "lingbot_v2"),
    )
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key-env", default="KUAVO_POLICY_API_KEY")
    parser.add_argument("--lingbot-root", default="")
    parser.add_argument("--qwen-path", default="")
    parser.add_argument("--robot-name", default="")
    parser.add_argument("--norm-stats-file", default="")
    parser.add_argument("--task-prompt", default="")
    parser.add_argument("--use-compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worker = load_local_worker(
        backend=args.backend,
        policy_path=args.policy_path,
        device=args.device,
        lingbot_root=args.lingbot_root,
        qwen_path=args.qwen_path,
        robot_name=args.robot_name,
        norm_stats_file=args.norm_stats_file,
        task_prompt=args.task_prompt,
        use_compile=args.use_compile,
    )
    api_key = os.getenv(args.api_key_env) if args.api_key_env else None
    server = WebSocketPolicyServer(
        worker,
        host=args.host,
        port=args.port,
        metadata=worker.metadata,
        api_key=api_key,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
