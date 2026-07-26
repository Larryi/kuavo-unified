"""Manage isolated policy workers required by the standard ROS entrypoint."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from typing import Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]


def _port_is_ready(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _tail(path: Path, line_count: int = 40) -> str:
    try:
        return "\n".join(
            path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]
        )
    except OSError as exc:
        return f"<unable to read {path}: {exc}>"


def _server_command(cfg) -> list[str]:
    backend = cfg.client_autostart_backend
    policy_dir = cfg.pretrained_path
    if not policy_dir:
        raise ValueError(
            "client_autostart requires inference.pretrained_path to point to "
            "the mounted checkpoint"
        )

    if backend == "openpi":
        command = [str(REPO_ROOT / "scripts/kuavo_openpi"), "serve"]
        if int(cfg.client_port) != 8000:
            command.append(f"--port={int(cfg.client_port)}")
        command.extend(
            [
                "policy:checkpoint",
                f"--policy.config={cfg.openpi_policy_config}",
                f"--policy.dir={policy_dir}",
            ]
        )
        return command

    if backend == "lingbot_v2":
        command = [
            "/opt/kuavo-env/bin/python",
            str(REPO_ROOT / "tools/policy_worker.py"),
            "--backend",
            "lingbot_v2",
            "--policy-path",
            policy_dir,
            "--lingbot-root",
            cfg.lingbot_v2_root,
            "--qwen-path",
            cfg.qwen3vl_path,
            "--robot-name",
            cfg.lingbot_v2_robot_name,
            "--norm-stats-file",
            cfg.lingbot_norm_stats_file,
            "--task-prompt",
            cfg.task_prompt or cfg.task,
            "--host",
            cfg.client_host,
            "--port",
            str(cfg.client_port),
        ]
        if cfg.lingbot_v2_use_compile:
            command.append("--use-compile")
        return command

    raise ValueError(
        "client_autostart_backend must be 'openpi' or 'lingbot_v2', "
        f"got {backend!r}"
    )


@contextmanager
def managed_policy_server(cfg, logger) -> Iterator[None]:
    """Start and clean up a local policy server declared by deploy YAML."""

    if cfg.policy_type != "client" or not cfg.client_autostart:
        yield
        return
    if cfg.client_host not in {"127.0.0.1", "localhost"}:
        raise ValueError("client_autostart only supports a localhost policy server")

    host = cfg.client_host
    port = int(cfg.client_port)
    if _port_is_ready(host, port):
        logger.info(
            "Policy server already listening at %s:%s; reusing it without ownership",
            host,
            port,
        )
        yield
        return

    command = _server_command(cfg)
    log_path = Path(cfg.client_server_log).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    logger.info(
        "Starting %s policy server; log: %s",
        cfg.client_autostart_backend,
        log_path,
    )
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    try:
        deadline = time.monotonic() + float(cfg.client_server_startup_timeout_s)
        while not _port_is_ready(host, port):
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"{cfg.client_autostart_backend} policy server exited with "
                    f"status {return_code} before accepting connections.\n"
                    f"Last 40 log lines:\n{_tail(log_path)}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{cfg.client_autostart_backend} policy server did not become "
                    f"ready at {host}:{port} within "
                    f"{cfg.client_server_startup_timeout_s:g}s.\n"
                    f"Last 40 log lines:\n{_tail(log_path)}"
                )
            time.sleep(1.0)
        logger.info(
            "%s policy server is ready at %s:%s",
            cfg.client_autostart_backend,
            host,
            port,
        )
        yield
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        log_file.close()
