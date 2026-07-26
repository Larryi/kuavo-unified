#!/usr/bin/env python3
"""Bootstrap a VastAI SSH endpoint from Git, with explicit rsync fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
SSH_OPTIONS_WITH_VALUE = {
    "-B", "-b", "-c", "-D", "-E", "-e", "-F", "-I", "-i", "-J", "-L",
    "-l", "-m", "-O", "-o", "-p", "-Q", "-R", "-S", "-W", "-w",
}
SSH_FLAG_OPTIONS = {"-4", "-6", "-A", "-a", "-C", "-f", "-G", "-g", "-K", "-k", "-M", "-N", "-n", "-q", "-s", "-T", "-t", "-V", "-v", "-X", "-x", "-Y", "-y"}
SAFE_REMOTE_PATH = re.compile(r"^/[A-Za-z0-9._/-]+$")


class BootstrapError(ValueError):
    pass


def parse_ssh_command(command: str) -> tuple[list[str], str]:
    """Return SSH options and user@host, accepting options around the target."""
    tokens = shlex.split(command)
    if not tokens or Path(tokens[0]).name != "ssh":
        raise BootstrapError("SSH command must start with ssh")
    options: list[str] = []
    target = ""
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token in SSH_OPTIONS_WITH_VALUE:
            if index + 1 >= len(tokens):
                raise BootstrapError(f"{token} requires a value")
            options.extend((token, tokens[index + 1]))
            index += 2
            continue
        if token in SSH_FLAG_OPTIONS:
            options.append(token)
            index += 1
            continue
        if token.startswith("-p") and token != "-p":
            options.extend(("-p", token[2:]))
            index += 1
            continue
        if token.startswith("-") and "=" in token:
            options.append(token)
            index += 1
            continue
        if token.startswith("-"):
            raise BootstrapError(f"Unsupported SSH option: {token}")
        if target:
            raise BootstrapError(
                "Remote shell commands are not accepted in --ssh-command; "
                f"unexpected token: {token}"
            )
        target = token
        index += 1
    if not target or target.startswith("-") or "/" in target:
        raise BootstrapError(f"Invalid SSH target: {target!r}")
    return options, target


def options_without_forwarding(options: list[str]) -> list[str]:
    """Remove tunnel-only options from short-lived rsync SSH transports."""
    result: list[str] = []
    index = 0
    while index < len(options):
        token = options[index]
        if token in {"-L", "-R", "-D"}:
            index += 2
            continue
        if token == "-N":
            index += 1
            continue
        result.append(token)
        if token in SSH_OPTIONS_WITH_VALUE:
            result.append(options[index + 1])
            index += 2
        else:
            index += 1
    return result


def git_output(*args: str, cwd: Path = REPO_ROOT) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def build_sync_manifest() -> dict:
    submodules = []
    raw = git_output("submodule", "status", "--recursive")
    for line in raw.splitlines():
        if not line:
            continue
        state = line[0]
        fields = line[1:].split()
        path = fields[1]
        module_root = REPO_ROOT / path
        submodules.append(
            {
                "path": path,
                "commit": fields[0],
                "state": state,
                "dirty_paths": git_output("status", "--short", cwd=module_root).splitlines(),
            }
        )
    return {
        "schema_version": 1,
        "main_commit": git_output("rev-parse", "HEAD"),
        "branch": git_output("branch", "--show-current"),
        "dirty_paths": git_output("status", "--short").splitlines(),
        "submodules": submodules,
        "sync_mode": "rsync-working-tree-including-submodules",
    }


def verify_git_ready(repo_url: str, git_ref: str) -> None:
    """Require the main ref and every pinned submodule commit to be published."""
    dirty = git_output("status", "--short")
    if dirty:
        raise BootstrapError(
            "Local repository is not clean. Commit and push it before Git bootstrap, "
            "or explicitly use --sync-working-tree."
        )
    local_head = git_output("rev-parse", "HEAD")
    remote = subprocess.run(
        ["git", "ls-remote", "--heads", repo_url, git_ref],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    remote_heads = {line.split()[0] for line in remote if line.strip()}
    if local_head not in remote_heads:
        raise BootstrapError(
            f"Local HEAD {local_head} is not the published tip of {repo_url} branch "
            f"{git_ref}; push it before bootstrapping VastAI."
        )

    entries = git_output(
        "config", "--file", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$"
    )
    for line in entries.splitlines():
        key, path = line.split(maxsplit=1)
        name = key.removeprefix("submodule.").removesuffix(".path")
        url = git_output(
            "config", "--file", ".gitmodules", "--get", f"submodule.{name}.url"
        )
        commit = git_output("rev-parse", "HEAD", cwd=REPO_ROOT / path)
        reachable = subprocess.run(
            [
                "git", "-C", str(REPO_ROOT / path), "fetch", "--dry-run",
                "--depth=1", url, commit,
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if reachable.returncode:
            raise BootstrapError(
                f"Submodule {path} commit {commit} is not fetchable from "
                f"{url}; push the fork branch before bootstrapping VastAI."
            )


def run(command: list[str], *, dry_run: bool) -> None:
    print("+", shlex.join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh-command", required=True)
    parser.add_argument("--remote-root", default="/workspace/kuavo_unified_stack")
    parser.add_argument("--repo-url")
    parser.add_argument("--git-ref")
    parser.add_argument(
        "--sync-working-tree",
        action="store_true",
        help="Explicit fallback for local changes that are not available from Git.",
    )
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ssh_options, target = parse_ssh_command(args.ssh_command)
    if not SAFE_REMOTE_PATH.fullmatch(args.remote_root):
        raise SystemExit(f"Unsafe --remote-root: {args.remote_root}")

    sync_ssh_options = options_without_forwarding(ssh_options)
    ssh_transport = shlex.join(["ssh", *sync_ssh_options])
    if not args.sync_working_tree:
        repo_url = args.repo_url or git_output("remote", "get-url", "personal")
        git_ref = args.git_ref or git_output("branch", "--show-current")
        if not repo_url or not git_ref:
            raise SystemExit("--repo-url and --git-ref are required outside a named local branch")
        if args.dry_run:
            print("Git publication preflight skipped in dry-run mode.")
        else:
            verify_git_ready(repo_url, git_ref)
        remote_script = REPO_ROOT / "scripts/vast/remote_clone_and_restore.sh"
        remote_bootstrap = "/tmp/kuavo-vast-remote-clone.sh"
        run(
            [
                "rsync", "-az", "-e", ssh_transport, str(remote_script),
                f"{target}:{remote_bootstrap}",
            ],
            dry_run=args.dry_run,
        )
        remote_command = [
            "bash", remote_bootstrap,
            "--repo-url", repo_url,
            "--git-ref", git_ref,
            "--remote-root", args.remote_root,
        ]
        if args.no_launch:
            remote_command.append("--no-launch")
        run(
            ["ssh", "-t", *ssh_options, target, shlex.join(remote_command)],
            dry_run=args.dry_run,
        )
        print(
            f"Uploaded one bootstrap script; remote Git will resolve {repo_url}@{git_ref} "
            "and all pinned submodules."
        )
        return 0

    print("Explicit fallback enabled: syncing the local working tree and submodule contents.")
    excludes = [
        ".git",
        ".secrets",
        ".venv",
        "__pycache__",
        "outputs",
        "checkpoints",
        "wandb",
        "*.env",
        "*.pem",
        "*.key",
        "*.tar",
        "*.tar.gz",
    ]
    mkdir_command = f"mkdir -p {shlex.quote(args.remote_root)}"
    run(["ssh", *sync_ssh_options, target, mkdir_command], dry_run=args.dry_run)

    rsync = ["rsync", "-az", "--info=progress2", "-e", ssh_transport]
    for pattern in excludes:
        rsync.extend(("--exclude", pattern))
    rsync.extend((f"{REPO_ROOT}/", f"{target}:{args.remote_root}/"))
    run(rsync, dry_run=args.dry_run)

    with tempfile.TemporaryDirectory(prefix="kuavo-vast-sync-") as temp:
        manifest = Path(temp) / "sync_manifest.json"
        manifest.write_text(
            json.dumps(build_sync_manifest(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        run(
            [
                "rsync", "-az", "-e", ssh_transport, str(manifest),
                f"{target}:{args.remote_root}/.vast_sync_manifest.json",
            ],
            dry_run=args.dry_run,
        )

    print(f"Synced local main worktree and submodule contents to {target}:{args.remote_root}")
    if not args.no_launch:
        remote = (
            f"cd {shlex.quote(args.remote_root)} && "
            "exec bash scripts/vast/restore_and_launch.sh"
        )
        run(["ssh", "-t", *ssh_options, target, remote], dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
