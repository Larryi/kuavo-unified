#!/usr/bin/env python3
"""Resolve an official flash-attn wheel for an exact Python/Torch runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import urllib.request
import urllib.error
import zipfile


RELEASE_API = (
    "https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/v{version}"
)
RELEASE_DOWNLOAD = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v{version}/{name}"
)


def inspect_runtime(python: str) -> dict[str, str]:
    code = r"""
import json
import platform
import sys
import torch

print(json.dumps({
    "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    "torch": torch.__version__.split("+", 1)[0],
    "cuda": str(torch.version.cuda or ""),
    "cxx11_abi": "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE",
    "machine": platform.machine(),
    "libc": platform.libc_ver()[0],
    "libc_version": platform.libc_ver()[1],
}))
"""
    result = subprocess.run(
        [python, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def wheel_name(version: str, runtime: dict[str, str]) -> str:
    python_major, python_minor = runtime["python"].split(".", 1)
    torch_major, torch_minor = runtime["torch"].split(".", 2)[:2]
    cuda_major = runtime["cuda"].split(".", 1)[0]
    if not cuda_major:
        raise ValueError("The selected Torch runtime has no CUDA version")
    machine = {"amd64": "x86_64"}.get(runtime["machine"], runtime["machine"])
    if machine not in {"x86_64", "aarch64"}:
        raise ValueError(f"Unsupported wheel architecture: {machine}")
    python_tag = f"cp{python_major}{python_minor}"
    return (
        f"flash_attn-{version}+cu{cuda_major}torch{torch_major}.{torch_minor}"
        f"cxx11abi{runtime['cxx11_abi']}-{python_tag}-{python_tag}"
        f"-linux_{machine}.whl"
    )


def release_asset(version: str, expected_name: str) -> dict[str, str]:
    request = urllib.request.Request(
        RELEASE_API.format(version=version),
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "kuavo-unified-flash-attn-resolver",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            release = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code not in {403, 429}:
            raise
        # GitHub's unauthenticated API is rate limited independently from
        # release downloads. The asset filename is already derived from the
        # exact runtime tuple, so retry the canonical release URL directly.
        return {
            "name": expected_name,
            "browser_download_url": RELEASE_DOWNLOAD.format(
                version=version,
                name=expected_name,
            ),
        }
    matches = [asset for asset in release.get("assets", []) if asset["name"] == expected_name]
    if not matches:
        raise FileNotFoundError(
            f"Official release v{version} has no exact wheel {expected_name}. "
            "Do not install a wheel built for a different Python/Torch/ABI tuple."
        )
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version_tuple(version: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", version)
    if not numbers:
        raise ValueError(f"Cannot parse version: {version!r}")
    return tuple(int(number) for number in numbers)


def required_glibc(path: Path) -> str | None:
    """Return the newest GLIBC symbol version referenced by wheel binaries."""
    required: set[tuple[int, ...]] = set()
    with zipfile.ZipFile(path) as wheel:
        for member in wheel.infolist():
            if not member.filename.endswith(".so"):
                continue
            binary = wheel.read(member)
            for match in re.findall(rb"GLIBC_(\d+(?:\.\d+)+)", binary):
                required.add(version_tuple(match.decode("ascii")))
    if not required:
        return None
    return ".".join(str(part) for part in max(required))


def assert_glibc_compatible(runtime: dict[str, str], requirement: str | None) -> None:
    if requirement is None:
        return
    libc = runtime.get("libc", "")
    installed = runtime.get("libc_version", "")
    if libc != "glibc" or not installed:
        raise RuntimeError(
            f"Wheel requires GLIBC_{requirement}, but target runtime reports "
            f"{libc or 'unknown libc'} {installed or 'unknown version'}."
        )
    if version_tuple(installed) < version_tuple(requirement):
        raise RuntimeError(
            f"Wheel requires GLIBC_{requirement}, but target runtime has glibc "
            f"{installed}. Use a newer base system/cloud image or a wheel built "
            "on a compatible older system; matching Python/Torch/CUDA/ABI alone "
            "is not sufficient."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable, help="Target environment Python")
    parser.add_argument("--version", default="2.8.3")
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = inspect_runtime(args.python)
    expected_name = wheel_name(args.version, runtime)
    output = {"runtime": runtime, "wheel": expected_name}
    if args.dry_run:
        print(json.dumps(output, indent=2))
        return

    asset = release_asset(args.version, expected_name)
    output["url"] = asset["browser_download_url"]
    expected_size = int(asset["size"]) if "size" in asset else None
    if expected_size is not None:
        output["size"] = str(expected_size)
    if args.download_dir is not None:
        args.download_dir.mkdir(parents=True, exist_ok=True)
        destination = args.download_dir / expected_name
        if (
            not destination.is_file()
            or (expected_size is not None and destination.stat().st_size != expected_size)
        ):
            request = urllib.request.Request(
                asset["browser_download_url"],
                headers={"User-Agent": "kuavo-unified-flash-attn-resolver"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                with destination.open("wb") as file:
                    while chunk := response.read(1024 * 1024):
                        file.write(chunk)
        output["path"] = str(destination.resolve())
        output["sha256"] = sha256(destination)
        requirement = required_glibc(destination)
        output["required_glibc"] = requirement
        assert_glibc_compatible(runtime, requirement)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
