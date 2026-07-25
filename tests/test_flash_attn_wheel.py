from __future__ import annotations

import pytest

from docker.flash_attn_wheel import assert_glibc_compatible, version_tuple, wheel_name


def test_lingbot_v2_official_wheel_tuple() -> None:
    assert wheel_name(
        "2.8.3",
        {
            "python": "3.12",
            "torch": "2.8.0",
            "cuda": "12.8",
            "cxx11_abi": "TRUE",
            "machine": "x86_64",
        },
    ) == (
        "flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-"
        "cp312-cp312-linux_x86_64.whl"
    )


def test_lingbot_v1_wheel_uses_torch_cuda_not_driver_cuda() -> None:
    assert wheel_name(
        "2.8.3",
        {
            "python": "3.10",
            "torch": "2.7.1",
            "cuda": "12.6",
            "cxx11_abi": "TRUE",
            "machine": "amd64",
        },
    ).startswith("flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310")


def test_cpu_torch_is_rejected() -> None:
    with pytest.raises(ValueError, match="no CUDA"):
        wheel_name(
            "2.8.3",
            {
                "python": "3.12",
                "torch": "2.8.0",
                "cuda": "",
                "cxx11_abi": "TRUE",
                "machine": "x86_64",
            },
        )


def test_glibc_version_comparison_is_numeric() -> None:
    assert version_tuple("2.9") < version_tuple("2.32")
    assert_glibc_compatible(
        {"libc": "glibc", "libc_version": "2.35"},
        "2.32",
    )


def test_old_glibc_is_rejected() -> None:
    with pytest.raises(RuntimeError, match=r"requires GLIBC_2\.32.*glibc 2\.31"):
        assert_glibc_compatible(
            {"libc": "glibc", "libc_version": "2.31"},
            "2.32",
        )
