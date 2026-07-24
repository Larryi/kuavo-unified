"""Runtime dependency checks for LingBot-VLA v2 training."""

from __future__ import annotations

import importlib


UTILS3D_INSTALL_COMMAND = (
    "python -m pip uninstall -y utils3d && "
    "python -m pip install --no-deps "
    "'git+https://github.com/EasternJournalist/utils3d.git@"
    "3fab839f0be9931dac7c8488eb0e1600c236e183'"
)


def validate_utils3d() -> None:
    """Reject the unrelated PyPI package that shadows MoGe's utils3d."""

    try:
        utils3d = importlib.import_module("utils3d")
    except ImportError as exc:
        raise RuntimeError(
            "LingBot-VLA v2 Depth/MoGe requires EasternJournalist/utils3d. "
            f"Install it with: {UTILS3D_INSTALL_COMMAND}"
        ) from exc

    pt = getattr(utils3d, "pt", None)
    if pt is None or not hasattr(pt, "intrinsics_from_focal_center"):
        module_path = getattr(utils3d, "__file__", "unknown")
        version = getattr(utils3d, "__version__", "unknown")
        raise RuntimeError(
            "An incompatible package named 'utils3d' is installed "
            f"(version={version}, path={module_path}). LingBot-VLA v2 requires "
            "the MoGe utils3d implementation, not PyPI utils3d==0.1.1. "
            f"Repair the environment with: {UTILS3D_INSTALL_COMMAND}"
        )
