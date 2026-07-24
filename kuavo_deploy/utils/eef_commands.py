"""Pure end-effector command conversions shared by ROS deployment adapters."""

from __future__ import annotations

import numpy as np


LEJU_CLAW_MAX_POSITION = 80.0


def normalized_leju_claw_positions(left: float, right: float) -> np.ndarray:
    """Convert normalized [0, 1] claw targets to the Leju device's [0, 80] range."""

    values = np.asarray([left, right], dtype=float)
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"Leju claw targets must be normalized to [0, 1], got {values.tolist()}.")
    return values * LEJU_CLAW_MAX_POSITION
