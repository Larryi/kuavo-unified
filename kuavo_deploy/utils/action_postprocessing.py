"""Causal action post-processors shared by deployment diagnostics and viewers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch


def _indices_tensor(indices: Iterable[int], action_dim: int, device: torch.device) -> torch.Tensor:
    values = tuple(dict.fromkeys(int(index) for index in indices))
    if not values:
        raise ValueError("Enter at least one joint action index.")
    if any(index < 0 or index >= action_dim for index in values):
        raise ValueError(f"Joint action indices {values} are outside action_dim={action_dim}.")
    return torch.as_tensor(values, dtype=torch.long, device=device)


def _limit_tensor(values: Iterable[float], count: int, like: torch.Tensor, name: str) -> torch.Tensor:
    result = torch.as_tensor(tuple(float(value) for value in values), dtype=like.dtype, device=like.device)
    if result.numel() != count:
        raise ValueError(f"{name} must contain {count} values, got {result.numel()}.")
    if not torch.isfinite(result).all() or torch.any(result < 0):
        raise ValueError(f"{name} must contain finite non-negative values.")
    return result


@dataclass(frozen=True)
class JointRateLimitConfig:
    action_indices: tuple[int, ...]
    max_delta: tuple[float, ...]
    max_second_delta: Optional[tuple[float, ...]] = None


class CausalJointRateLimiter:
    """Project joint targets onto per-step velocity and optional acceleration limits."""

    def __init__(self, config: JointRateLimitConfig):
        self.config = config
        self.previous_action: Optional[torch.Tensor] = None
        self.previous_delta: Optional[torch.Tensor] = None

    def reset(self, initial_action: Optional[torch.Tensor] = None) -> None:
        self.previous_action = None if initial_action is None else initial_action.detach().clone().flatten()
        self.previous_delta = None

    def process_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        if chunk.ndim != 2:
            raise ValueError(f"Expected [time, action_dim], got {tuple(chunk.shape)}")
        result = chunk.clone()
        indices = _indices_tensor(self.config.action_indices, chunk.shape[-1], chunk.device)
        max_delta = _limit_tensor(self.config.max_delta, len(indices), chunk, "max_delta")
        max_second_delta = None
        if self.config.max_second_delta is not None:
            max_second_delta = _limit_tensor(
                self.config.max_second_delta, len(indices), chunk, "max_second_delta"
            )

        if self.previous_action is None:
            self.previous_action = chunk[0].detach().clone()
        else:
            self.previous_action = self.previous_action.to(dtype=chunk.dtype, device=chunk.device)
        if self.previous_action.numel() != chunk.shape[-1]:
            raise ValueError(
                f"Initial action dimension {self.previous_action.numel()} does not match {chunk.shape[-1]}."
            )
        if self.previous_delta is None:
            self.previous_delta = torch.zeros_like(max_delta)
        else:
            self.previous_delta = self.previous_delta.to(dtype=chunk.dtype, device=chunk.device)

        previous = self.previous_action.clone()
        previous_delta = self.previous_delta.clone()
        for row in range(len(chunk)):
            desired_delta = torch.clamp(chunk[row, indices] - previous[indices], -max_delta, max_delta)
            if max_second_delta is not None:
                desired_delta = torch.maximum(
                    torch.minimum(desired_delta, previous_delta + max_second_delta),
                    previous_delta - max_second_delta,
                )
            result[row, indices] = previous[indices] + desired_delta
            previous = result[row].detach().clone()
            previous_delta = desired_delta.detach().clone()

        self.previous_action = previous
        self.previous_delta = previous_delta
        return result


@dataclass(frozen=True)
class ChunkBoundaryBlendConfig:
    action_indices: tuple[int, ...]
    blend_steps: int = 4


class CausalChunkBoundaryBlender:
    """Blend the prefix of each newly available chunk from the last emitted target."""

    def __init__(self, config: ChunkBoundaryBlendConfig):
        if config.blend_steps < 1:
            raise ValueError("blend_steps must be at least 1.")
        self.config = config
        self.previous_action: Optional[torch.Tensor] = None

    def reset(self, initial_action: Optional[torch.Tensor] = None) -> None:
        self.previous_action = None if initial_action is None else initial_action.detach().clone().flatten()

    def process_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        if chunk.ndim != 2:
            raise ValueError(f"Expected [time, action_dim], got {tuple(chunk.shape)}")
        result = chunk.clone()
        indices = _indices_tensor(self.config.action_indices, chunk.shape[-1], chunk.device)
        if self.previous_action is None:
            self.previous_action = chunk[0].detach().clone()
        else:
            self.previous_action = self.previous_action.to(dtype=chunk.dtype, device=chunk.device)
        if self.previous_action.numel() != chunk.shape[-1]:
            raise ValueError(
                f"Initial action dimension {self.previous_action.numel()} does not match {chunk.shape[-1]}."
            )

        anchor = self.previous_action[indices]
        prefix = min(self.config.blend_steps, len(chunk))
        for row in range(prefix):
            weight = float(row + 1) / float(self.config.blend_steps)
            result[row, indices] = (1.0 - weight) * anchor + weight * chunk[row, indices]
        self.previous_action = result[-1].detach().clone()
        return result
