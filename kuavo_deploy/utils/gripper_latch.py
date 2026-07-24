"""Policy-independent gripper intent detection and command latching."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GripperLatchConfig:
    action_indices: tuple[int, ...]
    intent_steps: int = 5
    close_threshold: float = 0.7
    open_threshold: float = 0.3
    close_ratio: float = 0.6
    open_ratio: float = 0.8
    min_close_steps: int = 10
    min_open_steps: int = 4
    closed_value: float = 1.0
    open_value: float = 0.0


class GripperIntentLatch:
    """Convert noisy chunk predictions into independent latched gripper commands."""

    def __init__(self, config: GripperLatchConfig):
        self.config = config
        self.closed = {index: False for index in config.action_indices}
        self.steps_in_state = {index: config.min_open_steps for index in config.action_indices}

    def process_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        if chunk.ndim != 2:
            raise ValueError(f"Expected [time, action_dim], got {tuple(chunk.shape)}")
        result = chunk.clone()
        prefix = chunk[: max(1, min(self.config.intent_steps, len(chunk)))]
        for index in self.config.action_indices:
            if index < 0 or index >= chunk.shape[-1]:
                raise ValueError(f"Gripper action index {index} is outside action_dim={chunk.shape[-1]}.")
            values = prefix[:, index]
            close_score = float((values >= self.config.close_threshold).float().mean())
            open_score = float((values <= self.config.open_threshold).float().mean())
            if not self.closed[index]:
                if self.steps_in_state[index] >= self.config.min_open_steps and close_score >= self.config.close_ratio:
                    self.closed[index] = True
                    self.steps_in_state[index] = 0
            elif self.steps_in_state[index] >= self.config.min_close_steps and open_score >= self.config.open_ratio:
                self.closed[index] = False
                self.steps_in_state[index] = 0
            result[:, index] = self.config.closed_value if self.closed[index] else self.config.open_value
        return result

    def advance(self, executed_steps: int) -> None:
        if executed_steps < 0:
            raise ValueError("executed_steps must be non-negative.")
        for index in self.steps_in_state:
            self.steps_in_state[index] += executed_steps


def parse_action_indices(value: str) -> tuple[int, ...]:
    indices = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",") if part.strip()))
    if not indices:
        raise ValueError("Enter at least one gripper action index.")
    if any(index < 0 for index in indices):
        raise ValueError("Gripper action indices must be non-negative.")
    return indices
