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
    """Convert noisy predictions into independent, causally debounced commands.

    ``intent_steps`` is the number of consecutive opposite-intent actions that
    must be observed before switching state. Evidence is retained across calls,
    so this behaves identically for full chunks and one-action deployment calls.
    """

    def __init__(self, config: GripperLatchConfig):
        self.config = config
        self.closed = {index: False for index in config.action_indices}
        self.steps_in_state = {index: config.min_open_steps for index in config.action_indices}
        self.opposite_intent_steps = {index: 0 for index in config.action_indices}

    def reset(self, initial_values: dict[int, float] | None = None) -> None:
        initial_values = initial_values or {}
        for index in self.config.action_indices:
            value = float(initial_values.get(index, self.config.open_value))
            self.closed[index] = value >= 0.5 * (self.config.open_value + self.config.closed_value)
            self.steps_in_state[index] = (
                self.config.min_close_steps if self.closed[index] else self.config.min_open_steps
            )
            self.opposite_intent_steps[index] = 0

    def process_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        if chunk.ndim != 2:
            raise ValueError(f"Expected [time, action_dim], got {tuple(chunk.shape)}")
        if self.config.intent_steps < 1:
            raise ValueError("intent_steps must be at least 1.")
        result = chunk.clone()
        for index in self.config.action_indices:
            if index < 0 or index >= chunk.shape[-1]:
                raise ValueError(f"Gripper action index {index} is outside action_dim={chunk.shape[-1]}.")
            for row in range(len(chunk)):
                value = float(chunk[row, index])
                opposite_intent = (
                    value <= self.config.open_threshold
                    if self.closed[index]
                    else value >= self.config.close_threshold
                )
                if opposite_intent:
                    self.opposite_intent_steps[index] += 1
                else:
                    self.opposite_intent_steps[index] = 0

                minimum_hold = (
                    self.config.min_close_steps if self.closed[index] else self.config.min_open_steps
                )
                if (
                    self.steps_in_state[index] >= minimum_hold
                    and self.opposite_intent_steps[index] >= self.config.intent_steps
                ):
                    self.closed[index] = not self.closed[index]
                    self.steps_in_state[index] = 0
                    self.opposite_intent_steps[index] = 0
                result[row, index] = (
                    self.config.closed_value if self.closed[index] else self.config.open_value
                )
                self.steps_in_state[index] += 1
        return result

    def advance(self, executed_steps: int) -> None:
        """Advance dwell time for steps that did not pass through ``process_chunk``."""
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
