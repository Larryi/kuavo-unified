"""Small, dependency-free helpers shared by policy training loops."""


def accumulation_window_size(
    batch_index: int,
    total_batches: int,
    accumulation_steps: int,
) -> int:
    """Return the divisor for this batch's complete or final partial window."""
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be at least 1")
    if not 0 <= batch_index < total_batches:
        raise ValueError("batch_index must refer to an existing batch")
    window_start = (batch_index // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, total_batches - window_start)


def should_optimizer_step(
    batch_index: int,
    total_batches: int,
    accumulation_steps: int,
) -> bool:
    """Step after a full accumulation window and flush the final partial one."""
    accumulation_window_size(batch_index, total_batches, accumulation_steps)
    return (
        (batch_index + 1) % accumulation_steps == 0
        or batch_index + 1 == total_batches
    )
