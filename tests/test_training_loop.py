import pytest

from kuavo_train.utils.training_loop import (
    accumulation_window_size,
    should_optimizer_step,
)


def test_accumulation_steps_after_full_windows():
    assert [
        should_optimizer_step(index, total_batches=8, accumulation_steps=4)
        for index in range(8)
    ] == [False, False, False, True, False, False, False, True]


def test_accumulation_flushes_and_rescales_final_partial_window():
    assert [
        accumulation_window_size(index, total_batches=6, accumulation_steps=4)
        for index in range(6)
    ] == [4, 4, 4, 4, 2, 2]
    assert [
        should_optimizer_step(index, total_batches=6, accumulation_steps=4)
        for index in range(6)
    ] == [False, False, False, True, False, True]


def test_accumulation_rejects_invalid_values():
    with pytest.raises(ValueError):
        should_optimizer_step(0, total_batches=1, accumulation_steps=0)
