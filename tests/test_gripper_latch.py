import torch

from kuavo_deploy.utils.gripper_latch import GripperIntentLatch, GripperLatchConfig


def make_latch(**overrides):
    values = {
        "action_indices": (1,),
        "intent_steps": 3,
        "min_close_steps": 2,
        "min_open_steps": 1,
    }
    values.update(overrides)
    return GripperIntentLatch(GripperLatchConfig(**values))


def test_latch_retains_consecutive_intent_across_single_action_calls():
    latch = make_latch()

    outputs = [
        latch.process_chunk(torch.tensor([[5.0, value]]))[0, 1].item()
        for value in (0.9, 0.8, 0.95)
    ]

    assert outputs == [0.0, 0.0, 1.0]


def test_latch_clears_interrupted_intent_evidence():
    latch = make_latch(action_indices=(0,))

    outputs = [
        latch.process_chunk(torch.tensor([[value]])).item()
        for value in (0.9, 0.8, 0.5, 0.9, 0.8, 0.9)
    ]

    assert outputs == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def test_latch_respects_minimum_hold_before_reopening():
    latch = make_latch(action_indices=(0,), intent_steps=2, min_close_steps=4)
    closed = latch.process_chunk(torch.tensor([[0.9], [0.9]]))
    opening_attempt = latch.process_chunk(torch.tensor([[0.0], [0.0]]))
    opened = latch.process_chunk(torch.tensor([[0.0], [0.0]]))

    assert torch.equal(closed, torch.tensor([[0.0], [1.0]]))
    assert torch.equal(opening_attempt, torch.tensor([[1.0], [1.0]]))
    assert torch.equal(opened, torch.tensor([[1.0], [0.0]]))


def test_latch_reset_seeds_independent_gripper_states():
    latch = make_latch(action_indices=(0, 2), intent_steps=2)
    latch.reset({0: 1.0, 2: 0.0})

    output = latch.process_chunk(torch.tensor([[0.5, 7.0, 0.5]]))

    assert torch.equal(output, torch.tensor([[1.0, 7.0, 0.0]]))
