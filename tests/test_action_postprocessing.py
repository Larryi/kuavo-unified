import torch

from kuavo_deploy.utils.action_postprocessing import (
    CausalChunkBoundaryBlender,
    CausalJointRateLimiter,
    ChunkBoundaryBlendConfig,
    JointRateLimitConfig,
)


def test_rate_limiter_preserves_unselected_dimensions_and_state_across_chunks():
    limiter = CausalJointRateLimiter(
        JointRateLimitConfig(action_indices=(0, 2), max_delta=(1.0, 0.5))
    )
    limiter.reset(torch.tensor([0.0, 10.0, 0.0]))

    first = limiter.process_chunk(
        torch.tensor([[3.0, 11.0, -2.0], [3.0, 12.0, -2.0]])
    )
    second = limiter.process_chunk(torch.tensor([[-3.0, 13.0, 2.0]]))

    assert torch.equal(first[:, 1], torch.tensor([11.0, 12.0]))
    assert torch.allclose(first[:, (0, 2)], torch.tensor([[1.0, -0.5], [2.0, -1.0]]))
    assert torch.allclose(second, torch.tensor([[1.0, 13.0, -0.5]]))


def test_rate_limiter_applies_second_delta_limit():
    limiter = CausalJointRateLimiter(
        JointRateLimitConfig(
            action_indices=(0,),
            max_delta=(10.0,),
            max_second_delta=(0.25,),
        )
    )
    limiter.reset(torch.tensor([0.0]))

    result = limiter.process_chunk(torch.tensor([[5.0], [5.0], [-5.0]]))

    assert torch.allclose(result.flatten(), torch.tensor([0.25, 0.75, 1.0]))


def test_boundary_blender_uses_previous_emitted_action_as_anchor():
    blender = CausalChunkBoundaryBlender(
        ChunkBoundaryBlendConfig(action_indices=(0,), blend_steps=2)
    )
    blender.reset(torch.tensor([0.0, 7.0]))

    first = blender.process_chunk(torch.tensor([[2.0, 8.0], [4.0, 9.0], [6.0, 10.0]]))
    second = blender.process_chunk(torch.tensor([[10.0, 11.0], [12.0, 12.0]]))

    assert torch.allclose(first, torch.tensor([[1.0, 8.0], [4.0, 9.0], [6.0, 10.0]]))
    assert torch.allclose(second, torch.tensor([[8.0, 11.0], [12.0, 12.0]]))


def assert_value_error(processor, chunk, message):
    try:
        processor.process_chunk(chunk)
    except ValueError as exc:
        assert message in str(exc)
    else:
        raise AssertionError(f"Expected ValueError containing {message!r}")


def test_action_postprocessors_reject_invalid_configuration():
    cases = [
        (
            CausalJointRateLimiter(JointRateLimitConfig((2,), (1.0,))),
            torch.zeros(2, 2),
            "outside action_dim",
        ),
        (
            CausalJointRateLimiter(JointRateLimitConfig((0, 1), (1.0,))),
            torch.zeros(2, 2),
            "must contain 2 values",
        ),
        (
            CausalChunkBoundaryBlender(ChunkBoundaryBlendConfig((0,), blend_steps=2)),
            torch.zeros(2, 2, 1),
            "Expected",
        ),
    ]
    for processor, chunk, message in cases:
        assert_value_error(processor, chunk, message)
