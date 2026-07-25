from types import SimpleNamespace

import torch

from tools.open_loop_eval import (
    build_open_loop_delta_timestamps,
    predict_classic_chunk,
)


def test_queue_mode_does_not_prebatch_observation_history():
    policy = SimpleNamespace(
        config=SimpleNamespace(
            observation_delta_indices=[-1, 0],
            action_delta_indices=list(range(8)),
        )
    )
    metadata = SimpleNamespace(fps=10)

    assert build_open_loop_delta_timestamps(policy, metadata, "queue", "act") is None


def test_chunk_mode_loads_only_future_ground_truth_actions():
    policy = SimpleNamespace(
        config=SimpleNamespace(
            observation_delta_indices=[-1, 0],
            action_delta_indices=[0, 1, 2],
        )
    )
    metadata = SimpleNamespace(fps=10)

    assert build_open_loop_delta_timestamps(policy, metadata, "chunk", "act") == {
        "action": [0.0, 0.1, 0.2],
    }


def test_diffusion_chunk_ground_truth_matches_executable_window():
    policy = SimpleNamespace(
        config=SimpleNamespace(
            action_delta_indices=[-1, 0, 1, 2, 3],
            n_obs_steps=2,
            n_action_steps=3,
        )
    )
    metadata = SimpleNamespace(fps=10)

    assert build_open_loop_delta_timestamps(
        policy, metadata, "chunk", "diffusion"
    ) == {
        "action": [0.0, 0.1, 0.2],
    }


def test_diffusion_chunk_follows_public_action_queue_contract():
    class FakeDiffusion:
        config = SimpleNamespace(n_action_steps=3)

        def reset(self):
            self.step = 0

        def select_action(self, batch):
            del batch
            action = torch.tensor([[self.step, self.step + 1]], dtype=torch.float32)
            self.step += 1
            return action

    chunk = predict_classic_chunk(FakeDiffusion(), {}, "diffusion")

    assert chunk.shape == (1, 3, 2)
    torch.testing.assert_close(
        chunk,
        torch.tensor([[[0, 1], [1, 2], [2, 3]]], dtype=torch.float32),
    )
