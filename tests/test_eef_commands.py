import numpy as np

from kuavo_deploy.utils.eef_commands import normalized_leju_claw_positions


def test_leju_claw_positions_cover_both_normalized_command_bounds():
    assert np.array_equal(normalized_leju_claw_positions(0.0, 1.0), np.array([0.0, 80.0]))
    assert np.array_equal(normalized_leju_claw_positions(1.0, 0.0), np.array([80.0, 0.0]))


def test_leju_claw_positions_reject_out_of_range_commands():
    for command in ((-0.01, 0.0), (0.0, 1.01)):
        try:
            normalized_leju_claw_positions(*command)
        except ValueError as exc:
            assert "normalized to [0, 1]" in str(exc)
        else:
            raise AssertionError(f"Expected {command} to be rejected")
