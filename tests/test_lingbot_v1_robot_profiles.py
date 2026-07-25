from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _arm_subtract_state(profile: str) -> bool:
    config = yaml.safe_load(
        (ROOT / "configs/robot_configs" / f"{profile}.yaml").read_text(encoding="utf-8")
    )
    arm = config["actions"][0]["action.arm.position"]
    return bool(arm["subtract_state"])


def test_relative_and_legacy_absolute_profiles_are_explicitly_separate() -> None:
    assert _arm_subtract_state("kuavo_v1_right_arm")
    assert not _arm_subtract_state("kuavo_v1_right_arm_absolute")


def test_current_lingbot_deploy_config_uses_legacy_absolute_profile() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/deploy/kuavo_env.lingbot.yaml").read_text(encoding="utf-8")
    )
    assert config["inference"]["lingbot_robot_name"] == "kuavo_v1_right_arm_absolute"
