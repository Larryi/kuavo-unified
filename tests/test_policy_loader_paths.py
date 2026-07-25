import json

from kuavo_deploy.utils.policy_loader import resolve_processor_root


def test_processor_root_accepts_run_directory(tmp_path):
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (tmp_path / name).write_text("{}")

    assert resolve_processor_root(tmp_path) == tmp_path.resolve()


def test_processor_root_walks_up_from_epoch_checkpoint(tmp_path):
    run_dir = tmp_path / "run_123"
    epoch_dir = run_dir / "epoch20"
    epoch_dir.mkdir(parents=True)
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (run_dir / name).write_text("{}")

    assert resolve_processor_root(epoch_dir) == run_dir.resolve()


def test_processor_root_skips_nonportable_training_snapshot(tmp_path):
    run_dir = tmp_path / "run_123"
    epoch_dir = run_dir / "epochbest"
    epoch_dir.mkdir(parents=True)

    portable = {"name": "policy_preprocessor", "steps": []}
    nonportable = {
        "name": "policy_preprocessor",
        "steps": [
            {
                "class": "__main__.AugmentationProcessorStep",
                "config": {},
            }
        ],
    }
    (run_dir / "policy_preprocessor.json").write_text(json.dumps(portable))
    (run_dir / "policy_postprocessor.json").write_text(json.dumps(portable))
    (epoch_dir / "policy_preprocessor.json").write_text(json.dumps(nonportable))
    (epoch_dir / "policy_postprocessor.json").write_text(json.dumps(portable))

    assert resolve_processor_root(epoch_dir) == run_dir.resolve()


def test_processor_root_rejects_incomplete_artifact(tmp_path):
    (tmp_path / "policy_preprocessor.json").write_text("{}")

    try:
        resolve_processor_root(tmp_path)
    except FileNotFoundError as exc:
        assert "policy_postprocessor.json" in str(exc)
    else:
        raise AssertionError("Incomplete processor artifacts must be rejected")
