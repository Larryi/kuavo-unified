from kuavo_train.accelerate_utils import (
    dataloader_worker_options,
    resolve_accelerate_options,
    save_final_policy,
)


def test_accelerate_options_preserve_legacy_amp_default():
    assert resolve_accelerate_options({}, {"use_amp": True}) == {
        "mixed_precision": "fp16",
        "ddp_find_unused_parameters": True,
        "allow_tf32": True,
    }


def test_accelerate_options_accept_bf16_and_explicit_ddp_settings():
    result = resolve_accelerate_options(
        {
            "mixed_precision": "bf16",
            "ddp_find_unused_parameters": False,
            "allow_tf32": False,
        },
        {"use_amp": False},
    )
    assert result == {
        "mixed_precision": "bf16",
        "ddp_find_unused_parameters": False,
        "allow_tf32": False,
    }


def test_accelerate_options_reject_unknown_precision():
    try:
        resolve_accelerate_options({"mixed_precision": "fp32"}, {})
    except ValueError as exc:
        assert "Unsupported mixed precision" in str(exc)
    else:
        raise AssertionError("Expected unsupported precision to be rejected")


def test_dataloader_worker_options_disable_worker_only_settings_at_zero():
    assert dataloader_worker_options({"num_workers": 0}) == {
        "prefetch_factor": None,
        "persistent_workers": False,
    }
    assert dataloader_worker_options(
        {"num_workers": 4, "prefetch_factor": 3, "persistent_workers": False}
    ) == {
        "prefetch_factor": 3,
        "persistent_workers": False,
    }


def test_final_policy_is_saved_once_by_main_process(tmp_path):
    class Policy:
        def __init__(self):
            self.saved = []

        def save_pretrained(self, path):
            self.saved.append(path)

    class Accelerator:
        def __init__(self, is_main_process):
            self.is_main_process = is_main_process
            self.messages = []

        def unwrap_model(self, policy):
            return policy

        def print(self, message):
            self.messages.append(message)

    policy = Policy()
    worker = Accelerator(False)
    main = Accelerator(True)

    assert save_final_policy(worker, policy, tmp_path) is None
    checkpoint = save_final_policy(main, policy, tmp_path)

    assert checkpoint == tmp_path / "epochlast"
    assert policy.saved == [checkpoint]
