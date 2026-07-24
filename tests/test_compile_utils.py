from types import SimpleNamespace

import torch

from kuavo_train.compile_utils import maybe_compile_policy


class Config(dict):
    pass


def make_config(*, enabled=True, policy_name="diffusion", components=("compute_loss",)):
    return Config(
        policy_name=policy_name,
        training={
            "torch_compile": {
                "enabled": enabled,
                "components": list(components),
                "mode": "default",
                "fullgraph": False,
                "dynamic": False,
                "backend": None,
            }
        },
    )


def test_compile_disabled_is_a_noop():
    policy = SimpleNamespace(diffusion=SimpleNamespace(compute_loss=lambda: "eager"))
    assert maybe_compile_policy(policy, make_config(enabled=False)) == 0
    assert policy.diffusion.compute_loss() == "eager"


def test_compile_wraps_selected_callable_and_module_list():
    original_compile = torch.compile
    compiled_labels = []

    def fake_compile(fn, **kwargs):
        compiled_labels.append((fn, kwargs))
        return lambda *args, **call_kwargs: ("compiled", fn(*args, **call_kwargs))

    torch.compile = fake_compile
    try:
        diffusion = SimpleNamespace(
            compute_loss=lambda: "loss",
            rgb_encoder=torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()]),
        )
        policy = SimpleNamespace(diffusion=diffusion)
        count = maybe_compile_policy(
            policy,
            make_config(components=("compute_loss", "rgb_encoder")),
        )
    finally:
        torch.compile = original_compile

    assert count == 3
    assert diffusion.compute_loss() == ("compiled", "loss")
    assert len(compiled_labels) == 3


def test_compile_failure_falls_back_to_eager_callable():
    original_compile = torch.compile
    messages = []

    def failed_compile(_fn, **_kwargs):
        raise RuntimeError("unsupported graph")

    eager = lambda: "eager"
    diffusion = SimpleNamespace(compute_loss=eager)
    torch.compile = failed_compile
    try:
        count = maybe_compile_policy(
            SimpleNamespace(diffusion=diffusion),
            make_config(),
            log=messages.append,
        )
    finally:
        torch.compile = original_compile

    assert count == 0
    assert diffusion.compute_loss is eager
    assert any("using eager mode" in message for message in messages)
