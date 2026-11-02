#!/usr/bin/env python
"""``torch_compile`` without a ``torch_compile_mode`` compiles in ``default`` mode, on every launch path.

``reduce-overhead`` (CUDA graphs) measures slower on EP MoE, so it runs only when the config names it. The
mixin's compile call decides the mode; a multi-process accelerate-managed launch compiles its DDP wrapper again
with its own dynamo plugin, whose unset mode resolves to the same one.

    python tests/cpu/trainers/test_torch_compile_mode_fallback.py
"""

from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import TorchDynamoPlugin
from transformers import TrainingArguments

from halo.trainers.parallelism.mode_setup import DEFAULT_TORCH_COMPILE_MODE, ModeSetupMixin

_DYNAMO_ENV = ("ACCELERATE_DYNAMO_BACKEND", "ACCELERATE_DYNAMO_MODE")


@pytest.fixture
def compile_calls(monkeypatch):
    calls = []

    def record(model, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(compiled=model)

    monkeypatch.setattr(torch, "compile", record)
    return calls


@pytest.fixture
def clean_dynamo_env(monkeypatch):
    """Clear ACCELERATE_DYNAMO_* vars inherited from the launch env (an ``accelerate launch`` sets them) so
    ``TorchDynamoPlugin`` reads its own default."""
    for key in _DYNAMO_ENV:
        monkeypatch.delenv(key, raising=False)


def _host(args):
    return SimpleNamespace(args=args, model=torch.nn.Linear(2, 2), model_wrapped=None)


def _args(**overrides):
    fields = {
        "torch_compile": True,
        "torch_compile_mode": None,
        "torch_compile_backend": None,
        "packing": False,
        "per_device_train_batch_size": 1,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_unset_mode_compiles_in_default_mode(compile_calls):
    host = _host(_args())
    eager = host.model
    ModeSetupMixin._apply_torch_compile(host)
    assert compile_calls == [{"mode": "default", "backend": "inductor"}]
    assert host.model.compiled is eager and host.model_wrapped is host.model


@pytest.mark.parametrize("mode", ["reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
def test_an_explicit_mode_is_passed_through(compile_calls, mode):
    ModeSetupMixin._apply_torch_compile(_host(_args(torch_compile_mode=mode, torch_compile_backend="aot_eager")))
    assert compile_calls == [{"mode": mode, "backend": "aot_eager"}]


def test_compile_off_does_not_compile(compile_calls):
    host = _host(_args(torch_compile=False))
    eager = host.model
    ModeSetupMixin._apply_torch_compile(host)
    assert compile_calls == [] and host.model is eager


def test_hf_training_arguments_reach_the_default_mode(compile_calls, tmp_path):
    """HF's own config object leaves the mode unset; the trainer's fallback is what decides it."""
    args = TrainingArguments(output_dir=str(tmp_path), torch_compile=True, report_to=[], use_cpu=True)
    ModeSetupMixin._apply_torch_compile(_host(args))
    assert compile_calls == [{"mode": "default", "backend": "inductor"}]


def test_accelerate_dynamo_plugin_defaults_to_the_same_mode(clean_dynamo_env):
    """The compile ``prepare`` puts around a multi-process accelerate launch's DDP wrapper uses the plugin's mode,
    which HF passes through from ``torch_compile_mode``; left unset, it must match the trainer's fallback."""
    assert TorchDynamoPlugin(backend="inductor").mode == DEFAULT_TORCH_COMPILE_MODE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
