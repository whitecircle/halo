#!/usr/bin/env python
"""Ulysses attention hands its flash kernel the run's ``deterministic`` flag, as transformers does off CP.

``UlyssesAttention._attend`` is the one flash call under context parallelism, and it replaces the attention
forward through which transformers would read ``FLASH_ATTENTION_DETERMINISTIC`` on every call. A call that
passes no flag leaves a ``full_determinism`` CP run with a non-deterministic attention backward. The flag must
also reach each kernel the probe resolves, through the adapter the probe applies.

Run: python tests/cpu/parallelism/test_cp_deterministic_attention.py  (or pytest)
"""

import importlib
import inspect
from collections import defaultdict
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from transformers import Qwen3Config
from transformers import modeling_flash_attention_utils as flash_utils

import halo.models.attention.selection as selection
from halo.distributed.context_parallel import ulysses_attention
from halo.distributed.context_parallel.ulysses_attention import UlyssesAttention
from halo.models.attention.selection import FLASH_ATTENTION_DETERMINISTIC_ENV
from tests.common.models import TINY_QWEN3_CONFIG

# Spellings around transformers' own rule, which turns the flag on for "1" alone.
_ENV_VALUES = ("1", "0", "", "true")


class _Attention(nn.Module):
    """The attributes ``UlyssesAttention`` reads off the module it wraps."""

    def __init__(self):
        super().__init__()
        self.config = Qwen3Config(**TINY_QWEN3_CONFIG)
        self.layer_idx = 0
        self.head_dim = self.config.head_dim
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = True


def _attend_inputs() -> tuple[torch.Tensor, ...]:
    config = Qwen3Config(**TINY_QWEN3_CONFIG)
    heads = (config.num_attention_heads, config.num_key_value_heads, config.num_key_value_heads)
    return tuple(torch.randn(1, 4, h, config.head_dim) for h in heads)


def _transformers_flag() -> bool:
    """The ``deterministic`` transformers hands a flash kernel whose signature names it."""
    return flash_utils._process_flash_attention_kwargs(
        query_length=4, key_length=4, is_causal=True, supports_mapping=defaultdict(bool, deterministic=True)
    )["deterministic"]


class _BindingKernel:
    """A flash entry point stand-in carrying the real kernel's signature: each call is bound against it, so a
    keyword the kernel does not take raises as the kernel would, and the bound arguments are recorded."""

    def __init__(self, real):
        self.__signature__ = inspect.signature(real)
        self.__name__ = real.__name__
        self.__module__ = real.__module__
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(dict(self.__signature__.bind(*args, **kwargs).arguments))
        return args[0]


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:  # noqa: BLE001 — any import failure means the probe skips this backend too
        return False
    return True


@pytest.mark.parametrize("value", _ENV_VALUES)
def test_ulysses_passes_the_flag_transformers_passes(value, monkeypatch):
    monkeypatch.setenv(FLASH_ATTENTION_DETERMINISTIC_ENV, value)
    calls = []

    def kernel(query, key, value, **kwargs):
        calls.append(kwargs)
        return query

    with patch.object(ulysses_attention, "_flash_attn_func", lambda allow_fa4: kernel):
        UlyssesAttention(_Attention(), cp_group=None, cp_size=1)._attend(*_attend_inputs())

    assert [kwargs.get("deterministic") for kwargs in calls] == [_transformers_flag()]


@pytest.mark.parametrize(
    ("blackwell", "hopper", "module"),
    [(True, False, "flash_attn.cute"), (False, True, "flash_attn_interface"), (False, False, "flash_attn")],
    ids=["fa4", "fa3", "fa2"],
)
@pytest.mark.parametrize("value", ["1", "0"])
def test_every_kernel_the_probe_resolves_takes_the_flag(blackwell, hopper, module, value, monkeypatch):
    """Through the probe and its FA2-contract adapter, whose dropped keywords would lose the flag silently."""
    if not _importable(module):
        pytest.skip(f"{module} does not import in this image")
    monkeypatch.setenv(FLASH_ATTENTION_DETERMINISTIC_ENV, value)
    kernels = []

    def binding_kernel(attn_impl, *, varlen=True):
        kernels.append(_BindingKernel(selection.flash_kernel(attn_impl, varlen=varlen)))
        return kernels[-1]

    ulysses_attention._flash_attn_func.cache_clear()
    try:
        with (
            patch.object(selection, "is_blackwell_gpu", return_value=blackwell),
            patch.object(selection, "is_hopper_gpu", return_value=hopper),
            patch.object(ulysses_attention, "flash_kernel", binding_kernel),
            patch.object(ulysses_attention, "install_fa4_contiguous_dedicated_head_inputs"),
        ):
            UlyssesAttention(_Attention(), cp_group=None, cp_size=1)._attend(*_attend_inputs())
    finally:
        ulysses_attention._flash_attn_func.cache_clear()

    (kernel,) = kernels
    assert kernel.__module__.startswith(module), kernel.__module__
    assert [call.get("deterministic") for call in kernel.calls] == [value == "1"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
