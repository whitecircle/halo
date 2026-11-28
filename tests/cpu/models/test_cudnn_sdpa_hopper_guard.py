#!/usr/bin/env python
"""The guard on pytorch/pytorch#196678: cuDNN SDPA turns off on SM 9.0 under a cuDNN before 9.26 alone, the other SDPA
kernels untouched, and every model load reaches it whatever the attention label."""

from __future__ import annotations

import contextlib
from unittest import mock

import pytest
import torch
from transformers import LlamaConfig

from halo.model_loading import model_preparation
from halo.models.attention import backend_setup
from halo.models.attention.backend_setup import ensure_cudnn_sdpa_off_on_affected_hopper

# The images' cuDNN (torch 2.11.0), the last minor the bug holds for, and the release whose kernel the fix ships.
_IMAGE_CUDNN, _LAST_AFFECTED_CUDNN, _FIXED_CUDNN = 91900, 92500, 92600


class _CudnnSdpFlag:
    def __init__(self):
        self.enabled = True

    def set(self, enabled: bool) -> None:
        self.enabled = enabled


@contextlib.contextmanager
def _device(capability: tuple[int, int], cudnn: int | None, flag: _CudnnSdpFlag):
    with (
        mock.patch.object(torch.cuda, "is_available", lambda: True),
        mock.patch.object(torch.cuda, "get_device_capability", lambda *args: capability),
        mock.patch.object(torch.backends.cudnn, "version", lambda: cudnn),
        mock.patch.object(torch.backends.cuda, "cudnn_sdp_enabled", lambda: flag.enabled),
        mock.patch.object(torch.backends.cuda, "enable_cudnn_sdp", flag.set),
        mock.patch.object(torch.backends.cuda, "enable_flash_sdp") as flash,
        mock.patch.object(torch.backends.cuda, "enable_mem_efficient_sdp") as mem_efficient,
        mock.patch.object(torch.backends.cuda, "enable_math_sdp") as math,
    ):
        yield flash, mem_efficient, math


@pytest.mark.parametrize(
    ("capability", "cudnn", "turned_off"),
    [
        ((9, 0), _IMAGE_CUDNN, True),
        ((9, 0), _LAST_AFFECTED_CUDNN, True),
        ((10, 0), _IMAGE_CUDNN, False),
        ((10, 3), _IMAGE_CUDNN, False),
        ((9, 0), _FIXED_CUDNN, False),
        ((9, 0), None, False),
    ],
    ids=[
        "hopper-cudnn-9.19",
        "hopper-cudnn-9.25",
        "b200-cudnn-9.19",
        "b300-cudnn-9.19",
        "hopper-cudnn-9.26",
        "hopper-no-cudnn",
    ],
)
def test_only_hopper_under_cudnn_926_loses_cudnn_sdpa(capability, cudnn, turned_off):
    flag = _CudnnSdpFlag()
    with _device(capability, cudnn, flag) as other_kernels:
        assert ensure_cudnn_sdpa_off_on_affected_hopper() is turned_off
    assert flag.enabled is not turned_off
    assert not any(setter.called for setter in other_kernels), "flash, mem-efficient and math SDPA stay as they were"


def test_no_gpu_leaves_the_flag_alone():
    with (
        mock.patch.object(torch.cuda, "is_available", lambda: False),
        mock.patch.object(torch.backends.cuda, "enable_cudnn_sdp") as enable_cudnn_sdp,
    ):
        assert ensure_cudnn_sdpa_off_on_affected_hopper() is False
    enable_cudnn_sdp.assert_not_called()


def test_it_logs_once_with_the_issue_and_the_cudnn_that_lifts_it():
    flag = _CudnnSdpFlag()
    with (
        _device((9, 0), _IMAGE_CUDNN, flag),
        mock.patch.object(backend_setup, "is_global_main_process", lambda: True),
        mock.patch.object(backend_setup.logger, "warning") as warn,
    ):
        assert ensure_cudnn_sdpa_off_on_affected_hopper() is True
        assert ensure_cudnn_sdpa_off_on_affected_hopper() is False, "a second load finds it off"
    assert warn.call_count == 1
    message = warn.call_args.args[0]
    assert all(part in message for part in ("pytorch/pytorch#196678", "9.19.0", "9.26.0"))


def test_another_rank_turns_it_off_and_logs_nothing():
    flag = _CudnnSdpFlag()
    with (
        _device((9, 0), _IMAGE_CUDNN, flag),
        mock.patch.object(backend_setup, "is_global_main_process", lambda: False),
        mock.patch.object(backend_setup.logger, "warning") as warn,
    ):
        assert ensure_cudnn_sdpa_off_on_affected_hopper() is True
    assert not flag.enabled
    warn.assert_not_called()


@pytest.mark.parametrize("attn_implementation", [None, "eager", "sdpa", "flex_attention"])
def test_every_load_reaches_the_guard_whatever_the_attention_label(attn_implementation):
    """DSA layers, QSA tiles and flex-sliding's fallbacks reach SDPA under labels other than ``sdpa``."""
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2
    )
    with mock.patch.object(model_preparation, "ensure_cudnn_sdpa_off_on_affected_hopper") as guard:
        model_preparation.apply_family_patches(config, attn_implementation)
    guard.assert_called_once_with()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
