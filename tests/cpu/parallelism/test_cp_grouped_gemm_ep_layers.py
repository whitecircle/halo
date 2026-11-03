#!/usr/bin/env python
"""Pure CP on a MoE still gets the ``ep_size == 1`` grouped-GEMM EP layers, and a dense model none. Without them a
MoE pays the Liger force-off while keeping the stock per-expert loop."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config, Qwen3MoeConfig

import halo.model_loading.parallel.cp as cp_loading
import halo.model_loading.parallel.loader as loading
from halo.model_loading.parallel.model_source import ModelSource
from tests.common.parallelism import make_parallelism_config

_SENTINEL_EP_CONFIG = object()


class _Dispatched(Exception):
    def __init__(self, wraps_experts: bool):
        super().__init__(wraps_experts)
        self.wraps_experts = wraps_experts


def _dispatched_wraps_experts(*, use_grouped_gemm: bool, config) -> bool:
    def stop(source, pc, model_class, from_pretrained_kwargs, *, preserve_checkpoint_precision, wraps_experts):
        raise _Dispatched(wraps_experts)

    pc = make_parallelism_config(world_size=8, gpus_per_node=8, cp_size=2, use_grouped_gemm=use_grouped_gemm)
    with (
        patch.object(
            loading, "resolve_model_source", lambda path, revision, **kwargs: ModelSource(path, revision, path)
        ),
        patch.object(loading.AutoConfig, "from_pretrained", lambda *args, **kwargs: config),
        patch.object(loading.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: SimpleNamespace()),
        # Its own test covers it; it would refuse the eager label standing in for attention.
        patch.object(loading, "reject_unsupported_structure", lambda *args, **kwargs: None),
        patch.object(loading, "load_cp_model", stop),
        pytest.raises(_Dispatched) as dispatched,
    ):
        loading.load_distributed_model(
            "org/base", pc, dtype=torch.bfloat16, attn_implementation="eager", trust_remote_code=True
        )
    return dispatched.value.wraps_experts


def _rung_ep_config(*, wraps_experts: bool) -> object | None:
    pc = SimpleNamespace(
        max_concurrent_loading=0,
        fp32_non_ep_params=False,
        cp_config=lambda: SimpleNamespace(cp_size=8),
        ep_config=lambda: _SENTINEL_EP_CONFIG,
    )
    seen = {}

    def fake_load(model_class, source, pc, **kwargs):
        seen["ep_config"] = kwargs["ep_config"]
        return SimpleNamespace()

    with (
        patch.object(cp_loading, "load_whole_model", fake_load),
        patch.object(cp_loading, "apply_ep_layers", lambda model, ep_config: model),
        patch.object(cp_loading, "UlyssesModel", lambda model, cp_config: model),
    ):
        cp_loading.load_cp_model(
            ModelSource("dummy/path", None, "dummy/path"),
            pc,
            None,
            {"config": SimpleNamespace()},
            preserve_checkpoint_precision=False,
            wraps_experts=wraps_experts,
        )
    return seen["ep_config"]


def test_moe_under_pure_cp_gets_the_grouped_gemm_ep_config():
    assert _dispatched_wraps_experts(use_grouped_gemm=True, config=Qwen3MoeConfig()) is True
    assert _rung_ep_config(wraps_experts=True) is _SENTINEL_EP_CONFIG


def test_dense_under_pure_cp_gets_no_ep_config():
    """``needs_ep_layers`` defaults True on a dense model too."""
    assert _dispatched_wraps_experts(use_grouped_gemm=True, config=Qwen3Config()) is False
    assert _rung_ep_config(wraps_experts=False) is None


def test_moe_under_pure_cp_with_grouped_gemm_off_gets_no_ep_config():
    assert _dispatched_wraps_experts(use_grouped_gemm=False, config=Qwen3MoeConfig()) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
