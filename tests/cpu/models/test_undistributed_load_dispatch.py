#!/usr/bin/env python
"""The undistributed load path — one loader, and what decides whether it wraps the MoE experts.

``_dispatch_model_loading`` falls through to ``_load_undistributed_model`` for every shape that
distributes no experts: plain DDP/FSDP data parallelism. That path still applies the EP MoE wrappers
when the run wants them without distribution — grouped-GEMM expert compute, and the FSDP-managed
ep1 expert shard (``fsdp_shard_ep1_experts``) that rides on the same wrappers — and must not apply
them otherwise. Both halves are silent when wrong: wrapping a dense model patches zero MoE layers
after the whole model is on the GPU, and skipping the wrappers on a MoE leaves the stock per-expert
loop while the run pays the Liger force-off that assumes the wrappers are there.

    python tests/cpu/models/test_undistributed_load_dispatch.py
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.nn as nn
from accelerate import PartialState
from transformers import CONFIG_MAPPING

import src.distributed.loading.model_loading as model_loading
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.parallelism import make_parallelism_config

PartialState()  # the loader's accelerate logger needs accelerate state initialized

_PC_MOD = "src.distributed.parallelism_config"


def _pc(**kwargs) -> ParallelismConfig:
    """A single-node 8-GPU ``ParallelismConfig``, built off the real validators."""
    with patch(f"{_PC_MOD}.validate_nvlink_domain_against_fabric"):
        return make_parallelism_config(world_size=8, gpus_per_node=8, **kwargs)


def _dispatch(pc: ParallelismConfig, model_type: str) -> list:
    """Run the dispatcher with the weight load stubbed; return the EP configs the wrappers got."""
    wrapped: list = []
    loaded = nn.Linear(2, 2)

    with (
        patch.object(model_loading, "get_local_rank", return_value=0),
        patch.object(model_loading, "_from_pretrained_on_local_gpu", return_value=loaded),
        patch.object(model_loading, "_apply_ep_wrappers", side_effect=lambda m, cfg: wrapped.append(cfg) or m),
    ):
        model = model_loading._dispatch_model_loading(
            "stub/checkpoint", pc, SimpleNamespace(), {"config": CONFIG_MAPPING[model_type]()}
        )

    assert model is loaded, "the undistributed path must return the loaded model itself"
    return wrapped


def test_a_moe_gets_the_expert_wrappers_without_any_expert_distribution():
    """``ep_size == 1`` with the wrappers on: no rank owns a subset of experts, but the layers are
    still swapped — that is what puts the experts under grouped GEMM and, with the shipped
    ``fsdp_shard_ep1_experts``, under FSDP's own reduce-scatter."""
    pc = _pc(use_grouped_gemm=True)
    assert pc.ep_size == 1 and pc.needs_ep_wrappers and pc.fsdp_shard_ep1_experts, "premise"

    wrapped = _dispatch(pc, "qwen3_moe")

    assert len(wrapped) == 1
    assert wrapped[0].ep_size == 1, "an ep1 wrapper distributes nothing"


def test_a_dense_model_is_never_wrapped_even_with_grouped_gemm_requested():
    """``use_grouped_gemm`` is a default, not a declaration that the model has experts: a dense model
    reaching ``_apply_ep_wrappers`` patches zero MoE layers and raises."""
    assert _dispatch(_pc(use_grouped_gemm=True), "llama") == []


def test_the_wrappers_are_off_when_the_run_turns_grouped_gemm_off():
    """The knob is the whole gate at ``ep_size == 1`` — a MoE run that opted out keeps HF's own
    expert loop."""
    pc = _pc(use_grouped_gemm=False)
    assert not pc.needs_ep_wrappers, "premise"

    assert _dispatch(pc, "qwen3_moe") == []


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
