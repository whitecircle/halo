#!/usr/bin/env python
"""``tests.common.parallel_shape.parallel_shape_checks`` fails on a model the axis never reached.

The SFT mode suites gate "the mode engaged" on these probes and run only on the GPU tiers, so a probe
that stopped reading the model (and started agreeing with the config again) would go unnoticed. Each
probe is driven here with the model the axis leaves behind and with one it left untouched: real
Ulysses wrappers on a tiny Qwen3, a real TP DTensor over a fake process group, and EP layers stubbed
down to the class and attributes the probe reads.

Run: python tests/cpu/conventions/test_parallel_shape_checks.py
"""

import pytest
import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig

from src.distributed.context_parallel.patching import patch_attention_for_ulysses
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.mesh import MeshDim
from tests.common.distributed import fake_process_group_mesh
from tests.common.models import TINY_QWEN3_CONFIG, TINY_QWEN3_MOE_CONFIG
from tests.common.parallel_shape import parallel_shape_checks
from tests.common.parallelism import make_parallelism_config

WORLD_SIZE = 2
NUM_EXPERTS = TINY_QWEN3_MOE_CONFIG["num_experts"]
HIDDEN = 8


class _StubEPLayer(EPMoELayerBase):
    """The attributes and expert weights an EP wrapper carries, for ``expert_count`` local experts.
    The base ``__init__`` needs a live process group, so only the ``nn.Module`` half is initialized."""

    def __init__(self, ep_size: int, *, expert_count: int | None = None, expert_tp_size: int = 1):
        nn.Module.__init__(self)
        self.num_experts = NUM_EXPERTS
        self.ep_size = ep_size
        self.experts_per_rank = NUM_EXPERTS // ep_size
        self.expert_start, self.expert_end = 0, self.experts_per_rank
        self.expert_tp_size = expert_tp_size
        count = self.experts_per_rank if expert_count is None else expert_count
        self.gate_up_proj = nn.Parameter(torch.zeros(count, HIDDEN, 2 * HIDDEN))

    def expert_named_params(self):
        return [("gate_up_proj", self.gate_up_proj)]

    def forward(self, hidden_states, **kwargs):
        raise NotImplementedError


def _moe(*layers) -> nn.Module:
    model = nn.Sequential(*layers)
    model.config = Qwen3MoeConfig(**TINY_QWEN3_MOE_CONFIG)
    return model


def _config(**axes):
    return make_parallelism_config(world_size=WORLD_SIZE, gpus_per_node=WORLD_SIZE, **axes)


def _failed(checks: dict[str, bool]) -> list[str]:
    return [name for name, ok in checks.items() if not ok]


def test_an_ep_run_passes_on_split_expert_banks():
    checks = parallel_shape_checks(
        _moe(_StubEPLayer(WORLD_SIZE), _StubEPLayer(WORLD_SIZE)), _config(ep_size=WORLD_SIZE)
    )
    assert checks == {"ep_layers_wrapped": True, "expert_bank_split_ep_way": True}


@pytest.mark.parametrize(
    ("layers", "failed"),
    [
        ((), ["ep_layers_wrapped", "expert_bank_split_ep_way"]),
        # A wrapper built for one rank: it owns the whole bank.
        ((_StubEPLayer(1),), ["expert_bank_split_ep_way"]),
        # The attributes claim the split, but the loader materialized every expert on this rank.
        ((_StubEPLayer(WORLD_SIZE, expert_count=NUM_EXPERTS),), ["expert_bank_split_ep_way"]),
    ],
    ids=["unwrapped", "whole-bank-wrapper", "whole-bank-weights"],
)
def test_an_ep_run_fails_where_the_bank_was_not_split(layers, failed):
    assert _failed(parallel_shape_checks(_moe(*layers), _config(ep_size=WORLD_SIZE))) == failed


@pytest.mark.parametrize(("expert_tp_size", "ok"), [(WORLD_SIZE, True), (1, False)])
def test_an_etp_run_reads_the_sharding_off_every_layer(expert_tp_size, ok):
    model = _moe(_StubEPLayer(1, expert_tp_size=expert_tp_size))
    checks = parallel_shape_checks(model, _config(ep_size=1, expert_tp_size=WORLD_SIZE))
    assert checks["expert_ffn_sharded_etp_way"] is ok
    assert "expert_bank_split_ep_way" not in checks, "ep_size=1 splits no bank"


def test_a_dense_model_owes_no_ep_wrapper():
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    assert parallel_shape_checks(model, _config()) == {}, "grouped GEMM wraps experts, and a dense model has none"


@pytest.mark.parametrize(("dim_name", "ok"), [(MeshDim.TP, True), (MeshDim.DP, False), (None, False)])
def test_a_tp_run_needs_a_param_sharded_over_a_tp_mesh(dim_name, ok):
    model = nn.Linear(HIDDEN, HIDDEN)
    model.config = None
    with fake_process_group_mesh(0, WORLD_SIZE):
        if dim_name is not None:
            mesh = init_device_mesh("cpu", (WORLD_SIZE,), mesh_dim_names=(dim_name,))
            model.weight = nn.Parameter(distribute_tensor(model.weight.data, mesh, [Shard(0)], src_data_rank=None))
        checks = parallel_shape_checks(model, _config(tp_size=WORLD_SIZE))
    assert checks == {"tp_sharded_params": ok}


@pytest.mark.parametrize("patched", [True, False])
def test_a_cp_run_needs_the_attention_swapped_for_ulysses(patched):
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    if patched:
        patch_attention_for_ulysses(model, cp_group=None, cp_size=WORLD_SIZE, validate=False)
    assert parallel_shape_checks(model, _config(cp_size=WORLD_SIZE)) == {"cp_attention_wrapped": patched}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
