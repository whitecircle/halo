#!/usr/bin/env python
"""``gather_expert_grads`` returns every expert's gradient in its family's hub layout.

The layout must not depend on what the family stores locally: separately stored GLU halves (with or
without expert-TP shards) reach Qwen3's per-expert layout, a fused ``gate_up_proj`` GLM-4's per-expert
split, and GptOss keeps its interleaved matmul layout with biases from either of its stored layouts.

Collectives are emulated in-process: each group's rank ``r`` contributes the local tensor plus
``r`` times a group-specific offset, so a wrong concatenation axis, rank order or group shows up as a
value mismatch. Gradients differ from the weights everywhere, so a gather that read the weights
fails too.

Run: python tests/cpu/parallelism/test_ep_expert_grad_gather.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.distributed.expert_parallel.families.glm4 import EPGlm4MoELayer
from src.distributed.expert_parallel.families.gpt_oss import EPGptOssMoELayer
from src.distributed.expert_parallel.families.qwen3 import EPQwen3MoELayer
from src.runtime import collectives
from tests.common.pp_equivalence import gather_expert_grads

EP_SIZE = 2
LOCAL_EXPERTS, HIDDEN, INTER = 2, 6, 4
EP_GROUP, TP_GROUP = "fake-dispatch-ep-group", "fake-expert-tp-group"
RANK_OFFSET = {EP_GROUP: 100.0, TP_GROUP: 10.0}


@pytest.fixture(autouse=True)
def fake_collectives(monkeypatch):
    """Rank ``r`` of a group contributes ``tensor + r * RANK_OFFSET[group]``."""

    def all_gather_into_tensor(output, tensor, group=None):
        rows = tensor.shape[0]
        for rank in range(output.shape[0] // rows):
            output[rank * rows : (rank + 1) * rows].copy_(tensor + rank * RANK_OFFSET[group])

    def all_gather(shards, tensor, group=None):
        for rank, shard in enumerate(shards):
            shard.copy_(tensor + rank * RANK_OFFSET[group])

    monkeypatch.setattr(collectives.dist, "all_gather_into_tensor", all_gather_into_tensor)
    monkeypatch.setattr(collectives.dist, "all_gather", all_gather)


def _layer(cls, expert_tp_size: int = 1, **shapes) -> nn.Module:
    """``cls`` without its constructor (which needs live EP groups), holding random expert weights of
    ``shapes`` whose gradients are a different random draw."""
    layer = object.__new__(cls)
    nn.Module.__init__(layer)
    layer._expert_lora_attrs = frozenset()
    layer.expert_tp_size = expert_tp_size
    layer.expert_tp_group = TP_GROUP
    layer.ep_config = SimpleNamespace(ep_size=EP_SIZE, dispatch_ep_group=EP_GROUP)
    for attr, shape in shapes.items():
        param = nn.Parameter(torch.randn(*shape))
        param.grad = torch.randn(*shape)
        setattr(layer, attr, param)
    return layer


def _tp_assembled(shard: torch.Tensor, dim: int, tp_size: int) -> torch.Tensor:
    return torch.cat([shard + rank * RANK_OFFSET[TP_GROUP] for rank in range(tp_size)], dim=dim)


def _assert_per_expert(gathered: torch.Tensor, expected_local: torch.Tensor) -> None:
    """Global expert ``r * LOCAL_EXPERTS + j`` is dispatch rank ``r``'s local expert ``j``."""
    assert gathered.shape[0] == EP_SIZE * LOCAL_EXPERTS
    for rank in range(EP_SIZE):
        for j in range(LOCAL_EXPERTS):
            torch.testing.assert_close(
                gathered[rank * LOCAL_EXPERTS + j], expected_local[j] + rank * RANK_OFFSET[EP_GROUP]
            )


def _assert_split_per_expert(gathered: dict, names: tuple[str, ...], expected_local: torch.Tensor) -> None:
    """Global expert ``e``'s ``experts.{e}.<name>.weight`` tensors are the even split of rank ``e // LOCAL_EXPERTS``'s
    local expert ``e % LOCAL_EXPERTS`` along its first axis (``F.linear`` convention)."""
    for rank in range(EP_SIZE):
        for j in range(LOCAL_EXPERTS):
            parts = (expected_local[j] + rank * RANK_OFFSET[EP_GROUP]).chunk(len(names))
            for name, part in zip(names, parts, strict=True):
                torch.testing.assert_close(gathered[f"experts.{rank * LOCAL_EXPERTS + j}.{name}.weight"], part)


@pytest.mark.parametrize("expert_tp_size", [1, 2])
def test_separately_stored_glu_grads_gather_into_the_per_expert_layout(expert_tp_size):
    """Qwen3/Bailing store gate and up apart (and ETP shards each along the intermediate dim); the
    gather reassembles each and writes one ``F.linear`` tensor per expert and projection."""
    shard = INTER // expert_tp_size
    torch.manual_seed(0)
    layer = _layer(
        EPQwen3MoELayer,
        expert_tp_size,
        gate_proj=(LOCAL_EXPERTS, HIDDEN, shard),
        up_proj=(LOCAL_EXPERTS, HIDDEN, shard),
        down_proj=(LOCAL_EXPERTS, shard, HIDDEN),
    )
    gate = _tp_assembled(layer.gate_proj.grad, 2, expert_tp_size)
    up = _tp_assembled(layer.up_proj.grad, 2, expert_tp_size)
    down = _tp_assembled(layer.down_proj.grad, 1, expert_tp_size)

    gathered = gather_expert_grads(layer)

    experts = EP_SIZE * LOCAL_EXPERTS
    assert set(gathered) == {f"experts.{e}.{p}_proj.weight" for e in range(experts) for p in ("gate", "up", "down")}
    _assert_split_per_expert(gathered, ("gate_proj",), gate.transpose(1, 2))
    _assert_split_per_expert(gathered, ("up_proj",), up.transpose(1, 2))
    _assert_split_per_expert(gathered, ("down_proj",), down.transpose(1, 2))


def test_fused_glu_grads_gather_into_the_split_per_expert_layout():
    """A family storing the fused ``gate_up_proj`` whose hub layout splits it per expert (GLM-4)."""
    torch.manual_seed(1)
    layer = _layer(
        EPGlm4MoELayer,
        gate_up_proj=(LOCAL_EXPERTS, HIDDEN, 2 * INTER),
        down_proj=(LOCAL_EXPERTS, INTER, HIDDEN),
    )

    gathered = gather_expert_grads(layer)

    assert len(gathered) == 3 * EP_SIZE * LOCAL_EXPERTS
    _assert_split_per_expert(gathered, ("gate_proj", "up_proj"), layer.gate_up_proj.grad.transpose(1, 2))
    _assert_split_per_expert(gathered, ("down_proj",), layer.down_proj.grad.transpose(1, 2))


@pytest.mark.parametrize("grouped_gemm", [False, True])
def test_gptoss_grads_keep_the_interleaved_matmul_layout_with_biases(grouped_gemm):
    """GptOss exports ``[g0, u0, g1, u1, ...]`` untransposed; the grouped-GEMM path stores the pair
    de-interleaved, so its gradients are re-interleaved on the way out."""
    torch.manual_seed(2)
    down = {"down_proj": (LOCAL_EXPERTS, INTER, HIDDEN), "down_proj_bias": (LOCAL_EXPERTS, HIDDEN)}
    if grouped_gemm:
        layer = _layer(
            EPGptOssMoELayer,
            gate_proj_gmm=(LOCAL_EXPERTS, HIDDEN, INTER),
            up_proj_gmm=(LOCAL_EXPERTS, HIDDEN, INTER),
            gate_proj_gmm_bias=(LOCAL_EXPERTS, INTER),
            up_proj_gmm_bias=(LOCAL_EXPERTS, INTER),
            **down,
        )
        gate_up = torch.stack([layer.gate_proj_gmm.grad, layer.up_proj_gmm.grad], dim=-1).flatten(-2)
        gate_up_bias = torch.stack([layer.gate_proj_gmm_bias.grad, layer.up_proj_gmm_bias.grad], dim=-1).flatten(-2)
    else:
        layer = _layer(
            EPGptOssMoELayer,
            gate_up_proj=(LOCAL_EXPERTS, HIDDEN, 2 * INTER),
            gate_up_proj_bias=(LOCAL_EXPERTS, 2 * INTER),
            **down,
        )
        gate_up, gate_up_bias = layer.gate_up_proj.grad, layer.gate_up_proj_bias.grad

    gathered = gather_expert_grads(layer)

    assert set(gathered) == {
        "experts.gate_up_proj",
        "experts.gate_up_proj_bias",
        "experts.down_proj",
        "experts.down_proj_bias",
    }
    _assert_per_expert(gathered["experts.gate_up_proj"], gate_up)
    _assert_per_expert(gathered["experts.gate_up_proj_bias"], gate_up_bias)
    _assert_per_expert(gathered["experts.down_proj"], layer.down_proj.grad)
    _assert_per_expert(gathered["experts.down_proj_bias"], layer.down_proj_bias.grad)


def test_a_missing_expert_gradient_raises():
    """Gathering zeros for an expert the backward never reached would make the comparison vacuous."""
    layer = _layer(
        EPQwen3MoELayer,
        gate_proj=(LOCAL_EXPERTS, HIDDEN, INTER),
        up_proj=(LOCAL_EXPERTS, HIDDEN, INTER),
        down_proj=(LOCAL_EXPERTS, INTER, HIDDEN),
    )
    layer.up_proj.grad = None

    with pytest.raises(RuntimeError, match=r"up_proj has no gradient"):
        gather_expert_grads(layer)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
