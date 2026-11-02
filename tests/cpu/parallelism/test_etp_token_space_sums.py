#!/usr/bin/env python
"""The expert-TP sums of an EP MoE layer, run by the real ``_dispatch_compute_combine`` on two gloo ranks.

Under expert TP each partner holds a slice of every local expert's FFN, so its combined output is a partial sum and so
are the gradients it sends back into the tokens and the routing weights. ``SumAcrossGroup`` sums the outputs after the
combine and ``SumGradAcrossGroup`` sums those gradients before the dispatch, both in token space
(``halo/distributed/expert_parallel/moe_layer/layer.py``). A pure-ETP layer (``ep_size`` 1, ``expert_tp_size`` 2) over a
pass-through transport must reproduce one rank holding the whole FFN: the output, the token and routing-weight
gradients, and each rank's expert-weight gradients as its slice of the whole, and the output sum must leave the tensor
the combine handed back as it was (it sums a copy in place). The negative controls replace one sum at its call site
with the identity and must fail exactly the checks that sum carries.

Run: python tests/cpu/parallelism/test_etp_token_space_sums.py
"""

import contextlib
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from halo.distributed.expert_parallel.moe_layer import layer as layer_module
from tests.common.ep_stubs import StubEPLayerBase
from tests.common.gloo import run_gloo_ranks

ETP = 2
EXPERTS, HIDDEN, INTER, TOKENS, TOP_K = 4, 8, 8, 6, 2
SEED = 0
# Two partial down-projections summed against one over the whole intermediate dim: fp32 reassociation only.
TOLERANCE = 1e-5
CHECKS = ("output", "token_grad", "routing_weight_grad", "expert_weight_grads", "combine_output_untouched")
# Each replaces one expert-TP sum at its call site with the identity: the layer-module name, the checks it must fail.
BREAKS = {
    "output_sum": ("SumAcrossGroup", {"output"}),
    "input_grad_sum": ("SumGradAcrossGroup", {"token_grad", "routing_weight_grad"}),
}


class _Identity:
    @staticmethod
    def apply(x, _group):
        return x


class _PassThroughTransport:
    """The dispatch group of one rank at ``ep_size`` 1: every token reaches this rank's experts, and the combine
    returns the weighted expert outputs as computed, keeping the tensor it returned and a copy of its values."""

    def dispatch(self, flat, experts, weights, *, owner):
        return flat, experts, weights, None, None

    def combine(self, output, _recv_topk_weights, _handle, *, owner):
        self.returned, self.returned_values = output, output.detach().clone()
        return output


class _Layer(StubEPLayerBase):
    """The EP state ``_dispatch_compute_combine`` and the per-expert loop read, holding this ETP rank's slice of
    ``gate`` / ``up`` ``[E, M, H]`` and ``down`` ``[E, H, M]`` through the real ETP storage."""

    def __init__(self, gate, up, down, *, expert_tp_size: int, expert_tp_rank: int, expert_tp_group):
        super().__init__()
        self.dispatcher = _PassThroughTransport()
        self.ep_size = 1
        self.experts_per_rank = EXPERTS
        self.expert_tp_size = expert_tp_size
        self.expert_tp_rank = expert_tp_rank
        self.expert_tp_group = expert_tp_group
        self.fp32_experts = False
        self._use_grouped_mm = False
        self._activation_warmed = True
        self._capture_routing = False
        self._expert_adapters_enabled = True
        self._perf = lambda _label: contextlib.nullcontext()
        self.act_fn = F.silu
        self._store_separate_glu_params(gate, up, down)


def _step(layer: _Layer, tokens: torch.Tensor, experts: torch.Tensor, weights: torch.Tensor, probe: torch.Tensor):
    """One forward and backward through the layer: its output and the gradients of the tokens and routing weights."""
    tokens = tokens.clone().requires_grad_()
    weights = weights.clone().requires_grad_()
    output = layer._dispatch_compute_combine(tokens, experts, weights, torch.float32)
    (output * probe).sum().backward()
    return output.detach(), tokens.grad, weights.grad


def _close(got: torch.Tensor, expected: torch.Tensor) -> bool:
    return got.shape == expected.shape and torch.allclose(got, expected, rtol=TOLERANCE, atol=TOLERANCE)


def _worker(rank: int, output_dir: str, broken: str | None) -> None:
    if broken is not None:
        setattr(layer_module, BREAKS[broken][0], _Identity)
    generator = torch.Generator().manual_seed(SEED)
    gate, up = (torch.randn(EXPERTS, INTER, HIDDEN, generator=generator) for _ in range(2))
    down = torch.randn(EXPERTS, HIDDEN, INTER, generator=generator)
    tokens = torch.randn(TOKENS, HIDDEN, generator=generator)
    experts = torch.stack([torch.randperm(EXPERTS, generator=generator)[:TOP_K] for _ in range(TOKENS)])
    weights = torch.rand(TOKENS, TOP_K, generator=generator)
    probe = torch.randn(TOKENS, HIDDEN, generator=generator)

    whole = _Layer(gate, up, down, expert_tp_size=1, expert_tp_rank=0, expert_tp_group=None)
    sharded = _Layer(gate, up, down, expert_tp_size=ETP, expert_tp_rank=rank, expert_tp_group=dist.group.WORLD)
    expected = _step(whole, tokens, experts, weights, probe)
    got = _step(sharded, tokens, experts, weights, probe)
    # This rank's slice of each whole-FFN expert gradient, cut by the layer's own ETP narrowing.
    expected_expert_grads = {
        "gate_proj": sharded._etp_narrow(whole.gate_proj.grad, 2),
        "up_proj": sharded._etp_narrow(whole.up_proj.grad, 2),
        "down_proj": sharded._etp_narrow(whole.down_proj.grad, 1),
    }
    result = {
        "output": _close(got[0], expected[0]),
        "token_grad": _close(got[1], expected[1]),
        "routing_weight_grad": _close(got[2], expected[2]),
        "expert_weight_grads": all(
            _close(getattr(sharded, name).grad, grad) for name, grad in expected_expert_grads.items()
        ),
        "combine_output_untouched": torch.equal(sharded.dispatcher.returned, sharded.dispatcher.returned_values),
    }
    Path(output_dir, f"rank{rank}.json").write_text(json.dumps(result))


def _run(tmp_path: Path, broken: str | None) -> list[dict]:
    run_gloo_ranks(_worker, ETP, str(tmp_path), broken)
    return [json.loads(Path(tmp_path, f"rank{rank}.json").read_text()) for rank in range(ETP)]


def test_the_expert_tp_layer_reproduces_the_whole_ffn(tmp_path):
    for rank, result in enumerate(_run(tmp_path, None)):
        assert result == dict.fromkeys(CHECKS, True), f"rank {rank}: {result}"


@pytest.mark.parametrize("broken", sorted(BREAKS))
def test_each_dropped_sum_is_caught(tmp_path, broken):
    """Exactly the checks the dropped sum carries fail, on every rank: the output sum owns the output, the gradient
    sum the token and routing-weight gradients, and neither the expert-weight gradients."""
    expected = {check: check not in BREAKS[broken][1] for check in CHECKS}
    for rank, result in enumerate(_run(tmp_path, broken)):
        assert result == expected, f"rank {rank} with {broken} dropped: {result}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
