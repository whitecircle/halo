#!/usr/bin/env python
"""A single EP group's expert grads are divided by DP width once per step, however often reentrant checkpointing
re-enters the backward. A divide on a hook is not idempotent; the router's average is."""

import datetime
import functools
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.state import GradientState
from torch.utils.checkpoint import checkpoint

import halo.distributed.expert_parallel.moe_layer.layer as layer_module
from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.gloo import run_gloo_ranks
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

PG_TIMEOUT = datetime.timedelta(seconds=90)
SEED = 0
GRAD_ACCUM = 3
K_VALUES = (1, 4)
# One EP group spanning the ranks, without and with expert TP.
LAYOUTS = {"ep2": (2, {"ep_size": 2}), "ep2_etp2": (4, {"ep_size": 2, "expert_tp_size": 2})}
OVERRIDES = {
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "hidden_size": 64,
    "head_dim": 16,
    "intermediate_size": 64,
    "moe_intermediate_size": 32,
    "num_experts": 4,
}
# Far above any norm here: the clip runs its sync and scales by exactly 1.
NO_CLIP = 1e9
# The K>1 segments sum in a different order than the twin's single graph (fp32).
RTOL = 1e-6


def _segment(x: torch.Tensor, params: list[nn.Parameter], weights: list[torch.Tensor]) -> torch.Tensor:
    return x * sum((param * weight).sum() for param, weight in zip(params, weights, strict=True))


def _scales(k: int, micro: int, rank: int) -> list[float]:
    return [1.0 + 0.25 * index + 0.5 * rank + 0.1 * micro for index in range(k)]


class _SegmentsBackward:
    def training_step(self, model, inputs, num_items_in_batch=None):
        params, weights, scales = inputs
        segment = functools.partial(_segment, params=params, weights=weights)
        loss = sum(checkpoint(segment, torch.tensor(s, requires_grad=True), use_reentrant=True) for s in scales)
        loss.backward()
        return loss.detach()


class _Trainer(DistributedTrainerMixin, _SegmentsBackward):
    def __init__(self, model: nn.Module, ep_config: EPConfig, max_grad_norm: float):
        self.model = model
        self.args = SimpleNamespace(max_grad_norm=max_grad_norm)
        self.parallelism_config = SimpleNamespace(
            is_tp_mode=False, fsdp_reshard_after_backward=True, fsdp_defer_grad_sync=False
        )
        self.accelerator = SimpleNamespace(sync_gradients=True)
        self.state = SimpleNamespace(global_step=0)
        self._ep_config = ep_config
        self._fsdp_wrapped = False
        self._pp_runner = None
        self._pp_stage_group = None
        self._pp_chain_group = None
        self._device_mesh = None
        self._memory_margin_checked = True
        self._attention_work_bound = True
        self._attention_layout = None
        self._param_id_caches = {}
        self._install_grad_accum_window()
        self._install_ep_clip()
        self.optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)
        self._register_grad_sync_step_hooks()


def _ep_model(world: int, fields: dict) -> tuple[nn.Module, EPConfig]:
    """COLLECTIVE: its dispatcher is never reached, since no forward runs."""
    ep_config = EPConfig(
        stage_world_size=world, nvlink_domain_size=world, use_grouped_gemm=False, experts_fsdp_managed=False, **fields
    )
    torch.manual_seed(SEED)
    model = tiny_family_model(TINY_MOE_FAMILIES["qwen3_moe"], overrides=OVERRIDES)
    with patch.object(layer_module, "DeepEPDispatcher", lambda *args, **kwargs: object()):
        apply_ep_layers(model, ep_config)
    return model, ep_config


def _run(world: int, fields: dict, *, k: int, clip: bool) -> dict:
    rank = dist.get_rank()
    model, ep_config = _ep_model(world, fields)
    trainer = _Trainer(model, ep_config, max_grad_norm=NO_CLIP if clip else 0.0)
    experts, routers = [], []
    for _, layer in named_ep_layers(model):
        expert_ids = {id(p) for _, p in layer.expert_named_params()}
        experts += [p for p in layer.parameters() if p.requires_grad and id(p) in expert_ids]
        routers += [p for p in layer.parameters() if p.requires_grad and id(p) not in expert_ids]
    params = experts + routers
    generator = torch.Generator().manual_seed(SEED + rank)
    weights = [torch.randn(p.shape, generator=generator) for p in params]
    twin = [nn.Parameter(p.detach().clone()) for p in params]
    for micro in range(GRAD_ACCUM):
        sync = micro == GRAD_ACCUM - 1
        trainer.accelerator.sync_gradients = sync
        GradientState()._set_sync_gradients(sync)
        scales = _scales(k, micro, rank)
        trainer.training_step(model, (params, weights, scales))
        sum(_segment(torch.tensor(s), twin, weights) for s in scales).backward()
    if clip:
        trainer.accelerator.clip_grad_norm_(model.parameters(), NO_CLIP)
    trainer.optimizer.step()
    divisor = ep_config.stage_world_size // ep_config.expert_tp_size
    for param in twin[: len(experts)]:
        param.grad.div_(divisor)
    for param in twin[len(experts) :]:
        dist.all_reduce(param.grad)
        param.grad.div_(ep_config.stage_world_size)
    return {
        "experts": [p.grad.tolist() for p in experts],
        "routers": [p.grad.tolist() for p in routers],
        "twin_experts": [p.grad.tolist() for p in twin[: len(experts)]],
        "twin_routers": [p.grad.tolist() for p in twin[len(experts) :]],
    }


def _worker(rank: int, out: str, layout: str) -> None:
    world, fields = LAYOUTS[layout]
    results = {f"k{k}-clip{clip}": _run(world, fields, k=k, clip=clip) for k in K_VALUES for clip in (True, False)}
    with open(f"{out}.{rank}", "w") as fh:
        json.dump(results, fh)


@pytest.fixture(scope="module", params=list(LAYOUTS))
def runs(request, tmp_path_factory):
    world, _ = LAYOUTS[request.param]
    out = str(tmp_path_factory.mktemp(request.param) / "grads")
    run_gloo_ranks(_worker, world, out, request.param, pg_timeout=PG_TIMEOUT)
    per_rank = []
    for rank in range(world):
        with open(f"{out}.{rank}") as fh:
            per_rank.append(json.load(fh))
    return per_rank


def _close(actual: list, expected: list) -> bool:
    return all(
        float((torch.tensor(a) - torch.tensor(e)).norm()) <= RTOL * float(torch.tensor(e).norm())
        for a, e in zip(actual, expected, strict=True)
    )


@pytest.mark.parametrize("clip", [True, False], ids=["ep_clip", "step_pre_hook"])
def test_expert_grads_are_divided_once_per_step(runs, clip):
    for rank_runs in runs:
        for k in K_VALUES:
            run = rank_runs[f"k{k}-clip{clip}"]
            assert _close(run["experts"], run["twin_experts"]), k
            assert _close(run["routers"], run["twin_routers"]), k
            if k == 1:
                assert run["experts"] == run["twin_experts"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
