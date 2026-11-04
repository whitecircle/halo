#!/usr/bin/env python
"""``tests.common.routed_first_step`` sees the routed-path faults the SMPO layouts GPU rows rely on it for. A CPU
twin of that suite's tiny Qwen3-MoE at ep1 and pure expert TP 2, against the stock model pinned on the recorded
picks: the clean path passes and each fault fails by a clear factor."""

import copy
import datetime
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from transformers import GptOssConfig, Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP, GptOssTopKRouter
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock, Qwen3MoeTopKRouter

from halo.distributed.expert_parallel.families.gpt_oss import EPGptOssMoELayer
from halo.distributed.expert_parallel.families.qwen3 import EPQwen3MoELayer
from halo.distributed.expert_parallel.moe_layer import layer as layer_module
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from tests.common.ep_reference import random_token_batch, sweep_ep_grads
from tests.common.gloo import run_gloo_ranks
from tests.common.models import TINY_GPTOSS_CONFIG, TINY_QWEN3_MOE_CONFIG
from tests.common.parallelism import single_process_ep_config
from tests.common.routed_first_step import (
    PINNED_GATE_WEIGHTS,
    PinnedReference,
    RoutedStep,
    capturing_routed_forwards,
    gather_routed_grads,
    pinned_reference_routing,
    record_routed_first_step,
    routed_first_step_checks,
)
from tests.common.tolerances import TOL

PG_TIMEOUT = datetime.timedelta(seconds=180)
SEED = 1234
MICROBATCHES = 2
ROWS, SEQ = 4, 48
# How far the twin's compared errors must sit from the bounds, either way: the clean path under bound / CLEAR_FACTOR,
# each fault past CLEAR_FACTOR x bound (measured: clean outputs 8x under, a halved combine 5x past, a dropped
# expert-TP reduce 7x past). The pick shortfall moves in whole bf16 steps of the logits (the clean etp2 one is a
# single step, 7.8e-3), so only its own check holds it.
CLEAR_FACTOR = 3
# The routed checks a fault on the routed path may flip besides its own: a fault reaching the next layers' inputs
# moves their routers' own top-k off the recorded picks.
FAULT_VISIBLE = {
    "routed_outputs_match_reference",
    "routed_grads_match_reference",
    "routed_picks_are_reference_near_ties",
}
TWINS = {
    "ep1": (1, ("clean", "halved", "uncovered", "zeroed", "duplicated", "unrecorded")),
    "etp2": (2, ("clean", "halved", "dropped")),
}
# Each pinned stock router's MoE block, built from its tiny config, and the EP layer class that names its router and
# experts.
PIN_BLOCKS = {
    Qwen3MoeTopKRouter: (lambda: Qwen3MoeSparseMoeBlock(Qwen3MoeConfig(**TINY_QWEN3_MOE_CONFIG)), EPQwen3MoELayer),
    GptOssTopKRouter: (lambda: GptOssMLP(GptOssConfig(**TINY_GPTOSS_CONFIG)), EPGptOssMoELayer),
}


def _stock() -> Qwen3MoeForCausalLM:
    """The SMPO layouts suite's tiny Qwen3-MoE, seeded, in bf16."""
    torch.manual_seed(SEED)
    config = Qwen3MoeConfig(**TINY_QWEN3_MOE_CONFIG, pad_token_id=0, eos_token_id=1)
    return Qwen3MoeForCausalLM(config).to(torch.bfloat16)


def _step(model, batches) -> None:
    model.train()
    for ids, labels in batches:
        (model(input_ids=ids, labels=labels).loss / len(batches)).backward()


def _ep_config(twin: str):
    """The twin's EP config, the experts on the eager loop the CPU runs."""
    if twin == "ep1":
        return single_process_ep_config()
    return ParallelismConfig(expert_tp_size=2, use_grouped_gemm=False).ep_config()


def _faults(variant: str) -> list:
    """The variant's fault, as the GPU controls apply it: the combined output halved, or the expert-TP sum skipped."""
    combine = layer_module.EPMoELayerBase._dispatch_compute_combine

    def halved(self, *args, **kwargs):
        return 0.5 * combine(self, *args, **kwargs)

    if variant == "halved":
        return [patch.object(layer_module.EPMoELayerBase, "_dispatch_compute_combine", halved)]
    if variant == "dropped":
        return [patch.object(layer_module.SumAcrossGroup, "apply", lambda x, group: x)]
    return []


def _run(twin: str, variant: str) -> dict:
    """One recorded step of the twin's EP model and its pinned stock reference, scored; the verdicts and metrics."""
    reference = _stock()
    model = apply_ep_layers(copy.deepcopy(reference), _ep_config(twin))
    vocab = TINY_QWEN3_MOE_CONFIG["vocab_size"]
    batches = [random_token_batch(vocab, ROWS, SEQ, "cpu", seed=SEED + m) for m in range(MICROBATCHES)]
    routed = RoutedStep()
    faults = _faults(variant)
    for fault in faults:
        fault.start()
    try:
        with capturing_routed_forwards(model, routed):
            _step(model, batches)
    finally:
        for fault in faults:
            fault.stop()
    sweep_ep_grads(model)
    gather_routed_grads(model, routed)
    with pinned_reference_routing(reference, routed) as pinned:
        _step(reference, batches)
    grads = {name: param.grad.float() for name, param in reference.named_parameters() if param.grad is not None}
    if variant == "uncovered":
        routed.router_grads.pop(next(iter(routed.router_grads)))
    if variant == "zeroed":
        banks = next(iter(routed.expert_grads.values()))
        banks["down_proj"][0].zero_()
    if variant == "duplicated":
        # A layer dispatching one token to its first pick twice, and a reference pinned on the same picks.
        picks = next(iter(routed.picks.values()))[0]
        picks[0, 1] = picks[0, 0]
    if variant == "unrecorded":
        routed, pinned = RoutedStep(), PinnedReference()
    checks, metrics = routed_first_step_checks(routed, pinned, grads, device="cpu")
    return {"failed": sorted(name for name, ok in checks.items() if not ok), "metrics": metrics}


def _worker(rank: int, out: str, twin: str) -> None:
    torch.set_num_threads(8)
    _, variants = TWINS[twin]
    with open(f"{out}.{rank}", "w") as fh:
        json.dump({variant: _run(twin, variant) for variant in variants}, fh)


def _spawn(twin: str, tmp_path_factory) -> list[dict]:
    """Every rank's ``variant -> {failed, metrics}`` for ``twin``."""
    world, _ = TWINS[twin]
    out = str(tmp_path_factory.mktemp(twin) / "routed")
    run_gloo_ranks(_worker, world, out, twin, pg_timeout=PG_TIMEOUT)
    per_rank = []
    for rank in range(world):
        with open(f"{out}.{rank}") as fh:
            per_rank.append(json.load(fh))
    return per_rank


@pytest.fixture(scope="module")
def ep1_runs(tmp_path_factory) -> list[dict]:
    return _spawn("ep1", tmp_path_factory)


@pytest.fixture(scope="module")
def etp2_runs(tmp_path_factory) -> list[dict]:
    return _spawn("etp2", tmp_path_factory)


def _ratio_deviation(metrics: dict) -> float:
    return max(1 - metrics["routed_grad_min_norm_ratio"], metrics["routed_grad_max_norm_ratio"] - 1)


@pytest.mark.parametrize("twin", list(TWINS))
def test_the_clean_path_passes_every_routed_check_a_clear_factor_inside(request, twin):
    per_rank = request.getfixturevalue(f"{twin}_runs")
    low, high = TOL.routed_grad_norm_ratio_band
    for runs in per_rank:
        clean = runs["clean"]
        assert clean["failed"] == []
        metrics = clean["metrics"]
        assert metrics["routed_output_max_rel_l2"] <= TOL.routed_output_rel_l2 / CLEAR_FACTOR
        assert 1 - metrics["routed_grad_min_cos"] <= (1 - TOL.routed_grad_cosine_min) / CLEAR_FACTOR
        assert _ratio_deviation(metrics) <= min(1 - low, high - 1) / CLEAR_FACTOR


@pytest.mark.parametrize("twin", list(TWINS))
def test_a_halved_combine_fails_the_outputs_and_the_gradients_a_clear_factor_past(request, twin):
    per_rank = request.getfixturevalue(f"{twin}_runs")
    low, _ = TOL.routed_grad_norm_ratio_band
    for runs in per_rank:
        halved = runs["halved"]
        assert {"routed_outputs_match_reference", "routed_grads_match_reference"} <= set(halved["failed"])
        assert set(halved["failed"]) <= FAULT_VISIBLE
        assert halved["metrics"]["routed_output_max_rel_l2"] >= CLEAR_FACTOR * TOL.routed_output_rel_l2
        assert 1 - halved["metrics"]["routed_grad_max_norm_ratio"] >= CLEAR_FACTOR * (1 - low)


def test_a_dropped_expert_tp_reduce_fails_the_outputs_a_clear_factor_past(etp2_runs):
    for runs in etp2_runs:
        dropped = runs["dropped"]
        assert "routed_outputs_match_reference" in dropped["failed"]
        assert set(dropped["failed"]) <= FAULT_VISIBLE
        assert dropped["metrics"]["routed_output_max_rel_l2"] >= CLEAR_FACTOR * TOL.routed_output_rel_l2


def test_an_uncompared_router_or_an_expert_zero_on_one_side_fails(ep1_runs):
    (runs,) = ep1_runs
    assert runs["uncovered"]["failed"] == ["routed_grads_cover_every_routing_param"]
    assert runs["zeroed"]["failed"] == ["routed_grads_match_reference"]


def test_a_token_naming_one_expert_twice_fails_by_name(ep1_runs):
    (runs,) = ep1_runs
    assert runs["duplicated"]["failed"] == ["routed_picks_are_distinct_experts"]
    assert runs["duplicated"]["metrics"]["routed_duplicate_pick_tokens"] == 1


def test_a_step_with_nothing_recorded_or_pinned_fails_by_name(ep1_runs):
    (runs,) = ep1_runs
    assert runs["unrecorded"]["failed"] == [
        "routed_forwards_all_pinned",
        "routed_grads_cover_every_routing_param",
        "routed_outputs_match_reference",
        "routed_picks_are_reference_near_ties",
    ]


class _Trainer:
    """The trainer surface :func:`record_routed_first_step` reads: ``model``, the accumulation steps, ``compute_loss``
    and ``add_callback``."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.args = SimpleNamespace(gradient_accumulation_steps=MICROBATCHES)
        self.callbacks = []

    def compute_loss(self, model, inputs):
        return model(**inputs).loss

    def add_callback(self, callback):
        self.callbacks.append(callback)


def test_a_trainers_first_step_records_its_training_microbatches_and_their_gradients():
    model = apply_ep_layers(copy.deepcopy(_stock()), single_process_ep_config())
    trainer = _Trainer(model)
    routed = record_routed_first_step(trainer)
    vocab = TINY_QWEN3_MOE_CONFIG["vocab_size"]
    batches = [random_token_batch(vocab, ROWS, SEQ, "cpu", seed=SEED + m) for m in range(MICROBATCHES + 1)]
    model.eval()
    with torch.no_grad():
        trainer.compute_loss(model, {"input_ids": batches[0][0], "labels": batches[0][1]})
    model.train()
    for ids, labels in batches:
        (trainer.compute_loss(model, {"input_ids": ids, "labels": labels}) / MICROBATCHES).backward()
    (callback,) = trainer.callbacks
    callback.on_pre_optimizer_step(trainer.args, SimpleNamespace(global_step=0), None)
    first = {name: grad.clone() for name, grad in routed.router_grads.items()}
    for param in model.parameters():
        param.grad = None if param.grad is None else torch.zeros_like(param.grad)
    callback.on_pre_optimizer_step(trainer.args, SimpleNamespace(global_step=1), None)

    layers = TINY_QWEN3_MOE_CONFIG["num_hidden_layers"]
    tokens = ROWS * SEQ
    assert len(routed.layer_classes) == layers
    assert all(len(chunks) == MICROBATCHES for chunks in (*routed.picks.values(), *routed.outputs.values()))
    assert all(
        chunk.shape == (tokens, TINY_QWEN3_MOE_CONFIG["num_experts_per_tok"])
        for chunk in routed.picks["model.layers.0.mlp"]
    )
    assert sorted(routed.router_grads) == sorted(f"model.layers.{i}.mlp.gate.weight" for i in range(layers))
    assert all(torch.equal(routed.router_grads[name], grad) and grad.any() for name, grad in first.items())
    experts, hidden = TINY_QWEN3_MOE_CONFIG["num_experts"], TINY_QWEN3_MOE_CONFIG["hidden_size"]
    intermediate = TINY_QWEN3_MOE_CONFIG["moe_intermediate_size"]
    for banks in routed.expert_grads.values():
        assert banks["gate_up_proj"].shape == (experts, 2 * intermediate, hidden)
        assert banks["down_proj"].shape == (experts, hidden, intermediate)
    assert all(
        not layer._capture_routing and not layer._captured_routing_chunks for _, layer in named_ep_layers(model)
    )


def _block(router_cls: type) -> tuple[nn.Module, type, torch.Tensor]:
    """``router_cls``'s stock MoE block under ``mlp`` of a container, its EP layer class, and a hidden batch for it."""
    build, layer_cls = PIN_BLOCKS[router_cls]
    torch.manual_seed(SEED)
    container = nn.Module()
    container.mlp = build().to(torch.bfloat16)
    for param in container.mlp.parameters():
        nn.init.normal_(param, std=0.02)
    router = layer_cls._declared_router(container.mlp)
    assert isinstance(router, router_cls)
    return container, layer_cls, torch.randn(2, 16, router.weight.shape[1], dtype=torch.bfloat16)


def _router_call(container: nn.Module, layer_cls: type, hidden: torch.Tensor):
    return layer_cls._declared_router(container.mlp)(hidden.reshape(-1, hidden.shape[-1]))


def _block_output(container: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """The block's hidden-state output (GPT-OSS's block also returns its router scores)."""
    output = container.mlp(hidden)
    return output[0] if isinstance(output, tuple) else output


def _recorded(layer_cls: type, *picks: torch.Tensor) -> RoutedStep:
    """A run that recorded ``picks`` for the container's ``mlp``, one forward each."""
    return RoutedStep(layer_classes={"mlp": layer_cls}, picks={"mlp": list(picks)})


@pytest.mark.parametrize("router_cls", list(PINNED_GATE_WEIGHTS), ids=lambda cls: cls.__name__)
def test_a_block_pinned_on_its_routers_own_picks_is_the_stock_block(router_cls):
    container, layer_cls, hidden = _block(router_cls)
    with torch.no_grad():
        want = _block_output(container, hidden)
        _, _, own = _router_call(container, layer_cls, hidden)
        with pinned_reference_routing(container, _recorded(layer_cls, own)) as pinned:
            got = _block_output(container, hidden)
    assert torch.equal(got, want)
    assert pinned.regrets == [0.0]
    assert pinned.unconsumed == {"mlp": 0}
    assert torch.equal(pinned.outputs["mlp"][0], want)
    assert pinned.expert_paths == {"mlp": "mlp.experts"}


@pytest.mark.parametrize("router_cls", list(PINNED_GATE_WEIGHTS), ids=lambda cls: cls.__name__)
def test_a_pick_off_the_routers_top_k_reports_its_shortfall(router_cls):
    container, layer_cls, hidden = _block(router_cls)
    with torch.no_grad():
        logits, _, own = _router_call(container, layer_cls, hidden)
        worst = logits.float().topk(own.shape[-1], dim=-1, largest=False).indices
        with pinned_reference_routing(container, _recorded(layer_cls, worst)) as pinned:
            got = _block_output(container, hidden)
        want = _block_output(container, hidden)
    assert pinned.regrets[0] > TOL.router_near_tie_logit_abs
    assert not torch.equal(got, want)


def test_the_pin_raises_past_the_recorded_forwards_and_reports_unconsumed_ones():
    container, layer_cls, hidden = _block(Qwen3MoeTopKRouter)
    with torch.no_grad():
        _, _, own = _router_call(container, layer_cls, hidden)
        with pinned_reference_routing(container, _recorded(layer_cls, own)):
            container.mlp(hidden)
            with pytest.raises(RuntimeError, match="more forwards than the run recorded"):
                container.mlp(hidden)
        with pinned_reference_routing(container, _recorded(layer_cls, own, own)) as pinned:
            container.mlp(hidden)
    assert pinned.unconsumed == {"mlp": 1}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
