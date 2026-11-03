#!/usr/bin/env python
"""``tests.common.first_step`` captures every trainable parameter's gradient whole, whatever holds it. A CPU twin of
the SMPO ep2tp2 row on two gloo ranks at DP 1: EP layers hold routers and experts plain, attention is TP-sharded,
and the reference is the stock model under FSDP2."""

import dataclasses
import datetime
import json
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from transformers import GptOssConfig, GptOssForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.fsdp import IdentityParamSet, apply_fsdp2_per_layer, mixed_precision_policy
from halo.distributed.tensor_parallel.parallelize_attention import apply_tp_to_attention_only
from halo.distributed.tensor_parallel.plain_params import hand_sliced_suffixes
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.ep_reference import random_token_batch
from tests.common.first_step import ROUTING_DEPENDENT_PARAM, dense_gradient_checks, train_recording_first_step
from tests.common.gloo import run_gloo_ranks
from tests.common.models import TINY_GPTOSS_CONFIG, TINY_QWEN3_MOE_CONFIG
from tests.common.parallelism import single_process_ep_config

WORLD = 2
SEED = 1234
MICROBATCHES = 2
ROWS, SEQ = 2, 32
PG_TIMEOUT = datetime.timedelta(seconds=180)
FAMILIES = {
    "qwen3_moe": (Qwen3MoeConfig, Qwen3MoeForCausalLM, TINY_QWEN3_MOE_CONFIG),
    "gpt_oss": (GptOssConfig, GptOssForCausalLM, TINY_GPTOSS_CONFIG),
}
POLICY = mixed_precision_policy(torch.float32, fp32_reduce=True, cast_forward_inputs=True)
# A replica the capture must take plain, which each negative control takes away.
DROPPED = "model.norm.weight"
# fp32 on both sides, so only the TP and FSDP2 reduction orders separate the two gradients.
EXACT_COSINE_MIN = 1 - 1e-9
EXACT_RATIO_RTOL = 1e-6


class _Trainer:
    """The trainer surface :func:`train_recording_first_step` reads; ``train()`` takes one optimizer step over
    ``batches``."""

    def __init__(self, model, batches, sweep):
        self.model = model
        self.args = SimpleNamespace(gradient_accumulation_steps=len(batches))
        self.state = SimpleNamespace(global_step=0, log_history=[])
        self._batches, self._sweep, self._callbacks = batches, sweep, []

    def compute_loss(self, model, inputs):
        return model(**inputs).loss

    def add_callback(self, callback):
        self._callbacks.append(callback)

    def train(self):
        self.model.train()
        losses = []
        for batch in self._batches:
            loss = self.compute_loss(self.model, batch)
            (loss / len(self._batches)).backward()
            losses.append(float(loss))
        self._sweep(self.model)
        for callback in self._callbacks:
            callback.on_pre_optimizer_step(self.args, self.state, None)
        self.state.log_history.append({"loss": sum(losses) / len(losses)})


def _sweep_tp_grads(model) -> None:
    """COLLECTIVE over the TP group: the trainer's TP-replicated gradient sync, once per optimizer step (per-head
    norms summed, other replicas averaged, hand-sliced ones left alone)."""
    expert_ids = {id(param) for _, layer in named_ep_layers(model) for _, param in layer.expert_named_params()}
    host = SimpleNamespace(
        _device_mesh=model._device_mesh,
        state=SimpleNamespace(global_step=0),
        parallelism_config=SimpleNamespace(fp32_grad_reduce=True),
        _param_id_caches={},
        _top_level_model=lambda: model,
        _get_sharded_expert_param_ids=lambda: expert_ids,
    )
    for name in ("_memoized_param_ids", "_tp_sharded_plain_param_ids", "_tp_per_head_norm_param_ids"):
        setattr(host, name, MethodType(getattr(DistributedTrainerMixin, name), host))
    params = DistributedTrainerMixin._non_expert_params(host, model.parameters())
    DistributedTrainerMixin._sync_tp_replicated_grads(host, params)


def _stock(family: str):
    config_cls, model_cls, fields = FAMILIES[family]
    torch.manual_seed(SEED)
    return model_cls(config_cls(**fields, pad_token_id=0, eos_token_id=1))


def _reference(family: str):
    """The stock model under FSDP2 over the world: every parameter a DTensor."""
    model = _stock(family)
    apply_fsdp2_per_layer(model, init_device_mesh("cpu", (WORLD,)), POLICY, False, IdentityParamSet())
    return model


def _run(family: str):
    """The ep2tp2 capture layout at DP 1: EP layers holding the routers and every expert plain on each rank, and the
    attention TP-sharded over the world, the loader's mesh stamped on the model."""
    model = apply_ep_layers(_stock(family), single_process_ep_config())
    tp_mesh = init_device_mesh("cpu", (WORLD,), mesh_dim_names=("tp",))
    apply_tp_to_attention_only(model, tp_mesh)
    model._device_mesh = tp_mesh
    return model


def _failed(checks: dict[str, bool]) -> list[str]:
    return sorted(name for name, ok in checks.items() if not ok)


def _twin(family: str) -> dict:
    """The clean capture's verdicts and what they read, and each negative control's outcome."""
    vocab = FAMILIES[family][2]["vocab_size"]
    # One batch stream on every rank: the TP group at DP 1 trains one batch.
    batches = [
        dict(zip(("input_ids", "labels"), random_token_batch(vocab, ROWS, SEQ, "cpu", seed=SEED + m), strict=True))
        for m in range(MICROBATCHES)
    ]
    _, want = train_recording_first_step(_Trainer(_reference(family), batches, lambda model: None))
    run = _run(family)
    _, got = train_recording_first_step(_Trainer(run, batches, _sweep_tp_grads))
    checks, metrics = dense_gradient_checks(got, want)
    params = dict(run.named_parameters())
    in_ep_layer = tuple(f"{path}." for path, _ in named_ep_layers(run))
    sliced = hand_sliced_suffixes(run)
    result = {
        "failed": _failed(checks),
        "metrics": metrics,
        "captured": sorted(got.grads),
        "dense": sorted(name for name in want.grads if not ROUTING_DEPENDENT_PARAM.search(name)),
        "dtensors": sorted(name for name, param in params.items() if isinstance(param, DTensor)),
        "plain": sorted(name for name, param in params.items() if not isinstance(param, DTensor)),
        "plain_in_ep_layers": sorted(
            name for name, param in params.items() if name.startswith(in_ep_layer) and not isinstance(param, DTensor)
        ),
        "hand_sliced": {
            name: [params[name].numel(), got.grads[name].numel() if name in got.grads else 0]
            for name in params
            if sliced and name.endswith(sliced)
        },
        "misshapen": sorted(n for n in got.grads if n in want.grads and got.grads[n].shape != want.grads[n].shape),
    }
    uncaptured = dataclasses.replace(got, grads={name: grad for name, grad in got.grads.items() if name != DROPPED})
    result["uncaptured_failed"] = _failed(dense_gradient_checks(uncaptured, want)[0])

    def dropping_sweep(model):
        _sweep_tp_grads(model)
        model.get_parameter(DROPPED).grad = None

    try:
        train_recording_first_step(_Trainer(_run(family), batches, dropping_sweep))
        result["dropped_raised"] = None
    except AssertionError as error:
        result["dropped_raised"] = str(error)
    return result


def _worker(rank: int, out: str) -> None:
    torch.set_num_threads(4)
    with open(f"{out}.{rank}", "w") as fh:
        json.dump({family: _twin(family) for family in FAMILIES}, fh)


@pytest.fixture(scope="module")
def twins(tmp_path_factory) -> list[dict]:
    """Every rank's ``family -> outcome``."""
    out = str(tmp_path_factory.mktemp("first_step_capture") / "capture")
    run_gloo_ranks(_worker, WORLD, out, pg_timeout=PG_TIMEOUT)
    per_rank = []
    for rank in range(WORLD):
        with open(f"{out}.{rank}") as fh:
            per_rank.append(json.load(fh))
    return per_rank


@pytest.mark.parametrize("family", list(FAMILIES))
def test_the_layout_holds_plain_replicas_and_plain_routed_params(twins, family):
    for twin in (runs[family] for runs in twins):
        assert DROPPED in twin["plain"]
        assert twin["dtensors"], "the attention must be TP-sharded"
        assert twin["plain_in_ep_layers"], "the EP layers must hold their routers and experts plain"


@pytest.mark.parametrize("family", list(FAMILIES))
def test_every_dense_parameter_is_captured_whole_and_no_plain_routed_one(twins, family):
    for twin in (runs[family] for runs in twins):
        assert twin["captured"] == twin["dense"]
        assert not set(twin["captured"]) & set(twin["plain_in_ep_layers"])
        assert twin["misshapen"] == []
        assert twin["failed"] == []


@pytest.mark.parametrize("family", list(FAMILIES))
def test_the_captured_gradient_is_the_references(twins, family):
    for twin in (runs[family] for runs in twins):
        metrics = twin["metrics"]
        assert metrics["step1_grad_cosine"] >= EXACT_COSINE_MIN, metrics
        assert abs(metrics["step1_grad_norm_ratio"] - 1) <= EXACT_RATIO_RTOL, metrics


def test_a_hand_sliced_parameter_is_gathered_over_the_tp_group(twins):
    for twin in (runs["gpt_oss"] for runs in twins):
        assert len(twin["hand_sliced"]) == FAMILIES["gpt_oss"][2]["num_hidden_layers"]
        assert all(whole == WORLD * local for local, whole in twin["hand_sliced"].values()), twin["hand_sliced"]


@pytest.mark.parametrize("family", list(FAMILIES))
def test_a_parameter_left_out_of_the_capture_fails_dense_coverage(twins, family):
    for twin in (runs[family] for runs in twins):
        assert twin["uncaptured_failed"] == ["step1_grads_cover_every_dense_param"]


@pytest.mark.parametrize("family", list(FAMILIES))
def test_a_parameter_without_its_gradient_raises(twins, family):
    for twin in (runs[family] for runs in twins):
        assert twin["dropped_raised"] == f"{DROPPED} carries no gradient at the optimizer step"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
