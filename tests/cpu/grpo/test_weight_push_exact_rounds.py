#!/usr/bin/env python
"""The rounds of the push-exactness suite against a stand-in engine: they pass on a push that leaves the server alone
and fail on the ones that do not, so a green live row means what it says."""

from __future__ import annotations

import argparse
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from halo.distributed.expert_parallel.families.bailing import EPBailingMoELayer
from halo.distributed.expert_parallel.moe_layer.gather import EPGatherMixin
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.models.moe.family_specs import BAILING
from tests.common import weight_push_exact
from tests.common.parallelism import single_process_ep_config
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model
from tests.common.weight_sync import RecordingSender

_ROWS = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]
_GATE_WEIGHT = "gate.weight"
_GATE_BIAS = "gate.expert_bias"


class _Engine:
    """Holds a tensor per name and takes a push as the named engine's loader does: vLLM rebuilds a gate that received
    its weight alone, SGLang copies what it is sent. Its score moves with every tensor it holds."""

    DROPPED_BALANCING_BIAS_MODEL_TYPES: frozenset[str] = frozenset()

    def __init__(self, held: dict[str, torch.Tensor], backend: str, drop_from_push: str | None = None):
        self.held = {name: tensor.clone() for name, tensor in held.items()}
        self._backend = backend
        self._drop = drop_from_push
        self._incoming: dict[str, torch.Tensor] = {}

    def __call__(self, **_client_args) -> _Engine:
        return self

    def init_communicator(self, device) -> None:
        pass

    def served_max_model_len(self) -> None:
        return None

    def close_communicator(self) -> None:
        pass

    def scope_co_load_groups(self, module_names) -> None:
        pass

    def abort_weight_update(self) -> None:
        pass

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        if self._drop is None or not name.endswith(self._drop):
            self._incoming[name] = weights.detach().clone()

    def finish_weight_update(self) -> None:
        for name, tensor in self._incoming.items():
            self.held[name] = tensor
        if self._backend == "vllm":
            for name in self._incoming:
                bias = name.replace(_GATE_WEIGHT, _GATE_BIAS)
                if name.endswith(_GATE_WEIGHT) and bias in self.held and bias not in self._incoming:
                    self.held[bias] = torch.zeros_like(self.held[bias])
        self._incoming = {}

    def score_completion_logprobs(self, prompt_ids: list[int], completion_ids: list[int]) -> list[float]:
        total = sum(tensor.double().sum().item() for tensor in self.held.values())
        return [total + token for token in completion_ids]


def _tiny_ep_ling():
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES["bailing_moe"]).to(torch.bfloat16)
    apply_ep_layers(model, single_process_ep_config())
    for index, (_, layer) in enumerate(named_ep_layers(model)):
        bias = layer.get_buffer(_GATE_BIAS)
        bias.copy_(torch.linspace(-0.25, 0.25, bias.numel()) + index / 8)
    return model


def _engine_scores_of_the_trainer(model, rows, device) -> list[float]:
    """What the stand-in engine would score holding this model's push: its scores are no log-probs a forward could match."""
    recorder = RecordingSender(keep_values=True)
    weight_push_exact.sync_weights_to_client(model, recorder, True, True)
    return weight_push_exact._score(_Engine({param.name: param.value for param in recorder.params}, "sglang"), rows)


@pytest.fixture
def run_row(monkeypatch):
    """Drive one row against ``engine_args`` with ``as_served`` on the server. Single rank, on the host."""
    gather = EPGatherMixin.gather_expert_state_dict
    monkeypatch.setattr(
        EPGatherMixin,
        "gather_expert_state_dict",
        lambda self, *, device="cpu", merge_lora=False, retain=True: gather(
            self, device="cpu", merge_lora=merge_lora, retain=retain
        ),
    )
    model = _tiny_ep_ling()
    recorder = RecordingSender(keep_values=True)
    weight_push_exact.sync_weights_to_client(model, recorder, True, True)
    as_served = {param.name: param.value for param in recorder.params}
    ctx = SimpleNamespace(rank=0, device="cpu", barrier=lambda: None, broadcast_checks=lambda checks: checks)
    monkeypatch.setattr(weight_push_exact, "load_policy", lambda *args, **kwargs: (model, None))
    monkeypatch.setattr(weight_push_exact, "_token_rows", lambda checkpoint, limit: _ROWS)
    monkeypatch.setattr(weight_push_exact, "weight_transfer_port", lambda knob: 0)
    monkeypatch.setattr(weight_push_exact, "_trainer_scores", _engine_scores_of_the_trainer)

    def run(
        backend: str, drop_from_push: str | None = None, reach: bool = False, reach_leave_out: str | None = None
    ) -> dict[str, bool]:
        engine = _Engine(as_served, backend, drop_from_push)
        monkeypatch.setattr(weight_push_exact, "resolve_weight_sync_client", lambda name: engine)
        row = argparse.Namespace(
            family="bailing_moe", backend=backend, ep_size=1, tp_size=1, reach=reach, reach_leave_out=reach_leave_out
        )
        return weight_push_exact.run_push_exactness(ctx, row)["checks"]

    return run


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_a_push_that_leaves_the_server_alone_passes_every_round(run_row, backend):
    checks = run_row(backend)

    assert checks.pop("leaving_the_declared_buffers_out_acts_as_the_engine_loader_says")
    assert all(checks.values()), checks


def test_a_push_missing_a_tensor_vllm_counts_fails_on_vllm(run_row):
    """The bug itself: the gate's weight arrives without its bias."""
    checks = run_row("vllm", drop_from_push=_GATE_BIAS)

    assert not checks["unchanged_push_leaves_the_served_log_probs"]
    assert not checks["unchanged_push_restores_the_control"]


def test_a_push_that_changes_a_weight_fails(run_row, monkeypatch):
    """A server launched from other weights moves on the first push, which is what the row exists to catch."""
    update = _Engine.update_named_param

    def nudged(self, name, weights):
        update(self, name, weights + 0.5 if name.endswith("lm_head.weight") else weights)

    monkeypatch.setattr(_Engine, "update_named_param", nudged)

    checks = run_row("sglang")

    assert not checks["unchanged_push_leaves_the_served_log_probs"]


def test_a_control_that_moves_nothing_fails(run_row, monkeypatch):
    """A bar that swallows the control proves nothing."""
    monkeypatch.setattr(weight_push_exact, "_CONTROL_SCALE", 1.0)

    checks = run_row("sglang")

    assert not checks["control_moved_past_the_bar"]


def test_a_real_step_that_moves_nothing_fails(run_row, monkeypatch):
    """The round proves nothing about the sync unless the trainer left what the engine serves."""
    monkeypatch.setattr(weight_push_exact, "_REAL_STEP_LEARNING_RATE", 0.0)

    checks = run_row("sglang")

    assert not checks["a_real_step_moved_the_trainer_past_the_floor"]
    assert not checks["a_push_after_a_real_step_moved_the_server"]


def test_a_push_that_misses_the_real_step_fails(run_row, monkeypatch):
    """The engine keeps serving what it held before the trainer stepped."""
    stepped = []
    take_step = weight_push_exact._take_one_real_step

    def recorded_step(*args) -> None:
        take_step(*args)
        stepped.append(True)

    update = _Engine.update_named_param

    def deaf_after_the_step(self, name, weights):
        if not stepped:
            update(self, name, weights)

    monkeypatch.setattr(weight_push_exact, "_take_one_real_step", recorded_step)
    monkeypatch.setattr(_Engine, "update_named_param", deaf_after_the_step)

    checks = run_row("sglang")

    assert checks["a_real_step_moved_the_trainer_past_the_floor"]
    assert not checks["a_push_after_a_real_step_moved_the_server"]
    assert not checks["a_push_after_a_real_step_serves_the_trainer"]


def test_a_family_with_nothing_declared_runs_no_leave_out_round(run_row, monkeypatch):
    monkeypatch.setattr(EPBailingMoELayer, "FAMILY", replace(BAILING, engine_loaded_buffers=None))

    checks = run_row("sglang")

    assert "unchanged_push_leaves_the_served_log_probs" in checks
    assert not any("declared_buffers" in name for name in checks)


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_every_name_a_healthy_engine_holds_reaches_it(run_row, backend):
    checks = run_row(backend, reach=True)

    assert checks["every_forwarded_name_reaches_the_served_model"]
    assert checks["unchanged_push_restores_the_reach_rounds"]


def test_a_name_the_engine_drops_passes_the_exact_push_and_fails_the_reach_round(run_row):
    """The failure no other round can see: the engine answers 200 on a name it never maps, so the served log-probs never
    move, on an unchanged push or on a real one whose bar a large floor swallows."""
    checks = run_row("sglang", drop_from_push="lm_head.weight", reach=True)

    assert checks["unchanged_push_leaves_the_served_log_probs"]
    assert checks["control_moved_past_the_bar"]
    assert not checks["every_forwarded_name_reaches_the_served_model"]


def test_leaving_a_pattern_out_of_its_own_round_fails_it(run_row):
    """The row's negative control: it must read red on a healthy engine."""
    checks = run_row("sglang", reach=True, reach_leave_out="lm_head.weight")

    assert checks["unchanged_push_leaves_the_served_log_probs"]
    assert not checks["every_forwarded_name_reaches_the_served_model"]


def test_a_zero_tensor_and_a_declared_drop_are_no_reach_rounds():
    """Scaling zeros moves nothing, as on a release Laguna's bias, and an engine that declares the drop owes no move."""
    client = SimpleNamespace(DROPPED_BALANCING_BIAS_MODEL_TYPES=frozenset({"laguna"}))
    forwarded = {
        "model.layers.N.self_attn.q_proj.weight": True,
        "model.layers.N.mlp.gate.weight": False,
        "model.layers.N.mlp.experts.e_score_correction_bias": True,
        "model.vision_tower.layers.N.attn.weight": True,
    }
    assert weight_push_exact._reach_patterns(forwarded, client, {"laguna"}) == [
        "model.layers.N.self_attn.q_proj.weight"
    ]
    assert "model.layers.N.mlp.experts.e_score_correction_bias" in weight_push_exact._reach_patterns(
        forwarded, client, {"bailing_moe"}
    )


def test_the_control_never_lands_in_a_tower_no_text_row_reaches():
    """A scaled vision tensor leaves text log-probs exactly where they were, and the control then reads as a failure."""
    sent: dict[str, torch.Tensor] = {}
    client = SimpleNamespace(update_named_param=lambda name, weights: sent.__setitem__(name, weights))
    sender = weight_push_exact._RoundSender(client, scale=True)
    weight = torch.ones(2, 2)

    sender.update_named_param("model.vision_tower.encoder.layers.1.self_attn.q_proj.weight", weight)
    assert sender.scaled is None
    sender.update_named_param("model.layers.1.self_attn.q_proj.bias", torch.ones(2))
    assert sender.scaled is None
    sender.update_named_param("model.language_model.layers.0.self_attn.q_proj.weight", weight)
    assert sender.scaled is None
    sender.update_named_param("model.language_model.layers.3.self_attn.q_proj.weight", weight)
    assert sender.scaled == "model.language_model.layers.3.self_attn.q_proj.weight"
    assert torch.equal(sent[sender.scaled], weight * weight_push_exact._CONTROL_SCALE)
    sender.update_named_param("model.language_model.layers.4.self_attn.q_proj.weight", weight)
    assert torch.equal(sent["model.language_model.layers.4.self_attn.q_proj.weight"], weight)


def _row() -> argparse.Namespace:
    return argparse.Namespace(family="qwen3", backend="vllm", ep_size=1, tp_size=1)


def _models_reply(served: str) -> SimpleNamespace:
    return SimpleNamespace(json=lambda: {"data": [{"id": served}]})


def test_a_row_runs_only_on_a_fresh_server_holding_its_checkpoint(monkeypatch, capsys):
    held = weight_push_exact.CHECKPOINTS["qwen3"]
    monkeypatch.setattr(weight_push_exact.requests, "get", lambda url, timeout: _models_reply(held))

    monkeypatch.delenv("HALO_TEST_PUSH_EXACT_FRESH", raising=False)
    with pytest.raises(SystemExit) as stale:
        weight_push_exact.skip_unless_a_fresh_server_holds_the_row(_row())
    assert stale.value.code == 0 and "HALO_TEST_PUSH_EXACT_FRESH" in capsys.readouterr().out

    monkeypatch.setenv("HALO_TEST_PUSH_EXACT_FRESH", "1")
    weight_push_exact.skip_unless_a_fresh_server_holds_the_row(_row())

    monkeypatch.setattr(weight_push_exact.requests, "get", lambda url, timeout: _models_reply("someone/else"))
    with pytest.raises(SystemExit) as other:
        weight_push_exact.skip_unless_a_fresh_server_holds_the_row(_row())
    assert other.value.code == 0 and "someone/else" in capsys.readouterr().out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
