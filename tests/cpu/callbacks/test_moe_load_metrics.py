#!/usr/bin/env python
"""``compute_moe_load_metrics`` is the one source of the ``moe/`` keys, so both callbacks agree on units.
The :class:`MoEMetricsCallback` hook and step aggregation are driven directly, with no GPU or real model."""

from __future__ import annotations

import math

import pytest
import torch

import halo.callbacks.moe_metrics as mm
from halo.callbacks.moe_metrics import (
    MoEMetricsCallback,
    _extract_router_logits,
    compute_moe_load_metrics,
)
from halo.models.moe.config_fields import resolve_expert_count, resolve_router_topk


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def test_uniform_distribution() -> None:
    L, E = 4, 8
    counts = torch.full((L, E), 100.0)
    m = compute_moe_load_metrics(counts)

    assert _approx(m["moe/load_max"], 1.0)
    assert _approx(m["moe/load_min"], 1.0)
    assert _approx(m["moe/load_cv"], 0.0, tol=1e-5)
    assert _approx(m["moe/dead_frac"], 0.0)
    assert _approx(m["moe/load_max_first"], 1.0)
    assert _approx(m["moe/load_max_last"], 1.0)
    assert _approx(m["moe/num_layers"], float(L))


def test_single_hot_expert() -> None:
    L, E = 3, 4
    counts = torch.zeros((L, E))
    counts[:, 0] = 200.0
    m = compute_moe_load_metrics(counts)

    assert _approx(m["moe/load_max"], float(E))
    assert _approx(m["moe/load_min"], 0.0)
    assert _approx(m["moe/dead_frac"], (E - 1) / E)
    assert _approx(m["moe/load_max_first"], float(E))
    assert _approx(m["moe/load_max_last"], float(E))


def test_load_cv_known_value() -> None:
    counts = torch.tensor([[2.0, 1.0, 1.0]])
    m = compute_moe_load_metrics(counts)

    fracs = [0.5, 0.25, 0.25]
    mean = sum(fracs) / 3
    var = sum((f - mean) ** 2 for f in fracs) / 3
    expected_cv = math.sqrt(var) / mean
    assert _approx(m["moe/load_cv"], expected_cv, tol=1e-5), m["moe/load_cv"]
    # load_max ratio = 0.5 / (1/3) = 1.5
    assert _approx(m["moe/load_max"], 1.5)


def test_per_layer_variation() -> None:
    L, E = 3, 4
    counts = torch.zeros((L, E))
    counts[0] = 50.0
    counts[1, 0] = 100.0
    counts[1, 1:] = 0.0
    counts[2, 0] = 200.0
    counts[2, 1:] = 0.0
    m = compute_moe_load_metrics(counts)

    assert _approx(m["moe/load_max_first"], 1.0)
    assert _approx(m["moe/load_max_last"], float(E))
    # The mean-over-layers load_max sits between the per-layer extremes.
    assert 1.0 < m["moe/load_max"] < float(E)


def test_exclude_last_slot() -> None:
    """Covers Zaya's discard slot."""
    L, E_plus_one = 2, 5  # 4 real experts + 1 discard
    counts = torch.zeros((L, E_plus_one))
    counts[:, :4] = 25.0
    counts[:, 4] = 9999.0  # must not skew the stats
    m = compute_moe_load_metrics(counts, exclude_last_slot=True)

    assert _approx(m["moe/load_max"], 1.0)
    assert _approx(m["moe/load_min"], 1.0)
    assert _approx(m["moe/load_cv"], 0.0, tol=1e-5)
    assert _approx(m["moe/dead_frac"], 0.0)
    assert _approx(m["moe/num_layers"], float(L))


def test_exclude_last_slot_single_column_noop() -> None:
    counts = torch.full((2, 1), 7.0)
    m = compute_moe_load_metrics(counts, exclude_last_slot=True)
    assert _approx(m["moe/load_max"], 1.0)
    assert _approx(m["moe/dead_frac"], 0.0)


def test_empty_layer_counts() -> None:
    L, E = 2, 4
    counts = torch.zeros((L, E))
    m = compute_moe_load_metrics(counts)

    assert _approx(m["moe/dead_frac"], 1.0)
    assert math.isfinite(m["moe/load_max"])
    assert math.isfinite(m["moe/load_cv"])


def test_int_counts_accepted() -> None:
    """The bincount path produces int64."""
    L, E = 2, 4
    counts = torch.full((L, E), 10, dtype=torch.int64)
    m = compute_moe_load_metrics(counts)

    assert _approx(m["moe/load_max"], 1.0)
    assert _approx(m["moe/load_min"], 1.0)


def test_metric_keys_stable() -> None:
    """The wandb dashboards read these keys."""
    counts = torch.ones((2, 4))
    m = compute_moe_load_metrics(counts)
    expected = {
        "moe/load_max",
        "moe/load_min",
        "moe/load_cv",
        "moe/dead_frac",
        "moe/load_max_first",
        "moe/load_max_last",
        "moe/num_layers",
    }
    assert set(m.keys()) == expected, sorted(m.keys())


def test_extract_router_logits_tuple() -> None:
    class _Out:
        router_logits = (torch.randn(8, 4), torch.randn(8, 4))

    out = _extract_router_logits(_Out())
    assert out is not None and len(out) == 2


def test_extract_router_logits_single_tensor_wrapped() -> None:
    class _Out:
        router_logits = torch.randn(8, 4)

    out = _extract_router_logits(_Out())
    assert out is not None and len(out) == 1


def test_extract_router_logits_dict_fallback() -> None:
    out = _extract_router_logits({"router_logits": (torch.randn(8, 4),)})
    assert out is not None and len(out) == 1


def test_extract_router_logits_absent() -> None:
    class _Out:
        pass

    assert _extract_router_logits(_Out()) is None
    assert _extract_router_logits({}) is None


def test_extract_router_logits_filters_non_2d() -> None:
    junk = _extract_router_logits({"router_logits": (torch.randn(8), None)})
    assert junk is None
    mixed = _extract_router_logits({"router_logits": (torch.randn(8), torch.randn(8, 4))})
    assert mixed is not None and len(mixed) == 1


class _State:
    def __init__(self, global_step: int = 1) -> None:
        self.global_step = global_step


def _make_output(router_logits):
    class _Out:
        pass

    o = _Out()
    o.router_logits = router_logits
    return o


def test_callback_init_validation() -> None:
    with pytest.raises(ValueError):
        MoEMetricsCallback(topk=0)
    with pytest.raises(ValueError):
        MoEMetricsCallback(log_every_n_steps=0)


def test_callback_hook_accumulates_and_emits() -> None:
    cb = MoEMetricsCallback(topk=1)

    class _Mod:
        training = True

    mod = _Mod()

    # Layer 0: 6 tokens all routed to expert 0 (E=3). Layer 1: 2 tokens per expert.
    layer0 = torch.tensor([[5.0, 0.0, 0.0]] * 6)
    layer1 = torch.tensor([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]] * 2)

    out = _make_output((layer0, layer1))
    cb._model_output_hook(mod, None, out)
    cb._model_output_hook(mod, None, out)  # second micro-batch

    assert cb._counters[0].tolist() == [12.0, 0.0, 0.0]
    assert cb._counters[1].tolist() == [4.0, 4.0, 4.0]

    cb.on_step_end(args=None, state=_State(global_step=1), control=None)
    logs: dict = {}
    cb.on_log(args=None, state=_State(global_step=1), control=None, logs=logs)

    assert _approx(logs["moe/load_max_first"], 3.0)
    assert _approx(logs["moe/load_max_last"], 1.0)
    assert _approx(logs["moe/num_layers"], 2.0)
    assert all((c == 0).all() for c in cb._counters)
    # on_log drains _pending so a second log call does not re-emit stale metrics.
    logs2: dict = {}
    cb.on_log(args=None, state=_State(global_step=1), control=None, logs=logs2)
    assert logs2 == {}


def test_callback_hook_skips_when_not_training() -> None:
    cb = MoEMetricsCallback(topk=1)

    class _Mod:
        training = False

    cb._model_output_hook(_Mod(), None, _make_output((torch.randn(4, 3),)))
    assert cb._counters == []


def test_callback_topk_counts_multiple_experts() -> None:
    cb = MoEMetricsCallback(topk=2)

    class _Mod:
        training = True

    logits = torch.tensor([[5.0, 4.0, 0.0, 0.0]] * 4)
    cb._model_output_hook(_Mod(), None, _make_output((logits,)))
    assert cb._counters[0].tolist() == [4.0, 4.0, 0.0, 0.0]


def test_callback_log_every_n_resets_without_emit() -> None:
    cb = MoEMetricsCallback(topk=1, log_every_n_steps=4)

    class _Mod:
        training = True

    cb._model_output_hook(_Mod(), None, _make_output((torch.tensor([[9.0, 0.0]] * 3),)))
    cb.on_step_end(args=None, state=_State(global_step=2), control=None)
    assert cb._pending == {}
    assert (cb._counters[0] == 0).all()


class _CfgModel:
    """Declares no ``_can_record_outputs`` router, so the callback takes its model-output fallback."""

    def __init__(self, router_aux_loss_coef: float) -> None:
        self.config = type("Cfg", (), {"output_router_logits": False, "router_aux_loss_coef": router_aux_loss_coef})()

    def modules(self):
        return iter([self])

    def named_modules(self):
        return iter([("", self)])

    def register_forward_hook(self, fn):
        return type("H", (), {"remove": lambda self: None})()


def test_callback_warns_once_when_no_router_logits() -> None:
    """EP recorder bypass or a GRPO backbone-only forward: one loud warning instead of silently no metrics."""
    cb = MoEMetricsCallback(topk=1)
    model = _CfgModel(router_aux_loss_coef=0.0)
    model.config.output_router_logits = True  # wiring proceeds; only the capture comes up empty
    cb.on_train_begin(args=None, state=None, control=None, model=model)
    assert cb._hook_handles, "this scenario is a WIRED callback whose hook saw nothing"

    calls: list[str] = []
    orig = mm.logger.warning
    mm.logger.warning = lambda msg, *a, **k: calls.append(str(msg))
    try:
        cb.on_step_end(args=None, state=_State(global_step=1), control=None)
        cb.on_step_end(args=None, state=_State(global_step=2), control=None)
    finally:
        mm.logger.warning = orig

    assert cb._warned_empty is True
    assert len(calls) == 1, f"expected exactly one warning, got {len(calls)}"
    assert "captured no routing" in calls[0]
    assert "moe_balancing=bias_update" in calls[0]
    assert cb._pending == {}


def test_on_train_begin_enables_router_logits_when_permitted() -> None:
    """The coefficient is left alone, since aux_loss balancing is what it is for."""
    cb = MoEMetricsCallback(topk=1, enable_router_logits=True)
    model = _CfgModel(router_aux_loss_coef=0.001)
    cb.on_train_begin(args=None, state=None, control=None, model=model)
    assert model.config.output_router_logits is True
    assert model.config.router_aux_loss_coef == 0.001
    cb.on_train_end(args=None, state=None, control=None)


def test_metrics_never_change_training_by_themselves() -> None:
    """``output_router_logits=True`` makes HF MoE forwards add the aux loss and materialize a ``[B*S, num_experts]``
    tensor per MoE layer, and zeroing the coefficient disables aux balancing set in YAML. Metrics must do neither."""
    cb = MoEMetricsCallback(topk=1)
    model = _CfgModel(router_aux_loss_coef=0.001)
    cb.on_train_begin(args=None, state=None, control=None, model=model)
    assert model.config.output_router_logits is False, "metrics enabled router logits without permission"
    assert model.config.router_aux_loss_coef == 0.001, "metrics disabled the aux balancing the YAML configured"
    assert cb._hook_handles == [], "hook wired although router_logits can never be populated"


def test_unwired_callback_does_not_warn_about_a_cause_that_does_not_apply() -> None:
    """``on_step_end``'s only local signal is no counters, which an EP or GRPO forward also produces, so the diagnosis
    is gated on having hooked the model. Its advice (``moe_balancing=bias_update``) is wrong here."""
    cb = MoEMetricsCallback(topk=1)
    model = _CfgModel(router_aux_loss_coef=0.0)
    cb.on_train_begin(args=None, state=None, control=None, model=model)
    assert cb._hook_handles == []

    calls: list[str] = []
    orig = mm.logger.warning
    mm.logger.warning = lambda msg, *a, **k: calls.append(str(msg))
    try:
        cb.on_step_end(args=None, state=_State(global_step=1), control=None)
    finally:
        mm.logger.warning = orig
    assert calls == [], f"unwired callback emitted a misdiagnosing warning: {calls}"


def test_metrics_ride_along_on_already_enabled_router_logits() -> None:
    cb = MoEMetricsCallback(topk=1)
    model = _CfgModel(router_aux_loss_coef=0.001)
    model.config.output_router_logits = True
    cb.on_train_begin(args=None, state=None, control=None, model=model)
    assert model.config.router_aux_loss_coef == 0.001
    assert cb._hook_handles
    cb.on_train_end(args=None, state=None, control=None)


class _CompositeCfg:
    """Like ``Qwen3_5MoeConfig``: the wrapper has no ``num_experts`` (no ``__getattr__`` on ``PreTrainedConfig``)."""

    def __init__(self, **text_fields):
        self.text_config = type("TextCfg", (), text_fields)()

    def get_text_config(self):
        return self.text_config


def test_detect_moe_reads_composite_text_config() -> None:
    """Reading only the top level returns (0, 0), so ``moe_balancing: auto`` would resolve to none and the run would
    train with no router balancing and no moe/* metrics."""
    model = type("M", (), {"config": _CompositeCfg(num_experts=256, num_experts_per_tok=8)})()
    assert (resolve_expert_count(model.config), resolve_router_topk(model.config)) == (256, 8)


def test_detect_moe_still_reads_flat_config() -> None:
    class _FlatCfg:
        num_local_experts = 128
        num_experts_per_tok = 4

        def get_text_config(self):
            return self

    model = type("M", (), {"config": _FlatCfg()})()
    assert (resolve_expert_count(model.config), resolve_router_topk(model.config)) == (128, 4)


def test_detect_moe_dense_model_reports_zero() -> None:
    class _DenseCfg:
        hidden_size = 64

        def get_text_config(self):
            return self

    model = type("M", (), {"config": _DenseCfg()})()
    assert (resolve_expert_count(model.config), resolve_router_topk(model.config)) == (0, 0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
