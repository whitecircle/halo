#!/usr/bin/env python
"""Unit tests for :mod:`src.callbacks.moe_metrics`.

``compute_moe_load_metrics`` is the single source of truth for the ``moe/``
metric keys used by both :class:`MoEMetricsCallback` (router-logits path,
``moe_balancing`` ``aux_loss`` / ``none``) and :class:`RouterBiasBalancingCallback`
(EP ``expert_load_counter`` path, ``moe_balancing: bias_update``). Locking down the
math here means the two callbacks agree on units across every supported MoE family.

In addition to the pure metric math we drive the :class:`MoEMetricsCallback`
forward hook + step-end aggregation directly (no GPU / no real model) to pin the
router-logits → top-k bincount → all-step accumulation → ``moe/`` payload path.

Run::

    python tests/cpu/callbacks/test_moe_load_metrics.py
"""

from __future__ import annotations

import math

import pytest
import torch

from src.callbacks.moe_metrics import (
    MoEMetricsCallback,
    _extract_router_logits,
    compute_moe_load_metrics,
)
from src.models.moe_balancing import detect_moe_experts_topk


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


# compute_moe_load_metrics — pure metric math


def test_uniform_distribution() -> None:
    """Perfectly uniform routing: max/min/mean all equal 1.0x uniform, cv=0, no dead experts."""
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
    """One expert gets all tokens in every layer: load_max == E, load_min == 0, dead_frac == (E-1)/E."""
    L, E = 3, 4
    counts = torch.zeros((L, E))
    counts[:, 0] = 200.0  # all tokens go to expert 0 in every layer
    m = compute_moe_load_metrics(counts)

    assert _approx(m["moe/load_max"], float(E))
    assert _approx(m["moe/load_min"], 0.0)
    assert _approx(m["moe/dead_frac"], (E - 1) / E)
    assert _approx(m["moe/load_max_first"], float(E))
    assert _approx(m["moe/load_max_last"], float(E))


def test_load_cv_known_value() -> None:
    """CV of a known skewed single layer matches the closed-form value.

    Loads [2, 1, 1] over 3 experts → fracs [0.5, 0.25, 0.25], mean=1/3,
    population std = sqrt(((1/6)^2 + 2*(1/12)^2)/3). cv = std/mean.
    """
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
    """First layer perfectly balanced, last layer fully collapsed — first/last keys must differ."""
    L, E = 3, 4
    counts = torch.zeros((L, E))
    counts[0] = 50.0  # uniform
    counts[1, 0] = 100.0  # other layer fully collapsed
    counts[1, 1:] = 0.0
    counts[2, 0] = 200.0  # last layer fully collapsed
    counts[2, 1:] = 0.0
    m = compute_moe_load_metrics(counts)

    assert _approx(m["moe/load_max_first"], 1.0)
    assert _approx(m["moe/load_max_last"], float(E))
    # The mean-over-layers load_max sits between the per-layer extremes.
    assert 1.0 < m["moe/load_max"] < float(E)


def test_exclude_last_slot() -> None:
    """Trailing slot is dropped before stats — covers Zaya's discard slot."""
    L, E_plus_one = 2, 5  # 4 real experts + 1 discard
    counts = torch.zeros((L, E_plus_one))
    counts[:, :4] = 25.0  # uniform over real experts
    counts[:, 4] = 9999.0  # large discard slot — must NOT skew the stats
    m = compute_moe_load_metrics(counts, exclude_last_slot=True)

    assert _approx(m["moe/load_max"], 1.0)
    assert _approx(m["moe/load_min"], 1.0)
    assert _approx(m["moe/load_cv"], 0.0, tol=1e-5)
    assert _approx(m["moe/dead_frac"], 0.0)
    # The discard column is gone, so num_layers stays L (rows), and the
    # effective expert count used for the uniform baseline is 4, not 5: had it
    # NOT been dropped, the 9999 column would push load_max far above 1.0.
    assert _approx(m["moe/num_layers"], float(L))


def test_exclude_last_slot_single_column_noop() -> None:
    """exclude_last_slot must not drop the only column (guard against empty stats)."""
    counts = torch.full((2, 1), 7.0)
    m = compute_moe_load_metrics(counts, exclude_last_slot=True)
    # One expert holds 100% of tokens → load_max == 1x uniform (uniform == 1.0).
    assert _approx(m["moe/load_max"], 1.0)
    assert _approx(m["moe/dead_frac"], 0.0)


def test_empty_layer_counts() -> None:
    """All-zero counts (e.g. eval-time skip) clamp safely: dead_frac == 1, load_* finite."""
    L, E = 2, 4
    counts = torch.zeros((L, E))
    m = compute_moe_load_metrics(counts)

    assert _approx(m["moe/dead_frac"], 1.0)
    assert math.isfinite(m["moe/load_max"])
    assert math.isfinite(m["moe/load_cv"])


def test_int_counts_accepted() -> None:
    """Integer counters (the bincount path produces int64) are cast internally."""
    L, E = 2, 4
    counts = torch.full((L, E), 10, dtype=torch.int64)
    m = compute_moe_load_metrics(counts)

    assert _approx(m["moe/load_max"], 1.0)
    assert _approx(m["moe/load_min"], 1.0)


def test_metric_keys_stable() -> None:
    """The set of returned keys is the contract — pin it so wandb dashboards stay valid."""
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


# _extract_router_logits — output-shape normalization


def test_extract_router_logits_tuple() -> None:
    """A tuple of per-layer 2D logits is returned as-is."""

    class _Out:
        router_logits = (torch.randn(8, 4), torch.randn(8, 4))

    out = _extract_router_logits(_Out())
    assert out is not None and len(out) == 2


def test_extract_router_logits_single_tensor_wrapped() -> None:
    """A bare tensor (single MoE layer) is wrapped into a length-1 tuple."""

    class _Out:
        router_logits = torch.randn(8, 4)

    out = _extract_router_logits(_Out())
    assert out is not None and len(out) == 1


def test_extract_router_logits_dict_fallback() -> None:
    """Dict-style outputs are supported via .get()."""
    out = _extract_router_logits({"router_logits": (torch.randn(8, 4),)})
    assert out is not None and len(out) == 1


def test_extract_router_logits_absent() -> None:
    """No router_logits anywhere → None (no-op for dense models)."""

    class _Out:
        pass

    assert _extract_router_logits(_Out()) is None
    assert _extract_router_logits({}) is None


def test_extract_router_logits_filters_non_2d() -> None:
    """1D / None entries are filtered; an all-junk tuple collapses to None."""
    junk = _extract_router_logits({"router_logits": (torch.randn(8), None)})
    assert junk is None
    mixed = _extract_router_logits({"router_logits": (torch.randn(8), torch.randn(8, 4))})
    assert mixed is not None and len(mixed) == 1


# MoEMetricsCallback — hook + step aggregation (no GPU, no real model)


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
    """Two micro-batch forwards accumulate, on_step_end computes metrics, on_log drains them.

    Construct router logits whose argmax sends every token to expert 0 in layer 0
    and is uniform in layer 1; assert the resulting moe/ values.
    """
    cb = MoEMetricsCallback(topk=1)

    class _Mod:
        training = True

    mod = _Mod()

    # Layer 0: 6 tokens all routed to expert 0 (E=3). Layer 1: 2 tokens per expert.
    layer0 = torch.tensor([[5.0, 0.0, 0.0]] * 6)
    layer1 = torch.tensor([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]] * 2)

    out = _make_output((layer0, layer1))
    cb._hook(mod, None, out)
    cb._hook(mod, None, out)  # second micro-batch — must accumulate

    # After two forwards, layer 0 expert 0 saw 12 tokens, layer 1 saw 4 each.
    assert cb._counters[0].tolist() == [12.0, 0.0, 0.0]
    assert cb._counters[1].tolist() == [4.0, 4.0, 4.0]

    cb.on_step_end(args=None, state=_State(global_step=1), control=None)
    logs: dict = {}
    cb.on_log(args=None, state=_State(global_step=1), control=None, logs=logs)

    # Layer 0 fully collapsed (load_max == E == 3), layer 1 uniform (1.0).
    assert _approx(logs["moe/load_max_first"], 3.0)
    assert _approx(logs["moe/load_max_last"], 1.0)
    assert _approx(logs["moe/num_layers"], 2.0)
    # Counters reset after the step so the next step measures fresh.
    assert all((c == 0).all() for c in cb._counters)
    # on_log drains _pending so a second log call does not re-emit stale metrics.
    logs2: dict = {}
    cb.on_log(args=None, state=_State(global_step=1), control=None, logs=logs2)
    assert logs2 == {}


def test_callback_hook_skips_when_not_training() -> None:
    """Eval-mode forwards must not accumulate (the hot path pays nothing in eval)."""
    cb = MoEMetricsCallback(topk=1)

    class _Mod:
        training = False

    cb._hook(_Mod(), None, _make_output((torch.randn(4, 3),)))
    assert cb._counters == []


def test_callback_topk_counts_multiple_experts() -> None:
    """topk>1 records every selected expert per token."""
    cb = MoEMetricsCallback(topk=2)

    class _Mod:
        training = True

    # 4 tokens, E=4; each token strongly prefers experts 0 and 1.
    logits = torch.tensor([[5.0, 4.0, 0.0, 0.0]] * 4)
    cb._hook(_Mod(), None, _make_output((logits,)))
    # top-2 per token = experts {0,1}; 4 tokens → 4 hits each on 0 and 1, 0 on 2,3.
    assert cb._counters[0].tolist() == [4.0, 4.0, 0.0, 0.0]


def test_callback_log_every_n_resets_without_emit() -> None:
    """Off-cadence steps zero the counters and emit nothing."""
    cb = MoEMetricsCallback(topk=1, log_every_n_steps=4)

    class _Mod:
        training = True

    cb._hook(_Mod(), None, _make_output((torch.tensor([[9.0, 0.0]] * 3),)))
    # global_step=2 is not a multiple of 4 → no metrics, counters cleared.
    cb.on_step_end(args=None, state=_State(global_step=2), control=None)
    assert cb._pending == {}
    assert (cb._counters[0] == 0).all()


class _CfgModel:
    """Minimal stand-in for an HF MoE model exposing config + forward-hook registration.

    It declares no ``_can_record_outputs`` router, so the callback takes its model-output fallback —
    which is the path these wiring tests are about.
    """

    def __init__(self, router_aux_loss_coef: float) -> None:
        self.config = type("Cfg", (), {"output_router_logits": False, "router_aux_loss_coef": router_aux_loss_coef})()

    def modules(self):
        return iter([self])

    def named_modules(self):
        return iter([("", self)])

    def register_forward_hook(self, fn):
        return type("H", (), {"remove": lambda self: None})()


def test_callback_warns_once_when_no_router_logits() -> None:
    """No router_logits ever captured (EP recorder bypass / GRPO backbone-only forward): the
    callback must emit a single loud warning, not silently produce no moe/* metrics.

    Fails if the loud-no-op guard is removed (then it would silently no-op as before).
    """
    import src.callbacks.moe_metrics as mm

    cb = MoEMetricsCallback(topk=1)
    model = _CfgModel(router_aux_loss_coef=0.0)
    model.config.output_router_logits = True  # wiring proceeds; only the CAPTURE comes up empty
    cb.on_train_begin(args=None, state=None, control=None, model=model)
    assert cb._hook_handles, "this scenario is a WIRED callback whose hook saw nothing"

    calls: list[str] = []
    orig = mm.logger.warning
    mm.logger.warning = lambda msg, *a, **k: calls.append(str(msg))
    try:
        # _hook never fired (no router_logits) → counters stay empty across steps.
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
    """Metrics need router logits emitted from the forward; aux_loss mode grants that permission.

    The coefficient is left alone: aux_loss balancing is exactly what it is there for.
    """
    cb = MoEMetricsCallback(topk=1, enable_router_logits=True)
    model = _CfgModel(router_aux_loss_coef=0.001)
    cb.on_train_begin(args=None, state=None, control=None, model=model)
    assert model.config.output_router_logits is True
    assert model.config.router_aux_loss_coef == 0.001
    cb.on_train_end(args=None, state=None, control=None)


def test_metrics_never_change_training_by_themselves() -> None:
    """Without permission the callback touches NOTHING and wires no hook.

    Both writes it could make change training, not just observability: ``output_router_logits=True``
    makes HF MoE forwards add ``router_aux_loss_coef * aux_loss`` to the loss and materialize a
    ``[B*S, num_experts]`` tensor per MoE layer, and zeroing the coefficient DISABLES aux balancing the
    user configured in YAML. Asking for metrics must do neither.
    """
    cb = MoEMetricsCallback(topk=1)
    model = _CfgModel(router_aux_loss_coef=0.001)
    cb.on_train_begin(args=None, state=None, control=None, model=model)
    assert model.config.output_router_logits is False, "metrics enabled router logits without permission"
    assert model.config.router_aux_loss_coef == 0.001, "metrics disabled the aux balancing the YAML configured"
    assert cb._hook_handles == [], "hook wired although router_logits can never be populated"


def test_unwired_callback_does_not_warn_about_a_cause_that_does_not_apply() -> None:
    """A callback that declined to wire must not then blame EP / GRPO for the missing metrics.

    ``on_step_end``'s only local signal is "no counters", which is also what an EP or GRPO forward
    produces — so the diagnosis has to be gated on having actually hooked the model. Its advice
    (``moe_balancing=bias_update``) is wrong here and unavailable on weight-sync RL runs.
    """
    cb = MoEMetricsCallback(topk=1)
    model = _CfgModel(router_aux_loss_coef=0.0)
    cb.on_train_begin(args=None, state=None, control=None, model=model)
    assert cb._hook_handles == []

    import src.callbacks.moe_metrics as mm

    calls: list[str] = []
    orig = mm.logger.warning
    mm.logger.warning = lambda msg, *a, **k: calls.append(str(msg))
    try:
        cb.on_step_end(args=None, state=_State(global_step=1), control=None)
    finally:
        mm.logger.warning = orig
    assert calls == [], f"unwired callback emitted a misdiagnosing warning: {calls}"


def test_metrics_ride_along_on_already_enabled_router_logits() -> None:
    """When something else already turned router logits on, metrics are free — collect them."""
    cb = MoEMetricsCallback(topk=1)
    model = _CfgModel(router_aux_loss_coef=0.001)
    model.config.output_router_logits = True
    cb.on_train_begin(args=None, state=None, control=None, model=model)
    assert model.config.router_aux_loss_coef == 0.001
    assert cb._hook_handles
    cb.on_train_end(args=None, state=None, control=None)


class _CompositeCfg:
    """Multimodal wrapper config: expert fields live on ``text_config`` only.

    Mirrors ``Qwen3_5MoeConfig`` / mistral3-wrapped Mistral4 — ``PreTrainedConfig`` defines no
    ``__getattr__``, so the wrapper genuinely does not expose ``num_experts``.
    """

    def __init__(self, **text_fields):
        self.text_config = type("TextCfg", (), text_fields)()

    def get_text_config(self):
        return self.text_config


def test_detect_moe_reads_composite_text_config() -> None:
    """A MoE VLM must be detected through its text sub-config.

    Regression guard: reading only the top level returned (0, 0), so ``moe_balancing: auto``
    resolved to "none" and the flagship Qwen3.5/3.6 MoE runs trained with NO router balancing and
    no moe/* metrics — silently. No family name is involved: any composite config must work.
    """
    model = type("M", (), {"config": _CompositeCfg(num_experts=256, num_experts_per_tok=8)})()
    assert detect_moe_experts_topk(model) == (256, 8)


def test_detect_moe_still_reads_flat_config() -> None:
    """A plain (non-composite) config keeps working — ``get_text_config()`` returns self."""

    class _FlatCfg:
        num_local_experts = 128
        num_experts_per_tok = 4

        def get_text_config(self):
            return self

    model = type("M", (), {"config": _FlatCfg()})()
    assert detect_moe_experts_topk(model) == (128, 4)


def test_detect_moe_dense_model_reports_zero() -> None:
    """A dense model must stay (0, 0) so balancing resolves to 'none'."""

    class _DenseCfg:
        hidden_size = 64

        def get_text_config(self):
            return self

    model = type("M", (), {"config": _DenseCfg()})()
    assert detect_moe_experts_topk(model) == (0, 0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
