#!/usr/bin/env python
"""The environmental trainer's importance weight and KL tail clamp built in the loss forward from its detached
log-probs (``loss_forward_importance_weights``): the same loss and gradient as the no-grad pass for every admitted loss
type, packed or not; the pass kept wherever a consumer needs the round's log-probs earlier; the seam into TRL's loss
that raises when it moves."""

import types
from collections import defaultdict
from typing import get_args

import pytest
import torch
from trl import GRPOTrainer

from halo.configs.grpo_environmental import AsyncRolloutConfig, ISRMaskConfig
from halo.trainers.grpo.environmental import loss_forward_weights as lfw
from halo.trainers.grpo.environmental.loss_forward_weights import (
    LossForwardImportanceMixin,
    loss_forward_weight_blocker,
)
from halo.trainers.grpo.environmental.row_order import permute_rows
from halo.trainers.grpo.environmental.step_batch import IS_WEIGHTS_FROM_LOSS_FORWARD_KEY, ZERO_GRADIENT_ROUND_KEY
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.logprob_scoring import ProtectedTokenEntropyMixin
from halo.trainers.grpo.microbatch_plan import agree_microbatch_geometry
from halo.trainers.grpo.objective.logratio import (
    LOGRATIO_MEAN_KEY,
    clamp_ref_logps,
    compute_is_ratio,
    zero_engine_forced_closes,
)
from halo.trainers.grpo.objective.loss_inputs import (
    IS_CORRECTED_MASK_KEY,
    IS_FORCED_ZERO_KEY,
    IS_SAMPLING_LOGPS_KEY,
    NEGATIVE_ONLY_MASS_KEY,
    NEGATIVE_ONLY_ROWS_KEY,
    NET_TOKEN_MASS_KEY,
    UNCLAMPED_REF_LOGPS_KEY,
    record_token_mass,
)
from halo.trainers.token_metrics import effective_loss_mask
from halo.trainers.world_metrics import WorldMetrics
from tests.common.env_grpo_batch import batch_host, build, episode, row
from tests.common.length_deal import (
    packed_host,
    ragged_round,
    single_process_accelerator,
    stacked,
    tiny_llama,
    train_window,
)

_VOCAB, _ROWS, _CLOSE, _CLIP = 64, 8, 7, 3.0
_ADMITTED = ("grpo", "sapo", "dr_grpo", "dapo", "cispo", "luspo")

_PASS_FACTS = {
    "runs_pass": True,
    "num_iterations": 1,
    "misaligned_accumulation": False,
    "isr_masks": ISRMaskConfig(),
    "engine_reference": False,
    "skip_update_masked_frac": None,
    "balance_token_mass": False,
    "routing_replay": "none",
}
_BLOCKERS = [
    ("runs_pass", False, "nothing to fuse"),
    ("num_iterations", 2, "PPO ratio's denominator"),
    ("misaligned_accumulation", True, "straddles a generation round"),
    ("isr_masks", ISRMaskConfig(geo_band_min=0.9, geo_band_max=1.1), "isr_* mask stages"),
    ("isr_masks", ISRMaskConfig(veto_min=0.1), "isr_* mask stages"),
    ("isr_masks", ISRMaskConfig(opsm_delta=0.5), "isr_* mask stages"),
    ("engine_reference", True, "isr_engine_reference"),
    ("skip_update_masked_frac", 0.5, "skip_update_masked_frac"),
    ("balance_token_mass", True, "balance_token_mass"),
    ("routing_replay", "recompute", "routing_replay: recompute"),
]


def test_a_plain_aligned_one_step_round_is_fusable():
    assert loss_forward_weight_blocker(**_PASS_FACTS) is None


@pytest.mark.parametrize(("fact", "value", "named"), _BLOCKERS)
def test_every_consumer_of_the_round_start_log_probs_blocks_the_fusion(fact, value, named):
    blocker = loss_forward_weight_blocker(**{**_PASS_FACTS, fact: value})
    assert blocker is not None and named in blocker


def test_every_routing_replay_mode_is_judged():
    modes = get_args(AsyncRolloutConfig.__annotations__["routing_replay"])
    verdicts = {mode: loss_forward_weight_blocker(**{**_PASS_FACTS, "routing_replay": mode}) for mode in modes}
    assert {mode for mode, blocker in verdicts.items() if blocker is not None} == {"recompute"}


def _resolving_host(*, requested=True, is_correction=True, beta=0.0, iterations=1, accumulation=4, **config):
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host.async_config = AsyncRolloutConfig(loss_forward_importance_weights=requested, **config)
    host.args = types.SimpleNamespace(steps_per_generation=4, gradient_accumulation_steps=accumulation)
    host.num_iterations, host.beta, host._is_correction = iterations, beta, is_correction
    host._isr_mask_config = host.async_config.build_isr_mask_config()
    return host


def test_the_knob_defaults_off_and_opts_in():
    assert AsyncRolloutConfig().loss_forward_importance_weights is False
    for requested, fuses in ((False, False), (True, True)):
        host = _resolving_host(requested=requested)
        host._configure_loss_forward_weights()
        assert host._skips_logprob_pass is fuses
        assert (host._loss_forward_metrics is not None) is fuses


def test_true_raises_where_a_consumer_needs_the_pass_naming_it_and_false_never_does():
    with pytest.raises(
        ValueError, match=r"loss_forward_importance_weights: true, but balance_token_mass .* Set it to false"
    ):
        _resolving_host(balance_token_mass=True)._configure_loss_forward_weights()
    kept = _resolving_host(requested=False, balance_token_mass=True)
    kept._configure_loss_forward_weights()
    assert kept._skips_logprob_pass is False


def test_nothing_to_fuse_is_refused():
    with pytest.raises(ValueError, match="nothing to fuse"):
        _resolving_host(is_correction=False)._configure_loss_forward_weights()


def test_an_early_stop_threshold_is_accepted_under_true():
    host = _resolving_host(early_stop_logratio_gap=0.1)
    host._configure_loss_forward_weights()
    assert host._skips_logprob_pass is True


@pytest.mark.parametrize("requested", [True, False])
def test_routing_replay_recompute_never_loses_its_pass_to_the_knob(requested):
    """R2's construction gate reads ``_runs_logprob_recompute`` ahead of the knob and must not follow it."""
    host = _resolving_host(requested=requested, routing_replay="recompute")
    if requested:
        with pytest.raises(ValueError, match="routing_replay: recompute"):
            host._configure_loss_forward_weights()
    else:
        host._configure_loss_forward_weights()
    assert host._skips_logprob_pass is False and host._runs_logprob_recompute is True


def test_a_misaligned_window_is_refused_under_true():
    with pytest.raises(ValueError, match="straddles a generation round"):
        _resolving_host(accumulation=6)._configure_loss_forward_weights()


# ---- the loss: fused equals the pass -------------------------------------------------------------------------------


def _sampling_side_round(seed: int = 5, rows: int = _ROWS) -> dict:
    """A round whose engine log-probs sit within a couple of nats of any tiny model's, so weights straddle the clamp,
    with a forced close, a row without sampling log-probs and an untrainable row."""
    gen = torch.Generator().manual_seed(seed)
    batch = ragged_round(
        torch.randint(2, 6, (rows,), generator=gen).tolist(),
        torch.randint(3, 9, (rows,), generator=gen).tolist(),
        vocab=_VOCAB,
        seed=seed,
    )
    batch["completion_ids"][1, 1] = _CLOSE
    batch["completion_ids"][3, 0] = _CLOSE
    batch["tool_mask"][2, 1] = 0
    batch["advantages"][4] = 0.0
    return batch


def _score(model, batch: dict) -> torch.Tensor:
    """The pass's recompute: every row, no grad, no entropy."""
    host = packed_host(model, slots=_ROWS, accelerator=single_process_accelerator())
    host._packs_microbatches = False
    ids, mask, keep = stacked(batch)
    with torch.no_grad():
        return host._get_per_token_logps_and_entropies(model, ids, mask, keep, compute_entropy=False)[0]


def _inputs(batch: dict, beta: float, *, seed: int = 11) -> tuple[dict, dict]:
    """``(pass batch, fused batch)`` of one round: the weight and clamped reference the pass would build, and the inputs
    the loss forward builds them from."""
    gen = torch.Generator().manual_seed(seed)
    model = tiny_llama(_VOCAB, 0)
    recompute = _score(model, batch)
    shape = recompute.shape
    has_sampling = torch.ones(shape[0], dtype=torch.bool)
    has_sampling[5] = False
    noise = torch.randn(shape, generator=gen) * 0.9
    sampling = (recompute - noise) * has_sampling[:, None]
    sampling[1, 1] = 0.0  # the engine emitted the close with probability 1
    sampling[3, 0] = 0.0
    completion_mask = batch["completion_mask"].bool()
    ratio, _, corrected = compute_is_ratio(recompute, sampling, completion_mask, has_sampling, _CLIP)
    ratio, forced = zero_engine_forced_closes(
        ratio, sampling, completion_mask, has_sampling, batch["completion_ids"], (_CLOSE,)
    )
    assert forced.any() and (ratio > _CLIP - 1e-6).any() and (ratio < 1).any() and not ratio.eq(1).all()
    ref = recompute + (torch.rand(shape, generator=gen) - 0.5) * 16
    clamped_ref, was_clamped = clamp_ref_logps(ref, recompute)
    assert was_clamped.any() and not was_clamped.all()
    negative_only = torch.zeros(shape[0], dtype=torch.bool)
    negative_only[2] = True
    shared = {k: v for k, v in batch.items() if k != "importance_sampling_ratio"}
    in_pass = shared | {"importance_sampling_ratio": ratio}
    fused = shared | {
        IS_SAMPLING_LOGPS_KEY: sampling,
        IS_CORRECTED_MASK_KEY: corrected,
        IS_FORCED_ZERO_KEY: forced,
        NEGATIVE_ONLY_ROWS_KEY: negative_only,
    }
    if beta:
        in_pass["ref_per_token_logps"] = clamped_ref
        fused[UNCLAMPED_REF_LOGPS_KEY] = ref
    return in_pass, fused


def _host(*, loss_type: str, beta: float, fused: bool, packed: bool = False):
    host = packed_host(
        tiny_llama(_VOCAB, 0),
        slots=_ROWS if not packed else 4,
        loss_type=loss_type,
        accelerator=single_process_accelerator(),
    )
    host._packs_microbatches = packed
    host.beta, host.vllm_importance_sampling_clip_max, host._is_correction = beta, _CLIP, True
    host.args.use_bias_correction_kl = False
    host._skips_logprob_pass = fused
    host._loss_forward_metrics = WorldMetrics() if fused else None
    return host


def _train(batch: dict, *, loss_type: str, beta: float, fused: bool, packed: bool = False, hook=None):
    host = _host(loss_type=loss_type, beta=beta, fused=fused, packed=packed)
    plan = None
    if packed:
        lengths = (batch["prompt_mask"].sum(1) + batch["completion_mask"].sum(1)).clamp(min=2).tolist()
        plan = agree_microbatch_geometry(lengths, 40, slots=4, ceiling=None, round_rows=_ROWS)
    total, grads = train_window(host, batch, plan)
    return host, total, grads


def _close_to(got, want) -> bool:
    return got[0] == pytest.approx(want[0], rel=1e-4, abs=1e-6) and all(
        torch.allclose(g, w, rtol=1e-3, atol=1e-6) for g, w in zip(got[1], want[1], strict=True)
    )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("beta", [0.0, 0.04])
@pytest.mark.parametrize("loss_type", _ADMITTED)
def test_the_loss_forward_weight_gives_the_pass_loss_and_gradient(loss_type, beta, packed):
    in_pass, fused = _inputs(_sampling_side_round(), beta)
    _, *want = _train(in_pass, loss_type=loss_type, beta=beta, fused=False, packed=packed)
    host, *got = _train(fused, loss_type=loss_type, beta=beta, fused=True, packed=packed)
    assert abs(want[0]) > 1e-4, "a vanishing loss compares nothing"
    assert _close_to(got, want)
    assert host._loss_forward_seam is None


def _leaky_hook(self, model, input_ids, attention_mask, logits_to_keep, *args, **kwargs):
    """The hook with the ``detach()`` taken out."""
    scored = ProtectedTokenEntropyMixin._get_per_token_logps_and_entropies(
        self, model, input_ids, attention_mask, logits_to_keep, *args, **kwargs
    )
    seam = self._loss_forward_seam
    if seam is not None and torch.is_grad_enabled() and kwargs.get("compute_entropy"):
        seam.fired += 1
        self._weigh_from_loss_forward(seam, scored[0])
    return scored


def test_a_weight_that_carries_a_graph_changes_the_gradient(monkeypatch):
    """Negative control: ``w`` built from the live log-probs adds ``A·logπ·∂w/∂θ`` the objective does not have."""
    in_pass, fused = _inputs(_sampling_side_round(), 0.0)
    want = _train(in_pass, loss_type="dapo", beta=0.0, fused=False)[1:]
    monkeypatch.setattr(LossForwardImportanceMixin, "_get_per_token_logps_and_entropies", _leaky_hook)
    got = _train(fused, loss_type="dapo", beta=0.0, fused=True)[1:]
    assert not _close_to(got, want)


def test_a_weight_clamped_from_below_changes_the_loss(monkeypatch):
    in_pass, fused = _inputs(_sampling_side_round(), 0.0)
    want = _train(in_pass, loss_type="dapo", beta=0.0, fused=False)[1:]

    def from_below(logps, sampling_logps, corrected_mask, clip_max):
        diff = (logps - sampling_logps) * corrected_mask
        return torch.clamp(torch.exp(diff), min=clip_max), diff

    monkeypatch.setattr(lfw, "compute_token_is_ratio", from_below)
    assert not _close_to(_train(fused, loss_type="dapo", beta=0.0, fused=True)[1:], want)


def test_forced_closes_left_unzeroed_change_the_loss():
    in_pass, fused = _inputs(_sampling_side_round(), 0.0)
    want = _train(in_pass, loss_type="dapo", beta=0.0, fused=False)[1:]
    no_forced = {k: v for k, v in fused.items() if k != IS_FORCED_ZERO_KEY}
    assert not _close_to(_train(no_forced, loss_type="dapo", beta=0.0, fused=True)[1:], want)


# ---- the seam into TRL's loss --------------------------------------------------------------------------------------


def _fused_window(monkeypatch, fake_loss):
    """One micro-step of a fused host whose TRL loss is ``fake_loss``."""
    _, fused = _inputs(_sampling_side_round(), 0.0)
    host = _host(loss_type="dapo", beta=0.0, fused=True)
    monkeypatch.setattr(DistributedAsyncEnvironmentalGRPOTrainer, "_compute_loss", fake_loss, raising=False)
    host._generate_and_score_completions = lambda _inputs: dict(fused)
    host._step, host._buffered_inputs = 0, None
    inputs = host._prepare_inputs({})
    return lambda: DistributedAsyncEnvironmentalGRPOTrainer.compute_loss(host, host.model, inputs)


def _scoring_loss(calls: int, reads: tuple[str, ...] = ()):
    def fake(self, model, inputs):
        ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], 1)
        mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], 1)
        loss = 0.0
        for _ in range(calls):
            loss = (
                loss
                + self._get_per_token_logps_and_entropies(
                    model, ids, mask, inputs["completion_ids"].size(1), compute_entropy=True
                )[0].sum()
            )
        for key in reads:
            loss = loss + inputs[key].sum()
        return loss

    return fake


@pytest.mark.parametrize(("calls", "message"), [(0, "scored 0 time"), (2, "scored 2 time")])
def test_a_loss_forward_that_does_not_score_exactly_once_raises(monkeypatch, calls, message):
    run = _fused_window(monkeypatch, _scoring_loss(calls, reads=("importance_sampling_ratio",) if calls else ()))
    with pytest.raises(RuntimeError, match=message):
        run()


def test_a_trl_that_stops_reading_the_weight_raises(monkeypatch):
    run = _fused_window(monkeypatch, _scoring_loss(1))
    with pytest.raises(RuntimeError, match=r"never read \['importance_sampling_ratio'\]"):
        run()


def test_trls_loss_reads_both_written_keys_only_after_its_one_scoring_call(monkeypatch):
    """The order the hook relies on, against the TRL this image pins: the batch carries neither key before the
    forward, and the real ``_compute_loss`` reads both after it."""
    events = []
    note = lfw._WeightSeam._note
    monkeypatch.setattr(
        lfw._WeightSeam,
        "_note",
        lambda self, key: (events.append(f"read:{key}") if key in self._written else None, note(self, key))[1],
    )
    scorer = ProtectedTokenEntropyMixin._get_per_token_logps_and_entropies
    monkeypatch.setattr(
        ProtectedTokenEntropyMixin,
        "_get_per_token_logps_and_entropies",
        lambda self, *a, **kw: (events.append("forward"), scorer(self, *a, **kw))[1],
    )
    _, fused = _inputs(_sampling_side_round(), 0.04)
    host = _host(loss_type="dapo", beta=0.04, fused=True)
    host._generate_and_score_completions = lambda _inputs: dict(fused)
    host._step, host._buffered_inputs = 0, None
    inputs = host._prepare_inputs({})
    assert "importance_sampling_ratio" not in inputs and "ref_per_token_logps" not in inputs
    events.clear()
    DistributedAsyncEnvironmentalGRPOTrainer.compute_loss(host, host.model, inputs)
    assert events.count("forward") == 1
    after = events[events.index("forward") :]
    assert {"read:importance_sampling_ratio", "read:ref_per_token_logps"} <= set(after)
    assert not {e for e in events[: events.index("forward")] if e.startswith("read:")}


# ---- the metrics the pass fed --------------------------------------------------------------------------------------


def _pass_metrics(in_pass: dict, fused: dict) -> dict[str, float]:
    """What the pass path logs for the round, from the build's own metric code."""
    host = types.SimpleNamespace(
        vllm_importance_sampling_clip_max=_CLIP,
        _is_correction=True,
        _isr_mask_config=types.SimpleNamespace(any_mask_active=False),
        _metrics={"train": defaultdict(list)},
    )
    host._world_metrics = WorldMetrics()
    trainer = DistributedAsyncEnvironmentalGRPOTrainer
    loss_mask = (in_pass["completion_mask"] * in_pass["tool_mask"]).bool()
    recompute = _score(tiny_llama(_VOCAB, 0), in_pass)
    step = types.SimpleNamespace(
        sampling_logps=fused[IS_SAMPLING_LOGPS_KEY],
        completion_mask=in_pass["completion_mask"].bool(),
        has_sampling=fused[IS_SAMPLING_LOGPS_KEY].ne(0).any(1),
    )
    rows = types.SimpleNamespace(traj_row_ids=lambda device: torch.zeros(recompute.size(0), dtype=torch.long))
    ratio, _, corrected, _ = trainer._apply_is_correction(host, recompute, step, rows)
    trainer._narrow_corrected_mask_and_record(host, ratio, corrected, loss_mask)
    host._world_metrics.covariance(
        "logps/advantage_cov", recompute, in_pass["advantages"].unsqueeze(1).expand_as(recompute), loss_mask
    )
    record_token_mass(
        in_pass["advantages"],
        loss_mask,
        in_pass["importance_sampling_ratio"],
        lambda x: x,
        host._metrics["train"],
        False,
        negative_only=fused[NEGATIVE_ONLY_ROWS_KEY],
    )
    host._world_metrics.flush(host._metrics["train"])
    return {key: values[-1] for key, values in host._metrics["train"].items()}


def test_the_round_end_flush_logs_what_the_pass_logged_over_loss_tokens():
    batch = _sampling_side_round()
    batch["tool_mask"] = batch["completion_mask"].clone()  # every attended token is a loss token, so D1 cannot differ
    in_pass, fused = _inputs(batch, 0.0)
    host, *_ = _train(fused, loss_type="dapo", beta=0.0, fused=True)
    host._metrics = {"train": defaultdict(list)}
    host._flush_loss_forward_metrics()
    got = {key: values[-1] for key, values in host._metrics["train"].items()}
    want = _pass_metrics(in_pass, fused)
    for key in (
        LOGRATIO_MEAN_KEY,
        "sampling/is_ratio_extreme_frac",
        "sampling/is_ratio_mean",
        "sampling/is_ratio_max",
        "sampling/is_ess_frac",
        "logps/advantage_cov",
        NET_TOKEN_MASS_KEY,
        NEGATIVE_ONLY_MASS_KEY,
    ):
        assert got[key] == pytest.approx(want[key], rel=1e-4, abs=1e-6), key


def test_a_dropped_row_leaves_the_logged_log_ratio_as_decision_d1_says():
    """The pass logged the log-ratio over every corrected token, a dropped row's included; the loss forward reads loss
    tokens only, so the two differ exactly by that row."""
    batch = _sampling_side_round()
    in_pass, fused = _inputs(batch, 0.0)
    dropped = {**fused, "completion_mask": fused["completion_mask"].clone(), "tool_mask": fused["tool_mask"].clone()}
    dropped["completion_mask"][0] = 0
    dropped["tool_mask"][0] = 0
    host, *_ = _train(dropped, loss_type="dapo", beta=0.0, fused=True)
    host._metrics = {"train": defaultdict(list)}
    host._flush_loss_forward_metrics()
    fused_mean = host._metrics["train"][LOGRATIO_MEAN_KEY][-1]
    recompute = _score(tiny_llama(_VOCAB, 0), in_pass)
    corrected = fused[IS_CORRECTED_MASK_KEY]
    diff = (recompute - fused[IS_SAMPLING_LOGPS_KEY]) * corrected
    over_loss = corrected & (dropped["completion_mask"] * dropped["tool_mask"]).bool()
    over_all = corrected
    assert fused_mean == pytest.approx((diff * over_loss).sum().item() / over_loss.sum().item(), rel=1e-4)
    assert abs(fused_mean - diff.sum().item() / over_all.sum().item()) > 1e-4


def test_a_round_without_loss_tokens_logs_no_ratio_as_the_pass_does():
    batch = _sampling_side_round()
    batch["tool_mask"].zero_()  # every group degenerate: no token enters the loss
    in_pass, fused = _inputs(batch, 0.0)
    host = types.SimpleNamespace(vllm_importance_sampling_clip_max=_CLIP, _loss_forward_metrics=WorldMetrics())
    loss_mask = effective_loss_mask(fused).bool()
    assert not loss_mask.any()
    ratio = torch.ones_like(fused[IS_SAMPLING_LOGPS_KEY])
    seam = {"advantages": fused["advantages"], NEGATIVE_ONLY_ROWS_KEY: fused[NEGATIVE_ONLY_ROWS_KEY]}
    LossForwardImportanceMixin._record_loss_forward_metrics(
        host, seam, torch.zeros_like(ratio), loss_mask, ratio, torch.zeros_like(ratio), fused[IS_CORRECTED_MASK_KEY]
    )
    got = host._loss_forward_metrics.flush()
    want = _pass_metrics(in_pass, fused)
    assert "sampling/is_ratio_mean" not in want
    assert "sampling/is_ratio_mean" not in got
    assert LOGRATIO_MEAN_KEY not in got


def test_a_non_finite_token_mass_raises_at_the_flush():
    host = types.SimpleNamespace(_metrics={"train": defaultdict(list)})
    host._loss_forward_metrics = WorldMetrics()
    for key in lfw.TOKEN_MASS_SUM_KEYS:
        host._loss_forward_metrics.sum(key, torch.tensor(float("nan")))
    with pytest.raises(RuntimeError, match="not finite"):
        LossForwardImportanceMixin._flush_loss_forward_metrics(host)


# ---- the batch build -----------------------------------------------------------------------------------------------


def _fused_build_host(**kwargs):
    host = batch_host(
        [row([6, 8], [-1.0, -1.0]), row([6, 9], [-1.0, -0.5])], torch.zeros(2, 2), training=True, **kwargs
    )
    host._skips_logprob_pass = True
    host._loss_forward_metrics = WorldMetrics()

    def never(*_args, **_kwargs):
        raise AssertionError("the no-grad pass ran on a fused round")

    host._get_per_token_logps_and_entropies = never
    return host


def test_a_fused_round_builds_no_pass_and_hands_the_loss_its_sampling_side_inputs():
    host = _fused_build_host(is_correction=True, beta=0.0)
    batch = build(host, [episode(1.0), episode(0.0)])
    assert "importance_sampling_ratio" not in batch
    assert batch["old_per_token_logps"] is None
    assert {IS_SAMPLING_LOGPS_KEY, IS_CORRECTED_MASK_KEY, NEGATIVE_ONLY_ROWS_KEY} <= batch.keys()
    assert host._metrics["train"][IS_WEIGHTS_FROM_LOSS_FORWARD_KEY] == [1.0]
    assert "sampling/is_correction_coverage" in host._metrics["train"]
    assert LOGRATIO_MEAN_KEY not in host._metrics["train"], "the log-ratio is the loss forward's to report"


def test_a_fused_round_with_a_kl_term_hands_the_loss_the_unclamped_reference():
    host = _fused_build_host(is_correction=True, beta=0.04, reference=torch.full((2, 2), -9.0))
    batch = build(host, [episode(1.0), episode(0.0)])
    assert "ref_per_token_logps" not in batch
    assert torch.equal(batch[UNCLAMPED_REF_LOGPS_KEY], torch.full((2, 2), -9.0))


def test_a_kl_only_round_keeps_the_all_ones_ratio_the_loss_multiplies_by():
    host = _fused_build_host(is_correction=False, beta=0.04, reference=torch.full((2, 2), -9.0))
    batch = build(host, [episode(1.0), episode(0.0)])
    assert torch.equal(batch["importance_sampling_ratio"], torch.ones(2, 2))
    assert IS_SAMPLING_LOGPS_KEY not in batch


def test_an_eval_round_keeps_the_pass_and_its_ratio():
    host = batch_host([row([6], [-1.0]), row([7], [-1.0])], torch.zeros(2, 1), training=False, is_correction=True)
    host._skips_logprob_pass = True
    batch = build(host, [episode(1.0), episode(0.0)])
    assert "importance_sampling_ratio" in batch and IS_SAMPLING_LOGPS_KEY not in batch
    assert host._metrics["eval"][IS_WEIGHTS_FROM_LOSS_FORWARD_KEY] == [0.0]


def test_the_zero_gradient_tracker_counts_a_forced_close_only_round_as_no_gradient():
    """The ratio form's ``ratio > 0`` becomes the static ``~forced``: the round whose only loss token is a forced close
    trains nothing, and a round with one other token does."""
    forced_only = _fused_build_host(is_correction=True, forced_close_ids=(6,))
    forced_only._tokenize_step_rows = lambda results: [[r] for r in (row([6], [0.0]), row([6], [0.0]))]
    forced_only._get_per_token_logps_and_entropies = None
    build(forced_only, [episode(1.0), episode(0.0)])
    assert forced_only._metrics["train"][ZERO_GRADIENT_ROUND_KEY] == [1.0]
    trained = _fused_build_host(is_correction=True, forced_close_ids=(6,))
    trained._tokenize_step_rows = lambda results: [[r] for r in (row([6, 9], [0.0, -1.0]), row([6], [0.0]))]
    build(trained, [episode(1.0), episode(0.0)])
    assert trained._metrics["train"][ZERO_GRADIENT_ROUND_KEY] == [0.0]


def test_the_length_deal_keeps_every_weight_input_with_its_row():
    host = _fused_build_host(is_correction=True, beta=0.04, reference=torch.zeros(4, 2))
    host._tokenize_step_rows = lambda results: [
        [r] for r in (row([6, 8], [-1, -2]), row([6, 9], [-3, -4]), row([5, 8], [-5, -6]), row([5, 9], [-7, -8]))
    ]
    batch = build(host, [episode(1.0), episode(0.0), episode(1.0), episode(0.0)])
    order = torch.tensor([2, 0, 3, 1])
    dealt = permute_rows(batch, order)
    for key in (IS_SAMPLING_LOGPS_KEY, IS_CORRECTED_MASK_KEY, NEGATIVE_ONLY_ROWS_KEY, UNCLAMPED_REF_LOGPS_KEY):
        assert torch.equal(dealt[key], batch[key][order]), key
    assert dealt[IS_SAMPLING_LOGPS_KEY][0].tolist() == [-5.0, -6.0]


def test_the_log_ratio_formula_has_one_home():
    """The pass and the loss forward share ``compute_token_is_ratio``: same logps, same weight, bit for bit."""
    recompute = torch.randn(3, 5)
    sampling = torch.randn(3, 5)
    mask, has = torch.ones(3, 5, dtype=torch.bool), torch.ones(3, dtype=torch.bool)
    from_pass, diff, corrected = compute_is_ratio(recompute, sampling, mask, has, _CLIP)
    from_hook, hook_diff = lfw.compute_token_is_ratio(recompute, sampling, corrected, _CLIP)
    assert torch.equal(from_pass, from_hook) and torch.equal(diff, hook_diff)


# ---- the round-end flush's cadence ---------------------------------------------------------------------------------


class _TrlStep:
    """TRL's ``training_step``: it advances ``_step`` after the micro-step, before the caller reads it."""

    def __init__(self, advance: int = 1):
        self._advance = advance

    def training_step(self, model, inputs, num_items_in_batch=None):
        self._step += self._advance
        return torch.tensor(0.0)


class _Cadence(LossForwardImportanceMixin, _TrlStep):
    def __init__(self, spg: int, advance: int = 1):
        super().__init__(advance)
        self.args = types.SimpleNamespace(steps_per_generation=spg)
        self._step, self.flushes, self._skips_logprob_pass = 0, [], True

    def _flush_loss_forward_metrics(self):
        self.flushes.append(self._step)


@pytest.mark.parametrize("spg", [1, 2, 4])
def test_the_round_flush_fires_once_per_round_on_its_last_micro_step(spg):
    host = _Cadence(spg)
    for _ in range(3 * spg):  # a packed window's no-op slots are micro-steps like any other
        _Cadence.training_step(host, None, {})
    assert host.flushes == [spg, 2 * spg, 3 * spg]


def test_the_round_flush_follows_trls_step_not_the_micro_step_count():
    """Negative control: a step counter that runs one ahead moves every flush off the round's last micro-step."""
    host = _Cadence(4, advance=2)
    for _ in range(4):
        _Cadence.training_step(host, None, {})
    assert host.flushes != [4]


def test_an_unfused_host_never_flushes():
    host = _Cadence(2)
    host._skips_logprob_pass = False
    for _ in range(4):
        _Cadence.training_step(host, None, {})
    assert host.flushes == []


def test_the_mixin_runs_after_trls_step_increment_in_the_real_trainer():
    mro = DistributedAsyncEnvironmentalGRPOTrainer.__mro__
    assert mro.index(LossForwardImportanceMixin) < mro.index(GRPOTrainer)


def test_a_micro_batch_without_loss_tokens_leaves_the_round_maximum_to_the_others():
    host = types.SimpleNamespace(vllm_importance_sampling_clip_max=_CLIP, _loss_forward_metrics=WorldMetrics())
    ratio = torch.tensor([[0.5, 2.0]])
    seam = {"advantages": torch.tensor([1.0]), NEGATIVE_ONLY_ROWS_KEY: torch.tensor([False])}
    for loss_mask in (torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool)):
        LossForwardImportanceMixin._record_loss_forward_metrics(
            host, seam, torch.zeros(1, 2), loss_mask, ratio, torch.zeros(1, 2), loss_mask
        )
    assert host._loss_forward_metrics.flush()["sampling/is_ratio_max"] == 2.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
