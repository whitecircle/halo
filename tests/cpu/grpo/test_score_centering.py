#!/usr/bin/env python
"""Score centering (arXiv:2609.20807, App. A): the per-token estimator against a full-vocabulary oracle, its drift
cancellation, its degenerate inputs, and its composition with TRL's loss through the scorer's surrogate."""

import math
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from trl import GRPOTrainer

from halo.configs.grpo import ScoreCenteringConfig
from halo.trainers.grpo.environmental.centered_scoring import ScoreCenteringMixin
from halo.trainers.grpo.objective.logratio import compute_token_is_ratio
from halo.trainers.grpo.objective.loss_inputs import IS_SAMPLING_LOGPS_KEY
from halo.trainers.grpo.objective.score_centering import TOPK_IDS_KEY, TOPK_LOGPS_KEY, centered_logps
from halo.trainers.world_metrics import WorldMetrics

VOCAB = 48
CLIP, LOW, HIGH = 2.0, 0.5, 5.0
WEIGHTS = ("none", "tis", "mis")


def _logits(seed: int, scale: float) -> torch.Tensor:
    return torch.randn(VOCAB, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) * scale


def _centered(theta, sampler_logits, ys, head_ids, weights, *, eps=1e-6):
    """``X`` of tokens ``ys`` drawn from the sampler, scored under ``theta``, over candidate ids ``head_ids``."""
    lp = torch.log_softmax(theta, dim=-1)
    q = torch.log_softmax(sampler_logits, dim=-1)
    n = len(ys)
    ids = head_ids.expand(n, -1)
    mask = ids >= 0
    lp_head = torch.where(mask, lp[ids.clamp(min=0)], -math.inf)
    q_head = torch.where(mask, q[ids.clamp(min=0)], -math.inf)
    x, metrics = centered_logps(
        lp[ys],
        lp_head,
        q[ys],
        q_head,
        mask,
        torch.ones(n, dtype=torch.bool),
        weights=weights,
        tis_clip=CLIP,
        mis_low=LOW,
        mis_high=HIGH,
        eps=eps,
    )
    return x, metrics


def _reference_gradient(theta, sampler_logits, y, head, weights, eps=1e-6):
    """The estimator's gradient wrt the logits, written out per scalar: grad lp_v = onehot_v - p."""
    p, q = torch.softmax(theta, -1).tolist(), torch.softmax(sampler_logits, -1).tolist()

    def f(ratio):
        if weights == "none":
            return 1.0
        if weights == "tis":
            return min(ratio, CLIP)
        return ratio if LOW <= ratio <= HIGH else 0.0

    def mass_f(pv, qv):
        """q f(p / q), without the division where q = 0."""
        return 0.0 if qv == 0.0 else qv * f(pv / qv)

    mass_p, mass_q = sum(p[v] for v in head), sum(q[v] for v in head)
    rho = max(1 - mass_q, eps) / max(1 - mass_p, eps)
    alpha = rho * f(1 / rho)
    grad = [0.0] * VOCAB
    weight = f(p[y] / q[y])
    coefficients = {y: weight}
    for v in head:
        coefficients[v] = coefficients.get(v, 0.0) - (mass_f(p[v], q[v]) - alpha * p[v])
    for v, c in coefficients.items():
        for k in range(VOCAB):
            grad[k] += c * ((1.0 if k == v else 0.0) - p[k])
    return torch.tensor(grad, dtype=torch.float64)


def _gradient(theta, sampler_logits, ys, head_ids, weights, per_token_weights=None):
    x, _ = _centered(theta, sampler_logits, ys, head_ids, weights)
    coefficients = torch.ones(len(ys), dtype=torch.float64) if per_token_weights is None else per_token_weights
    (grad,) = torch.autograd.grad((coefficients * x).sum(), theta)
    return grad


def _theta():
    return _logits(0, 1.5).requires_grad_()


@pytest.mark.parametrize("weights", WEIGHTS)
@pytest.mark.parametrize("head", ["full", "top16", "y_outside"])
def test_gradient_is_the_centered_estimator_token_by_token(weights, head):
    theta, sampler = _theta(), _logits(1, 1.5)
    order = torch.argsort(sampler, descending=True)
    ids = {"full": order, "top16": order[:16], "y_outside": order[:8]}[head]
    ys = order[-3:] if head == "y_outside" else torch.cat([order[:2], order[20:22]])
    for y in ys.tolist():
        got = _gradient(theta, sampler, torch.tensor([y]), ids.unsqueeze(0), weights)
        torch.testing.assert_close(got, _reference_gradient(theta.detach(), sampler, y, ids.tolist(), weights))


@pytest.mark.parametrize("weights", WEIGHTS)
def test_full_head_gives_exact_centering_so_a_constant_reward_has_zero_expected_update(weights):
    theta, sampler = _theta(), _logits(1, 1.5)
    q = torch.softmax(sampler, -1)
    ys = torch.arange(VOCAB)
    ids = torch.arange(VOCAB).unsqueeze(0)
    expected = _gradient(theta, sampler, ys, ids, weights, per_token_weights=q)
    assert expected.abs().max() < 1e-9


def test_plain_policy_gradient_under_a_mismatched_sampler_drifts():
    """The negative control: a constant reward trained as sampled tokens' scores moves the logits by ``q - p``."""
    theta, sampler = _theta(), _logits(1, 1.5)
    q = torch.softmax(sampler, -1)
    lp = torch.log_softmax(theta, -1)
    (plain,) = torch.autograd.grad((q * lp).sum(), theta)
    assert plain.abs().max() > 1e-2
    torch.testing.assert_close(plain, q - torch.softmax(theta.detach(), -1))


@pytest.mark.parametrize("weights", WEIGHTS)
def test_a_partial_head_cancels_most_of_the_drift(weights):
    theta, sampler = _theta(), _logits(1, 1.5)
    q = torch.softmax(sampler, -1)
    lp = torch.log_softmax(theta, -1)
    (plain,) = torch.autograd.grad((q * lp).sum(), theta)
    head = torch.argsort(sampler, descending=True)[:16].unsqueeze(0)
    centered = _gradient(theta, sampler, torch.arange(VOCAB), head, weights, per_token_weights=q)
    assert centered.norm() < 0.5 * plain.norm()


def test_unused_slots_change_nothing_and_stay_finite():
    theta, sampler = _theta(), _logits(1, 1.5)
    order = torch.argsort(sampler, descending=True)
    ys = torch.tensor([order[3], order[30]])
    exact = order[:6].unsqueeze(0)
    padded = torch.cat([order[:4], torch.tensor([-1]), order[4:6], torch.tensor([-1, -1])]).unsqueeze(0)
    for weights in WEIGHTS:
        torch.testing.assert_close(
            _gradient(theta, sampler, ys, padded, weights), _gradient(theta, sampler, ys, exact, weights)
        )
        assert torch.isfinite(_gradient(theta, sampler, ys, padded, weights)).all()


def test_a_candidate_the_sampler_clamped_to_zero_mass_contributes_no_head_term():
    theta, sampler = _theta(), _logits(1, 1.5)
    ids = torch.argsort(sampler, descending=True)[:6].unsqueeze(0)
    lp, q = torch.log_softmax(theta, -1), torch.log_softmax(sampler, -1)
    q_clamped = q.clone()
    q_clamped[ids[0, 5]] = -9999.0
    mask = torch.ones(1, 6, dtype=torch.bool)
    for weights in ("none", "tis"):
        x, _ = centered_logps(
            lp[ids[0, 0]].reshape(1),
            lp[ids],
            q[ids[0, 0]].reshape(1),
            q_clamped[ids],
            mask,
            torch.ones(1, dtype=torch.bool),
            weights=weights,
            tis_clip=CLIP,
            mis_low=LOW,
            mis_high=HIGH,
        )
        assert torch.isfinite(x).all()


@pytest.mark.parametrize("weights", WEIGHTS)
def test_a_missing_head_is_the_importance_weighted_plain_score(weights):
    theta, sampler = _theta(), _logits(1, 1.5)
    ys = torch.arange(VOCAB)
    lp = torch.log_softmax(theta, -1)[ys]
    q = torch.log_softmax(sampler, -1)[ys]
    empty = torch.empty(VOCAB, 0, dtype=torch.long)
    x, _ = centered_logps(
        lp, lp[:, None][:, :0], q, lp[:, None][:, :0], empty >= 0, torch.ones(VOCAB, dtype=torch.bool),
        weights=weights, tis_clip=CLIP, mis_low=LOW, mis_high=HIGH,
    )  # fmt: skip
    (got,) = torch.autograd.grad(x.sum(), lp)
    if weights == "tis":
        # Bitwise today's truncated ratio.
        ratio, _ = compute_token_is_ratio(lp.detach(), q, torch.ones(VOCAB, dtype=torch.bool), CLIP)
        assert torch.equal(got, ratio)
    elif weights == "none":
        assert torch.equal(got, torch.ones_like(got))
    else:
        ratio = torch.exp(lp.detach() - q)
        assert torch.equal(got != 0, (ratio >= LOW) & (ratio <= HIGH))


def test_inactive_and_nan_tokens_train_on_their_plain_score_with_a_finite_gradient():
    lp_y = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float64, requires_grad=True)
    lp_head = torch.tensor([[-1.0, -2.0]] * 3, dtype=torch.float64, requires_grad=True)
    q_y = torch.tensor([-1.5, math.nan, -2.5], dtype=torch.float64)
    q_head = torch.tensor([[-1.0, -2.0], [math.nan, -2.0], [-1.0, -2.0]], dtype=torch.float64)
    mask = torch.ones(3, 2, dtype=torch.bool)
    active = torch.tensor([True, True, False])
    x, _ = centered_logps(
        lp_y, lp_head, q_y, q_head, mask, active, weights="tis", tis_clip=CLIP, mis_low=LOW, mis_high=HIGH
    )
    assert torch.equal(x[1:].detach(), lp_y[1:].detach())
    grad_y, grad_head = torch.autograd.grad(x.sum(), (lp_y, lp_head))
    assert torch.isfinite(grad_y).all() and torch.isfinite(grad_head).all()
    assert grad_y[1:].tolist() == [1.0, 1.0]
    assert grad_head[1:].abs().sum() == 0


def test_the_weights_are_detached_so_the_gradient_has_no_weight_derivative_term():
    """With ``tis`` the weight is a function of ``lp_y``; the oracle above fails if any of it carried a graph, and
    this pins the mechanism directly."""
    lp_y = torch.tensor([-1.0], dtype=torch.float64, requires_grad=True)
    lp_head = torch.tensor([[-1.5]], dtype=torch.float64, requires_grad=True)
    x, _ = centered_logps(
        lp_y, lp_head, torch.tensor([-1.2], dtype=torch.float64), torch.tensor([[-1.4]], dtype=torch.float64),
        torch.ones(1, 1, dtype=torch.bool), torch.ones(1, dtype=torch.bool), weights="tis", tis_clip=CLIP,
        mis_low=LOW, mis_high=HIGH,
    )  # fmt: skip
    assert x.grad_fn is not None
    (grad,) = torch.autograd.grad(x.sum(), lp_y)
    assert grad.item() == pytest.approx(math.exp(-1.0 + 1.2))


def test_a_refused_tis_clip_names_the_value():
    with pytest.raises(ValueError, match="TIS clip"):
        centered_logps(
            torch.zeros(1), torch.zeros(1, 1), torch.zeros(1), torch.zeros(1, 1),
            torch.ones(1, 1, dtype=torch.bool), torch.ones(1, dtype=torch.bool), weights="tis", tis_clip=0.0,
            mis_low=LOW, mis_high=HIGH,
        )  # fmt: skip


# --- composition with TRL's loss through the scorer's surrogate -------------------------------------------------


class _DenseScorer:
    """Stands in for the chunked scorer: the full-vocabulary log-probs at every completion position, gathered at the
    sampled token and, when candidates are armed, at them."""

    _armed = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return GRPOTrainer._compute_loss(self, model, inputs)

    def _scoring_with_candidates(self, ids):
        host = self

        class _Armed:
            def __enter__(self):
                host._armed = ids

            def __exit__(self, *exc):
                host._armed = None

        return _Armed()

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, **kwargs):
        lp = torch.log_softmax(self.theta, dim=-1)  # [rows, T, V]
        sampled = input_ids[:, -logits_to_keep:]
        picked = lp.gather(-1, sampled.unsqueeze(-1))
        if self._armed is not None:
            ids = self._armed
            picked = torch.cat([picked, torch.where(ids >= 0, lp.gather(-1, ids.clamp(min=0).long()), -math.inf)], -1)
        entropy = -(lp.exp() * lp).sum(-1).detach()
        return picked.squeeze(-1) if self._armed is None else picked, entropy, None


class _CenteredHost(ScoreCenteringMixin, _DenseScorer):
    """TRL's real ``_compute_loss`` over the scorer, on a bare host."""

    def __init__(self, theta, loss_type, config):
        self.theta = theta
        self._score_centering = config
        self._centering_metrics = WorldMetrics()
        self.vllm_importance_sampling_clip_max = CLIP
        self._metrics = defaultdict(lambda: defaultdict(list))
        self.loss_type = loss_type
        self.top_entropy_quantile, self.off_policy_mask_threshold = 1.0, None
        self.importance_sampling_level, self.beta = "token", 0.0
        self.epsilon_low, self.epsilon_high = 0.2, 0.2
        self.use_vllm, self.vllm_importance_sampling_correction = False, False
        self.args = SimpleNamespace(
            delta=None,
            sapo_temperature_pos=1.0,
            sapo_temperature_neg=1.05,
            use_bias_correction_kl=False,
            steps_per_generation=1,
        )
        self._entropy_bonus_enabled, self.aux_loss_enabled = False, False
        self.current_gradient_accumulation_steps = 1
        self.max_completion_length = theta.shape[1]
        self.accelerator = SimpleNamespace(
            num_processes=1, gather=lambda t: t, reduce=lambda t, reduction: t, sync_gradients=True
        )
        self.model = SimpleNamespace(training=True)

    def _loss(self, batch):
        return self.compute_loss(self.model, batch)

    def get_high_entropy_mask(self, *args):
        raise AssertionError("top_entropy_quantile is 1")


def _batch(rows=2, width=6, k=5, seed=3):
    generator = torch.Generator().manual_seed(seed)
    theta = (torch.randn(rows, width, VOCAB, generator=generator, dtype=torch.float64) * 1.5).requires_grad_()
    sampler = torch.randn(rows, width, VOCAB, generator=generator, dtype=torch.float64) * 1.5
    sampled = torch.distributions.Categorical(logits=sampler).sample()
    q = torch.log_softmax(sampler, -1)
    top = torch.topk(q, k, dim=-1).indices
    batch = {
        "prompt_ids": torch.zeros(rows, 2, dtype=torch.long),
        "prompt_mask": torch.ones(rows, 2, dtype=torch.long),
        "completion_ids": sampled,
        "completion_mask": torch.ones(rows, width, dtype=torch.long),
        "advantages": torch.tensor([1.0, -0.5], dtype=torch.float64)[:rows],
        "num_items_in_batch": torch.tensor(float(rows * width)),
        TOPK_IDS_KEY: top.int(),
        TOPK_LOGPS_KEY: q.gather(-1, top),
        IS_SAMPLING_LOGPS_KEY: q.gather(-1, sampled.unsqueeze(-1)).squeeze(-1),
    }
    return theta, batch


@pytest.mark.parametrize("loss_type", ["dapo", "cispo", "grpo", "bnpo", "dr_grpo", "luspo", "sapo"])
@pytest.mark.parametrize("weights", WEIGHTS)
def test_trls_loss_keeps_its_value_and_takes_the_centered_gradient(loss_type, weights):
    theta, batch = _batch()
    config = ScoreCenteringConfig(top_k=5, weights=weights, mis_low=LOW, mis_high=HIGH)
    centered = _CenteredHost(theta, loss_type, config)
    plain = _CenteredHost(theta, loss_type, None)
    value = centered._loss(batch)
    baseline = plain._loss(batch)
    assert torch.equal(value.detach(), baseline.detach())
    (grad,) = torch.autograd.grad(value, theta, retain_graph=True)
    (plain_grad,) = torch.autograd.grad(baseline, theta)
    assert not torch.allclose(grad, plain_grad, atol=1e-6), "centering must change the gradient"

    # The same loss with the scorer's log-probs replaced by the estimator's own surrogate, built per token.
    lp = torch.log_softmax(theta, -1)
    ids = batch[TOPK_IDS_KEY].long()
    sampled = batch["completion_ids"]
    lp_y = lp.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    x, _ = centered_logps(
        lp_y.reshape(-1),
        lp.gather(-1, ids).reshape(-1, ids.shape[-1]),
        batch[IS_SAMPLING_LOGPS_KEY].reshape(-1),
        batch[TOPK_LOGPS_KEY].reshape(-1, ids.shape[-1]),
        (ids >= 0).reshape(-1, ids.shape[-1]),
        (batch[IS_SAMPLING_LOGPS_KEY] < 0).reshape(-1),
        weights=weights,
        tis_clip=CLIP,
        mis_low=LOW,
        mis_high=HIGH,
    )
    x = x.reshape(lp_y.shape)

    class _Reference(_CenteredHost):
        def _get_per_token_logps_and_entropies(self, model, *args, **kwargs):
            return lp_y.detach() + (x - x.detach()), -(lp.exp() * lp).sum(-1).detach(), None

    expected = _Reference(theta, loss_type, None)._loss(batch)
    (expected_grad,) = torch.autograd.grad(expected, theta)
    torch.testing.assert_close(grad, expected_grad, atol=1e-12, rtol=1e-9)
    assert grad.abs().sum() > 0


def test_a_token_the_sampler_emitted_with_probability_one_trains_on_its_plain_score():
    theta, batch = _batch(rows=1, width=3)
    batch[IS_SAMPLING_LOGPS_KEY] = torch.zeros_like(batch[IS_SAMPLING_LOGPS_KEY])
    config = ScoreCenteringConfig(top_k=5, weights="tis")
    centered = _CenteredHost(theta, "dapo", config)._loss(batch)
    plain = _CenteredHost(theta, "dapo", None)._loss(batch)
    (got,) = torch.autograd.grad(centered, theta)
    (want,) = torch.autograd.grad(plain, theta)
    assert torch.equal(got, want)


def test_the_loss_forward_needs_the_candidates_in_its_batch():
    theta, batch = _batch(rows=1, width=3)
    del batch[TOPK_LOGPS_KEY]
    with pytest.raises(RuntimeError, match=TOPK_LOGPS_KEY):
        _CenteredHost(theta, "dapo", ScoreCenteringConfig(top_k=5))._loss(batch)


def test_the_metrics_cover_only_the_centered_loss_tokens():
    theta, batch = _batch(rows=2, width=4)
    batch["completion_mask"][1] = 0
    host = _CenteredHost(theta, "dapo", ScoreCenteringConfig(top_k=5))
    host._loss(batch)
    logged = host._metrics["train"]
    host._centering_metrics.flush(logged)
    assert logged["score_centering/active_frac"] == [1.0]
    assert 0 < logged["score_centering/head_mass_mean"][0] <= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
