"""Score centering of the GRPO token gradient (arXiv:2609.20807, App. A): the sampled token's score is weighted by the
importance weight, and the score-function mean of the sampler's top-K candidates, tail included, is subtracted.
:func:`centered_logps` returns the per-token term whose gradient is that estimator."""

import math
from dataclasses import dataclass, fields

import torch

from halo.configs.grpo import ScoreCenteringWeights

TOPK_IDS_KEY = "rollout_topk_ids"
"""Batch key of the sampler's candidate ids: int32 ``[rows, completion_width, K]``, unused slots -1."""

TOPK_LOGPS_KEY = "rollout_topk_logps"
"""Batch key of the candidates' sampler log-probs, the log-softmax of the sampling logits over the full vocabulary:
float32, the shape of :data:`TOPK_IDS_KEY`, unused slots -inf."""


@dataclass(frozen=True)
class ScoreCenteringMetrics:
    """Detached per-token diagnostics, over every token; the caller selects the ones it reports."""

    correction: torch.Tensor
    head_mass: torch.Tensor
    sampler_head_mass: torch.Tensor
    tail_ratio: torch.Tensor
    weight: torch.Tensor

    def as_log_dict(self) -> dict[str, torch.Tensor]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def _importance_terms(
    weights: ScoreCenteringWeights,
    log_ratio_y: torch.Tensor,
    p_head: torch.Tensor,
    q_head: torch.Tensor,
    log_ratio_head: torch.Tensor,
    rho: torch.Tensor,
    *,
    tis_clip: float,
    mis_low: float,
    mis_high: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(w_y, q f(p/q) on the head, alpha)`` of the weight ``f``: ``w_y = f(p_y / q_y)`` and the tail scale
    ``alpha = rho f(1 / rho)``. Ratios are tested in log space, so no term exponentiates an unbounded value."""
    if weights == "none":
        return torch.ones_like(log_ratio_y), q_head, rho
    if weights == "tis":
        if not math.isfinite(tis_clip) or tis_clip <= 0:
            raise ValueError(f"The score-centering TIS clip must be positive and finite, got {tis_clip}")
        weight = log_ratio_y.exp().clamp_max(tis_clip)
        # q min(p / q, c) without 0 * inf at q = 0.
        return weight, torch.minimum(p_head, tis_clip * q_head), (tis_clip * rho).clamp_max(1.0)
    if weights == "mis":
        low, high = math.log(mis_low), math.log(mis_high)
        inside_y = (log_ratio_y >= low) & (log_ratio_y <= high)
        inside_head = (log_ratio_head >= low) & (log_ratio_head <= high)
        alpha = (rho >= 1 / mis_high) & (rho <= 1 / mis_low)
        return (
            torch.where(inside_y, log_ratio_y.clamp(max=high).exp(), 0.0),
            torch.where(inside_head, p_head, 0.0),
            alpha.to(rho.dtype),
        )
    raise ValueError(f"Unknown score-centering weights {weights!r}")


def centered_logps(
    lp_y: torch.Tensor,
    lp_head: torch.Tensor,
    q_y: torch.Tensor,
    q_head: torch.Tensor,
    head_mask: torch.Tensor,
    active: torch.Tensor,
    *,
    weights: ScoreCenteringWeights,
    tis_clip: float,
    mis_low: float,
    mis_high: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, ScoreCenteringMetrics]:
    """``(X, metrics)`` with ``grad X`` the centered estimator of the token's policy gradient:
    ``X = w_y lp_y - sum_v c_v lp_v`` over the head candidates ``v``, ``w_y = f(p_y / q_y)``,
    ``c_v = q_v f(p_v / q_v) - alpha p_v``, ``alpha = rho f(1 / rho)`` and ``rho`` the sampler-to-trainer ratio of the
    mass outside the head. Every weight is detached, as differentiating one changes the estimator.

    ``lp_y`` ``[N]`` and ``lp_head`` ``[N, K]`` are the trainer's live log-probs of the sampled token and the
    candidates, ``q_y`` and ``q_head`` the sampler's, ``head_mask`` ``[N, K]`` marks the candidates a token has (the
    sampled token need not be one), ``active`` ``[N]`` the tokens that are centered. An inactive token, or one whose
    sampler log-prob is not finite, returns ``lp_y``: its plain score, with no weight. The tail masses are floored at
    ``eps``, so the cancellation is exact for a full head and approximate for a true tail that differs from the
    trainer's rescaled one."""
    usable = active & torch.isfinite(q_y)
    slot = head_mask & usable.unsqueeze(-1) & torch.isfinite(lp_head.detach())
    with torch.no_grad():
        p_head = lp_head.detach().masked_fill(~slot, -math.inf).exp()
        q_log = q_head.masked_fill(~slot, -math.inf)
        q_prob = q_log.exp()
        mass_p, mass_q = p_head.sum(-1), q_prob.sum(-1)
        rho = (1 - mass_q).clamp_min(eps) / (1 - mass_p).clamp_min(eps)
        log_ratio_y = torch.where(usable, lp_y.detach() - q_y, 0.0)
        log_ratio_head = torch.where(slot, lp_head.detach() - q_log, 0.0)
        weight, head_mass, alpha = _importance_terms(
            weights,
            log_ratio_y,
            p_head,
            q_prob,
            log_ratio_head,
            rho,
            tis_clip=tis_clip,
            mis_low=mis_low,
            mis_high=mis_high,
        )
        residual = head_mass - alpha.unsqueeze(-1) * p_head
    correction = (residual * torch.where(slot, lp_head, 0.0)).sum(-1)
    centered = torch.where(usable, weight * lp_y - correction, lp_y)
    return centered, ScoreCenteringMetrics(
        correction=correction.detach(),
        head_mass=mass_p,
        sampler_head_mass=mass_q,
        tail_ratio=rho,
        weight=weight,
    )
