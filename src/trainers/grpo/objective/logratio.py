"""Truncated log-ratio corrections in the environmental-GRPO objective.

Two loss terms are log-prob ratios with an unbounded tail one token can dominate, each truncated:
:func:`compute_is_ratio` (vLLM↔trainer drift, clamped at ``vllm_importance_sampling_clip_max``) and
:func:`clamp_ref_logps` (k3 KL estimator, capped at :data:`KL_LOGRATIO_CLAMP`).

On top, :func:`apply_is_masks` and :func:`apply_opsm` add masking-over-reweighting stages for MoE-scale
mismatch (token band, trajectory geometric-mean band, catastrophic-token veto, OPSM), all default off.
A masked token/trajectory gets ratio 0 (policy-gradient term vanishes, DAPO normalizer unchanged); the
β·k3 KL term is added after the ratio multiply, so masked tokens stay anchored to the reference.

Trajectory aggregation pools all of an episode's turn rows via ``traj_ids`` — the drift compounds over
the whole episode. Pure functions: no trainer state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

KL_LOGRATIO_CLAMP = 5.0
"""Cap on ``ref − logp`` (nats) in the k3 KL estimator, bounding per-token KL at ``exp(5) ≈ 148``."""


def clamp_ref_logps(ref_logps: torch.Tensor, policy_logps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Bound the exploding tail of TRL's k3 KL estimator by capping the log-ratio at
    :data:`KL_LOGRATIO_CLAMP` nats — unconditional, not a knob.

    ``per_token_kl = exp(ref − logp) − (ref − logp) − 1`` is unbounded where the policy suppresses a token
    the reference likes; capping ``ref`` at ``policy + KL_LOGRATIO_CLAMP`` truncates that tail.
    Returns ``(clamped_ref, fraction_clamped)``.
    """
    ceiling = policy_logps + KL_LOGRATIO_CLAMP
    return torch.minimum(ref_logps, ceiling), (ref_logps > ceiling).float().mean()


def compute_is_ratio(
    recompute_logps: torch.Tensor,
    sampling_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    row_has_sampling: torch.Tensor,
    clip_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Truncated per-token vLLM→trainer importance ratio ``clamp(exp(logπ_recompute − logπ_sampling), clip_max)``.

    Rows flagged ``False`` in ``row_has_sampling`` (rollout error dropped their vLLM logprobs) and
    non-policy tokens get ratio exactly 1, so one bad row cannot perturb the others.
    Returns ``(ratio, logps_diff, corrected_mask)``, all shaped like ``completion_mask``.
    """
    corrected_mask = completion_mask.bool() & row_has_sampling.unsqueeze(1)
    logps_diff = (recompute_logps - sampling_logps) * corrected_mask
    return torch.clamp(torch.exp(logps_diff), max=clip_max), logps_diff, corrected_mask


@dataclass(frozen=True)
class ISMaskConfig:
    """Mask/veto stages layered on the truncated IS ratio (see module docstring). All default off.

    * ``band_min``/``band_max`` — TOKEN band: mask a corrected token whose ratio falls outside it.
    * ``geo_band_min``/``geo_band_max`` — TRAJECTORY geometric-mean band: mask the whole trajectory when
      ``exp(mean log-ratio over its corrected tokens)`` leaves the band. Both bounds must be set.
    * ``veto_min`` — catastrophic-token veto: mask the trajectory when ANY corrected token's ratio is below it.
    * ``opsm_delta`` — see :func:`apply_opsm` (applied separately, once advantages exist).
    """

    band_min: float | None = None
    band_max: float | None = None
    geo_band_min: float | None = None
    geo_band_max: float | None = None
    veto_min: float | None = None
    opsm_delta: float | None = None

    def __post_init__(self):
        for lo, hi, name in (
            (self.band_min, self.band_max, "band"),
            (self.geo_band_min, self.geo_band_max, "geo_band"),
        ):
            if (lo is None) != (hi is None):
                raise ValueError(f"is_{name}_min and is_{name}_max must be set together")
            if lo is not None and not 0 < lo < 1 < hi:
                raise ValueError(f"is_{name} bounds must satisfy 0 < min < 1 < max, got [{lo}, {hi}]")
        if self.veto_min is not None and not 0 < self.veto_min < 1:
            raise ValueError(f"isr_veto_min must be in (0, 1), got {self.veto_min}")
        if self.opsm_delta is not None and self.opsm_delta <= 0:
            raise ValueError(f"isr_opsm_delta must be > 0 (nats), got {self.opsm_delta}")

    @property
    def any_mask_active(self) -> bool:
        return any(v is not None for v in (self.band_min, self.geo_band_min, self.veto_min))


def _traj_scatter(values: torch.Tensor, traj_ids: torch.Tensor, num_trajs: int, reduce: str) -> torch.Tensor:
    """Per-trajectory reduction: rows sharing ``traj_ids`` pool; id < 0 (padding) reduces into a discarded slot."""
    out = torch.zeros(num_trajs + 1, device=values.device, dtype=values.dtype)
    idx = torch.where(traj_ids >= 0, traj_ids, torch.full_like(traj_ids, num_trajs))
    out.scatter_reduce_(0, idx, values, reduce=reduce, include_self=False)
    return out[:num_trajs]


def _traj_mean_logratio(
    logps_diff: torch.Tensor, corrected_mask: torch.Tensor, traj_ids: torch.Tensor, num_trajs: int
) -> torch.Tensor:
    """Mean log-ratio per trajectory over the corrected tokens (the geometric-mean ratio in log space)."""
    row_sum = (logps_diff * corrected_mask).sum(dim=1)
    row_cnt = corrected_mask.sum(dim=1).to(row_sum.dtype)
    token_count = _traj_scatter(row_cnt, traj_ids, num_trajs, "sum").clamp(min=1.0)
    return _traj_scatter(row_sum, traj_ids, num_trajs, "sum") / token_count


def apply_is_masks(
    ratio: torch.Tensor,
    logps_diff: torch.Tensor,
    corrected_mask: torch.Tensor,
    traj_ids: torch.Tensor,
    config: ISMaskConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply the token-band / geometric-band / veto stages to the truncated ratio.

    ``traj_ids`` maps each ROW to its trajectory (−1 for dummy rows). The band tests use the raw
    (pre-truncation) ratio ``exp(logps_diff)`` so the clip cannot hide an out-of-band token.
    Returns the masked ratio and diagnostic fractions.
    """
    if not config.any_mask_active:
        return ratio, {}
    raw_ratio = torch.exp(logps_diff)
    keep = torch.ones_like(ratio, dtype=torch.bool)
    stats: dict[str, float] = {}
    corrected = corrected_mask.sum().clamp(min=1)
    num_trajs = int(traj_ids.max().item()) + 1 if traj_ids.numel() else 0

    if config.band_min is not None:
        in_band = (raw_ratio >= config.band_min) & (raw_ratio <= config.band_max)
        token_keep = in_band | ~corrected_mask
        stats["sampling/is_token_band_masked_frac"] = ((~token_keep) & corrected_mask).sum().item() / corrected.item()
        keep &= token_keep

    traj_keep = None
    if num_trajs and (config.geo_band_min is not None or config.veto_min is not None):
        traj_keep = torch.ones(num_trajs, device=ratio.device, dtype=torch.bool)
        if config.geo_band_min is not None:
            geo = torch.exp(_traj_mean_logratio(logps_diff, corrected_mask, traj_ids, num_trajs))
            in_geo = (geo >= config.geo_band_min) & (geo <= config.geo_band_max)
            stats["sampling/is_geo_band_masked_frac"] = (~in_geo).float().mean().item()
            traj_keep &= in_geo
        if config.veto_min is not None:
            # uncorrected tokens read as 1.0 so they never trip the veto.
            row_min = torch.where(corrected_mask, raw_ratio, torch.ones_like(raw_ratio)).min(dim=1).values
            traj_min = _traj_scatter(row_min, traj_ids, num_trajs, "amin")
            # An id with no rows keeps the 0 scatter init, which reads as vetoed — restrict to contributing ids.
            present = _traj_scatter(torch.ones_like(row_min), traj_ids, num_trajs, "sum") > 0
            vetoed = (traj_min < config.veto_min) & present
            stats["sampling/is_veto_masked_frac"] = vetoed.float().mean().item()
            traj_keep &= ~vetoed
    if traj_keep is not None:
        row_keep = torch.where(
            traj_ids >= 0, traj_keep.gather(0, traj_ids.clamp(min=0)), torch.ones_like(traj_ids).bool()
        )
        keep &= row_keep.unsqueeze(1)

    return torch.where(keep, ratio, torch.zeros_like(ratio)), stats


def apply_opsm(
    ratio: torch.Tensor,
    logps_diff: torch.Tensor,
    corrected_mask: torch.Tensor,
    traj_ids: torch.Tensor,
    row_advantages: torch.Tensor,
    delta: float,
) -> tuple[torch.Tensor, float]:
    """Off-Policy Sequence Masking (DeepSeek-V3.2): zero the ratio of NEGATIVE-advantage trajectories
    whose mean log-ratio magnitude exceeds ``delta`` nats. Positive-advantage trajectories are never
    masked. Returns the masked ratio and the masked-trajectory fraction.
    """
    num_trajs = int(traj_ids.max().item()) + 1 if traj_ids.numel() else 0
    if num_trajs == 0:
        return ratio, 0.0
    traj_mean = _traj_mean_logratio(logps_diff, corrected_mask, traj_ids, num_trajs)
    traj_negative = _traj_scatter(row_advantages, traj_ids, num_trajs, "amin") < 0
    masked = traj_negative & (traj_mean.abs() > delta)
    row_masked = torch.where(traj_ids >= 0, masked.gather(0, traj_ids.clamp(min=0)), torch.zeros_like(traj_ids).bool())
    out = torch.where(row_masked.unsqueeze(1), torch.zeros_like(ratio), ratio)
    return out, masked.float().mean().item()
