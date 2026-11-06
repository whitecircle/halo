"""Truncated log-ratio terms of the GRPO objectives: the k3 KL tail clamp, the importance ratio, and the masking stages
(default off), which give a masked token ratio 0 and leave the DAPO normalizer and the KL anchor in place."""

from __future__ import annotations

import torch

from halo.configs.grpo_environmental import ISRMaskConfig

KL_LOGRATIO_CLAMP = 5.0
"""Cap on ``ref − logp`` (nats) in the k3 KL estimator, bounding per-token KL at ``exp(5) ≈ 148``."""

KL_CLAMP_FRAC_KEY = "kl_clamp_frac"
"""Metric key of the share of reference log-probs :func:`clamp_ref_logps` capped (online and environmental)."""

LOGRATIO_MEAN_KEY = "sampling/logratio_mean"
"""Metric key of the environmental trainer's mean :func:`compute_is_ratio` log-ratio over the corrected tokens."""

UPDATE_SKIPPED_KEY = "sampling/update_skipped"
"""Metric key of the environmental trainer's trust-region breaker verdict over the mask stages: 1 when it skipped
the round's update."""

SAMPLER_CERTAIN_LOGPROB = 0.0
"""A sampling logprob of exactly 0 is a token the engine emitted with probability 1: a logits processor
forced it (vLLM's thinking budget closing ``</think>``) or the nucleus collapsed onto it."""


def sampled_token_mask(completion_mask: torch.Tensor, row_has_sampling: torch.Tensor) -> torch.Tensor:
    """The only tokens a sampler-side quantity (a forced close, a certain token, the IS ratio) can be read on."""
    return completion_mask.bool() & row_has_sampling.unsqueeze(1)


def sampler_certain_mask(
    sampling_logps: torch.Tensor, completion_mask: torch.Tensor, row_has_sampling: torch.Tensor
) -> torch.Tensor:
    """Tokens the sampler emitted with probability 1: no sampling choice was made there, so they carry no importance
    weight."""
    return sampled_token_mask(completion_mask, row_has_sampling) & (sampling_logps >= SAMPLER_CERTAIN_LOGPROB)


def engine_forced_close_mask(
    sampling_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    row_has_sampling: torch.Tensor,
    completion_ids: torch.Tensor,
    reasoning_end_ids: tuple[int, ...],
) -> torch.Tensor:
    """Tokens of a reasoning close the engine forced at a turn's thinking cap, from the sampling side alone."""
    certain = sampler_certain_mask(sampling_logps, completion_mask, row_has_sampling)
    width = len(reasoning_end_ids)
    forced = torch.zeros_like(certain)
    if completion_ids.shape[1] >= width:
        end = completion_ids.new_tensor(reasoning_end_ids)
        starts = ((completion_ids.unfold(1, width, 1) == end) & certain.unfold(1, width, 1)).all(-1)
        for offset in range(width):
            forced[:, offset : offset + starts.shape[1]] |= starts
    return forced


def zero_engine_forced_closes(
    ratio: torch.Tensor,
    sampling_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    row_has_sampling: torch.Tensor,
    completion_ids: torch.Tensor,
    reasoning_end_ids: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ratio zeroed at reasoning closes the engine forced at a turn's thinking cap. Trained with the episode's
    advantage, repeated forced closes drive down the model's own probability of ending its reasoning."""
    forced = engine_forced_close_mask(
        sampling_logps, completion_mask, row_has_sampling, completion_ids, reasoning_end_ids
    )
    return ratio.masked_fill(forced, 0.0), forced


def clamp_ref_logps(ref_logps: torch.Tensor, policy_logps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cap the k3 estimator's log-ratio at :data:`KL_LOGRATIO_CLAMP` nats, as ``exp(ref − logp) − (ref − logp) − 1`` is
    unbounded where the policy suppresses a token the reference favors. Returns ``(clamped_ref, clamped)``."""
    ceiling = policy_logps + KL_LOGRATIO_CLAMP
    return torch.minimum(ref_logps, ceiling), ref_logps > ceiling


def kl_clamp_counts(clamped: torch.Tensor, loss_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Over the loss mask the KL term is averaged on, so a cap that bit on padding, a tool-output token or a dropped
    row counts nowhere."""
    loss_mask = loss_mask.bool()
    return (clamped & loss_mask).sum(), loss_mask.sum()


def corrected_token_mask(
    sampling_logps: torch.Tensor, completion_mask: torch.Tensor, row_has_sampling: torch.Tensor
) -> torch.Tensor:
    """The tokens the importance ratio corrects: sampled, and not emitted with probability 1."""
    return sampled_token_mask(completion_mask, row_has_sampling) & ~sampler_certain_mask(
        sampling_logps, completion_mask, row_has_sampling
    )


def compute_token_is_ratio(
    logps: torch.Tensor, sampling_logps: torch.Tensor, corrected_mask: torch.Tensor, clip_max: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``clamp(exp(logπ − logπ_sampling), clip_max)`` on the corrected tokens, exactly 1 elsewhere, and the masked
    log-ratio it came from. Returns ``(ratio, logps_diff)``."""
    logps_diff = (logps - sampling_logps) * corrected_mask
    return torch.clamp(torch.exp(logps_diff), max=clip_max), logps_diff


def compute_is_ratio(
    recompute_logps: torch.Tensor,
    sampling_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    row_has_sampling: torch.Tensor,
    clip_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Truncated per-token ratio ``clamp(exp(logπ_recompute − logπ_sampling), clip_max)``. Rows without sampling
    logprobs, non-policy tokens and sampler-certain tokens get exactly 1, so one bad row cannot perturb the others and
    no band or veto trips. Returns ``(ratio, logps_diff, corrected_mask)``."""
    corrected_mask = corrected_token_mask(sampling_logps, completion_mask, row_has_sampling)
    ratio, logps_diff = compute_token_is_ratio(recompute_logps, sampling_logps, corrected_mask, clip_max)
    return ratio, logps_diff, corrected_mask


def select_mask_logratio(
    logps_diff: torch.Tensor,
    recompute_logps: torch.Tensor,
    sampling_logps: torch.Tensor,
    engine_logps: torch.Tensor,
    corrected_mask: torch.Tensor,
    row_has_engine: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """``logπ_engine_now − logπ_sampling`` on rows the engine re-scored under current weights (pure staleness, as both
    engine passes share numerics), the trainer diff elsewhere; plus on-device ``(numerator, denominator)``
    diagnostics."""
    use_engine = corrected_mask & row_has_engine.unsqueeze(1)
    engine_diff = (engine_logps - sampling_logps) * use_engine
    mask_diff = torch.where(use_engine, engine_diff, logps_diff)
    n = use_engine.sum()
    stats = {
        "sampling/engine_logratio_mean": (engine_diff.sum(), n),
        "sampling/numerics_logratio_mean": (((recompute_logps - engine_logps) * use_engine).sum(), n),
        "sampling/engine_rescore_coverage": (n, corrected_mask.sum()),
    }
    return mask_diff, stats


def _num_trajs(traj_ids: torch.Tensor) -> int:
    """Ids ``0..n-1``, −1 for dummy rows. A host read: it sizes the per-trajectory reductions."""
    return int(traj_ids.max()) + 1 if traj_ids.numel() else 0


def _traj_scatter(values: torch.Tensor, traj_ids: torch.Tensor, num_trajs: int, reduce: str) -> torch.Tensor:
    """Per-trajectory reduction: rows sharing ``traj_ids`` pool; id < 0 (padding) reduces into a discarded slot."""
    out = torch.zeros(num_trajs + 1, device=values.device, dtype=values.dtype)
    idx = torch.where(traj_ids >= 0, traj_ids, torch.full_like(traj_ids, num_trajs))
    out.scatter_reduce_(0, idx, values, reduce=reduce, include_self=False)
    return out[:num_trajs]


def _traj_to_rows(per_traj: torch.Tensor, traj_ids: torch.Tensor, dummy: bool) -> torch.Tensor:
    """Each row's entry of the per-trajectory bool ``per_traj``; a dummy row (id < 0) reads ``dummy``."""
    dummy_rows = torch.full_like(traj_ids, dummy, dtype=torch.bool)
    return torch.where(traj_ids >= 0, per_traj.gather(0, traj_ids.clamp(min=0)), dummy_rows)


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
    config: ISRMaskConfig,
) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, int]]]:
    """The veto tests the raw ratio ``exp(logps_diff)``, so the clip cannot hide a catastrophic token. Returns the
    masked ratio and each active stage's ``(masked, total)`` counts, the masked count left on device."""
    if not config.any_mask_active:
        return ratio, {}
    num_trajs = _num_trajs(traj_ids)
    if not num_trajs:
        return ratio, {}
    raw_ratio = torch.exp(logps_diff)
    stats: dict[str, tuple[int, int]] = {}
    traj_keep = torch.ones(num_trajs, device=ratio.device, dtype=torch.bool)
    if config.geo_band_min is not None:
        geo = torch.exp(_traj_mean_logratio(logps_diff, corrected_mask, traj_ids, num_trajs))
        in_geo = (geo >= config.geo_band_min) & (geo <= config.geo_band_max)
        stats["sampling/is_geo_band_masked_frac"] = ((~in_geo).sum(), num_trajs)
        traj_keep &= in_geo
    if config.veto_min is not None:
        row_min = torch.where(corrected_mask, raw_ratio, torch.ones_like(raw_ratio)).min(dim=1).values
        traj_min = _traj_scatter(row_min, traj_ids, num_trajs, "amin")
        # An id with no rows keeps the 0 scatter init, which reads as vetoed; restrict to contributing ids.
        present = _traj_scatter(torch.ones_like(row_min), traj_ids, num_trajs, "sum") > 0
        vetoed = (traj_min < config.veto_min) & present
        stats["sampling/is_veto_masked_frac"] = (vetoed.sum(), num_trajs)
        traj_keep &= ~vetoed
    row_keep = _traj_to_rows(traj_keep, traj_ids, dummy=True)
    return torch.where(row_keep.unsqueeze(1), ratio, torch.zeros_like(ratio)), stats


def apply_opsm(
    ratio: torch.Tensor,
    logps_diff: torch.Tensor,
    corrected_mask: torch.Tensor,
    traj_ids: torch.Tensor,
    row_advantages: torch.Tensor,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Off-Policy Sequence Masking (DeepSeek-V3.2): zero negative-advantage trajectories whose mean log-ratio magnitude
    exceeds ``delta`` nats. Returns the masked ratio and the per-trajectory mask."""
    num_trajs = _num_trajs(traj_ids)
    if num_trajs == 0:
        return ratio, torch.zeros(0, dtype=torch.bool, device=ratio.device)
    traj_mean = _traj_mean_logratio(logps_diff, corrected_mask, traj_ids, num_trajs)
    traj_negative = _traj_scatter(row_advantages, traj_ids, num_trajs, "amin") < 0
    masked = traj_negative & (traj_mean.abs() > delta)
    row_masked = _traj_to_rows(masked, traj_ids, dummy=False)
    out = torch.where(row_masked.unsqueeze(1), torch.zeros_like(ratio), ratio)
    return out, masked
