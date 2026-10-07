"""Truncated log-ratio terms of the GRPO objectives.

Two loss terms are log-prob ratios with an unbounded tail one token can dominate, each truncated:
:func:`clamp_ref_logps` caps the k3 KL estimator at :data:`KL_LOGRATIO_CLAMP` in all three GRPO
trainers, and :func:`compute_is_ratio` clamps the environmental trainer's sampling-to-trainer importance
ratio at ``vllm_importance_sampling_clip_max``. Everything else here serves that ratio: its exemptions
(sampler-certain tokens, engine-forced reasoning closes), the engine re-score's log-ratio and the mask
stages.

:func:`apply_is_masks` and :func:`apply_opsm` add masking-over-reweighting stages for MoE-scale
mismatch (trajectory geometric-mean band, catastrophic-token veto, OPSM), all default off.
A masked token/trajectory gets ratio 0 (policy-gradient term vanishes, DAPO normalizer unchanged); the
β·k3 KL term is added after the ratio multiply, so masked tokens stay anchored to the reference.

Trajectory aggregation pools all of an episode's turn rows via ``traj_ids``, since the drift compounds
over the episode. Pure functions: no trainer state.
"""

from __future__ import annotations

import torch

from src.configs.async_training_config import ISMaskConfig

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
    """The completion tokens of rows that carry sampling logprobs: the only tokens a sampler-side
    quantity (a forced close, a certain token, the IS ratio) can be read on."""
    return completion_mask.bool() & row_has_sampling.unsqueeze(1)


def sampler_certain_mask(
    sampling_logps: torch.Tensor, completion_mask: torch.Tensor, row_has_sampling: torch.Tensor
) -> torch.Tensor:
    """Policy tokens the sampler emitted with probability 1 (:data:`SAMPLER_CERTAIN_LOGPROB`), on rows that
    carry sampling logprobs: no sampling choice was made there, so they carry no importance weight."""
    return sampled_token_mask(completion_mask, row_has_sampling) & (sampling_logps >= SAMPLER_CERTAIN_LOGPROB)


def zero_engine_forced_closes(
    ratio: torch.Tensor,
    sampling_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    row_has_sampling: torch.Tensor,
    completion_ids: torch.Tensor,
    reasoning_end_ids: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero the ratio at reasoning closes the engine forced: a run of the reasoning-end ids every token of
    which the sampler emitted at probability 1, which is what vLLM's thinking budget appends when a turn
    reaches its cap (one token for ``</think>``, five for gpt-oss's final-channel opener). Returns
    ``(ratio, forced)``.

    The close was not the policy's action; trained with the episode's advantage it moves the model's own
    probability of ending its reasoning, which repeated forced closes can drive down until the model stops
    closing at all. Ratio 0 drops the policy-gradient term and keeps the DAPO normalizer, like every mask
    stage here. A naturally certain token outside such a run (a collapsed nucleus) is left alone; a natural
    run emitted wholly at probability 1 is zeroed too, which costs nothing without nucleus truncation (the
    budgeted recipes sample at top_p 1) and otherwise drops the little gradient its near-certain tokens carry."""
    certain = sampler_certain_mask(sampling_logps, completion_mask, row_has_sampling)
    width = len(reasoning_end_ids)
    forced = torch.zeros_like(certain)
    if completion_ids.shape[1] >= width:
        end = completion_ids.new_tensor(reasoning_end_ids)
        starts = ((completion_ids.unfold(1, width, 1) == end) & certain.unfold(1, width, 1)).all(-1)
        for offset in range(width):
            forced[:, offset : offset + starts.shape[1]] |= starts
    return ratio.masked_fill(forced, 0.0), forced


def clamp_ref_logps(ref_logps: torch.Tensor, policy_logps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Bound the tail of the k3 KL estimator by capping the log-ratio at :data:`KL_LOGRATIO_CLAMP`
    nats. Unconditional; not configurable.

    ``per_token_kl = exp(ref − logp) − (ref − logp) − 1`` is unbounded where the policy suppresses a
    token the reference gives high probability; capping ``ref`` at ``policy + KL_LOGRATIO_CLAMP``
    truncates that tail. Returns ``(clamped_ref, clamped)``, the second the bool mask of the positions
    the cap bit.
    """
    ceiling = policy_logps + KL_LOGRATIO_CLAMP
    return torch.minimum(ref_logps, ceiling), ref_logps > ceiling


def kl_clamp_counts(clamped: torch.Tensor, loss_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(clamped loss tokens, loss tokens)``, the pair :data:`KL_CLAMP_FRAC_KEY` is the world ratio of.

    Over the loss mask the KL term is averaged on (TRL's ``kl``: the completion mask times the tool mask,
    after every drop), so a cap that bit on padding, on a tool-output token or on a dropped row is not
    counted, nor is any of them in the denominator.
    """
    loss_mask = loss_mask.bool()
    return (clamped & loss_mask).sum(), loss_mask.sum()


def compute_is_ratio(
    recompute_logps: torch.Tensor,
    sampling_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    row_has_sampling: torch.Tensor,
    clip_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Truncated per-token engine→trainer importance ratio ``clamp(exp(logπ_recompute − logπ_sampling), clip_max)``.

    Rows flagged ``False`` in ``row_has_sampling`` (rollout error dropped their sampling logprobs) and
    non-policy tokens get ratio exactly 1, so one bad row cannot perturb the others. So does a token
    the sampler reports as certain (:data:`SAMPLER_CERTAIN_LOGPROB`): no sampling choice was made
    there, so it carries no importance weight and cannot trip a band or the veto.
    Returns ``(ratio, logps_diff, corrected_mask)``, all shaped like ``completion_mask``.
    """
    corrected_mask = sampled_token_mask(completion_mask, row_has_sampling) & ~sampler_certain_mask(
        sampling_logps, completion_mask, row_has_sampling
    )
    logps_diff = (recompute_logps - sampling_logps) * corrected_mask
    return torch.clamp(torch.exp(logps_diff), max=clip_max), logps_diff, corrected_mask


def select_mask_logratio(
    logps_diff: torch.Tensor,
    recompute_logps: torch.Tensor,
    sampling_logps: torch.Tensor,
    engine_logps: torch.Tensor,
    corrected_mask: torch.Tensor,
    row_has_engine: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """The log-ratio the mask stages read when the engine re-scored the rows under the trainer's
    current weights: ``logπ_engine_now − logπ_sampling`` on rows that carry a re-score — pure policy
    staleness, since the two engine passes share their numerics — and the trainer diff elsewhere.
    Returns it with the step's diagnostics as ``(numerator, denominator)`` pairs over the re-scored
    tokens: the staleness mean, the numerics mean ``logπ_recompute − logπ_engine_now`` (the floor the
    bands would otherwise read) and the coverage of the re-score over the corrected tokens. The pairs
    stay on device, for the caller to read with the step's other counts.
    """
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
    """Trajectories ``traj_ids`` spans (ids ``0..n-1``, −1 for dummy rows). A host read: it sizes the
    per-trajectory reductions."""
    return int(traj_ids.max()) + 1 if traj_ids.numel() else 0


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
) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, int]]]:
    """Apply the geometric-band / veto stages to the truncated ratio.

    ``traj_ids`` maps each row to its trajectory (−1 for dummy rows). The veto tests the raw
    (pre-truncation) ratio ``exp(logps_diff)``, so the clip cannot hide a catastrophic token.
    Returns the masked ratio and the diagnostic ``(masked, total)`` counts of each active stage, the
    masked count left on device.
    """
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
        # uncorrected tokens read as 1.0 so they never trip the veto.
        row_min = torch.where(corrected_mask, raw_ratio, torch.ones_like(raw_ratio)).min(dim=1).values
        traj_min = _traj_scatter(row_min, traj_ids, num_trajs, "amin")
        # An id with no rows keeps the 0 scatter init, which reads as vetoed; restrict to contributing ids.
        present = _traj_scatter(torch.ones_like(row_min), traj_ids, num_trajs, "sum") > 0
        vetoed = (traj_min < config.veto_min) & present
        stats["sampling/is_veto_masked_frac"] = (vetoed.sum(), num_trajs)
        traj_keep &= ~vetoed
    row_keep = torch.where(traj_ids >= 0, traj_keep.gather(0, traj_ids.clamp(min=0)), torch.ones_like(traj_ids).bool())
    return torch.where(row_keep.unsqueeze(1), ratio, torch.zeros_like(ratio)), stats


def apply_opsm(
    ratio: torch.Tensor,
    logps_diff: torch.Tensor,
    corrected_mask: torch.Tensor,
    traj_ids: torch.Tensor,
    row_advantages: torch.Tensor,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Off-Policy Sequence Masking (DeepSeek-V3.2): zero the ratio of negative-advantage trajectories
    whose mean log-ratio magnitude exceeds ``delta`` nats. Positive-advantage trajectories are never
    masked. Returns the masked ratio and the per-trajectory bool mask of what it masked.
    """
    num_trajs = _num_trajs(traj_ids)
    if num_trajs == 0:
        return ratio, torch.zeros(0, dtype=torch.bool, device=ratio.device)
    traj_mean = _traj_mean_logratio(logps_diff, corrected_mask, traj_ids, num_trajs)
    traj_negative = _traj_scatter(row_advantages, traj_ids, num_trajs, "amin") < 0
    masked = traj_negative & (traj_mean.abs() > delta)
    row_masked = torch.where(traj_ids >= 0, masked.gather(0, traj_ids.clamp(min=0)), torch.zeros_like(traj_ids).bool())
    out = torch.where(row_masked.unsqueeze(1), torch.zeros_like(ratio), ratio)
    return out, masked
