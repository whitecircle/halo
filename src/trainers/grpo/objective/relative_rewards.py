"""Relative-reward shaping for group-based RL (RLRR).

Implements Niu et al., arXiv:2601.23058: replaces absolute group-mean/std advantage with relative
intra-group rankings, fixing GRPO signal sparsity (all-correct/all-wrong groups) and scale
instability (drifting RM scores).

Two shaping modes: ``hrr`` (Hybrid, Eq. 3) adds a bounded rank correction to the binary rule reward;
``prr`` (Pure, Eq. 4) maps rank linearly to ``[0, 1]``, discarding the absolute score. Ranks come
from hierarchical re-ranking (Eq. 6) on ``(correctness, length-bin, raw-reward)``; advantages use
the group-relative estimator (Eq. 1) then correctness-aware clipping (Eq. 5). Framework-agnostic
numpy, operating on a single group of ``G`` responses. Hyperparameters live in
:class:`~src.args.mixins.RLRRConfig`, which the RLVR script args build.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from src.args.mixins import RLRRConfig

ArrayLike = Sequence[float] | np.ndarray
BoolArrayLike = Sequence[bool] | np.ndarray

# Guards the ``std_normalize`` divisor against an all-equal shaped group; not a tuning knob.
_STD_EPS = 1e-6


def dense_ranks(keys: Sequence[tuple]) -> np.ndarray:
    """Dense-rank sortable keys, ``1`` = best (smallest key); equal keys share a rank."""
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    ranks = np.empty(len(keys), dtype=np.int64)
    current = 0
    prev: tuple | None = None
    for i in order:
        if prev is None or keys[i] != prev:
            current += 1
            prev = keys[i]
        ranks[i] = current
    return ranks


def hierarchical_ranks(
    rewards: ArrayLike,
    correct: BoolArrayLike,
    lengths: ArrayLike | None,
    config: RLRRConfig,
) -> np.ndarray:
    """Intra-group ranks via hierarchical re-ranking (Eq. 6).

    Lexicographic sort key (ascending, smallest = best): correctness, then length bin
    ``floor(len / λ)`` (``+inf`` for incorrect), then raw reward.
    """
    n = len(rewards)
    keys: list[tuple] = []
    for i in range(n):
        is_correct = bool(correct[i])
        correctness_key = 0 if is_correct else 1
        if config.length_rerank and lengths is not None:
            length_bin = math.floor(lengths[i] / config.lam) if is_correct else math.inf
        else:
            length_bin = 0.0
        # Negate reward so higher reward -> smaller key -> better rank.
        keys.append((correctness_key, length_bin, -float(rewards[i])))
    return dense_ranks(keys)


def shape_rewards(ranks: np.ndarray, rule_scores: np.ndarray, config: RLRRConfig) -> np.ndarray:
    """Map ranks to shaped scalar rewards (Eq. 3 for HRR, Eq. 4 for PRR)."""
    r = ranks.astype(np.float64)
    r_max = float(ranks.max())
    if config.mode == "hrr":
        correction = config.tau * np.tanh(r_max / r - 1.0)
        return rule_scores.astype(np.float64) + correction
    # Eq. 4 (PRR): s_i = (r_max - r_i) / (r_max - 1); degenerate single-rank -> 0.
    if r_max <= 1.0:
        return np.zeros_like(r)
    return (r_max - r) / (r_max - 1.0)


def relative_advantages(
    rewards: ArrayLike,
    config: RLRRConfig,
    *,
    lengths: ArrayLike | None = None,
) -> np.ndarray:
    """Compute RLRR advantages for a single group of responses.

    Correctness is the ``rewards >= config.correctness_threshold`` label, which is also HRR's
    binary rule score ``s_rule`` (Eq. 3): the reward this toolkit scores is the rule reward.

    Args:
        rewards: Raw scalar reward / RM score per response (group size ``G``).
        config: :class:`RLRRConfig` hyperparameters.
        lengths: Optional response token lengths for length re-ranking (Eq. 6).

    Returns:
        ``np.ndarray`` of shape ``(G,)`` with the per-response advantage.
    """
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    n = len(rewards_arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    correct_arr = rewards_arr >= config.correctness_threshold
    rule_scores_arr = correct_arr.astype(np.float64)

    ranks = hierarchical_ranks(rewards_arr, correct_arr, lengths, config)
    shaped = shape_rewards(ranks, rule_scores_arr, config)

    advantages = shaped - shaped.mean()
    if config.std_normalize:
        advantages = advantages / (shaped.std() + _STD_EPS)

    if config.correctness_clip:
        # Eq. 5: floor correct advantages at ξ⁻, cap incorrect ones at ξ⁺.
        advantages = np.where(
            correct_arr,
            np.maximum(advantages, config.xi_neg),
            np.minimum(advantages, config.xi_pos),
        )
    return advantages


def relative_advantages_grouped(
    rewards: ArrayLike,
    group_size: int,
    config: RLRRConfig,
    *,
    lengths: ArrayLike | None = None,
) -> np.ndarray:
    """Apply :func:`relative_advantages` to a flat batch of contiguous groups.

    ``rewards`` is laid out as ``num_prompts`` consecutive blocks of
    ``group_size`` responses (the standard GRPO rollout layout). Returns a flat
    advantage array of the same length.
    """
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    total = len(rewards_arr)
    if group_size <= 0 or total % group_size != 0:
        raise ValueError(f"rewards length {total} is not divisible by group_size {group_size}")

    out = np.empty(total, dtype=np.float64)
    for start in range(0, total, group_size):
        end = start + group_size
        out[start:end] = relative_advantages(
            rewards_arr[start:end],
            config,
            lengths=None if lengths is None else lengths[start:end],
        )
    return out
