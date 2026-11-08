"""Environmental GRPO's reasoning terms: the run's reasoning cap, the reasoning end marker and the length terms,
each against the episode's level budget (its effort profile's ``reasoning_budget``, summed over turns): a per-level
price on what the episode reasoned past it and an under-use floor on a failed episode's shortfall under it, waived in
a training group where no episode earned the objective."""

import logging
from collections.abc import Sequence
from functools import cached_property

import torch

from halo.environments.episodes.episode_keys import REWARD_COMPONENTS_KEY
from halo.environments.episodes.messages import Trajectory
from halo.environments.episodes.reasoning_effort import EpisodeEffort, efforts_by_level, has_thinking_cap
from halo.environments.rollout_config import SGLANG_BACKEND, find_reasoning_end_ids
from halo.environments.rollouts.engine_wire import reject_unenforceable_thinking_budget
from halo.environments.rollouts.episode_loop import RolloutResult
from halo.models.introspection.config_access import config_model_types
from halo.models.introspection.unwrapping import unwrap_framework_wrappers
from halo.rewards.terms import OBJECTIVE_REWARD_KEY
from halo.runtime.ranks import is_global_main_process
from halo.weight_sync.registry import resolve_weight_sync_client

logger = logging.getLogger(__name__)


def reasoning_token_counts(tokenizer, trajectory: Trajectory | None) -> list[int]:
    if trajectory is None:
        return []
    return [
        len(tokenizer(m.thinking, add_special_tokens=False)["input_ids"]) if m.thinking else 0
        for m in trajectory.messages
        if m.role == "assistant"
    ]


def reasoning_price_term(reasoning_tokens: list[int], price_per_1k: float, cap: float, budget: int) -> float:
    """``-min(cap, price_per_1k * past / 1000)``, ``past`` being what the episode's turns reasoned together beyond
    ``budget`` (0 prices every reasoning token). Within the budget reasoning is free, so a short correct solve pays
    nothing; the cap keeps a long trace from outweighing the task reward."""
    past = sum(reasoning_tokens) - budget
    if past <= 0:
        return 0.0
    return -min(cap, price_per_1k * past / 1000)


def reasoning_floor_term(reasoning_tokens: list[int], budget: int, weight: float) -> float:
    """``-weight * shortfall / budget`` while the episode's reasoning, summed over its turns, falls short of
    ``budget``. The caller charges it only to a failed episode, so a short solve pays nothing. An episode with no
    assistant turn is a lost one and pays nothing."""
    shortfall = budget - sum(reasoning_tokens)
    if not reasoning_tokens or budget <= 0 or weight <= 0 or shortfall <= 0:
        return 0.0
    return -weight * shortfall / budget


def earned_objective(result: RolloutResult) -> bool:
    """Whether a baseline episode earned the objective: a positive ``reward/objective`` component, which a veto
    zeroes. An episode outside the baseline earns nothing: it does not train, and a sibling's advantage does not read
    it."""
    return (
        result.counts_toward_baseline
        and result.trajectory is not None
        and result.trajectory.info.get(REWARD_COMPONENTS_KEY, {}).get(OBJECTIVE_REWARD_KEY, 0.0) > 0
    )


def failed_group_mask(rollout_results: Sequence[RolloutResult], group_size: int) -> list[bool]:
    """Per episode, whether no baseline episode of its group earned the objective. There the length terms are most of
    what tells attempts apart, so the advantage would train reasoning length rather than the task."""
    earned = [earned_objective(r) for r in rollout_results]
    failed = []
    for start in range(0, len(earned), group_size):
        group = earned[start : start + group_size]
        failed.extend([not any(group)] * len(group))
    return failed


class ReasoningBudgetMixin:
    def _efforts_by_level(self) -> dict[str | None, EpisodeEffort]:
        cfg = self.async_config
        return efforts_by_level(
            self._rollout_env,
            max_tokens=cfg.rollout_max_tokens,
            max_thinking_tokens=cfg.rollout_max_thinking_tokens,
            max_answer_tokens=cfg.rollout_max_answer_tokens,
            max_episode_tokens=cfg.rollout_max_episode_tokens,
        )

    @cached_property
    def _level_reasoning_terms(self) -> dict[str | None, tuple[int | None, float]]:
        """Per level this run can draw, its reasoning budget and price (:meth:`reasoning_terms_for_effort`)."""
        return {level: self._rollout_env.reasoning_terms_for_effort(level) for level in self._efforts_by_level()}

    @property
    def _reasoning_price_on(self) -> bool:
        """Whether some level this run draws prices reasoning past its budget."""
        return any(price > 0 for _, price in self._level_reasoning_terms.values())

    @property
    def _any_episode_capped(self) -> bool:
        return has_thinking_cap(self._efforts_by_level())

    def _resolve_reasoning_end_ids(self) -> tuple[int, ...] | None:
        """The reasoning-end ids on a capped vLLM run, else ``None``; a marker the tokenizer does not write resolves
        none, warned on the main process."""
        cfg = self.async_config
        if cfg.rollout_backend == SGLANG_BACKEND or not self._any_episode_capped:
            return None
        model_config = getattr(unwrap_framework_wrappers(self.model), "config", None)
        reject_unenforceable_thinking_budget(
            config_model_types(model_config),
            resolve_weight_sync_client(cfg.rollout_backend).UNENFORCEABLE_THINKING_BUDGET_MODEL_TYPES,
        )
        return find_reasoning_end_ids(
            self._tokenizer, cfg.rollout_reasoning_end_token, logger=logger if is_global_main_process() else None
        )

    def _resolve_forced_close_ids(self, reasoning_end_ids: tuple[int, ...] | None) -> tuple[int, ...] | None:
        """``reasoning_end_ids`` where a reasoning cap can force them, closes the loss must not train on."""
        return reasoning_end_ids if self._any_episode_capped else None

    def _build_rollout_rewards(
        self,
        rollout_results: list[RolloutResult],
        reasoning_tokens: list[list[int]],
        device: torch.device,
        mode: str,
    ) -> torch.Tensor:
        """Each episode charged its level's price and under-use floor, in its level's ``reasoning_budget``: a level
        with no budget pays neither, the floor only a baseline episode that earned no objective (a solve pays nothing),
        and a training episode whose group earned no objective neither, unless ``reasoning_terms_in_failed_groups``."""
        rewards = torch.tensor([r.total_reward for r in rollout_results], device=device, dtype=torch.float32)
        cfg = self.async_config
        price_on, floor_on = self._reasoning_price_on, cfg.reasoning_floor > 0
        if price_on or floor_on or self._carry_reasoning:
            self._warn_if_no_reasoning_captured(rollout_results, price_on=price_on, floor_on=floor_on)
        if price_on or floor_on:
            waived = [False] * len(rollout_results)
            # Mode- and config-derived, so every rank records the waived share or none does.
            if mode == "train" and not cfg.reasoning_terms_in_failed_groups:
                waived = failed_group_mask(rollout_results, self.num_generations)
                self._world_metrics.fraction("outcome/reasoning_terms_waived_frac", sum(waived), len(waived))
            prices, floors = [], []
            for i, (r, tokens) in enumerate(zip(rollout_results, reasoning_tokens, strict=True)):
                traj = r.trajectory
                price = floor = 0.0
                if traj is not None and not waived[i]:
                    budget, rate = self._rollout_env.reasoning_terms_for_effort(traj.reasoning_effort)
                    if budget is not None and rate > 0:
                        price = reasoning_price_term(tokens, rate, cfg.reasoning_price_cap, budget)
                    if budget is not None and floor_on and r.counts_toward_baseline and not earned_objective(r):
                        floor = reasoning_floor_term(tokens, budget, cfg.reasoning_floor)
                rewards[i] += price + floor
                prices.append(price)
                floors.append(floor)
            # A config gate is rank-uniform, so every rank records the same keys.
            if price_on:
                self._world_metrics.fraction("reward/reasoning_price", sum(prices), len(prices))
            if floor_on:
                self._world_metrics.fraction("reward/reasoning_floor", sum(floors), len(floors))
        return rewards
