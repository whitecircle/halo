"""Environmental GRPO's reasoning budget: the per-level thinking contract, the reasoning end marker and the length
terms (a per-level price and an under-use floor), waived in a training group where no episode earned the objective."""

import logging
from collections.abc import Sequence

import torch

from halo.environments.engine_wire import reject_unenforceable_thinking_budget
from halo.environments.episode import RolloutResult
from halo.environments.episode_keys import REWARD_COMPONENTS_KEY
from halo.environments.messages import Trajectory
from halo.environments.reasoning_effort import EpisodeEffort, efforts_by_level, has_thinking_budget
from halo.environments.rollout_config import SGLANG_BACKEND, find_reasoning_end_ids
from halo.models.config_access import config_model_types
from halo.models.unwrapping import unwrap_framework_wrappers
from halo.rewards.terms import OBJECTIVE_REWARD_KEY
from halo.runtime.ranks import is_global_main_process
from halo.weight_sync.registry import resolve_weight_sync_client

logger = logging.getLogger(__name__)

# The share of the per-turn thinking budget the floor asks an episode for, summed over its turns; under 1, so
# one turn can clear it without hitting an enforced budget.
REASONING_TARGET_SHARE = 0.75


def reasoning_token_counts(tokenizer, trajectory: Trajectory | None) -> list[int]:
    if trajectory is None:
        return []
    return [
        len(tokenizer(m.thinking, add_special_tokens=False)["input_ids"]) if m.thinking else 0
        for m in trajectory.messages
        if m.role == "assistant"
    ]


def reasoning_price_term(
    reasoning_tokens: list[int], price_per_1k: float, cap: float, budget: int | None = None
) -> float:
    """``-min(cap, price_per_1k * priced / 1000)``, ``priced`` being every turn's reasoning tokens or, with a per-turn
    ``budget``, what each turn reasoned past it. The cap keeps a long trace from outweighing the task reward."""
    priced = sum(reasoning_tokens) if budget is None else sum(max(0, tokens - budget) for tokens in reasoning_tokens)
    if priced <= 0:
        return 0.0
    return -min(cap, price_per_1k * priced / 1000)


def reasoning_floor_term(reasoning_tokens: list[int], budget: int, weight: float) -> float:
    """``-weight * shortfall / target`` below :data:`REASONING_TARGET_SHARE` of ``budget``. Summed over the episode, so
    a short repair turn is not under-use."""
    target = round(budget * REASONING_TARGET_SHARE)
    shortfall = target - sum(reasoning_tokens)
    if not reasoning_tokens or target <= 0 or weight <= 0 or shortfall <= 0:
        return 0.0
    return -weight * shortfall / target


def failed_group_mask(rollout_results: Sequence[RolloutResult], group_size: int) -> list[bool]:
    """Per episode, whether no baseline episode of its group earned the objective. There the length terms are most of
    what tells attempts apart, so the advantage would train reasoning length rather than the task."""
    earned = [
        r.counts_toward_baseline
        and r.trajectory is not None
        and r.trajectory.info.get(REWARD_COMPONENTS_KEY, {}).get(OBJECTIVE_REWARD_KEY, 0.0) > 0
        for r in rollout_results
    ]
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

    @property
    def _every_budget_soft(self) -> bool:
        """A soft budget is one a turn can reason past."""
        return all(effort.budget_soft for effort in self._efforts_by_level().values())

    def _resolve_reasoning_end_ids(self) -> tuple[int, ...] | None:
        """On a budgeted vLLM run; a marker the tokenizer does not write resolves none, warned on the main process."""
        cfg = self.async_config
        if cfg.rollout_backend == SGLANG_BACKEND or not has_thinking_budget(self._efforts_by_level()):
            return None
        model_config = getattr(unwrap_framework_wrappers(self.model), "config", None)
        reject_unenforceable_thinking_budget(
            config_model_types(model_config),
            resolve_weight_sync_client(cfg.rollout_backend).UNENFORCEABLE_THINKING_BUDGET_MODEL_TYPES,
        )
        also_lost = (
            ", and the closes the engine forces at a thinking cap stay in the policy loss"
            if self._any_episode_capped
            else ""
        )
        return find_reasoning_end_ids(
            self._tokenizer,
            cfg.rollout_reasoning_end_token,
            also_lost=also_lost,
            logger=logger if is_global_main_process() else None,
        )

    @property
    def _any_episode_capped(self) -> bool:
        return any(effort.thinking_cap is not None for effort in self._efforts_by_level().values())

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
        """Each episode charged its level's price and under-use floor, except: no level pays no price, no budget no
        floor, and a training episode whose group earned no objective neither, unless
        ``reasoning_terms_in_failed_groups``."""
        rewards = torch.tensor([r.total_reward for r in rollout_results], device=device, dtype=torch.float32)
        cfg = self.async_config
        price_on, floor_on = cfg.reasoning_price is not None, cfg.reasoning_floor > 0
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
                charged = traj is not None and not waived[i]
                price = floor = 0.0
                if price_on and charged and traj.reasoning_effort in cfg.reasoning_price:
                    level_price = cfg.reasoning_price[traj.reasoning_effort]
                    if not cfg.reasoning_price_past_budget:
                        price = reasoning_price_term(tokens, level_price, cfg.reasoning_price_cap)
                    elif traj.reasoning_budget and traj.reasoning_budget_soft:
                        price = reasoning_price_term(
                            tokens, level_price, cfg.reasoning_price_cap, traj.reasoning_budget
                        )
                if floor_on and charged and traj.reasoning_budget:
                    floor = reasoning_floor_term(tokens, traj.reasoning_budget, cfg.reasoning_floor)
                rewards[i] += price + floor
                prices.append(price)
                floors.append(floor)
            # A config gate is rank-uniform, so every rank records the same keys.
            if price_on:
                self._world_metrics.fraction("reward/reasoning_price", sum(prices), len(prices))
            if floor_on:
                self._world_metrics.fraction("reward/reasoning_floor", sum(floors), len(floors))
        return rewards
