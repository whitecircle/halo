"""Environmental GRPO's reasoning budget: the per-level thinking contract a run draws, the reasoning end marker
it resolves, and the length terms it charges (a per-level price and an under-use floor)."""

import torch
from accelerate.logging import get_logger

from src.environments.engine_wire import reject_unenforceable_thinking_budget
from src.environments.episode import RolloutResult
from src.environments.messages import Trajectory
from src.environments.reasoning_effort import EpisodeEffort, efforts_by_level
from src.environments.rollout_config import SGLANG_BACKEND, resolve_reasoning_end_ids
from src.models.config_access import config_model_types
from src.models.unwrapping import unwrap_framework_wrappers

logger = get_logger(__name__, log_level="INFO")

# The share of the per-turn thinking budget the floor asks an episode for, summed over its turns; under 1, so
# one turn can clear it without hitting an enforced budget.
REASONING_TARGET_SHARE = 0.75


def reasoning_token_counts(tokenizer, trajectory: Trajectory | None) -> list[int]:
    """Each assistant turn's reasoning token count under ``tokenizer``, 0 for a thinking-free turn; none for an
    episode without a trajectory."""
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
    """The capped reasoning price in ``[-cap, 0]``: ``-min(cap, price_per_1k * priced / 1000)``, ``priced`` being
    every turn's reasoning tokens, or with a per-turn ``budget`` what each turn reasoned past it. The cap keeps a
    long trace from outweighing the task reward."""
    priced = sum(reasoning_tokens) if budget is None else sum(max(0, tokens - budget) for tokens in reasoning_tokens)
    if priced <= 0:
        return 0.0
    return -min(cap, price_per_1k * priced / 1000)


def reasoning_floor_term(reasoning_tokens: list[int], budget: int, weight: float) -> float:
    """Under-use floor in ``[-weight, 0]``: ``-weight * shortfall / target`` while the episode's reasoning falls
    short of :data:`REASONING_TARGET_SHARE` of its per-turn thinking ``budget``. Summed over the episode, so a
    short repair turn is not under-use; an episode with no assistant turn pays nothing."""
    target = round(budget * REASONING_TARGET_SHARE)
    shortfall = target - sum(reasoning_tokens)
    if not reasoning_tokens or target <= 0 or weight <= 0 or shortfall <= 0:
        return 0.0
    return -weight * shortfall / target


class ReasoningBudgetMixin:
    """The per-level reasoning contract a run draws, its reasoning end marker and the length terms."""

    @property
    def _any_episode_budgeted(self) -> bool:
        """Whether some level the run draws carries a per-turn thinking budget."""
        return any(effort.thinking_budget is not None for effort in self._efforts_by_level().values())

    def _efforts_by_level(self) -> dict[str | None, EpisodeEffort]:
        """The contract every level this run can draw binds (:func:`efforts_by_level`, which refuses one the
        turn cannot hold)."""
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
        """Whether every level the run draws runs under a soft budget, one a turn can reason past."""
        return all(effort.budget_soft for effort in self._efforts_by_level().values())

    def _resolve_reasoning_end_ids(self) -> tuple[int, ...] | None:
        """The ids the server's reasoning parser ends reasoning with on a budgeted vLLM run, else ``None``; a
        marker the tokenizer does not write resolves none, with a warning."""
        cfg = self.async_config
        if cfg.rollout_backend == SGLANG_BACKEND or not self._any_episode_budgeted:
            return None
        model_config = getattr(unwrap_framework_wrappers(self.model), "config", None)
        reject_unenforceable_thinking_budget(config_model_types(model_config), cfg.rollout_backend)
        try:
            return resolve_reasoning_end_ids(self._tokenizer, cfg.rollout_reasoning_end_token)
        except ValueError as e:
            lost = "A turn's reasoning goes uncounted"
            if self._any_episode_capped:
                lost += ", and the closes the engine forces at a thinking cap stay in the policy loss"
            logger.warning("%s %s.", e, lost)
            return None

    @property
    def _any_episode_capped(self) -> bool:
        """Whether some level the run draws binds a reasoning cap the engine closes at."""
        return any(effort.thinking_cap is not None for effort in self._efforts_by_level().values())

    def _resolve_forced_close_ids(self, reasoning_end_ids: tuple[int, ...] | None) -> tuple[int, ...] | None:
        """``reasoning_end_ids`` where a reasoning cap can force them, closes the loss must not train on."""
        return reasoning_end_ids if self._any_episode_capped else None

    def _resolve_reasoning_end_token_id(self) -> int | None:
        """The one-token reasoning end id the rollout counts a turn's reasoning up to, ``None`` with no marker
        or a multi-token one."""
        ids = self._reasoning_end_ids
        return ids[0] if ids is not None and len(ids) == 1 else None

    def _build_rollout_rewards(
        self, rollout_results: list[RolloutResult], reasoning_tokens: list[list[int]], device: torch.device
    ) -> torch.Tensor:
        """Per-trajectory environment rewards, each episode charged its level's reasoning price and its under-use
        floor while that term is on, each term's batch mean recorded. An episode with no level pays no price, one
        with no budget no floor."""
        rewards = torch.tensor([r.total_reward for r in rollout_results], device=device, dtype=torch.float32)
        cfg = self.async_config
        price_on, floor_on = cfg.reasoning_price is not None, cfg.reasoning_floor > 0
        if price_on or floor_on or self._carry_reasoning:
            self._warn_if_no_reasoning_captured(rollout_results, price_on=price_on, floor_on=floor_on)
        if price_on or floor_on:
            prices, floors = [], []
            for i, (r, tokens) in enumerate(zip(rollout_results, reasoning_tokens, strict=True)):
                traj = r.trajectory
                price = floor = 0.0
                if price_on and traj and traj.reasoning_effort in cfg.reasoning_price:
                    level_price = cfg.reasoning_price[traj.reasoning_effort]
                    if not cfg.reasoning_price_past_budget:
                        price = reasoning_price_term(tokens, level_price, cfg.reasoning_price_cap)
                    elif traj.reasoning_budget and traj.reasoning_budget_soft:
                        price = reasoning_price_term(
                            tokens, level_price, cfg.reasoning_price_cap, traj.reasoning_budget
                        )
                if floor_on and traj and traj.reasoning_budget:
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
