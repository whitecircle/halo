"""Rollout diagnostics for environmental GRPO: completion logs and per-episode metrics, every number over the
world's episode population."""

import json
import logging
from collections.abc import Mapping, Sequence

import torch

from halo.environments.episodes.episode_keys import (
    EPISODE_SLICES_KEY,
    REWARD_COMPONENTS_KEY,
    REWARD_DETAILS_KEY,
    SOLVE_RATE_KEY,
)
from halo.environments.episodes.messages import Trajectory
from halo.environments.rollouts.episode_loop import RolloutResult
from halo.rewards.samples import CUT_CALL_NOTE
from halo.rewards.terms import REWARD_COMPONENT_PREFIX
from halo.runtime.ranks import is_global_main_process
from halo.trainers.grpo.rollout.completions_logging import emit_completion_artifacts
from halo.trainers.metrics.world_metrics import WorldMetrics

logger = logging.getLogger(__name__)


def _episode_info_json(trajectory: Trajectory | None, key: str) -> str:
    """As sorted-key JSON; ``{}`` for an episode that kept none (a lost rollout, an unscored term)."""
    value = trajectory.info.get(key, {}) if trajectory is not None else {}
    return json.dumps(value, sort_keys=True, default=str)


def group_solve_counts(solved: Sequence[bool | None], group_size: int) -> tuple[int, int, int, int]:
    """Each group is judged on the episodes that carry a verdict; a group with none is not counted."""
    groups = all_pass = all_fail = any_pass = 0
    for start in range(0, len(solved), group_size):
        verdicts = [v for v in solved[start : start + group_size] if v is not None]
        if verdicts:
            groups += 1
            all_pass += all(verdicts)
            all_fail += not any(verdicts)
            any_pass += any(verdicts)
    return groups, all_pass, all_fail, any_pass


class RolloutMetricsMixin:
    # Each rank keeps and writes its own rows: the rendered trajectories are the heaviest payload this trainer
    # holds, and gathering them would stage the world's text through a device.
    RANK_LOCAL_COMPLETION_LOGS = True
    # Cumulative totals over the world's population, never this rank's shard; ``+=`` rebinds per instance.
    _total_rollouts = 0
    _total_rollout_latency = 0.0
    _total_generation_tokens = 0
    _completion_logs_mode: str | None = None
    _truncation_alarmed: frozenset[str] = frozenset()
    # Train rounds in a row over episode_error_rate_limit; resumed from the checkpoint's sidecar.
    _episode_error_rounds = 0

    def _populate_completion_logs(
        self,
        rollout_results: list[RolloutResult],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        mode: str,
    ) -> None:
        """A TP/ETP follower, which holds its leader's rollouts, keeps no rows. Rows of the other mode still waiting
        are written under their own mode first (COLLECTIVE then, a verdict every rank shares)."""
        if not (self._save_completions or self.log_completions):
            return
        if self._completion_logs_mode not in (None, mode):
            emit_completion_artifacts(
                self, console=False, save=self._save_completions, mode=self._completion_logs_mode
            )
        self._completion_logs_mode = mode
        if self._is_tp_follower():
            return
        self._logs["prompt"].extend(r.prompt for r in rollout_results)
        self._logs["completion"].extend(self._render_trajectory_for_log(r.trajectory) for r in rollout_results)
        self._logs["rewards"]["environment_reward"].extend(rewards.tolist())
        self._logs["advantages"].extend(advantages.tolist())
        self._logs["extra"][REWARD_COMPONENTS_KEY].extend(
            _episode_info_json(r.trajectory, REWARD_COMPONENTS_KEY) for r in rollout_results
        )
        self._logs["extra"][REWARD_DETAILS_KEY].extend(
            _episode_info_json(r.trajectory, REWARD_DETAILS_KEY) for r in rollout_results
        )

    @staticmethod
    def _render_trajectory_for_log(trajectory: "Trajectory | None") -> str:
        """A cut turn's unrun calls are marked :data:`~halo.rewards.samples.CUT_CALL_NOTE`."""
        if trajectory is None or not trajectory.messages:
            return "(empty trajectory)"
        parts = []
        for m in trajectory.messages:
            if m.role == "system":
                continue
            seg = f"[{m.role}] {m.content or ''}".rstrip()
            if m.thinking:
                seg += f"\n  <reasoning> {m.thinking}"
            calls = [(tc, "") for tc in m.tool_calls or []] + [
                (tc, f" ({CUT_CALL_NOTE})") for tc in m.cut_tool_calls or []
            ]
            for tc, note in calls:
                fn = tc.get("function", tc)
                seg += f"\n  <tool_call {fn.get('name', '?')}{note}> {fn.get('arguments', '')}"
            parts.append(seg)
        return "\n".join(parts)

    def _log_rollout_metrics(self, results: list[RolloutResult], mode: str, reasoning_tokens: list[list[int]]):
        """COLLECTIVE. World values over every rank's ``results`` in one flush; ``reasoning_tokens`` is each episode's
        per-turn count, taken once for the step."""
        metrics = WorldMetrics()
        for result, tokens in zip(results, reasoning_tokens, strict=True):
            self._record_episode(metrics, result, sum(tokens))
        world = metrics.flush()
        if not world:
            return
        self._total_rollouts += int(world.pop("async/total_rollouts"))
        self._total_rollout_latency += world.pop("async/total_rollout_latency")
        self._total_generation_tokens += int(world.pop("async/total_generation_tokens"))
        self._sound_truncation_alarm(world["episode/truncation_rate"], mode)
        if mode == "train":
            self._track_episode_error_round(world["episode/error_rate"], results)
        for key, value in world.items():
            self._metrics[mode][key].append(value)

    def _record_episode(self, metrics: WorldMetrics, result: RolloutResult, reasoning_tokens: int) -> None:
        truncated = float(bool(result.trajectory and result.trajectory.truncated))
        metrics.sum("async/total_rollouts", 1)
        metrics.sum("async/total_rollout_latency", result.latency)
        metrics.sum("async/total_generation_tokens", result.generation_tokens)
        metrics.mean("async/mean_rollout_latency", result.latency)
        # A count: each one threw away a turn the engine had already started.
        metrics.sum("async/requests_expired_in_sync", result.requests_expired_in_sync)
        metrics.mean("episode/generation_tokens", result.generation_tokens)
        metrics.maximum("episode/generation_tokens_max", result.generation_tokens)
        metrics.percentile("episode/generation_tokens_p90", result.generation_tokens, 90.0)
        metrics.mean("episode/turns", result.episode_length)
        metrics.mean("episode/natural_termination_rate", float(bool(result.terminated_naturally)))
        metrics.mean("episode/truncation_rate", truncated)
        metrics.mean("episode/error_rate", float(bool(result.error)))
        for key, value in result.metrics.items():
            metrics.mean(key, value)
        components = [v for k, v in result.metrics.items() if k.startswith(REWARD_COMPONENT_PREFIX)]
        if components:
            # Components must sum EXACTLY to the reward; a nonzero mean |residue| means a channel bypasses them.
            metrics.mean("reward/composition_residue", abs(result.total_reward - sum(components)))
        for name, value in self._episode_slices(result.trajectory).items():
            prefix = f"{name}/{value}"
            metrics.sum(f"{prefix}/count", 1)
            metrics.mean(f"{prefix}/reward", result.total_reward)
            metrics.mean(f"{prefix}/generation_tokens", result.generation_tokens)
            metrics.mean(f"{prefix}/reasoning_tokens", reasoning_tokens)
            metrics.mean(f"{prefix}/turns", result.episode_length)
            metrics.mean(f"{prefix}/truncation_rate", truncated)
            if SOLVE_RATE_KEY in result.metrics:
                metrics.mean(f"{prefix}/solve_rate", result.metrics[SOLVE_RATE_KEY])
            # The env's episode/* metrics, sliced by the value that conditions them; any environment's keys split
            # automatically.
            for key, value in result.metrics.items():
                if key.startswith("episode/"):
                    metrics.mean(f"{prefix}/{key.removeprefix('episode/')}", value)

    @staticmethod
    def _episode_slices(traj) -> dict[str, str]:
        """Its effort level and the env's string stamps under :data:`EPISODE_SLICES_KEY`."""
        if traj is None:
            return {}
        slices: dict[str, str] = {}
        if traj.reasoning_effort is not None:
            slices["effort"] = str(traj.reasoning_effort)
        stamps = traj.info.get(EPISODE_SLICES_KEY)
        if not isinstance(stamps, Mapping):
            return slices
        for name, value in stamps.items():
            # ``effort`` is the trainer's own slice; an env stamp cannot rename the resolved level.
            if name != "effort" and isinstance(value, str) and value:
                slices[str(name)] = value
        return slices

    def _sound_truncation_alarm(self, truncation_rate: float, mode: str) -> None:
        threshold = self.async_config.truncation_alarm_rate
        if threshold is None:
            return
        alarmed = truncation_rate > threshold
        self._metrics[mode]["episode/truncation_alarm"].append(float(alarmed))
        if alarmed and mode not in self._truncation_alarmed and is_global_main_process():
            in_loss = (
                "mask_truncated_completions drops those episodes from the loss"
                if self.args.mask_truncated_completions
                else "a truncated episode is priced like a failure"
            )
            logger.warning(
                f"{truncation_rate:.0%} of this {mode} round's episodes ended truncated, over "
                f"truncation_alarm_rate={threshold}: the turn cap (max_turns), the episode "
                f"output budget (rollout_max_episode_tokens) or unrecovered cut turns (rollout_max_tokens) bind, and "
                f"{in_loss}. Warned again once "
                f"the rate has dropped back under the threshold."
            )
        self._truncation_alarmed = self._truncation_alarmed | {mode} if alarmed else self._truncation_alarmed - {mode}

    def _track_episode_error_round(self, error_rate: float, results: list[RolloutResult]) -> None:
        """Raise on the ``episode_error_round_limit``-th train round in a row whose world-wide error rate is over
        ``episode_error_rate_limit``. Errored episodes train as masked rows, so a run that times out its slow episodes
        keeps training on the fast ones. Every rank reads the same world value, so all raise together."""
        cfg = self.async_config
        limit = cfg.episode_error_rate_limit
        if limit is None:
            return
        if error_rate <= limit:
            self._episode_error_rounds = 0
            return
        self._episode_error_rounds += 1
        sample = next((r.error for r in results if r.error), "no error on this rank")
        detail = (
            f"{error_rate:.0%} of this train round's episodes errored or timed out, over episode_error_rate_limit="
            f"{limit} for {self._episode_error_rounds} round(s) in a row (an example: {sample})"
        )
        if self._episode_error_rounds >= cfg.episode_error_round_limit:
            raise RuntimeError(
                f"{detail}. The batch is the episodes that finished, biased toward short answers, or the rollout "
                f"path is down. A timeout means a turn outran request_timeout={cfg.request_timeout:g}s at "
                f"rollout_max_tokens={cfg.rollout_max_tokens}: raise request_timeout or lower rollout_max_tokens. "
                f"Raise episode_error_rate_limit or set it to null to train through it."
            )
        if self._episode_error_rounds == 1 and is_global_main_process():
            logger.warning(f"{detail}; halting after {cfg.episode_error_round_limit}.")

    def cumulative_rollout_metrics(self) -> dict[str, float]:
        return {
            "async/total_rollouts": float(self._total_rollouts),
            "async/cumulative_mean_rollout_latency": self._total_rollout_latency / max(1, self._total_rollouts),
            "async/total_generation_tokens": float(self._total_generation_tokens),
        }
