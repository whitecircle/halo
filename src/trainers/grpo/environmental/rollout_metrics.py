"""Rollout diagnostics for environmental GRPO: completion logs and per-episode metrics, every number over the
GATHERED-GLOBAL episode population."""

import json
import logging
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch
import torch.distributed as dist
from accelerate.utils import gather_object

from src.environments.episode import RolloutResult
from src.environments.episode_keys import EPISODE_SLICES_KEY, REWARD_COMPONENTS_KEY, REWARD_DETAILS_KEY, SOLVE_RATE_KEY
from src.environments.messages import Trajectory
from src.rewards.samples import CUT_CALL_NOTE
from src.rewards.terms import REWARD_COMPONENT_PREFIX
from src.runtime.filesystem import is_fs_save_rank, is_output_shared_filesystem
from src.runtime.ranks import is_global_main_process, is_multi_rank_run
from src.trainers.grpo.rollout.completions_logging import emit_completion_artifacts

logger = logging.getLogger(__name__)


def _gather_to_completion_writers(values: list) -> list | None:
    """COLLECTIVE. ``values`` gathered across the world in rank order to the ranks that write the completions
    artifact (:func:`is_fs_save_rank`), ``None`` elsewhere: the payload is the heaviest this trainer moves, and
    an all-gather would stage the whole world's text through every rank's device."""
    if not is_multi_rank_run():
        return list(values)
    if not is_output_shared_filesystem():
        return list(gather_object(values))
    # Shared output FS ⇒ is_fs_save_rank() is global rank 0, which is what dst names.
    chunks: list | None = [None] * dist.get_world_size() if is_fs_save_rank() else None
    dist.gather_object(values, chunks, dst=0)
    if chunks is None:
        return None
    return [item for chunk in chunks for item in chunk]


def _episode_info_json(trajectory: Trajectory | None, key: str) -> str:
    """One mapping the episode keeps under ``key`` in ``trajectory.info``, as sorted-key JSON for the
    completion record; ``{}`` for an episode that kept none (a lost rollout, an unscored term)."""
    value = trajectory.info.get(key, {}) if trajectory is not None else {}
    return json.dumps(value, sort_keys=True, default=str)


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (``q`` in [0, 100]) of a non-empty numeric list."""
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(q / 100.0 * len(ordered))))
    return float(ordered[rank - 1])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def group_solve_counts(solved: Sequence[bool | None], group_size: int) -> tuple[int, int, int, int]:
    """``(groups, all_pass, all_fail, any_pass)`` over consecutive groups of ``group_size`` solve verdicts, each
    group judged on the episodes that carry one (not ``None``); a group with none is not counted."""
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
    """Completion logging and rollout diagnostics for the environmental GRPO trainer; no training signal."""

    # Cumulative totals over the gathered-global population, never this rank's shard; ``+=`` rebinds per
    # instance.
    _total_rollouts = 0
    _total_rollout_latency = 0.0
    _total_generation_tokens = 0
    # The mode whose rows ``self._logs`` holds.
    _completion_logs_mode: str | None = None
    # ``truncation_alarm_rate`` (``None``: no alarm) and the modes whose last round was over it.
    _truncation_alarm_rate: float | None = None
    _truncation_alarmed: frozenset[str] = frozenset()

    def _populate_completion_logs(
        self,
        rollout_results: list[RolloutResult],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        mode: str,
    ) -> None:
        """COLLECTIVE. Fill TRL's ``self._logs`` from this step's rollouts: text, reward, advantage, and as JSON
        the settled reward components and every scored term's rationale. Every gather runs on every rank before
        any is consumed; rows of the other mode still waiting are written under their own mode first."""
        if not (self._save_completions or self.log_completions):
            return
        if self._completion_logs_mode not in (None, mode):
            emit_completion_artifacts(
                self, console=False, save=self._save_completions, mode=self._completion_logs_mode
            )
        self._completion_logs_mode = mode
        prompts_text = [r.prompt for r in rollout_results]
        completions_text = [self._render_trajectory_for_log(r.trajectory) for r in rollout_results]
        components_json = [_episode_info_json(r.trajectory, REWARD_COMPONENTS_KEY) for r in rollout_results]
        details_json = [_episode_info_json(r.trajectory, REWARD_DETAILS_KEY) for r in rollout_results]
        prompts, completions, reward_values, advantage_values, components, details = [
            _gather_to_completion_writers(values)
            for values in (
                prompts_text,
                completions_text,
                rewards.tolist(),
                advantages.tolist(),
                components_json,
                details_json,
            )
        ]
        if prompts is None:
            return
        self._logs["prompt"].extend(prompts)
        self._logs["completion"].extend(completions)
        self._logs["rewards"]["environment_reward"].extend(reward_values)
        self._logs["advantages"].extend(advantage_values)
        self._logs["extra"][REWARD_COMPONENTS_KEY].extend(components)
        self._logs["extra"][REWARD_DETAILS_KEY].extend(details)

    @staticmethod
    def _render_trajectory_for_log(trajectory: "Trajectory | None") -> str:
        """A readable multi-turn render for the completions log, a cut turn's unrun calls marked
        :data:`~src.rewards.samples.CUT_CALL_NOTE`."""
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
        """COLLECTIVE. Log per-rollout diagnostics grouped by prefix (``async/*``, ``episode/*``, ``outcome/*``,
        ``reward/*``), means over the gathered-global population of this rank's ``results``.
        ``reasoning_tokens`` is each episode's per-turn count, taken once for the step."""
        # Lightweight, picklable per-episode summary (the full RolloutResult carries a heavy trajectory).
        local = [
            {
                "latency": r.latency,
                "generation_tokens": r.generation_tokens,
                "requests_expired_in_sync": r.requests_expired_in_sync,
                "turns": r.episode_length,
                "success": bool(r.success),
                "truncated": bool(r.trajectory and r.trajectory.truncated),
                "error": bool(r.error),
                "total_reward": r.total_reward,
                "slices": self._episode_slices(r.trajectory),
                "reasoning_tokens": sum(tokens),
                "metrics": r.metrics,
            }
            for r, tokens in zip(results, reasoning_tokens, strict=True)
        ]
        episodes = gather_object(local)
        if not episodes:
            return

        self._total_rollouts += len(episodes)
        self._total_rollout_latency += sum(e["latency"] for e in episodes)
        self._total_generation_tokens += sum(e["generation_tokens"] for e in episodes)
        m = self._metrics[mode]
        self._log_episode_metrics(m, episodes, mode)
        self._log_env_metrics(m, episodes)
        self._log_slice_metrics(m, episodes)

    @staticmethod
    def _episode_slices(traj) -> dict[str, str]:
        """The categorical facts an episode's metrics are sliced by: its effort level and the env's string stamps
        under :data:`EPISODE_SLICES_KEY`."""
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

    def _log_episode_metrics(self, m: dict[str, list[float]], episodes: list[dict], mode: str) -> None:
        """The ``async/*`` latency and ``episode/*`` length and outcome rates of the gathered episodes."""
        m["async/mean_rollout_latency"].append(_mean([e["latency"] for e in episodes]))
        # A count, not a mean: each one threw away a turn the engine had already started.
        m["async/requests_expired_in_sync"].append(float(sum(e["requests_expired_in_sync"] for e in episodes)))
        # Each episode's generated tokens, summed across its turns.
        generation_tokens = [e["generation_tokens"] for e in episodes]
        m["episode/generation_tokens"].append(_mean(generation_tokens))
        m["episode/generation_tokens_max"].append(float(max(generation_tokens)))
        m["episode/generation_tokens_p90"].append(_percentile(generation_tokens, 90.0))
        m["episode/turns"].append(_mean([e["turns"] for e in episodes]))
        m["episode/natural_termination_rate"].append(_mean([1.0 if e["success"] else 0.0 for e in episodes]))
        truncation_rate = _mean([1.0 if e["truncated"] else 0.0 for e in episodes])
        m["episode/truncation_rate"].append(truncation_rate)
        self._sound_truncation_alarm(truncation_rate, mode)
        m["episode/error_rate"].append(_mean([1.0 if e["error"] else 0.0 for e in episodes]))

    def _sound_truncation_alarm(self, truncation_rate: float, mode: str) -> None:
        """``episode/truncation_alarm`` for this round, and a warning when the rate crosses over
        ``truncation_alarm_rate``."""
        if self._truncation_alarm_rate is None:
            return
        alarmed = truncation_rate > self._truncation_alarm_rate
        self._metrics[mode]["episode/truncation_alarm"].append(float(alarmed))
        if alarmed and mode not in self._truncation_alarmed and is_global_main_process():
            in_loss = (
                "mask_truncated_completions drops those episodes from the loss"
                if self.args.mask_truncated_completions
                else "a truncated episode is priced like a failure"
            )
            logger.warning(
                f"{truncation_rate:.0%} of this {mode} round's episodes ended truncated, over "
                f"truncation_alarm_rate={self._truncation_alarm_rate}: the turn cap (max_turns), the episode "
                f"output budget (rollout_max_episode_tokens) or unrecovered cut turns (rollout_max_tokens) bind, and "
                f"{in_loss}. Warned again once "
                f"the rate has dropped back under the threshold."
            )
        self._truncation_alarmed = self._truncation_alarmed | {mode} if alarmed else self._truncation_alarmed - {mode}

    @staticmethod
    def _log_env_metrics(m: dict[str, list[float]], episodes: list[dict]) -> None:
        """The mean of every metric an environment reported, and how far the reward components fall short of
        summing to the reward."""
        for key in {k for e in episodes for k in e["metrics"]}:
            vals = [e["metrics"][key] for e in episodes if key in e["metrics"]]
            if vals:
                m[key].append(_mean(vals))
        # Components must sum EXACTLY to the reward; a nonzero mean |residue| means a channel bypasses them.
        residues = [
            abs(e["total_reward"] - sum(v for k, v in e["metrics"].items() if k.startswith(REWARD_COMPONENT_PREFIX)))
            for e in episodes
            if any(k.startswith(REWARD_COMPONENT_PREFIX) for k in e["metrics"])
        ]
        if residues:
            m["reward/composition_residue"].append(_mean(residues))

    @staticmethod
    def _log_slice_metrics(m: dict[str, list[float]], episodes: list[dict]) -> None:
        """The per-value breakdown (``<slice>/<value>/count``, ``.../solve_rate`` ...) of every slice an episode
        carries."""
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for e in episodes:
            for name, value in e["slices"].items():
                groups[(name, value)].append(e)
        for (name, value), group in groups.items():
            prefix = f"{name}/{value}"
            m[f"{prefix}/count"].append(float(len(group)))
            m[f"{prefix}/reward"].append(_mean([e["total_reward"] for e in group]))
            m[f"{prefix}/generation_tokens"].append(_mean([e["generation_tokens"] for e in group]))
            m[f"{prefix}/reasoning_tokens"].append(_mean([e["reasoning_tokens"] for e in group]))
            m[f"{prefix}/turns"].append(_mean([e["turns"] for e in group]))
            m[f"{prefix}/truncation_rate"].append(_mean([1.0 if e["truncated"] else 0.0 for e in group]))
            solves = [e["metrics"][SOLVE_RATE_KEY] for e in group if SOLVE_RATE_KEY in e["metrics"]]
            if solves:
                m[f"{prefix}/solve_rate"].append(_mean(solves))
            # The env's per-episode strategy metrics (episode/*), sliced by the value that conditions
            # them — env-agnostic: any environment's episode/* keys split automatically.
            for key in {k for e in group for k in e["metrics"] if k.startswith("episode/")}:
                vals = [e["metrics"][key] for e in group if key in e["metrics"]]
                m[f"{prefix}/{key.removeprefix('episode/')}"].append(_mean(vals))

    def cumulative_rollout_metrics(self) -> dict[str, float]:
        """The ``async/*`` totals since train start, as the trainer logs them."""
        return {
            "async/total_rollouts": float(self._total_rollouts),
            "async/cumulative_mean_rollout_latency": self._total_rollout_latency / max(1, self._total_rollouts),
            "async/total_generation_tokens": float(self._total_generation_tokens),
        }
