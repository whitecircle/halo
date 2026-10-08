"""Rollout diagnostics for environmental GRPO: completion logs and per-episode metric aggregation.

Every number here is computed over the GATHERED-GLOBAL episode population, so a DP rank's own rows never
set a logged value on their own.
"""

import json
import logging
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch
import torch.distributed as dist
from accelerate.utils import gather_object

from src.distributed.runtime import (
    fs_aware_save_rank,
    is_global_main_process,
    is_multi_rank_run,
    is_output_shared_filesystem,
)
from src.environments.base import (
    EPISODE_SLICES_KEY,
    REWARD_COMPONENTS_KEY,
    REWARD_DETAILS_KEY,
    SOLVE_RATE_KEY,
    Trajectory,
)
from src.environments.episode import RolloutResult
from src.rewards.terms import REWARD_COMPONENT_PREFIX
from src.trainers.grpo.rollout.completions_logging import emit_completion_artifacts

logger = logging.getLogger(__name__)


def _gather_to_completion_writers(values: list) -> list | None:
    """Gather ``values`` across the world in rank order, delivering only to the ranks that WRITE the
    completions artifact; ``None`` on every other rank. COLLECTIVE — every rank must call it.

    The payload is the full multi-turn trajectory render, the heaviest object this trainer moves. An
    all-gather hands the whole world's text to every rank and (on NCCL) stages the pickle through
    that rank's CUDA device, so the transient grows linearly in world size while the only consumer
    is ``emit_completion_artifacts`` on the writer rank. With a shared output filesystem that writer
    is global rank 0 alone, so the payload is gathered there and nowhere else; without one, every
    node's local rank 0 writes its own copy of the world record, and the all-gather is what feeds
    them. The receiving set is :func:`fs_aware_save_rank` itself — the same predicate that elects
    the writer — so the two cannot drift into gathering to a rank that does not write.
    """
    if not is_multi_rank_run():
        return list(values)
    if not is_output_shared_filesystem():
        return list(gather_object(values))
    # Shared output FS ⇒ fs_aware_save_rank() is global rank 0, which is what dst names.
    chunks: list | None = [None] * dist.get_world_size() if fs_aware_save_rank() else None
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


def _summarize_episode_generation_tokens(generation_tokens: list[float]) -> dict[str, float]:
    """Per-episode generation-token summary (mean / max / p90) of a non-empty step. Each element is one
    episode's total generated tokens summed across its turns."""
    return {
        "episode/generation_tokens": sum(generation_tokens) / len(generation_tokens),
        "episode/generation_tokens_max": float(max(generation_tokens)),
        "episode/generation_tokens_p90": _percentile(generation_tokens, 90.0),
    }


def group_solve_counts(solved: Sequence[bool | None], group_size: int) -> tuple[int, int, int, int]:
    """``(groups, all_pass, all_fail, any_pass)`` over consecutive groups of ``group_size`` episodes.

    Each entry is an episode's solve verdict, or ``None`` for one that carries none (the environment
    reports no verdict, or the episode left the baseline). A group is judged on the episodes that carry
    a verdict; a group with none is not counted.
    """
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
    """Completion logging and rollout diagnostics for the environmental GRPO trainer.

    Reads the trainer's ``self._logs``/``self._metrics`` accumulators and the accelerate gathers; it
    computes no training signal, so every method here is safe to call on any rank.
    """

    # Cumulative infra totals, accumulated from the GATHERED-GLOBAL population :meth:`_log_rollout_metrics`
    # already builds — never from this rank's own shard, which would under-report the job by the DP
    # size while reading as a job total. Class-level so no __init__ is needed; ``+=`` rebinds per
    # instance. Under TP/ETP each group's leader rollouts appear once per member, matching the
    # duplicate generation those ranks really performed.
    _total_rollouts = 0
    _total_rollout_latency = 0.0
    _total_generation_tokens = 0
    # The mode whose rows ``self._logs`` holds; rebinds per instance like the counters above.
    _completion_logs_mode: str | None = None
    # ``truncation_alarm_rate``, set by the trainer (``None`` = no alarm), and the modes whose last round
    # was over it: the warning fires on a crossing, not on every round above the line.
    _truncation_alarm_rate: float | None = None
    _truncation_alarmed: frozenset[str] = frozenset()

    def cumulative_rollout_metrics(self) -> dict[str, float]:
        """The ``async/*`` totals since train start, as the trainer logs them."""
        return {
            "async/total_rollouts": float(self._total_rollouts),
            "async/cumulative_mean_rollout_latency": self._total_rollout_latency / max(1, self._total_rollouts),
            "async/total_generation_tokens": float(self._total_generation_tokens),
        }

    def _populate_completion_logs(
        self,
        rollout_results: list[RolloutResult],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        mode: str,
    ) -> None:
        """Fill TRL's ``self._logs`` from this step's rollouts (TRL's base does so in its own generation
        path, which this trainer overrides). Gathered across ranks in lock-step when parquet/table is wanted.

        Beside the text, reward and advantage, each row carries the episode's settled reward components
        and every scored term's rationale (a veto judge's fired checks with their quotes) as JSON, so a
        vetoed or oddly priced episode can be read off the record without scoring it again.

        Every gather runs on every rank before any of them is consumed, so the writer's early return
        cannot skip a collective. Rows of the other mode still waiting for their log (an eval round on
        a step ``logging_steps`` skipped) are written under their own mode first, never into this
        round's file; every rank enters that write, whose failure is raised on all of them."""
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
        """Readable multi-turn render for completion logging: each non-system message plus tool calls and reasoning."""
        if trajectory is None or not trajectory.messages:
            return "(empty trajectory)"
        parts = []
        for m in trajectory.messages:
            if m.role == "system":
                continue
            seg = f"[{m.role}] {m.content or ''}".rstrip()
            if m.thinking:
                seg += f"\n  <reasoning> {m.thinking}"
            for tc in m.tool_calls or []:
                fn = tc.get("function", tc)
                seg += f"\n  <tool_call {fn.get('name', '?')}> {fn.get('arguments', '')}"
            parts.append(seg)
        return "\n".join(parts)

    @staticmethod
    def _episode_slices(traj) -> dict[str, str]:
        """The categorical facts an episode's metrics are sliced by: its resolved effort level under
        ``effort`` plus whatever the env stamped under :data:`EPISODE_SLICES_KEY` (string values only;
        a non-string stamp would fan out into one metric key per object repr)."""
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

    def _log_rollout_metrics(self, results: list[RolloutResult], mode: str, reasoning_tokens: list[list[int]]):
        """Log per-rollout diagnostics grouped by prefix (``async/*``, ``episode/*``, ``outcome/*``,
        ``reward/*``). Means are over the gathered-global population; ``results`` is rank-local, gathered here.
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

        def _mean(vals: list[float]) -> float:
            return sum(vals) / len(vals)

        m["async/mean_rollout_latency"].append(_mean([e["latency"] for e in episodes]))
        # A count, not a mean: each one threw away a turn the engine had already started.
        m["async/requests_expired_in_sync"].append(float(sum(e["requests_expired_in_sync"] for e in episodes)))

        for key, val in _summarize_episode_generation_tokens([e["generation_tokens"] for e in episodes]).items():
            m[key].append(val)

        m["episode/turns"].append(_mean([e["turns"] for e in episodes]))
        m["episode/natural_termination_rate"].append(_mean([1.0 if e["success"] else 0.0 for e in episodes]))
        truncation_rate = _mean([1.0 if e["truncated"] else 0.0 for e in episodes])
        m["episode/truncation_rate"].append(truncation_rate)
        self._sound_truncation_alarm(truncation_rate, mode)
        m["episode/error_rate"].append(_mean([1.0 if e["error"] else 0.0 for e in episodes]))

        env_keys = {k for e in episodes for k in e["metrics"]}
        for key in env_keys:
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

        # Every slice an episode carries (its effort level, the env's own categorical stamps such as
        # the language it submitted in) gets the same per-value breakdown, so a run's balance across
        # the values and each value's outcome read off ``<slice>/<value>/count`` and ``.../solve_rate``.
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

    def _sound_truncation_alarm(self, truncation_rate: float, mode: str) -> None:
        """``episode/truncation_alarm`` for this round, and a warning when the rate crosses over
        ``truncation_alarm_rate``, naming what the loss does with a truncated episode. The rate is
        gathered-global, so every rank reaches the same verdict."""
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
