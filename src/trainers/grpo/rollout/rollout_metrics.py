"""Rollout diagnostics for environmental GRPO: completion logs and per-episode metric aggregation.

Every metric here is computed over the GATHERED-GLOBAL episode population, so a DP rank's own
rollouts never set the logged mean on their own.
"""

import logging
import math
from collections import defaultdict

import torch
import torch.distributed as dist
from accelerate.utils import gather_object

from src.distributed.runtime import (
    fs_aware_save_rank,
    is_multi_rank_run,
    is_output_shared_filesystem,
)
from src.environments.base import SOLVE_RATE_KEY, Trajectory
from src.environments.episode import RolloutResult

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


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (``q`` in [0, 100]) of a numeric list. Empty list → 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(q / 100.0 * len(ordered))))
    return float(ordered[rank - 1])


def _summarize_episode_generation_tokens(generation_tokens: list[float]) -> dict[str, float]:
    """Per-episode generation-token summary (mean / max / p90). Each element is one episode's total
    generated tokens summed across its turns. Empty batch → all zeros."""
    if not generation_tokens:
        return {
            "episode/generation_tokens": 0.0,
            "episode/generation_tokens_max": 0.0,
            "episode/generation_tokens_p90": 0.0,
        }
    return {
        "episode/generation_tokens": sum(generation_tokens) / len(generation_tokens),
        "episode/generation_tokens_max": float(max(generation_tokens)),
        "episode/generation_tokens_p90": _percentile(generation_tokens, 90.0),
    }


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
    ) -> None:
        """Fill TRL's ``self._logs`` from this step's rollouts (TRL's base does so in its own generation
        path, which this trainer overrides). Gathered across ranks in lock-step when parquet/table is wanted.

        All four gathers run on every rank before any of them is consumed, so the writer's early
        return cannot skip a collective."""
        if not (self._save_completions or self.log_completions):
            return
        prompts_text = [r.prompt for r in rollout_results]
        completions_text = [self._render_trajectory_for_log(r.trajectory) for r in rollout_results]
        prompts, completions, reward_values, advantage_values = [
            _gather_to_completion_writers(values)
            for values in (prompts_text, completions_text, rewards.tolist(), advantages.tolist())
        ]
        if prompts is None:
            return
        self._logs["prompt"].extend(prompts)
        self._logs["completion"].extend(completions)
        self._logs["rewards"]["environment_reward"].extend(reward_values)
        self._logs["advantages"].extend(advantage_values)

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

    def _assistant_turn_reasoning_tokens(self, traj) -> list[int]:
        """Per-assistant-turn CoT token counts for calibration and the per-effort metrics.

        Every assistant turn counts, a thinking-free one as 0: skipping them would score "no thinking
        at all" as 0 while brief thinking pays the under-band penalty — a preference for dropping CoT
        entirely, the opposite of the calibration term's intent.
        """
        return [
            len(self._tokenizer(m.thinking, add_special_tokens=False)["input_ids"]) if m.thinking else 0
            for m in traj.messages
            if m.role == "assistant"
        ]

    def _log_rollout_metrics(self, results: list[RolloutResult], mode: str):
        """Log per-rollout diagnostics grouped by prefix (``async/*``, ``episode/*``, ``outcome/*``,
        ``reward/*``). Means are over the gathered-global population; ``results`` is rank-local, gathered here."""
        # Lightweight, picklable per-episode summary (the full RolloutResult carries a heavy trajectory).
        local = [
            {
                "latency": r.latency,
                "generation_tokens": r.generation_tokens,
                "turns": r.episode_length,
                "success": bool(r.success),
                "truncated": bool(r.trajectory and r.trajectory.truncated),
                "error": bool(r.error),
                "total_reward": r.total_reward,
                "effort": getattr(r.trajectory, "reasoning_effort", None) if r.trajectory else None,
                "reasoning_tokens": sum(self._assistant_turn_reasoning_tokens(r.trajectory)) if r.trajectory else 0,
                "metrics": r.metrics,
            }
            for r in results
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

        for key, val in _summarize_episode_generation_tokens([e["generation_tokens"] for e in episodes]).items():
            m[key].append(val)

        m["episode/turns"].append(_mean([e["turns"] for e in episodes]))
        m["episode/natural_termination_rate"].append(_mean([1.0 if e["success"] else 0.0 for e in episodes]))
        m["episode/truncation_rate"].append(_mean([1.0 if e["truncated"] else 0.0 for e in episodes]))
        m["episode/error_rate"].append(_mean([1.0 if e["error"] else 0.0 for e in episodes]))

        env_keys = {k for e in episodes for k in e["metrics"]}
        for key in env_keys:
            vals = [e["metrics"][key] for e in episodes if key in e["metrics"]]
            if vals:
                m[key].append(_mean(vals))

        # Components must sum EXACTLY to the reward; a nonzero mean |residue| means a channel bypasses them.
        residues = [
            abs(e["total_reward"] - sum(v for k, v in e["metrics"].items() if k.startswith("reward/")))
            for e in episodes
            if any(k.startswith("reward/") for k in e["metrics"])
        ]
        if residues:
            m["reward/composition_residue"].append(_mean(residues))

        by_effort: dict[str, list[dict]] = defaultdict(list)
        for e in episodes:
            if e["effort"] is not None:
                by_effort[e["effort"]].append(e)
        for effort, group in by_effort.items():
            prefix = f"effort/{effort}"
            m[f"{prefix}/count"].append(float(len(group)))
            m[f"{prefix}/reward"].append(_mean([e["total_reward"] for e in group]))
            m[f"{prefix}/generation_tokens"].append(_mean([e["generation_tokens"] for e in group]))
            m[f"{prefix}/reasoning_tokens"].append(_mean([e["reasoning_tokens"] for e in group]))
            m[f"{prefix}/turns"].append(_mean([e["turns"] for e in group]))
            m[f"{prefix}/truncation_rate"].append(_mean([1.0 if e["truncated"] else 0.0 for e in group]))
            solves = [e["metrics"][SOLVE_RATE_KEY] for e in group if SOLVE_RATE_KEY in e["metrics"]]
            if solves:
                m[f"{prefix}/solve_rate"].append(_mean(solves))
            # The env's per-episode strategy metrics (episode/*), sliced by the effort level that
            # conditions them — env-agnostic: any environment's episode/* keys split automatically.
            for key in {k for e in group for k in e["metrics"] if k.startswith("episode/")}:
                vals = [e["metrics"][key] for e in group if key in e["metrics"]]
                m[f"{prefix}/{key.removeprefix('episode/')}"].append(_mean(vals))
