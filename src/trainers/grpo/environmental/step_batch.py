"""One environmental GRPO step's training batch: rollouts to rows, scores, advantages and loss masks."""

import contextlib
import dataclasses
import logging
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate.utils import gather
from trl.trainer.utils import pad

from src.environments.episode import RolloutResult
from src.environments.episode_keys import EPISODE_INVALID_REASON_KEY, solve_verdict
from src.runtime.consensus import DeferredRankFailure, reject_across_ranks
from src.runtime.ranks import is_global_main_process, is_multi_rank_run
from src.trainers.grpo.environmental.rollout_metrics import group_solve_counts
from src.trainers.grpo.environmental.trajectory_tokenize import TurnRow, single_trajectory_row
from src.trainers.grpo.logprob_scoring import dense_row_spans
from src.trainers.grpo.objective.advantages import degenerate_group_mask, group_relative_advantages, valid_group_stats
from src.trainers.grpo.objective.logratio import (
    KL_CLAMP_FRAC_KEY,
    clamp_ref_logps,
    kl_clamp_counts,
)
from src.trainers.grpo.objective.loss_inputs import (
    DEGENERATE_GROUP_FRAC_KEY,
    expand_traj_to_rows,
    gathered_num_items,
    narrow_loss_masks,
    record_token_mass,
)
from src.trainers.token_metrics import effective_loss_mask

logger = logging.getLogger(__name__)

# The batch key the routing-replay masks ride under, row-aligned with the rest of the batch.
ROUTING_MASKS_KEY = "routing_masks"
# Consecutive steps with no valid episode anywhere that halt the run: one can be a grader blip, a second
# means the rollout path is down.
EMPTY_ROLLOUT_STEP_LIMIT = 2


@dataclasses.dataclass(frozen=True)
class BatchRows:
    """This step's rollouts and how they map onto its rows: each trajectory's row count (past
    ``rollout_results`` by an eval round's padding trajectories), this rank's dummy rows, the per-turn split."""

    rollout_results: list[RolloutResult]
    turns_per_traj: list[int]
    num_dummy_rows: int
    per_turn: bool

    def to_rows(self, values: torch.Tensor, dummy_fill: float = 0) -> torch.Tensor:
        """Per-trajectory ``values`` laid out over this step's rows (see :func:`expand_traj_to_rows`)."""
        return expand_traj_to_rows(values, self.turns_per_traj, self.num_dummy_rows, self.per_turn, dummy_fill)

    def traj_row_ids(self, device: torch.device) -> torch.Tensor:
        """Each row's index into ``rollout_results``; -1 on the padding and dummy rows, which are no rollout's."""
        ids = torch.arange(len(self.rollout_results), device=device)
        return self.to_rows(F.pad(ids, (0, len(self.turns_per_traj) - len(ids)), value=-1), dummy_fill=-1)


@dataclasses.dataclass(frozen=True)
class StepRows:
    """This step's rows stacked into the padded tensors every forward reads; ``completion_mask`` is
    attention-valid, ``tool_mask`` the loss mask."""

    prompt_ids: torch.Tensor
    completion_ids: torch.Tensor
    prompt_mask: torch.Tensor
    completion_mask: torch.Tensor
    tool_mask: torch.Tensor
    sampling_logps: torch.Tensor | None
    has_sampling: torch.Tensor | None
    negative_only: torch.Tensor
    prompt_completion_ids: torch.Tensor
    attention_mask: torch.Tensor
    logits_to_keep: int


@dataclasses.dataclass(frozen=True)
class StepScores:
    """What the step's no-grad pass scored; ``old_logps`` is the recompute where the loss reads it."""

    recompute_logps: torch.Tensor | None
    old_logps: torch.Tensor | None
    routing_masks: torch.Tensor | None
    ratio: torch.Tensor
    logps_diff: torch.Tensor | None
    corrected_mask: torch.Tensor | None
    traj_row_ids: torch.Tensor | None
    ref_logps: torch.Tensor | None
    kl_clamped: torch.Tensor | None


class BatchBuildFence(DeferredRankFailure):
    """The rank-uniform fence of the batch construction, reused every round: a failure is recorded, never
    raised on one rank (which would strand its peers in the next collective), and :meth:`reject` raises it
    on every rank, naming the first failing rank."""

    def __init__(self) -> None:
        super().__init__("Environmental GRPO batch construction", exc_type=ValueError)

    def record(self, message: str) -> None:
        """Keep the first failure: a later one on the same batch would replace the root cause."""
        if self.reason is None:
            self.reason = message

    def reject(self) -> None:
        """COLLECTIVE. Raise on every rank if any recorded a failure; this rank's starts the next round clear."""
        reason, self.reason = self.reason, None
        reject_across_ranks(reason, self.what, exc_type=self.exc_type)


def _row_on_device(row: TurnRow, device: torch.device) -> TurnRow:
    """``row`` on ``device``, its loss mask as bool: the ``[rows, max_len]`` sink every mask stage reads."""
    return row._replace(
        prompt_ids=row.prompt_ids.to(device),
        completion_ids=row.completion_ids.to(device),
        completion_mask=row.completion_mask.to(device=device, dtype=torch.bool),
        sampling_logps=None if row.sampling_logps is None else row.sampling_logps.to(device),
    )


def rollout_valid_mask(rollout_results: list[RolloutResult], device: torch.device) -> torch.Tensor:
    """Per-rollout bool mask of :attr:`RolloutResult.counts_toward_baseline`."""
    return torch.tensor([r.counts_toward_baseline for r in rollout_results], device=device, dtype=torch.bool)


def batch_reward_std(rewards: torch.Tensor) -> float:
    """Sample std of a gathered reward batch; 0.0 for a single reward, where ``Tensor.std()`` returns NaN."""
    return rewards.std().item() if rewards.numel() > 1 else 0.0


class StepBatchMixin:
    """Builds one step's training batch for the environmental GRPO trainer."""

    def _build_training_tensors(
        self,
        rollout_results: list[RolloutResult],
        device: torch.device,
        mode: str,
        num_padding: int,
        reasoning_tokens: list[list[int]],
    ) -> dict[str, torch.Tensor]:
        """One step's training batch from its rollouts, each one's per-turn ``reasoning_tokens`` and the
        ``num_padding`` inert trajectories that pad an eval split's final round (:meth:`eval_split_rows`).

        The phases run in this order on every rank, and every collective inside them sits behind a config- or
        mode-derived gate, never a local-data one: a rank-dependent skip or early-out desyncs the world.
        """
        rows, turns_per_traj = self._rows_for_step(rollout_results, num_padding, device)
        # After tokenizing: an episode the template cannot re-render leaves the baseline the waiver reads.
        rewards = F.pad(self._build_rollout_rewards(rollout_results, reasoning_tokens, device, mode), (0, num_padding))
        # Fences the row-shape failures tokenization recorded (a context overflow, a malformed routing payload).
        self._batch_errors.reject()
        # After the fence and ahead of every collective: the re-score never raises, but it must not sit between
        # a rank's collectives. Train mode only: eval runs the ratio and mask stages on the trainer's log-probs.
        engine_logps = None
        if self._is_correction and self.async_config.isr_engine_reference and mode == "train":
            engine_logps = self._rescore_rows_on_engine(rows, turns_per_traj)
        num_dummy_rows = self._pad_rows_across_ranks(rows, engine_logps, device)
        batch_rows = BatchRows(
            rollout_results, turns_per_traj, num_dummy_rows, self.async_config.train_on_sampled_tokens
        )
        step = self._stack_rows(rows, device)
        rollout_routing_masks = self._assemble_rollout_routing(rows, step, device, mode)

        with torch.no_grad():
            scores = self._score_step(step, batch_rows, rollout_routing_masks, engine_logps, mode)

        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        episode_state, traj_advantages, local_advantages = self._step_advantages(
            rewards, batch_rows, num_padding, num_generations
        )
        scores = self._apply_opsm_stage(scores, local_advantages)
        gathered_rewards, gathered_state = gather(rewards), gather(episode_state)
        if mode == "train":
            self._verify_step_has_valid_episodes(rollout_results, gathered_state == 1)
        completion_mask, tool_mask, loss_mask, num_items_in_batch = self._narrow_masks_and_normalizer(
            batch_rows, step, episode_state == 1, num_generations, local_advantages, mode
        )
        if scores.kl_clamped is not None:
            self._world_metrics.fraction(KL_CLAMP_FRAC_KEY, *kl_clamp_counts(scores.kl_clamped, loss_mask))
        if mode == "train":
            local_advantages, traj_advantages = self._apply_token_mass_balance(
                local_advantages, traj_advantages, loss_mask, scores.ratio, step.negative_only
            )
        local_advantages = self._record_step_metrics(
            rewards,
            gathered_rewards,
            gathered_state,
            batch_rows,
            scores,
            loss_mask,
            local_advantages,
            traj_advantages,
            num_generations,
            mode,
        )

        result = {
            "prompt_ids": step.prompt_ids,
            "prompt_mask": step.prompt_mask,
            "completion_ids": step.completion_ids,
            "completion_mask": completion_mask,
            "tool_mask": tool_mask,
            "advantages": local_advantages,
            "old_per_token_logps": scores.old_logps,
            "ref_per_token_logps": scores.ref_logps,
            "importance_sampling_ratio": scores.ratio,
            "num_items_in_batch": num_items_in_batch,
        }
        if scores.routing_masks is not None:
            # Rides the batch dict like old_per_token_logps so TRL's shuffle/slice keep it row-aligned.
            result[ROUTING_MASKS_KEY] = scores.routing_masks
        return result

    def _rows_for_step(
        self, rollout_results: list[RolloutResult], num_padding: int, device: torch.device
    ) -> tuple[list[TurnRow], list[int]]:
        """This step's training rows on ``device`` and how many each trajectory contributed: one per assistant
        turn under ``train_on_sampled_tokens``, else one re-rendered row, and one inert row per padding
        trajectory."""
        padding = [single_trajectory_row(self._masked_trajectory_tensors())] * num_padding
        per_trajectory = self._tokenize_step_rows(rollout_results) + padding
        rows = [_row_on_device(row, device) for turn_rows in per_trajectory for row in turn_rows]
        return rows, [len(turn_rows) for turn_rows in per_trajectory]

    def _pad_rows_across_ranks(
        self, rows: list[TurnRow], engine_logps: list[torch.Tensor | None] | None, device: torch.device
    ) -> int:
        """COLLECTIVE under ``train_on_sampled_tokens``. Append masked dummy rows (no engine score each) up to the
        world's largest row count, rounded up to a multiple of ``steps_per_generation``, and return how many: the
        per-turn split gives each rank a variable row count, unequal forwards deadlock, and TRL drops a remainder."""
        num_dummy_rows = 0
        if self.async_config.train_on_sampled_tokens:
            global_rows = torch.tensor([len(rows)], device=device)
            if is_multi_rank_run():
                dist.all_reduce(global_rows, op=dist.ReduceOp.MAX)
            chunks = self.args.steps_per_generation
            num_dummy_rows = math.ceil(int(global_rows.item()) / chunks) * chunks - len(rows)
            # One attended prompt token (no NaN), a fully masked completion (zero loss), natural routing.
            (inert,) = single_trajectory_row(self._masked_trajectory_tensors())
            rows.extend([_row_on_device(inert, device)] * num_dummy_rows)
            if engine_logps is not None:
                engine_logps.extend([None] * num_dummy_rows)
        return num_dummy_rows

    def _stack_rows(self, rows: list[TurnRow], device: torch.device) -> StepRows:
        """The step's rows padded into the batch tensors; the sampling log-probs under the IS correction only,
        a config-derived gate every rank takes alike."""
        prompt_ids = pad([row.prompt_ids for row in rows], padding_value=self.pad_token_id, padding_side="left")
        completion_ids = pad(
            [row.completion_ids for row in rows], padding_value=self.pad_token_id, padding_side="right"
        )
        prompt_mask = pad(
            [torch.ones_like(row.prompt_ids, dtype=torch.bool) for row in rows], padding_value=0, padding_side="left"
        )
        # ``completion_mask`` is attention-valid (env/tool tokens conditioned sampling), ``tool_mask`` the loss
        # mask: conflating them hides tool output from attention. A fully masked row stays invisible to both.
        completion_mask = pad(
            [
                torch.ones_like(row.completion_mask) if row.completion_mask.any() else row.completion_mask
                for row in rows
            ],
            padding_value=0,
            padding_side="right",
        )
        tool_mask = pad([row.completion_mask for row in rows], padding_value=0, padding_side="right")
        sampling_logps = has_sampling = None
        if self._is_correction:
            # A turn missing sampling log-probs keeps ratio ≡ 1 for that row alone: has_sampling masks its zeros.
            sampling_logps = pad(
                [
                    row.sampling_logps
                    if row.sampling_logps is not None
                    else torch.zeros(len(row.completion_ids), device=device)
                    for row in rows
                ],
                padding_value=0,
                padding_side="right",
            )
            has_sampling = torch.tensor([row.sampling_logps is not None for row in rows], device=device)
        return StepRows(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            prompt_mask=prompt_mask,
            completion_mask=completion_mask,
            tool_mask=tool_mask,
            sampling_logps=sampling_logps,
            has_sampling=has_sampling,
            negative_only=torch.tensor([row.negative_only for row in rows], device=device, dtype=torch.bool),
            prompt_completion_ids=torch.cat([prompt_ids, completion_ids], dim=1),
            attention_mask=torch.cat([prompt_mask, completion_mask], dim=1),
            logits_to_keep=completion_ids.size(1),
        )

    def _score_step(
        self,
        step: StepRows,
        rows: BatchRows,
        rollout_routing_masks: torch.Tensor | None,
        engine_logps: list[torch.Tensor | None] | None,
        mode: str,
    ) -> StepScores:
        """The step's no-grad scores: the policy recompute, the IS weights with the engine's forced closes
        zeroed, and the reference log-probs clamped against the policy's."""
        recompute_logps, routing_masks = self._recompute_logps_and_routing_masks(
            step.prompt_completion_ids, step.attention_mask, step.logits_to_keep, rollout_routing_masks, mode
        )
        ratio, logps_diff, corrected_mask, traj_row_ids = self._apply_is_correction(
            recompute_logps, step, rows, engine_logps
        )
        if self._is_correction and self._forced_close_ids is not None:
            ratio = self._zero_forced_closes(ratio, step)
        ref_logps = self._compute_ref_logps(step.prompt_completion_ids, step.attention_mask, step.logits_to_keep)
        kl_clamped = None
        if ref_logps is not None and recompute_logps is not None:
            ref_logps, kl_clamped = clamp_ref_logps(ref_logps, recompute_logps)
        # None on the aligned single-iteration path: TRL's detach() reuse makes the ratio exactly 1.
        old_logps = recompute_logps if self.num_iterations > 1 or self._misaligned_accumulation else None
        return StepScores(
            recompute_logps,
            old_logps,
            routing_masks,
            ratio,
            logps_diff,
            corrected_mask,
            traj_row_ids,
            ref_logps,
            kl_clamped,
        )

    def _recompute_logps_and_routing_masks(
        self,
        prompt_completion_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        rollout_routing_masks: torch.Tensor | None,
        mode: str,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """The no-grad log-prob recompute and the routing mask the gradient forward replays. The gate is
        config-derived, so the forward and its EP dispatch are entered by every rank or by none."""
        # Recomputed once and chunked: a full-batch DeepEP dispatch overflows its 32-bit limit.
        recompute_logps = None
        # Either the recompute forward below captures the mask, or the engine already supplied it.
        routing_masks = rollout_routing_masks
        if self._runs_logprob_recompute:
            # Capture is scoped to this call: the ref-logps forward that follows is a different policy.
            replay_spans = (
                dense_row_spans(attention_mask)
                if self._routing_injector is not None and self._replay_row_spans and mode == "train"
                else None
            )
            with contextlib.ExitStack() as capture_stack:
                captured = (
                    capture_stack.enter_context(
                        self._routing_injector.capture(
                            prompt_completion_ids.size(0), prompt_completion_ids.size(1), row_spans=replay_spans
                        )
                    )
                    if self._routing_injector is not None and mode == "train" and not self._rollout_routing_replay
                    else None
                )
                if rollout_routing_masks is not None:
                    # The recompute must run under the same routing the update pass replays, or
                    # the ratio comes from a different token distribution than the policy trains on.
                    self._routing_injector.arm(rollout_routing_masks, row_spans=replay_spans)
                try:
                    recompute_logps, _, _ = self._get_per_token_logps_and_entropies(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        compute_entropy=False,
                    )
                finally:
                    if rollout_routing_masks is not None:
                        self._routing_injector.disarm()
            if captured:
                routing_masks = captured[0]
        return recompute_logps, routing_masks

    def _compute_ref_logps(
        self,
        prompt_completion_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor | None:
        """Reference-model log-probs for the KL penalty (``None`` when ``beta == 0``)."""
        if self.beta == 0.0:
            return None
        # Chunked: a whole-batch DeepEP dispatch exceeds its 32-bit limit on long trajectories.
        with self._reference_policy() as model:
            ref_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                model,
                prompt_completion_ids,
                attention_mask,
                logits_to_keep,
                compute_entropy=False,
            )
        return ref_per_token_logps

    def _step_advantages(
        self, rewards: torch.Tensor, rows: BatchRows, num_padding: int, num_generations: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(episode_state, traj_advantages, local_advantages)``; the state is 1 valid, 0 invalid, -1 an eval
        round's padding."""
        episode_state = F.pad(
            rollout_valid_mask(rows.rollout_results, rewards.device).to(torch.int8), (0, num_padding), value=-1
        )
        traj_advantages = group_relative_advantages(
            rewards,
            num_generations,
            self.args.scale_rewards,
            valid_mask=episode_state == 1,
            std_floor=self._scale_rewards_std_floor,
        )
        return episode_state, traj_advantages, rows.to_rows(traj_advantages)

    def _verify_step_has_valid_episodes(
        self, rollout_results: list[RolloutResult], gathered_valid: torch.Tensor
    ) -> None:
        """Warn on a step where no episode in the world survived the rollout, and raise on the
        :data:`EMPTY_ROLLOUT_STEP_LIMIT`-th in a row: it trains a zero gradient while logging ``loss=0``. Judged
        on the world-gathered ``gathered_valid``, so every rank decides alike."""
        if gathered_valid.any():
            self._empty_rollout_steps = 0
            return
        self._empty_rollout_steps += 1
        errors = sorted({r.error for r in rollout_results if r.error})
        untrainable = sorted(
            {
                r.trajectory.info[EPISODE_INVALID_REASON_KEY]
                for r in rollout_results
                if r.trajectory is not None and EPISODE_INVALID_REASON_KEY in r.trajectory.info
            }
        )
        if errors:
            detail = f" Rollout error: {errors[0]}"
        elif untrainable:
            detail = f" Episodes were dropped as untrainable: {untrainable[0]}"
        else:
            detail = " No rollout carried an error: the environment marked every episode invalid."
        if self._empty_rollout_steps < EMPTY_ROLLOUT_STEP_LIMIT:
            if is_global_main_process():
                logger.warning(
                    f"Every rollout in this step failed or was marked invalid — the step contributes no "
                    f"gradient. Halting if the next step is empty too.{detail}"
                )
            return
        raise RuntimeError(
            f"{self._empty_rollout_steps} consecutive steps produced no valid episode anywhere in the "
            f"world, so the run is training on all-masked batches (zero gradient) while logging "
            f"loss=0.{detail}"
        )

    def _narrow_masks_and_normalizer(
        self,
        rows: BatchRows,
        step: StepRows,
        valid_mask: torch.Tensor,
        num_generations: int,
        advantages: torch.Tensor,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(completion_mask, tool_mask, loss_mask, num_items_in_batch)``: the step's excluded trajectories
        (train mode) masked out of both loss masks, a ``negative_only`` row kept only under a negative advantage,
        and the DAPO normalizer, gathered world-wide in every mode."""
        world = self._world_metrics
        completion_mask, negative_only, device = step.completion_mask, step.negative_only, step.completion_mask.device
        (tool_mask,) = narrow_loss_masks(negative_only & (advantages >= 0), step.tool_mask)
        # Masking, not deleting: excluded rows still run the forward so every rank's collective sequence matches.
        if mode == "train":
            # An env-invalid episode's forced failure-reward advantage is noise, so its tokens go too.
            drop_traj = ~valid_mask
            world.fraction("sampling/invalid_episode_frac", drop_traj.sum(), drop_traj.numel())
            if self._drop_degenerate_groups:
                # Judged on the environment's reward: the trainer's reasoning terms are length regularizers,
                # and a group tied on everything else would train on its members' reasoning lengths alone.
                env_rewards = torch.tensor([r.total_reward for r in rows.rollout_results], device=device)
                degenerate = degenerate_group_mask(env_rewards, num_generations, valid_mask=valid_mask)
                drop_traj |= degenerate
                world.fraction(DEGENERATE_GROUP_FRAC_KEY, degenerate.sum(), degenerate.numel())
            if self.args.mask_truncated_completions:
                # TRL enforces this in its own generation path, which this trainer replaces.
                truncated = torch.tensor(
                    [bool(r.trajectory and r.trajectory.truncated) for r in rows.rollout_results], device=device
                )
                drop_traj |= truncated
                world.fraction("sampling/truncated_masked_frac", truncated.sum(), truncated.numel())
            if drop_traj.any():
                # TRL parity: tool_mask narrows with completion_mask, so the normalizer never counts them.
                completion_mask, tool_mask = narrow_loss_masks(rows.to_rows(drop_traj), completion_mask, tool_mask)
        rollout_rows = sum(rows.turns_per_traj[: len(rows.rollout_results)])
        world.fraction("sampling/untrainable_rows_frac", negative_only.sum(), rollout_rows)
        # After every drop: a kept row is one whose tokens reach the normalizer below.
        kept = negative_only & tool_mask.any(dim=1)
        world.fraction("sampling/untrainable_rows_trained_frac", kept.sum(), negative_only.sum())
        # DAPO normalizes by the gathered-global loss-token count / num_processes; a local count mis-scales.
        loss_mask = effective_loss_mask({"completion_mask": completion_mask, "tool_mask": tool_mask})
        num_items_in_batch = gathered_num_items(loss_mask, self.accelerator.gather)
        return completion_mask, tool_mask, loss_mask, num_items_in_batch

    def _apply_token_mass_balance(
        self,
        local_advantages: torch.Tensor,
        traj_advantages: torch.Tensor,
        loss_mask: torch.Tensor,
        ratio: torch.Tensor,
        negative_only: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """COLLECTIVE. Record the step's token mass; under ``balance_token_mass`` both advantage sets come back
        balanced, the per-row ones the loss reads and the per-trajectory ones the completions record reports."""
        balance = record_token_mass(
            local_advantages,
            loss_mask,
            ratio,
            self.accelerator.gather,
            self._metrics["train"],
            self._balance_token_mass,
            negative_only=negative_only,
        )
        if balance is not None:
            local_advantages = balance.apply(local_advantages, negative_only)
            traj_advantages = balance.apply(traj_advantages)
        return local_advantages, traj_advantages

    def _record_step_metrics(
        self,
        rewards: torch.Tensor,
        gathered_rewards: torch.Tensor,
        gathered_state: torch.Tensor,
        rows: BatchRows,
        scores: StepScores,
        loss_mask: torch.Tensor,
        local_advantages: torch.Tensor,
        traj_advantages: torch.Tensor,
        num_generations: int,
        mode: str,
    ) -> torch.Tensor:
        """COLLECTIVE. Record the step's metrics, run the trust-region breaker, write the completions record and
        fold the rank-local counts; returns the per-row advantages the breaker left."""
        rollout_results = rows.rollout_results
        # Padding fills whole groups, so dropping its rows keeps every episode's group aligned.
        episodes = gathered_state >= 0
        self._log_headline_rewards(gathered_rewards[episodes], gathered_state[episodes] == 1, num_generations, mode)
        self._metrics[mode]["sampling/is_correction_active"].append(float(self._is_correction))
        if scores.corrected_mask is not None:
            eff_corrected = self._score_is_correction(scores.ratio, scores.corrected_mask, loss_mask)
            if self._update_breaker_tripped(
                scores.ratio, eff_corrected, scores.traj_row_ids, len(rollout_results), mode
            ):
                # Both: the per-row one carries the zeroed policy gradient, the per-trajectory one is what the
                # completions record writes.
                local_advantages = torch.zeros_like(local_advantages)
                traj_advantages = torch.zeros_like(traj_advantages)
        # After the breaker: the diagnostics and the completions record report what the gradient carries.
        self._record_step_diagnostics(
            rollout_results, num_generations, mode, scores.recompute_logps, local_advantages, loss_mask
        )
        num_episodes = len(rollout_results)
        self._populate_completion_logs(rollout_results, rewards[:num_episodes], traj_advantages[:num_episodes], mode)
        # Every rank arrives here once per step: the one collective over the step's rank-local counts.
        self._world_metrics.flush(self._metrics[mode])
        return local_advantages

    def _log_headline_rewards(
        self, gathered_rewards: torch.Tensor, gathered_valid: torch.Tensor, num_generations: int, mode: str
    ) -> None:
        """``reward``, its global ``reward_std`` and the ``reward/within_group_std`` the advantage reads, over the
        gathered VALID episodes only: an invalid episode's forced failure reward would read a grader outage as
        a policy collapse."""
        valid_rewards = gathered_rewards[gathered_valid]
        if valid_rewards.numel() == 0:
            return
        self._metrics[mode]["reward"].append(valid_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(batch_reward_std(valid_rewards))
        if num_generations > 1 and gathered_rewards.numel() % num_generations == 0:
            self._metrics[mode]["reward/within_group_std"].append(
                valid_group_stats(gathered_rewards, num_generations, gathered_valid)[1].mean().item()
            )

    def _record_step_diagnostics(
        self,
        rollout_results: list[RolloutResult],
        num_generations: int,
        mode: str,
        recompute_logps: torch.Tensor | None,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> None:
        """Record the step's solve-group split over this rank's groups (an episode outside the baseline or
        without a verdict does not vote), an eval round's success@k, and the covariance of each loss token's
        pre-update log-prob with its row's advantage."""
        world = self._world_metrics
        solved = [solve_verdict(r.metrics) if r.counts_toward_baseline else None for r in rollout_results]
        groups, all_pass, all_fail, any_pass = group_solve_counts(solved, num_generations)
        if groups:
            world.fraction("outcome/all_pass_group_frac", all_pass, groups)
            world.fraction("outcome/all_fail_group_frac", all_fail, groups)
            if mode == "eval" and num_generations > 1:
                world.fraction(f"outcome/success@{num_generations}", any_pass, groups)
        if recompute_logps is not None:
            world.covariance(
                "logps/advantage_cov", recompute_logps, advantages.unsqueeze(1).expand_as(recompute_logps), loss_mask
            )
