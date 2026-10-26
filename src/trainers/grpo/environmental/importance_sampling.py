"""The engine-to-trainer importance-sampling correction of environmental GRPO and its trust-region breaker."""

import dataclasses
import math
import weakref
from concurrent.futures import ThreadPoolExecutor

import torch
from accelerate.logging import get_logger
from transformers import TrainerCallback

from src.runtime.ranks import get_global_rank
from src.trainers.grpo.early_stop import build_early_stop_callback
from src.trainers.grpo.environmental.environmental_rules import (
    reject_inert_is_stages,
    reject_unneutralized_forced_closes,
    reject_unsupported_engine_reference,
)
from src.trainers.grpo.environmental.step_batch import BatchRows, StepRows, StepScores
from src.trainers.grpo.environmental.trajectory_tokenize import TurnRow
from src.trainers.grpo.objective.logratio import (
    LOGRATIO_MEAN_KEY,
    UPDATE_SKIPPED_KEY,
    apply_is_masks,
    apply_opsm,
    compute_is_ratio,
    sampled_token_mask,
    sampler_certain_mask,
    select_mask_logratio,
    zero_engine_forced_closes,
)
from src.trainers.world_metrics import world_sums
from src.weight_sync.registry import resolve_weight_sync_client

logger = get_logger(__name__, log_level="INFO")

# Concurrent re-score prefills per rank under isr_engine_reference (each is one HTTP request in flight).
_ENGINE_RESCORE_CONCURRENCY = 16


class _BreakerOptimizerSkipCallback(TrainerCallback):
    """Skips the optimizer step while the trust-region breaker is tripped, at the one hook between gradient
    clipping and ``optimizer.step``; weak, since the trainer owns the handler that owns this."""

    def __init__(self, trainer):
        self._trainer = weakref.ref(trainer)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Set every gradient to ``None``, which makes an optimizer skip the parameter (a zero gradient still
        steps Adam on momentum). The flag stays up for every optimizer step the round feeds."""
        trainer = self._trainer()
        if trainer is not None and trainer._breaker_tripped_this_step:
            trainer.optimizer.zero_grad(set_to_none=True)


def _masked_fraction_counts(
    ratio: torch.Tensor, corrected: torch.Tensor, traj_row_ids: torch.Tensor, num_trajectories: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The trust-region breaker's two ``(masked, total)`` counts: the IS-corrected trajectories whose every
    corrected token had its ratio zeroed, and the corrected tokens zeroed. Padding and dummy rows
    (``traj_row_ids`` -1) count in neither."""
    rows_valid = traj_row_ids >= 0
    corrected_tokens = corrected & rows_valid.unsqueeze(1)
    masked_tokens = corrected_tokens & (ratio <= 0)
    row_corrected = corrected.any(dim=1) & rows_valid
    row_surviving = ((ratio > 0) & corrected).any(dim=1) & rows_valid
    traj_corrected = torch.zeros(num_trajectories, dtype=torch.bool, device=ratio.device)
    traj_surviving = torch.zeros(num_trajectories, dtype=torch.bool, device=ratio.device)
    traj_corrected[traj_row_ids[row_corrected]] = True
    traj_surviving[traj_row_ids[row_surviving]] = True
    masked = traj_corrected & ~traj_surviving
    return [(masked.sum(), traj_corrected.sum()), (masked_tokens.sum(), corrected_tokens.sum())]


class ImportanceSamplingMixin:
    """The engine-to-trainer importance ratio, its mask stages and the trust-region breaker."""

    def _arm_update_breaker(self) -> None:
        """Take ``skip_update_masked_frac`` and, with it set, attach the hook that acts on the breaker's
        per-round verdict (:meth:`_update_breaker_tripped`)."""
        self._breaker_tripped_this_step = False
        if self.async_config.skip_update_masked_frac is not None:
            self.add_callback(_BreakerOptimizerSkipCallback(self))

    def _configure_importance_sampling(self) -> None:
        """The importance-sampling correction the config asks for, scored at the sampling temperature, refusing
        every stage it would leave inert."""
        cfg = self.async_config
        # Score log-probs at the engine's sampling temperature, or the IS ratio picks up a systematic bias.
        if self.temperature != cfg.rollout_temperature:
            logger.info(
                f"Scoring log-probs at the sampling temperature: temperature "
                f"{self.temperature} -> rollout_temperature {cfg.rollout_temperature}"
            )
            self.temperature = cfg.rollout_temperature
            self.args.temperature = cfg.rollout_temperature
        # Truncated per-token ratio vs the captured sampling logprobs, so it needs train_on_sampled_tokens.
        self._is_correction = cfg.train_on_sampled_tokens and self.vllm_importance_sampling_correction
        if self.vllm_importance_sampling_correction and not cfg.train_on_sampled_tokens:
            logger.warning(
                "vllm_importance_sampling_correction is ON but train_on_sampled_tokens is off, so the "
                "correction is DISABLED: the ratio needs the sampling log-probs the server returns "
                "alongside the sampled tokens. Every batch then trains uncorrected for the engine-vs-trainer "
                "numerics gap, and for a weight-sync lag where one exists (enable_prefetch, "
                "sync_weights_every_n_steps > 1). Set train_on_sampled_tokens: true (a vLLM server needs "
                "--return-tokens-as-token-ids), or set the correction to false deliberately."
            )
        reject_unneutralized_forced_closes(self._forced_close_ids, self._is_correction)
        reject_inert_is_stages(self._is_mask_config, self._is_correction, cfg.skip_update_masked_frac)
        early_stop = build_early_stop_callback(
            cfg.build_early_stop_config(), gap_key=LOGRATIO_MEAN_KEY, gap_logged=self._is_correction
        )
        if early_stop is not None:
            self.add_callback(early_stop)
        if cfg.isr_engine_reference:
            reject_unsupported_engine_reference(
                resolve_weight_sync_client(self._rollout_backend), cfg, self._is_correction
            )
        if self._is_correction:
            self.use_vllm = True

    def _rescore_rows_on_engine(self, rows: list[TurnRow], turns_per_traj: list[int]) -> list[torch.Tensor | None]:
        """Every sampled row's completion re-scored on the rollout engine under this step's weights, a
        trajectory's rows on one server so its turns share the prefix cache; ``None`` where a row carries no
        sampling log-probs or its request failed. Never raises: a rank-local raise would desync the collectives
        that follow, so a miss is reported and its row reads the trainer's log-ratio.
        """
        clients = self._engine_rescore_clients
        if not clients:
            raise RuntimeError(
                "isr_engine_reference re-scored a batch before the rollout components started: no score-only "
                "client was built (_init_async_components builds them on every rank)."
            )
        row_traj = [traj for traj, turns in enumerate(turns_per_traj) for _ in range(turns)]
        indices = [i for i, row in enumerate(rows) if row.sampling_logps is not None]

        def score(i: int) -> list[float]:
            client = clients[row_traj[i] % len(clients)]
            return client.score_completion_logprobs(rows[i].prompt_ids.tolist(), rows[i].completion_ids.tolist())

        scored: list[torch.Tensor | None] = [None] * len(rows)
        failures: list[Exception] = []
        with ThreadPoolExecutor(max_workers=_ENGINE_RESCORE_CONCURRENCY) as pool:
            futures = {i: pool.submit(score, i) for i in indices}
            for i, future in futures.items():
                completion_ids = rows[i].completion_ids
                try:
                    values = future.result()
                    if len(values) != completion_ids.numel():
                        raise ValueError(f"row {i}: {len(values)} log-probs for {completion_ids.numel()} tokens")
                    scored[i] = torch.tensor(values, dtype=torch.float32, device=completion_ids.device)
                except Exception as e:  # noqa: BLE001 — a transport/shape failure on one row is that row's miss
                    failures.append(e)
        self._world_metrics.fraction("sampling/engine_rescore_miss_frac", len(failures), len(indices))
        if failures:
            report = logger.error if len(failures) == len(indices) else logger.warning
            # main_process_only=False: the failure is this rank's own requests, invisible to rank 0.
            report(
                f"[rank {get_global_rank()}] isr_engine_reference: {len(failures)}/{len(indices)} re-score "
                f"requests failed this step (those rows read the trainer's log-ratio instead); last error: "
                f"{failures[-1]}",
                main_process_only=False,
            )
        return scored

    def _apply_is_correction(
        self,
        recompute_logps: torch.Tensor | None,
        step: StepRows,
        rows: BatchRows,
        engine_logps: list[torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """``(ratio, logps_diff, corrected_mask, traj_row_ids)``: the rollout-vs-trainer importance ratio after its
        mask stages, all-ones with the rest ``None`` when the correction is off. ``logps_diff`` is what the mask
        stages read, the engine's diff on re-scored rows under ``isr_engine_reference``; the weight itself is
        always trainer-vs-sampling."""
        importance_sampling_ratio = torch.ones_like(step.completion_mask, dtype=torch.float32)
        corrected_mask = None
        logps_diff = None
        traj_row_ids = None
        if self._is_correction:
            importance_sampling_ratio, logps_diff, corrected_mask = compute_is_ratio(
                recompute_logps,
                step.sampling_logps,
                step.completion_mask,
                step.has_sampling,
                self.vllm_importance_sampling_clip_max,
            )
            # Unclamped mean log-ratio (nats): ~0 when conditioning matches the engine's exact prompt.
            self._world_metrics.fraction(LOGRATIO_MEAN_KEY, logps_diff.sum(), corrected_mask.sum())
            clip_max = self.vllm_importance_sampling_clip_max
            # Past the truncation point in either direction, on the raw trainer-vs-sampler ratio; the
            # band [1/clip_max, clip_max] is empty at or below 1.
            if clip_max > 1:
                extreme = corrected_mask & (logps_diff.abs() > math.log(clip_max))
                self._world_metrics.fraction("sampling/is_ratio_extreme_frac", extreme.sum(), corrected_mask.sum())
            # Policy tokens the sampler emitted with probability 1 (budget-forced closes): uncorrected.
            certain = sampler_certain_mask(step.sampling_logps, step.completion_mask, step.has_sampling)
            sampled = sampled_token_mask(step.completion_mask, step.has_sampling)
            self._world_metrics.fraction("sampling/sampler_certain_frac", certain.sum(), sampled.sum())
            traj_row_ids = rows.traj_row_ids(step.completion_mask.device)
            if engine_logps is not None:
                logps_diff = self._engine_mask_logratio(
                    logps_diff, recompute_logps, step, corrected_mask, engine_logps
                )
            if self._is_mask_config.any_mask_active:
                importance_sampling_ratio, mask_stats = apply_is_masks(
                    importance_sampling_ratio, logps_diff, corrected_mask, traj_row_ids, self._is_mask_config
                )
                for key, (masked, total) in mask_stats.items():
                    self._world_metrics.fraction(key, masked, total)
        return importance_sampling_ratio, logps_diff, corrected_mask, traj_row_ids

    def _engine_mask_logratio(
        self,
        logps_diff: torch.Tensor,
        recompute_logps: torch.Tensor,
        step: StepRows,
        corrected_mask: torch.Tensor,
        engine_logps: list[torch.Tensor | None],
    ) -> torch.Tensor:
        """The log-ratio the mask stages read under ``isr_engine_reference``: the engine's current-vs-sampling
        diff on every re-scored row, the trainer diff elsewhere (:func:`select_mask_logratio`)."""
        scored_logps = torch.zeros_like(step.sampling_logps)
        for i, scored in enumerate(engine_logps):
            if scored is not None:
                scored_logps[i, : scored.numel()] = scored
        has_engine = torch.tensor([scored is not None for scored in engine_logps], device=scored_logps.device)
        logps_diff, engine_stats = select_mask_logratio(
            logps_diff, recompute_logps, step.sampling_logps, scored_logps, corrected_mask, has_engine
        )
        for key, (numerator, denominator) in engine_stats.items():
            self._world_metrics.fraction(key, numerator, denominator)
        return logps_diff

    def _zero_forced_closes(self, ratio: torch.Tensor, step: StepRows) -> torch.Tensor:
        """``ratio`` with every reasoning close the engine forced at a thinking cap set to 0, so none trains with
        its episode's advantage."""
        ratio, forced = zero_engine_forced_closes(
            ratio,
            step.sampling_logps,
            step.completion_mask,
            step.has_sampling,
            step.completion_ids,
            self._forced_close_ids,
        )
        # Over the tokens that carry sampling log-probs: no other token can be a forced close.
        sampled = sampled_token_mask(step.completion_mask, step.has_sampling)
        self._world_metrics.fraction("sampling/forced_close_frac", forced.sum(), sampled.sum())
        return ratio

    def _apply_opsm_stage(self, scores: StepScores, advantages: torch.Tensor) -> StepScores:
        """``scores`` with the OPSM stage on their IS ratio, where it is configured: it masks negative-advantage
        trajectories only, so it runs once the per-row ``advantages`` exist."""
        delta = self._is_mask_config.opsm_delta
        if delta is not None and scores.corrected_mask is not None:
            ratio, opsm_masked = apply_opsm(
                scores.ratio, scores.logps_diff, scores.corrected_mask, scores.traj_row_ids, advantages, delta
            )
            self._world_metrics.fraction("sampling/is_opsm_masked_frac", opsm_masked.sum(), opsm_masked.numel())
            scores = dataclasses.replace(scores, ratio=ratio)
        return scores

    def _score_is_correction(
        self,
        importance_sampling_ratio: torch.Tensor,
        corrected_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """The drop-narrowed corrected mask the trust-region breaker reads; also records the step's
        IS-correction coverage and surviving-ratio metrics."""
        # corrected_mask predates the drop narrowing; intersect so coverage cannot exceed 1.
        eff_corrected = corrected_mask & loss_mask
        world = self._world_metrics
        world.fraction("sampling/is_correction_coverage", eff_corrected.sum(), loss_mask.sum())
        # Masked ratios are zeroed in place, so a raw mean collapses as masking grows; report survivors.
        ratios = importance_sampling_ratio[eff_corrected]
        surviving = ratios[ratios > 0]
        world.fraction("sampling/is_masked_frac", ratios.numel() - surviving.numel(), ratios.numel())
        if surviving.numel():
            world.fraction("sampling/is_ratio_mean", surviving.sum(), surviving.numel())
            world.maximum("sampling/is_ratio_max", surviving.max())
        # Over the weights the loss applies, masked ones as zeros: a thrown-away token is a lost sample.
        world.effective_sample_frac("sampling/is_ess_frac", importance_sampling_ratio, eff_corrected)
        return eff_corrected

    def _update_breaker_tripped(
        self,
        importance_sampling_ratio: torch.Tensor,
        eff_corrected: torch.Tensor,
        traj_row_ids: torch.Tensor | None,
        num_trajectories: int,
        mode: str,
    ) -> bool:
        """Whether the trust-region breaker (``skip_update_masked_frac``) trips: the GLOBAL share of fully masked
        corrected trajectories or of masked corrected tokens is over the threshold, so the survivors are a
        selection-biased sample that would amplify the drift. On a trip the caller zeroes both advantage sets
        and every optimizer step of the round is skipped; the next sync re-anchors the rollouts."""
        threshold = self.async_config.skip_update_masked_frac
        if threshold is None or mode != "train" or traj_row_ids is None:
            return False
        counts = _masked_fraction_counts(importance_sampling_ratio, eff_corrected, traj_row_ids, num_trajectories)
        traj_masked, traj_total, token_masked, token_total = world_sums([count for pair in counts for count in pair])
        traj_frac = traj_masked / traj_total if traj_total else 0.0
        token_frac = token_masked / token_total if token_total else 0.0
        self._metrics[mode]["sampling/is_masked_traj_frac"].append(traj_frac)
        self._metrics[mode]["sampling/is_masked_token_frac"].append(token_frac)
        skipped = max(traj_frac, token_frac) > threshold
        self._metrics[mode][UPDATE_SKIPPED_KEY].append(float(skipped))
        # The flag stays up for every optimizer step the round feeds and comes down only here.
        self._breaker_tripped_this_step = skipped
        if not skipped:
            return False
        logger.warning(
            f"IS trust region: {traj_frac:.0%} of corrected trajectories fully masked, {token_frac:.0%} of "
            f"corrected tokens masked (> skip_update_masked_frac={threshold}) — zeroing "
            "this step's policy gradient and skipping its optimizer step; rollouts re-anchor at the next weight sync."
        )
        return True
