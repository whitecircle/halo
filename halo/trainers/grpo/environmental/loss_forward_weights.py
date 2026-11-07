"""The importance weight and KL tail clamp of an aligned one-step round, built in the loss forward from its detached
log-probs in place of the no-grad pass, with the metrics that pass fed."""

import torch

from halo.configs.grpo_environmental import ISRMaskConfig
from halo.trainers.grpo.environmental.step_batch import StepRows, StepScores
from halo.trainers.grpo.loss_forward_seam import (
    BALANCE_TOKEN_MASS_BLOCKER,
    LOSS_FORWARD_KNOB,
    LossForwardSeam,
    LossForwardSeamMixin,
    pass_consumer_blocker,
)
from halo.trainers.grpo.objective.logratio import (
    KL_CLAMP_FRAC_KEY,
    LOGRATIO_MEAN_KEY,
    clamp_ref_logps,
    compute_token_is_ratio,
    kl_clamp_counts,
    record_is_ratio_extreme_frac,
    record_is_ratio_survivors,
)
from halo.trainers.grpo.objective.loss_inputs import (
    IMPORTANCE_RATIO_KEY,
    IS_CORRECTED_MASK_KEY,
    IS_FORCED_ZERO_KEY,
    IS_SAMPLING_LOGPS_KEY,
    NEGATIVE_ONLY_ROWS_KEY,
    REF_LOGPS_KEY,
    TOKEN_MASS_SUM_KEYS,
    UNCLAMPED_REF_LOGPS_KEY,
    record_world_token_mass,
    token_mass_sums,
)
from halo.trainers.token_metrics import effective_loss_mask


def loss_forward_weight_blocker(
    *,
    runs_pass: bool,
    num_iterations: int,
    misaligned_accumulation: bool,
    isr_masks: ISRMaskConfig,
    engine_reference: bool,
    skip_update_masked_frac: float | None,
    balance_token_mass: bool,
    routing_replay: str,
) -> str | None:
    """Why the round's weight cannot come from the loss forward, or ``None``: every consumer of the trainer's
    log-probs before the first backward is named."""
    shared = pass_consumer_blocker(
        pass_off=None if runs_pass else "the IS correction and the KL term are both off",
        num_iterations=num_iterations,
        misaligned_accumulation=misaligned_accumulation,
    )
    if shared is not None:
        return shared
    if isr_masks.any_stage_active:
        return (
            "the isr_* mask stages read each trajectory's mean log-ratio over rows that run in different "
            "micro-batches, which only a pass before the loss supplies"
        )
    if engine_reference:
        return "isr_engine_reference compares the engine's and the trainer's log-ratio per trajectory before the loss"
    if skip_update_masked_frac is not None:
        return (
            "skip_update_masked_frac decides from the round's global masked fractions before the first optimizer step"
        )
    if balance_token_mass:
        return BALANCE_TOKEN_MASS_BLOCKER
    if routing_replay == "recompute":
        return (
            "routing_replay: recompute pins the update to the no-grad pass's routing, and with one forward there is "
            "nothing to pin to (use none or rollout)"
        )
    return None


class LossForwardImportanceMixin(LossForwardSeamMixin):
    """Mixed in ahead of ``ProtectedTokenEntropyMixin``, whose scorer override returns through ``super()``, and of
    ``RowSpanStashMixin``. The host calls :meth:`_configure_loss_forward_weights` once its config and routing replay are
    resolved, and :meth:`_with_loss_forward_inputs` on the last statement of a fused train round's batch build."""

    def _configure_loss_forward_weights(self) -> None:
        """Take the weight from the loss forward when the knob is set, refusing it where something needs the round's
        log-probs earlier."""
        cfg = self.async_config
        if not getattr(cfg, LOSS_FORWARD_KNOB):
            return
        self._adopt_loss_forward_weights(
            loss_forward_weight_blocker(
                runs_pass=self._runs_logprob_recompute,
                num_iterations=self.num_iterations,
                misaligned_accumulation=self._misaligned_accumulation,
                isr_masks=self._isr_mask_config,
                engine_reference=cfg.isr_engine_reference,
                skip_update_masked_frac=cfg.skip_update_masked_frac,
                balance_token_mass=cfg.balance_token_mass,
                routing_replay=cfg.routing_replay,
            )
        )

    def _uses_loss_forward_weights(self, mode: str) -> bool:
        """Whether this mode's rounds take the weight from the loss forward. Eval keeps the pass."""
        return self._skips_logprob_pass and mode == "train"

    def _with_loss_forward_inputs(self, result: dict, step: StepRows, scores: StepScores) -> dict:
        """``result`` with the weight's sampling-side inputs in place of the weight, and the reference log-probs
        unclamped, for the loss forward to complete. The weight is read by TRL only after that forward, so a
        batch that still carried one would fail there as a missing key, not train on a stale one."""
        result[NEGATIVE_ONLY_ROWS_KEY] = step.negative_only
        if self.beta != 0.0:
            result[UNCLAMPED_REF_LOGPS_KEY] = result.pop(REF_LOGPS_KEY)
        if self._is_correction:
            del result[IMPORTANCE_RATIO_KEY]
            result[IS_SAMPLING_LOGPS_KEY] = step.sampling_logps
            result[IS_CORRECTED_MASK_KEY] = scores.corrected_mask
            if scores.forced_zero is not None:
                result[IS_FORCED_ZERO_KEY] = scores.forced_zero
        return result

    def _weigh_from_loss_forward(self, seam: LossForwardSeam, logps: torch.Tensor) -> None:
        """Write the weight and the clamped reference into ``seam`` from the loss forward's detached ``logps``, and
        record the metrics. Detached, as a weight that carried a graph would add ``A·logπ·∂w/∂θ`` to the gradient."""
        loss_mask = effective_loss_mask(seam).bool()
        corrected = logps_diff = None
        if self._is_correction:
            corrected = seam[IS_CORRECTED_MASK_KEY]
            ratio, logps_diff = compute_token_is_ratio(
                logps, seam[IS_SAMPLING_LOGPS_KEY], corrected, self.vllm_importance_sampling_clip_max
            )
            forced = seam.get(IS_FORCED_ZERO_KEY)
            if forced is not None:
                ratio = ratio.masked_fill(forced, 0.0)
            seam[IMPORTANCE_RATIO_KEY] = ratio
        else:
            ratio = seam[IMPORTANCE_RATIO_KEY]
        self._record_loss_forward_metrics(seam, logps, loss_mask, ratio, logps_diff, corrected)
        if self.beta != 0.0:
            seam[REF_LOGPS_KEY], clamped = clamp_ref_logps(seam[UNCLAMPED_REF_LOGPS_KEY], logps)
            self._loss_forward_metrics.fraction(KL_CLAMP_FRAC_KEY, *kl_clamp_counts(clamped, loss_mask))

    def _record_loss_forward_metrics(
        self,
        seam: LossForwardSeam,
        logps: torch.Tensor,
        loss_mask: torch.Tensor,
        ratio: torch.Tensor,
        logps_diff: torch.Tensor | None,
        corrected: torch.Tensor | None,
    ) -> None:
        """This micro-batch's additive share of the metrics the pass fed, over loss-mask tokens, on the device."""
        world = self._loss_forward_metrics
        advantages = seam["advantages"]
        world.covariance("logps/advantage_cov", logps, advantages.unsqueeze(1).expand_as(logps), loss_mask)
        masses = token_mass_sums(advantages, loss_mask, ratio, seam[NEGATIVE_ONLY_ROWS_KEY])
        for key, mass in zip(TOKEN_MASS_SUM_KEYS, masses.unbind(), strict=True):
            world.sum(key, mass)
        if corrected is None:
            return
        counted = corrected & loss_mask
        # A mean, not a fraction: a round with no loss token logs no ratio, as the pass does, not a 0.0.
        num_counted = counted.sum()
        world.mean(LOGRATIO_MEAN_KEY, (logps_diff * counted).sum() / num_counted.clamp(min=1), num_counted)
        record_is_ratio_extreme_frac(world, logps_diff, counted, self.vllm_importance_sampling_clip_max)
        record_is_ratio_survivors(world, ratio, counted)

    def _flush_loss_forward_metrics(self) -> None:
        """COLLECTIVE. The round's accumulated sums folded across the ranks and logged; a non-finite token mass raises
        on every rank."""
        logged = self._metrics["train"]
        world = self._loss_forward_metrics.flush()
        masses = [world.pop(key, None) for key in TOKEN_MASS_SUM_KEYS]
        for key, value in world.items():
            logged[key].append(value)
        if masses[0] is not None:
            record_world_token_mass(masses, logged, enabled=False, with_negative_only=True)
