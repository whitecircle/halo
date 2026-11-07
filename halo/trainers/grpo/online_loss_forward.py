"""Online GRPO's vLLM importance-sampling ratio and KL tail clamp, built in the loss forward from its detached log-probs
in place of TRL's no-grad old-policy pass, with the metrics that pass fed."""

import contextlib
import math
from collections.abc import Iterator

import torch

from halo.trainers.grpo.loss_forward_seam import (
    BALANCE_TOKEN_MASS_BLOCKER,
    LOSS_FORWARD_KNOB,
    LossForwardSeam,
    LossForwardSeamMixin,
    pass_consumer_blocker,
)
from halo.trainers.grpo.objective.logratio import KL_CLAMP_FRAC_KEY, clamp_ref_logps, kl_clamp_counts
from halo.trainers.grpo.objective.loss_inputs import (
    IMPORTANCE_RATIO_KEY,
    REF_LOGPS_KEY,
    TOKEN_MASS_SUM_KEYS,
    record_world_token_mass,
    token_mass_sums,
)
from halo.trainers.token_metrics import effective_loss_mask

SAMPLING_LOGPS_KEY = "sampling_per_token_logps"

# The positive and negative token masses; a negative-only mass is the environmental trainer's.
_MASS_KEYS = TOKEN_MASS_SUM_KEYS[:2]

# The statistics TRL logs of the round's log-prob gap and of its ratio, by metric stem.
_SAMPLING_STATS = {
    "sampling/sampling_logp_difference": ("mean", "max"),
    "sampling/importance_sampling_ratio": ("min", "mean", "max"),
}


def online_loss_forward_blocker(
    *, runs_pass: bool, num_iterations: int, misaligned_accumulation: bool, balance_token_mass: bool
) -> str | None:
    """Why the round's ratio cannot come from the loss forward, or ``None``: every consumer of the trainer's log-probs
    before the first backward is named."""
    shared = pass_consumer_blocker(
        runs_pass=runs_pass, num_iterations=num_iterations, misaligned_accumulation=misaligned_accumulation
    )
    if shared is None and balance_token_mass:
        return BALANCE_TOKEN_MASS_BLOCKER
    return shared


class OnlineLossForwardMixin(LossForwardSeamMixin):
    """Mixed in ahead of ``ProtectedTokenEntropyMixin`` and of ``RowSpanStashMixin``, like the environmental trainer's.
    The host sets ``_requests_loss_forward_weights``, calls :meth:`_configure_loss_forward_weights` once TRL's
    constructor has run, and builds each train round inside :meth:`_without_trl_old_pass`."""

    _requests_loss_forward_weights = False

    def _configure_loss_forward_weights(self) -> None:
        """Take the ratio from the loss forward when the knob is set, refusing it where something needs the round's
        log-probs earlier."""
        if not self._requests_loss_forward_weights:
            return
        self._adopt_loss_forward_weights(
            online_loss_forward_blocker(
                runs_pass=self._is_correction,
                num_iterations=self.num_iterations,
                misaligned_accumulation=self._misaligned_accumulation,
                balance_token_mass=self._balance_token_mass,
            )
        )

    @property
    def _is_correction(self) -> bool:
        return self.use_vllm and self.vllm_importance_sampling_correction

    @property
    def _fuses_round(self) -> bool:
        """Whether the round being built takes its ratio from the loss forward. Eval keeps the pass."""
        return self._skips_logprob_pass and self.model.training

    @contextlib.contextmanager
    def _without_trl_old_pass(self) -> Iterator[None]:
        """TRL's old-policy pass runs under its importance-sampling gate, which stays off while a fused round is
        built, so the round carries the engine's log-probs and no ratio, and logs none of the gap statistics."""
        if not self._fuses_round:
            yield
            return
        self.vllm_importance_sampling_correction = False
        try:
            yield
        finally:
            self.vllm_importance_sampling_correction = True

    def _loss_forward_written_keys(self) -> frozenset[str]:
        written = {IMPORTANCE_RATIO_KEY} if self._is_correction else set()
        if self.beta != 0.0:
            written.add(REF_LOGPS_KEY)
        return frozenset(written)

    def _weigh_from_loss_forward(self, seam: LossForwardSeam, logps: torch.Tensor) -> None:
        """Write the ratio and the clamped reference into ``seam`` from the loss forward's detached ``logps``, and
        record the metrics. Detached, as a ratio that carried a graph would add ``A·logπ·∂w/∂θ`` to the gradient."""
        sampling = seam.get(SAMPLING_LOGPS_KEY)
        if sampling is None:
            raise RuntimeError(
                "The batch carries no sampling_per_token_logps, so the importance ratio cannot be built in the loss "
                f"forward: the rollout returned no engine log-probs. Set {LOSS_FORWARD_KNOB}: false."
            )
        loss_mask = effective_loss_mask(seam)
        delta = (logps - sampling) * loss_mask
        mode = self.vllm_importance_sampling_mode
        sequence_level = mode.startswith("sequence")
        # A token whose engine log-prob is NaN is uncorrected: its difference is zero, so its ratio is 1.
        diff = torch.nan_to_num(delta, nan=0.0)
        if sequence_level:
            diff = diff.sum(dim=-1, keepdim=True)
        ratio = self._clip_ratio(torch.exp(diff), mode)
        seam[IMPORTANCE_RATIO_KEY] = ratio
        self._record_loss_forward_metrics(seam, logps, sampling, loss_mask.bool(), ratio, sequence_level)
        if self.beta != 0.0:
            seam[REF_LOGPS_KEY], clamped = clamp_ref_logps(seam.peek(REF_LOGPS_KEY), logps)
            self._loss_forward_metrics.fraction(KL_CLAMP_FRAC_KEY, *kl_clamp_counts(clamped, loss_mask))

    def _clip_ratio(self, ratio: torch.Tensor, mode: str) -> torch.Tensor:
        """TRL's truncation or masking of the ratio at ``vllm_importance_sampling_clip_min`` / ``_max``."""
        low, high = self.vllm_importance_sampling_clip_min, self.vllm_importance_sampling_clip_max
        if mode.endswith("truncate"):
            return torch.clamp(ratio, min=low, max=high)
        if mode.endswith("mask"):
            low = -math.inf if low is None else low
            high = math.inf if high is None else high
            return ratio.masked_fill((ratio < low) | (ratio > high), 0.0)
        raise ValueError(f"Unknown vLLM importance sampling mode {mode!r}.")

    def _record_loss_forward_metrics(
        self,
        seam: LossForwardSeam,
        logps: torch.Tensor,
        sampling: torch.Tensor,
        loss_mask: torch.Tensor,
        ratio: torch.Tensor,
        sequence_level: bool,
    ) -> None:
        """This micro-batch's additive share of the metrics the pass fed, over loss-mask tokens (rows, at sequence
        level), on the device."""
        world = self._loss_forward_metrics
        gap = (logps - sampling).abs()
        world.moments("sampling/sampling_logp_difference", gap, loss_mask & ~gap.isnan())
        counted = loss_mask.any(dim=-1, keepdim=True) if sequence_level else loss_mask
        world.moments("sampling/importance_sampling_ratio", ratio, counted)
        advantages = seam["advantages"]
        masses = token_mass_sums(advantages, loss_mask.to(advantages.dtype), ratio)
        for key, mass in zip(_MASS_KEYS, masses.unbind(), strict=True):
            world.sum(key, mass)

    def _flush_loss_forward_metrics(self) -> None:
        """COLLECTIVE. The round's accumulated sums folded across the ranks and logged; a non-finite token mass raises
        on every rank."""
        logged = self._metrics["train"]
        world = self._loss_forward_metrics.flush()
        masses = [world.pop(key, None) for key in _MASS_KEYS]
        for stem, stats in _SAMPLING_STATS.items():
            for stat in stats:
                if (value := world.get(f"{stem}_{stat}")) is not None:
                    logged[f"{stem}/{stat}"].append(value)
        if KL_CLAMP_FRAC_KEY in world:
            logged[KL_CLAMP_FRAC_KEY].append(world[KL_CLAMP_FRAC_KEY])
        if masses[0] is not None:
            record_world_token_mass(masses, logged, enabled=False, with_negative_only=False)
