"""The seam through which a GRPO loss forward completes its own batch with the importance weight, in place of the
no-grad pass that would have supplied it, and the refusals every trainer taking that seam shares."""

import logging
from collections.abc import Mapping

import torch

from halo.runtime.ranks import is_global_main_process
from halo.trainers.grpo.objective.loss_inputs import IMPORTANCE_RATIO_KEY, REF_LOGPS_KEY
from halo.trainers.world_metrics import WorldMetrics

logger = logging.getLogger(__name__)

LOSS_FORWARD_KNOB = "loss_forward_importance_weights"

BALANCE_TOKEN_MASS_BLOCKER = (
    "balance_token_mass needs the round's global importance-weighted token mass before the first backward"
)


def pass_consumer_blocker(*, pass_off: str | None, num_iterations: int, misaligned_accumulation: bool) -> str | None:
    """Why the round's weight cannot come from the loss forward whatever else reads the log-probs, or ``None``.
    ``pass_off`` is the trainer's reason no log-prob pass runs on the config, ``None`` where one does."""
    if pass_off is not None:
        return f"no log-prob pass runs on this config ({pass_off}), so there is nothing to fuse"
    if num_iterations > 1:
        return f"num_iterations={num_iterations} needs the round-start log-probs as the PPO ratio's denominator"
    if misaligned_accumulation:
        return (
            "gradient_accumulation_steps straddles a generation round, so later micro-batches run on moved weights "
            "while the weight needs the round-start policy"
        )
    return None


class LossForwardSeam(dict):
    """A loss forward's batch, completed by the scoring hook with the keys TRL reads after the forward. It records
    those reads and the hook's firings, so a TRL that stops reading a key, or a forward that bypasses the hook, raises
    in place of training an unweighted loss."""

    def __init__(self, inputs: Mapping, written: frozenset[str]):
        super().__init__(inputs)
        self._written = written
        self._read: set[str] = set()
        self.fired = 0

    def __getitem__(self, key):
        self._note(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._note(key)
        return super().get(key, default)

    def peek(self, key):
        """``key``'s value, not counted as TRL's read of it."""
        return super().__getitem__(key)

    def _note(self, key: str) -> None:
        if key in self._written:
            self._read.add(key)

    def verify(self) -> None:
        if self.fired != 1:
            raise RuntimeError(
                f"The loss forward scored {self.fired} time(s) with entropy, where the importance weight is built from "
                f"exactly one: the scorer was bypassed or TRL's loss forward changed. Set {LOSS_FORWARD_KNOB}: false "
                f"until the hook follows it."
            )
        unread = sorted(self._written - self._read)
        if unread:
            raise RuntimeError(
                f"TRL's loss never read {unread} after the loss forward wrote them, so the weight or KL clamp went "
                f"unused: TRL moved where it reads them. Set {LOSS_FORWARD_KNOB}: false until the hook follows it."
            )


class LossForwardSeamMixin:
    """Mixed in ahead of ``ProtectedTokenEntropyMixin``, whose scorer override returns through ``super()``, and of
    ``RowSpanStashMixin``. The host provides ``_is_correction`` and ``beta``, defines :meth:`_weigh_from_loss_forward`
    and :meth:`_flush_loss_forward_metrics`, and adopts the seam through :meth:`_adopt_loss_forward_weights`."""

    _skips_logprob_pass = False
    _loss_forward_seam: LossForwardSeam | None = None
    _loss_forward_metrics: WorldMetrics | None = None

    def _adopt_loss_forward_weights(self, blocker: str | None) -> None:
        """Raises where ``blocker`` names a consumer of the round's log-probs ahead of the loss; else takes the weight
        from the loss forward."""
        if blocker is not None:
            raise ValueError(f"{LOSS_FORWARD_KNOB}: true, but {blocker}. Set it to false.")
        self._skips_logprob_pass = True
        self._loss_forward_metrics = WorldMetrics()
        if is_global_main_process():
            logger.info("Importance weights come from the loss forward; no separate log-prob pass runs.")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """A fused train micro-batch's loss runs on a copy of ``inputs`` the scoring hook completes, and raises unless
        the hook fired once and TRL read what it wrote."""
        if not (self._skips_logprob_pass and model.training and isinstance(inputs, dict)):
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        seam = LossForwardSeam(inputs, self._loss_forward_written_keys())
        self._loss_forward_seam = seam
        try:
            loss = super().compute_loss(
                model, seam, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        finally:
            self._loss_forward_seam = None
        seam.verify()
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """COLLECTIVE on a fused round's last micro-step: the round's metrics that need the trainer's log-probs fold
        into one flush, ahead of the optimizer step the round feeds."""
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self._skips_logprob_pass and self._step % self.args.steps_per_generation == 0:
            self._flush_loss_forward_metrics()
        return loss

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, *args, **kwargs):
        """TRL's one grad-enabled scoring call with entropy is the loss forward; the other forwards (no-grad passes, the
        reference, warm-ups) leave the batch alone."""
        scored = super()._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, *args, **kwargs
        )
        seam = self._loss_forward_seam
        if seam is not None and torch.is_grad_enabled() and kwargs.get("compute_entropy"):
            seam.fired += 1
            if seam.fired == 1:
                self._weigh_from_loss_forward(seam, scored[0].detach())
        return scored

    def _loss_forward_written_keys(self) -> frozenset[str]:
        """The keys the scoring hook writes into a fused batch for TRL's loss to read: the weight under the IS
        correction, the clamped reference under a KL term."""
        written = {IMPORTANCE_RATIO_KEY} if self._is_correction else set()
        if self.beta != 0.0:
            written.add(REF_LOGPS_KEY)
        return frozenset(written)
