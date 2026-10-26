"""The privileged-teacher OPD term both SDPG arms add to their base objective (arXiv:2606.04036): its
weight schedule, the frozen teacher pass and the mixin that adopts its tunables."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from src.configs.distillation import SDPGArguments
from src.trainers.distillation.divergences import require_divergence


def beta_warmup_decay(
    step: int,
    total_steps: int,
    beta_base: float,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    """SDPG distillation-coefficient schedule (warmup → hold → linear decay).

    ``beta(k) = beta_base * min(1, k / T_warm) * min(1, (T - k) / T_decay)``. Constant ``beta_base``
    when ``warmup_steps=decay_steps=0``.
    """
    if beta_base == 0.0:
        return 0.0
    warm = 1.0 if warmup_steps <= 0 else min(1.0, step / warmup_steps)
    # An unset total (<=0) must not zero OPD for the whole run.
    decay_inactive = decay_steps <= 0 or total_steps <= 0
    decay = 1.0 if decay_inactive else min(1.0, max(0.0, (total_steps - step) / decay_steps))
    return beta_base * warm * decay


@contextmanager
def privileged_teacher_pass(model: torch.nn.Module) -> Iterator[None]:
    """Run the same model as the frozen privileged teacher, under ``no_grad`` and in eval mode.

    eval() as well as no_grad: train-mode dropout would perturb the teacher target. The training
    flag is restored on exit whether or not the forward raised.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        if was_training:
            model.train()


class OPDTermMixin:
    """Adopts the :class:`SDPGArguments` tunables, resolves the OPD divergence and schedules its weight.

    Mixed into the offline self-distillation trainer and the on-policy SDPG trainer, which read
    ``self.state`` (the HF trainer's) for the schedule.
    """

    def _adopt_sdpg_arguments(self, kwargs: dict, *, exclude: frozenset[str] = frozenset()) -> None:
        """Pop the SDPG tunables from the ctor ``kwargs`` onto ``self``, under the names and defaults
        :class:`SDPGArguments` declares."""
        vars(self).update(SDPGArguments.pop_from(kwargs, exclude=exclude))
        self.sdpg_loss_fn = require_divergence(self.sdpg_loss)

    def _opd_beta(self) -> float:
        """The OPD coefficient ``beta(k)`` at the current optimizer step."""
        return beta_warmup_decay(
            int(self.state.global_step),
            int(self.state.max_steps),
            self.sdpg_beta_base,
            self.sdpg_beta_warmup_steps,
            self.sdpg_beta_decay_steps,
        )

    @staticmethod
    def _opd_metrics(opd_loss: torch.Tensor, beta: float) -> dict[str, torch.Tensor | float]:
        """The two OPD metric keys both arms log, on-device for ``store_metrics``."""
        return {"opd_loss": opd_loss.detach(), "opd_beta": beta}
