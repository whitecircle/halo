"""The privileged-teacher OPD term both SDPG arms add to their base objective (arXiv:2606.04036)."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from halo.configs.distillation import SDPGArguments
from halo.trainers.distillation.divergences import require_divergence


def beta_warmup_decay(
    step: int,
    total_steps: int,
    beta_base: float,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    """``beta(k) = beta_base * min(1, k / T_warm) * min(1, (T - k) / T_decay)``; constant when
    ``warmup_steps=decay_steps=0``."""
    if beta_base == 0.0:
        return 0.0
    warm = 1.0 if warmup_steps <= 0 else min(1.0, step / warmup_steps)
    # An unset total (<=0) must not zero OPD for the whole run.
    decay_inactive = decay_steps <= 0 or total_steps <= 0
    decay = 1.0 if decay_inactive else min(1.0, max(0.0, (total_steps - step) / decay_steps))
    return beta_base * warm * decay


@contextmanager
def privileged_teacher_pass(model: torch.nn.Module) -> Iterator[None]:
    """eval() as well as no_grad: train-mode dropout would perturb the teacher target."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        if was_training:
            model.train()


class OPDTermMixin:
    """Holds the :class:`SDPGArguments` tunables as ``self.sdpg``; the host trainer's ``self.state`` drives the
    schedule."""

    def _adopt_sdpg_arguments(self, kwargs: dict, *, exclude: frozenset[str] = frozenset()) -> None:
        self.sdpg = SDPGArguments.pop_from(kwargs, exclude=exclude)
        self.sdpg_loss_fn = require_divergence(self.sdpg.sdpg_loss)

    def _opd_beta(self) -> float:
        return beta_warmup_decay(
            int(self.state.global_step),
            int(self.state.max_steps),
            self.sdpg.sdpg_beta_base,
            self.sdpg.sdpg_beta_warmup_steps,
            self.sdpg.sdpg_beta_decay_steps,
        )

    @staticmethod
    def _opd_metrics(opd_loss: torch.Tensor, beta: float) -> dict[str, torch.Tensor | float]:
        return {"opd_loss": opd_loss.detach(), "opd_beta": beta}
