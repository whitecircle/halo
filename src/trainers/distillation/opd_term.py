"""The privileged-teacher OPD term both SDPG arms add to their base objective (arXiv:2606.04036)."""

import torch

from src.args.mixins import SDPGArguments
from src.trainers.distillation.losses import beta_warmup_decay, get_divergence


class OPDTermMixin:
    """Adopts the :class:`SDPGArguments` tunables, resolves the OPD divergence and schedules its weight.

    Mixed into the offline self-distillation trainer and the on-policy SDPG trainer, which read
    ``self.state`` (the HF trainer's) for the schedule.
    """

    def _adopt_sdpg_arguments(self, kwargs: dict, *, exclude: frozenset[str] = frozenset()) -> None:
        """Pop the SDPG tunables from the ctor ``kwargs`` onto ``self``, under the names and defaults
        :class:`SDPGArguments` declares."""
        vars(self).update(SDPGArguments.pop_from(kwargs, exclude=exclude))
        self.sdpg_loss_fn = get_divergence(self.sdpg_loss, jsd_beta=self.sdpg_jsd_beta)

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
