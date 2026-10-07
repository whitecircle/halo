"""The KL reference model a GRPO trainer forwards, shared by the environmental and offline trainers."""

from collections.abc import Iterator
from contextlib import contextmanager

from torch import nn


@contextmanager
def reference_policy(trainer) -> Iterator[nn.Module]:
    """Yield the model that scores the KL reference: the trainer's frozen ``ref_model`` when it holds
    one, else the PEFT policy itself, with its adapters disabled for the duration."""
    if trainer.ref_model is not None:
        yield trainer.ref_model
        return
    with trainer.accelerator.unwrap_model(trainer.model).disable_adapter():
        yield trainer.model
