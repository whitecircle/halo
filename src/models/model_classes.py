"""Which transformers class, or which remote-code class, a live model was built as."""

import torch
from transformers.modeling_utils import PreTrainedModel


def transformers_model_class(model: torch.nn.Module) -> type[PreTrainedModel] | None:
    """The transformers class ``model`` was built as, or None for a non-transformers carrier.

    The first ``PreTrainedModel`` subclass in the MRO that is not torch's FSDP2 in-place class swap
    (``FSDP<Name>``, module ``torch.*``), so a sharded model resolves to its family's class.
    """
    return next(
        (
            cls
            for cls in type(model).__mro__
            if isinstance(cls, type)
            and issubclass(cls, PreTrainedModel)
            and cls is not PreTrainedModel
            and not cls.__module__.startswith("torch")
        ),
        None,
    )


def remote_code_model_class(model: torch.nn.Module) -> type | None:
    """The remote-code class ``model`` was built from (module under ``transformers_modules``), or None.

    Read from the MRO rather than ``type(model)``, so an FSDP2 in-place class swap still resolves it.
    """
    return next((cls for cls in type(model).__mro__ if cls.__module__.startswith("transformers_modules")), None)
