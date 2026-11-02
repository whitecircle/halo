"""Which class a live model was built as (transformers or remote-code), or an Auto class maps a config to."""

import torch
from transformers.modeling_utils import PreTrainedModel

# Set on a module that stands in for a cut model (a pipeline stage), whose own MRO no longer names it.
REMOTE_CODE_MODEL_CLASS_ATTR = "_halo_remote_code_model_class"
# The package transformers imports ``trust_remote_code`` modeling and config files under.
_REMOTE_CODE_PACKAGE = "transformers_modules"


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


def is_remote_code_class(cls: type) -> bool:
    """Whether ``cls`` comes from a ``trust_remote_code`` file rather than transformers."""
    return cls.__module__.startswith(_REMOTE_CODE_PACKAGE)


def remote_code_model_class(model: torch.nn.Module) -> type | None:
    """The remote-code class ``model`` was built from (module under ``transformers_modules``), or None.

    The class stamped under :data:`REMOTE_CODE_MODEL_CLASS_ATTR` when set, else read from the MRO rather
    than ``type(model)``, so an FSDP2 in-place class swap still resolves it.
    """
    return getattr(model, REMOTE_CODE_MODEL_CLASS_ATTR, None) or next(
        (cls for cls in type(model).__mro__ if is_remote_code_class(cls)), None
    )


def mapped_model_class(config, *auto_classes) -> type | None:
    """The class the first of ``auto_classes`` whose mapping knows ``type(config)`` dispatches it to, or
    None (a remote-code config, an auto class without a ``_model_mapping``)."""
    for auto_class in auto_classes:
        mapping = getattr(auto_class, "_model_mapping", None)
        if mapping is None:
            continue
        try:
            return mapping[type(config)]
        except KeyError:
            continue
    return None
