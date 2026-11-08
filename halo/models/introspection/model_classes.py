"""Which class a live model was built as (transformers or remote-code), or an Auto class maps a config to."""

import torch
from transformers.modeling_utils import PreTrainedModel

# Set on a module that stands in for a cut model (a pipeline stage), whose own MRO no longer names it.
REMOTE_CODE_MODEL_CLASS_ATTR = "_halo_remote_code_model_class"
# The package transformers imports ``trust_remote_code`` modeling and config files under.
_REMOTE_CODE_PACKAGE = "transformers_modules"


def transformers_model_class(model: torch.nn.Module) -> type[PreTrainedModel] | None:
    """The first ``PreTrainedModel`` subclass in the MRO that is not torch's FSDP2 in-place class swap
    (``FSDP<Name>``), so a sharded model resolves to its family's class."""
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


def auto_map_references(entry: str | list | tuple | dict) -> tuple[str, ...]:
    """The class references one ``auto_map`` entry holds: one, a tokenizer's ``[slow, fast]`` pair, or an image
    processor's mapping of backend to class."""
    values = entry.values() if isinstance(entry, dict) else (entry,) if isinstance(entry, str) else entry
    return tuple(value for value in values if value)


def is_remote_code_class(cls: type) -> bool:
    return cls.__module__.startswith(_REMOTE_CODE_PACKAGE)


def remote_code_model_class(model: torch.nn.Module) -> type | None:
    """Read from the MRO rather than ``type(model)``, so an FSDP2 in-place class swap still resolves it."""
    return getattr(model, REMOTE_CODE_MODEL_CLASS_ATTR, None) or next(
        (cls for cls in type(model).__mro__ if is_remote_code_class(cls)), None
    )


def mapped_model_class(config, *auto_classes) -> type | None:
    """None for a remote-code config or an auto class without a ``_model_mapping``. Remote code is ruled out by module,
    as the mapping keys on the class name and would answer a remote ``LagunaConfig`` with the in-library class."""
    if is_remote_code_class(type(config)):
        return None
    for auto_class in auto_classes:
        mapping = getattr(auto_class, "_model_mapping", None)
        if mapping is None:
            continue
        try:
            return mapping[type(config)]
        except KeyError:
            continue
    return None
