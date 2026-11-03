"""Checkpoint-coverage gate: refuse a load that leaves live weights randomly initialized. ``from_pretrained`` reports
re-initialized keys only through the transformers logger, which ``configure_training_run`` silences off the logging
rank."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import torch.nn as nn
from transformers import PreTrainedModel

from halo.env import env_flag
from halo.model_loading.remote_code import reject_untieable_list_ties
from halo.models.config_access import drop_toolkit_stamps

logger = logging.getLogger(__name__)

ALLOW_MISSING_CHECKPOINT_KEYS_ENV = "HALO_ALLOW_MISSING_CHECKPOINT_KEYS"
# accelerate's switch under an FSDP launch for keeping the checkpoint's weights on each node's local rank 0 only;
# transformers reads it inside ``from_pretrained``.
_FSDP_CPU_RAM_EFFICIENT_LOADING = "FSDP_CPU_RAM_EFFICIENT_LOADING"


def unexpected_missing_keys(
    model: nn.Module, missing_keys: Iterable[str], *, excuse_task_head: bool = True
) -> list[str]:
    """The ``missing_keys`` that cannot be legitimate. Excused, off the model class: a tie, a task head outside
    ``base_model_prefix`` (unless ``excuse_task_head=False``, for a caller consuming a head it cannot train), and
    ``_keys_to_ignore_on_load_missing``."""
    missing = set(missing_keys)
    if not missing:
        return []

    # keep_vars so a tie is the same object under both names: ``state_dict()`` detaches, which would
    # hide the sharing and report a tied shadow as an absence.
    state = model.state_dict(keep_vars=True)
    loaded_tensor_ids = {id(tensor) for key, tensor in state.items() if key not in missing}
    prefix = getattr(model, "base_model_prefix", "")
    backbone_prefix = f"{prefix}." if prefix and hasattr(model, prefix) else ""
    declared_ignores = tuple(getattr(type(model), "_keys_to_ignore_on_load_missing", None) or ())

    unexpected: list[str] = []
    for key in sorted(missing):
        if key in state and id(state[key]) in loaded_tensor_ids:
            continue
        if any(re.search(pattern, key) for pattern in declared_ignores):
            continue
        if excuse_task_head and backbone_prefix and not key.startswith(backbone_prefix):
            continue
        unexpected.append(key)
    return unexpected


def verify_checkpoint_coverage(
    model: nn.Module, missing_keys: Iterable[str], *, source: str, excuse_task_head: bool = True
) -> None:
    """Raise when ``missing_keys`` holds an absence not legitimate; ``source`` names the checkpoint or caller."""
    unexpected = unexpected_missing_keys(model, missing_keys, excuse_task_head=excuse_task_head)
    if not unexpected:
        return

    detail = (
        f"{source}: {len(unexpected)} tensor(s) of {type(model).__name__} are absent from the "
        f"checkpoint and were randomly initialized. First few: {unexpected[:8]}"
    )
    if env_flag(ALLOW_MISSING_CHECKPOINT_KEYS_ENV):
        logger.warning(f"{ALLOW_MISSING_CHECKPOINT_KEYS_ENV} is set — accepting random weights. {detail}")
        return
    excused = (
        "A task head the architecture adds on top of a base checkpoint is excused automatically; "
        if excuse_task_head
        else "This caller consumes the task head rather than training it, so its absence is not excused; "
    )
    raise RuntimeError(
        f"{detail}. Whatever runs next would use random weights for these, with no other symptom "
        f"than results that look plausible. The usual cause is a truncated, partially uploaded or "
        f"wrong-architecture checkpoint directory. {excused}set "
        f"{ALLOW_MISSING_CHECKPOINT_KEYS_ENV}=1 to accept these absences deliberately."
    )


@contextmanager
def full_weights_on_every_rank() -> Iterator[None]:
    """Turn ``FSDP_CPU_RAM_EFFICIENT_LOADING`` off for the scope. Under an accelerate FSDP launch it leaves off-rank-0
    weights as CPU zero copies until ``accelerator.prepare``'s broadcast, which a frozen model never gets and which
    anything using the policy first (a reference sweep, a teacher on its device) precedes."""
    previous = os.environ.get(_FSDP_CPU_RAM_EFFICIENT_LOADING)
    os.environ[_FSDP_CPU_RAM_EFFICIENT_LOADING] = "false"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_FSDP_CPU_RAM_EFFICIENT_LOADING, None)
        else:
            os.environ[_FSDP_CPU_RAM_EFFICIENT_LOADING] = previous


def from_pretrained_verified(
    model_class, model_name_or_path: str, *, excuse_task_head: bool = True, **kwargs
) -> PreTrainedModel:
    """``from_pretrained`` with its missing-key report gated, inside :func:`full_weights_on_every_rank`. A passed
    ``config`` carries this run's stamps and is kept; one read from the checkpoint takes
    :func:`drop_toolkit_stamps`."""
    with full_weights_on_every_rank():
        model, loading_info = model_class.from_pretrained(model_name_or_path, output_loading_info=True, **kwargs)
    if kwargs.get("config") is None:
        drop_toolkit_stamps(model.config)
    reject_untieable_list_ties(model)
    verify_checkpoint_coverage(
        model, loading_info["missing_keys"], source=model_name_or_path, excuse_task_head=excuse_task_head
    )
    return model
