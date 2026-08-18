"""Checkpoint-coverage gate: refuse a load that leaves live weights randomly initialized.

``from_pretrained`` re-initializes every key it could not find and reports the set through the
transformers logger only, which ``setup_logging`` silences off the logging rank — so a truncated or
key-mismatched directory trains from random weights without an exception.
:func:`from_pretrained_verified` reads that report off the return value instead; the lazy EP/PP
loaders bypass ``from_pretrained`` and reach the same gate with the absent set they computed.

Which absences are legitimate is derived from the model class, never from a key list. Depends on
torch, transformers and :mod:`src.env` only, so any loader may import it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

import torch.nn as nn
from transformers import PreTrainedModel

from src.env import env_flag

logger = logging.getLogger(__name__)

ALLOW_MISSING_CHECKPOINT_KEYS_ENV = "HALO_ALLOW_MISSING_CHECKPOINT_KEYS"


def unexpected_missing_keys(
    model: nn.Module, missing_keys: Iterable[str], *, excuse_task_head: bool = True
) -> list[str]:
    """The subset of ``missing_keys`` that cannot be a legitimate absence.

    Legitimate, all read off the model class:

    * a tie (``lm_head.weight`` under ``tie_word_embeddings``) — one object answering to two keys;
    * a task head the architecture adds ON TOP of the backbone: anything outside
      ``base_model_prefix``, so a class that IS the backbone (``AutoModel``) excuses nothing;
    * ``_keys_to_ignore_on_load_missing``, the class's own declaration.

    ``excuse_task_head=False`` withdraws the second excuse, for a caller that CONSUMES a head it
    cannot train: a random ``score`` produces plausible, meaningless numbers instead of a loud error.
    """
    missing = set(missing_keys)
    if not missing:
        return []

    # keep_vars so a tie is the SAME object under both names — ``state_dict()`` detaches, which would
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
    """Raise when ``missing_keys`` holds an absence that is not a legitimate one.

    ``source`` names the checkpoint (or the caller, for the lazy loaders) in the message;
    ``excuse_task_head`` is passed through to :func:`unexpected_missing_keys`.
    """
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


def from_pretrained_verified(
    model_class, model_name_or_path: str, *, excuse_task_head: bool = True, **kwargs
) -> PreTrainedModel:
    """``model_class.from_pretrained`` with the missing-key report captured and gated.

    The report is taken from ``output_loading_info`` rather than the transformers logger so it
    survives the ``set_verbosity_error`` that ``setup_logging`` applies off the logging rank.
    """
    model, loading_info = model_class.from_pretrained(model_name_or_path, output_loading_info=True, **kwargs)
    if getattr(type(model), "_halo_tied_keys_were_list", False) and getattr(
        model.config.get_text_config(), "tie_word_embeddings", False
    ):
        raise RuntimeError(
            f"{type(model).__name__} declares tie_word_embeddings=True but shipped the legacy "
            "list-form _tied_weights_keys, which transformers cannot tie (the compat shim "
            "converts it to {}) — the head would silently train untied. Fix the remote modeling "
            "file's _tied_weights_keys to the dict form."
        )
    verify_checkpoint_coverage(
        model, loading_info["missing_keys"], source=model_name_or_path, excuse_task_head=excuse_task_head
    )
    return model
