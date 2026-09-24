#!/usr/bin/env python
"""Every trainer writes its card through the checkpointing mixin, which adds the Halo Hub tag.

``CheckpointingMixin.create_model_card`` unions the tag into the ``tags`` keyword and hands off to the
library writer behind it. A trainer class, or a base ahead of the mixin, that defines its own writer
drops the tag; a library writer that stops taking ``tags`` as a keyword breaks the hand-off.

    python tests/cpu/trainers/test_trainer_card_tags.py
"""

import inspect

import pytest

from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.mixins.checkpointing import CheckpointingMixin
from tests.common.rosters import import_all_trainers

import_all_trainers()

_WRITER = "create_model_card"
_KEYWORD_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def _subclasses(cls: type):
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


# Derived from the class hierarchy, not listed: a new trainer joins by existing.
TRAINERS = sorted(
    {cls for cls in _subclasses(DistributedTrainerMixin) if cls.__module__.startswith("src.trainers")},
    key=lambda cls: cls.__name__,
)


def _writer_owner(classes) -> type:
    return next(base for base in classes if _WRITER in base.__dict__)


def test_the_derived_roster_covers_every_trainer():
    """Anti-vacuity: a roster that stopped finding trainers would make the check below assert nothing."""
    names = {cls.__name__ for cls in TRAINERS}
    assert len(names) >= 13, f"the roster collapsed to {sorted(names)}"
    assert {"DistributedSFTTrainer", "SmoothMarginPOTrainer", "EmbeddingTrainer"} <= names, sorted(names)


@pytest.mark.parametrize("trainer", TRAINERS, ids=lambda cls: cls.__name__)
def test_the_card_writer_is_the_mixins_and_hands_tags_on(trainer):
    mro = trainer.__mro__
    owner = _writer_owner(mro)
    assert owner is CheckpointingMixin, f"{trainer.__name__}.{_WRITER} resolves to {owner.__name__}, which adds no tag"
    library = _writer_owner(mro[mro.index(CheckpointingMixin) + 1 :])
    tags = inspect.signature(getattr(library, _WRITER)).parameters.get("tags")
    assert tags is not None and tags.kind in _KEYWORD_KINDS, (
        f"{library.__module__}.{library.__name__}.{_WRITER} takes no ``tags`` keyword"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
