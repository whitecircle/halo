#!/usr/bin/env python
"""Every trainer writes its card through the checkpointing mixin, which adds the Halo Hub tag.
A trainer, or a base ahead of the mixin, that defines its own writer drops the tag.
"""

import inspect

import pytest

from halo.trainers.checkpointing import CheckpointingMixin
from tests.common.rosters import distributed_trainer_classes

_WRITER = "create_model_card"
_KEYWORD_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)

# Derived from the class hierarchy, so a new trainer joins by existing.
TRAINERS = distributed_trainer_classes()


def _writer_owner(classes) -> type:
    return next(base for base in classes if _WRITER in base.__dict__)


def test_the_derived_roster_covers_every_trainer():
    """A roster that stopped finding trainers would make the check below assert nothing."""
    names = {cls.__name__ for cls in TRAINERS}
    assert len(names) >= 13, f"the roster collapsed to {sorted(names)}"
    assert {"DistributedSFTTrainer", "SMPOTrainer", "EmbeddingTrainer"} <= names, sorted(names)


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
