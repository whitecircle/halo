#!/usr/bin/env python
"""Every toolkit override of a TRL ``GRPOTrainer`` method takes the arguments TRL's signature declares.

TRL calls its own methods with its own signature: when a release adds a parameter (1.14's
``num_generations`` on ``_generate_single_turn``), an override still spelled to the old one raises
``TypeError`` on every rollout, and a toolkit method that happens to share a TRL name shadows it for
TRL's internal calls. Each override is bound against TRL's declared parameters: the required ones
positionally, the rest by keyword. A TRL method taking ``*args`` cannot be bound this way; its
override has its own test.

    python tests/cpu/grpo/test_grpo_override_signatures.py
"""

import inspect

import pytest
from trl import GRPOTrainer

from src.trainers.distillation.sdpg import DistributedSDPGTrainer
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from src.trainers.grpo.online import DistributedGRPOTrainer

TRAINERS = [DistributedGRPOTrainer, DistributedAsyncEnvironmentalGRPOTrainer, DistributedSDPGTrainer]


def _trl_call(trl_method) -> tuple[list, dict] | None:
    """Placeholder arguments for a call spelled to ``trl_method``'s signature, ``None`` past ``*args``."""
    positional, keywords = [], {}
    for param in list(inspect.signature(trl_method).parameters.values())[1:]:
        if param.kind is param.VAR_POSITIONAL:
            return None
        if param.kind is param.POSITIONAL_OR_KEYWORD and param.default is param.empty:
            positional.append(None)
        elif param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            keywords[param.name] = None
    return positional, keywords


def _overrides(trainer_cls):
    """``(owner, name, method)`` for every toolkit method in the MRO that TRL's GRPOTrainer also defines."""
    for owner in trainer_cls.__mro__:
        if owner.__module__.startswith(("trl.", "transformers.")) or owner is object:
            continue
        for name, method in vars(owner).items():
            if (
                inspect.isfunction(method)
                and not name.startswith("__")
                and inspect.isfunction(getattr(GRPOTrainer, name, None))
            ):
                yield owner, name, method


@pytest.mark.parametrize("trainer_cls", TRAINERS, ids=lambda cls: cls.__name__)
def test_every_override_binds_trls_signature(trainer_cls):
    unbindable = []
    for owner, name, method in _overrides(trainer_cls):
        call = _trl_call(getattr(GRPOTrainer, name))
        if call is None:
            continue
        positional, keywords = call
        try:
            inspect.signature(method).bind(None, *positional, **keywords)
        except TypeError as error:
            unbindable.append(f"{owner.__name__}.{name}: {error}")
    assert not unbindable, "overrides TRL's own calls cannot bind:\n" + "\n".join(unbindable)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
