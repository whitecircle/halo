"""Neither freeze knob may act on a pattern that matches nothing.

``unfreeze_modules_by_patterns`` freezes every parameter first, so a pattern that matches nothing
leaves ZERO trainable parameters. AdamW accepts an empty parameter group without complaint, so the
run trains for its full schedule with a flat loss and no error. The patterns are fnmatch against
full MODULE names, which is exactly the confusion that produces a non-matching pattern.

``freeze_modules_by_patterns`` fails the other way and just as quietly: it only subtracts, so a
typo'd entry leaves exactly the parameters it named training while every other pattern works — the
loss curve, the logs and the checkpoint all look normal. Its patterns are fnmatch against full
PARAMETER names, the opposite convention of its counterpart, which is where the typo comes from.

    python tests/cpu/models/test_freeze_patterns.py
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from src.distributed.loading.peft_setup import freeze_modules_by_patterns, unfreeze_modules_by_patterns


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))


def test_a_matching_pattern_unfreezes_only_that_module():
    model = _model()
    unfreeze_modules_by_patterns(model, ["1"])
    assert [param.requires_grad for param in model[0].parameters()] == [False, False]
    assert [param.requires_grad for param in model[1].parameters()] == [True, True]


def test_a_pattern_matching_nothing_raises():
    """Pre-gate this returned normally with every parameter frozen."""
    with pytest.raises(ValueError, match="matched no parameter-bearing module"):
        unfreeze_modules_by_patterns(_model(), ["model.layer.0.*"])


def test_a_parameter_name_instead_of_a_module_name_raises():
    """The most likely typo: fnmatch here runs against module names, so a parameter name matches
    nothing and would freeze the model."""
    with pytest.raises(ValueError, match="full MODULE names"):
        unfreeze_modules_by_patterns(_model(), ["1.weight"])


def test_a_matching_freeze_pattern_freezes_only_those_parameters():
    model = _model()
    freeze_modules_by_patterns(model, ["0.weight"])
    assert model[0].weight.requires_grad is False
    assert model[0].bias.requires_grad is True
    assert all(param.requires_grad for param in model[1].parameters())


def test_a_freeze_pattern_matching_nothing_raises():
    """Pre-gate this returned normally, having frozen nothing: the module the user meant to hold
    still trained for the whole run with no other symptom."""
    with pytest.raises(ValueError, match="matched no parameter"):
        freeze_modules_by_patterns(_model(), ["nothing.matches.this"])


def test_one_typo_among_working_patterns_is_named():
    """The realistic case — the other patterns work, so nothing else looks wrong."""
    with pytest.raises(ValueError, match=r"0\.bais"):
        freeze_modules_by_patterns(_model(), ["0.weight", "0.bais"])


def test_a_module_name_instead_of_a_parameter_name_raises():
    """The mirror of the unfreeze typo: this side matches PARAMETER names, so a bare module name
    (which is what the counterpart takes) matches nothing."""
    with pytest.raises(ValueError, match="full PARAMETER names"):
        freeze_modules_by_patterns(_model(), ["0"])


def test_overlapping_patterns_are_all_credited():
    """A parameter must be tested against every pattern, not just until the first hit: stopping at
    the first would leave a broader pattern looking unmatched and raise on a correct config.

    Both patterns here match ``0.weight`` and NOTHING else, so the only parameter that can credit
    ``0.we*`` is one an earlier pattern has already claimed — a break after the first hit leaves it
    unmatched and raises. A broad partner like ``0.*`` cannot show this: it would pick up ``0.bias``
    on its own and be credited even with the break in place."""
    model = _model()
    freeze_modules_by_patterns(model, ["0.weight", "0.we*"])
    assert model[0].weight.requires_grad is False
    assert model[0].bias.requires_grad is True, "neither pattern names the bias"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
