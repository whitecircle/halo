"""Neither freeze knob may act on a pattern that matches nothing: unfreezing would train nothing with a flat loss,
and freezing would leave what the typo named training. Both fail silently."""

from __future__ import annotations

import pytest
import torch.nn as nn

from halo.model_loading.parallel.freeze_patterns import freeze_modules_by_patterns, unfreeze_modules_by_patterns


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))


def test_a_matching_pattern_unfreezes_only_that_module():
    model = _model()
    unfreeze_modules_by_patterns(model, ["1"])
    assert [param.requires_grad for param in model[0].parameters()] == [False, False]
    assert [param.requires_grad for param in model[1].parameters()] == [True, True]


def test_a_pattern_matching_nothing_raises():
    with pytest.raises(ValueError, match="matched no parameter-bearing module"):
        unfreeze_modules_by_patterns(_model(), ["model.layer.0.*"])


def test_a_parameter_name_instead_of_a_module_name_raises():
    """fnmatch runs against module names here, so a parameter name matches nothing."""
    with pytest.raises(ValueError, match="full MODULE names"):
        unfreeze_modules_by_patterns(_model(), ["1.weight"])


def test_a_matching_freeze_pattern_freezes_only_those_parameters():
    model = _model()
    freeze_modules_by_patterns(model, ["0.weight"])
    assert model[0].weight.requires_grad is False
    assert model[0].bias.requires_grad is True
    assert all(param.requires_grad for param in model[1].parameters())


def test_a_freeze_pattern_matching_nothing_raises():
    with pytest.raises(ValueError, match="matched no parameter"):
        freeze_modules_by_patterns(_model(), ["nothing.matches.this"])


def test_one_typo_among_working_patterns_is_named():
    with pytest.raises(ValueError, match=r"0\.bais"):
        freeze_modules_by_patterns(_model(), ["0.weight", "0.bais"])


def test_a_module_name_instead_of_a_parameter_name_raises():
    """This side matches parameter names, so a bare module name matches nothing."""
    with pytest.raises(ValueError, match="full PARAMETER names"):
        freeze_modules_by_patterns(_model(), ["0"])


def test_overlapping_patterns_are_all_credited():
    """Both match only ``0.weight``, so stopping at the first hit leaves ``0.we*`` unmatched and raises.
    A broad partner like ``0.*`` cannot show this: it picks up ``0.bias`` on its own and is credited anyway."""
    model = _model()
    freeze_modules_by_patterns(model, ["0.weight", "0.we*"])
    assert model[0].weight.requires_grad is False
    assert model[0].bias.requires_grad is True, "neither pattern names the bias"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
