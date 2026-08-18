"""revert_load_conversions must not re-add a wrapper prefix the saved model does not have.

A text-only load of a multimodal checkpoint (``text_only_model``) consumes
``PrefixChange(prefix_to_remove="language_model")``; reverting it at save would emit
``model.language_model.*`` keys under a text-only config — a checkpoint transformers re-strips on
reload but engine loaders keyed on ``architectures`` cannot read. These tests fail if the guard
stops consulting the module tree, and prove the revert still fires for a model that really has the
wrapper child.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.core_model_loading import PrefixChange

from src.checkpoint.format import revert_load_conversions


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)


class _TextOnly(nn.Module):
    """CausalLM-shaped: base tree has no `language_model` child."""

    base_model_prefix = "model"

    def __init__(self):
        super().__init__()
        self.model = _Inner()
        self.config = SimpleNamespace()  # revert_weight_conversion passes model.config to convert()
        self._weight_conversions = [PrefixChange(prefix_to_remove="language_model", model_prefix="model")]


class _WrapperInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _Inner()


class _Wrapper(nn.Module):
    """VLM-wrapper-shaped: the `language_model` child is real."""

    base_model_prefix = "model"

    def __init__(self):
        super().__init__()
        self.model = _WrapperInner()
        self.config = SimpleNamespace()
        self._weight_conversions = [PrefixChange(prefix_to_remove="language_model", model_prefix="model")]


def test_text_only_save_keeps_causal_lm_keys():
    model = _TextOnly()
    state = {"model.linear.weight": torch.zeros(2, 2)}
    out = revert_load_conversions(model, dict(state))
    assert set(out) == {"model.linear.weight"}, out
    # The model's own conversions list must be restored for later saves.
    assert isinstance(model._weight_conversions[0], PrefixChange)


def test_wrapper_save_still_reverts_the_prefix():
    model = _Wrapper()
    state = {"model.linear.weight": torch.zeros(2, 2)}
    out = revert_load_conversions(model, dict(state))
    assert set(out) == {"model.language_model.linear.weight"}, out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
