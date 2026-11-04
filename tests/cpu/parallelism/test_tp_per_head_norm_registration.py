#!/usr/bin/env python
"""Per-head norm registration accumulates across its two writers.
The reader SUMs registered names and AVGs the rest, so a dropped name trains on ``1/tp_size`` of its gradient."""

from __future__ import annotations

import pytest
import torch.nn as nn

from halo.distributed.tensor_parallel.plain_params import per_head_norm_param_names, register_per_head_norm_params


class _TwoNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.plan_norm = nn.RMSNorm(4)
        self.selective_norm = nn.RMSNorm(4)
        self.proj = nn.Linear(4, 4)


def test_second_writer_does_not_clobber_the_first():
    model = _TwoNormModel()

    register_per_head_norm_params(model, [model.plan_norm])
    register_per_head_norm_params(model, [model.selective_norm])

    assert per_head_norm_param_names(model) == ("plan_norm.weight", "selective_norm.weight"), (
        "the second registration replaced the first: its norms fall out of the step-time SUM and "
        "get AVG-reduced instead, i.e. 1/tp_size of their gradient"
    )


def test_registration_is_idempotent_and_sorted():
    """The reader builds an id set."""
    model = _TwoNormModel()

    register_per_head_norm_params(model, [model.selective_norm, model.plan_norm])
    register_per_head_norm_params(model, [model.plan_norm])

    assert per_head_norm_param_names(model) == ("plan_norm.weight", "selective_norm.weight")


def test_empty_module_list_leaves_an_existing_registration_intact():
    """The early return must not be the only thing protecting a prior writer."""
    model = _TwoNormModel()

    register_per_head_norm_params(model, [model.plan_norm])
    register_per_head_norm_params(model, [])

    assert per_head_norm_param_names(model) == ("plan_norm.weight",)


def test_only_parameters_of_the_given_modules_are_registered():
    model = _TwoNormModel()

    register_per_head_norm_params(model, [model.plan_norm])

    assert per_head_norm_param_names(model) == ("plan_norm.weight",)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
