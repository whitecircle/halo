#!/usr/bin/env python
"""CPU test: per-head norm registration must ACCUMULATE across its two writers.

``_register_per_head_norm_params`` is called from both TP entry points —
``retarget_hf_replicated_grad_hooks`` (HF-native ``tp_plan="auto"``) and
``apply_tp_to_attention_only`` (toolkit selective TP). The single reader,
``DistributedTrainerMixin._sync_tp_replicated_grads``, SUMs the registered names over the TP group
and AVG-reduces everything else: a name dropped from the set trains on exactly ``1/tp_size`` of its
true gradient, silently — the partial tripwire only fires when NO registered name resolves.

Run: ``python tests/cpu/parallelism/test_tp_per_head_norm_registration.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from src.distributed.tensor_parallel.parallelize_attention import _register_per_head_norm_params


class _TwoNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.plan_norm = nn.RMSNorm(4)
        self.selective_norm = nn.RMSNorm(4)
        self.proj = nn.Linear(4, 4)


def test_second_writer_does_not_clobber_the_first():
    """Two writers, disjoint non-empty module sets -> the union survives."""
    model = _TwoNormModel()

    _register_per_head_norm_params(model, [model.plan_norm])
    _register_per_head_norm_params(model, [model.selective_norm])

    assert model._tp_per_head_norm_params == ["plan_norm.weight", "selective_norm.weight"], (
        "the second registration replaced the first: its norms fall out of the step-time SUM and "
        "get AVG-reduced instead, i.e. 1/tp_size of their gradient"
    )


def test_registration_is_idempotent_and_sorted():
    """Re-registering the same modules must not duplicate a name (the reader builds an id set)."""
    model = _TwoNormModel()

    _register_per_head_norm_params(model, [model.selective_norm, model.plan_norm])
    _register_per_head_norm_params(model, [model.plan_norm])

    assert model._tp_per_head_norm_params == ["plan_norm.weight", "selective_norm.weight"]


def test_empty_module_list_leaves_an_existing_registration_intact():
    """The empty-list early return must not be the only thing protecting a prior writer."""
    model = _TwoNormModel()

    _register_per_head_norm_params(model, [model.plan_norm])
    _register_per_head_norm_params(model, [])

    assert model._tp_per_head_norm_params == ["plan_norm.weight"]


def test_only_parameters_of_the_given_modules_are_registered():
    model = _TwoNormModel()

    _register_per_head_norm_params(model, [model.plan_norm])

    assert model._tp_per_head_norm_params == ["plan_norm.weight"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
