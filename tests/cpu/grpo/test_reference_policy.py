#!/usr/bin/env python
"""The environmental trainer's KL reference forward reads the one reference rule.

A run holding a frozen ``ref_model`` forwards it with the policy's adapters untouched; a PEFT run with
none forwards the policy itself with its adapters disabled, and only for that forward (the offline
trainer's side of the same rule is pinned in ``test_offline_grpo_kl_reference.py``).

    python tests/cpu/grpo/test_reference_policy.py
"""

import contextlib
import types

import pytest
import torch

from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer


class _PeftPolicy:
    def __init__(self):
        self.adapters_off = False

    @contextlib.contextmanager
    def disable_adapter(self):
        self.adapters_off = True
        try:
            yield
        finally:
            self.adapters_off = False


def _ref_forward(ref_model):
    """The model ``_compute_ref_logps`` forwarded and whether the policy's adapters were off during it."""
    policy = _PeftPolicy()
    seen = {}

    def forward(model, ids, mask, keep, compute_entropy):
        seen.update(model=model, adapters_off=policy.adapters_off)
        return torch.zeros(1, keep), None

    host = types.SimpleNamespace(
        beta=0.04,
        model=policy,
        ref_model=ref_model,
        accelerator=types.SimpleNamespace(unwrap_model=lambda model: model),
        _get_per_token_logps_and_entropies=forward,
    )
    DistributedAsyncEnvironmentalGRPOTrainer._compute_ref_logps(host, torch.zeros(1, 3), torch.ones(1, 3), 2)
    return seen, policy


def test_without_a_reference_model_the_policy_forwards_with_its_adapters_off():
    seen, policy = _ref_forward(ref_model=None)
    assert seen == {"model": policy, "adapters_off": True}
    assert policy.adapters_off is False, "the adapters stayed off past the reference forward"


def test_a_held_reference_model_forwards_with_the_adapters_untouched():
    reference = object()
    seen, _ = _ref_forward(ref_model=reference)
    assert seen == {"model": reference, "adapters_off": False}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
