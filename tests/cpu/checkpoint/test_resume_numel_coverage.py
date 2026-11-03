#!/usr/bin/env python
"""The FSDP2 resume coverage gate weighs parameters, since experts are two keys per layer but most of the bytes.
A key count passes a foreign expert namespace, and strict=False silently keeps base experts."""

import pytest
import torch
import torch.nn as nn

from halo.checkpoint.parallel.resume_weights import resume_numel_coverage


class _MoEish(nn.Module):
    """Many tiny non-expert tensors and one huge fused expert tensor."""

    def __init__(self):
        super().__init__()
        for i in range(20):
            self.register_parameter(f"small_{i}", nn.Parameter(torch.zeros(4)))
        self.experts_gate_up = nn.Parameter(torch.zeros(4, 512, 64))
        self.register_buffer("inv_freq", torch.zeros(8), persistent=False)


def test_expert_namespace_mismatch_is_refused():
    model = _MoEish()
    checkpoint_keys = {f"small_{i}" for i in range(20)}  # Experts saved under a foreign namespace.

    ok, unmatched, matched_numel, total_numel = resume_numel_coverage(model, checkpoint_keys)
    assert not ok, (
        "the gate passed a checkpoint missing the expert tensor — 20/21 keys matched but "
        f"only {matched_numel}/{total_numel} parameters did; strict=False would silently keep "
        "base expert weights"
    )
    assert unmatched == {"experts_gate_up"}


def test_small_real_gaps_still_pass():
    model = _MoEish()
    checkpoint_keys = {"experts_gate_up"} | {f"small_{i}" for i in range(10)}  # E.g. tied head absent.

    ok, unmatched, _, _ = resume_numel_coverage(model, checkpoint_keys)
    assert ok and len(unmatched) == 10


def test_non_persistent_buffers_do_not_count():
    model = _MoEish()
    full = {name for name, _ in model.named_parameters()}
    ok, unmatched, _, _ = resume_numel_coverage(model, full)
    assert ok and not unmatched  # inv_freq absent from the checkpoint must not register as a gap.


def test_compiled_model_is_peeled_before_matching():
    """Live FQNs are ``_orig_mod.*``; matched raw, every key misses, though torch strips the prefix itself."""
    model = torch.compile(_MoEish())
    assert any(name.startswith("_orig_mod.") for name, _ in model.named_parameters()), (
        "premise: torch.compile must have wrapped the model"
    )
    checkpoint_keys = {name for name, _ in _MoEish().named_parameters()}

    ok, unmatched, matched_numel, total_numel = resume_numel_coverage(model, checkpoint_keys)

    assert ok and not unmatched, (
        f"the gate would raise on a compiled model: only {matched_numel}/{total_numel} parameters matched"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
