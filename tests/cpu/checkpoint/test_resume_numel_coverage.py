#!/usr/bin/env python
"""The FSDP2 resume coverage gate must weigh PARAMETERS, not keys.

MoE experts are two keys per layer but most of the bytes: a checkpoint written under a different
expert namespace matches almost every key while losing the majority of the model. A key-count gate
reads that as a healthy resume and strict=False then silently keeps base values for the experts
(measured 62% of parameters lost). The numel-weighted gate refuses it.

    python tests/cpu/checkpoint/test_resume_numel_coverage.py
"""

import pytest
import torch
import torch.nn as nn

from src.distributed.checkpoint.loader import resume_numel_coverage


class _MoEish(nn.Module):
    """Many tiny non-expert tensors + one huge fused expert tensor — the shape of the failure."""

    def __init__(self):
        super().__init__()
        for i in range(20):
            self.register_parameter(f"small_{i}", nn.Parameter(torch.zeros(4)))
        self.experts_gate_up = nn.Parameter(torch.zeros(4, 512, 64))  # dwarfs everything else
        self.register_buffer("inv_freq", torch.zeros(8), persistent=False)


def test_expert_namespace_mismatch_is_refused():
    model = _MoEish()
    checkpoint_keys = {f"small_{i}" for i in range(20)}  # experts saved under a foreign namespace

    ok, unmatched, matched_numel, total_numel = resume_numel_coverage(model, checkpoint_keys)
    assert not ok, (
        "the gate passed a checkpoint missing the expert tensor — 20/21 keys matched but "
        f"only {matched_numel}/{total_numel} parameters did; strict=False would silently keep "
        "base expert weights"
    )
    assert unmatched == {"experts_gate_up"}


def test_small_real_gaps_still_pass():
    model = _MoEish()
    checkpoint_keys = {"experts_gate_up"} | {f"small_{i}" for i in range(10)}  # e.g. tied head absent

    ok, unmatched, _, _ = resume_numel_coverage(model, checkpoint_keys)
    assert ok and len(unmatched) == 10


def test_non_persistent_buffers_do_not_count():
    model = _MoEish()
    full = {name for name, _ in model.named_parameters()}
    ok, unmatched, _, _ = resume_numel_coverage(model, full)
    assert ok and not unmatched  # inv_freq absent from the checkpoint must not register as a gap


def test_compiled_model_is_peeled_before_matching():
    """A ``torch.compile``d model's live FQNs are ``_orig_mod.*``: matched raw, EVERY key misses and
    the gate refuses a resume ``set_model_state_dict`` handles (torch strips the prefix itself)."""
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
