#!/usr/bin/env python
"""CPU tests for the Adam-beta plumbing in :func:`src.optimizers.muon.build_muon_optimizer`.

Muon's internal scalar (1D / embedding / head) AdamW has its own tuned betas ``(0.9, 0.95)``, while
HF's ``TrainingArguments.adam_beta2`` defaults to ``0.999``. Forwarding the training args
unconditionally would silently retune the scalar optimizer of every Muon config that never mentions
betas, so ``build_muon_optimizer`` forwards a beta only when the config explicitly overrides it.

Run: python tests/cpu/optimizers/test_muon_scalar_betas.py  (or pytest)
"""

import pytest
import torch
import torch.nn as nn
from transformers import TrainingArguments

from src.optimizers.muon import DEFAULT_SCALAR_BETAS, _is_explicitly_set, build_muon_optimizer


class _TinyModel(nn.Module):
    """One Muon-routed matrix plus scalar-routed params (bias, embedding, head)."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8, dtype=torch.bfloat16)
        self.q_proj = nn.Linear(8, 8, bias=True, dtype=torch.bfloat16)
        self.lm_head = nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)


def _scalar_betas(args) -> tuple[float, float]:
    optimizer = build_muon_optimizer(_TinyModel(), args, decay_parameters=[])
    groups = optimizer.scalar_optimizer.param_groups
    betas = {tuple(g["betas"]) for g in groups}
    assert len(betas) == 1, f"scalar param groups disagree on betas: {betas}"
    return betas.pop()


def _args(tmp_path, **overrides) -> TrainingArguments:
    return TrainingArguments(output_dir=str(tmp_path), **overrides)


def test_default_args_keep_muon_tuned_scalar_betas(tmp_path):
    """Untouched adam_beta* must NOT drag HF's 0.999 into Muon's scalar AdamW."""
    assert TrainingArguments.__dataclass_fields__["adam_beta2"].default == 0.999, (
        "test premise: HF's adam_beta2 default differs from Muon's tuned scalar beta2"
    )
    assert _scalar_betas(_args(tmp_path)) == DEFAULT_SCALAR_BETAS == (0.9, 0.95)


def test_explicit_beta2_is_forwarded(tmp_path):
    """A config that actually sets adam_beta2 still steers the scalar optimizer."""
    assert _scalar_betas(_args(tmp_path, adam_beta2=0.98)) == (0.9, 0.98)


def test_explicit_beta1_is_forwarded(tmp_path):
    """beta1 overrides apply independently of whether beta2 was set."""
    assert _scalar_betas(_args(tmp_path, adam_beta1=0.85)) == (0.85, 0.95)


def test_is_explicitly_set_reads_declared_defaults(tmp_path):
    args = _args(tmp_path, adam_beta2=0.98)
    assert _is_explicitly_set(args, "adam_beta2") is True
    assert _is_explicitly_set(args, "adam_beta1") is False
    with pytest.raises(AttributeError):
        _is_explicitly_set(args, "not_a_field")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
