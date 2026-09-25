#!/usr/bin/env python
"""CPU test for the parameter split :func:`src.optimizers.muon.build_muon_optimizer` reports.

The token embedding and output head are 2D yet routed to Muon's internal scalar AdamW, so a summary
counted by ``ndim`` reports them under Newton-Schulz. The logged split must match the routing the
built optimizer actually holds.

Run: pytest tests/cpu/optimizers/test_muon_build_summary.py
"""

import logging

import pytest
import torch
import torch.nn as nn
from transformers import TrainingArguments

from src.optimizers.muon import build_muon_optimizer

_VOCAB = 1000
_HIDDEN = 1024


class _TinyModel(nn.Module):
    """A Muon-routed hidden matrix beside a 2D embedding and head of the same order of size."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN, dtype=torch.bfloat16)
        self.q_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=False, dtype=torch.bfloat16)
        self.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False, dtype=torch.bfloat16)


def test_summary_counts_the_embedding_and_head_under_the_scalar_optimizer(tmp_path, caplog):
    model = _TinyModel()
    with caplog.at_level(logging.INFO, logger="src.optimizers.muon"):
        optimizer = build_muon_optimizer(model, TrainingArguments(output_dir=str(tmp_path)), decay_parameters=[])

    scalar_ids = {id(p) for g in optimizer.scalar_optimizer.param_groups for p in g["params"]}
    assert scalar_ids == {id(model.embed_tokens.weight), id(model.lm_head.weight)}, "test premise: routing"

    muon_m = model.q_proj.weight.numel() / 1e6
    scalar_m = (model.embed_tokens.weight.numel() + model.lm_head.weight.numel()) / 1e6
    summary = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Muon optimizer:"))
    assert f"{muon_m:.1f}M params (Newton-Schulz)" in summary, summary
    assert f"{scalar_m:.1f}M params (AdamW scalar)" in summary, summary


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
