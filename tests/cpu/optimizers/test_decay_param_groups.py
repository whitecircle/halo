#!/usr/bin/env python
"""The weight-decay param-group split each custom-optimizer builder hands its optimizer. Group count and order are the
index space saved state and schedulers key on, so drift silently restores moments onto the wrong parameters."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from halo.optimizers import flash_adamw
from halo.optimizers.adamw_bf16 import adamw_bf16_optimizer_from_args
from halo.optimizers.flash_adamw import build_flash_adamw_optimizer
from halo.optimizers.muon import build_muon_optimizer

WEIGHT_DECAY = 0.123
SCALAR_WEIGHT_DECAY = 0.456  # Distinct, so a scalar group serving the matrix value is visible.

# HF's decay rule for _TinyNet; the frozen matrix is named on purpose, and the builders must still drop it.
DECAY_NAMES = (
    "expert_weight",
    "embed_tokens.weight",
    "q_proj.weight",
    "lm_head.weight",
    "frozen_proj.weight",
)
EXPECTED_DECAY = ["expert_weight", "embed_tokens.weight", "q_proj.weight", "lm_head.weight"]
EXPECTED_NO_DECAY = ["expert_bias", "rms_weight", "q_proj.bias", "norm.weight", "norm.bias"]
EXPECTED_ALL_TRAINABLE = [
    "expert_weight",
    "expert_bias",
    "rms_weight",
    "embed_tokens.weight",
    "q_proj.weight",
    "q_proj.bias",
    "norm.weight",
    "norm.bias",
    "lm_head.weight",
]


class _TinyNet(nn.Module):
    """The 3D ``expert_weight`` / 2D ``expert_bias`` pair: a non-decaying matrix keeps Muon's no-decay matrix group
    populated."""

    def __init__(self):
        super().__init__()
        self.expert_weight = nn.Parameter(torch.randn(2, 8, 8, dtype=torch.bfloat16))
        self.expert_bias = nn.Parameter(torch.zeros(2, 8, dtype=torch.bfloat16))
        self.rms_weight = nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
        self.embed_tokens = nn.Embedding(16, 8, dtype=torch.bfloat16)
        self.q_proj = nn.Linear(8, 8, bias=True, dtype=torch.bfloat16)
        self.norm = nn.LayerNorm(8, dtype=torch.bfloat16)
        self.lm_head = nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)
        self.frozen_proj = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
        self.frozen_proj.requires_grad_(False)


class _FakeFlashAdamW(torch.optim.Optimizer):
    """flashoptim's FlashAdamW refuses CPU parameters at construction; a torch ``Optimizer`` fills group defaults as
    the real one does."""

    def __init__(self, params, lr, betas, eps, weight_decay, master_weight_bits):
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        weight_decay=WEIGHT_DECAY,
        learning_rate=1e-3,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
    )


def _names(model: nn.Module, groups) -> list[list[str]]:
    by_id = {id(p): n for n, p in model.named_parameters()}
    return [[by_id[id(p)] for p in group["params"]] for group in groups]


def test_bf16_builder_splits_by_decay_name_in_order():
    model = _TinyNet()
    optimizer = adamw_bf16_optimizer_from_args(model, _args(), DECAY_NAMES)

    assert _names(model, optimizer.param_groups) == [EXPECTED_DECAY, EXPECTED_NO_DECAY]
    assert [g["weight_decay"] for g in optimizer.param_groups] == [WEIGHT_DECAY, 0.0]


def test_bf16_builder_keeps_an_empty_no_decay_group():
    """Collapsing to one group renumbers every saved group and drops the shard restore."""
    model = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    optimizer = adamw_bf16_optimizer_from_args(model, _args(), ["weight"])

    assert _names(model, optimizer.param_groups) == [["weight"], []]
    assert [g["weight_decay"] for g in optimizer.param_groups] == [WEIGHT_DECAY, 0.0]


def test_flash_builder_splits_by_decay_name_in_order(monkeypatch):
    monkeypatch.setattr(flash_adamw, "FlashAdamW", _FakeFlashAdamW)
    model = _TinyNet()
    optimizer = build_flash_adamw_optimizer(model, weight_decay=WEIGHT_DECAY, decay_parameters=DECAY_NAMES)

    assert _names(model, optimizer.param_groups) == [EXPECTED_DECAY, EXPECTED_NO_DECAY]
    assert [g["weight_decay"] for g in optimizer.param_groups] == [WEIGHT_DECAY, 0.0]


def test_flash_builder_drops_the_empty_group(monkeypatch):
    """FlashAdamW's own convention."""
    monkeypatch.setattr(flash_adamw, "FlashAdamW", _FakeFlashAdamW)
    model = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    optimizer = build_flash_adamw_optimizer(model, weight_decay=WEIGHT_DECAY, decay_parameters=["weight"])

    assert _names(model, optimizer.param_groups) == [["weight"]]
    assert [g["weight_decay"] for g in optimizer.param_groups] == [WEIGHT_DECAY]


def test_flash_builder_without_decay_names_decays_every_trainable_param(monkeypatch):
    monkeypatch.setattr(flash_adamw, "FlashAdamW", _FakeFlashAdamW)
    model = _TinyNet()
    optimizer = build_flash_adamw_optimizer(model, weight_decay=WEIGHT_DECAY, decay_parameters=None)

    assert _names(model, optimizer.param_groups) == [EXPECTED_ALL_TRAINABLE]
    assert [g["weight_decay"] for g in optimizer.param_groups] == [WEIGHT_DECAY]


def test_muon_builder_splits_both_legs_by_decay_name():
    model = _TinyNet()
    optimizer = build_muon_optimizer(
        model,
        weight_decay=WEIGHT_DECAY,
        scalar_weight_decay=SCALAR_WEIGHT_DECAY,
        decay_parameters=DECAY_NAMES,
        ns_use_kernels=False,
    )

    # gram_newton_schulz keys its 3D->2D split on the first tensor's dim-0.
    assert _names(model, optimizer._muon_param_groups) == [["q_proj.weight", "expert_weight"], ["expert_bias"]]
    assert [g["weight_decay"] for g in optimizer._muon_param_groups] == [WEIGHT_DECAY, 0.0]

    scalar_groups = optimizer.scalar_optimizer.param_groups
    assert _names(model, scalar_groups) == [
        ["embed_tokens.weight", "lm_head.weight"],
        ["rms_weight", "q_proj.bias", "norm.weight", "norm.bias"],
    ]
    assert [g["weight_decay"] for g in scalar_groups] == [SCALAR_WEIGHT_DECAY, 0.0]

    # The schedulers walk the combined view: matrix groups first, scalar groups after.
    assert len(optimizer.param_groups) == 4


def test_muon_builder_drops_the_empty_groups():
    model = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    optimizer = build_muon_optimizer(
        model, weight_decay=WEIGHT_DECAY, decay_parameters=["weight"], ns_use_kernels=False
    )

    assert _names(model, optimizer._muon_param_groups) == [["weight"]]
    assert optimizer.scalar_optimizer is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
