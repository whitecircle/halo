#!/usr/bin/env python
"""The weight-decay param-group split every custom-optimizer builder hands its optimizer.

Membership, per-group ``weight_decay``, group COUNT and group ORDER are all load-bearing and none of
them raises when it drifts: a name that slips out of the decay group trains a norm/bias with decay
(or an FFN matrix without) for the whole run, while the count and the order are the index space a
saved optimizer state and every ``lr_scheduler`` per-group update are keyed on — an inserted,
dropped or swapped group silently restores moments onto the wrong parameters.

Pinned per builder: AdamWBF16 keeps BOTH groups even when one is empty, FlashAdamW emits only the
non-empty ones (and a single group of everything when no decay names are given), and Muon splits
each of its two legs — matrix and scalar — with the scalar leg carrying its own decay value and the
matrix leg ordered 2D-before-3D.

Run: pytest tests/cpu/optimizers/test_decay_param_groups.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.optimizers import flash_adamw
from src.optimizers.adamw_bf16 import build_bf16_optimizer
from src.optimizers.flash_adamw import create_flash_adamw_optimizer
from src.optimizers.muon import create_muon_optimizer

WEIGHT_DECAY = 0.123
SCALAR_WEIGHT_DECAY = 0.456  # distinct, so a scalar group serving the matrix value is visible

# What HF's decay-parameter rule yields for _TinyNet: every weight but the norms and biases. The
# frozen matrix is named here on purpose — the builders must still drop it (requires_grad=False).
DECAY_NAMES = (
    "expert_weight",
    "embed_tokens.weight",
    "q_proj.weight",
    "lm_head.weight",
    "frozen_proj.weight",
)
EXPECTED_DECAY = ["expert_weight", "embed_tokens.weight", "q_proj.weight", "lm_head.weight"]
EXPECTED_NO_DECAY = ["expert_bias", "rms_weight", "q_proj.bias", "norm.weight", "norm.bias"]
# Every trainable parameter, in ``named_parameters`` order — what an undifferentiated group holds.
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
    """Biases, a LayerNorm, a bare RMSNorm-style weight, 2D and 3D matrices, one frozen matrix.

    The 3D ``expert_weight`` / 2D ``expert_bias`` pair is the MoE expert stack shape: the bias is a
    matrix that must NOT decay, which is what keeps Muon's no-decay matrix group populated.
    """

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
    """Stand-in for flashoptim's FlashAdamW, which refuses CPU parameters at construction.

    The split under test is toolkit code above the optimizer, and a torch ``Optimizer`` fills the
    group defaults exactly as the real one does — so these are the groups the run would get.
    """

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
    """Parameter names per group, in group order and in each group's own order."""
    by_id = {id(p): n for n, p in model.named_parameters()}
    return [[by_id[id(p)] for p in group["params"]] for group in groups]


def test_bf16_builder_splits_by_decay_name_in_order():
    model = _TinyNet()
    optimizer = build_bf16_optimizer(model, _args(), DECAY_NAMES)

    assert _names(model, optimizer.param_groups) == [EXPECTED_DECAY, EXPECTED_NO_DECAY]
    assert [g["weight_decay"] for g in optimizer.param_groups] == [WEIGHT_DECAY, 0.0]


def test_bf16_builder_keeps_an_empty_no_decay_group():
    """Two groups always: the group count is what a resumed optimizer state indexes.

    A model whose every trainable parameter decays still ships a (possibly empty) no-decay group —
    collapsing it to one group renumbers every saved group and drops the shard restore.
    """
    model = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    optimizer = build_bf16_optimizer(model, _args(), ["weight"])

    assert _names(model, optimizer.param_groups) == [["weight"], []]
    assert [g["weight_decay"] for g in optimizer.param_groups] == [WEIGHT_DECAY, 0.0]


def test_flash_builder_splits_by_decay_name_in_order(monkeypatch):
    monkeypatch.setattr(flash_adamw, "FlashAdamW", _FakeFlashAdamW)
    model = _TinyNet()
    optimizer = create_flash_adamw_optimizer(model, weight_decay=WEIGHT_DECAY, decay_parameters=DECAY_NAMES)

    assert _names(model, optimizer.param_groups) == [EXPECTED_DECAY, EXPECTED_NO_DECAY]
    assert [g["weight_decay"] for g in optimizer.param_groups] == [WEIGHT_DECAY, 0.0]


def test_flash_builder_drops_the_empty_group(monkeypatch):
    """FlashAdamW's own convention: only non-empty groups reach it."""
    monkeypatch.setattr(flash_adamw, "FlashAdamW", _FakeFlashAdamW)
    model = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    optimizer = create_flash_adamw_optimizer(model, weight_decay=WEIGHT_DECAY, decay_parameters=["weight"])

    assert _names(model, optimizer.param_groups) == [["weight"]]
    assert [g["weight_decay"] for g in optimizer.param_groups] == [WEIGHT_DECAY]


def test_flash_builder_without_decay_names_decays_every_trainable_param(monkeypatch):
    monkeypatch.setattr(flash_adamw, "FlashAdamW", _FakeFlashAdamW)
    model = _TinyNet()
    optimizer = create_flash_adamw_optimizer(model, weight_decay=WEIGHT_DECAY, decay_parameters=None)

    assert _names(model, optimizer.param_groups) == [EXPECTED_ALL_TRAINABLE]
    assert [g["weight_decay"] for g in optimizer.param_groups] == [WEIGHT_DECAY]


def test_muon_builder_splits_both_legs_by_decay_name():
    """Matrix leg and scalar leg each split decay-first, and the scalar leg keeps its own decay."""
    model = _TinyNet()
    optimizer = create_muon_optimizer(
        model,
        weight_decay=WEIGHT_DECAY,
        scalar_weight_decay=SCALAR_WEIGHT_DECAY,
        decay_parameters=DECAY_NAMES,
        ns_use_kernels=False,
    )

    # 2D before 3D: gram_newton_schulz keys its 3D->2D split on the first tensor's dim-0.
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
    """A model with no undecayed matrix and no scalar parameter ships exactly one group."""
    model = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    optimizer = create_muon_optimizer(
        model, weight_decay=WEIGHT_DECAY, decay_parameters=["weight"], ns_use_kernels=False
    )

    assert _names(model, optimizer._muon_param_groups) == [["weight"]]
    assert optimizer.scalar_optimizer is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
