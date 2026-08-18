#!/usr/bin/env python
"""PEFT vLLM weight-sync must broadcast the *merged* adapter, not the reverted base.

``gather_and_send_weights`` merges the LoRA adapter into the base for the body, forwards the base
params to the vLLM client, then unmerges. The vendored client buffers ``update_named_param`` payloads
**by reference** (``.contiguous()`` is a no-op on an already-contiguous tensor) and flushes them later
via ``reset_prefix_cache``. For a plain (non-DTensor) param — single-process or DDP PEFT runs — the
unmerge then reverts that aliased storage *before* the flush, so vLLM would serve the base model and
on-policy RL would silently train against the wrong policy.

This reproduces the buffer-by-reference + late-flush ordering with a recording fake client and asserts
the flushed weight equals the merged weight (``W + B @ A``), not the base ``W``. It is a CPU test: the
aliasing is independent of the device or the real NCCL transport.

Run: ``python tests/cpu/peft/test_vllm_weight_sync_peft.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from src.models.structure import normalize_peft_param_name
from src.trainers.grpo.rollout.weight_sync import gather_and_send_weights


class _TinyModel(nn.Module):
    """Minimal module with a single LoRA-targetable linear (no download)."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):  # pragma: no cover - never called
        return self.proj(x)


class _RecordingClient:
    """Faithful stand-in for the vendored NCCL client's buffer-by-reference + late-flush semantics."""

    def __init__(self):
        self._buffer: list[tuple[str, torch.Tensor]] = []
        self.flushed: dict[str, torch.Tensor] = {}

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        # Mirrors BaseWeightSyncClient.update_named_param exactly (no clone).
        self._buffer.append((name, weights.contiguous()))

    def reset_prefix_cache(self) -> None:
        # The broadcast happens here, after gather_and_send_weights has unmerged the adapter.
        self.flushed = {name: tensor.detach().clone() for name, tensor in self._buffer}
        self._buffer = []


def _build_lora_model() -> nn.Module:
    torch.manual_seed(0)
    base = _TinyModel()
    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["proj"], task_type=None)
    model = get_peft_model(base, cfg)
    # PEFT zero-inits lora_B, which would make merged == base and hide the bug. Give the adapter a
    # non-trivial delta so W + B@A differs from W.
    for name, param in model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                param.copy_(torch.randn_like(param))
    return model


def _merged_reference(model: nn.Module) -> dict[str, torch.Tensor]:
    """The base-named weights as they should appear *with the adapter folded in*."""
    model.merge_adapter()
    try:
        ref = {}
        for name, param in model.named_parameters():
            normalized = normalize_peft_param_name(name, model.prefix)
            if normalized is not None:
                ref[normalized] = param.detach().clone()
    finally:
        model.unmerge_adapter()
    return ref


def test_peft_sync_broadcasts_merged_not_base():
    model = _build_lora_model()
    merged = _merged_reference(model)
    assert "proj.weight" in merged

    client = _RecordingClient()
    peft = gather_and_send_weights(model, client)
    # Caller flushes after the gather returns (adapter already unmerged) — the moment the real bug bit.
    client.reset_prefix_cache()

    assert peft is True
    assert "proj.weight" in client.flushed, "base weight was never forwarded"
    # No adapter-only params should leak to vLLM.
    assert not any("lora_" in k for k in client.flushed)

    base_weight = model.base_model.model.proj.base_layer.weight.detach()
    sent = client.flushed["proj.weight"]

    # The fix: the flushed weight is the merged adapter, decoupled from the later unmerge.
    assert torch.allclose(sent, merged["proj.weight"], atol=1e-6), (
        "vLLM received weights that do not match the merged adapter (W + B@A)"
    )
    # And it must NOT be the reverted base (the adapter delta is non-trivial by construction).
    assert not torch.allclose(sent, base_weight, atol=1e-6), (
        "vLLM received the reverted base weights — the merge was aliased away by unmerge"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
