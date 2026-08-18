#!/usr/bin/env python
"""Offline GRPO's dense KL reference must refuse a checkpoint that does not cover it.

``_create_full_ft_reference`` deepcopies the policy for dense models, but a wrapped-MoE policy holds
live NCCL groups that cannot be pickled, so the reference is RELOADED from the checkpoint path.
``from_pretrained`` randomly initializes every key it cannot find and reports it through the
transformers logger only — which ``setup_logging`` silences off the logging rank. A KL reference that
is part noise biases every step's objective (the policy is pulled toward random logits) with no
exception, no NaN and no other symptom, so the load has to be coverage-gated.

Run: ``python tests/cpu/grpo/test_offline_grpo_kl_reference_coverage.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.models.loading.checkpoint_coverage import ALLOW_MISSING_CHECKPOINT_KEYS_ENV
from src.trainers.grpo.offline import OfflineGRPOTrainer

PartialState()  # the trainer's accelerate logger requires an initialized state

# Present in the model, dropped from the checkpoint: inside the backbone, tied to nothing, so no
# class-derived allowance can excuse it.
DROPPED_KEY = "model.layers.0.mlp.down_proj.weight"


class _EPLayerStub(EPMoELayerBase):
    """Marks the policy as wrapped-MoE — the branch that must reload instead of deepcopying."""

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, hidden_states, **kwargs):  # pragma: no cover - never called
        return hidden_states

    def expert_named_params(self):  # pragma: no cover - never called
        return []


def _tiny_llama():
    config = AutoConfig.for_model(
        "llama",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.config._attn_implementation = "eager"
    return model


def _checkpoint(tmp_path, *, drop: str | None) -> str:
    path = tmp_path / ("truncated" if drop else "complete")
    _tiny_llama().save_pretrained(path, safe_serialization=True)
    if drop is not None:
        shard = path / "model.safetensors"
        state = load_file(shard)
        assert drop in state, f"{drop} is not in this checkpoint — the test drops nothing"
        del state[drop]
        save_file(state, shard, metadata={"format": "pt"})
    return str(path)


def _wrapped_moe_policy():
    """A policy the KL reference cannot be deepcopied from (holds an EP layer)."""
    policy = _tiny_llama()
    policy.model.layers[0].mlp = _EPLayerStub()
    assert any(isinstance(module, EPMoELayerBase) for module in policy.modules())
    return policy


def test_truncated_checkpoint_is_refused(tmp_path):
    """The reference would be part pretrained, part RNG — and nothing downstream could tell."""
    policy = _wrapped_moe_policy()

    with pytest.raises(RuntimeError, match="randomly initialized"):
        OfflineGRPOTrainer._create_full_ft_reference(policy, _checkpoint(tmp_path, drop=DROPPED_KEY))


def test_complete_checkpoint_still_loads(tmp_path):
    """Anti-over-rejection: a legitimate reference load must not be blocked."""
    policy = _wrapped_moe_policy()

    reference = OfflineGRPOTrainer._create_full_ft_reference(policy, _checkpoint(tmp_path, drop=None))

    assert reference is not policy
    assert reference.config._attn_implementation == "eager", "the policy's attn implementation was not carried over"
    assert reference.dtype == policy.dtype


def test_the_escape_hatch_accepts_the_absence(tmp_path, monkeypatch):
    """The gate is deliberately overridable — a run that knows better must be able to proceed."""
    monkeypatch.setenv(ALLOW_MISSING_CHECKPOINT_KEYS_ENV, "1")
    policy = _wrapped_moe_policy()

    reference = OfflineGRPOTrainer._create_full_ft_reference(policy, _checkpoint(tmp_path, drop=DROPPED_KEY))

    assert isinstance(reference, nn.Module)


def test_dense_policy_still_deepcopies():
    """Anti-vacuity for the branch under test: a dense policy never reaches the reload at all."""
    policy = _tiny_llama()

    reference = OfflineGRPOTrainer._create_full_ft_reference(policy, "")  # no path needed for a deepcopy

    assert reference is not policy
    assert torch.equal(reference.lm_head.weight, policy.lm_head.weight)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
