#!/usr/bin/env python
"""QLoRA bases must fail at trainer construction on either rollout engine, not opaquely at the first sync.

A dense-model QLoRA RL run constructs cleanly (the loader's rejection covers only MoE + EP/TP/
grouped-GEMM), and ``_send_dense_weights`` then ships the bnb ``Params4bit`` packed uint8 storage
under base-weight names — the server fails opaquely after full startup. ``validate_weight_sync_support``
is the construction gate, run by the init spine both the online and environmental GRPO trainers close through.

Run: ``python tests/cpu/grpo/test_grpo_weight_sync_qlora_gate.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import inspect
import types

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState

# The online install logs through accelerate's logger, which requires an initialized state.
PartialState()

from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from src.trainers.grpo.mixins.on_policy_init import OnPolicyGRPOInitMixin
from src.trainers.grpo.online import DistributedGRPOTrainer
from src.trainers.grpo.rollout.weight_sync import validate_weight_sync_support


class _QuantizedStub(nn.Module):
    """Float adapter over a bnb-shaped packed base: uint8 storage, requires_grad=False —
    exactly what ``Params4bit`` looks like to ``named_parameters``."""

    def __init__(self):
        super().__init__()
        self.packed_base = nn.Parameter(torch.zeros(8, dtype=torch.uint8), requires_grad=False)
        self.lora_A = nn.Parameter(torch.zeros(4, 4))


class _FloatStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_gate_rejects_quantized_model(backend):
    with pytest.raises(ValueError, match="QLoRA .* not supported with rollout-engine weight sync"):
        validate_weight_sync_support(_QuantizedStub(), backend)


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_gate_passes_float_model(backend):
    validate_weight_sync_support(_FloatStub(), backend)  # must not raise


def _install_host(model, vllm_generation):
    """Real trainer object (never ``__init__``-ed) carrying only what the install reads —
    a real one so ``type(self)._distributed_sync_weights`` resolves."""
    me = object.__new__(DistributedGRPOTrainer)
    me.model = model
    me.vllm_generation = vllm_generation
    me.parallelism_config = types.SimpleNamespace(is_ep_mode=False, is_tp_mode=False, is_expert_tp_mode=False)
    return me


def test_online_setup_weight_sync_replaces_trl_sync_on_float_model():
    """The install must actually bind the distributed-aware sync — TRL's own ``sync_weights``
    forwards DTensors verbatim and deadlocks against the trainer↔vLLM NCCL group."""
    me = _install_host(_FloatStub(), types.SimpleNamespace())
    DistributedGRPOTrainer._setup_weight_sync(me)
    assert me.vllm_generation.sync_weights.__func__ is DistributedGRPOTrainer._distributed_sync_weights


def test_online_setup_weight_sync_raises_without_vllm_generation():
    """TRL builds ``vllm_generation`` on EVERY rank whenever ``use_vllm`` is set (only its client is
    main-only), and this trainer requires server-mode vLLM — so a missing one means generation is
    not wired at all. Returning quietly would leave TRL's own ``sync_weights`` in place and every
    rollout would come from an engine that never receives the trained weights."""
    me = _install_host(_FloatStub(), None)
    with pytest.raises(RuntimeError, match="vllm_generation"):
        DistributedGRPOTrainer._setup_weight_sync(me)


_SPINE_STEPS = (
    "_setup_distributed_modes",
    "_validate_implicit_reference_model",
    "_resolve_chunked_head_transform",
    "_setup_weight_sync",
    "_disable_dropout_for_onpolicy",
)


def _spine_host(model, backend: str, ran: list[str]):
    return types.SimpleNamespace(
        **{step: (lambda step=step: ran.append(step)) for step in _SPINE_STEPS},
        model=model,
        _rollout_backend=backend,
        _loss_logits_width=lambda: 7,
        _check_full_logits_fit=lambda width: ran.append(f"_check_full_logits_fit({width})"),
    )


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_the_spine_gates_a_quantized_model_before_anything_can_push(backend):
    """The gate runs on the trainer's own engine, ahead of the install: a QLoRA model fails at
    construction rather than mid-broadcast at the first push."""
    ran: list[str] = []
    with pytest.raises(ValueError, match="QLoRA .* not supported with rollout-engine weight sync"):
        OnPolicyGRPOInitMixin._finish_on_policy_init(_spine_host(_QuantizedStub(), backend, ran))
    assert "_setup_weight_sync" not in ran, "the sync was wired for a model it cannot ship"


def test_both_trainers_reach_the_gate_through_the_shared_init_spine():
    """Both on-policy trainers close their ctor through ``_finish_on_policy_init``, the one place the
    gate is called — so a refactor that drops the call, or a trainer that bypasses the spine, fails
    here rather than at the first sync."""
    ran: list[str] = []
    OnPolicyGRPOInitMixin._finish_on_policy_init(_spine_host(_FloatStub(), "vllm", ran))

    # Order matters as much as presence: dropout must be killed on the modules the mode setup
    # realized, the logits plane is weighed against memory the placed model left free, and the gates
    # must run before anything can push weights.
    assert ran == [
        "_setup_distributed_modes",
        "_validate_implicit_reference_model",
        "_resolve_chunked_head_transform",
        "_check_full_logits_fit(7)",
        "_setup_weight_sync",
        "_disable_dropout_for_onpolicy",
    ], f"the shared init spine no longer runs its gates in order: {ran}"
    for cls in (DistributedGRPOTrainer, DistributedAsyncEnvironmentalGRPOTrainer):
        assert "_finish_on_policy_init" in inspect.getsource(cls.__init__), (
            f"{cls.__name__}.__init__ no longer closes through the shared spine"
        )
        assert cls._loss_logits_width is not OnPolicyGRPOInitMixin._loss_logits_width, (
            f"{cls.__name__} never states its logits width"
        )
    assert DistributedGRPOTrainer._setup_weight_sync is not OnPolicyGRPOInitMixin._setup_weight_sync, (
        "the online trainer no longer installs its distributed-aware sync"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
