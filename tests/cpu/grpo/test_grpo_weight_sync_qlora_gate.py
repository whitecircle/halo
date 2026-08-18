#!/usr/bin/env python
"""QLoRA × vLLM weight sync must fail at trainer construction, not opaquely at the first sync.

A dense-model QLoRA RL run constructs cleanly (the loader's rejection covers only MoE + EP/TP/
grouped-GEMM), and ``_send_dense_weights`` then ships the bnb ``Params4bit`` packed uint8 storage
under base-weight names — the vLLM server fails opaquely after full startup, and each sync's LoRA
merge/unmerge round-trip through 4-bit weights is lossy. ``validate_weight_sync_support`` is the
construction gate; both the online and environmental GRPO trainers must wire it.

Run: ``python tests/cpu/grpo/test_grpo_weight_sync_qlora_gate.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import inspect
import types

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState

# The install logs through accelerate's logger, which requires an initialized state.
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


def test_gate_rejects_quantized_model():
    with pytest.raises(ValueError, match="QLoRA .* not supported with vLLM weight sync"):
        validate_weight_sync_support(_QuantizedStub())


def test_gate_passes_float_model():
    validate_weight_sync_support(_FloatStub())  # must not raise


def _install_host(model, vllm_generation):
    """Real trainer object (never ``__init__``-ed) carrying only what the install reads —
    a real one so ``type(self)._distributed_sync_weights`` resolves."""
    me = object.__new__(DistributedGRPOTrainer)
    me.model = model
    me.vllm_generation = vllm_generation
    me.parallelism_config = types.SimpleNamespace(is_ep_mode=False, is_tp_mode=False, is_expert_tp_mode=False)
    return me


def test_online_setup_weight_sync_gates_quantized_model():
    """The online trainer's weight-sync install (called in ``__init__``) must run the gate BEFORE
    anything else, so a QLoRA model fails at construction rather than at the first sync."""
    me = _install_host(_QuantizedStub(), types.SimpleNamespace())
    with pytest.raises(ValueError, match="QLoRA .* not supported with vLLM weight sync"):
        DistributedGRPOTrainer._setup_weight_sync(me)


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


def _env_sync_host(model):
    """Env trainer object carrying only what its ``_setup_weight_sync`` reads."""
    me = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    me.model = model
    me.async_config = types.SimpleNamespace(rollout_backend="vllm")
    me.parallelism_config = types.SimpleNamespace(
        is_ep_mode=False, is_tp_mode=False, is_expert_tp_mode=False, ep_size=1, expert_tp_size=1, ep_group_size=1
    )
    return me


def test_environmental_setup_weight_sync_gates_quantized_model():
    """The env trainer has no sync to install, so its gate IS the whole seam: it must reject a
    quantized model at construction rather than mid-broadcast at the first push."""
    with pytest.raises(ValueError, match="QLoRA .* not supported with vLLM weight sync"):
        DistributedAsyncEnvironmentalGRPOTrainer._setup_weight_sync(_env_sync_host(_QuantizedStub()))


def test_environmental_setup_weight_sync_passes_a_float_model():
    """Anti-over-rejection: an ordinary float model must construct."""
    DistributedAsyncEnvironmentalGRPOTrainer._setup_weight_sync(_env_sync_host(_FloatStub()))


def test_both_trainers_reach_the_gate_through_the_shared_init_spine():
    """Both on-policy trainers close their ctor through ``_finish_on_policy_init``, which is the one
    place the gate is called — so a refactor that drops the call, or leaves a trainer on the mixin's
    ``NotImplementedError`` stub, must fail here rather than at the first vLLM sync."""
    ran: list[str] = []
    host = types.SimpleNamespace(
        **{
            step: (lambda step=step: ran.append(step))
            for step in (
                "_setup_distributed_modes",
                "_validate_implicit_reference_model",
                "_setup_weight_sync",
                "_disable_dropout_for_onpolicy",
            )
        }
    )
    OnPolicyGRPOInitMixin._finish_on_policy_init(host)

    # Order matters as much as presence: dropout must be killed on the modules the mode setup
    # realized, and the gate must run before anything can push weights.
    assert ran == [
        "_setup_distributed_modes",
        "_validate_implicit_reference_model",
        "_setup_weight_sync",
        "_disable_dropout_for_onpolicy",
    ], f"the shared init spine no longer runs the weight-sync gate in order: {ran}"
    for cls in (DistributedGRPOTrainer, DistributedAsyncEnvironmentalGRPOTrainer):
        assert "_finish_on_policy_init" in inspect.getsource(cls.__init__), (
            f"{cls.__name__}.__init__ no longer closes through the shared spine"
        )
        assert cls._setup_weight_sync is not OnPolicyGRPOInitMixin._setup_weight_sync, (
            f"{cls.__name__} never overrides the gate stub"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
