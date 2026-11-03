#!/usr/bin/env python
"""QLoRA bases fail at trainer construction on either engine. Otherwise ``_send_dense_weights`` ships the packed
uint8 storage under base-weight names and the server fails after full startup."""

from __future__ import annotations

import inspect
import types

import pytest
import torch
import torch.nn as nn

from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.on_policy import OnPolicyGRPOInitMixin
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer
from halo.trainers.grpo.rollout.weight_push import reject_unsyncable_model


class _QuantizedStub(nn.Module):
    """Float adapter over a packed base, as ``Params4bit`` looks to ``named_parameters``."""

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
        reject_unsyncable_model(_QuantizedStub(), backend)


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_gate_passes_float_model(backend):
    reject_unsyncable_model(_FloatStub(), backend)


def _install_host(model, vllm_generation):
    """A real trainer object, never ``__init__``-ed, so ``type(self)._distributed_sync_weights`` resolves."""
    me = object.__new__(DistributedOnlineGRPOTrainer)
    me.model = model
    me.vllm_generation = vllm_generation
    me.parallelism_config = types.SimpleNamespace(has_ep_group=False, is_tp_mode=False, is_expert_tp_mode=False)
    return me


def test_online_install_weight_sync_replaces_trl_sync_on_float_model():
    """TRL's own ``sync_weights`` forwards DTensors verbatim and deadlocks against the trainer-vLLM NCCL group."""
    me = _install_host(_FloatStub(), types.SimpleNamespace())
    DistributedOnlineGRPOTrainer._install_weight_sync(me)
    assert me.vllm_generation.sync_weights.__func__ is DistributedOnlineGRPOTrainer._distributed_sync_weights


def test_online_install_weight_sync_raises_without_vllm_generation():
    """TRL builds ``vllm_generation`` on every rank under ``use_vllm``, so a missing one means generation is not wired.
    Returning quietly would leave rollouts on an engine that never receives the trained weights."""
    me = _install_host(_FloatStub(), None)
    with pytest.raises(RuntimeError, match="vllm_generation"):
        DistributedOnlineGRPOTrainer._install_weight_sync(me)


_SPINE_STEPS = (
    "_configure_distributed_modes",
    "_validate_held_reference_model",
    "_resolve_chunked_head_transform",
    "_install_weight_sync",
    "_disable_dropout_for_onpolicy",
)


def _spine_host(model, backend: str, ran: list[str]):
    return types.SimpleNamespace(
        **{step: (lambda step=step: ran.append(step)) for step in _SPINE_STEPS},
        model=model,
        _rollout_backend=backend,
        _loss_logits_width=lambda: 7,
        _verify_full_logits_fit=lambda width: ran.append(f"_verify_full_logits_fit({width})"),
    )


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_the_spine_gates_a_quantized_model_before_anything_can_push(backend):
    ran: list[str] = []
    with pytest.raises(ValueError, match="QLoRA .* not supported with rollout-engine weight sync"):
        OnPolicyGRPOInitMixin._finish_on_policy_init(_spine_host(_QuantizedStub(), backend, ran))
    assert "_install_weight_sync" not in ran, "the sync was wired for a model it cannot ship"


def test_both_trainers_reach_the_gate_through_the_shared_init_spine():
    """``_finish_on_policy_init`` is the one place the gate is called."""
    ran: list[str] = []
    OnPolicyGRPOInitMixin._finish_on_policy_init(_spine_host(_FloatStub(), "vllm", ran))

    # Order matters: dropout is killed on the modules the mode setup realized, the logits plane is weighed against
    # memory the placed model left free, and the gates run before anything can push weights.
    assert ran == [
        "_configure_distributed_modes",
        "_validate_held_reference_model",
        "_resolve_chunked_head_transform",
        "_verify_full_logits_fit(7)",
        "_install_weight_sync",
        "_disable_dropout_for_onpolicy",
    ], f"the shared init spine no longer runs its gates in order: {ran}"
    for cls in (DistributedOnlineGRPOTrainer, DistributedAsyncEnvironmentalGRPOTrainer):
        assert "_finish_on_policy_init" in inspect.getsource(cls.__init__), (
            f"{cls.__name__}.__init__ no longer closes through the shared spine"
        )
        assert "_loss_logits_width" in vars(cls), f"{cls.__name__} never states its logits width"
    assert DistributedOnlineGRPOTrainer._install_weight_sync is not OnPolicyGRPOInitMixin._install_weight_sync, (
        "the online trainer no longer installs its distributed-aware sync"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
