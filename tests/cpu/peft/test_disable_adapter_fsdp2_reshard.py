#!/usr/bin/env python
"""``disable_adapter()`` under FSDP2 restores the sharded adapters, and reshards only when it must.

peft clears ``requires_grad`` on the adapter params registered at entry and restores it on those
registered at exit, while FSDP2 swaps the registered params between the sharded DTensors and their
unsharded copies. A single-rank gloo mesh makes those swaps for real on CPU. The trainer's wrapper
(``make_disable_adapter_fsdp2_safe``) meets two orders:

- a reference pass that is the first forward after a reshard (online GRPO / SDPG at ``beta > 0``)
  enters on the sharded params and exits on the unsharded ones. The wrapper reshards before peft's
  exit, or every sharded adapter (and ``modules_to_save`` copy) stays frozen for the rest of the run;
- a reference pass behind the policy forward (DPO, KTO, offline GRPO) enters and exits on the same
  params. Resharding there would make the backward re-gather the whole model once more per
  micro-step, so the params registered after the pass must be the ones registered before it.

Run: pytest tests/cpu/peft/test_disable_adapter_fsdp2_reshard.py
"""

import pytest
import torch
import torch.distributed as dist
from peft import get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from transformers import Qwen3Config, Qwen3ForCausalLM
from trl import ModelConfig

from src.distributed.fsdp import (
    IdentityParamSet,
    apply_fsdp2_per_layer,
    make_disable_adapter_fsdp2_safe,
    reshard_fsdp2_modules,
)
from src.distributed.loading.peft_setup import build_peft_config
from tests.common.models import TINY_QWEN3_CONFIG

TOKENS = torch.tensor([[6, 7, 8, 9, 10, 11]])


@pytest.fixture
def reshard_after_forward():
    """The toolkit default (SHARD_GRAD_OP): a forward leaves its params unsharded."""
    return False


@pytest.fixture(params=[None, ["lm_head"]], ids=["adapters", "adapters_and_modules_to_save"])
def modules_to_save(request):
    return request.param


@pytest.fixture
def sharded_lora(tmp_path, reshard_after_forward, modules_to_save):
    """A tiny LoRA Qwen3 wrapped as the trainer wraps it; yields it with the names trained at the start."""
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG)).train()
        model_config = ModelConfig(
            use_peft=True,
            lora_r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            lora_target_modules=["q_proj", "v_proj"],
            lora_modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, build_peft_config(model, model_config))
        apply_fsdp2_per_layer(
            model, init_device_mesh("cpu", (1,)), MixedPrecisionPolicy(), reshard_after_forward, IdentityParamSet()
        )
        make_disable_adapter_fsdp2_safe(model, model)
        trained = {name for name, param in model.named_parameters() if param.requires_grad}
        yield model, trained
    finally:
        dist.destroy_process_group()


def _reference_pass(model) -> None:
    with torch.no_grad(), model.disable_adapter():
        model(input_ids=TOKENS)


def _assert_sharded_training_set_unchanged(model, trained: set[str]) -> None:
    """The flags every later unshard copies live on the sharded params: exactly the trained set requires grad."""
    reshard_fsdp2_modules(model)
    params = dict(model.named_parameters())
    assert all(isinstance(params[name], DTensor) for name in trained)
    frozen = sorted(name for name in trained if not params[name].requires_grad)
    assert not frozen, f"sharded params left frozen by the reference pass: {frozen[:5]}"
    thawed = sorted(name for name, param in params.items() if param.requires_grad and name not in trained)
    assert not thawed, f"base params left trainable by the reference pass: {thawed[:5]}"


@pytest.mark.parametrize("reshard_after_forward", [False, True], ids=["shard_grad_op", "full_shard"])
def test_reference_pass_first_after_a_reshard_restores_the_sharded_params(sharded_lora):
    model, trained = sharded_lora
    reshard_fsdp2_modules(model)
    _reference_pass(model)
    _assert_sharded_training_set_unchanged(model, trained)


def test_reference_pass_behind_the_policy_forward_keeps_its_params_registered(sharded_lora):
    model, trained = sharded_lora
    loss = model(input_ids=TOKENS, labels=TOKENS).loss
    registered = dict(model.named_parameters())
    assert any(not isinstance(param, DTensor) for param in registered.values()), "the policy forward unsharded nothing"

    _reference_pass(model)

    resharded = sorted(name for name, param in registered.items() if model.get_parameter(name) is not param)
    assert not resharded, f"the reference pass resharded params the backward then re-gathers: {resharded[:5]}"
    loss.backward()
    _assert_sharded_training_set_unchanged(model, trained)
    params = dict(model.named_parameters())
    assert all(params[name].grad is not None for name in trained), "a trained param received no gradient"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
