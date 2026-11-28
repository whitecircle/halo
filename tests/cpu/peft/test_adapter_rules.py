#!/usr/bin/env python
"""Adapter refusals read off a live model: each fires on its own condition and names its mechanism. The refusal of a
``PeftModel`` below the FSDP2 root fires exactly where a reference pass would leave adapters frozen."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from transformers import Qwen3Config, Qwen3ForCausalLM

from halo.distributed.fsdp import IdentityParamSet, apply_fsdp2_per_layer, reshard_fsdp2_modules
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.parallelism.adapter_rules import (
    reject_expert_lora_peft_conflicts,
    reject_peft_lora_in_ep_layers,
    reject_peft_model_below_fsdp2_root,
    reject_unrealized_expert_lora,
)
from tests.common.cp_model import unpatched_cp_wrapper
from tests.common.ep_stubs import StubEPLayerBase
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.parallelism import make_parallelism_config

HIDDEN, EXPERTS, RANK = 8, 2, 2
_SPEC = ExpertLoraSpec(r=RANK, alpha=2 * RANK)


class _MoEHost(nn.Module):
    """An attention projection beside an EP layer's shared-expert ``nn.Linear`` and fused expert bank, with or without
    native adapters."""

    def __init__(self, *, native_adapters: bool):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.mlp = StubEPLayerBase()
        self.mlp.shared_expert = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.mlp.down_proj = nn.Parameter(torch.zeros(EXPERTS, HIDDEN, HIDDEN))
        if native_adapters:
            self.mlp._expert_lora_attrs = frozenset({"gate_up_proj"})
            self.mlp.gate_up_proj_lora_A = nn.Parameter(torch.zeros(EXPERTS, HIDDEN, RANK))
            self.mlp.gate_up_proj_lora_B = nn.Parameter(torch.zeros(EXPERTS, RANK, 2 * HIDDEN))

    def forward(self, x):
        return self.q_proj(x)


def _peft(targets: list[str], *, native_adapters: bool = True, **lora):
    return get_peft_model(_MoEHost(native_adapters=native_adapters), LoraConfig(target_modules=targets, **lora))


def test_the_supported_adapter_layouts_pass():
    """Native adapters beside a PEFT attention adapter of a matching recipe, an expert-only run, and no expert LoRA."""
    attention_only = _peft(["q_proj"], r=RANK, lora_alpha=2 * RANK)
    reject_peft_lora_in_ep_layers(attention_only)
    reject_expert_lora_peft_conflicts(attention_only, _SPEC)
    reject_unrealized_expert_lora(attention_only, _SPEC)

    expert_only = _MoEHost(native_adapters=True)
    reject_expert_lora_peft_conflicts(expert_only, _SPEC)
    reject_unrealized_expert_lora(expert_only, _SPEC)
    reject_unrealized_expert_lora(_MoEHost(native_adapters=False), None)


@pytest.mark.parametrize(
    ("refusal", "match"),
    [
        (
            lambda: reject_peft_lora_in_ep_layers(_peft(["shared_expert"])),
            r"PEFT-wrapped LoRA adapters detected inside or around EP-wrapped MoE layers"
            r"(?s:.*)mlp\.shared_expert\.lora_A\.default\.weight",
        ),
        (
            lambda: reject_peft_lora_in_ep_layers(_peft(["q_proj"], target_parameters=["mlp.down_proj"])),
            r"inside or around EP-wrapped MoE layers(?s:.*)base_model\.model\.mlp \(a ParamWrapper around an EP layer\)",
        ),
        (
            lambda: reject_expert_lora_peft_conflicts(_peft(["q_proj"], use_rslora=True), _SPEC),
            r"settings the native grouped expert adapters cannot reproduce(?s:.*)use_rslora=True",
        ),
        (
            lambda: reject_expert_lora_peft_conflicts(_peft(["q_proj"], use_dora=True), _SPEC),
            r"settings the native grouped expert adapters cannot reproduce(?s:.*)use_dora=True",
        ),
        (
            lambda: reject_unrealized_expert_lora(_MoEHost(native_adapters=False), _SPEC),
            r"routed to native EP grouped-LoRA .* no EP/grouped MoE layer built the adapters",
        ),
    ],
    ids=["peft-lora-in-ep-layer", "peft-lora-around-ep-layer", "rslora-mismatch", "dora", "unrealized-expert-lora"],
)
def test_each_adapter_refusal_names_its_mechanism(refusal, match):
    with pytest.raises(ValueError, match=match):
        refusal()


TOKENS = torch.tensor([[6, 7, 8, 9, 10, 11]])
# Adapters by where they sit: in the decoder layers' groups, on the head the root group holds, or on the embedding the
# backbone's group holds.
_ADAPTERS = {
    "attention": {"target_modules": ["q_proj", "v_proj"]},
    "head_copy": {"target_modules": ["q_proj", "v_proj"], "modules_to_save": ["lm_head"]},
    "head_lora": {"target_modules": ["q_proj", "lm_head"]},
    "embedding_lora": {"target_modules": ["q_proj", "embed_tokens"]},
}
_IN_THE_ROOT_GROUP = {"head_copy", "head_lora"}


@pytest.fixture
def one_gloo_rank(tmp_path):
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        yield
    finally:
        dist.destroy_process_group()


def _lora_qwen3(adapters: str, *, below_root: bool):
    """A tiny LoRA Qwen3, under the CP wrapper (the PeftModel below the root) when ``below_root``."""
    torch.manual_seed(0)
    base = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG)).train()
    peft_model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, init_lora_weights=False, **_ADAPTERS[adapters]))
    return (unpatched_cp_wrapper(peft_model) if below_root else peft_model), peft_model


def _shard(root: nn.Module, *, zero3: bool) -> None:
    """The trainer's FSDP2 wrap of a mode's root: decoder-layer units, the embedding backbone, the root."""
    apply_fsdp2_per_layer(root, init_device_mesh("cpu", (1,)), MixedPrecisionPolicy(), zero3, IdentityParamSet())


def _left_frozen_by_a_reference_pass(root: nn.Module, peft_model) -> list[str]:
    """The trained params a ``disable_adapter()`` pass, first after a reshard, leaves frozen on the shards."""
    trained = {name for name, param in root.named_parameters() if param.requires_grad}
    reshard_fsdp2_modules(root)
    with torch.no_grad(), peft_model.disable_adapter():
        root(input_ids=TOKENS)
    reshard_fsdp2_modules(root)
    params = dict(root.named_parameters())
    return sorted(name for name in trained if not params[name].requires_grad)


@pytest.mark.parametrize("zero3", [False, True], ids=["zero2", "zero3"])
@pytest.mark.parametrize("below_root", [True, False], ids=["peft_below_root", "peft_root"])
@pytest.mark.parametrize("adapters", sorted(_ADAPTERS))
def test_the_below_root_refusal_fires_where_a_reference_pass_leaves_adapters_frozen(
    one_gloo_rank, adapters, below_root, zero3
):
    root, peft_model = _lora_qwen3(adapters, below_root=below_root)
    _shard(root, zero3=zero3)
    fails = below_root and not zero3 and adapters in _IN_THE_ROOT_GROUP

    try:
        reject_peft_model_below_fsdp2_root(root, reshard_after_forward=zero3)
        refusal = None
    except ValueError as error:
        refusal = str(error)

    assert (refusal is not None) == fails, refusal
    frozen = _left_frozen_by_a_reference_pass(root, peft_model)
    assert bool(frozen) == fails, frozen
    if fails:
        assert "under ZeRO-2" in refusal and "lm_head" in refusal


class _PastTheRootCheck(Exception):
    """Carries control out of the construction spine after the below-root refusal."""


class _ConstructionHost(DistributedTrainerMixin):
    """The real ``_configure_distributed_modes`` on one gloo rank, with the trainer's FSDP2 wrap of the root."""

    _deferred_liger_kernel = False
    _accelerate_manages_fsdp = False
    label_smoother = None
    compute_loss_func = None

    def __init__(self, model, *, zero3: bool):
        self.model = model
        self.parallelism_config = make_parallelism_config(
            world_size=1, gpus_per_node=1, fsdp_reshard_after_forward=zero3
        )
        self.args = SimpleNamespace(
            load_best_model_at_end=False,
            optim="adamw_torch",
            per_device_train_batch_size=1,
            eval_strategy="no",
            eval_on_start=False,
        )
        self._bf16_optimizer = False
        self._ep_config = None
        self._fsdp_wrapped = False

    def _cast_peft_params_to_compute_dtype(self):
        """The tiny model is fp32 throughout."""

    def _wrap_for_parallel_mode(self):
        _shard(self.model, zero3=self.parallelism_config.fsdp_reshard_after_forward)
        self._fsdp_wrapped = True

    def _fsdp_excluded_params(self):
        raise _PastTheRootCheck


@pytest.mark.parametrize(
    ("adapters", "zero3", "refused"),
    [("head_copy", False, True), ("attention", False, False), ("head_copy", True, False)],
    ids=["head_copy-zero2", "attention-zero2", "head_copy-zero3"],
)
def test_trainer_construction_runs_the_below_root_refusal_after_the_wrap(one_gloo_rank, adapters, zero3, refused):
    root, _ = _lora_qwen3(adapters, below_root=True)
    error, match = (ValueError, "PeftModel sits inside a UlyssesModel") if refused else (_PastTheRootCheck, None)

    with patch("halo.trainers.distributed_trainer.agree_rank_uniform_env"), pytest.raises(error, match=match):
        DistributedTrainerMixin._configure_distributed_modes(_ConstructionHost(root, zero3=zero3))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
