#!/usr/bin/env python
"""What PEFT's wrap leaves trainable, repaired and checked at trainer construction. PEFT keeps trainable only the
parameters whose ``named_parameters()`` name lies under a tuner layer's ``named_modules()`` name, so a wrapper whose
names drop a level (``UlyssesModel``) freezes every adapter, and the native grouped expert adapters, in no tuner layer,
freeze too. Construction marks both trainable again; the check refuses any tuner layer still left with nothing that
trains."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from peft import IA3Config, LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from halo.distributed.context_parallel.config import CPConfig
from halo.distributed.context_parallel.model_rules import SUPPORTED_ATTN_IMPLEMENTATIONS
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.model_loading.parallel.peft_setup import (
    apply_cp_adapter_trainability,
    apply_expert_lora_trainability,
    build_peft_model,
    verify_adapters_trainable,
)
from halo.models.introspection.unwrapping import plain_adapter_key
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.parallelism import make_parallelism_config, single_process_ep_config
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

_TINY_QWEN3 = Qwen3Config(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    max_position_embeddings=64,
    tie_word_embeddings=False,
)
_ARGS = SimpleNamespace(gradient_checkpointing=False, bf16=False)
_FROZEN = r"hold no trainable parameter after the PEFT wrap.*peft#4181"


def _tiny_qwen3(**config) -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    return Qwen3ForCausalLM._from_config(
        Qwen3Config(**{**_TINY_QWEN3.to_dict(), **config}), attn_implementation="sdpa"
    )


def _qwen3(**lora) -> LoraConfig:
    return LoraConfig(**{"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"], **lora})


def _under_ulysses(peft_config: LoraConfig, **config) -> PeftModel:
    """``cp_size=1`` still wraps and patches every attention layer, so the names are the real CP ones."""
    inner = _tiny_qwen3(**config)
    inner.config._attn_implementation = SUPPORTED_ATTN_IMPLEMENTATIONS[0]
    wrapper = UlyssesModel(inner, CPConfig(1, stage_world_size=1, stage_base_rank=0, nvlink_domain_size=1))
    model, _ = build_peft_model(wrapper, peft_config, _ARGS)
    return model


def _freeze_adapters(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(False)


def test_peft_leaves_every_adapter_under_the_real_cp_wrapper_frozen():
    """The failure the CP repair exists for, and that the check alone refuses."""
    model = _under_ulysses(_qwen3())

    with pytest.raises(RuntimeError, match=_FROZEN):
        verify_adapters_trainable(model)


# A LoRA bias and bias="lora_only" need biased projections; bias="all" trains every bias, adapter or not.
_BIASED = {"attention_bias": True}
_ADAPTER_KINDS = {
    "lora": ({}, {}),
    "dora": ({"use_dora": True}, {}),
    "rslora_lora_bias": ({"use_rslora": True, "lora_bias": True}, _BIASED),
    "bias_all": ({"bias": "all"}, _BIASED),
    "bias_lora_only": ({"bias": "lora_only"}, _BIASED),
    "modules_to_save": ({"modules_to_save": ["lm_head"]}, {}),
    "modules_to_save_embedding_and_head": ({"modules_to_save": ["embed_tokens", "lm_head"]}, {}),
    "trainable_tokens": ({"trainable_token_indices": [1, 2, 3]}, {}),
    "embedding_lora": ({"target_modules": ["q_proj", "embed_tokens"]}, {}),
    "target_parameters": ({"target_modules": ["q_proj"], "target_parameters": ["layers.0.mlp.down_proj.weight"]}, {}),
    "inference_mode": ({"inference_mode": True}, {}),
    "mica": ({"init_lora_weights": "mica"}, {}),
}


@pytest.mark.parametrize("kind", sorted(_ADAPTER_KINDS))
def test_the_cp_repair_trains_what_the_plain_wrap_trains(kind):
    lora, config = _ADAPTER_KINDS[kind]
    plain = get_peft_model(_tiny_qwen3(**config), _qwen3(**lora))
    expected = {name for name, param in plain.named_parameters() if param.requires_grad}
    model = _under_ulysses(_qwen3(**lora), **config)

    apply_cp_adapter_trainability(model)

    trained = {plain_adapter_key(name) for name, param in model.named_parameters() if param.requires_grad}
    assert trained == expected, (sorted(expected - trained)[:3], sorted(trained - expected)[:3])


@pytest.mark.parametrize("active", ["default", "other"])
def test_the_cp_repair_trains_the_active_adapter_alone(active):
    model = _under_ulysses(_qwen3())
    model.add_adapter("other", _qwen3(target_modules=["q_proj", "k_proj"]))
    model.set_adapter(active)
    _freeze_adapters(model)

    apply_cp_adapter_trainability(model)

    trained = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trained == {name for name, _ in model.named_parameters() if f".{active}." in name}


@pytest.mark.parametrize(
    "lora",
    [{"modules_to_save": ["lm_head"]}, {"trainable_token_indices": [1, 2, 3]}],
    ids=["modules_to_save", "tokens"],
)
def test_the_cp_repair_trains_what_peft_wraps_beside_its_tuner_layers(lora):
    model = _under_ulysses(_qwen3(**lora))
    wrapped = {name for name, param in model.named_parameters() if param.requires_grad and "lora_" not in name}
    assert wrapped, "premise: the wrap left the wrapper's copy trainable"
    for param in model.parameters():
        param.requires_grad_(False)

    apply_cp_adapter_trainability(model)

    assert wrapped <= {name for name, param in model.named_parameters() if param.requires_grad}


def test_frozen_adapters_are_refused():
    """The wrap finished with every adapter frozen, whatever froze them."""
    model = get_peft_model(_tiny_qwen3(), _qwen3())
    _freeze_adapters(model)

    with pytest.raises(RuntimeError, match=_FROZEN):
        verify_adapters_trainable(model)


def test_one_frozen_adapter_layer_among_trainable_ones_is_refused():
    model = get_peft_model(_tiny_qwen3(), _qwen3())
    frozen_path = "base_model.model.model.layers.1.self_attn.q_proj"
    for param in model.get_submodule(frozen_path).parameters():
        param.requires_grad_(False)

    with pytest.raises(RuntimeError, match=rf"1 adapter layers.*{frozen_path}"):
        verify_adapters_trainable(model)


@pytest.mark.parametrize(
    "lora",
    [{}, {"use_dora": True}, {"modules_to_save": ["lm_head"]}, {"target_modules": ["embed_tokens", "q_proj"]}],
    ids=["lora", "dora", "modules_to_save", "embedding"],
)
def test_a_trained_adapter_of_every_kind_passes(lora):
    peft_config = _qwen3()
    for key, value in lora.items():
        setattr(peft_config, key, value)
    model = get_peft_model(_tiny_qwen3(), peft_config)
    assert any(p.requires_grad for p in model.parameters()), "premise: something trains"

    verify_adapters_trainable(model)


def test_only_the_active_adapter_must_train():
    model = get_peft_model(_tiny_qwen3(), _qwen3())
    model.add_adapter("other", _qwen3())
    model.set_adapter("other")
    for name, param in model.named_parameters():
        if "lora_" in name and ".default." in name:
            param.requires_grad_(False)
    assert not any(p.requires_grad for n, p in model.named_parameters() if ".default." in n), "premise"

    verify_adapters_trainable(model)


def test_an_inference_mode_adapter_is_not_expected_to_train():
    model = get_peft_model(_tiny_qwen3(), _qwen3(inference_mode=True))
    assert not any(p.requires_grad for p in model.parameters()), "premise: PEFT froze everything"

    verify_adapters_trainable(model)


def test_a_non_lora_tuner_is_counted_by_its_own_layer_names():
    model = get_peft_model(
        _tiny_qwen3(),
        IA3Config(target_modules=["q_proj", "down_proj"], feedforward_modules=["down_proj"]),
    )
    verify_adapters_trainable(model)
    _freeze_adapters_by_prefix = [p for n, p in model.named_parameters() if "ia3_" in n]
    for param in _freeze_adapters_by_prefix:
        param.requires_grad_(False)

    with pytest.raises(RuntimeError, match=_FROZEN):
        verify_adapters_trainable(model)


class _PastAdapterCheck(Exception):
    """Carries control out of the construction spine after the adapter check."""


class _ConstructionHost(DistributedTrainerMixin):
    def __init__(self, model, **parallelism):
        self.model = model
        self.args = SimpleNamespace(optim="adamw_torch")
        self._bf16_optimizer = False
        with patch("halo.distributed.parallelism_config.validate_nvlink_domain_against_fabric"):
            self.parallelism_config = make_parallelism_config(world_size=2, gpus_per_node=2, **parallelism)

    def _restore_trl_overrides(self):
        """No TRL constructor ran, so there is nothing to take back."""

    def _disable_inert_router_logits(self):
        raise _PastAdapterCheck


@pytest.mark.parametrize(
    ("freeze", "error", "fragment"),
    [(True, RuntimeError, _FROZEN), (False, _PastAdapterCheck, None)],
    ids=["frozen", "trained"],
)
def test_trainer_construction_runs_the_check(freeze, error, fragment):
    """The check sits before every wrap, so it covers each mode alike; adapters frozen outside a CP wrapper are what
    no repair reaches."""
    model = get_peft_model(_tiny_qwen3(), _qwen3())
    if freeze:
        _freeze_adapters(model)
    host = _ConstructionHost(model)

    with patch("halo.trainers.distributed_trainer.agree_rank_uniform_env"), pytest.raises(error, match=fragment):
        DistributedTrainerMixin._configure_distributed_modes(host)


def test_trainer_construction_repairs_the_cp_wrap():
    model = _under_ulysses(_qwen3())
    _freeze_adapters(model)
    host = _ConstructionHost(model, cp_size=2)

    with patch("halo.trainers.distributed_trainer.agree_rank_uniform_env"), pytest.raises(_PastAdapterCheck):
        DistributedTrainerMixin._configure_distributed_modes(host)
    assert all(p.requires_grad for n, p in model.named_parameters() if "lora_" in n)


def _mixed_expert_lora_model() -> tuple[PeftModel, list]:
    """Native grouped expert adapters made trainable as ``configure_adapter_training`` does, then the stock PEFT wrap
    of the attention targets."""
    model = tiny_family_model(TINY_MOE_FAMILIES["qwen3_moe"])
    ep_config = single_process_ep_config(expert_lora=ExpertLoraSpec(r=8, alpha=16), experts_fsdp_managed=False)
    apply_ep_layers(model, ep_config)
    for param in model.parameters():
        param.requires_grad_(False)
    apply_expert_lora_trainability(model)
    experts = [param for _, layer in named_ep_layers(model) for _, param in layer.expert_lora_named_params()]
    assert experts and all(param.requires_grad for param in experts), "premise: expert adapters built and trained"
    return get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])), experts


def test_peft_freezes_the_native_expert_adapters():
    _, experts = _mixed_expert_lora_model()
    assert not any(param.requires_grad for param in experts)


def test_trainer_construction_trains_the_native_expert_adapters():
    model, experts = _mixed_expert_lora_model()
    host = _ConstructionHost(model)

    with patch("halo.trainers.distributed_trainer.agree_rank_uniform_env"), pytest.raises(_PastAdapterCheck):
        DistributedTrainerMixin._configure_distributed_modes(host)
    frozen = sum(not param.requires_grad for param in experts)
    assert not frozen, f"{frozen} of {len(experts)} native expert adapters left frozen after construction"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
