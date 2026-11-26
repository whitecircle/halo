#!/usr/bin/env python
"""A PEFT wrap that leaves a tuner layer with no trainable adapter parameter is refused at trainer construction.
PEFT >= 0.21 matches parameter names against module names, so a wrapper whose ``named_parameters()`` drops a level
(``UlyssesModel``) freezes every adapter silently; the check reads the live parameters and does not depend on the
PEFT release."""

from importlib.metadata import version
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from packaging.version import Version
from peft import IA3Config, LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from halo.distributed.context_parallel.config import CPConfig
from halo.distributed.context_parallel.model_rules import SUPPORTED_ATTN_IMPLEMENTATIONS
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.model_loading.parallel.peft_setup import build_peft_model, verify_adapters_trainable
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.parallelism import make_parallelism_config

_PEFT_MATCHES_BY_NAME = Version(version("peft")) >= Version("0.21")
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


def _tiny_qwen3() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM._from_config(_TINY_QWEN3, attn_implementation="sdpa")


def _qwen3(**lora) -> LoraConfig:
    return LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], **lora)


def _under_ulysses(peft_config: LoraConfig) -> PeftModel:
    """``cp_size=1`` still wraps and patches every attention layer, so the names are the real CP ones."""
    inner = _tiny_qwen3()
    inner.config._attn_implementation = SUPPORTED_ATTN_IMPLEMENTATIONS[0]
    wrapper = UlyssesModel(inner, CPConfig(1, stage_world_size=1, stage_base_rank=0, nvlink_domain_size=1))
    model, _ = build_peft_model(wrapper, peft_config, _ARGS)
    return model


def _freeze_adapters(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(False)


def test_the_real_cp_wrapper_is_refused_exactly_where_peft_matches_names():
    model = _under_ulysses(_qwen3())

    if _PEFT_MATCHES_BY_NAME:
        with pytest.raises(RuntimeError, match=_FROZEN):
            verify_adapters_trainable(model)
    else:
        verify_adapters_trainable(model)
        assert any(p.requires_grad for n, p in model.named_parameters() if "lora_" in n)


def test_frozen_adapters_are_refused_on_every_peft_release():
    """The release-independent form of the failure: the wrap finished with every adapter frozen."""
    model = _under_ulysses(_qwen3())
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
    """CP is the mode the failure comes from; the check sits before every wrap, so it covers each mode alike."""
    model = _under_ulysses(_qwen3())
    if freeze:
        _freeze_adapters(model)
    elif _PEFT_MATCHES_BY_NAME:
        pytest.skip("PEFT >= 0.21 freezes the real CP wrapper; the frozen case above covers the call site")
    host = _ConstructionHost(model, cp_size=2)

    with patch("halo.trainers.distributed_trainer.agree_rank_uniform_env"), pytest.raises(error, match=fragment):
        DistributedTrainerMixin._configure_distributed_modes(host)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
