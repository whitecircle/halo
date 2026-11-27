#!/usr/bin/env python
"""PEFT's wrap rewrites a regex or ``all-linear`` target on a transformers v5 fused-expert MoE: it adds the fused
experts and the router as ``target_parameters`` and widens a regex to bare leaf names. The run's config build refuses
a target PEFT would rewrite, before any wrap, and keeps every target it would not."""

import pytest
from peft import LoraConfig
from transformers import Qwen3Config, Qwen3ForCausalLM

from halo.configs.script_arguments import ModelArguments
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.model_loading.parallel.peft_setup import build_peft_config, reject_peft_moe_target_rewrite
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.parallelism import single_process_ep_config
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

_EXPERTS_REGEX = r".*\.(q_proj|v_proj|gate_proj|up_proj|down_proj)"


def _moe(*, ep: bool):
    model = tiny_family_model(TINY_MOE_FAMILIES["qwen3_moe"])
    if ep:
        apply_ep_layers(model, single_process_ep_config(experts_fsdp_managed=False))
    return model


@pytest.mark.parametrize("ep", [False, True], ids=["hf_moe", "ep_moe"])
@pytest.mark.parametrize("target", [_EXPERTS_REGEX, "all-linear"], ids=["regex", "all_linear"])
def test_a_target_peft_would_turn_into_expert_parameters_is_refused(ep, target):
    with pytest.raises(ValueError, match=r"would adapt target_parameters \[.*'down_proj'.*'gate_up_proj'"):
        reject_peft_moe_target_rewrite(_moe(ep=ep), LoraConfig(target_modules=target))


def test_a_regex_peft_would_widen_to_every_layer_is_refused():
    with pytest.raises(ValueError, match=r"modules the pattern does not name \(e\.g\. model\.layers\.1\.self_attn"):
        reject_peft_moe_target_rewrite(_moe(ep=False), LoraConfig(target_modules=r".*\.layers\.0\.self_attn\.q_proj"))


@pytest.mark.parametrize(
    "target",
    [["q_proj", "v_proj"], r".*\.self_attn\.(q_proj|v_proj)", ["model.layers.0.self_attn.q_proj"]],
    ids=["names", "attention_regex", "paths"],
)
def test_a_target_peft_keeps_as_written_passes(target):
    reject_peft_moe_target_rewrite(_moe(ep=True), LoraConfig(target_modules=target))


def test_a_dense_model_keeps_its_regex():
    reject_peft_moe_target_rewrite(
        Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG)), LoraConfig(target_modules="all-linear")
    )


def test_the_runs_config_build_refuses_it():
    model_config = ModelArguments(use_peft=True, lora_r=2, lora_alpha=4, lora_target_modules=_EXPERTS_REGEX)
    with pytest.raises(ValueError, match="which the target as written does not name"):
        build_peft_config(_moe(ep=True), model_config)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
