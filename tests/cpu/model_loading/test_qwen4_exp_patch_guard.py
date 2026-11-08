#!/usr/bin/env python
"""A Qwen3.8-Flash-Next model built without the family patches is refused at the run policies: it would train its n-gram
table as a parameter and run transformers' per-query QSA loop and [T, T] masks."""

import pytest
from torch import nn
from transformers import AutoModelForCausalLM

from halo.model_loading.model_preparation import apply_family_patches, apply_run_model_policies
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.models.families.qwen4_exp.forwards import verify_qwen4_exp_patched
from halo.models.families.qwen4_exp.ngram_table import HostNGramTable
from tests.common.qwen4_exp import NGRAM_TABLE_PATH, tiny_qwen4_exp_text_config, transformers_forwards

REMEDY = "apply_family_patches"


def _unpatched_build():
    """The config as ``config.json`` gives it, with no stamp: the class builds the stock table."""
    return AutoModelForCausalLM.from_config(tiny_qwen4_exp_text_config())


def _patched_build():
    config = tiny_qwen4_exp_text_config()
    apply_family_patches(config, "sdpa")
    return AutoModelForCausalLM.from_config(config)


def test_the_premise_a_patched_build_holds_a_host_table_and_an_unpatched_one_a_parameter():
    assert isinstance(_patched_build().get_submodule(NGRAM_TABLE_PATH), HostNGramTable)
    assert _unpatched_build().get_parameter(f"{NGRAM_TABLE_PATH}.weight").requires_grad


def test_an_unstamped_build_is_refused_with_its_table_and_the_remedy():
    with pytest.raises(RuntimeError, match=REMEDY) as refused:
        verify_qwen4_exp_patched(_unpatched_build())
    assert "model.layers.1.ple.ple_embedding" in str(refused.value)


def test_a_patched_build_passes():
    verify_qwen4_exp_patched(_patched_build())


def test_a_model_built_before_the_forwards_were_installed_is_refused():
    model = _patched_build()
    with transformers_forwards(), pytest.raises(RuntimeError, match="forwards are transformers' own"):
        verify_qwen4_exp_patched(model)


def test_the_run_policies_refuse_an_unpatched_model():
    model = _unpatched_build()
    with pytest.raises(RuntimeError, match=REMEDY):
        apply_run_model_policies(model, model.config, sinks_policy=SinksPolicy.NEUTRALIZED, attn_implementation="sdpa")


def test_a_stamped_config_without_split_ngram_parts_is_refused():
    config = tiny_qwen4_exp_text_config()
    apply_family_patches(config, "sdpa")
    object.__setattr__(config, "split_ngram_parts", None)  # the strict config class refuses None on assignment
    with pytest.raises(ValueError, match="sets no split_ngram_parts"):
        AutoModelForCausalLM.from_config(config)


def test_a_model_without_flash_next_modules_is_not_checked():
    verify_qwen4_exp_patched(nn.Linear(2, 2))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
