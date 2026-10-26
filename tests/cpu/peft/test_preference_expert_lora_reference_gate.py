#!/usr/bin/env python
"""Which native EP expert-LoRA preference runs need ``precompute_ref_log_probs``.

With an attention target left after the expert peel the run is PEFT-wrapped, and TRL scores the
reference inside the ``PeftModel``'s ``disable_adapter()``, which also drops the expert adapters
(``install_ep_aware_disable_adapter``; the GPU pin is
``tests/gpu/trainers/preference/test_pref_ep_expert_lora_reference.py``). Such a run needs no
precomputed reference. An expert-only run has no ``PeftModel``: TRL would build its own unsharded fp32
dense reference on every rank, so the loader refuses it unless the reference log-probs are precomputed.

    python tests/cpu/peft/test_preference_expert_lora_reference_gate.py
"""

import types

import pytest

from src.configs.script_arguments import ModelArguments
from src.model_loading.parallel.frozen_models import load_reference_model_for_preference
from src.model_loading.parallel.peft_setup import wraps_in_peft_model
from src.models.moe.expert_lora_spec import ExpertLoraSpec

# What split_expert_lora_targets leaves behind: the attention targets of a mixed run, none of an
# expert-only one.
MIXED_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
EXPERT_ONLY_TARGETS: list[str] = []


def _reference(targets, *, precompute: bool, method: str):
    model_config = ModelArguments(model_name_or_path="org/moe", use_peft=True)
    model_config.lora_target_modules = targets
    return load_reference_model_for_preference(
        types.SimpleNamespace(),
        model_config,
        types.SimpleNamespace(precompute_ref_log_probs=precompute),
        types.SimpleNamespace(expert_lora=ExpertLoraSpec(r=8, alpha=16)),
        tokenizer=None,
        is_vlm=False,
        method=method,
        reset_sinks=True,
        attn_default=None,
    )


@pytest.mark.parametrize("method", ["DPO", "KTO"])
def test_a_mixed_run_takes_the_adapter_disabled_reference(method):
    assert _reference(MIXED_TARGETS, precompute=False, method=method) is None


@pytest.mark.parametrize("method", ["DPO", "KTO"])
def test_an_expert_only_run_without_precompute_is_refused(method):
    with pytest.raises(ValueError, match="expert-only.*precompute_ref_log_probs") as excinfo:
        _reference(EXPERT_ONLY_TARGETS, precompute=False, method=method)
    assert str(excinfo.value).startswith(method)


def test_an_expert_only_run_with_precompute_loads_no_reference():
    assert _reference(EXPERT_ONLY_TARGETS, precompute=True, method="DPO") is None


@pytest.mark.parametrize(
    ("use_peft", "targets", "expected"),
    [(True, None, True), (True, MIXED_TARGETS, True), (True, EXPERT_ONLY_TARGETS, False), (False, None, False)],
)
def test_attention_targets_distinguish_defaults_from_expert_only(use_peft, targets, expected):
    model_config = ModelArguments(model_name_or_path="org/moe", use_peft=use_peft)
    model_config.lora_target_modules = targets
    assert wraps_in_peft_model(model_config) is expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
