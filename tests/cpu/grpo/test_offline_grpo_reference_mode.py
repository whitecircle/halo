#!/usr/bin/env python
"""Every offline-GRPO reference decision follows one classification of the policy.

The constructor's run-start sweep, the CP adapter refusal, the explicit-reference gate, the script's
``requires_ref_model`` and ``_kl_reference`` all derive from ``_reference_mode``. ``_EXPECTED`` pins
each verdict over every policy kind × kl_beta × layout, mixed PEFT + native expert LoRA included.
"""

import itertools
import types
from unittest.mock import patch

import pytest
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

import src.trainers.grpo.offline as offline_module
from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.distributed.expert_parallel.config import ExpertLoraSpec
from src.trainers.grpo.offline import OfflineGRPOTrainer, ReferenceMode
from tests.common.offline_grpo import make_offline_tokenizer, offline_grpo_dataset

PartialState()

# Policy kind → (PEFT-wrapped model, peft_config passed, native expert LoRA, reference mode).
_POLICIES = {
    "full": (False, False, False, ReferenceMode.RUN_START),
    "peft_config": (False, True, False, ReferenceMode.ADAPTERS_OFF),
    "peft_model": (True, False, False, ReferenceMode.ADAPTERS_OFF),
    "expert_lora": (False, False, True, ReferenceMode.LIVE_BASE),
    "mixed_config": (False, True, True, ReferenceMode.ADAPTERS_OFF),
    "mixed_model": (True, False, True, ReferenceMode.ADAPTERS_OFF),
}
_LAYOUTS = {"plain": (False, False), "cp": (True, False), "pp": (False, True)}

# (policy, kl_beta, layout) → constructor outcome without / with an explicit ref_model,
# _holds_kl_reference, requires_ref_model, _kl_reference without / with an explicit ref_model.
_EXPECTED = {
    ("full", 0.0, "plain"): ("no-sweep", "no-sweep", False, False, "none", "refuse-unread"),
    ("full", 0.0, "cp"): ("no-sweep", "cp-ref", False, False, "none", "refuse-unread"),
    ("full", 0.0, "pp"): ("no-sweep", "no-sweep", False, False, "none", "refuse-unread"),
    ("full", 0.2, "plain"): ("sweep", "sweep", True, False, "none", "ref"),
    ("full", 0.2, "cp"): ("sweep", "cp-ref", False, False, "none", "refuse-unread"),
    ("full", 0.2, "pp"): ("sweep", "sweep", False, False, "none", "refuse-unread"),
    ("peft_config", 0.0, "plain"): ("no-sweep", "no-sweep", False, False, "none", "refuse-unread"),
    ("peft_config", 0.0, "cp"): ("cp-adapters", "cp-adapters", False, False, "none", "refuse-unread"),
    ("peft_config", 0.0, "pp"): ("no-sweep", "no-sweep", False, False, "none", "refuse-unread"),
    ("peft_config", 0.2, "plain"): ("no-sweep", "no-sweep", False, False, "none", "refuse-unread"),
    ("peft_config", 0.2, "cp"): ("cp-adapters", "cp-adapters", False, False, "none", "refuse-unread"),
    ("peft_config", 0.2, "pp"): ("no-sweep", "no-sweep", False, False, "none", "refuse-unread"),
    ("peft_model", 0.0, "plain"): ("no-sweep", "no-sweep", False, False, "none", "refuse-unread"),
    ("peft_model", 0.0, "cp"): ("cp-adapters", "cp-adapters", False, False, "none", "refuse-unread"),
    ("peft_model", 0.0, "pp"): ("no-sweep", "no-sweep", False, False, "none", "refuse-unread"),
    ("peft_model", 0.2, "plain"): ("no-sweep", "no-sweep", False, False, "none", "refuse-unread"),
    ("peft_model", 0.2, "cp"): ("cp-adapters", "cp-adapters", False, False, "none", "refuse-unread"),
    ("peft_model", 0.2, "pp"): ("no-sweep", "no-sweep", False, False, "none", "refuse-unread"),
    ("expert_lora", 0.0, "plain"): ("no-sweep", "no-sweep", False, False, "none", "refuse-unread"),
    ("expert_lora", 0.0, "cp"): ("cp-adapters", "cp-adapters", False, False, "none", "refuse-unread"),
    ("expert_lora", 0.0, "pp"): ("no-sweep", "no-sweep", False, False, "none", "refuse-unread"),
    ("expert_lora", 0.2, "plain"): ("no-sweep", "no-sweep", True, True, "refuse-needs-base", "ref"),
    ("expert_lora", 0.2, "cp"): ("cp-adapters", "cp-adapters", False, False, "none", "refuse-unread"),
    ("expert_lora", 0.2, "pp"): ("no-sweep", "no-sweep", False, False, "none", "refuse-unread"),
    ("mixed_config", 0.0, "plain"): ("no-sweep", "no-sweep", False, False, "none", "refuse-unread"),
    ("mixed_config", 0.0, "cp"): ("cp-adapters", "cp-adapters", False, False, "none", "refuse-unread"),
    ("mixed_config", 0.0, "pp"): ("no-sweep", "no-sweep", False, False, "none", "refuse-unread"),
    ("mixed_config", 0.2, "plain"): ("no-sweep", "no-sweep", False, False, "none", "refuse-unread"),
    ("mixed_config", 0.2, "cp"): ("cp-adapters", "cp-adapters", False, False, "none", "refuse-unread"),
    ("mixed_config", 0.2, "pp"): ("no-sweep", "no-sweep", False, False, "none", "refuse-unread"),
    ("mixed_model", 0.0, "plain"): ("no-sweep", "no-sweep", False, False, "none", "refuse-unread"),
    ("mixed_model", 0.0, "cp"): ("cp-adapters", "cp-adapters", False, False, "none", "refuse-unread"),
    ("mixed_model", 0.0, "pp"): ("no-sweep", "no-sweep", False, False, "none", "refuse-unread"),
    ("mixed_model", 0.2, "plain"): ("no-sweep", "no-sweep", False, False, "none", "refuse-unread"),
    ("mixed_model", 0.2, "cp"): ("cp-adapters", "cp-adapters", False, False, "none", "refuse-unread"),
    ("mixed_model", 0.2, "pp"): ("no-sweep", "no-sweep", False, False, "none", "refuse-unread"),
}


class _StopAtReferenceInputs(Exception):
    """Raised at the constructor seam that receives the run-start sweep verdict."""


def _tiny_llama():
    config = AutoConfig.for_model(
        "llama",
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
    )
    return AutoModelForCausalLM.from_config(config)


def _lora():
    return LoraConfig(r=2, target_modules=["q_proj", "v_proj"])


def _policy(kind: str):
    wrapped = _POLICIES[kind][0]
    return get_peft_model(_tiny_llama(), _lora()) if wrapped else _tiny_llama()


def _cell(kind: str, layout: str):
    _, with_config, expert_lora, _ = _POLICIES[kind]
    is_cp, is_pp = _LAYOUTS[layout]
    parallelism = types.SimpleNamespace(
        is_cp_mode=is_cp, is_pp_mode=is_pp, expert_lora=ExpertLoraSpec(r=2, alpha=4) if expert_lora else None
    )
    return parallelism, _lora() if with_config else None


def _constructor_outcome(tmp_path, model, kl_beta, parallelism, peft_config, ref_model) -> str:
    verdict = {}

    def stop(train_dataset, eval_dataset, *, active, presharded):
        verdict["sweep"] = active
        raise _StopAtReferenceInputs

    args = OfflineGRPOConfig(
        output_dir=str(tmp_path), kl_beta=kl_beta, use_cpu=True, bf16=False, use_liger_kernel=False, report_to="none"
    )
    try:
        with patch.object(offline_module, "reject_unsupported_reference_input", stop):
            OfflineGRPOTrainer(
                model=model,
                ref_model=ref_model,
                peft_config=peft_config,
                args=args,
                parallelism_config=parallelism,
                train_dataset=offline_grpo_dataset(1),
                processing_class=make_offline_tokenizer(),
            )
    except _StopAtReferenceInputs:
        return "sweep" if verdict["sweep"] else "no-sweep"
    except ValueError as error:
        if "requires a full policy fine-tune" in str(error):
            return "cp-adapters"
        if "Drop ref_model" in str(error):
            return "cp-ref"
        raise
    raise AssertionError("the constructor ran past the reference-input seam")


def _kl_reference_outcome(model, ref_model, kl_beta, parallelism, peft_config) -> str:
    try:
        held = OfflineGRPOTrainer._kl_reference(model, ref_model, kl_beta, parallelism, peft_config)
    except ValueError as error:
        if "never be read" in str(error):
            return "refuse-unread"
        if "load_frozen_reference_model" in str(error):
            return "refuse-needs-base"
        raise
    if held is None:
        return "none"
    assert held is ref_model, "the run must hold the reference it was given"
    return "ref"


def test_the_matrix_covers_every_policy_kind_kl_beta_and_layout():
    assert set(_EXPECTED) == set(itertools.product(_POLICIES, (0.0, 0.2), _LAYOUTS))


@pytest.mark.parametrize(("kind", "kl_beta", "layout"), sorted(_EXPECTED), ids=lambda value: str(value))
def test_every_reference_decision_matches_the_recorded_classification(tmp_path, kind, kl_beta, layout):
    parallelism, peft_config = _cell(kind, layout)
    ctor_bare, ctor_ref, holds, requires, kl_bare, kl_ref = _EXPECTED[(kind, kl_beta, layout)]
    reference = _tiny_llama()

    assert OfflineGRPOTrainer._reference_mode(_policy(kind), parallelism, peft_config) is _POLICIES[kind][3]
    assert _constructor_outcome(tmp_path, _policy(kind), kl_beta, parallelism, peft_config, None) == ctor_bare
    assert _constructor_outcome(tmp_path, _policy(kind), kl_beta, parallelism, peft_config, reference) == ctor_ref
    assert OfflineGRPOTrainer._holds_kl_reference(_policy(kind), kl_beta, parallelism, peft_config) is holds
    args = types.SimpleNamespace(kl_beta=kl_beta)
    assert OfflineGRPOTrainer.requires_ref_model(_policy(kind), args, parallelism, peft_config) is requires
    assert _kl_reference_outcome(_policy(kind), None, kl_beta, parallelism, peft_config) == kl_bare
    assert _kl_reference_outcome(_policy(kind), reference, kl_beta, parallelism, peft_config) == kl_ref


def test_a_checkpoint_id_policy_is_classified_by_its_config_alone():
    """Before loading, ``model`` is a path: only peft_config and expert_lora can classify it."""
    parallelism, _ = _cell("expert_lora", "plain")
    assert OfflineGRPOTrainer._reference_mode("org/base", parallelism, None) is ReferenceMode.LIVE_BASE
    assert OfflineGRPOTrainer._reference_mode("org/base", parallelism, _lora()) is ReferenceMode.ADAPTERS_OFF
    assert OfflineGRPOTrainer._reference_mode("org/base", None, None) is ReferenceMode.RUN_START


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
