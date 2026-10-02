#!/usr/bin/env python
"""Offline GRPO's KL reference for a wrapped-MoE full fine-tune is the canonical frozen load.

A dense policy's reference is a deepcopy of it. An EP / grouped-GEMM wrapped MoE policy holds live
NCCL process groups ``deepcopy`` cannot pickle, so its reference is a separate load of the policy's
weights, and a load outside the frozen-reference path skips what makes the two log-probs
comparable: the non-persistent buffer repair (a garbage ``inv_freq`` on remote-code families), the
GptOss sinks policy (neutralized policy sinks against live reference sinks put a KL on step 0), the
family attention patches, and the run's own remote-code flag. A biased reference shifts every step's
objective with no other symptom, so what is pinned: the script loads the reference exactly as it
loaded the policy, the trainer refuses a wrapped MoE that arrives without one instead of loading its
own, and a reference handed to a run that holds none is refused rather than ignored. The reference
forward itself runs with grad off and, where the run holds no reference model, through the PEFT
policy with its adapters disabled: either one dropped scores the KL against the live policy.

Run: ``python tests/cpu/grpo/test_offline_grpo_kl_reference.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import contextlib
import types

import pytest
import torch
from accelerate import PartialState
from peft import LoraConfig
from transformers import AutoConfig, AutoModelForCausalLM
from trl import ModelConfig

import scripts.training.offline_grpo as offline_grpo_script
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.grpo.offline import OfflineGRPOTrainer
from tests.common.ep_stubs import StubEPLayerBase
from tests.common.frozen_loader import captured_load, stub_frozen_loader

PartialState()  # the trainer's accelerate logger requires an initialized state

BASE = "org/base"
RESUMED = "/runs/offline-grpo/checkpoint-40"
PIPELINE = types.SimpleNamespace(is_pp_mode=True)


def _tiny_llama():
    config = AutoConfig.for_model(
        "llama",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
    )
    return AutoModelForCausalLM.from_config(config)


def _wrapped_moe_policy():
    """A policy holding an EP layer: the shape ``deepcopy`` cannot reproduce."""
    policy = _tiny_llama()
    policy.model.layers[0].mlp = StubEPLayerBase()
    return policy


def _config(kl_beta: float = 0.1):
    return types.SimpleNamespace(kl_beta=kl_beta, bf16=True, fp16=False)


@pytest.mark.parametrize(
    "policy_fn,kl_beta,parallelism,peft_config,expected",
    [
        (_wrapped_moe_policy, 0.1, ParallelismConfig(), None, True),
        (_tiny_llama, 0.1, ParallelismConfig(), None, False),
        (_wrapped_moe_policy, 0.0, ParallelismConfig(), None, False),
        (_wrapped_moe_policy, 0.1, ParallelismConfig(), LoraConfig(), False),
        (_wrapped_moe_policy, 0.1, PIPELINE, None, False),
    ],
    ids=["wrapped-moe", "dense", "no-kl", "peft", "pipeline"],
)
def test_only_a_wrapped_moe_full_finetune_needs_a_loaded_reference(
    policy_fn, kl_beta, parallelism, peft_config, expected
):
    assert OfflineGRPOTrainer.requires_ref_model(policy_fn(), _config(kl_beta), parallelism, peft_config) is expected


def test_a_wrapped_moe_without_a_loaded_reference_is_refused():
    """The trainer holds none of the load flags the reference must share with the policy, so it
    refuses rather than loading a reference of its own."""
    with pytest.raises(ValueError, match="load_frozen_reference_model"):
        OfflineGRPOTrainer._kl_reference(_wrapped_moe_policy(), None, 0.1, ParallelismConfig(), None)


def test_a_loaded_reference_is_the_one_the_run_holds():
    reference = _tiny_llama()
    held = OfflineGRPOTrainer._kl_reference(_wrapped_moe_policy(), reference, 0.1, ParallelismConfig(), None)
    assert held is reference


def test_a_dense_policy_still_deepcopies():
    policy = _tiny_llama()
    reference = OfflineGRPOTrainer._kl_reference(policy, None, 0.1, ParallelismConfig(), None)
    assert reference is not policy
    assert torch.equal(reference.lm_head.weight, policy.lm_head.weight)
    assert not any(param.requires_grad for param in reference.parameters())


@pytest.mark.parametrize(
    "kl_beta,parallelism,peft_config",
    [(0.0, ParallelismConfig(), None), (0.1, ParallelismConfig(), LoraConfig()), (0.1, PIPELINE, None)],
    ids=["no-kl", "peft", "pipeline"],
)
def test_a_reference_the_run_would_never_read_is_refused(kl_beta, parallelism, peft_config):
    policy = _wrapped_moe_policy()
    assert OfflineGRPOTrainer._kl_reference(policy, None, kl_beta, parallelism, peft_config) is None
    with pytest.raises(ValueError, match="never be read"):
        OfflineGRPOTrainer._kl_reference(policy, _tiny_llama(), kl_beta, parallelism, peft_config)


def _script_args():
    return types.SimpleNamespace(
        eos_token=None,
        bos_token=None,
        pad_token=None,
        chat_template=None,
        force_chat_template=False,
        added_special_tokens=None,
        tokenizer_backend="hf",
    )


def _tokenizer():
    return types.SimpleNamespace(
        eos_token="<e>",
        eos_token_id=1,
        bos_token="<b>",
        bos_token_id=2,
        pad_token="<p>",
        pad_token_id=3,
        chat_template="{{ messages }}",
    )


def _load(policy, *, model_source: str, reset_sinks: bool):
    """The script's reference load for ``policy``, as ``main`` calls it."""
    return offline_grpo_script._load_kl_reference(
        _script_args(),
        types.SimpleNamespace(parallelism_config=ParallelismConfig(), model_source=model_source),
        _config(),
        ModelConfig(model_name_or_path=BASE, model_revision="abc123", trust_remote_code=False),
        types.SimpleNamespace(reset_sinks=reset_sinks),
        policy=policy,
        tokenizer=_tokenizer(),
        peft_config=None,
        attn_default="sdpa",
    )


def test_the_script_loads_the_reference_as_it_loaded_the_policy():
    """Same weights source (the resumed checkpoint on a resume), revision, remote-code flag, class
    resolution, attention request and sinks policy; then the tokenizer's ids, as the policy got them."""
    with stub_frozen_loader() as caps:
        reference = _load(_wrapped_moe_policy(), model_source=RESUMED, reset_sinks=False)
    captured = captured_load(caps)

    assert reference is captured.model
    assert captured.load_positional[0] == RESUMED
    assert captured.config["revision"] == captured.load["revision"] == "abc123"
    assert captured.load["trust_remote_code"] is False
    assert captured.load["dtype"] is torch.bfloat16
    assert captured.resolver["attn_implementation"] == "sdpa"
    assert captured.resolver["sinks_reset"] is False
    captured.freeze_sinks.assert_called_once()
    assert reference.generation_config.pad_token_id == 3, "the tokenizer setup never reached the reference"


def test_the_script_loads_nothing_where_the_trainer_derives_the_reference():
    with stub_frozen_loader() as caps:
        assert _load(_tiny_llama(), model_source=BASE, reset_sinks=True) is None
    caps.auto_load.assert_not_called()


class _PeftPolicy:
    """The policy's ``disable_adapter()`` switch, recording whether a forward ran with the adapters off."""

    def __init__(self):
        self.adapters_off = False

    @contextlib.contextmanager
    def disable_adapter(self):
        self.adapters_off = True
        try:
            yield
        finally:
            self.adapters_off = False


class _ReferenceReached(Exception):
    """Ends ``_compute_loss_inner`` once the reference forward has been recorded."""


def _loss_forwards(ref_model) -> tuple[list[dict], _PeftPolicy]:
    """Each ``_get_per_token_logps`` call ``_compute_loss_inner`` makes under ``kl_beta > 0``: the model
    it forwarded, and whether grad and the policy's adapters were on during it."""
    policy = _PeftPolicy()
    forwards = []

    def per_token_logps(model, input_ids, *_args, **_kwargs):
        forwards.append({"model": model, "grad": torch.is_grad_enabled(), "adapters_off": policy.adapters_off})
        if len(forwards) == 2:
            raise _ReferenceReached
        logps = torch.zeros(input_ids.size(0), 2)
        return logps, logps

    host = types.SimpleNamespace(
        model=policy,
        min_log_prob=None,
        beta=0.1,
        ref_model=ref_model,
        accelerator=types.SimpleNamespace(unwrap_model=lambda model: model),
        _get_per_token_logps=per_token_logps,
    )
    inputs = {
        "prompt_input_ids": torch.zeros(1, 3, dtype=torch.long),
        "prompt_attention_mask": torch.ones(1, 3, dtype=torch.long),
        "completion_input_ids": torch.zeros(1, 2, dtype=torch.long),
        "completion_attention_mask": torch.ones(1, 2, dtype=torch.long),
        "advantage": torch.ones(1),
    }
    with pytest.raises(_ReferenceReached):
        OfflineGRPOTrainer._compute_loss_inner(host, policy, inputs)
    return forwards, policy


def test_without_a_reference_model_the_reference_is_the_policy_with_adapters_off_and_no_grad():
    (policy_forward, reference_forward), policy = _loss_forwards(ref_model=None)
    assert policy_forward == {"model": policy, "grad": True, "adapters_off": False}
    assert reference_forward == {"model": policy, "grad": False, "adapters_off": True}


def test_a_held_reference_model_forwards_with_no_grad_and_the_policy_adapters_untouched():
    reference = object()
    (_, reference_forward), _ = _loss_forwards(ref_model=reference)
    assert reference_forward == {"model": reference, "grad": False, "adapters_off": False}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
