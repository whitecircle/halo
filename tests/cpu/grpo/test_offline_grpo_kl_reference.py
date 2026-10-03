#!/usr/bin/env python
"""Offline GRPO precomputes full-finetuning anchors; native expert LoRA keeps its frozen base."""

from __future__ import annotations

import contextlib
import types

import pytest
import torch
from accelerate import PartialState
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoConfig, AutoModelForCausalLM
from trl import ModelConfig

import scripts.training.offline_grpo as offline_grpo_script
import src.trainers.grpo.reference_logps as reference_logps_module
from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from src.distributed.expert_parallel.config import ExpertLoraSpec
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.grpo.offline import OfflineGRPOTrainer
from tests.common.ep_stubs import StubEPLayerBase
from tests.common.frozen_loader import captured_load, stub_frozen_loader
from tests.common.offline_grpo import make_offline_tokenizer, offline_grpo_dataset
from tests.common.offline_grpo_reference import mapped_scores

PartialState()  # the trainer's accelerate logger requires an initialized state

BASE = "org/base"
RESUMED = "/runs/offline-grpo/checkpoint-40"
PIPELINE = types.SimpleNamespace(is_pp_mode=True, is_cp_mode=False)


def _native_expert_lora_config():
    return ParallelismConfig(expert_lora=ExpertLoraSpec(r=2, alpha=4))


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


@pytest.mark.parametrize("kl_beta", [0.0, 0.1])
@pytest.mark.parametrize("chunked", [False, True])
def test_missing_parallelism_config_preserves_the_required_config_diagnostic(tmp_path, kl_beta, chunked):
    args = OfflineGRPOConfig(
        output_dir=str(tmp_path),
        kl_beta=kl_beta,
        use_chunked_grpo_logprobs=chunked,
        use_cpu=True,
        bf16=False,
        use_liger_kernel=False,
        report_to="none",
    )
    with pytest.raises(ValueError, match="parallelism_config is required.*Pass a ParallelismConfig instance"):
        OfflineGRPOTrainer(
            model=_tiny_llama(),
            args=args,
            train_dataset=_anchor_dataset(),
            processing_class=types.SimpleNamespace(pad_token_id=0),
        )


@pytest.mark.parametrize(
    "policy_fn,kl_beta,parallelism,peft_config,expected",
    [
        (_wrapped_moe_policy, 0.1, ParallelismConfig(), None, False),
        (_wrapped_moe_policy, 0.1, _native_expert_lora_config(), None, True),
        (_tiny_llama, 0.1, ParallelismConfig(), None, False),
        (_wrapped_moe_policy, 0.0, ParallelismConfig(), None, False),
        (_wrapped_moe_policy, 0.1, ParallelismConfig(), LoraConfig(), False),
        (_wrapped_moe_policy, 0.1, PIPELINE, None, False),
    ],
    ids=["wrapped-full-ft", "native-expert-lora", "dense-full-ft", "no-kl", "peft", "pipeline"],
)
def test_only_native_expert_lora_needs_a_loaded_reference(policy_fn, kl_beta, parallelism, peft_config, expected):
    assert OfflineGRPOTrainer.requires_ref_model(policy_fn(), _config(kl_beta), parallelism, peft_config) is expected


def test_native_expert_lora_without_a_loaded_reference_is_refused():
    with pytest.raises(ValueError, match="load_frozen_reference_model"):
        OfflineGRPOTrainer._kl_reference(_wrapped_moe_policy(), None, 0.1, _native_expert_lora_config(), None)


def test_a_loaded_reference_is_the_one_the_run_holds():
    reference = _tiny_llama()
    held = OfflineGRPOTrainer._kl_reference(_wrapped_moe_policy(), reference, 0.1, ParallelismConfig(), None)
    assert held is reference


@pytest.mark.parametrize("native_expert_lora", [False, True], ids=["full-ft-sweep", "native-expert-lora-live"])
def test_constructor_places_and_freezes_every_external_reference(tmp_path, monkeypatch, native_expert_lora):
    policy, reference = _tiny_llama(), _tiny_llama()
    assert reference.training and all(parameter.requires_grad for parameter in reference.parameters())
    original_weights = [parameter.detach().clone() for parameter in reference.parameters()]
    args = OfflineGRPOConfig(
        output_dir=str(tmp_path),
        kl_beta=0.2,
        use_cpu=True,
        bf16=False,
        use_liger_kernel=False,
        report_to="none",
        max_prompt_length=16,
        max_completion_length=16,
        dataset_num_proc=1,
        remove_unused_columns=False,
    )
    # Native adapter installation needs DeepEP; reference ownership is the constructor's seam.
    monkeypatch.setattr(OfflineGRPOTrainer, "_setup_distributed_modes", lambda self: None)
    trainer = OfflineGRPOTrainer(
        model=policy,
        ref_model=reference,
        args=args,
        train_dataset=offline_grpo_dataset(2),
        processing_class=make_offline_tokenizer(),
        parallelism_config=_native_expert_lora_config() if native_expert_lora else ParallelismConfig(),
    )
    assert trainer._precompute_reference is not native_expert_lora
    assert (trainer.ref_model is reference) is native_expert_lora
    assert not reference.training
    assert all(not parameter.requires_grad for parameter in reference.parameters())
    assert all(parameter.requires_grad for parameter in policy.parameters()), "reference freezing reached the policy"
    assert reference.device == policy.device
    for before, after in zip(original_weights, reference.parameters(), strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def _raw_model_reference(model, dataset):
    rows = []
    with torch.no_grad():
        for row in dataset:
            prompt, completion = row["prompt_input_ids"], row["completion_input_ids"]
            ids = torch.tensor([prompt + completion])
            logits = model(input_ids=ids).logits[:, len(prompt) - 1 : -1].float()
            targets = torch.tensor(completion).view(1, -1, 1)
            rows.append(logits.log_softmax(-1).gather(-1, targets).flatten())
    return rows


def test_explicit_reference_scores_both_splits_once_instead_of_the_live_policy(tmp_path):
    torch.manual_seed(711)
    policy = _tiny_llama()
    torch.manual_seed(991)
    reference = _tiny_llama()
    reference_calls = []
    original_forward = reference.forward

    def observed_forward(*args, **kwargs):
        reference_calls.append(1)
        return original_forward(*args, **kwargs)

    reference.forward = observed_forward
    args = OfflineGRPOConfig(
        output_dir=str(tmp_path),
        kl_beta=0.2,
        use_cpu=True,
        bf16=False,
        use_liger_kernel=False,
        report_to="none",
        max_prompt_length=16,
        max_completion_length=16,
        per_device_train_batch_size=2,
        dataset_num_proc=1,
        remove_unused_columns=False,
    )
    trainer = OfflineGRPOTrainer(
        model=policy,
        ref_model=reference,
        args=args,
        train_dataset=offline_grpo_dataset(2),
        eval_dataset=offline_grpo_dataset(2, 4),
        processing_class=make_offline_tokenizer(),
        parallelism_config=ParallelismConfig(),
    )
    assert len(reference_calls) == 4, "the supplied reference was not swept once over each split"
    assert trainer.ref_model is None
    assert not reference.training and not any(parameter.requires_grad for parameter in reference.parameters())
    for dataset in (trainer.train_dataset, trainer.eval_dataset):
        reference.forward = original_forward
        expected = _raw_model_reference(reference, dataset)
        policy.eval()
        live = _raw_model_reference(policy, dataset)
        assert max((ref - own).abs().max().item() for ref, own in zip(expected, live, strict=True)) > 1e-3
        for row, score in zip(dataset, expected, strict=True):
            torch.testing.assert_close(torch.tensor(row[REF_PER_TOKEN_LOGPS_COLUMN]), score, rtol=1e-6, atol=1e-6)
    reference.forward = lambda *args, **kwargs: pytest.fail("released reference was scored again")
    batch = trainer.data_collator([trainer.train_dataset[index] for index in range(2)])
    assert torch.isfinite(trainer._compute_loss_inner(policy, batch))
    assert len(reference_calls) == 4


@pytest.mark.parametrize("policy_fn", [_tiny_llama, _wrapped_moe_policy], ids=["dense", "wrapped-moe"])
def test_full_finetuning_sweeps_the_policy_without_constructing_a_reference_copy(policy_fn):
    policy = policy_fn()
    assert OfflineGRPOTrainer._kl_reference(policy, None, 0.1, ParallelismConfig(), None) is None
    assert all(parameter.requires_grad for parameter in policy.parameters())


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


def _load(policy, *, model_source: str, reset_sinks: bool, parallelism=None):
    """The script's reference load for ``policy``, as ``main`` calls it."""
    return offline_grpo_script._load_kl_reference(
        _script_args(),
        types.SimpleNamespace(parallelism_config=parallelism or ParallelismConfig(), model_source=model_source),
        _config(),
        ModelConfig(model_name_or_path=BASE, model_revision="abc123", trust_remote_code=False),
        types.SimpleNamespace(reset_sinks=reset_sinks),
        policy=policy,
        tokenizer=_tokenizer(),
        peft_config=None,
        attn_default="sdpa",
    )


def test_native_expert_lora_keeps_the_original_base_when_the_policy_resumes():
    """The unadapted base uses the policy's load settings, never its trained checkpoint."""
    with stub_frozen_loader() as caps:
        reference = _load(
            _wrapped_moe_policy(), model_source=RESUMED, reset_sinks=False, parallelism=_native_expert_lora_config()
        )
    captured = captured_load(caps)

    assert reference is captured.model
    assert captured.load_positional[0] == BASE
    assert captured.config["revision"] == captured.load["revision"] == "abc123"
    assert captured.load["trust_remote_code"] is False
    assert captured.load["dtype"] is torch.bfloat16
    assert captured.resolver["attn_implementation"] == "sdpa"
    assert captured.resolver["sinks_reset"] is False
    captured.freeze_sinks.assert_called_once()
    assert reference.generation_config.pad_token_id == 3, "the tokenizer setup never reached the reference"


@pytest.mark.parametrize("policy_fn", [_tiny_llama, _wrapped_moe_policy], ids=["dense", "wrapped-moe"])
def test_the_script_loads_nothing_for_a_precomputed_full_finetuning_reference(policy_fn):
    with stub_frozen_loader() as caps:
        assert _load(policy_fn(), model_source=BASE, reset_sinks=True) is None
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
        _precompute_reference=False,
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


def _anchor_trainer(*, checkpoint=None, trained=False, output_dir=None):
    trainer = OfflineGRPOTrainer.__new__(OfflineGRPOTrainer)
    trainer.model = _tiny_llama()
    with torch.no_grad():
        trainer.model.lm_head.weight.fill_(-7.0 if trained else -0.25)
    trainer.ref_model = None
    trainer.parallelism_config = ParallelismConfig()
    trainer.temperature = 0.75
    trainer.beta = 0.2
    trainer.args = types.SimpleNamespace(
        disable_dropout=True, resume_from_checkpoint=checkpoint, output_dir=output_dir or checkpoint
    )
    trainer._dataset_presharded = False
    trainer._init_reference_logps(resume_checkpoint=checkpoint)
    trainer.sweep_count = 0

    def sweep(dataset, split):
        trainer.sweep_count += 1
        initial = float(trainer.model.lm_head.weight[0, 0].detach())
        rows = [torch.arange(len(row), dtype=torch.float32).neg() + initial for row in dataset["completion_input_ids"]]
        return mapped_scores(trainer.args.output_dir, dataset, rows)

    trainer._sweep_reference_logps = sweep
    return trainer


def _anchor_dataset():
    return Dataset.from_dict({"prompt_input_ids": [[1, 2], [5]], "completion_input_ids": [[3, 4], [6]]})


def test_noncp_full_finetuning_restores_original_reference_instead_of_sweeping_trained_weights(tmp_path):
    first = _anchor_trainer(output_dir=str(tmp_path))
    original = first._precompute_reference_logps(_anchor_dataset(), "training")
    assert first.sweep_count == 1
    assert original[REF_PER_TOKEN_LOGPS_COLUMN] == [[-0.25, -1.25], [-0.25]]
    first._persist_trainer_sidecars(str(tmp_path))

    resumed = _anchor_trainer(checkpoint=str(tmp_path), trained=True)
    restored = resumed._precompute_reference_logps(_anchor_dataset(), "training")
    assert resumed.sweep_count == 0, "resume re-anchored KL to the trained policy"
    assert restored[REF_PER_TOKEN_LOGPS_COLUMN] == original[REF_PER_TOKEN_LOGPS_COLUMN]
    assert float(resumed.model.lm_head.weight[0, 0].detach()) == -7.0


def test_fresh_reference_split_hashes_each_token_column_once_before_scoring(tmp_path, monkeypatch):
    trainer = _anchor_trainer(output_dir=str(tmp_path))
    dataset = _anchor_dataset()
    original_digest = reference_logps_module.token_digest
    digests = {}

    def observed_digest(dataset, column):
        assert column not in digests, f"fresh reference split hashed '{column}' twice"
        digests[column] = original_digest(dataset, column)
        return digests[column]

    monkeypatch.setattr(reference_logps_module, "token_digest", observed_digest)
    original_sweep = trainer._sweep_reference_logps

    def sweep_after_identity(dataset, split):
        assert set(digests) == {"prompt_input_ids", "completion_input_ids"}, "scoring preceded input validation"
        return original_sweep(dataset, split)

    trainer._sweep_reference_logps = sweep_after_identity
    attached = trainer._precompute_reference_logps(dataset, "training")
    assert attached[REF_PER_TOKEN_LOGPS_COLUMN] == [[-0.25, -1.25], [-0.25]]
    assert trainer._reference_logps_by_split["training"]["token_digests"] == digests
    assert trainer.sweep_count == 1


def test_explicit_reference_does_not_allow_a_resume_to_replace_its_missing_anchor(tmp_path):
    resumed = _anchor_trainer(checkpoint=str(tmp_path), trained=True)
    resumed.ref_model = _tiny_llama()
    with pytest.raises(RuntimeError, match="TRAINED checkpoint policy"):
        resumed._precompute_reference_logps(_anchor_dataset(), "training")
    assert resumed.sweep_count == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
