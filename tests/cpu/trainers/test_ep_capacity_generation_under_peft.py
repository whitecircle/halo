#!/usr/bin/env python
"""The EP capacity generation must keep advancing after PEFT wraps the EP-patched model.

``_ElasticBackend.ensure`` sizes the DeepEP arena once per forward and scopes that capacity with a
generation token a forward pre-hook bumps. EP patching leaves that hook on the model it patched, and
a task-typed ``PeftModel`` — what every adapter run builds — then reaches that model through
``BaseTuner.forward``'s direct ``self.model.forward(...)`` call, which runs no pre-hook. The
generation then freezes at the run's first forward: every later step is served the first batch's
capacity, and the first longer batch is refused outright by the dedup guard. Silent for a step or
two, then fatal mid-run, so the hook has to ride the module the training loop calls — after TRL's
own ``get_peft_model`` that is the trainer's ``model_wrapped``, not the model the loader patched.

Run: ``python tests/cpu/trainers/test_ep_capacity_generation_under_peft.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import inspect

import pytest
import torch
from accelerate import PartialState
from datasets import Dataset
from peft import LoraConfig, PeftModel
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Trainer
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
from trl import SFTConfig

PartialState()  # the trainer logs through accelerate's logger, which refuses an uninitialized state

from src.distributed.expert_parallel import dispatcher as dispatcher_mod
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.patching import patch_moe_model_for_ep
from src.trainers.sft import DistributedSFTTrainer
from tests.common.models import TINY_GEMMA4_MOE_CONFIG
from tests.common.parallelism import make_parallelism_config, single_process_ep_config

_INPUT_IDS = [3, 4, 5, 6]


def _tokenizer() -> PreTrainedTokenizerFast:
    """A vocab-wide WordLevel tokenizer: TRL demands a processing class, this run never tokenizes."""
    vocab = {f"t{index}": index for index in range(TINY_GEMMA4_MOE_CONFIG["vocab_size"])}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="t0"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="t0", pad_token="t1", eos_token="t2")


def _ep_patched_model() -> Gemma4ForCausalLM:
    """A tiny MoE model carrying EP wrappers — and the generation hook EP patching registers with them."""
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(Gemma4TextConfig(**TINY_GEMMA4_MOE_CONFIG))
    patch_moe_model_for_ep(model, single_process_ep_config(TINY_GEMMA4_MOE_CONFIG["num_experts"]))
    # Anti-vacuity: a zero-patch is only a warning at ep_group_size 1, and nothing below would notice.
    assert [name for name, module in model.named_modules() if isinstance(module, EPMoELayerBase)]
    return model


def _trainer(tmp_path, peft_config: LoraConfig | None) -> DistributedSFTTrainer:
    """A real SFT trainer over the patched model, at the grouped-GEMM-only shape a CPU box can run."""
    args = SFTConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        max_length=len(_INPUT_IDS),
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
    )
    return DistributedSFTTrainer(
        model=_ep_patched_model(),
        args=args,
        train_dataset=Dataset.from_dict({"input_ids": [_INPUT_IDS] * 2, "labels": [_INPUT_IDS] * 2}),
        processing_class=_tokenizer(),
        parallelism_config=make_parallelism_config(world_size=1, gpus_per_node=1),
        peft_config=peft_config,
        moe_balancing="none",
    )


def _generation_delta(module: torch.nn.Module, forwards: int) -> int:
    """How far ``forwards`` calls of ``module`` moved the process-global generation."""
    before = dispatcher_mod._FORWARD_GENERATION
    for _ in range(forwards):
        module(input_ids=torch.tensor([_INPUT_IDS]))
    return dispatcher_mod._FORWARD_GENERATION - before


def test_an_attention_lora_run_still_advances_the_ep_capacity_generation(tmp_path):
    """EP + attention LoRA is a supported shape, and its forwards must each size their own capacity.

    Frozen, the run trains until a batch dispatches more tokens per rank than its first one did and
    then dies inside the dedup guard, blaming the model's later MoE layers.
    """
    trainer = _trainer(tmp_path, LoraConfig(r=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))

    # Premise: TRL wrapped the patched model, and the loop calls that wrapper, not what it wraps.
    assert isinstance(trainer.model, PeftModel), type(trainer.model).__name__
    assert "self._wrap_model(self.model_wrapped)" in inspect.getsource(Trainer._prepare_for_training)
    assert trainer.model_wrapped is trainer.model

    assert _generation_delta(trainer.model_wrapped, forwards=2) == 2, (
        "the wrapper reaches the patched model through .forward(), which runs no pre-hook, so the "
        "capacity generation stayed at the value the run's first forward cached"
    )


def test_the_same_run_without_an_adapter_advances_it_exactly_once_per_forward(tmp_path):
    """Anti-vacuity for the LoRA case, and the ceiling on the fix.

    An unwrapped model already carries the hook from EP patching, so this measures the harness
    itself: a forward that never reaches an EP-patched module, or a counter nothing owns, would
    fail here too. It also refuses a double bump — re-registering on a model that already has the
    hook would degrade the dedup back to one all-reduce per MoE layer.
    """
    trainer = _trainer(tmp_path, None)

    assert not isinstance(trainer.model, PeftModel)
    assert _generation_delta(trainer.model_wrapped, forwards=2) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
