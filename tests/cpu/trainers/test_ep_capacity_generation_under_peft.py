#!/usr/bin/env python
"""The EP capacity generation must keep advancing after PEFT wraps the EP-patched model.
``PeftModel`` calls ``.forward()``, which runs no pre-hook; frozen, a later longer batch fails in the dedup guard."""

from __future__ import annotations

import inspect

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Trainer
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
from trl import SFTConfig

from halo.distributed.expert_parallel.deepep.capacity import capacity_scope
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.models import TINY_GEMMA4_MOE_CONFIG
from tests.common.parallelism import make_parallelism_config, single_process_ep_config

_INPUT_IDS = [3, 4, 5, 6]


def _tokenizer() -> PreTrainedTokenizerFast:
    """TRL demands a processing class; this run never tokenizes."""
    vocab = {f"t{index}": index for index in range(TINY_GEMMA4_MOE_CONFIG["vocab_size"])}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="t0"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="t0", pad_token="t1", eos_token="t2")


def _ep_patched_model() -> Gemma4ForCausalLM:
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(Gemma4TextConfig(**TINY_GEMMA4_MOE_CONFIG))
    apply_ep_layers(model, single_process_ep_config(TINY_GEMMA4_MOE_CONFIG["num_experts"]))
    # A zero-patch is only a warning at ep_group_size 1, and nothing below would notice.
    assert [name for name, module in model.named_modules() if isinstance(module, EPMoELayerBase)]
    return model


def _trainer(tmp_path, peft_config: LoraConfig | None) -> DistributedSFTTrainer:
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
    before = capacity_scope.generation
    for _ in range(forwards):
        module(input_ids=torch.tensor([_INPUT_IDS]))
    return capacity_scope.generation - before


def test_an_attention_lora_run_still_advances_the_ep_capacity_generation(tmp_path):
    """Frozen, the run trains until a batch dispatches more tokens per rank than its first, then dies in the guard."""
    trainer = _trainer(tmp_path, LoraConfig(r=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))

    # TRL wrapped the patched model, and the loop calls that wrapper, not what it wraps.
    assert isinstance(trainer.model, PeftModel), type(trainer.model).__name__
    assert "self._wrap_model(self.model_wrapped)" in inspect.getsource(Trainer._prepare_for_training)
    assert trainer.model_wrapped is trainer.model

    assert _generation_delta(trainer.model_wrapped, forwards=2) == 2, (
        "the wrapper reaches the patched model through .forward(), which runs no pre-hook, so the "
        "capacity generation stayed at the value the run's first forward cached"
    )


def test_the_same_run_without_an_adapter_advances_it_exactly_once_per_forward(tmp_path):
    """Measures the harness: a forward that misses the EP layers would fail here too.
    A double bump would degrade the dedup back to one all-reduce per MoE layer."""
    trainer = _trainer(tmp_path, None)

    assert not isinstance(trainer.model, PeftModel)
    assert _generation_delta(trainer.model_wrapped, forwards=2) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
