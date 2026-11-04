#!/usr/bin/env python
"""An adapter-only PEFT resume on one process or under DDP keeps the adapters at the run's dtype.
PEFT's ``autocast_adapter_dtype`` recasts bf16 adapters to fp32, and a resume loads before the optimizer exists."""

import json
import os

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file
from trl import SFTConfig

from halo.checkpoint.format import ADAPTER_SAFETENSORS_FILE, SAFETENSORS_WEIGHTS_FILE
from halo.checkpoint.parallel.resume_weights import load_resume_weights
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.checkpoint_io import load_context
from tests.common.gloo import run_gloo_ranks
from tests.common.parallelism import make_parallelism_config
from tests.common.tiny_models import TINY_DENSE_FAMILY, tiny_family_model, word_level_tokenizer

WORDS = ("hello", "world", "again")
SAVE_STEP = 2


def _peft_model(seed: int, tokenizer) -> PeftModel:
    """A bf16 base with bf16 LoRA adapters, the dtype the trainer aligns a bf16 run's adapters to."""
    torch.manual_seed(seed)
    base = tiny_family_model(TINY_DENSE_FAMILY, tokenizer, overrides={"num_hidden_layers": 2}).to(torch.bfloat16)
    config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
    return get_peft_model(base, config, autocast_adapter_dtype=False)


def _trainer(model, tokenizer, output_dir: str) -> DistributedSFTTrainer:
    """A real single-process bf16 SFT trainer, saving its adapter-only checkpoint at step 2."""
    ids = tokenizer("hello world again hello")["input_ids"]
    args = SFTConfig(
        output_dir=output_dir,
        use_cpu=True,
        bf16=True,
        max_length=len(ids),
        max_steps=SAVE_STEP,
        per_device_train_batch_size=1,
        learning_rate=1e-2,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=SAVE_STEP,
        report_to="none",
    )
    return DistributedSFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_dict({"input_ids": [ids] * 4, "labels": [ids] * 4}),
        processing_class=tokenizer,
        parallelism_config=make_parallelism_config(world_size=1, gpus_per_node=1),
        moe_balancing="none",
    )


def _trainable(model) -> dict[str, torch.Tensor]:
    """The trained tensors under the keys an adapter file spells them with."""
    live = {name: param for name, param in model.named_parameters() if param.requires_grad}
    return get_peft_model_state_dict(model, state_dict=live)


def test_a_resumed_single_process_run_keeps_its_bf16_adapters_and_moments(tmp_path):
    tokenizer = word_level_tokenizer(WORDS)
    _trainer(_peft_model(0, tokenizer), tokenizer, str(tmp_path / "run")).train()
    checkpoint = tmp_path / "run" / f"checkpoint-{SAVE_STEP}"
    saved = load_file(str(checkpoint / ADAPTER_SAFETENSORS_FILE))
    assert {tensor.dtype for tensor in saved.values()} == {torch.bfloat16}, "premise: the save keeps the bf16 adapters"

    resumed = _trainer(_peft_model(1, tokenizer), tokenizer, str(tmp_path / "resumed"))
    resumed.train(resume_from_checkpoint=str(checkpoint))

    live = _trainable(resumed.model)
    assert set(live) == set(saved)
    recast = sorted(key for key, tensor in live.items() if tensor.dtype != torch.bfloat16)
    assert not recast, f"the resume recast {len(recast)} adapters off bf16: {recast[:3]}"
    assert all(torch.equal(live[key], saved[key]) for key in saved), "the resumed adapters are not the saved ones"
    moments = {param.dtype for group in resumed.optimizer.param_groups for param in group["params"]}
    assert moments == {torch.bfloat16}, f"the optimizer was built over {moments}"


@pytest.fixture
def adapter_dir(tmp_path) -> str:
    """A bf16 adapter-only checkpoint, as PEFT's own save writes it."""
    directory = str(tmp_path / "adapter")
    trained = _peft_model(0, word_level_tokenizer(WORDS))
    with torch.no_grad():
        for param in trained.parameters():
            if param.requires_grad:
                param.normal_()
    trained.save_pretrained(directory)
    return directory


def test_the_weight_resume_restores_an_adapter_only_checkpoint_in_place(adapter_dir):
    model = _peft_model(1, word_level_tokenizer(WORDS))
    saved = load_file(os.path.join(adapter_dir, ADAPTER_SAFETENSORS_FILE))

    assert load_resume_weights(load_context(model), adapter_dir, model) is True, "left to the base loader's recast"
    live = _trainable(model)
    assert all(live[key].dtype == torch.bfloat16 and torch.equal(live[key], saved[key]) for key in saved)


def test_a_reload_the_base_loader_owns_or_a_whole_weight_peft_checkpoint_stays_with_it(adapter_dir):
    """Accelerate-managed FSDP reads its own format, and a PEFT run never writes whole weights beside its adapter."""
    model = _peft_model(1, word_level_tokenizer(WORDS))
    before = {key: tensor.clone() for key, tensor in _trainable(model).items()}
    assert load_resume_weights(load_context(model, base_owns_whole_weight_load=True), adapter_dir, model) is False
    open(os.path.join(adapter_dir, SAFETENSORS_WEIGHTS_FILE), "wb").close()
    assert load_resume_weights(load_context(model), adapter_dir, model) is False
    assert all(torch.equal(tensor, before[key]) for key, tensor in _trainable(model).items())


def _resume_on_rank(rank: int, adapter_dir: str, outcome_dir: str) -> None:
    """One rank of a two-rank data-parallel run resuming its replica from the shared adapter-only checkpoint."""
    model = _peft_model(1, word_level_tokenizer(WORDS))
    context = load_context(model, make_parallelism_config(world_size=2, gpus_per_node=2, rank=rank))
    restored = load_resume_weights(context, adapter_dir, model)
    saved = load_file(os.path.join(adapter_dir, ADAPTER_SAFETENSORS_FILE))
    live = _trainable(model)
    result = {
        "restored": restored,
        "recast": sorted(key for key, tensor in live.items() if tensor.dtype != torch.bfloat16),
        "differs": sorted(key for key in saved if not torch.equal(live[key], saved[key])),
    }
    with open(os.path.join(outcome_dir, f"rank{rank}.json"), "w") as fh:
        json.dump(result, fh)


def test_every_ddp_rank_restores_the_bf16_adapters(adapter_dir, tmp_path):
    outcome_dir = tmp_path / "outcomes"
    outcome_dir.mkdir()
    run_gloo_ranks(_resume_on_rank, 2, adapter_dir, str(outcome_dir))
    outcomes = [json.loads((outcome_dir / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert outcomes == [{"restored": True, "recast": [], "differs": []}] * 2, outcomes


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
