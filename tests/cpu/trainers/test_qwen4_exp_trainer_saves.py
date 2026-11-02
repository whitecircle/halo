#!/usr/bin/env python
"""Every trainer save of a model holding an external table leaves a checkpoint that loads, or the run is refused.

Qwen3.8-Flash-Next holds its n-gram table outside the module state, so transformers' ``save_pretrained``, the base
Trainer's save, writes none of it. A run no parallel saver claims (one process or DDP at ``use_grouped_gemm: false``)
streams through the gathered writer instead, which writes the table beside the weights: pinned through a real
single-process SFT trainer whose ``save_model`` export reloads in stock transformers into the trained model's
logits. Under accelerate-managed FSDP the save keeps the base Trainer's path, so the trainer refuses the model at
construction, before any step, unless it trains adapters, whose save needs no table.

    python tests/cpu/trainers/test_qwen4_exp_trainer_saves.py
"""

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from trl import SFTConfig

import halo.trainers.distributed_trainer as distributed_trainer
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.checkpoint_io import loading_problems, safetensors_state_dict
from tests.common.models import TINY_QWEN4_EXP_CONFIG
from tests.common.parallelism import make_parallelism_config
from tests.common.qwen4_exp import (
    NGRAM_TABLE_PATH,
    build_tiny_qwen4_exp_checkpoint,
    load_host_table_model,
    transformers_forwards,
)
from tests.common.tiny_models import word_level_tokenizer

TABLE_KEYS = {
    f"{NGRAM_TABLE_PATH}.shard_{index}.weight" for index in range(TINY_QWEN4_EXP_CONFIG["split_ngram_parts"])
}
WORDS = ("solve", "the", "problem")


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> str:
    directory = str(tmp_path_factory.mktemp("flash_next"))
    build_tiny_qwen4_exp_checkpoint(directory)
    return directory


def _trainer(model, output_dir: str) -> DistributedSFTTrainer:
    tokenizer = word_level_tokenizer(WORDS)
    ids = tokenizer("solve the problem")["input_ids"]
    args = SFTConfig(
        output_dir=output_dir,
        use_cpu=True,
        bf16=False,
        max_length=len(ids),
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
        model=model,
        args=args,
        train_dataset=Dataset.from_dict({"input_ids": [ids] * 2, "labels": [ids] * 2}),
        processing_class=tokenizer,
        parallelism_config=make_parallelism_config(world_size=1, gpus_per_node=1, use_grouped_gemm=False),
        moe_balancing="none",
    )


def _logits(model, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad(), transformers_forwards():
        return model.eval()(input_ids=ids, use_cache=False).logits


def test_a_single_process_save_writes_the_table_and_reloads_into_the_trained_model(source, tmp_path):
    """In bf16, the checkpoint's dtype, so the export's cast to it is exact and the reload compares bit for bit."""
    model = load_host_table_model(source, dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(
                torch.randn(parameter.shape, generator=torch.Generator().manual_seed(1)).to(parameter) * 0.01
            )
    ids = torch.randint(0, TINY_QWEN4_EXP_CONFIG["vocab_size"], (2, 23), generator=torch.Generator().manual_seed(0))
    expected = _logits(model, ids)
    out = str(tmp_path / "export")

    _trainer(model, str(tmp_path / "run")).save_model(out)

    assert set(safetensors_state_dict(out)) >= TABLE_KEYS, "the export lacks the n-gram table"
    reloaded, info = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16, output_loading_info=True)
    assert loading_problems(info) == {}
    assert torch.equal(_logits(reloaded, ids), expected)


def test_accelerate_managed_fsdp_is_refused_at_construction_unless_it_trains_adapters(source, tmp_path, monkeypatch):
    monkeypatch.setattr(distributed_trainer, "accelerate_manages_fsdp", lambda config: True)
    with pytest.raises(ValueError, match="external tables"):
        _trainer(load_host_table_model(source), str(tmp_path / "full"))

    adapters = get_peft_model(load_host_table_model(source), LoraConfig(r=4, target_modules=["q_proj", "v_proj"]))
    _trainer(adapters, str(tmp_path / "lora"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
