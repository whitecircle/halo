#!/usr/bin/env python
"""A ``text_only_model`` full fine-tune saved through the base Trainer writes its text-only layout.

``text_only_model: true`` loads a multimodal Qwen3.5/3.6 checkpoint as its ``*ForCausalLM`` sibling,
and transformers records the ``PrefixChange(prefix_to_remove="language_model")`` that load consumed.
Where no parallel saver claims the write (single GPU, DDP, accelerate FSDP), the trainer falls back to
HF's ``Trainer.save_model``, whose stock ``save_pretrained`` replays that record: a ``*_text`` config
over ``model.language_model.*`` keys. transformers re-strips the prefix on reload, so a reload alone
proves nothing; engine loaders keyed on the architectures and ``reattach_vision_tower.py`` do not.
These tests pin the on-disk keys to what a fresh model of the saved config writes.

    python tests/cpu/trainers/test_text_only_base_save_layout.py
"""

from __future__ import annotations

# Ahead of the Qwen3.5 modeling modules, which bind transformers' hub-kernel fallback at import.
import src.models.patches.kernel_dispatch  # noqa: F401  # isort: skip

from pathlib import Path

import pytest
import torch
from accelerate import PartialState
from datasets import Dataset
from transformers import AutoConfig
from trl import SFTConfig

PartialState()  # the trainer logs through accelerate's logger, which refuses an uninitialized state

from src.distributed.checkpoint.save import select_checkpoint_saver
from src.models.loading.model_preparation import auto_load_model
from src.trainers.sft import DistributedSFTTrainer
from tests.common.parallelism import make_parallelism_config
from tests.cpu.checkpoint.test_merge_text_only_adapter_e2e import (
    _FAMILIES,
    _WRAPPER_TEXT_PREFIX,
    _build_multimodal_base,
    _declared_layout,
    _on_disk_keys,
    _tiny_tokenizer,
)

_INPUT_IDS = [2, 3, 2, 3]


def _trainer(model, output_dir: Path) -> DistributedSFTTrainer:
    """A real single-process SFT trainer: no parallel saver claims its save."""
    args = SFTConfig(
        output_dir=str(output_dir),
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
        model=model,
        args=args,
        train_dataset=Dataset.from_dict({"input_ids": [_INPUT_IDS] * 2, "labels": [_INPUT_IDS] * 2}),
        processing_class=_tiny_tokenizer(),
        parallelism_config=make_parallelism_config(world_size=1, gpus_per_node=1),
        moe_balancing="none",
    )


@pytest.mark.parametrize("family_name", sorted(_FAMILIES))
def test_a_text_only_full_fine_tune_saved_by_the_base_trainer_writes_its_text_only_layout(family_name, tmp_path):
    family = _FAMILIES[family_name]
    base, out = tmp_path / "base", tmp_path / "out"
    _build_multimodal_base(base, family)
    model = auto_load_model(str(base), text_only=True, dtype=torch.float32)
    recorded = model._weight_conversions
    assert type(model) is family.text_cls
    assert any(getattr(c, "prefix_to_remove", None) == "language_model" for c in recorded), (
        "premise: the text-only load no longer records the prefix it stripped, so this test pins nothing"
    )
    trainer = _trainer(model, tmp_path / "run")
    assert select_checkpoint_saver(trainer._checkpoint_context()) is None, "premise: the base Trainer must save"

    trainer.save_model(str(out))

    keys = _on_disk_keys(out)
    assert not any(key.startswith(_WRAPPER_TEXT_PREFIX) for key in keys), (
        f"the base save replayed the load's PrefixChange: {sorted(keys)[:3]} under a "
        f"{AutoConfig.from_pretrained(out).model_type} config"
    )
    assert keys == _declared_layout(out, tmp_path / "fresh")
    assert model._weight_conversions is recorded, "the save must leave the load's record as it found it"

    reloaded = auto_load_model(str(out), dtype=torch.float32)
    assert type(reloaded) is family.text_cls
    _, loading_info = family.text_cls.from_pretrained(out, dtype=torch.float32, output_loading_info=True)
    assert not loading_info["missing_keys"] and not loading_info["unexpected_keys"], loading_info
    saved = reloaded.state_dict()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(saved[name], tensor.to(saved[name].dtype), msg=f"{name} did not round-trip")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
