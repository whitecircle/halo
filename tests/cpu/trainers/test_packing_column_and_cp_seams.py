#!/usr/bin/env python
"""Five seams that keep packed-batch inputs intact on their way into a trainer.

1. Column pruning: ``seq_lengths`` is a collator input, not a model-forward parameter, so HF's
   signature-based pruning drops it unless the trainer's signature set names it. TRL's SFT trainer
   happens to hard-code it; every other base does not. ``DataParallelLoaderMixin`` unions the
   collator's declared ``required_dataset_columns`` so the pin cannot drift from the collator.
2. The CP × packing rejection must see a collator passed POSITIONALLY (at TRL's own data_collator
   slot), not only via keyword — a packed collator slipping past the gate under CP attends across
   documents silently.
3. The compressed-KV rejection must see a packing / padding-free collator handed straight to the
   trainer, which skips the factory: a multi-document row on DeepSeek-V4's compressors reads another
   document's compressed KV.
4. The M-RoPE rejection must see a padding-free collator handed straight to the trainer: a flattened M-RoPE
   row (every Qwen3.5/3.6 decoder) faults the flash varlen kernels, while ``packing`` serves the same rows.
5. The unmarked-linear-attention rejection must see a packing / padding-free collator handed straight to the
   trainer too: a KDA or Lightning Attention-2 layer carries state across a multi-document row's documents. A
   collator the factory built under the run's ``allow_cross_document_linear_attention`` carries the opt-in, so the
   trainer admits what the factory did; a tree collator's plan segments those layers.

Run: python tests/cpu/trainers/test_packing_column_and_cp_seams.py
"""

import contextlib
import types
from unittest import mock

import pytest
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.deepseek_v4 import DeepseekV4Config
from trl import SFTTrainer

from halo.data.collators.factory import select_data_collator
from halo.data.collators.packing import (
    DataCollatorWithFlattening,
    DataCollatorWithFlatteningAndCompletionMask,
    DataCollatorWithPacking,
)
from halo.data.collators.tree_packing import DataCollatorWithTreePacking
from halo.trainers.dataloaders import DataParallelLoaderMixin
from halo.trainers.sft import _CTOR_POSITIONS, DistributedSFTTrainer
from tests.common.models import TINY_DSV4_CONFIG
from tests.common.tiny_models import unknown_hybrid_config


class _Base:
    """Stands in for the HF Trainer base: sets the model-signature columns when unset."""

    def __init__(self):
        self._signature_columns = None
        self.data_collator = None

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["input_ids", "labels"]


class _Trainer(DataParallelLoaderMixin, _Base):
    pass


def test_collator_columns_survive_signature_pruning():
    trainer = _Trainer()
    trainer.data_collator = types.SimpleNamespace(required_dataset_columns=("seq_lengths",))
    trainer._set_signature_columns_if_needed()
    assert "seq_lengths" in trainer._signature_columns

    # Idempotent: a second call must not duplicate the column.
    trainer._set_signature_columns_if_needed()
    assert trainer._signature_columns.count("seq_lengths") == 1


def test_collator_without_declared_columns_changes_nothing():
    trainer = _Trainer()
    trainer._set_signature_columns_if_needed()
    assert trainer._signature_columns == ["input_ids", "labels"]


def test_packing_collator_declares_seq_lengths():
    assert "seq_lengths" in DataCollatorWithPacking.required_dataset_columns


def test_union_override_wins_over_trl_base():
    """TRL's SFT base defines the method too; the mixin's union must be first in the MRO, or the
    collator's declaration silently stops reaching the pruning set."""
    assert (
        DistributedSFTTrainer._set_signature_columns_if_needed
        is DataParallelLoaderMixin._set_signature_columns_if_needed
    )


def _construct_sft(*, cp: bool, model=None, collator=None) -> None:
    """``DistributedSFTTrainer`` up to its gates, ``model`` and ``collator`` at TRL's own positional
    slots, as a signature-following caller would pass them."""

    def _init_cfg(self, kwargs, **_):
        self.parallelism_config = types.SimpleNamespace(is_cp_mode=cp)
        return kwargs

    ctor_args: list = [None] * (max(_CTOR_POSITIONS["data_collator"], _CTOR_POSITIONS["model"]) + 1)
    ctor_args[_CTOR_POSITIONS["data_collator"]] = collator
    ctor_args[_CTOR_POSITIONS["model"]] = model
    with (
        mock.patch.object(DistributedSFTTrainer, "_init_distributed_config", _init_cfg),
        mock.patch.object(SFTTrainer, "__init__", return_value=None),
        mock.patch.object(DistributedSFTTrainer, "_configure_distributed_modes", return_value=None),
    ):
        DistributedSFTTrainer(*ctor_args)


def test_positional_collator_reaches_cp_gate():
    # ``object.__new__``: isinstance is all the gate reads, and the real ctor wants a tokenizer.
    with pytest.raises(ValueError, match="context parallelism"):
        _construct_sft(cp=True, collator=object.__new__(DataCollatorWithPacking))


@pytest.mark.parametrize(
    ("root", "mode"), [(DataCollatorWithPacking, "packing"), (DataCollatorWithFlattening, "padding_free")]
)
@pytest.mark.parametrize("compressed", [True, False], ids=["compressed-kv", "masked-attention-only"])
def test_a_hand_built_multi_document_collator_reaches_the_compressed_kv_gate(root, mode, compressed):
    """Refused for a model with compressed-KV layers; a model without them constructs."""
    layer_types = TINY_DSV4_CONFIG["layer_types"] if compressed else ["sliding_attention"] * 3
    model = types.SimpleNamespace(config=DeepseekV4Config(**{**TINY_DSV4_CONFIG, "layer_types": layer_types}))
    refusal = f"{mode} is refused .* layers pool KV over windows"
    with pytest.raises(ValueError, match=refusal) if compressed else contextlib.nullcontext():
        _construct_sft(cp=False, model=model, collator=object.__new__(root))


@pytest.mark.parametrize(
    ("collator_class", "flattens"),
    [
        (DataCollatorWithFlattening, True),
        (DataCollatorWithFlatteningAndCompletionMask, True),
        (DataCollatorWithPacking, False),
    ],
)
@pytest.mark.parametrize(
    ("model_type", "mrope"), [("qwen3_5", True), ("qwen3_5_moe_text", True), ("qwen3_next", False)]
)
def test_a_hand_built_padding_free_collator_reaches_the_mrope_gate(collator_class, flattens, model_type, mrope):
    """Refused for an M-RoPE decoder, read through a composite wrapper too; packing there, and a padding-free
    collator on a plain-RoPE decoder, construct."""
    model = types.SimpleNamespace(config=CONFIG_MAPPING[model_type]())
    with pytest.raises(ValueError, match="M-RoPE") if flattens and mrope else contextlib.nullcontext():
        _construct_sft(cp=False, model=model, collator=object.__new__(collator_class))


_ROOTS = [(DataCollatorWithPacking, "packing"), (DataCollatorWithFlattening, "padding_free")]
# model_type -> whether its rows cross in linear-attention layers no segment marker reaches.
_LINEAR_ATTENTION = {"glm5_next_text": True, "qwen3_next": False, "qwen3": False}


@pytest.mark.parametrize(("root", "mode"), _ROOTS)
@pytest.mark.parametrize("model_type", sorted(_LINEAR_ATTENTION))
def test_a_hand_built_multi_document_collator_reaches_the_linear_attention_gate(root, mode, model_type):
    """Refused for GLM-5 Next's KDA layers; GatedDeltaNet reads its markers and softmax-only models have no such
    layer, so both construct."""
    crosses = _LINEAR_ATTENTION[model_type]
    model = types.SimpleNamespace(config=CONFIG_MAPPING[model_type]())
    refusal = f"{mode}=True is refused for model_type='{model_type}': its linear-attention layers"
    with pytest.raises(ValueError, match=refusal) if crosses else contextlib.nullcontext():
        _construct_sft(cp=False, model=model, collator=object.__new__(root))


@pytest.mark.parametrize(("root", "mode"), _ROOTS)
def test_a_factory_built_collator_carries_the_run_opt_in(root, mode):
    """The factory admits the crossing row under the opt-in and records it, so the trainer admits it too; the
    same collator without the record is refused, so the trainer reads the record and nothing else."""
    config = unknown_hybrid_config()
    config._attn_implementation = "flash_attention_4"
    tokenizer = types.SimpleNamespace(pad_token_id=0, eos_token_id=1, padding_side="right")
    collator = select_data_collator(
        tokenizer,
        train_on_completions_only=False,
        model_config=config,
        allow_cross_document_linear_attention=True,
        **{mode: True},
    )
    assert isinstance(collator, root) and collator.allow_cross_document_linear_attention is True
    model = types.SimpleNamespace(config=config)
    _construct_sft(cp=False, model=model, collator=collator)
    collator.allow_cross_document_linear_attention = False
    with pytest.raises(ValueError, match=f"{mode}=True is refused"):
        _construct_sft(cp=False, model=model, collator=collator)


def test_a_tree_collator_is_left_to_its_plan():
    """A tree row's plan segments the linear-attention layers of every family tree packing admits, and tree
    packing refuses the opt-in, so its collator never carries one."""
    model = types.SimpleNamespace(config=CONFIG_MAPPING["glm5_next_text"]())
    _construct_sft(cp=False, model=model, collator=object.__new__(DataCollatorWithTreePacking))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
