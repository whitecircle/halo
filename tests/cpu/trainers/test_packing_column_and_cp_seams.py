#!/usr/bin/env python
"""Seams that keep packed-batch inputs intact on their way into a trainer.
Hand-built collators skip the factory, so the trainer runs the CP, compressed-KV, dense-backend and linear-attention gates."""

import contextlib
import types
from unittest import mock

import pytest
from datasets import Dataset
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.deepseek_v4 import DeepseekV4Config
from trl import SFTConfig, SFTTrainer

from halo.data.collators.factory import select_data_collator
from halo.data.collators.packing import (
    DataCollatorWithFlattening,
    DataCollatorWithFlatteningAndCompletionMask,
    DataCollatorWithPacking,
)
from halo.data.collators.tree_packing import DataCollatorWithTreePacking
from halo.entrypoints.data_setup import disable_trl_dataset_prep
from halo.models import segment_markers
from halo.trainers.dataloaders import DataParallelLoaderMixin
from halo.trainers.sft import _CTOR_POSITIONS, DistributedSFTTrainer
from tests.common.models import TINY_DSV4_CONFIG, TINY_QWEN3_CONFIG
from tests.common.offline_grpo import make_offline_tokenizer
from tests.common.parallelism import make_parallelism_config
from tests.common.tiny_models import unknown_hybrid_config


@pytest.fixture(autouse=True)
def _segment_aware_kernels(monkeypatch):
    """The segment-aware kernel checks as the images answer them on a GPU host, which a CPU host's torch fails."""
    monkeypatch.setattr(
        segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", (("causal_conv1d", lambda: True), ("fla>=0.2.2", lambda: True))
    )


class _Base:
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

    trainer._set_signature_columns_if_needed()
    assert trainer._signature_columns.count("seq_lengths") == 1


def test_collator_without_declared_columns_changes_nothing():
    trainer = _Trainer()
    trainer._set_signature_columns_if_needed()
    assert trainer._signature_columns == ["input_ids", "labels"]


def test_packing_collator_declares_seq_lengths():
    assert "seq_lengths" in DataCollatorWithPacking.required_dataset_columns


def test_union_override_wins_over_trl_base():
    """TRL's SFT base defines the method too, so the mixin's union must come first in the MRO."""
    assert (
        DistributedSFTTrainer._set_signature_columns_if_needed
        is DataParallelLoaderMixin._set_signature_columns_if_needed
    )


def _construct_sft(*, cp: bool, model=None, collator=None, built=None) -> None:
    """Up to the gates, ``model`` and ``collator`` at TRL's slots; the base builds ``built``, else ``model``."""

    def _init_cfg(self, kwargs, **_):
        self.parallelism_config = types.SimpleNamespace(is_cp_mode=cp)
        return kwargs

    def _base_init(self, *_args, **_kwargs):
        self.model = model if built is None else built

    ctor_args: list = [None] * (max(_CTOR_POSITIONS["data_collator"], _CTOR_POSITIONS["model"]) + 1)
    ctor_args[_CTOR_POSITIONS["data_collator"]] = collator
    ctor_args[_CTOR_POSITIONS["model"]] = model
    with (
        mock.patch.object(DistributedSFTTrainer, "_init_distributed_config", _init_cfg),
        mock.patch.object(SFTTrainer, "__init__", _base_init),
        mock.patch.object(DistributedSFTTrainer, "_configure_distributed_modes", return_value=None),
    ):
        DistributedSFTTrainer(*ctor_args)


def _on_flash(config):
    """``config`` as a load on FA2 leaves it, so the rows' backend admits multi-document rows."""
    config._attn_implementation = "flash_attention_2"
    return config


def test_positional_collator_reaches_cp_gate():
    # ``object.__new__``: isinstance is all the gate reads, and the real ctor wants a tokenizer.
    with pytest.raises(ValueError, match="context parallelism"):
        _construct_sft(cp=True, collator=object.__new__(DataCollatorWithPacking))


@pytest.mark.parametrize(
    ("root", "mode"), [(DataCollatorWithPacking, "packing"), (DataCollatorWithFlattening, "padding_free")]
)
@pytest.mark.parametrize("compressed", [True, False], ids=["compressed-kv", "masked-attention-only"])
def test_a_hand_built_multi_document_collator_reaches_the_compressed_kv_gate(root, mode, compressed):
    """A multi-document row on DeepSeek-V4's compressors reads another document's compressed KV."""
    layer_types = TINY_DSV4_CONFIG["layer_types"] if compressed else ["sliding_attention"] * 3
    model = types.SimpleNamespace(
        config=_on_flash(DeepseekV4Config(**{**TINY_DSV4_CONFIG, "layer_types": layer_types}))
    )
    refusal = f"{mode} is refused .* layers pool KV over windows"
    with pytest.raises(ValueError, match=refusal) if compressed else contextlib.nullcontext():
        _construct_sft(cp=False, model=model, collator=object.__new__(root))


@pytest.mark.parametrize(
    "collator_class",
    [DataCollatorWithFlattening, DataCollatorWithFlatteningAndCompletionMask, DataCollatorWithPacking],
)
@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe_text", "qwen3_next"])
def test_a_hand_built_flattening_collator_on_a_gated_delta_net_decoder_passes_on_flash(collator_class, model_type):
    """An M-RoPE decoder takes a flattened row on a flash backend like any other GatedDeltaNet decoder."""
    model = types.SimpleNamespace(config=_on_flash(CONFIG_MAPPING[model_type]()))
    _construct_sft(cp=False, model=model, collator=object.__new__(collator_class))


@pytest.mark.parametrize("collator_class", [DataCollatorWithFlattening, DataCollatorWithFlatteningAndCompletionMask])
def test_a_hand_built_flattening_collator_on_a_dense_backend_is_refused(collator_class):
    config = CONFIG_MAPPING["qwen3_5"]()
    config._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="padding_free requires a varlen"):
        _construct_sft(cp=False, model=types.SimpleNamespace(config=config), collator=object.__new__(collator_class))


_ROOTS = [(DataCollatorWithPacking, "packing"), (DataCollatorWithFlattening, "padding_free")]
# model_type -> whether its rows cross in linear-attention layers no segment marker reaches.
_LINEAR_ATTENTION = {"glm5_next_text": True, "qwen3_next": False, "qwen3": False}


@pytest.mark.parametrize(("root", "mode"), _ROOTS)
@pytest.mark.parametrize("model_type", sorted(_LINEAR_ATTENTION))
def test_a_hand_built_multi_document_collator_reaches_the_linear_attention_gate(root, mode, model_type):
    """KDA layers carry state across documents; GatedDeltaNet reads its markers and softmax-only models have none."""
    crosses = _LINEAR_ATTENTION[model_type]
    model = types.SimpleNamespace(config=_on_flash(CONFIG_MAPPING[model_type]()))
    refusal = f"{mode}=True is refused for model_type='{model_type}': its linear-attention layers"
    with pytest.raises(ValueError, match=refusal) if crosses else contextlib.nullcontext():
        _construct_sft(cp=False, model=model, collator=object.__new__(root))


@pytest.mark.parametrize(("root", "mode"), _ROOTS)
def test_a_factory_built_collator_carries_the_run_opt_in(root, mode):
    """The trainer reads the factory's record and nothing else: the same collator without it is refused."""
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
    """A tree row's plan segments those layers, and tree packing refuses the opt-in, so its collator carries none."""
    model = types.SimpleNamespace(config=CONFIG_MAPPING["glm5_next_text"]())
    _construct_sft(cp=False, model=model, collator=object.__new__(DataCollatorWithTreePacking))


def _script_built_sft(tmp_path, mode: str, *, seam: bool):
    """The factory's collator, and ``disable_trl_dataset_prep`` on the config only where ``seam``."""
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    # After the build, which refuses a flash backend on a CPU host.
    _on_flash(model.config)
    tokenizer = make_offline_tokenizer()
    args = SFTConfig(
        output_dir=str(tmp_path), use_cpu=True, bf16=False, use_liger_kernel=False, report_to="none", **{mode: True}
    )
    collator = select_data_collator(
        tokenizer,
        train_on_completions_only=False,
        model_config=model.config,
        padding_free=args.padding_free,
        packing=args.packing,
    )
    if seam:
        disable_trl_dataset_prep(args)
    trainer = DistributedSFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_dict({"input_ids": [[5, 6, 7]]}),
        processing_class=tokenizer,
        data_collator=collator,
        parallelism_config=make_parallelism_config(world_size=1, gpus_per_node=1),
    )
    return trainer, collator


@pytest.mark.parametrize(
    ("mode", "root"), [("padding_free", DataCollatorWithFlattening), ("packing", DataCollatorWithPacking)]
)
def test_the_scripts_trainer_trains_on_the_factory_collator(tmp_path, mode, root):
    """Without the seam TRL refuses a custom collator under its ``padding_free``, which its default packing forces."""
    trainer, collator = _script_built_sft(tmp_path, mode, seam=True)
    assert isinstance(collator, root) and trainer.data_collator is collator
    with pytest.raises(ValueError, match="custom data collator is not supported"):
        _script_built_sft(tmp_path, mode, seam=False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
