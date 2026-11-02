#!/usr/bin/env python
"""Packing under pipeline parallelism on models whose conv / linear-attention mixers need segment markers.

A pipeline stage's forward passes none of the ``seq_idx`` / ``cu_seq_lens`` markers GatedDeltaNet and ShortConv
layers read, and the linear-attention layers no marker reaches (KDA, Lightning Attention-2) have no boundary on
any path. Their state carries from
each document of a packed row into the next, and the loss shows nothing. The PP trainer's check of its final
collator and the collator factory (PP keeps the packed rows) refuse such a row; padded rows hold one document
each and stay allowed.

Run: python tests/cpu/parallelism/test_pp_packing_segment_markers.py
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from transformers import Glm5NextConfig, Qwen3_5MoeConfig, Step3p7Config
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from halo.data.collators.factory import select_data_collator
from halo.data.collators.fixed_shape import FixedShapeCollator
from halo.data.collators.packing import DataCollatorWithPacking
from halo.data.collators.padded import DataCollatorForCausalLMWithPadding
from halo.distributed.pipeline_parallel.families import PP_SPEC_MAP
from halo.models import segment_markers
from halo.trainers.parallelism.pipeline_rules import reject_pp_collator
from halo.trainers.parallelism.pipeline_trainer import PipelineTrainerMixin
from tests.common.models import (
    BAILING_MOE_RING_MINI,
    TINY_DSV4_CONFIG,
    TINY_GLM4_MOE_LITE_CONFIG,
    TINY_GLM5_CONFIG,
    TINY_GLM5_VISION_CONFIG,
    TINY_GPTOSS_CONFIG,
    TINY_INKLING_CONFIG,
    TINY_LAGUNA_CONFIG,
    TINY_LFM2_MOE_CONFIG,
    TINY_MISTRAL4_CONFIG,
    TINY_QWEN3_CONFIG,
    TINY_QWEN3_MOE_CONFIG,
    TINY_QWEN35_CONFIG,
    TINY_QWEN35_MOE_CONFIG,
    TINY_QWEN35_VISION_CONFIG,
    TINY_STEP3P7_CONFIG,
    TINY_STEP3P7_VISION_CONFIG,
)
from tests.common.tiny_models import remote_config, unknown_hybrid_config, word_level_tokenizer

_REFUSAL = "packing under pipeline parallelism is refused"


# Every registered PP backbone -> (the config its stage carries, whether a packed row is refused). The shipped
# GLM-5 Next and Step-3.7 checkpoints are composite, so their stage carries the wrapper's config.
_PP_ROSTER = {
    "Qwen3Model": (lambda: CONFIG_MAPPING["qwen3"](**TINY_QWEN3_CONFIG), False),
    "GptOssModel": (lambda: CONFIG_MAPPING["gpt_oss"](**TINY_GPTOSS_CONFIG), False),
    "Qwen3MoeModel": (lambda: CONFIG_MAPPING["qwen3_moe"](**TINY_QWEN3_MOE_CONFIG), False),
    "Glm4MoeLiteModel": (lambda: CONFIG_MAPPING["glm4_moe_lite"](**TINY_GLM4_MOE_LITE_CONFIG), False),
    "LagunaModel": (lambda: CONFIG_MAPPING["laguna"](**TINY_LAGUNA_CONFIG), False),
    # Its short convolutions read no document boundary on any path.
    "InklingTextModel": (lambda: CONFIG_MAPPING["inkling_text"](**TINY_INKLING_CONFIG), True),
    "Step3p7TextModel": (
        lambda: Step3p7Config(text_config=dict(TINY_STEP3P7_CONFIG), vision_config=dict(TINY_STEP3P7_VISION_CONFIG)),
        False,
    ),
    # Its compressed-KV layers are refused by reject_compressed_kv_rows, not by this rule.
    "DeepseekV4Model": (lambda: CONFIG_MAPPING["deepseek_v4"](**TINY_DSV4_CONFIG), False),
    "Qwen3_5TextModel": (lambda: CONFIG_MAPPING["qwen3_5_text"](**TINY_QWEN35_CONFIG), True),
    "Qwen3_5MoeTextModel": (lambda: CONFIG_MAPPING["qwen3_5_moe_text"](**TINY_QWEN35_MOE_CONFIG), True),
    "Glm5NextTextModel": (
        lambda: Glm5NextConfig(text_config=dict(TINY_GLM5_CONFIG), vision_config=dict(TINY_GLM5_VISION_CONFIG)),
        True,
    ),
}
# Read by the same detection: the composite config a multimodal Qwen3.5/3.6 MoE checkpoint's stage carries, and
# off the PP roster a seq_idx-only family, an unmarked layer-group family, a family no list names, and a
# softmax-only MoE.
_DETECTED = {
    "qwen3_5_moe_composite": (
        lambda: Qwen3_5MoeConfig(
            text_config=dict(TINY_QWEN35_MOE_CONFIG), vision_config=dict(TINY_QWEN35_VISION_CONFIG)
        ),
        True,
    ),
    "lfm2_moe_short_conv": (lambda: CONFIG_MAPPING["lfm2_moe"](**TINY_LFM2_MOE_CONFIG), True),
    "ring_lightning_attention": (lambda: remote_config(BAILING_MOE_RING_MINI), True),
    "unknown_family_linear_attention": (unknown_hybrid_config, True),
    "mistral4_softmax_only": (lambda: CONFIG_MAPPING["mistral4"](**TINY_MISTRAL4_CONFIG), False),
}
_CASES = {**_PP_ROSTER, **_DETECTED}


def _packing_collator() -> DataCollatorWithPacking:
    return DataCollatorWithPacking(word_level_tokenizer())


def test_the_roster_is_every_registered_pp_backbone():
    """A family registered for PP without an entry here would go unchecked."""
    assert set(_PP_ROSTER) == {name for name, spec in PP_SPEC_MAP.items() if spec.validated}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_a_packed_row_is_refused_where_a_mixer_needs_markers(name):
    build, refused = _CASES[name]
    config = build()
    with pytest.raises(ValueError, match=_REFUSAL) if refused else nullcontext():
        reject_pp_collator(_packing_collator(), per_device_train_batch_size=1, n_microbatches=1, model_config=config)
    padded = DataCollatorForCausalLMWithPadding(tokenizer=word_level_tokenizer(), mlm=False)
    reject_pp_collator(padded, per_device_train_batch_size=1, n_microbatches=1, model_config=config)


@pytest.mark.parametrize(
    ("name", "attn_implementation"),
    [
        ("Qwen3_5MoeTextModel", "flash_attention_4"),
        ("lfm2_moe_short_conv", "flash_attention_2"),
        ("unknown_family_linear_attention", "flash_attention_4"),
        ("Qwen3Model", "flash_attention_4"),
    ],
)
def test_the_collator_factory_applies_the_same_rule_under_pp(monkeypatch, name, attn_implementation):
    """PP keeps the packed rows, so the factory refuses the same rows; the run's opt-in admits crossing linear
    attention off PP, not the markers a stage drops."""
    # GatedDeltaNet's segment-aware kernels need CUDA; taken as present, as on the images.
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", ())
    build, refused = _CASES[name]
    config = build()
    config._attn_implementation = attn_implementation

    def select(keeps_packed_rows: bool):
        return select_data_collator(
            word_level_tokenizer(),
            train_on_completions_only=False,
            packing=True,
            model_config=config,
            keeps_packed_rows=keeps_packed_rows,
            allow_cross_document_linear_attention=True,
        )

    assert isinstance(select(keeps_packed_rows=False), DataCollatorWithPacking)
    if refused:
        with pytest.raises(ValueError, match=_REFUSAL):
            select(keeps_packed_rows=True)
    else:
        assert isinstance(select(keeps_packed_rows=True), DataCollatorWithPacking)


def _pp_trainer(collator, config) -> SimpleNamespace:
    """The trainer state ``_pp_wrap_collator`` reads: the final collator, the stage (which carries the
    unsplit model's config), the batch shape and the pad id."""
    return SimpleNamespace(
        data_collator=collator,
        model=SimpleNamespace(config=config),
        args=SimpleNamespace(per_device_train_batch_size=1, max_length=16),
        _pp_n_microbatches=1,
        processing_class=word_level_tokenizer(),
        _pp_adapter=SimpleNamespace(pad_spec=None),
    )


@pytest.mark.parametrize("name", ["Qwen3_5TextModel", "Glm5NextTextModel"])
def test_the_pp_seam_checks_the_collator_against_the_stage_config(name):
    trainer = _pp_trainer(_packing_collator(), _PP_ROSTER[name][0]())
    with pytest.raises(ValueError, match=_REFUSAL):
        PipelineTrainerMixin._pp_wrap_collator(trainer)


def test_the_pp_seam_wraps_a_packing_collator_on_a_softmax_model():
    """Anti-vacuity for the refusal above: the same seam wraps the collator where nothing crosses."""
    collator = _packing_collator()
    trainer = _pp_trainer(collator, _PP_ROSTER["Qwen3Model"][0]())
    PipelineTrainerMixin._pp_wrap_collator(trainer)
    assert isinstance(trainer.data_collator, FixedShapeCollator) and trainer.data_collator.inner is collator
    assert collator.flatten_to_single_row is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
