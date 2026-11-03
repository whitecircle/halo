#!/usr/bin/env python
"""Packing under PP on models whose conv / linear-attention mixers need segment markers. Mixers no marker reaches carry
state across documents with no sign in the loss, so a packed row is refused; padded rows stay allowed."""

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
    TINY_QWEN4_EXP_CONFIG,
    TINY_QWEN35_CONFIG,
    TINY_QWEN35_MOE_CONFIG,
    TINY_QWEN35_VISION_CONFIG,
    TINY_STEP3P7_CONFIG,
    TINY_STEP3P7_VISION_CONFIG,
)
from tests.common.tiny_models import remote_config, unknown_hybrid_config, word_level_tokenizer

_REFUSAL = "packing under pipeline parallelism is refused"


# Backbone -> (stage config, whether a packed row is refused); shipped GLM-5 Next and Step-3.7 stages carry the
# wrapper's config.
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
    "Qwen3_5TextModel": (lambda: CONFIG_MAPPING["qwen3_5_text"](**TINY_QWEN35_CONFIG), False),
    "Qwen3_5MoeTextModel": (lambda: CONFIG_MAPPING["qwen3_5_moe_text"](**TINY_QWEN35_MOE_CONFIG), False),
    "Qwen4ExpTextModel": (lambda: CONFIG_MAPPING["qwen4_exp_text"](**TINY_QWEN4_EXP_CONFIG), False),
    "Glm5NextTextModel": (
        lambda: Glm5NextConfig(text_config=dict(TINY_GLM5_CONFIG), vision_config=dict(TINY_GLM5_VISION_CONFIG)),
        True,
    ),
}
# The composite config a multimodal Qwen3.5/3.6 MoE stage carries, and off the roster a seq_idx-only family, an
# unmarked layer-group family, an unlisted family, a softmax-only MoE.
_DETECTED = {
    "qwen3_5_moe_composite": (
        lambda: Qwen3_5MoeConfig(
            text_config=dict(TINY_QWEN35_MOE_CONFIG), vision_config=dict(TINY_QWEN35_VISION_CONFIG)
        ),
        False,
    ),
    "lfm2_moe_short_conv": (lambda: CONFIG_MAPPING["lfm2_moe"](**TINY_LFM2_MOE_CONFIG), False),
    "ring_lightning_attention": (lambda: remote_config(BAILING_MOE_RING_MINI), True),
    "unknown_family_linear_attention": (unknown_hybrid_config, True),
    "mistral4_softmax_only": (lambda: CONFIG_MAPPING["mistral4"](**TINY_MISTRAL4_CONFIG), False),
}
_CASES = {**_PP_ROSTER, **_DETECTED}


def _packing_collator() -> DataCollatorWithPacking:
    return DataCollatorWithPacking(word_level_tokenizer())


def test_the_roster_is_every_registered_pp_backbone():
    """A family without an entry here would go unchecked."""
    assert set(_PP_ROSTER) == {name for name, spec in PP_SPEC_MAP.items() if spec.validated}


@pytest.fixture
def segment_aware_kernels(monkeypatch):
    """They need CUDA; taken as present, as on the images."""
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", ())


@pytest.mark.parametrize("name", sorted(_CASES))
def test_a_packed_row_is_refused_where_no_marker_reaches_a_mixer(segment_aware_kernels, name):
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
def test_the_collator_factory_applies_the_same_rule_under_pp(segment_aware_kernels, name, attn_implementation):
    """The opt-in admits crossing linear attention off PP, not a mixer no marker reaches."""
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
    """The stage carries the unsplit model's config."""
    return SimpleNamespace(
        data_collator=collator,
        model=SimpleNamespace(config=config),
        args=SimpleNamespace(per_device_train_batch_size=1, max_length=16),
        _pp_n_microbatches=1,
        processing_class=word_level_tokenizer(),
        _pp_adapter=SimpleNamespace(pad_spec=None),
    )


def test_the_pp_seam_checks_the_collator_against_the_stage_config():
    trainer = _pp_trainer(_packing_collator(), _PP_ROSTER["Glm5NextTextModel"][0]())
    with pytest.raises(ValueError, match=_REFUSAL):
        PipelineTrainerMixin._pp_wrap_collator(trainer)


def test_a_marker_reading_family_needs_the_kernels_that_read_them(monkeypatch):
    """The torch fallbacks drop the markers."""
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", (("fla>=0.2.2", lambda: False),))
    trainer = _pp_trainer(_packing_collator(), _PP_ROSTER["Qwen3_5MoeTextModel"][0]())
    with pytest.raises(ValueError, match="unavailable"):
        PipelineTrainerMixin._pp_wrap_collator(trainer)


@pytest.mark.parametrize("name", ["Qwen3Model", "Qwen3_5TextModel", "Qwen4ExpTextModel"])
def test_the_pp_seam_wraps_a_packing_collator_where_every_mixer_is_reached(segment_aware_kernels, name):
    """Anti-vacuity: the same seam wraps the collator where nothing crosses."""
    collator = _packing_collator()
    trainer = _pp_trainer(collator, _PP_ROSTER[name][0]())
    PipelineTrainerMixin._pp_wrap_collator(trainer)
    assert isinstance(trainer.data_collator, FixedShapeCollator) and trainer.data_collator.inner is collator
    assert collator.flatten_to_single_row is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
