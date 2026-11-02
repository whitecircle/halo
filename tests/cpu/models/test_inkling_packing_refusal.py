#!/usr/bin/env python
"""CPU test: Inkling's multi-document rows are refused unless the run accepts the short convolutions crossing them.

Inkling's attention isolates a packed row's documents, but its four depthwise causal short convolutions per layer
receive no segment marker (the modeling reads ``seq_idx``, its call sites forward no kwargs), so each document's
first tokens read the previous document's last ones on every backend, invisible in the loss. The family spec names
that mixer (``MoEFamilySpec.unmarked_sequence_mixer``), read through the multimodal wrapper too, so SFT's ``packing``
and ``padding_free`` are refused as an unmarked linear-attention row is, and ``allow_cross_document_linear_attention``
admits them with one warning.

    python tests/cpu/models/test_inkling_packing_refusal.py
"""

import logging
from unittest.mock import MagicMock

import pytest
import torch
from transformers import InklingConfig, InklingTextConfig

from halo.data.collators.factory import select_data_collator
from halo.models import segment_markers
from halo.models.segment_markers import find_unmarked_sequence_mixer, reject_pipeline_packed_rows
from tests.common.models import TINY_INKLING_CONFIG, TINY_INKLING_VISION_CONFIG
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

CONFIGS = {
    "inkling_text": lambda: InklingTextConfig(**TINY_INKLING_CONFIG),
    "inkling_mm_model": lambda: InklingConfig(
        text_config=TINY_INKLING_CONFIG, vision_config=TINY_INKLING_VISION_CONFIG
    ),
}
MIXER = "depthwise causal short convolutions"
DOCUMENTS = (12, 12)


def _tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.padding_side = 0, 2, "right"
    return tokenizer


def _select(config, flag: str, *, allow: bool):
    return select_data_collator(
        _tokenizer(),
        train_on_completions_only=False,
        model_config=config,
        allow_cross_document_linear_attention=allow,
        **{flag: True},
    )


@pytest.mark.parametrize("flag", ["packing", "padding_free"])
@pytest.mark.parametrize("model_type", sorted(CONFIGS))
def test_multi_document_rows_are_refused_naming_the_convolutions(model_type, flag):
    """On every backend: a flash label is the one where nothing else refuses ``padding_free``."""
    config = CONFIGS[model_type]()
    config._attn_implementation = "flash_attention_2"
    assert config.model_type == model_type
    assert find_unmarked_sequence_mixer(config).startswith(MIXER)
    with pytest.raises(ValueError, match=rf"{flag}=True is refused for model_type='{model_type}': its {MIXER}"):
        _select(config, flag, allow=False)


@pytest.mark.parametrize("model_type", sorted(CONFIGS))
def test_the_opt_in_admits_them_with_one_warning(model_type, monkeypatch, caplog):
    monkeypatch.setattr(segment_markers, "_crossing_allowed_warned", set())
    config = CONFIGS[model_type]()
    config._attn_implementation = "sdpa"
    with caplog.at_level(logging.WARNING, logger=segment_markers.__name__):
        for _ in range(2):
            assert _select(config, "packing", allow=True) is not None
    warnings = [record.getMessage() for record in caplog.records if MIXER in record.getMessage()]
    assert len(warnings) == 1, warnings


@pytest.mark.parametrize("model_type", sorted(CONFIGS))
def test_pipeline_packing_names_the_convolutions(model_type):
    with pytest.raises(ValueError, match=f"its {MIXER}.* read no document boundary on any path"):
        reject_pipeline_packed_rows(CONFIGS[model_type]())


def test_the_convolutions_carry_one_document_into_the_next_on_sdpa():
    """The crossing the refusal stands for: with the attention isolated by the restarting positions, changing the
    first document still moves the second one's hidden states."""
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES["inkling_text"]).eval()
    model.config._attn_implementation = "sdpa"
    generator = torch.Generator().manual_seed(3)
    input_ids = torch.randint(3, model.config.vocab_size, (1, sum(DOCUMENTS)), generator=generator)
    position_ids = torch.cat([torch.arange(length) for length in DOCUMENTS])[None]
    changed = input_ids.clone()
    changed[0, : DOCUMENTS[0]] = (changed[0, : DOCUMENTS[0]] + 7) % (model.config.vocab_size - 3) + 3
    with torch.no_grad():
        second, second_after_change = (
            model.model(input_ids=ids, position_ids=position_ids, use_cache=False).last_hidden_state[0, DOCUMENTS[0] :]
            for ids in (input_ids, changed)
        )
    drift = (second - second_after_change).abs().max().item()
    assert drift > 1e-3 * second.abs().max().item(), f"the second document did not move ({drift})"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
