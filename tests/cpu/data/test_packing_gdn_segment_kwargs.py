#!/usr/bin/env python
"""GatedDeltaNet families get segment kwargs from the packing collator: their layers read ``seq_idx`` and
``cu_seq_lens_q`` from forward kwargs, and nothing derives them model-side. A multi-document run is refused when
``causal_conv1d`` or ``fla`` is absent, since transformers' torch fallbacks read neither marker."""

import importlib.util
from unittest.mock import MagicMock

import pytest
import torch
from transformers import Mistral4Config
from transformers.models.qwen3_next import Qwen3NextConfig

from halo.data.collators.factory import select_data_collator
from halo.models import segment_markers
from halo.models.segment_markers import GDN_SEGMENT_AWARE_BACKENDS


def make_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    tok.padding_side = "right"

    def _pad(features, **kwargs):
        max_len = max(len(f["input_ids"]) for f in features)
        out = {"input_ids": [], "attention_mask": []}
        for f in features:
            ids = list(f["input_ids"])
            pad = max_len - len(ids)
            out["input_ids"].append(ids + [tok.pad_token_id] * pad)
            out["attention_mask"].append([1] * len(ids) + [0] * pad)
        return {key: torch.tensor(value) for key, value in out.items()}

    tok.pad.side_effect = _pad
    return tok


def _backends(monkeypatch, missing: tuple[str, ...] = ()) -> None:
    """Force transformers' GDN fast-path predicates, ``missing`` reading as unavailable. The real ones also require
    a CUDA-capable torch, so on a CPU host both read as fallback."""
    patched = tuple((name, (lambda ok=(name not in missing): ok)) for name, _ in GDN_SEGMENT_AWARE_BACKENDS)
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", patched)


def test_factory_emits_segment_kwargs_for_gdn_families(monkeypatch):
    _backends(monkeypatch)
    config = Qwen3NextConfig(num_hidden_layers=1)
    config._attn_implementation = "flash_attention_2"
    collator = select_data_collator(
        make_tokenizer(), train_on_completions_only=False, packing=True, model_config=config
    )

    rows = [{"input_ids": [5, 6, 7, 8, 9], "attention_mask": [1] * 5, "seq_lengths": [3, 2]}]
    batch = collator.torch_call(rows)

    assert batch["seq_idx"].tolist() == [[0, 0, 0, 1, 1]]
    assert batch["cu_seq_lens_q"].tolist() == [0, 3, 5]
    assert batch["cu_seq_lens_q"].dtype == torch.int32
    assert torch.equal(batch["cu_seq_lens_q"], batch["cu_seq_lens_k"])
    assert batch["max_length_q"] == 3 and batch["max_length_k"] == 3

    # The boundaries must agree with what flash attention re-derives from position_ids, or attention itself changes.
    positions = batch["position_ids"][0]
    starts = (positions == 0).nonzero(as_tuple=True)[0].tolist()
    assert batch["cu_seq_lens_q"].tolist() == [*starts, positions.numel()]


def test_pp_keeps_rows_and_leaves_the_markers_to_its_stages(monkeypatch):
    """PP keeps the packed rows, whose stages rebuild seq_idx and the varlen set per microbatch: the collator
    builds the row's seq_idx and no varlen set, which exists only on a flattened row."""
    _backends(monkeypatch)  # else the wheel refusal fires first
    config = Qwen3NextConfig(num_hidden_layers=1)
    config._attn_implementation = "flash_attention_2"
    collator = select_data_collator(
        make_tokenizer(), train_on_completions_only=False, packing=True, model_config=config, keeps_packed_rows=True
    )
    collator.flatten_to_single_row = False  # what the PP trainer sets on the collator it keeps
    batch = collator.torch_call([{"input_ids": [5, 6, 7, 8, 9], "attention_mask": [1] * 5, "seq_lengths": [2, 3]}])
    assert batch["seq_idx"].tolist() == [[0, 0, 1, 1, 1]]
    assert "cu_seq_lens_q" not in batch


def test_other_families_get_no_segment_kwargs():
    """Off flash, where the varlen set is attention's (``test_packing_flash_varlen_kwargs.py``), a family
    whose mixers read no marker gets none."""
    config = Mistral4Config(num_hidden_layers=1)
    config._attn_implementation = "sdpa"
    collator = select_data_collator(
        make_tokenizer(), train_on_completions_only=False, packing=True, model_config=config
    )
    batch = collator.torch_call([{"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "seq_lengths": [2, 1]}])
    assert "cu_seq_lens_q" not in batch and "seq_idx" not in batch


@pytest.mark.parametrize("missing", ["causal_conv1d", "fla>=0.2.2"])
def test_packing_refused_when_a_segment_aware_wheel_is_missing(monkeypatch, missing):
    """transformers picks the segment-aware kernels on those imports, and its torch fallbacks drop the markers
    without a word, training on cross-document conv and recurrent state."""
    _backends(monkeypatch, missing=(missing,))
    config = Qwen3NextConfig(num_hidden_layers=1)
    config._attn_implementation = "flash_attention_2"
    with pytest.raises(ValueError) as excinfo:
        select_data_collator(make_tokenizer(), train_on_completions_only=False, packing=True, model_config=config)
    message = str(excinfo.value)
    assert missing in message, message
    assert "qwen3_next" in message, message
    assert "boundaries" in message, message


def test_padding_free_refused_when_a_wheel_is_missing(monkeypatch):
    """padding_free puts several documents in one row too."""
    _backends(monkeypatch, missing=("fla>=0.2.2",))
    config = Qwen3NextConfig(num_hidden_layers=1)
    config._attn_implementation = "flash_attention_2"
    with pytest.raises(ValueError, match="padding_free"):
        select_data_collator(make_tokenizer(), train_on_completions_only=False, padding_free=True, model_config=config)


def test_packing_allowed_when_both_wheels_are_importable(monkeypatch):
    """With the backends available packing builds, and the image ships the wheels (the predicates also require
    CUDA)."""
    assert all(importlib.util.find_spec(wheel) is not None for wheel in ("causal_conv1d", "fla")), (
        "this image is missing a GDN wheel — the refusal above would not be provably caused by hiding it"
    )
    _backends(monkeypatch)
    config = Qwen3NextConfig(num_hidden_layers=1)
    config._attn_implementation = "flash_attention_2"
    assert (
        select_data_collator(make_tokenizer(), train_on_completions_only=False, packing=True, model_config=config)
        is not None
    )


def test_other_families_are_not_gated_on_the_gdn_wheels(monkeypatch):
    """The fallbacks only matter where the model has a GatedDeltaNet: nothing else reads the markers."""
    _backends(monkeypatch, missing=tuple(name for name, _ in GDN_SEGMENT_AWARE_BACKENDS))
    config = Mistral4Config(num_hidden_layers=1)
    config._attn_implementation = "flash_attention_2"
    assert (
        select_data_collator(make_tokenizer(), train_on_completions_only=False, packing=True, model_config=config)
        is not None
    )


def test_unpacked_gdn_run_is_not_gated(monkeypatch):
    """One document per row has no boundary to cross."""
    _backends(monkeypatch, missing=tuple(name for name, _ in GDN_SEGMENT_AWARE_BACKENDS))
    config = Qwen3NextConfig(num_hidden_layers=1)
    config._attn_implementation = "flash_attention_2"
    assert (
        select_data_collator(make_tokenizer(), train_on_completions_only=False, packing=False, model_config=config)
        is None
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
