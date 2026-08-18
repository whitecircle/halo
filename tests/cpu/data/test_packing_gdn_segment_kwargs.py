#!/usr/bin/env python
"""GatedDeltaNet families get segment kwargs from the packing collator.

``Qwen3_5(Moe)GatedDeltaNet`` and Qwen3-Next's linear-attention layers read ``seq_idx`` (their
causal conv) and ``cu_seq_lens_q`` (the chunked delta rule) from forward kwargs; nothing derives
them model-side, so without emission packed documents mix inside those layers while attention is
correctly isolated. The factory switches the emission on for those families only — and refuses a
multi-document run outright when the wheels that select the segment-aware kernels
(``causal_conv1d``, ``fla``) are absent, because transformers' torch fallbacks read neither marker.

    python tests/cpu/data/test_packing_gdn_segment_kwargs.py
"""

import importlib.util
from unittest.mock import MagicMock

import pytest
import torch
from accelerate import PartialState
from transformers.models.qwen3_next import Qwen3NextConfig

from src.data.collators import factory
from src.data.collators.factory import GDN_SEGMENT_AWARE_BACKENDS, select_data_collator

PartialState()  # the factory logs through accelerate's logger, which needs the state initialized


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
    """Force transformers' GDN fast-path predicates, ``missing`` reading as unavailable.

    The real predicates (``is_causal_conv1d_available`` / ``is_flash_linear_attention_available``)
    also require a CUDA-capable torch, so on a CPU test host both read as "fallback" — the allowed
    cases have to force them on, and the refusal cases pick one off. This is how an environment
    without them — a bare pip install, a slim CI image, a CPU box — reaches the factory.
    """
    patched = tuple((name, (lambda ok=(name not in missing): ok)) for name, _ in GDN_SEGMENT_AWARE_BACKENDS)
    monkeypatch.setattr(factory, "GDN_SEGMENT_AWARE_BACKENDS", patched)


def test_factory_emits_segment_kwargs_for_gdn_families(monkeypatch):
    _backends(monkeypatch)
    config = Qwen3NextConfig(num_hidden_layers=1)
    config._attn_implementation = "flash_attention_2"
    collator = select_data_collator(make_tokenizer(), packing=True, model_config=config)

    rows = [{"input_ids": [5, 6, 7, 8, 9], "attention_mask": [1] * 5, "seq_lengths": [3, 2]}]
    batch = collator.torch_call(rows)

    assert batch["seq_idx"].tolist() == [[0, 0, 0, 1, 1]]
    assert batch["cu_seq_lens_q"].tolist() == [0, 3, 5]
    assert batch["cu_seq_lens_q"].dtype == torch.int32
    assert torch.equal(batch["cu_seq_lens_q"], batch["cu_seq_lens_k"])
    assert batch["max_length_q"] == 3 and batch["max_length_k"] == 3

    # The boundaries must agree with what flash attention would re-derive from position_ids —
    # disagreement would silently change attention itself, not just the delta rule.
    positions = batch["position_ids"][0]
    starts = (positions == 0).nonzero(as_tuple=True)[0].tolist()
    assert batch["cu_seq_lens_q"].tolist() == [*starts, positions.numel()]


def test_pp_keeps_rows_is_refused(monkeypatch):
    """PP keeps the packed rows, and the delta rule's cu_seq_lens have no per-row convention —
    a half-isolated run (conv isolated, delta rule crossing) must be refused, not silently built."""
    _backends(monkeypatch)  # else the wheel refusal fires first and this would pass for it
    config = Qwen3NextConfig(num_hidden_layers=1)
    config._attn_implementation = "flash_attention_2"
    with pytest.raises(ValueError, match="GatedDeltaNet"):
        select_data_collator(make_tokenizer(), packing=True, model_config=config, keeps_packed_rows=True)


def test_other_families_get_no_segment_kwargs():
    from transformers import Mistral4Config

    config = Mistral4Config(num_hidden_layers=1)
    config._attn_implementation = "flash_attention_2"
    collator = select_data_collator(make_tokenizer(), packing=True, model_config=config)
    batch = collator.torch_call([{"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "seq_lengths": [2, 1]}])
    assert "cu_seq_lens_q" not in batch and "seq_idx" not in batch


@pytest.mark.parametrize("missing", ["causal_conv1d", "fla>=0.2.2"])
def test_packing_refused_when_a_segment_aware_wheel_is_missing(monkeypatch, missing):
    """Isolation here is not a property of the collator: transformers picks the segment-aware conv and
    delta-rule kernels on those imports, and its torch fallbacks drop ``seq_idx`` / ``cu_seq_lens_q``
    without a word. Emitting the markers into a fallback that ignores them trains on cross-document
    conv and recurrent state, so the run must be refused rather than silently wrong."""
    _backends(monkeypatch, missing=(missing,))
    config = Qwen3NextConfig(num_hidden_layers=1)
    config._attn_implementation = "flash_attention_2"
    with pytest.raises(ValueError) as excinfo:
        select_data_collator(make_tokenizer(), packing=True, model_config=config)
    message = str(excinfo.value)
    assert missing in message, message
    assert "qwen3_next" in message, message
    assert "boundaries" in message, message


def test_padding_free_refused_when_a_wheel_is_missing(monkeypatch):
    """padding_free puts several documents in one row too — same exposure, same refusal."""
    _backends(monkeypatch, missing=("fla>=0.2.2",))
    config = Qwen3NextConfig(num_hidden_layers=1)
    config._attn_implementation = "flash_attention_2"
    with pytest.raises(ValueError, match="padding_free"):
        select_data_collator(make_tokenizer(), padding_free=True, model_config=config)


def test_packing_allowed_when_both_wheels_are_importable(monkeypatch):
    """Anti-vacuity: with the backends available packing still builds, and the image really ships the
    wheels (the predicates additionally require CUDA, which a CPU test host does not have)."""
    assert all(importlib.util.find_spec(wheel) is not None for wheel in ("causal_conv1d", "fla")), (
        "this image is missing a GDN wheel — the refusal above would not be provably caused by hiding it"
    )
    _backends(monkeypatch)
    config = Qwen3NextConfig(num_hidden_layers=1)
    config._attn_implementation = "flash_attention_2"
    assert select_data_collator(make_tokenizer(), packing=True, model_config=config) is not None


def test_other_families_are_not_gated_on_the_gdn_wheels(monkeypatch):
    """The fallbacks only matter where the model has a GatedDeltaNet: nothing else reads the markers."""
    from transformers import Mistral4Config

    _backends(monkeypatch, missing=tuple(name for name, _ in GDN_SEGMENT_AWARE_BACKENDS))
    config = Mistral4Config(num_hidden_layers=1)
    config._attn_implementation = "flash_attention_2"
    assert select_data_collator(make_tokenizer(), packing=True, model_config=config) is not None


def test_unpacked_gdn_run_is_not_gated(monkeypatch):
    """One document per row has no boundary to cross — the wheels are irrelevant there."""
    _backends(monkeypatch, missing=tuple(name for name, _ in GDN_SEGMENT_AWARE_BACKENDS))
    config = Qwen3NextConfig(num_hidden_layers=1)
    config._attn_implementation = "flash_attention_2"
    assert select_data_collator(make_tokenizer(), packing=False, model_config=config) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
