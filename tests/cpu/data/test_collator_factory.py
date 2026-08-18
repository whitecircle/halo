#!/usr/bin/env python
"""
Tests for select_data_collator routing logic in collator_factory.py.

Run: python tests/cpu/data/test_collator_factory.py
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch
from accelerate import PartialState
from transformers import AutoTokenizer

from src.data.collators.factory import select_data_collator
from src.data.collators.packing import DataCollatorForCausalLMWithPadding
from src.models.patches.attention import VARLEN_ATTN_IMPLEMENTATIONS


def _make_tokenizer():
    """Create a minimal mock tokenizer for collator construction."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 2
    tok.padding_side = "right"
    return tok


def _mock_model_config(model_type: str = "qwen3") -> MagicMock:
    """Real strings where the factory reads them: ``model_type_matches`` does ``.startswith`` on
    ``model_type`` via ``get_text_config``, and a bare MagicMock is truthy for EVERY family gate."""
    config = MagicMock()
    config.model_type = model_type
    config.get_text_config.return_value = config
    return config


def _select(**kwargs):
    """Call select_data_collator with mocked PartialState (no accelerate init)."""
    with patch("src.data.collators.factory.PartialState") as MockPS:
        MockPS.return_value.is_main_process = False  # suppress logging
        return select_data_collator(tokenizer=_make_tokenizer(), **kwargs)


def test_packing_and_padding_free_conflict():
    """packing + padding_free raises ValueError."""
    try:
        _select(packing=True, padding_free=True)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "packing" in str(e).lower() and "padding_free" in str(e).lower()


def test_cp_with_padding_free_conflict():
    """CP + padding_free raises ValueError."""
    try:
        _select(use_context_parallel=True, padding_free=True)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "CP" in str(e) or "Context Parallelism" in str(e)


def test_completions_without_template():
    """train_on_completions_only without assistant_message_template raises ValueError."""
    try:
        _select(train_on_completions_only=True, assistant_message_template=None)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "assistant_message_template" in str(e)


def test_last_only_without_completions():
    """train_on_last_assistant_only without train_on_completions_only raises ValueError."""
    try:
        _select(train_on_last_assistant_only=True, train_on_completions_only=False)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "train_on_last_assistant_only" in str(e)


def test_cp_packing_rejected():
    """CP + packing is rejected: the Ulysses CP attention path has no per-document boundaries,
    so packed documents would silently attend across each other."""
    with pytest.raises(ValueError, match="packing is not supported with Context Parallelism"):
        _select(use_context_parallel=True, packing=True, pad_to_multiple_of=8)
    with pytest.raises(ValueError, match="packing is not supported with Context Parallelism"):
        _select(
            use_context_parallel=True,
            packing=True,
            train_on_completions_only=True,
            assistant_message_template="<|assistant|>",
            pad_to_multiple_of=8,
        )


def test_cp_no_packing():
    """CP without packing -> padded causal-LM collator that PRODUCES labels (a labels-less
    collator makes every CP SFT step fail with 'model did not return a loss')."""
    collator = _select(use_context_parallel=True, pad_to_multiple_of=8)
    assert type(collator).__name__ == "DataCollatorForCausalLMWithPadding"
    assert collator.pad_to_multiple_of == 8


def test_cp_completions_routing():
    """CP + train_on_completions_only must keep the completion mask, not silently drop it."""
    collator = _select(
        use_context_parallel=True,
        train_on_completions_only=True,
        assistant_message_template="<|assistant|>",
        pad_to_multiple_of=8,
    )
    assert type(collator).__name__ == "DataCollatorForCompletionOnlyLM"
    assert collator.pad_to_multiple_of == 8


def test_cp_collator_produces_and_preserves_labels():
    """Real-tokenizer check: labels built from input_ids when absent, precomputed (ragged,
    prompt-masked) labels padded with -100 — not crashed on, not discarded."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    collator = DataCollatorForCausalLMWithPadding(tokenizer=tok, mlm=False, pad_to_multiple_of=4)

    batch = collator([{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}])
    assert batch["labels"][0, :3].tolist() == [1, 2, 3]
    assert (batch["labels"][0, 3:] == -100).all()

    batch = collator(
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
        ]
    )
    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["labels"][0, :3].tolist() == [-100, 2, 3]
    assert batch["labels"][1, :2].tolist() == [-100, 5]
    assert (batch["labels"][1, 2:] == -100).all()
    assert isinstance(batch["labels"], torch.Tensor)


def test_cp_collator_restores_real_eos_when_pad_equals_eos():
    """Raw-labels path with pad == eos (GLM-4.7-Flash natively; any family via the pad=eos
    fallback): the parent LM collator masks every pad-valued label — including the REAL
    turn-ending EOS — so the model never learns to stop. Real-token EOS labels must be restored
    from input_ids (attention_mask == 1); true padding stays masked.

    Without the restore, labels at the real EOS positions are -100.
    """
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token  # gpt2 has no pad → the pad=eos fallback layout
    eos = tok.eos_token_id
    collator = DataCollatorForCausalLMWithPadding(tokenizer=tok, mlm=False)

    batch = collator(
        [
            {"input_ids": [1, 2, eos], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, eos], "attention_mask": [1, 1]},
        ]
    )
    assert batch["labels"][0, 2].item() == eos, "real turn-ending EOS erased on the raw-labels path (pad==eos)"
    assert batch["labels"][1, 1].item() == eos, "real turn-ending EOS erased on the padded row (pad==eos)"
    assert (batch["labels"][1, 2:] == -100).all(), "true padding must stay masked"

    # Precomputed labels stay authoritative: no restore over baked masks.
    baked = collator(
        [
            {"input_ids": [1, 2, eos], "attention_mask": [1, 1, 1], "labels": [-100, 2, -100]},
        ]
    )
    assert baked["labels"][0].tolist() == [-100, 2, -100], "baked labels must not be overwritten by the EOS restore"


def test_padding_free_with_completions():
    """padding_free + completions -> DataCollatorWithFlatteningAndCompletionMask."""
    collator = _select(
        padding_free=True,
        train_on_completions_only=True,
        assistant_message_template="<|assistant|>",
    )
    assert type(collator).__name__ == "DataCollatorWithFlatteningAndCompletionMask"


def test_padding_free_no_completions():
    """padding_free alone -> DataCollatorWithFlattening."""
    collator = _select(padding_free=True)
    assert type(collator).__name__ == "DataCollatorWithFlattening"


def test_packing_with_completions():
    """packing + completions -> DataCollatorForCompletionOnlyLMWithPacking."""
    collator = _select(
        packing=True,
        train_on_completions_only=True,
        assistant_message_template="<|assistant|>",
    )
    assert type(collator).__name__ == "DataCollatorForCompletionOnlyLMWithPacking"


def test_packing_no_completions():
    """packing alone -> DataCollatorWithPacking."""
    collator = _select(packing=True)
    assert type(collator).__name__ == "DataCollatorWithPacking"


def test_completions_only():
    """completions only -> DataCollatorForCompletionOnlyLM."""
    collator = _select(
        train_on_completions_only=True,
        assistant_message_template="<|assistant|>",
    )
    assert type(collator).__name__ == "DataCollatorForCompletionOnlyLM"


def test_default_returns_none():
    """No flags set -> returns None (TRL default)."""
    collator = _select()
    assert collator is None


def test_completions_template_propagated():
    """The assistant_message_template reaches the collator's response_prompt_template."""
    collator = _select(
        train_on_completions_only=True,
        assistant_message_template="<|MYASSISTANT|>",
    )
    assert collator.response_prompt_template == "<|MYASSISTANT|>"


def test_last_only_flag_propagated_to_completions_collator():
    """train_on_last_assistant_only flows into the standard completions collator."""
    collator = _select(
        train_on_completions_only=True,
        assistant_message_template="<|assistant|>",
        train_on_last_assistant_only=True,
    )
    assert type(collator).__name__ == "DataCollatorForCompletionOnlyLM"
    assert collator.train_on_last_assistant_only is True


def test_last_only_flag_propagated_to_packing_collator():
    """train_on_last_assistant_only flows into the packing completions collator too."""
    collator = _select(
        packing=True,
        train_on_completions_only=True,
        assistant_message_template="<|assistant|>",
        train_on_last_assistant_only=True,
    )
    assert type(collator).__name__ == "DataCollatorForCompletionOnlyLMWithPacking"
    assert collator.train_on_last_assistant_only is True


def test_last_only_default_false_when_unset():
    """When not requested, the selected completions collator keeps train_on_last_assistant_only=False."""
    collator = _select(
        train_on_completions_only=True,
        assistant_message_template="<|assistant|>",
    )
    assert collator.train_on_last_assistant_only is False


def test_padding_free_completions_template_propagated():
    """Padding-free + completions wires the template into the flattening completion-mask collator."""
    collator = _select(
        padding_free=True,
        train_on_completions_only=True,
        assistant_message_template="<|assistant|>",
        train_on_last_assistant_only=True,
    )
    assert type(collator).__name__ == "DataCollatorWithFlatteningAndCompletionMask"
    assert collator.response_prompt_template == "<|assistant|>"
    assert collator.train_on_last_assistant_only is True


# Varlen gate — derived from the RESOLVED attention impl, never from a family list.


class _Cfg:
    """Stand-in for a HF config carrying only the resolved attention implementation."""

    def __init__(self, attn_impl):
        self._attn_implementation = attn_impl


@pytest.mark.parametrize("attn_impl", ["eager", "sdpa", "flex_attention", None])
def test_non_varlen_attention_rejects_padding_free(attn_impl):
    """padding_free must be refused without a varlen kernel — the mask cost is unbounded.

    padding_free flattens the whole batch into ONE row, so a non-varlen backend materializes a dense
    mask over the summed token count: the area grows with the square of the batch size. No family
    name appears here or in the implementation — adding a new one must not require editing either.
    """
    with pytest.raises(ValueError, match="varlen"):
        _select(padding_free=True, model_config=_Cfg(attn_impl))


@pytest.mark.parametrize("attn_impl", ["eager", "sdpa", "flex_attention", None])
def test_non_varlen_attention_allows_packing(attn_impl):
    """packing must be ALLOWED without a varlen kernel: document isolation is not backend-specific.

    Both collators emit position_ids that reset per document, and transformers derives a
    packed_sequence_mask from them for every backend (``masking_utils.find_packed_sequence_indices``),
    so eager/SDPA isolate documents just as flash does — verified directly against the installed transformers.
    Refusing packing here would strand the models that CANNOT run a varlen kernel (Gemma 4's
    head_dim=512, Laguna) with no packing at all, on a false correctness premise. The real difference
    is cost — a dense ``[max_length, max_length]`` mask, the same one a padded batch builds — which
    warrants a warning, not a raise.

    The selected class is pinned, not merely non-None: a gate that quietly downgraded packing to the
    padded collator would keep every row padded to max_length and silently stop packing.
    """
    collator = _select(packing=True, model_config=_Cfg(attn_impl))
    assert type(collator).__name__ == "DataCollatorWithPacking"


@pytest.mark.parametrize("flag", ["padding_free", "packing"])
@pytest.mark.parametrize("attn_impl", ["flash_attention_2", "flash_attention_3", "flash_attention_4"])
def test_varlen_attention_allows_flattening(attn_impl, flag):
    """Every flash variant honours document boundaries, so the gate must let it through — into the
    collator the flag asks for, not a padded fallback that would undo the throughput win."""
    expected = {"padding_free": "DataCollatorWithFlattening", "packing": "DataCollatorWithPacking"}[flag]
    collator = _select(**{flag: True}, model_config=_Cfg(attn_impl))
    assert type(collator).__name__ == expected


def test_varlen_gate_covers_every_declared_flash_impl():
    """The accepted set IS ``VARLEN_ATTN_IMPLEMENTATIONS`` — no second hand-maintained copy.

    Fails if someone adds a flash backend to the constant but the collator keeps its own list.
    """
    assert VARLEN_ATTN_IMPLEMENTATIONS, "the varlen constant went empty — the loop below tests nothing"
    for impl in VARLEN_ATTN_IMPLEMENTATIONS:
        collator = _select(packing=True, model_config=_Cfg(impl))
        assert type(collator).__name__ == "DataCollatorWithPacking", impl


# packing + per_device_train_batch_size > 1 builds one merged row, not B sequences.


def _warnings_for(**kwargs) -> str:
    """The main-process warnings select_data_collator emits, as one lowercased blob."""
    with (
        patch("src.data.collators.factory.PartialState") as MockPS,
        patch("src.data.collators.factory.logger") as mock_logger,
    ):
        MockPS.return_value.is_main_process = True
        select_data_collator(tokenizer=_make_tokenizer(), **kwargs)
        rendered = []
        for call in mock_logger.warning.call_args_list:
            msg, *fmt_args = call.args
            rendered.append(msg % tuple(fmt_args) if fmt_args else msg)
        return " ".join(rendered).lower()


def test_packing_above_batch_one_warns_about_the_merge():
    """batch>1 under packing must say what it builds — one merged row, not B independent sequences.

    The merge is backend-independent (flatten_packed_batch), so the warning cannot hide behind the
    non-varlen branch: on FA4, the production path, the shape change is otherwise silent.
    """
    config = _mock_model_config()
    config._attn_implementation = VARLEN_ATTN_IMPLEMENTATIONS[0]
    blob = _warnings_for(packing=True, per_device_train_batch_size=2, model_config=config)
    assert "does not run 2 independent sequences" in blob, f"no merge warning on a varlen backend: {blob}"


def test_packing_at_batch_one_is_silent():
    """Anti-over-reach: the default shape must not warn."""
    config = _mock_model_config()
    config._attn_implementation = VARLEN_ATTN_IMPLEMENTATIONS[0]
    blob = _warnings_for(packing=True, per_device_train_batch_size=1, model_config=config)
    assert "independent sequences" not in blob, f"batch 1 must not warn: {blob}"


def test_batch_above_one_without_packing_is_silent():
    """Anti-over-reach: unpacked batches are never merged."""
    config = _mock_model_config()
    config._attn_implementation = VARLEN_ATTN_IMPLEMENTATIONS[0]
    blob = _warnings_for(packing=False, per_device_train_batch_size=8, model_config=config)
    assert "independent sequences" not in blob, f"unpacked batches must not warn: {blob}"


def test_pipeline_parallelism_is_not_warned():
    """PP keeps the rows and splits them into microbatches — batch>1 there IS B independent rows."""
    config = _mock_model_config()
    config._attn_implementation = VARLEN_ATTN_IMPLEMENTATIONS[0]
    blob = _warnings_for(packing=True, per_device_train_batch_size=4, keeps_packed_rows=True, model_config=config)
    assert "independent sequences" not in blob, f"PP must not be warned about a merge it never does: {blob}"


# The chat-template probe


_FACTORY_LOGGER = "src.data.collators.factory"


def _probe_tokenizer(*, renders: bool):
    """Mock tokenizer whose chat template either renders text or refuses every probe conversation."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 2
    tok.padding_side = "right"
    tok.chat_template = "{{ messages }}"
    tok.encode.return_value = [7, 7]
    if renders:
        tok.apply_chat_template.return_value = "user: q <|assistant|> a"
    else:
        tok.apply_chat_template.side_effect = TypeError("template requires a 'documents' argument")
    return tok


def _select_with_probe(tokenizer):
    # The real state, not a mock: accelerate's logger refuses to emit before it exists, and this
    # single-process one reports is_main_process — which is the gate the warning sits behind.
    PartialState()
    return select_data_collator(
        tokenizer=tokenizer,
        train_on_completions_only=True,
        assistant_message_template="<|assistant|>",
    )


def test_unrenderable_chat_template_probe_warns(caplog):
    """The probe verifies that the configured ``assistant_message_template`` occurs in the RENDERED
    template. When no probe conversation renders, the check does not run at all — indistinguishable
    from a pass, while the defect it exists to catch (a marker the template never emits) trains the
    run at loss ~0. It must say so out loud."""
    with caplog.at_level(logging.WARNING, logger=_FACTORY_LOGGER):
        assert _select_with_probe(_probe_tokenizer(renders=False)) is not None
    records = [r for r in caplog.records if "Could not verify assistant_message_template" in r.getMessage()]
    assert records, "an unverifiable probe must warn, not pass silently"
    assert "TypeError" in records[-1].getMessage(), "the warning must name why the probe failed"


def test_renderable_template_carrying_the_marker_stays_quiet(caplog):
    """The verified case must not warn — a warning on every good run trains operators to ignore it."""
    with caplog.at_level(logging.WARNING, logger=_FACTORY_LOGGER):
        assert _select_with_probe(_probe_tokenizer(renders=True)) is not None
    assert not [r for r in caplog.records if "Could not verify assistant_message_template" in r.getMessage()]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
