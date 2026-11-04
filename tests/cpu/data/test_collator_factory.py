#!/usr/bin/env python
import logging
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from halo.data.collators.collator_rules import reject_unsupported_collator_options
from halo.data.collators.factory import select_data_collator
from halo.data.collators.padded import DataCollatorForCausalLMWithPadding
from halo.models import segment_markers
from halo.models.attention.selection import VARLEN_ATTN_IMPLEMENTATIONS
from tests.common.models import BAILING_LING_3_TINY, BAILING_MOE_LING_MINI, BAILING_MOE_RING_MINI, QWEN3_0_6B
from tests.common.tiny_models import remote_config, unknown_hybrid_config
from tests.common.tokenizers import load_cached_tokenizer


def _make_tokenizer():
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 2
    tok.padding_side = "right"
    return tok


def _mock_model_config(model_type: str = "qwen3") -> MagicMock:
    """Real values where the factory reads them: a bare MagicMock is truthy for every family gate,
    ``layer_group_size`` included."""
    config = MagicMock()
    config.model_type = model_type
    config.layer_group_size = None
    config.get_text_config.return_value = config
    return config


def _select(**kwargs):
    return select_data_collator(tokenizer=_make_tokenizer(), **{"train_on_completions_only": False, **kwargs})


def test_packing_and_padding_free_conflict():
    try:
        _select(packing=True, padding_free=True)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "packing" in str(e).lower() and "padding_free" in str(e).lower()


def test_cp_with_padding_free_conflict():
    try:
        _select(use_context_parallel=True, padding_free=True)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "CP" in str(e) or "Context Parallelism" in str(e)


def test_completions_without_template():
    try:
        _select(train_on_completions_only=True, assistant_message_template=None)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "assistant_message_template" in str(e)


def test_last_only_without_completions():
    try:
        _select(train_on_last_assistant_only=True, train_on_completions_only=False)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "train_on_last_assistant_only" in str(e)


def test_cp_packing_rejected():
    """The Ulysses CP attention path has no per-document boundaries, so packed documents would silently attend
    across each other."""
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


def test_pad_to_multiple_of_reaches_the_completion_collator_outside_cp():
    collator = _select(
        train_on_completions_only=True, assistant_message_template="<|assistant|>", pad_to_multiple_of=8
    )
    assert type(collator).__name__ == "DataCollatorForCompletionOnlyLM"
    assert collator.pad_to_multiple_of == 8


@pytest.mark.parametrize("rows", ["packing", "padding_free"])
def test_pad_to_multiple_of_is_refused_on_rows_that_carry_no_padding(rows):
    with pytest.raises(ValueError, match="pad_to_multiple_of=8 pads a batch"):
        _select(**{rows: True}, pad_to_multiple_of=8)


def test_cp_no_packing():
    """A labels-less collator makes every CP SFT step fail with 'model did not return a loss'."""
    collator = _select(use_context_parallel=True, pad_to_multiple_of=8)
    assert type(collator).__name__ == "DataCollatorForCausalLMWithPadding"
    assert collator.pad_to_multiple_of == 8


def test_cp_completions_routing():
    collator = _select(
        use_context_parallel=True,
        train_on_completions_only=True,
        assistant_message_template="<|assistant|>",
        pad_to_multiple_of=8,
    )
    assert type(collator).__name__ == "DataCollatorForCompletionOnlyLM"
    assert collator.pad_to_multiple_of == 8


def test_cp_collator_produces_and_preserves_labels():
    """Labels built from input_ids when absent; precomputed ragged labels padded with -100."""
    tok = load_cached_tokenizer(QWEN3_0_6B)
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
    """With pad == eos the parent LM collator masks every pad-valued label, including the real turn-ending EOS, so
    the model never learns to stop. Real-token EOS labels are restored from input_ids; true padding stays masked."""
    tok = load_cached_tokenizer(QWEN3_0_6B)
    tok.pad_token = tok.eos_token  # the pad=eos fallback layout
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
    collator = _select(
        padding_free=True,
        train_on_completions_only=True,
        assistant_message_template="<|assistant|>",
    )
    assert type(collator).__name__ == "DataCollatorWithFlatteningAndCompletionMask"


def test_padding_free_no_completions():
    collator = _select(padding_free=True)
    assert type(collator).__name__ == "DataCollatorWithFlattening"


def test_packing_with_completions():
    collator = _select(
        packing=True,
        train_on_completions_only=True,
        assistant_message_template="<|assistant|>",
    )
    assert type(collator).__name__ == "DataCollatorForCompletionOnlyLMWithPacking"


def test_packing_no_completions():
    collator = _select(packing=True)
    assert type(collator).__name__ == "DataCollatorWithPacking"


def test_completions_only():
    collator = _select(
        train_on_completions_only=True,
        assistant_message_template="<|assistant|>",
    )
    assert type(collator).__name__ == "DataCollatorForCompletionOnlyLM"


def test_default_returns_none():
    collator = _select()
    assert collator is None


def test_completions_template_propagated():
    collator = _select(
        train_on_completions_only=True,
        assistant_message_template="<|MYASSISTANT|>",
    )
    assert collator.response_prompt_template == "<|MYASSISTANT|>"


def test_last_only_flag_propagated_to_completions_collator():
    collator = _select(
        train_on_completions_only=True,
        assistant_message_template="<|assistant|>",
        train_on_last_assistant_only=True,
    )
    assert type(collator).__name__ == "DataCollatorForCompletionOnlyLM"
    assert collator.train_on_last_assistant_only is True


def test_last_only_flag_propagated_to_packing_collator():
    collator = _select(
        packing=True,
        train_on_completions_only=True,
        assistant_message_template="<|assistant|>",
        train_on_last_assistant_only=True,
    )
    assert type(collator).__name__ == "DataCollatorForCompletionOnlyLMWithPacking"
    assert collator.train_on_last_assistant_only is True


def test_last_only_default_false_when_unset():
    collator = _select(
        train_on_completions_only=True,
        assistant_message_template="<|assistant|>",
    )
    assert collator.train_on_last_assistant_only is False


def test_padding_free_completions_template_propagated():
    collator = _select(
        padding_free=True,
        train_on_completions_only=True,
        assistant_message_template="<|assistant|>",
        train_on_last_assistant_only=True,
    )
    assert type(collator).__name__ == "DataCollatorWithFlatteningAndCompletionMask"
    assert collator.response_prompt_template == "<|assistant|>"
    assert collator.train_on_last_assistant_only is True


class _Cfg:
    """Stand-in for a HF config carrying only the resolved attention implementation."""

    def __init__(self, attn_impl):
        self._attn_implementation = attn_impl


@pytest.mark.parametrize("attn_impl", ["eager", "sdpa", "flex_attention", None])
def test_non_varlen_attention_rejects_padding_free(attn_impl):
    """padding_free flattens the batch into one row, so a non-varlen backend builds a dense mask whose area grows
    with the square of the batch size."""
    with pytest.raises(ValueError, match="varlen"):
        _select(padding_free=True, model_config=_Cfg(attn_impl))


@pytest.mark.parametrize("attn_impl", ["eager", "sdpa", "flex_attention", None])
def test_non_varlen_attention_allows_packing(attn_impl):
    """Transformers derives a packed_sequence_mask from reset position_ids on every backend, so eager and SDPA
    isolate documents too; the cost is a dense mask, which warrants a warning. The class is pinned, since a quiet
    downgrade to the padded collator would stop packing."""
    collator = _select(packing=True, model_config=_Cfg(attn_impl))
    assert type(collator).__name__ == "DataCollatorWithPacking"


@pytest.mark.parametrize("flag", ["padding_free", "packing"])
@pytest.mark.parametrize("attn_impl", ["flash_attention_2", "flash_attention_3", "flash_attention_4"])
def test_varlen_attention_allows_flattening(attn_impl, flag):
    """Into the collator the flag asks for, not a padded fallback that would undo the throughput win."""
    expected = {"padding_free": "DataCollatorWithFlattening", "packing": "DataCollatorWithPacking"}[flag]
    collator = _select(**{flag: True}, model_config=_Cfg(attn_impl))
    assert type(collator).__name__ == expected


def test_varlen_gate_covers_every_declared_flash_impl():
    """The accepted set is ``VARLEN_ATTN_IMPLEMENTATIONS``, with no second copy."""
    assert VARLEN_ATTN_IMPLEMENTATIONS, "the varlen constant went empty — the loop below tests nothing"
    for impl in VARLEN_ATTN_IMPLEMENTATIONS:
        collator = _select(packing=True, model_config=_Cfg(impl))
        assert type(collator).__name__ == "DataCollatorWithPacking", impl


# packing + per_device_train_batch_size > 1 builds one merged row, not B sequences.


def _warnings_for(**kwargs) -> str:
    """The main-process warnings select_data_collator emits, as one lowercased blob."""
    with patch("halo.data.collators.collator_rules.logger") as mock_logger:
        select_data_collator(tokenizer=_make_tokenizer(), **{"train_on_completions_only": False, **kwargs})
        rendered = []
        for call in mock_logger.warning.call_args_list:
            msg, *fmt_args = call.args
            rendered.append(msg % tuple(fmt_args) if fmt_args else msg)
        return " ".join(rendered).lower()


def test_packing_above_batch_one_warns_about_the_merge():
    """The merge is backend-independent (flatten_packed_batch), so on FA4 the shape change is otherwise silent."""
    config = _mock_model_config()
    config._attn_implementation = VARLEN_ATTN_IMPLEMENTATIONS[0]
    blob = _warnings_for(packing=True, per_device_train_batch_size=2, model_config=config)
    assert "does not run 2 independent sequences" in blob, f"no merge warning on a varlen backend: {blob}"


def test_packing_at_batch_one_is_silent():
    config = _mock_model_config()
    config._attn_implementation = VARLEN_ATTN_IMPLEMENTATIONS[0]
    blob = _warnings_for(packing=True, per_device_train_batch_size=1, model_config=config)
    assert "independent sequences" not in blob, f"batch 1 must not warn: {blob}"


def test_batch_above_one_without_packing_is_silent():
    config = _mock_model_config()
    config._attn_implementation = VARLEN_ATTN_IMPLEMENTATIONS[0]
    blob = _warnings_for(packing=False, per_device_train_batch_size=8, model_config=config)
    assert "independent sequences" not in blob, f"unpacked batches must not warn: {blob}"


def test_pipeline_parallelism_is_not_warned():
    """PP keeps the rows and splits them into microbatches, so batch>1 there is B independent rows."""
    config = _mock_model_config()
    config._attn_implementation = VARLEN_ATTN_IMPLEMENTATIONS[0]
    blob = _warnings_for(packing=True, per_device_train_batch_size=4, keeps_packed_rows=True, model_config=config)
    assert "independent sequences" not in blob, f"PP must not be warned about a merge it never does: {blob}"


_MARKER_LOGGER = "halo.data.turn_markers"


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
    return select_data_collator(
        tokenizer=tokenizer,
        train_on_completions_only=True,
        assistant_message_template="<|assistant|>",
    )


def test_unrenderable_chat_template_probe_warns(caplog):
    """When no probe conversation renders, the check does not run, indistinguishable from a pass, while a marker the
    template never emits trains at loss ~0."""
    with caplog.at_level(logging.WARNING, logger=_MARKER_LOGGER):
        assert _select_with_probe(_probe_tokenizer(renders=False)) is not None
    records = [r for r in caplog.records if "Could not verify assistant_message_template" in r.getMessage()]
    assert records, "an unverifiable probe must warn, not pass silently"
    assert "TypeError" in records[-1].getMessage(), "the warning must name why the probe failed"


def test_renderable_template_carrying_the_marker_stays_quiet(caplog):
    with caplog.at_level(logging.WARNING, logger=_MARKER_LOGGER):
        assert _select_with_probe(_probe_tokenizer(renders=True)) is not None
    assert not [r for r in caplog.records if "Could not verify assistant_message_template" in r.getMessage()]


# Linear-attention layers no segment marker reaches cross documents on every backend; flash attention
# isolating the softmax layers must not make that silent.


_BOTH_FLAGS = ("packing", "padding_free")
# name -> (config builder, whether its rows cross, the flags it is checked under)
_LINEAR_ATTENTION_CONFIGS = {
    "ring_lightning_attention": (lambda: remote_config(BAILING_MOE_RING_MINI), True, _BOTH_FLAGS),
    "ling3_kda": (lambda: remote_config(BAILING_LING_3_TINY), True, _BOTH_FLAGS),
    "glm5_next_kda": (lambda: CONFIG_MAPPING["glm5_next_text"](), True, _BOTH_FLAGS),
    "unknown_family_linear_attention": (unknown_hybrid_config, True, _BOTH_FLAGS),
    "ling2_softmax_only": (lambda: remote_config(BAILING_MOE_LING_MINI), False, _BOTH_FLAGS),
    # Each group's last layer is softmax attention, so groups of one hold no Lightning Attention-2 layer.
    "ring_groups_of_one_softmax_only": (
        lambda: remote_config(BAILING_MOE_RING_MINI, layer_group_size=1),
        False,
        _BOTH_FLAGS,
    ),
    "qwen3_next_gated_deltanet_marked": (lambda: CONFIG_MAPPING["qwen3_next"](), False, _BOTH_FLAGS),
    # qwen3_next carries the GatedDeltaNet exemption under padding_free.
    "qwen3_5_gated_deltanet_marked": (lambda: CONFIG_MAPPING["qwen3_5_text"](), False, ("packing",)),
    "qwen3_softmax_only": (lambda: CONFIG_MAPPING["qwen3"](), False, _BOTH_FLAGS),
}
_LINEAR_ATTENTION_CASES = [
    (name, flag) for name, (_, _, flags) in sorted(_LINEAR_ATTENTION_CONFIGS.items()) for flag in flags
]
_CROSSING_ALLOWED_MESSAGE = "under allow_cross_document_linear_attention"


@pytest.mark.parametrize(
    ("name", "flag"), _LINEAR_ATTENTION_CASES, ids=[f"{name}-{flag}" for name, flag in _LINEAR_ATTENTION_CASES]
)
def test_unmarked_linear_attention_is_refused_on_flash(caplog, monkeypatch, name, flag):
    """On a flash label the softmax layers isolate, so Ring's Lightning Attention-2 or a KDA layer would carry state
    across documents silently. GatedDeltaNet reads the cu_seq_lens markers and softmax-only models have no such
    layer, so they build without the opt-in."""
    # GatedDeltaNet's segment-aware kernels need CUDA; taken as present, as on the images.
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", ())
    monkeypatch.setattr(segment_markers, "_crossing_allowed_warned", set())
    build, crosses, _ = _LINEAR_ATTENTION_CONFIGS[name]
    config = build()
    config._attn_implementation = "flash_attention_4"
    if crosses:
        with pytest.raises(ValueError, match=f"{flag}=True is refused .*allow_cross_document_linear_attention"):
            _select(**{flag: True}, model_config=config)
    else:
        assert _select(**{flag: True}, model_config=config) is not None
    with caplog.at_level(logging.WARNING, logger=segment_markers.__name__):
        assert _select(**{flag: True}, model_config=config, allow_cross_document_linear_attention=True) is not None
    warned = any(_CROSSING_ALLOWED_MESSAGE in record.getMessage() for record in caplog.records)
    assert warned is crosses, f"{name}: warned={warned} under the opt-in, expected {crosses}"


def test_a_caller_that_names_no_opt_in_is_refused():
    """The rule's opt-in defaults off, so a direct caller that never names it refuses the crossing row."""
    config = CONFIG_MAPPING["glm5_next_text"]()
    config._attn_implementation = "flash_attention_4"
    with pytest.raises(ValueError, match="packing=True is refused .*allow_cross_document_linear_attention"):
        reject_unsupported_collator_options(
            _make_tokenizer(),
            padding_free=False,
            packing=True,
            train_on_completions_only=False,
            assistant_message_template=None,
            use_context_parallel=False,
            train_on_last_assistant_only=False,
            model_config=config,
            per_device_train_batch_size=1,
            keeps_packed_rows=False,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
