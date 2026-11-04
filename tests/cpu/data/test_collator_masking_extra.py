#!/usr/bin/env python
"""Extra collator masking and routing tests, using the hand-built token-id idiom of test_collators.py."""

import logging
from unittest.mock import MagicMock

import pytest
import torch

from halo.data.collators.factory import select_data_collator
from halo.data.collators.packing import (
    DataCollatorForCompletionOnlyLMWithPacking,
    DataCollatorWithFlatteningAndCompletionMask,
    DataCollatorWithPacking,
)
from halo.data.collators.padded import DataCollatorForCompletionOnlyLM

PAD = 0
EOS = 1
RESP_1, RESP_2 = 10, 11
RESPONSE_TEMPLATE_IDS = [RESP_1, RESP_2]
USER_A, USER_B = 20, 21
ASST_A, ASST_B, ASST_C, ASST_D = 30, 31, 32, 33
TOOL_R1, TOOL_R2 = 40, 41  # tool-result content tokens
IGNORE = -100


def make_tokenizer(pad_token_id: int = PAD, eos_token_id: int = EOS, response_template_ids=None) -> MagicMock:
    tok = MagicMock()
    tok.pad_token_id = pad_token_id
    tok.eos_token_id = eos_token_id
    tok.padding_side = "right"
    tok.model_max_length = 4096
    tok.encode.return_value = response_template_ids if response_template_ids is not None else RESPONSE_TEMPLATE_IDS

    def _pad(features, **kwargs):
        max_len = max(len(f["input_ids"]) for f in features)
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            ids = list(f["input_ids"])
            labs = list(f.get("labels", ids[:]))
            mask = list(f.get("attention_mask", [1] * len(ids)))
            pad_len = max_len - len(ids)
            out["input_ids"].append(ids + [pad_token_id] * pad_len)
            out["labels"].append(labs + [IGNORE] * pad_len)
            out["attention_mask"].append(mask + [0] * pad_len)
        return {k: torch.tensor(v) for k, v in out.items()}

    tok.pad = _pad
    return tok


def get_unmasked_positions(labels):
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    return sorted(i for i, v in enumerate(labels) if v != IGNORE)


def test_packing_drops_attention_mask_when_packed():
    """A present all-ones mask makes Flash Attention treat the whole pack as one sequence."""
    tok = make_tokenizer()
    collator = DataCollatorWithPacking(tokenizer=tok)

    conv1 = [USER_A, ASST_A, EOS]
    conv2 = [USER_B, ASST_B, EOS]
    examples = [
        {
            "input_ids": conv1 + conv2,
            "attention_mask": [1] * (len(conv1) + len(conv2)),
            "seq_lengths": [len(conv1), len(conv2)],
        }
    ]
    batch = collator.torch_call(examples)

    assert "attention_mask" not in batch, (
        "packed batch must not carry a dense attention_mask (FA derives cu_seqlens from "
        f"position_ids); got keys {list(batch.keys())}"
    )
    # position_ids must be present so FA can rebuild the block-diagonal mask.
    assert "position_ids" in batch, "packed batch lost the resetting position_ids"


def test_packing_keeps_attention_mask_when_not_packed():
    tok = make_tokenizer()
    collator = DataCollatorWithPacking(tokenizer=tok)

    ids = [USER_A, ASST_A, EOS]
    batch = collator.torch_call([{"input_ids": ids, "attention_mask": [1] * len(ids)}])

    assert "attention_mask" in batch, "non-packed batch must retain its attention_mask"
    assert batch["attention_mask"][0].tolist() == [1, 1, 1]


def test_completion_mask_binds_correct_eos_across_tool_result():
    """Two assistant turns separated by a tool-result turn: 0 USER_A, 1-2 template, 3 ASST_A, 4 EOS, 5-6 tool
    result, 7-8 template, 9 ASST_B, 10 EOS. A wrong EOS bind would unmask the tool-result span or drop a turn."""
    tok = make_tokenizer()
    collator = DataCollatorForCompletionOnlyLM(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=False,
    )

    ids = [
        USER_A,
        RESP_1,
        RESP_2,
        ASST_A,
        EOS,
        TOOL_R1,
        TOOL_R2,
        RESP_1,
        RESP_2,
        ASST_B,
        EOS,
    ]
    batch = collator.torch_call([{"input_ids": ids, "attention_mask": [1] * len(ids)}])
    unmasked = set(get_unmasked_positions(batch["labels"][0]))

    assert {1, 2, 3, 4} <= unmasked, f"turn-1 span not fully unmasked: {sorted(unmasked)}"
    assert 5 not in unmasked and 6 not in unmasked, (
        f"tool-result tokens leaked into the loss (wrong EOS bind): {sorted(unmasked)}"
    )
    assert {7, 8, 9, 10} <= unmasked, f"turn-2 span not fully unmasked: {sorted(unmasked)}"
    assert 0 not in unmasked, f"user token leaked into the loss: {sorted(unmasked)}"
    assert unmasked == {1, 2, 3, 4, 7, 8, 9, 10}, f"unexpected unmasked set: {sorted(unmasked)}"


def test_completion_marker_must_occur_in_rendered_template():
    """A marker the template never emits masks every label and trains zero tokens at loss ~0; the factory probes the
    rendered text at construction."""
    tok = make_tokenizer()
    tok.chat_template = "tmpl"
    tok.apply_chat_template = lambda conversation, tokenize=False: "<|im_start|>assistant\na"
    with pytest.raises(ValueError, match="does not occur"):
        select_data_collator(
            tok,
            train_on_completions_only=True,
            assistant_message_template="<|start_header_id|>assistant",
        )


def test_completion_marker_probe_accepts_matching_template():
    """The marker must also reach the collator: one dropped on the way masks every label as a wrong one would."""
    marker = "<|im_start|>assistant"
    tok = make_tokenizer()
    tok.chat_template = "tmpl"
    tok.apply_chat_template = lambda conversation, tokenize=False: f"{marker}\na"
    collator = select_data_collator(
        tok,
        train_on_completions_only=True,
        assistant_message_template=marker,
    )
    assert isinstance(collator, DataCollatorForCompletionOnlyLM)
    assert collator.response_prompt_template == marker


def test_completion_marker_probe_accepts_thinking_only_marker():
    """gpt-oss harmony emits ``<|channel|>final<|message|>`` only when the message carries ``thinking``, so the
    probe accepts a marker any renderable variant emits."""
    marker = "<|channel|>final<|message|>"
    tok = make_tokenizer()
    tok.chat_template = "tmpl"

    def _harmony_like(conversation, tokenize=False):
        out = []
        for message in conversation:
            if message["role"] == "assistant" and message.get("thinking"):
                out.append(f"<|channel|>analysis<|message|>{message['thinking']}")
                out.append(f"{marker}{message['content']}")
            else:
                out.append(f"<|start|>{message['role']}<|message|>{message['content']}")
        return "".join(out)

    tok.apply_chat_template = _harmony_like
    collator = select_data_collator(
        tok,
        train_on_completions_only=True,
        assistant_message_template=marker,
    )
    assert isinstance(collator, DataCollatorForCompletionOnlyLM)
    assert collator.response_prompt_template == marker


def test_completion_marker_probe_skips_unrenderable_template(caplog):
    """A template needing inputs the probe lacks cannot be verified: never a false raise, and the unverified pairing
    is still built and announced."""
    marker = "anything"
    tok = make_tokenizer()
    tok.chat_template = "tmpl"

    def _boom(conversation, tokenize=False):
        raise RuntimeError("template requires tools")

    tok.apply_chat_template = _boom
    with caplog.at_level(logging.WARNING, logger="halo.data.turn_markers"):
        collator = select_data_collator(
            tok,
            train_on_completions_only=True,
            assistant_message_template=marker,
        )
    assert isinstance(collator, DataCollatorForCompletionOnlyLM)
    assert collator.response_prompt_template == marker
    assert "UNCHECKED" in caplog.text, caplog.text
    assert "template requires tools" in caplog.text, "the probe failure's reason must be reported"


@pytest.mark.parametrize("collator_cls", [DataCollatorForCompletionOnlyLMWithPacking, DataCollatorForCompletionOnlyLM])
def test_unmatchable_template_warns_on_the_packed_and_padded_paths(collator_cls):
    """An all-(-100) row with no diagnostic turns a wrong assistant_message_template into a NaN or zero loss with
    nothing pointing at the cause."""
    tok = make_tokenizer(response_template_ids=[RESP_1, RESP_2])
    collator = collator_cls(response_prompt_template=RESPONSE_TEMPLATE_IDS, tokenizer=tok)
    ids = [USER_A, ASST_A, ASST_B, EOS]
    row = {"input_ids": ids, "attention_mask": [1] * len(ids), "seq_lengths": [len(ids)]}

    with pytest.warns(UserWarning, match="Could not find response key"):
        batch = collator.torch_call([row])

    assert all(v == IGNORE for v in batch["labels"].flatten().tolist())


def test_unmatchable_template_warns_on_the_padding_free_path():
    tok = make_tokenizer(response_template_ids=[RESP_1, RESP_2])
    collator = DataCollatorWithFlatteningAndCompletionMask(
        response_prompt_template=RESPONSE_TEMPLATE_IDS, tokenizer=tok
    )
    ids = [USER_A, ASST_A, ASST_B, EOS]

    with pytest.warns(UserWarning, match="Could not find response key"):
        batch = collator([{"input_ids": ids}])

    assert all(v == IGNORE for v in batch["labels"].flatten().tolist())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
