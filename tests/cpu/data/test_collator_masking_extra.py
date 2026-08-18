#!/usr/bin/env python
"""Extra collator masking + routing tests (companion to test_collators.py).

Uses the same hand-built token-id idiom: exact label-value asserts via
get_unmasked_positions / direct index comparison. Covers three correctness-critical
surfaces not pinned elsewhere:

  1. DataCollatorWithPacking DROPS the dense attention_mask on packed data (a dense
     all-ones mask reintroduces silent cross-document attention under Flash Attention),
     but keeps it for non-packed data.
  2. The completion mask binds each assistant span to the RIGHT EOS across an intervening
     TOOL_RESULT turn — both assistant spans unmasked exactly, the tool-result tokens
     masked.
  3. A response template that cannot match yields an all-(-100) batch (the loss-0 trap).

Run: pytest tests/cpu/data/test_collator_masking_extra.py
"""

import logging
from unittest.mock import MagicMock

import pytest
import torch

from src.data.collators.completions_only import DataCollatorForCompletionOnlyLM
from src.data.collators.factory import select_data_collator
from src.data.collators.packing import (
    DataCollatorForCompletionOnlyLMWithPacking,
    DataCollatorWithFlatteningAndCompletionMask,
    DataCollatorWithPacking,
)

# Token vocabulary (mirrors test_collators.py)
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
    """Packed rows must NOT carry a dense attention_mask — a present all-ones mask makes
    Flash Attention treat the whole pack as ONE sequence (cross-document attention leak)."""
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
    """Without seq_lengths the row is a single document — the dense attention_mask stays."""
    tok = make_tokenizer()
    collator = DataCollatorWithPacking(tokenizer=tok)

    ids = [USER_A, ASST_A, EOS]
    batch = collator.torch_call([{"input_ids": ids, "attention_mask": [1] * len(ids)}])

    assert "attention_mask" in batch, "non-packed batch must retain its attention_mask"
    assert batch["attention_mask"][0].tolist() == [1, 1, 1]


def test_completion_mask_binds_correct_eos_across_tool_result():
    """Two assistant turns separated by a tool-result turn.

    layout (indices):
      0 USER_A
      1 RESP_1   2 RESP_2     <- assistant turn 1 response template
      3 ASST_A   4 EOS        <- turn-1 content + its EOS
      5 TOOL_R1  6 TOOL_R2    <- tool-result turn (must stay masked)
      7 RESP_1   8 RESP_2     <- assistant turn 2 response template
      9 ASST_B  10 EOS        <- turn-2 content + its EOS

    Each assistant span must bind to ITS OWN turn's EOS (4 and 10), so both spans are
    unmasked exactly and the tool-result tokens (5,6) stay masked. A wrong EOS bind would
    unmask the tool-result span (turn 1 borrowing turn 2's EOS) or drop a turn."""
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
    # Exhaustive: nothing outside the two spans is unmasked.
    assert unmasked == {1, 2, 3, 4, 7, 8, 9, 10}, f"unexpected unmasked set: {sorted(unmasked)}"


def test_completion_marker_must_occur_in_rendered_template():
    """A marker the chat template never emits masks EVERY label — the run trains zero tokens at
    loss ~0. The factory probes the RENDERED text at construction and refuses the pairing."""
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
    """A marker the template does emit must survive the probe AND reach the collator: the factory
    threads ``assistant_message_template`` into every completions-only collator, and a marker
    dropped on the way masks every label exactly as a wrong one would."""
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
    """gpt-oss harmony emits ``<|channel|>final<|message|>`` only when the assistant message
    carries ``thinking`` — a content-only probe alone rejected the documented harmony + SFT
    pairing at collator construction. The probe must accept a marker any renderable variant emits."""
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
    """A template needing inputs the synthetic probe lacks cannot be verified — never a false raise.

    The unverified pairing still has to be built and still has to be announced: silence here reads
    exactly like a passing probe, and the defect it exists to catch surfaces only as a run at loss ~0.
    """
    marker = "anything"
    tok = make_tokenizer()
    tok.chat_template = "tmpl"

    def _boom(conversation, tokenize=False):
        raise RuntimeError("template requires tools")

    tok.apply_chat_template = _boom
    with caplog.at_level(logging.WARNING, logger="src.data.collators.factory"):
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
    """Every completion collator must REPORT the loss-0 trap, not just silently mask the row.

    An all-(-100) row returned with no diagnostic on the packed and padding-free paths turns a
    wrong assistant_message_template into a NaN/zero loss with nothing pointing at the cause.
    """
    tok = make_tokenizer(response_template_ids=[RESP_1, RESP_2])
    collator = collator_cls(response_prompt_template=RESPONSE_TEMPLATE_IDS, tokenizer=tok)
    ids = [USER_A, ASST_A, ASST_B, EOS]
    row = {"input_ids": ids, "attention_mask": [1] * len(ids), "seq_lengths": [len(ids)]}

    with pytest.warns(UserWarning, match="Could not find response key"):
        batch = collator.torch_call([row])

    assert all(v == IGNORE for v in batch["labels"].flatten().tolist())


def test_unmatchable_template_warns_on_the_padding_free_path():
    """Same fail-loud contract for the flattening (padding-free) collator."""
    tok = make_tokenizer(response_template_ids=[RESP_1, RESP_2])
    collator = DataCollatorWithFlatteningAndCompletionMask(
        response_prompt_template=RESPONSE_TEMPLATE_IDS, tokenizer=tok
    )
    ids = [USER_A, ASST_A, ASST_B, EOS]

    with pytest.warns(UserWarning, match="Could not find response key"):
        batch = collator([{"input_ids": ids}])

    assert all(v == IGNORE for v in batch["labels"].flatten().tolist())


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
