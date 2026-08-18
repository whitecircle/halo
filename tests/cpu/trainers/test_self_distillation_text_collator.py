#!/usr/bin/env python
"""Tests for the text SDPG self-distillation collator (src/data/collators/self_distill.py).

The OPD loss requires the student and teacher branches to select the SAME response tokens (row
alignment), with the teacher sequence longer because it carries the privileged hint. These tests
fail if either invariant breaks. A real tokenizer is needed for chat templating; the test skips
when one is not cached (CI without model access) rather than passing vacuously.

Run: pytest tests/cpu/trainers/test_self_distillation_text_collator.py
"""

import pytest
import torch

from src.data.collators.self_distill import (
    SelfDistillTextCollator,
    inject_privileged_hint,
    mean_normalized_confidence_weights,
)
from tests.common.models import QWEN3_0_6B
from tests.common.tokenizers import load_cached_tokenizer


def test_inject_hint_appends_to_last_user_turn():
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]
    out = inject_privileged_hint(history, " HINT={answer}", answer="42", solution=None)
    assert out[2]["content"] == "second HINT=42"
    assert out[0]["content"] == "first"  # earlier user turn untouched
    assert history[2]["content"] == "second"  # original not mutated


def test_confidence_weights_mean_normalized():
    examples = [{"c": 1.0}, {"c": 0.0}, {"c": 0.5}]
    w = mean_normalized_confidence_weights(examples, "c", power=2.0)
    # mean is normalized to 1.0 and the zero-confidence sample is gated to 0.
    assert torch.isclose(w.mean(), torch.tensor(1.0), atol=1e-5)
    assert float(w[1]) == 0.0
    assert float(w[0]) > float(w[2])  # higher confidence -> higher weight


def _tokenizer():
    return load_cached_tokenizer(QWEN3_0_6B)


def test_student_and_teacher_response_tokens_align():
    tok = _tokenizer()
    col = SelfDistillTextCollator(
        tok,
        max_length=128,
        conversation_field="messages",
        hint_template="\n[Hint] answer is {answer}\n",
        answer_field="answer",
        confidence_field="confidence",
        confidence_power=2.0,
        response_prompt_template="<|im_start|>assistant\n",
        train_on_completions_only=True,
    )
    examples = [
        {
            "messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}],
            "answer": "4",
            "confidence": 0.9,
        },
        {
            "messages": [{"role": "user", "content": "cap of France?"}, {"role": "assistant", "content": "Paris"}],
            "answer": "Paris",
            "confidence": 0.5,
        },
    ]
    batch = col(examples)

    assert set(batch) >= {
        "input_ids",
        "attention_mask",
        "labels",
        "teacher_input_ids",
        "teacher_attention_mask",
        "teacher_labels",
        "confidence_weights",
    }
    # Teacher carries the hint -> at least as long as the student.
    assert batch["teacher_input_ids"].shape[1] >= batch["input_ids"].shape[1]

    # The supervised (response) tokens must be byte-identical per sample across the two branches,
    # otherwise the OPD reverse-KL aligns mismatched positions.
    for b in range(len(examples)):
        student_resp = batch["input_ids"][b][batch["labels"][b] != -100]
        teacher_resp = batch["teacher_input_ids"][b][batch["teacher_labels"][b] != -100]
        assert student_resp.numel() > 0
        assert torch.equal(student_resp, teacher_resp)


def test_over_length_raises_instead_of_truncating():
    """Over-length must raise, naming the offending branch, never truncate: tokenizing student and
    teacher with truncation=True in separate calls costs the teacher (longer by the privileged hint)
    trailing RESPONSE tokens the student keeps, silently breaking the byte-identical response
    invariant OPD row alignment relies on."""
    tok = _tokenizer()
    col = SelfDistillTextCollator(
        tok,
        max_length=8,  # far below any templated conversation
        conversation_field="messages",
        hint_template=" answer={answer}",
        answer_field="answer",
    )
    examples = [
        {
            "messages": [
                {"role": "user", "content": "a fairly long question to overflow the tiny budget"},
                {"role": "assistant", "content": "a fairly long answer as well"},
            ],
            "answer": "42",
        }
    ]
    with pytest.raises(ValueError, match="Self-distillation .* branch"):
        col(examples)


def test_audit_row_applies_the_collate_time_contract_per_row():
    """audit_row is the prep-time twin of the collate raise: same contract, same message, one raw
    row — mapped over the dataset where the coordinated machinery makes the failure world-uniform
    (at collate time the raise is rank-local and peers hang to the NCCL watchdog)."""
    tok = _tokenizer()
    col = SelfDistillTextCollator(
        tok,
        max_length=8,
        conversation_field="messages",
        hint_template=" answer={answer}",
        answer_field="answer",
    )
    long_row = {
        "messages": [
            {"role": "user", "content": "a fairly long question to overflow the tiny budget"},
            {"role": "assistant", "content": "a fairly long answer as well"},
        ],
        "answer": "42",
    }
    with pytest.raises(ValueError, match="Self-distillation .* branch"):
        col.audit_row(long_row)

    roomy = SelfDistillTextCollator(
        tok,
        max_length=4096,
        conversation_field="messages",
        hint_template=" answer={answer}",
        answer_field="answer",
    )
    assert roomy.audit_row(long_row) == {}, "a within-budget row must pass and add no columns"


def test_teacher_overflow_names_teacher_branch():
    """When only the teacher overflows (student fits, hint pushes the teacher over), the error names
    the teacher branch so the user knows to budget max_length for the hint."""
    tok = _tokenizer()
    examples = [
        {
            "messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}],
            "answer": "4",
        }
    ]
    # Find the exact student length, then give the collator a budget that fits the student only.
    probe = SelfDistillTextCollator(
        tok,
        max_length=10_000,
        conversation_field="messages",
        hint_template=" here is a sufficiently long privileged hint: answer={answer}",
        answer_field="answer",
    )
    batch = probe(examples)
    student_len = int(batch["attention_mask"].sum())

    col = SelfDistillTextCollator(
        tok,
        max_length=student_len,
        conversation_field="messages",
        hint_template=" here is a sufficiently long privileged hint: answer={answer}",
        answer_field="answer",
    )
    with pytest.raises(ValueError, match="teacher branch"):
        col(examples)


def test_completion_only_masks_prompt():
    tok = _tokenizer()
    col = SelfDistillTextCollator(
        tok,
        max_length=128,
        conversation_field="messages",
        hint_template=" answer={answer}",
        answer_field="answer",
        response_prompt_template="<|im_start|>assistant\n",
        train_on_completions_only=True,
    )
    examples = [
        {
            "messages": [
                {"role": "user", "content": "long user prompt here"},
                {"role": "assistant", "content": "short"},
            ],
            "answer": "short",
        }
    ]
    batch = col(examples)
    labels = batch["labels"][0]
    # Some tokens supervised, but strictly fewer than the full sequence (prompt is masked).
    n_sup = int((labels != -100).sum())
    assert 0 < n_sup < labels.numel()


def _render(tok, **collator_kwargs) -> str:
    """Decode the student branch of a one-row batch — what the model actually sees."""
    col = SelfDistillTextCollator(
        tok,
        max_length=256,
        conversation_field="messages",
        hint_template=" answer={answer}",
        answer_field="answer",
        **collator_kwargs,
    )
    row = {
        "messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}],
        "answer": "4",
        "tools": [
            {
                "type": "function",
                "function": {"name": "add", "description": "adds", "parameters": {"type": "object", "properties": {}}},
            }
        ],
    }
    batch = col([row])
    return tok.decode(batch["input_ids"][0], skip_special_tokens=False)


def test_system_prompt_reaches_the_rendered_conversation():
    """The knob parses on the self-distillation script (SFTScriptArguments); rendering without it
    would train the model on a prompt shape it is never served with."""
    tok = _tokenizer()
    assert "BE TERSE" not in _render(tok)
    assert "BE TERSE" in _render(tok, system_prompt="BE TERSE")


def test_system_prompt_demoted_to_user_when_the_model_has_no_system_role():
    tok = _tokenizer()
    rendered = _render(tok, system_prompt="BE TERSE", model_supports_system_role=False)
    assert "BE TERSE" in rendered
    # The system turn is folded into the first user turn instead of emitted as its own system block.
    assert rendered.index("BE TERSE") > rendered.index("<|im_start|>user")


def test_tools_field_reaches_the_chat_template():
    tok = _tokenizer()
    assert '"add"' not in _render(tok)
    assert '"add"' in _render(tok, tools_field="tools")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
