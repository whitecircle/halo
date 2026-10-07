#!/usr/bin/env python
"""The SDPG self-distillation collators (src/data/collators/self_distill.py and the VLM sibling).

OPD pairs the k-th supervised token of the student branch with the k-th of the hinted teacher
branch, so both must supervise the same tokens in the same order while the teacher sequence is
longer by the hint; a row breaking that raises at prep, world-uniform, rather than distilling across
shifted tokens. Confidence weights divide by ONE dataset normalizer, so they weight rows at any
batch size. A real tokenizer is needed for chat templating; tests skip when one is not cached.

Run: pytest tests/cpu/data/test_self_distill_collator.py
"""

import types
from functools import partial

import pytest
import torch
from accelerate import PartialState
from datasets import Dataset, DatasetDict

from src.data.collators.self_distill import (
    SelfDistillTextCollator,
    confidence_normalizer,
    inject_privileged_hint,
    privileged_hint,
    teacher_history,
)
from src.data.collators.vlm import SelfDistillVLMDataCollator
from src.data.pipeline.vlm_dataset import _filter_vlm_over_length
from tests.common.models import QWEN3_0_6B
from tests.common.tokenizers import load_cached_tokenizer
from tests.common.utils import load_script_module

MARKER = "<|im_start|>assistant\n"

PartialState()  # the VLM filter logs through accelerate's logger, which requires an initialized state


def _tokenizer():
    return load_cached_tokenizer(QWEN3_0_6B)


def _collator(tokenizer, **overrides):
    kwargs = {
        "max_length": 256,
        "conversation_field": "messages",
        "hint_template": "\n[Hint] answer is {answer}\n",
        "response_prompt_template": MARKER,
        "train_on_completions_only": True,
    }
    return SelfDistillTextCollator(tokenizer, **{**kwargs, **overrides})


def _row(question="2+2?", answer="4", **extra):
    return {
        "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
        "answer": answer,
        **extra,
    }


def test_the_hint_lands_on_the_last_user_turn_of_a_copy():
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "second"}]},
    ]
    out = inject_privileged_hint(history, privileged_hint(" HINT={answer}", answer="42"))
    assert out[0]["content"] == "first"
    assert out[2]["content"][-1] == {"type": "text", "text": " HINT=42"}
    assert history[2]["content"][-1] == {"type": "text", "text": "second"}, "the input row must not be mutated"


@pytest.mark.parametrize("answer", [None, "", "  "])
def test_a_blank_answer_renders_no_hint_rather_than_an_empty_one(answer):
    """The template states the answer as fact: "The correct answer is: ." would mislead the teacher."""
    assert privileged_hint("The correct answer is: {answer}.", answer=answer) is None
    history = [{"role": "user", "content": "Q"}]
    assert (
        teacher_history(
            history, {"answer": answer}, hint_template="{answer}", answer_field="answer", solution_field=None
        )
        == history
    )


def test_only_the_slots_a_template_names_must_be_filled():
    """A solution-only template needs no answer; a blank named solution leaves no hint, like a blank answer."""
    assert privileged_hint("{answer}|{solution}", answer="42", solution="6 * 7") == "42|6 * 7"
    assert privileged_hint("worked: {solution}", answer=None, solution="6 * 7") == "worked: 6 * 7"
    assert privileged_hint("{answer}|{solution}", answer="42", solution=None) is None
    assert privileged_hint("no slots", answer=None) == "no slots"


def test_the_vlm_collator_hints_the_teacher_history_the_same_way():
    tokenizer = types.SimpleNamespace(eos_token_id=2, pad_token_id=0, get_vocab=dict)
    collator = SelfDistillVLMDataCollator(None, tokenizer, hint_template="\n[Hint] {answer}\n")
    history = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]
    assert collator._teacher_history(history, {"answer": "B"})[2]["content"] == "Q2\n[Hint] B\n"


def test_student_and_teacher_supervise_the_same_response_tokens():
    tokenizer = _tokenizer()
    batch = _collator(tokenizer)([_row(), _row("capital of France?", "Paris")])
    assert batch["teacher_input_ids"].shape[1] > batch["input_ids"].shape[1]
    for row in range(2):
        student = batch["input_ids"][row][batch["labels"][row] != -100]
        teacher = batch["teacher_input_ids"][row][batch["teacher_labels"][row] != -100]
        assert student.numel() > 0
        assert torch.equal(student, teacher)


def test_supervising_the_prompt_is_refused_at_the_audit():
    """With the prompt in the labels the hint shifts every pair after it onto another token."""
    tokenizer = _tokenizer()
    collator = _collator(tokenizer, train_on_completions_only=False, response_prompt_template=None)
    with pytest.raises(ValueError, match="not the same sequence"):
        collator.audit_row(_row())


def test_the_audit_writes_no_copy_of_its_input(tmp_path, monkeypatch):
    """The audit returns no columns, so its cache artifact must carry none: a map writes every column
    it leaves in place, so a VLM split's image bytes would be copied per node on a non-shared FS."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path))
    script = load_script_module("scripts/training/distillation/self_distill.py", "halo_test_self_distill_audit")
    produced = []
    real_map = script.coordinated_map
    monkeypatch.setattr(script, "coordinated_map", lambda *args, **kwargs: produced.append(real_map(*args, **kwargs)))
    rows = [_row(payload="x" * 4096), _row("capital of France?", "Paris", payload="y" * 4096)]
    split = Dataset.from_list(rows)

    script._audit_rows(_collator(_tokenizer()), split, None, 1)

    (audited,) = produced
    (artifact,) = audited.cache_files
    assert Dataset.from_file(artifact["filename"]).column_names == []


def test_no_teacher_branch_is_built_with_the_opd_term_off():
    tokenizer = _tokenizer()
    collator = _collator(tokenizer, hint_template=None, train_on_completions_only=False, response_prompt_template=None)
    assert not collator.builds_teacher_branch
    assert not any(key.startswith("teacher_") for key in collator([_row()]))


@pytest.mark.parametrize("max_length", [8, None], ids=["student-over", "teacher-over"])
def test_an_over_length_branch_raises_by_name_instead_of_truncating(max_length):
    """Truncating one branch would cut response tokens the other keeps; the raise names the branch so
    the budget is raised for the hint, and the audit raises it at prep."""
    tokenizer = _tokenizer()
    hint = " here is a sufficiently long privileged hint: answer={answer}"
    if max_length is None:
        max_length = int(_collator(tokenizer, hint_template=hint)([_row()])["attention_mask"].sum())
    collator = _collator(tokenizer, hint_template=hint, max_length=max_length)
    branch = "student" if max_length == 8 else "teacher"
    with pytest.raises(ValueError, match=f"{branch} branch"):
        collator.audit_row(_row())
    assert _collator(tokenizer, hint_template=hint, max_length=4096).audit_row(_row()) == {}


def _confidence_split(values):
    return Dataset.from_dict({"confidence": values})


def test_confidence_weights_divide_by_the_dataset_mean_at_any_batch_size():
    """A per-batch mean is 1.0 at batch size 1 — the weighting every shipped recipe would run as a no-op."""
    tokenizer = _tokenizer()
    normalizer = confidence_normalizer(_confidence_split([1.0, 0.5, 0.0]), None, "confidence", 2.0)
    assert normalizer == pytest.approx((1.0 + 0.25) / 3)
    collator = _collator(
        tokenizer, confidence_field="confidence", confidence_power=2.0, confidence_normalizer=normalizer
    )
    weight = collator([_row(confidence=0.5)])["confidence_weights"]
    assert weight.item() == pytest.approx(0.25 / normalizer)


@pytest.mark.parametrize("values", [[0.5, 1.5], [0.5, None], [-0.1, 0.5]], ids=["above", "missing", "below"])
def test_a_confidence_outside_the_unit_interval_is_refused(values):
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        confidence_normalizer(_confidence_split([0.5, 0.5]), _confidence_split(values), "confidence", 2.0)


def test_an_all_zero_confidence_split_is_refused():
    with pytest.raises(ValueError, match="is 0"):
        confidence_normalizer(_confidence_split([0.0, 0.0]), None, "confidence", 2.0)


def test_a_confidence_column_without_its_normalizer_is_refused_at_construction():
    with pytest.raises(ValueError, match="needs confidence_normalizer"):
        _collator(_tokenizer(), confidence_field="confidence")


def test_a_marker_the_chat_template_never_renders_is_refused_at_construction():
    """It would match no row, so completion-only masking would train zero tokens at loss ~0."""
    with pytest.raises(ValueError, match="does not occur in this tokenizer's rendered chat template"):
        _collator(_tokenizer(), response_prompt_template="<|start|>assistant<|channel|>final<|message|>")


class _JoinProcessor:
    def apply_chat_template(self, history, tokenize=False, add_generation_prompt=False, **_):
        return " ".join(str(message["content"]) for message in history)


class _WordTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class _QwenTemplateProcessor:
    """The Qwen3 tokenizer's own chat template behind the processor render seam."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, history, tokenize=False, add_generation_prompt=False, **kwargs):
        return self.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=False, **kwargs)


def _vlm_collator(tokenizer, **overrides):
    kwargs = {"response_prompt_template": MARKER, "train_on_completions_only": True, "hint_template": " hint {answer}"}
    return SelfDistillVLMDataCollator(_QwenTemplateProcessor(tokenizer), tokenizer, 256, **{**kwargs, **overrides})


def _vlm_row():
    history = [{"role": "user", "content": "What digit?"}, {"role": "assistant", "content": "Seven."}]
    return {"history": history, "images": [], "tools_json": None, "answer": "7"}


def test_the_vlm_audit_refuses_a_misaligned_row_at_prep():
    """The VLM path's alignment refusal runs at prep too, where it is world-uniform."""
    tokenizer = _tokenizer()
    assert _vlm_collator(tokenizer).audit_row(_vlm_row()) == {}
    misaligned = _vlm_collator(tokenizer, train_on_completions_only=False, response_prompt_template=None)
    with pytest.raises(ValueError, match="not the same sequence"):
        misaligned.audit_row(_vlm_row())
    assert _vlm_collator(tokenizer, hint_template=None, train_on_completions_only=False).audit_row(_vlm_row()) == {}


class _PlainVlmProcessor:
    """Renders a VLM history (string or list-of-parts content) as ``role: text`` lines."""

    def apply_chat_template(self, history, tokenize=False, add_generation_prompt=False, **_):
        def text(content):
            return content if isinstance(content, str) else "".join(part.get("text") or "" for part in content)

        return "\n".join(f"{message['role']}: {text(message['content'])}" for message in history)


def _vlm_script_args(**overrides):
    return types.SimpleNamespace(
        **{
            "dataset": "dummy/dataset",
            "conversation_field": "messages",
            "system_prompt": None,
            "model_supports_system_role": True,
            "tools_field": None,
            "images_field": None,
            "interleaved_thinking": False,
            "assistant_message_template": None,
            "train_on_completions_only": False,
            "sdpg_answer_field": "answer",
            "privileged_solution_field": None,
            "confidence_field": None,
            "confidence_power": 4.0,
            **overrides,
        }
    )


@pytest.mark.parametrize("hint_template", [" hint {answer}", None], ids=["opd-on", "opd-off"])
def test_the_script_runs_the_vlm_audit_at_prep(hint_template, tmp_path, monkeypatch):
    """The VLM builder must hand every row to the collator's audit before training: here the labels
    reach into the prompt, so the hinted branch supervises other tokens, and the build itself raises.
    With the OPD term off no teacher branch exists, so the same rows build."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path))
    script = load_script_module("scripts/training/distillation/self_distill.py", "halo_test_self_distill_vlm_audit")
    ds = DatasetDict({"train": Dataset.from_list([_row("What digit?", "7")])})
    build = partial(
        script._build_vlm_dataset_and_collator,
        ds,
        _vlm_script_args(),
        _PlainVlmProcessor(),
        _tokenizer(),
        256,
        1,
        None,
    )
    if hint_template is None:
        build(hint_template)
        return
    with pytest.raises(ValueError, match="not the same sequence"):
        build(hint_template)


def test_the_vlm_length_filter_measures_the_hinted_teacher_branch():
    """The teacher branch is the longer one; a row only its hint pushes over would otherwise raise in
    one rank's collator and hang the peers."""
    rows = {
        "history": [[{"role": "user", "content": "q q"}, {"role": "assistant", "content": "a"}]] * 2,
        "tools_json": [None, None],
        "answer": ["x", "x y z w"],
    }
    ds = DatasetDict({"train": Dataset.from_dict(rows)})
    plain = _filter_vlm_over_length(ds, _JoinProcessor(), _WordTokenizer(), 6, 1, None)
    assert len(plain["train"]) == 2
    measured = partial(teacher_history, hint_template=" hint {answer}", answer_field="answer", solution_field=None)
    hinted = _filter_vlm_over_length(ds, _JoinProcessor(), _WordTokenizer(), 6, 1, measured)
    assert hinted["train"]["answer"] == ["x"]


def _render(tok, **collator_kwargs) -> str:
    """Decode the student branch of a one-row batch — what the model actually sees."""
    row = _row(
        tools=[
            {
                "type": "function",
                "function": {"name": "add", "description": "adds", "parameters": {"type": "object", "properties": {}}},
            }
        ]
    )
    batch = _collator(tok, **collator_kwargs)([row])
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
    raise SystemExit(pytest.main([__file__, "-v"]))
