#!/usr/bin/env python
"""The SDPG self-distillation collators. OPD pairs the k-th supervised token of student and teacher branches, so a
row breaking that raises at prep, world-uniform. The text path encodes each row once at prep, and confidence
weights divide by one dataset normalizer. Tests skip when a real tokenizer is not cached."""

import types
from functools import partial

import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict
from PIL import Image

import halo.data.collators.self_distill as self_distill_collator
from halo.data.chat_render import render_conversation, tokenize_rendered
from halo.data.collators.self_distill import (
    SelfDistillTextCollator,
    SelfDistillVLMDataCollator,
    confidence_normalizer,
    inject_privileged_hint,
    privileged_hint,
    teacher_history,
)
from halo.data.completion_spans import build_training_labels
from halo.data.pipeline.vlm_rows import _filter_vlm_over_length
from halo.data.vlm_render import extract_conversation_images
from tests.common.effort_templates import LEVEL_LINES, QWEN3_6_EFFORT_TEMPLATE
from tests.common.models import GLM5_3_FLASH, PINNED_REVISIONS, QWEN3_0_6B
from tests.common.open_turn_templates import (
    GLM_ASSISTANT_MARKER,
    OPEN_FINAL_TURN_TEMPLATES,
    TWO_EXCHANGES,
    load_open_turn_tokenizer,
)
from tests.common.repo import load_script_module
from tests.common.tokenizers import load_cached_config, load_cached_processor, load_cached_tokenizer

MARKER = "<|im_start|>assistant\n"


def _tokenizer():
    return load_cached_tokenizer(QWEN3_0_6B)


def _collator(tokenizer, **overrides):
    kwargs = {
        "max_length": 256,
        "conversation_field": "messages",
        "hint_template": "\n[Hint] answer is {answer}\n",
        "answer_field": "answer",
        "solution_field": "solution",
        "confidence_power": 1.0,
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
    collator = SelfDistillVLMDataCollator(
        None,
        tokenizer,
        2048,
        train_on_completions_only=False,
        hint_template="\n[Hint] {answer}\n",
        answer_field="answer",
        solution_field="solution",
        confidence_power=1.0,
    )
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


def test_supervising_the_prompt_is_refused_at_the_encoding():
    """With the prompt in the labels the hint shifts every pair after it onto another token."""
    tokenizer = _tokenizer()
    collator = _collator(tokenizer, train_on_completions_only=False, response_prompt_template=None)
    with pytest.raises(ValueError, match="not the same sequence"):
        collator.encode_row(_row())


def _glm_collator(name, **overrides):
    tokenizer, config = load_open_turn_tokenizer(name)
    knobs = {"response_prompt_template": GLM_ASSISTANT_MARKER, "model_config": config, **overrides}
    return _collator(tokenizer, **knobs)


def _plain_ids(tokenizer, history, row):
    """The branch as its template renders it, with nothing appended."""
    rendered = render_conversation(tokenizer, history, row, conversation_field="messages")
    return tokenize_rendered(tokenizer, rendered)["input_ids"]


@pytest.mark.parametrize("name", sorted(OPEN_FINAL_TURN_TEMPLATES))
def test_both_branches_close_the_last_turn_an_open_template_leaves(name):
    """The student and the hinted teacher end on one assistant turn, so both take the closing eos and still supervise
    the same tokens. Left open, that turn trains to the row end and never a stop."""
    collator = _glm_collator(name)
    tokenizer, eos = collator.tokenizer, collator.tokenizer.eos_token_id
    row = {"messages": TWO_EXCHANGES, "answer": "delta"}
    encoded = collator.encode_row(row)
    for prefix, history in (("", TWO_EXCHANGES), ("teacher_", collator._teacher_history(TWO_EXCHANGES, row))):
        plain = _plain_ids(tokenizer, history, row)
        assert plain[-1] not in collator.eos_token_ids, "the template closes its last turn: this case tests nothing"
        assert encoded[f"{prefix}input_ids"] == [*plain, eos]
        assert encoded[f"{prefix}labels"][-1] == eos, "the stop is not trained"

    batch = collator([encoded])
    student = batch["input_ids"][0][batch["labels"][0] != -100]
    teacher = batch["teacher_input_ids"][0][batch["teacher_labels"][0] != -100]
    assert torch.equal(student, teacher)
    decoded = tokenizer.decode(student)
    assert "bravo" in decoded and "delta" in decoded and "charlie" not in decoded, decoded


def test_a_terminated_last_turn_and_a_row_ending_on_a_user_turn_take_no_eos():
    """Qwen3's template terminates every turn, and a row ending on a user turn holds no assistant turn to close."""
    tokenizer = _tokenizer()
    collator = _collator(tokenizer)
    row = _row()
    encoded = collator.encode_row(row)
    assert encoded["input_ids"] == _plain_ids(tokenizer, row["messages"], row)
    assert encoded["teacher_input_ids"] == _plain_ids(tokenizer, collator._teacher_history(row["messages"], row), row)

    full_sequence = _glm_collator("hub", hint_template=None, train_on_completions_only=False)
    ends_on_user = {"messages": TWO_EXCHANGES[:3]}
    plain = _plain_ids(full_sequence.tokenizer, ends_on_user["messages"], ends_on_user)
    assert full_sequence.encode_row(ends_on_user)["input_ids"] == plain


def test_the_vlm_branches_close_the_last_turn_an_open_template_leaves():
    """The VLM collator closes each branch at the text its processor tokenizes, so the student and the hinted teacher
    of an image row both end on the eos and stay aligned."""
    revision = PINNED_REVISIONS[GLM5_3_FLASH]
    processor = load_cached_processor(GLM5_3_FLASH, revision=revision)
    tokenizer, eos = processor.tokenizer, processor.tokenizer.eos_token_id
    collator = SelfDistillVLMDataCollator(
        processor,
        tokenizer,
        4096,
        response_prompt_template=GLM_ASSISTANT_MARKER,
        train_on_completions_only=True,
        hint_template=" hint {answer}",
        answer_field="answer",
        solution_field=None,
        confidence_power=1.0,
        model_config=load_cached_config(GLM5_3_FLASH, revision=revision),
    )
    image = Image.fromarray(np.random.RandomState(0).randint(0, 255, (100, 140, 3), dtype=np.uint8))
    history, images = extract_conversation_images(
        [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "What digit?"}]},
            {"role": "assistant", "content": "Seven."},
        ]
    )
    row = {"history": history, "images": images, "tools_json": None, "answer": "7"}
    assert collator.audit_row(row) == {}

    batch = collator([row])
    trained = {}
    for branch, prefix in (("student", ""), ("teacher", "teacher_")):
        input_ids, labels = batch[f"{prefix}input_ids"][0], batch[f"{prefix}labels"][0]
        assert input_ids[-1] == eos and labels[-1] == eos, f"the {branch} branch is left open"
        trained[branch] = input_ids[labels != -100]
    assert torch.equal(trained["student"], trained["teacher"])
    assert "Seven." in tokenizer.decode(trained["student"])


def _whole_batch_collation(collator, rows):
    """The batch the collator built from raw rows before rows were encoded at prep, spelled out: render and
    tokenize each row, ``tokenizer.pad`` the branch, then label the padded batch."""
    batch = {}
    branches = [("", [row["messages"] for row in rows])]
    if collator.builds_teacher_branch:
        branches.append(("teacher_", [collator._teacher_history(row["messages"], row) for row in rows]))
    for prefix, histories in branches:
        texts = [
            render_conversation(collator.tokenizer, history, row, conversation_field="messages")
            for history, row in zip(histories, rows, strict=True)
        ]
        encoded = collator.tokenizer.pad(
            [tokenize_rendered(collator.tokenizer, text) for text in texts], padding=True, return_tensors="pt"
        )
        batch[f"{prefix}input_ids"], batch[f"{prefix}attention_mask"] = encoded["input_ids"], encoded["attention_mask"]
        batch[f"{prefix}labels"] = build_training_labels(
            encoded["input_ids"],
            collator.tokenizer,
            MARKER,
            True,
            attention_mask=encoded["attention_mask"],
            eos_token_ids=collator.eos_token_ids,
        )
    return batch


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_collating_rows_encoded_at_prep_builds_the_whole_batch_collation(padding_side, monkeypatch):
    """Collation renders nothing and pads ``encode_row``'s rows into the batch the whole-batch collation builds, on
    either padding side."""
    tokenizer = _tokenizer()
    tokenizer.padding_side = padding_side
    collator = _collator(tokenizer)
    rows = [_row(), _row("capital of France, in one word please?", "Paris, the city of light")]
    encoded = [collator.encode_row(row) for row in rows]
    assert set(encoded[0]) == {"input_ids", "labels", "teacher_input_ids", "teacher_labels"}

    monkeypatch.setattr(
        self_distill_collator, "tokenize_training_conversation", lambda *a, **k: pytest.fail("rendered")
    )
    batch = collator(encoded)
    monkeypatch.undo()

    expected = _whole_batch_collation(collator, rows)
    assert batch.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(batch[key], value), key
    assert batch["input_ids"].shape[1] != batch["teacher_input_ids"].shape[1], "the rows must need padding"


def test_the_text_encoding_keeps_only_the_encoded_columns(tmp_path, monkeypatch):
    """Every column the map leaves in place is written to its cache file on every node."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path))
    script = load_script_module("scripts/training/distillation/self_distill.py", "halo_test_self_distill_encode")
    rows = [_row(payload="x" * 4096, confidence=0.5), _row("capital of France?", "Paris", payload="y", confidence=1.0)]

    encoded = script._encode_text_rows(_collator(_tokenizer()), Dataset.from_list(rows), "train", 1, ("confidence",))

    (artifact,) = encoded.cache_files
    assert sorted(Dataset.from_file(artifact["filename"]).column_names) == [
        "confidence",
        "input_ids",
        "labels",
        "teacher_input_ids",
        "teacher_labels",
    ]


def test_no_teacher_branch_is_built_with_the_opd_term_off():
    tokenizer = _tokenizer()
    collator = _collator(tokenizer, hint_template=None, train_on_completions_only=False, response_prompt_template=None)
    assert not collator.builds_teacher_branch
    assert not any(key.startswith("teacher_") for key in collator([_row()]))


@pytest.mark.parametrize("max_length", [8, None], ids=["student-over", "teacher-over"])
def test_an_over_length_branch_raises_by_name_instead_of_truncating(max_length):
    """Truncating one branch would cut response tokens the other keeps; the raise names the branch."""
    tokenizer = _tokenizer()
    hint = " here is a sufficiently long privileged hint: answer={answer}"
    if max_length is None:
        max_length = int(_collator(tokenizer, hint_template=hint)([_row()])["attention_mask"].sum())
    collator = _collator(tokenizer, hint_template=hint, max_length=max_length)
    branch = "student" if max_length == 8 else "teacher"
    with pytest.raises(ValueError, match=f"{branch} branch"):
        collator.encode_row(_row())
    assert _collator(tokenizer, hint_template=hint, max_length=4096).encode_row(_row())


def _confidence_split(values):
    return Dataset.from_dict({"confidence": values})


def test_confidence_weights_divide_by_the_dataset_mean_at_any_batch_size():
    """A per-batch mean is 1.0 at batch size 1, a no-op on every shipped recipe."""
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
    # No eos, so the filter measures every row as rendered.
    eos_token_id = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class _QwenTemplateProcessor:
    """The Qwen3 tokenizer's own chat template behind the processor render seam."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, history, tokenize=False, add_generation_prompt=False, **kwargs):
        return self.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=False, **kwargs)


def _vlm_collator(tokenizer, **overrides):
    kwargs = {
        "response_prompt_template": MARKER,
        "train_on_completions_only": True,
        "hint_template": " hint {answer}",
        "answer_field": "answer",
        "solution_field": "solution",
        "confidence_power": 1.0,
    }
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
            "reasoning_effort_field": None,
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
    """The labels here reach into the prompt, so the hinted branch supervises other tokens and the build raises.
    With the OPD term off the same rows build."""
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
        None,
    )
    if hint_template is None:
        build(hint_template)
        return
    with pytest.raises(ValueError, match="not the same sequence"):
        build(hint_template)


def test_the_vlm_length_filter_measures_the_hinted_teacher_branch():
    """A row only its hint pushes over would otherwise raise in one rank's collator and hang the peers."""
    rows = {
        "history": [[{"role": "user", "content": "q q"}, {"role": "assistant", "content": "a"}]] * 2,
        "tools_json": [None, None],
        "answer": ["x", "x y z w"],
    }
    ds = DatasetDict({"train": Dataset.from_dict(rows)})
    plain = _filter_vlm_over_length(ds, _JoinProcessor(), _WordTokenizer(), 6, 1, None, frozenset())
    assert len(plain["train"]) == 2
    measured = partial(teacher_history, hint_template=" hint {answer}", answer_field="answer", solution_field=None)
    hinted = _filter_vlm_over_length(ds, _JoinProcessor(), _WordTokenizer(), 6, 1, measured, frozenset())
    assert hinted["train"]["answer"] == ["x"]


def _render(tok, **collator_kwargs) -> str:
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
    """Rendering without it would train the model on a prompt shape it is never served with."""
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


def _effort_tokenizer():
    tok = _tokenizer()
    tok.chat_template = QWEN3_6_EFFORT_TEMPLATE.read_text()
    return tok


def test_the_row_level_reaches_the_text_collators_template():
    tok = _effort_tokenizer()
    row = _row(effort="high")

    def render(**collator_kwargs) -> str:
        batch = _collator(tok, hint_template=None, **collator_kwargs)([row])
        return tok.decode(batch["input_ids"][0], skip_special_tokens=False)

    assert LEVEL_LINES["high"] in render(reasoning_effort_field="effort")
    assert "Reasoning effort" not in render()


def test_the_row_level_reaches_the_vlm_collators_template():
    """A mapped VLM row carries its level in ``reasoning_effort``, which the collator's render hands the template."""
    collator = _vlm_collator(_effort_tokenizer(), hint_template=None)
    row = _vlm_row()
    assert LEVEL_LINES["low"] in collator._render(row["history"], {**row, "reasoning_effort": "low"})
    assert "Reasoning effort" not in collator._render(row["history"], {**row, "reasoning_effort": None})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
