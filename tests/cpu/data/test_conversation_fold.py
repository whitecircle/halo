#!/usr/bin/env python3
"""``fold_system_into_conversation`` under the text policy (demote a leading system turn the model lacks a role for)
and the VLM policy (fold only the config prompt), and the collapsed conversation render."""

import pytest

from halo.data.conversation import fold_system_into_conversation
from halo.data.pipeline.text_rows import render_row_conversation
from halo.data.vlm_render import extract_conversation_images


def _msgs(*pairs):
    return [{"role": r, "content": c} for r, c in pairs]


def test_text_supported_prepends_only_when_missing():
    out = fold_system_into_conversation(_msgs(("user", "hi")), "sys", True)
    assert out == _msgs(("system", "sys"), ("user", "hi"))

    existing = _msgs(("system", "already"), ("user", "hi"))
    assert fold_system_into_conversation(existing, "sys", True) == existing


def test_text_unsupported_merges_config_prompt_into_user():
    out = fold_system_into_conversation(_msgs(("user", "hi")), "sys", False)
    assert out == _msgs(("user", "sys\nhi"))


def test_text_unsupported_merges_dataset_system_turn():
    """A dataset-borne leading system turn is demoted even without a config system_prompt."""
    out = fold_system_into_conversation(_msgs(("system", "rules"), ("user", "hi")), None, False)
    assert out == _msgs(("user", "rules\nhi"))


def test_text_unsupported_demotes_role_when_no_user_follows():
    out = fold_system_into_conversation(_msgs(("system", "rules"), ("assistant", "yo")), None, False)
    assert out == _msgs(("user", "rules"), ("assistant", "yo"))


def test_text_unsupported_parts_list_user_content():
    out = fold_system_into_conversation(
        _msgs(("system", "rules"), ("user", [{"type": "text", "text": "hi"}])), None, False
    )
    assert out == _msgs(("user", [{"type": "text", "text": "rules\n"}, {"type": "text", "text": "hi"}]))


def test_vlm_no_prompt_leaves_conversation_untouched():
    conv = _msgs(("system", "rules"), ("user", "hi"))
    assert fold_system_into_conversation(conv, None, False, demote_existing_system=False) == conv


def test_vlm_supported_prepends_system_turn():
    out = fold_system_into_conversation(_msgs(("user", "hi")), "sys", True, demote_existing_system=False)
    assert out == _msgs(("system", "sys"), ("user", "hi"))


def test_vlm_supported_keeps_a_dataset_system_turn_and_takes_no_second_one():
    conv = _msgs(("system", "rules"), ("user", "hi"))
    out = fold_system_into_conversation(conv, "sys", True, demote_existing_system=False)
    assert out == _msgs(("system", "rules"), ("user", "hi"))


def test_vlm_unsupported_prefixes_first_user_turn():
    out = fold_system_into_conversation(_msgs(("user", "hi")), "sys", False, demote_existing_system=False)
    assert out == _msgs(("user", "sys\nhi"))


def test_vlm_unsupported_prefixes_dataset_system_turn():
    """The config prompt folds into a dataset-borne leading system turn, role preserved, and is not dropped."""
    out = fold_system_into_conversation(
        _msgs(("system", "rules"), ("user", "hi")), "sys", False, demote_existing_system=False
    )
    assert out == _msgs(("system", "sys\nrules"), ("user", "hi"))


def test_vlm_unsupported_parts_list_prepends_text_part():
    conv = _msgs(("user", [{"type": "image", "image": None}, {"type": "text", "text": "hi"}]))
    out = fold_system_into_conversation(conv, "sys", False, demote_existing_system=False)
    assert out[0]["content"] == [
        {"type": "text", "text": "sys\n"},
        {"type": "image", "image": None},
        {"type": "text", "text": "hi"},
    ]


def test_process_vlm_conversation_fold_unchanged():
    history, images = extract_conversation_images(
        _msgs(("user", "hi"), ("assistant", "hello")),
        system_prompt="sys",
        model_supports_system_role=False,
    )
    assert images == []
    assert history == [
        {"role": "user", "content": [{"type": "text", "text": "sys\nhi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]


class RenderTokenizer:
    bos_token = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")


def test_render_full_conversation_without_generation_prompt():
    row = {"messages": _msgs(("user", "q"), ("assistant", "a"))}
    out = render_row_conversation(row, RenderTokenizer(), conversation_field="messages")
    assert out == "<user>q</user><assistant>a</assistant>"


def test_render_generation_drops_last_turn_and_adds_generation_prompt():
    row = {"messages": _msgs(("user", "q"), ("assistant", "a"))}
    out = render_row_conversation(row, RenderTokenizer(), conversation_field="messages", add_generation_prompt=True)
    assert out == "<user>q</user><assistant>"


def test_render_system_prompt_folded_when_unsupported():
    row = {"messages": _msgs(("user", "q"))}
    out = render_row_conversation(
        row,
        RenderTokenizer(),
        conversation_field="messages",
        system_prompt="sys",
        model_supports_system_role=False,
    )
    assert out == "<user>sys\nq</user>"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
