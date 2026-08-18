#!/usr/bin/env python
"""The shared fixed test batch must not move with the calendar.

gpt-oss's harmony template stamps ``strftime_now("%Y-%m-%d")`` into its system message, so an
unpinned :func:`tests.common.ep_reference.fixed_chat_batch` tokenizes to different ids — and a
different loss — on every new day, silently re-deriving the EP/TP correctness thresholds that are
measured against it (at 2025-01-01 the rotated-expert control drops below ``control_min_loss_shift``).
``fixed_chat_batch`` shadows the template's clock with :data:`CHAT_TEMPLATE_NOW`; these tests fail
the moment that pin stops reaching the renderer.

Run: python tests/cpu/conventions/test_fixed_batch_date_pin.py
"""

from datetime import datetime

import pytest

from tests.common.ep_reference import CHAT_TEMPLATE_NOW, fixed_chat_batch
from tests.common.models import GPT_OSS_20B, QWEN3_0_6B
from tests.common.tokenizers import load_cached_tokenizer

SEQ_LEN = 128
PINNED_DATE = CHAT_TEMPLATE_NOW.strftime("%Y-%m-%d")
# A family whose template asks for a different directive still gets the same instant.
DATE_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}Date: {{ strftime_now('%d %B %Y') }}"
)


def _batch_text(tokenizer):
    input_ids, _, _ = fixed_chat_batch(tokenizer, SEQ_LEN, "cpu")
    return tokenizer.decode(input_ids[0])


def _tokenizer(name):
    tokenizer = load_cached_tokenizer(name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def test_harmony_batch_carries_the_pinned_date():
    text = _batch_text(_tokenizer(GPT_OSS_20B))
    assert f"Current date: {PINNED_DATE}" in text, (
        f"gpt-oss fixed batch does not carry the pinned date {PINNED_DATE}. Its harmony template "
        f"renders the live clock unless fixed_chat_batch shadows `strftime_now`. Rendered: {text[:300]!r}"
    )
    today = datetime.now().strftime("%Y-%m-%d")
    assert today == PINNED_DATE or today not in text, (
        f"today's date {today} leaked into the fixed batch — the pin is not the only clock reaching "
        "the template, so the batch (and every threshold derived from it) still drifts."
    )


def test_pin_covers_other_date_formats():
    tokenizer = _tokenizer(QWEN3_0_6B)
    tokenizer.chat_template = DATE_TEMPLATE
    expected = CHAT_TEMPLATE_NOW.strftime("%d %B %Y")
    assert f"Date: {expected}" in _batch_text(tokenizer), (
        f"a template asking for '%d %B %Y' did not render {expected!r}: the pin must be a clock the "
        "template formats itself, not one pre-formatted date string."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
