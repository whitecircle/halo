#!/usr/bin/env python3
"""tokenize_rendered against REAL cached tokenizers: per-family first/last-token pins.

Verifies the probed special-token contract on the actual model roster (training AND generation
modes), using tokenizers already present in the local HF cache (``local_files_only=True``) — each
test skips cleanly when its tokenizer isn't cached.

Pinned contracts:
- gemma-3: post-processor prepends BOS; rendered template also emits it → exactly one BOS.
- gemma-4: template emits BOS, post-processor adds NOTHING → the rendered BOS must survive.
- Qwen3 / GLM-4.7-Flash / gpt-oss: no specials added → verbatim template tokenization; gpt-oss's
  nominal ``bos_token`` must never be injected.
- Zaya: post-processor prepends BOS and appends ``<|im_end|>`` →
  training rows keep the trailing terminator, generation prompts drop it.

Usage:
    python tests/cpu/data/test_tokenize_rendered_real.py
"""

import sys

import pytest

from src.data.pipeline.rendered import probe_tokenizer_specials, tokenize_rendered
from src.data.pipeline.row_processors import create_llm_processor
from tests.common.models import ZAYA_8B
from tests.common.tokenizers import load_cached_tokenizer

MESSAGES = [
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "4."},
]
GENERATION_MESSAGES = [{"role": "user", "content": "What is 2 + 2?"}]


def _render(tokenizer, messages, add_generation_prompt):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def test_gemma3_single_bos_training_and_generation():
    tok = load_cached_tokenizer("google/gemma-3-4b-it")
    specials = probe_tokenizer_specials(tok)
    assert specials.adds_leading_bos and specials.trailing_special_ids == ()

    for add_gen, messages in ((False, MESSAGES), (True, GENERATION_MESSAGES)):
        text = _render(tok, messages, add_gen)
        ids = tokenize_rendered(tok, text, for_generation=add_gen)["input_ids"]
        assert ids[0] == tok.bos_token_id, "gemma-3 rows must open with BOS"
        assert ids.count(tok.bos_token_id) == 1, f"exactly one BOS expected, got {ids[:5]}..."


def test_gemma4_template_bos_survives():
    tok = load_cached_tokenizer("google/gemma-4-31B-it-qat-w4a16-ct")
    specials = probe_tokenizer_specials(tok)
    assert not specials.adds_leading_bos and specials.trailing_special_ids == ()

    for add_gen, messages in ((False, MESSAGES), (True, GENERATION_MESSAGES)):
        text = _render(tok, messages, add_gen)
        assert text.startswith(tok.bos_token), "precondition: the gemma-4 template emits BOS"
        ids = tokenize_rendered(tok, text, for_generation=add_gen)["input_ids"]
        assert ids[0] == tok.bos_token_id, "the template-emitted BOS must survive (regression: it was stripped)"
        assert ids.count(tok.bos_token_id) == 1


@pytest.mark.parametrize("name", ["Qwen/Qwen3-0.6B", "zai-org/GLM-4.7-Flash"])
def test_no_specials_families_verbatim(name):
    tok = load_cached_tokenizer(name)
    specials = probe_tokenizer_specials(tok)
    assert not specials.adds_leading_bos and specials.trailing_special_ids == ()

    for add_gen, messages in ((False, MESSAGES), (True, GENERATION_MESSAGES)):
        text = _render(tok, messages, add_gen)
        ids = tokenize_rendered(tok, text, for_generation=add_gen)["input_ids"]
        assert ids == tok(text, add_special_tokens=False)["input_ids"], f"{name}: rendered text must tokenize verbatim"


def test_gptoss_nominal_bos_never_injected():
    tok = load_cached_tokenizer("openai/gpt-oss-20b")
    specials = probe_tokenizer_specials(tok)
    assert not specials.adds_leading_bos and specials.trailing_special_ids == ()
    assert tok.bos_token_id is not None, "precondition: gpt-oss defines a nominal bos_token"

    for add_gen, messages in ((False, MESSAGES), (True, GENERATION_MESSAGES)):
        text = _render(tok, messages, add_gen)
        ids = tokenize_rendered(tok, text, for_generation=add_gen)["input_ids"]
        assert ids == tok(text, add_special_tokens=False)["input_ids"]
        # The harmony template opens with <|start|>, never the nominal <|startoftext|>.
        assert ids[0] != tok.bos_token_id


def test_zaya_training_keeps_terminator_generation_drops_it():
    tok = load_cached_tokenizer(ZAYA_8B, trust_remote_code=True)
    specials = probe_tokenizer_specials(tok)
    assert specials.adds_leading_bos, "Zaya's post-processor prepends BOS"
    assert specials.trailing_special_ids == (tok.eos_token_id,), "Zaya's post-processor appends <|im_end|>"

    # Training rows: single BOS, and the terminator appears exactly as the TEMPLATE emits it —
    # Zaya's template already closes the final assistant turn with <|im_end|>\n, so the
    # post-processor's appended copy is a duplicate and must be stripped (regression: training rows
    # ended [..., <|im_end|>, \n, <|im_end|>] — a doubled terminator).
    train_text = _render(tok, MESSAGES, add_generation_prompt=False)
    train_ids = tokenize_rendered(tok, train_text)["input_ids"]
    assert train_ids[0] == tok.bos_token_id and train_ids.count(tok.bos_token_id) == 1
    verbatim_ids = tok(train_text, add_special_tokens=False)["input_ids"]
    assert train_ids.count(tok.eos_token_id) == verbatim_ids.count(tok.eos_token_id), (
        "appended <|im_end|> doubled the template's own terminator on a training row"
    )
    assert train_ids[-2] == tok.eos_token_id, "the template's final <|im_end|> (before its \\n) must survive"

    # Generation prompts must NOT end with the appended terminator at the generation point.
    gen_text = _render(tok, GENERATION_MESSAGES, add_generation_prompt=True)
    gen_out = tokenize_rendered(tok, gen_text, for_generation=True)
    gen_ids = gen_out["input_ids"]
    assert gen_ids[0] == tok.bos_token_id and gen_ids.count(tok.bos_token_id) == 1
    assert gen_ids[-1] != tok.eos_token_id, "generation prompt must not end with the appended <|im_end|>"
    assert len(gen_out["attention_mask"]) == len(gen_ids)


def test_zaya_max_length_boundary_never_doubles_the_terminator():
    """Boundary regression on the real Zaya tokenizer, through the production SFT map fn.

    ``create_llm_processor`` drops rows longer than ``max_length``, so a surviving row is never cut
    by more than the render's trailing whitespace — yet a length-vs-budget dedup guard read those
    rows as "truncated" and let the appended ``<|im_end|>`` double the template's own.
    """
    tok = load_cached_tokenizer(ZAYA_8B, trust_remote_code=True)
    eos_id = tok.eos_token_id
    text = _render(tok, MESSAGES, add_generation_prompt=False)
    baseline = tokenize_rendered(tok, text, truncation=False, padding=False)["input_ids"]
    assert baseline.count(eos_id) >= 1 and baseline[-2] == eos_id, "precondition: the render terminates its last turn"

    row = {"messages": MESSAGES}
    for max_length in range(len(baseline) - 1, len(baseline) + 4):
        ids = create_llm_processor(tok, max_length=max_length, conversation_field="messages")(row)["input_ids"]
        if len(ids) <= 1:
            continue  # rejection sentinel: the row does not fit this budget at all
        assert ids.count(eos_id) == baseline.count(eos_id), (
            f"max_length={max_length}: terminator count drifted from the untruncated row "
            f"({ids.count(eos_id)} vs {baseline.count(eos_id)}); tail={ids[-4:]}"
        )
        assert not (ids[-1] == eos_id and ids[-2] == eos_id), f"max_length={max_length}: doubled terminator {ids[-4:]}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
