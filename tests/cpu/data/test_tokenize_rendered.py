#!/usr/bin/env python3
"""tokenize_rendered: probe-based special-token ownership for rendered chat-template text.

The seam probes each tokenizer once (``tokenizer("", add_special_tokens=True)``) to learn what the
post-processor really adds, then keeps every special exactly once. Pins all tokenizer contracts:
- no-BOS tokenizer (Qwen-style): token IDs identical to a plain ``tokenizer(text)`` call.
- BOS-post-processing tokenizer + BOS-emitting template (gemma-3/llama-style): exactly one BOS,
  counted inside the truncation budget.
- ``bos_token`` defined but NOT post-processed, template does NOT emit it (gpt-oss/Bailing-style):
  no BOS is invented — the served policy never sees one.
- ``bos_token`` defined but NOT post-processed, template DOES emit it (gemma-4-style): the
  template's BOS survives verbatim (stripping + add_special_tokens=True would silently drop it).
- trailing-special post-processor (Zaya-style ``<|im_end|>``): preserved on training rows, stripped
  from generation prompts (``for_generation=True``) so they never end with a turn terminator.

Usage:
    python tests/cpu/data/test_tokenize_rendered.py
"""

import sys

import pytest

from src.data.pipeline.preferences import build_reward_preprocess_fn
from src.data.pipeline.rendered import probe_tokenizer_specials, render_generation_prompt, tokenize_rendered
from tests.common.utils import load_script_module

BOS_ID = 1
TRAILING_EOS_ID = 555


class NoBosTokenizer:
    """Qwen-style: no BOS token; add_special_tokens is a no-op."""

    bos_token = None
    bos_token_id = None

    def _ids(self, text):
        return [100 + i for i, _ in enumerate(text.split())]

    def __call__(self, text, add_special_tokens=True, truncation=False, padding=False, max_length=None):
        ids = self._ids(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class BosTokenizer:
    """Gemma-3/Llama-style: BOS in the template AND prepended by the post-processor.

    A leading ``<bos>`` in the text tokenizes to the BOS id (special tokens parse from text),
    exactly like real tokenizers handle rendered templates.
    """

    bos_token = "<bos>"
    bos_token_id = BOS_ID

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        return self.bos_token + "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)

    def __call__(self, text, add_special_tokens=True, truncation=False, padding=False, max_length=None):
        ids = []
        if text.startswith(self.bos_token):
            ids.append(BOS_ID)
            text = text[len(self.bos_token) :]
        ids += [100 + i for i, _ in enumerate(text.split())]
        if add_special_tokens:
            ids = [BOS_ID] + ids
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class NominalBosTokenizer(NoBosTokenizer):
    """gpt-oss/Bailing-style: ``bos_token`` exists in the vocab but neither the template nor the
    add_special_tokens post-processor emits it."""

    bos_token = "<|startoftext|>"
    bos_token_id = 999


class Gemma4StyleTokenizer(NoBosTokenizer):
    """gemma-4-style: the template emits BOS but the post-processor does NOT re-add it —
    the rendered BOS must survive verbatim."""

    bos_token = "<bos>"
    bos_token_id = BOS_ID

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        return self.bos_token + "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)

    def __call__(self, text, add_special_tokens=True, truncation=False, padding=False, max_length=None):
        ids = []
        if text.startswith(self.bos_token):
            ids.append(BOS_ID)
            text = text[len(self.bos_token) :]
        ids += [100 + i for i, _ in enumerate(text.split())]
        # No post-processing: add_special_tokens adds nothing.
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


TRAILING_EOS_TOKEN = "<eos>"


class TrailingEosTokenizer(NoBosTokenizer):
    """Zaya-style trailing special: the add_special_tokens post-processor appends an EOS."""

    eos_token_id = TRAILING_EOS_ID
    eos_token = TRAILING_EOS_TOKEN

    def decode(self, ids):
        return "".join(self.eos_token if i == self.eos_token_id else f"[{i}]" for i in ids)

    def __call__(self, text, add_special_tokens=True, truncation=False, padding=False, max_length=None):
        out = super().__call__(text, truncation=truncation, max_length=max_length)
        if add_special_tokens:
            out["input_ids"] = out["input_ids"] + [self.eos_token_id]
            out["attention_mask"] = out["attention_mask"] + [1]
        return out


class BosAndTrailingEosTokenizer(BosTokenizer):
    """Zaya's real contract: post-processor prepends BOS AND appends a trailing ``<|im_end|>``."""

    eos_token_id = TRAILING_EOS_ID
    eos_token = TRAILING_EOS_TOKEN

    def decode(self, ids):
        return "".join(self.eos_token if i == self.eos_token_id else f"[{i}]" for i in ids)

    def __call__(self, text, add_special_tokens=True, truncation=False, padding=False, max_length=None):
        out = super().__call__(
            text, add_special_tokens=add_special_tokens, truncation=truncation, padding=padding, max_length=max_length
        )
        if add_special_tokens:
            out["input_ids"] = out["input_ids"] + [self.eos_token_id]
            out["attention_mask"] = out["attention_mask"] + [1]
        return out


class TerminatedRenderTokenizer(TrailingEosTokenizer):
    """Zaya's training-row shape: the chat template ITSELF ends the final assistant turn with the
    terminator (``…<eos>\\n``), and the post-processor would append another one on top."""

    def _ids(self, text):
        return [self.eos_token_id if w == self.eos_token else 100 + i for i, w in enumerate(text.split())]


class BudgetedTerminatedRenderTokenizer(TerminatedRenderTokenizer):
    """HF-faithful truncation on top of the terminated-render shape: the appended trailing special
    fits INSIDE the ``max_length`` budget (fast tokenizers truncate the text to make room for
    post-processor specials), so a truncated row ends with the appended terminator at exactly
    ``max_length`` tokens."""

    def __call__(self, text, add_special_tokens=True, truncation=False, padding=False, max_length=None):
        text_budget = max_length - 1 if (add_special_tokens and truncation and max_length is not None) else max_length
        out = NoBosTokenizer.__call__(self, text, truncation=truncation, max_length=text_budget)
        if add_special_tokens:
            out["input_ids"] = out["input_ids"] + [self.eos_token_id]
            out["attention_mask"] = out["attention_mask"] + [1]
        return out


def _unpack(specials):
    return (specials.adds_leading_bos, specials.trailing_special_ids)


def test_probe_contracts():
    """The probe classifies each stub's real add_special_tokens contract."""
    assert _unpack(probe_tokenizer_specials(NoBosTokenizer())) == (False, ())
    assert _unpack(probe_tokenizer_specials(BosTokenizer())) == (True, ())
    assert _unpack(probe_tokenizer_specials(NominalBosTokenizer())) == (False, ())
    assert _unpack(probe_tokenizer_specials(Gemma4StyleTokenizer())) == (False, ())
    assert _unpack(probe_tokenizer_specials(TrailingEosTokenizer())) == (False, (TRAILING_EOS_ID,))
    assert _unpack(probe_tokenizer_specials(BosAndTrailingEosTokenizer())) == (True, (TRAILING_EOS_ID,))


def test_probe_cached_per_instance():
    """The probe runs once per tokenizer instance (cached), and per-instance — not per-class."""
    calls = []

    class CountingTokenizer(NoBosTokenizer):
        def __call__(self, text, **kwargs):
            if text == "":
                calls.append(1)
            return super().__call__(text, **kwargs)

    tok = CountingTokenizer()
    probe_tokenizer_specials(tok)
    probe_tokenizer_specials(tok)
    assert len(calls) == 1, "probe must be cached after the first call"


def test_nominal_bos_tokenizer_gets_no_bos():
    """A tokenizer that defines bos_token without post-processing it must NOT gain a BOS —
    prepending one would train on a token the served policy never produces (gpt-oss regression)."""
    tok = NominalBosTokenizer()
    text = "hello world"
    ids = tokenize_rendered(tok, text)["input_ids"]
    assert tok.bos_token_id not in ids
    assert ids == tok(text)["input_ids"], "must match the tokenizer's own add_special_tokens output"


def test_gemma4_style_template_bos_survives():
    """gemma-4-style: the template's rendered BOS must survive — the historical strip +
    add_special_tokens=True dropped it (the post-processor never re-adds)."""
    tok = Gemma4StyleTokenizer()
    text = tok.apply_chat_template([{"role": "user", "content": "hi there"}])
    assert text.startswith(tok.bos_token), "precondition: the template emits BOS"

    ids = tokenize_rendered(tok, text)["input_ids"]
    assert ids[0] == BOS_ID, f"the template-emitted BOS must survive, got {ids}"
    assert ids.count(BOS_ID) == 1


def test_no_bos_tokenizer_equivalence():
    """No-BOS path is token-ID identical to the historical tokenizer(text) call."""
    tok = NoBosTokenizer()
    text = "hello world how are you"
    assert tokenize_rendered(tok, text)["input_ids"] == tok(text)["input_ids"]
    assert (
        tokenize_rendered(tok, text, truncation=True, max_length=3)["input_ids"]
        == tok(text, truncation=True, max_length=3)["input_ids"]
    )


def test_bos_template_no_double_bos():
    """Rendered text that already carries BOS yields exactly ONE leading BOS (reward-path fix:
    the historical add_special_tokens=True call produced two)."""
    tok = BosTokenizer()
    text = tok.bos_token + "hello world"

    legacy_ids = tok(text)["input_ids"]
    assert legacy_ids[:2] == [BOS_ID, BOS_ID], "precondition: the legacy call double-added BOS"

    ids = tokenize_rendered(tok, text)["input_ids"]
    assert ids[0] == BOS_ID and ids[1] != BOS_ID, f"expected a single leading BOS, got {ids}"


def test_stripped_render_gains_single_bos():
    """A BOS-stripped render regains exactly one BOS when the post-processor owns BOS
    (teacher-distill fix: the historical add_special_tokens=False call had zero)."""
    tok = BosTokenizer()
    text = "hello world"

    legacy_ids = tok(text, add_special_tokens=False)["input_ids"]
    assert BOS_ID not in legacy_ids, "precondition: the legacy call had no BOS"

    ids = tokenize_rendered(tok, text)["input_ids"]
    assert ids[0] == BOS_ID and ids[1] != BOS_ID


def test_bos_counts_toward_truncation_budget():
    tok = BosTokenizer()
    ids = tokenize_rendered(tok, "a b c d e", truncation=True, max_length=3)["input_ids"]
    assert len(ids) == 3 and ids[0] == BOS_ID


def test_trailing_eos_post_processing_preserved_on_training_rows():
    """Zaya-style trailing EOS from add_special_tokens=True must survive on training rows
    (add_special_tokens=False would silently drop the row terminator)."""
    tok = TrailingEosTokenizer()
    ids = tokenize_rendered(tok, "hello world")["input_ids"]
    assert ids[-1] == tok.eos_token_id


def test_for_generation_strips_trailing_special():
    """Generation prompts must not end with the tokenizer-appended turn terminator — the model
    would see its own end-of-turn at the generation point."""
    tok = TrailingEosTokenizer()
    out = tokenize_rendered(tok, "hello world", for_generation=True)
    assert out["input_ids"] == tok("hello world", add_special_tokens=False)["input_ids"]
    assert out["input_ids"][-1] != tok.eos_token_id
    assert len(out["attention_mask"]) == len(out["input_ids"]), "attention_mask must be trimmed too"


def test_for_generation_keeps_bos_strips_trailing():
    """Zaya's full contract under for_generation: single BOS kept, trailing terminator stripped."""
    tok = BosAndTrailingEosTokenizer()
    text = tok.bos_token + "hello world"
    ids = tokenize_rendered(tok, text, for_generation=True)["input_ids"]
    assert ids[0] == BOS_ID and ids.count(BOS_ID) == 1
    assert ids[-1] != tok.eos_token_id

    training_ids = tokenize_rendered(tok, text)["input_ids"]
    assert training_ids == ids + [tok.eos_token_id], "training rows keep the tokenizer's contract"


def test_trailing_special_not_doubled_when_render_already_terminated():
    """Zaya regression: the chat template already ends the final assistant turn with the terminator
    (``…<eos>\\n``), and add_special_tokens=True appends ANOTHER — doubling the terminator on
    training rows and violating the once-only contract. The appended duplicate must be stripped."""
    tok = TerminatedRenderTokenizer()
    text = f"hello world {TRAILING_EOS_TOKEN}\n"  # render terminates the turn itself (+ trailing newline)

    ids = tokenize_rendered(tok, text)["input_ids"]
    assert ids.count(TRAILING_EOS_ID) == 1, f"terminator doubled on the training row: {ids}"
    assert ids == tok(text, add_special_tokens=False)["input_ids"], (
        "training row must equal the verbatim render when it already carries the terminator"
    )

    # An unterminated render still gains the appended terminator (the post-processor's job).
    open_ids = tokenize_rendered(tok, "hello world")["input_ids"]
    assert open_ids[-1] == TRAILING_EOS_ID and open_ids.count(TRAILING_EOS_ID) == 1


def test_truncated_row_keeps_appended_terminator():
    """Truncation regression: when truncation cut the render's own terminator, the post-processor's
    appended copy is the row's ONLY terminator — the untruncated-text dedup must NOT strip it."""
    tok = BudgetedTerminatedRenderTokenizer()
    text = f"w0 w1 w2 w3 w4 w5 {TRAILING_EOS_TOKEN}\n"  # full render terminates the turn itself

    ids = tokenize_rendered(tok, text, truncation=True, max_length=4)["input_ids"]
    assert len(ids) == 4, f"truncation budget must hold, got {ids}"
    assert ids[-1] == TRAILING_EOS_ID and ids.count(TRAILING_EOS_ID) == 1, (
        f"the appended terminator is the only surviving one and must be kept, got {ids}"
    )


def test_untruncated_row_with_truncation_kwarg_still_deduped():
    """truncation=True that never engages (row under max_length) keeps the dedup: the render's own
    terminator survives, the appended duplicate is stripped."""
    tok = BudgetedTerminatedRenderTokenizer()
    text = f"hello world {TRAILING_EOS_TOKEN}\n"

    ids = tokenize_rendered(tok, text, truncation=True, max_length=64)["input_ids"]
    assert ids.count(TRAILING_EOS_ID) == 1, f"terminator doubled on the untruncated row: {ids}"
    assert ids == tok(text, add_special_tokens=False)["input_ids"]


def test_row_that_exactly_fills_the_budget_is_still_deduped():
    """Boundary regression: a render that EXACTLY fills ``max_length`` is never cut, so its own
    terminator survives and the appended copy is still a duplicate.

    Skipping the dedup whenever ``len >= max_length`` cannot distinguish "was cut" from "exactly
    fits" — every row landing on the budget then carries a doubled terminator.
    """
    tok = BudgetedTerminatedRenderTokenizer()
    text = f"w0 w1 w2 {TRAILING_EOS_TOKEN}\n"
    untruncated = tokenize_rendered(tok, text)["input_ids"]
    budget = len(untruncated) + 1  # the appended duplicate exactly fills it — nothing is cut

    ids = tokenize_rendered(tok, text, truncation=True, max_length=budget)["input_ids"]
    assert ids == untruncated, f"row at the max_length boundary diverged from the untruncated row: {ids}"
    assert ids.count(TRAILING_EOS_ID) == 1, f"terminator doubled at the max_length boundary: {ids}"


def test_for_generation_noop_without_trailing_specials():
    """for_generation must not change output for tokenizers that append nothing."""
    for tok in (NoBosTokenizer(), BosTokenizer(), NominalBosTokenizer(), Gemma4StyleTokenizer()):
        text = "hello world"
        assert (
            tokenize_rendered(tok, text, for_generation=True)["input_ids"] == tokenize_rendered(tok, text)["input_ids"]
        ), type(tok).__name__


def test_reward_preprocess_single_bos_for_bos_template():
    """build_reward_preprocess_fn routes through tokenize_rendered: single BOS on both branches."""
    fn = build_reward_preprocess_fn(BosTokenizer(), max_length=64)
    out = fn(
        {
            "prompt": [[{"role": "user", "content": "q"}]],
            "chosen": [[{"role": "assistant", "content": "good answer"}]],
            "rejected": [[{"role": "assistant", "content": "bad"}]],
        }
    )
    for key in ("input_ids_chosen", "input_ids_rejected"):
        ids = out[key][0]
        assert ids[0] == BOS_ID and BOS_ID not in ids[1:], f"{key} must carry exactly one BOS, got {ids}"


def test_reward_preprocess_keeps_gemma4_bos():
    """Reward path regression: gemma-4-style templates keep their BOS (a leading-BOS strip plus
    add_special_tokens=True drops it)."""
    tok = Gemma4StyleTokenizer()
    fn = build_reward_preprocess_fn(tok, max_length=64)
    out = fn(
        {
            "prompt": [[{"role": "user", "content": "q"}]],
            "chosen": [[{"role": "assistant", "content": "good answer"}]],
            "rejected": [[{"role": "assistant", "content": "bad"}]],
        }
    )
    for key in ("input_ids_chosen", "input_ids_rejected"):
        ids = out[key][0]
        assert ids[0] == BOS_ID and ids.count(BOS_ID) == 1, f"{key} must keep the template BOS, got {ids}"


def test_render_generation_prompt_keeps_bos_when_nothing_re_adds_it():
    """gemma-4-style: the template's BOS must stay in the returned prompt TEXT.

    Nothing downstream puts it back — TRL's GRPO tokenizes the prompt with
    ``add_special_tokens=False`` and this tokenizer's post-processor adds nothing — so an
    unconditional strip silently trains and rolls out on a BOS-less prompt.
    """
    tok = Gemma4StyleTokenizer()
    messages = [{"role": "user", "content": "hi there"}]
    text = render_generation_prompt(tok, messages)
    assert text.startswith(tok.bos_token), f"gemma-4-style prompt lost its BOS: {text[:40]!r}"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert ids[0] == BOS_ID and ids.count(BOS_ID) == 1


def test_render_generation_prompt_strips_bos_when_post_processor_re_adds_it():
    """gemma-3-style: the post-processor prepends BOS, so the render's own copy must go — keeping
    both would double it the moment anything tokenizes with add_special_tokens=True."""
    tok = BosTokenizer()
    text = render_generation_prompt(tok, [{"role": "user", "content": "hi there"}])
    assert not text.startswith(tok.bos_token), f"BOS-post-processing render kept its BOS: {text[:40]!r}"
    ids = tok(text)["input_ids"]
    assert ids[0] == BOS_ID and ids.count(BOS_ID) == 1


def test_render_generation_prompt_budget_counts_generation_tokenization():
    """The max_prompt_length filter must measure the ids the row will really carry.

    A Zaya-style post-processor appends a turn terminator that a generation prompt never keeps, so
    measuring with a bare ``tokenizer(text)`` counts a token that is not there and rejects prompts
    that fit.
    """
    tok = BosAndTrailingEosTokenizer()
    messages = [{"role": "user", "content": "one two three"}]
    text = render_generation_prompt(tok, messages)
    real_len = len(tokenize_rendered(tok, text, for_generation=True)["input_ids"])
    naive_len = len(tok(text)["input_ids"])
    assert naive_len > real_len, "precondition: the naive count over-counts the appended terminator"
    assert render_generation_prompt(tok, messages, max_prompt_length=real_len) is not None, (
        "a prompt that exactly fits its real length must pass the filter"
    )
    assert render_generation_prompt(tok, messages, max_prompt_length=real_len - 1) is None


def _classification_script():
    return load_script_module("scripts/training/classification.py", "halo_classification_script")


def test_classification_row_tokenization_never_doubles_bos():
    """The classification script's row processor routes through tokenize_rendered.

    A bare ``tokenizer(text=rendered)`` re-adds the post-processor's BOS on top of the template's,
    which is what gemma-3 / Zaya / LFM2 actually do — every classification row would open with two
    BOS tokens.
    """
    module = _classification_script()

    tok = BosTokenizer()
    row = {"prompt": [{"role": "user", "content": "some text here"}], "label": "spam"}
    out = module.tokenize_classification_row(
        dict(row), tokenizer=tok, max_length=64, label_to_id={"ham": 0, "spam": 1}, is_multi_label=False
    )
    ids = out["input_ids"]
    assert ids[0] == BOS_ID, f"row must open with BOS, got {ids}"
    assert ids.count(BOS_ID) == 1, f"doubled BOS on a classification row: {ids}"
    assert out["label"] == 1, "label must map through label_to_id"

    # Pin the naive call's doubling, so the single-BOS contract cannot silently regress.
    naive = tok(text=tok.apply_chat_template(row["prompt"]))["input_ids"]
    assert naive.count(BOS_ID) == 2, "precondition: the naive tokenization doubles BOS"


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ([0, 2], [1.0, 0.0, 1.0]),  # int-valued column: get_label_list stringified the keys
        (["0", "1"], [1.0, 1.0, 0.0]),  # already strings
    ],
)
def test_classification_row_maps_multi_labels_of_either_type(labels, expected):
    """``get_label_list`` stringifies every key, so both label paths must look up ``str(label)``.

    An int-valued multi-label column is what a HF dataset with a ``Sequence(ClassLabel)`` column
    yields, and a raw-value lookup ``KeyError``s on it while the single-label path accepts the very
    same column type.
    """
    module = _classification_script()

    out = module.tokenize_classification_row(
        {"prompt": [{"role": "user", "content": "some text here"}], "label": labels},
        tokenizer=BosTokenizer(),
        max_length=64,
        label_to_id={"0": 0, "1": 1, "2": 2},
        is_multi_label=True,
    )
    assert out["label"] == expected


@pytest.mark.parametrize("sentinel", [-1, "-1"])
def test_classification_row_skips_the_unlabeled_sentinel_in_a_multi_label_row(sentinel):
    """``get_label_list`` warns about and REMOVES ``-1`` from the label list, so it has no id to look
    up — the single-label branch passes it through, and a multi-hot row says the same thing by leaving
    every slot at 0. Without the skip the row ``KeyError``s on a key the pipeline deliberately dropped.
    """
    module = _classification_script()

    out = module.tokenize_classification_row(
        {"prompt": [{"role": "user", "content": "some text here"}], "label": [sentinel, 2]},
        tokenizer=BosTokenizer(),
        max_length=64,
        label_to_id={"0": 0, "1": 1, "2": 2},
        is_multi_label=True,
    )
    assert out["label"] == [0.0, 0.0, 1.0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
