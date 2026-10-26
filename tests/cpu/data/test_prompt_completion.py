#!/usr/bin/env python3
"""Prompt+completion tokenization under the two length budgets, on fixtures recorded from SMPO's joint tokenizer
and offline GRPO's per-completion one, both of which cut completions plainly at ``max_completion_length``:
boundary merges, both budgets' cuts, a completion of exactly the cap, terminator-ending completions, both
truncation modes, nominal and post-processor BOS, unset caps."""

import copy

import pytest

from src.data.pipeline.prompt_completion import tokenize_completions_jointly, tokenize_prompt_completions
from tests.common.models import QWEN3_0_6B
from tests.common.tokenizers import load_cached_tokenizer

BOS, EOS, END, MERGED = 1, 2, 3, 4

# Multi-character spellings the stub tokenizes to one id; "ab" is the merge a prompt boundary can split.
SPECIALS = {"<s>": BOS, "</s>": EOS, "<end>": END, "ab": MERGED}

QWEN_PROMPT = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
# The first merges its leading "\n" with the prompt's last one; the second ends without a terminator.
QWEN_COMPLETIONS = ("\nFour<|im_end|>\n", "Five is the answer")
QWEN_IM_END = 151645


class CharTokenizer:
    """One id per character (its code point) but the :data:`SPECIALS`; ``adds_bos`` makes the post-processor emit
    BOS (Llama-style), otherwise BOS is nominal (gpt-oss-style). Truncation follows HF: ``max_length=None`` cuts at
    ``model_max_length``."""

    bos_token_id = BOS
    model_max_length = 3

    def __init__(self, adds_bos=False, truncation_side="right", eos_token_id=EOS):
        self.adds_bos = adds_bos
        self.truncation_side = truncation_side
        self.eos_token_id = eos_token_id

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None):
        ids = []
        while text:
            special = next((spelling for spelling in SPECIALS if text.startswith(spelling)), text[0])
            ids.append(SPECIALS[special] if special in SPECIALS else ord(special))
            text = text[len(special) :]
        if truncation:
            bound = self.model_max_length if max_length is None else max_length
            ids = ids[:bound] if self.truncation_side == "right" else ids[-bound:]
        return {"input_ids": [BOS] + ids if add_special_tokens and self.adds_bos else ids}

    def decode(self, ids):
        names = {token_id: spelling for spelling, token_id in SPECIALS.items()} | {MERGED: "[ab]"}
        return "".join(names.get(token_id, chr(token_id)) for token_id in ids)


def _case(prompt, completions, expected_prompt, expected_completions, *, joint, tokenizer=None, **kwargs):
    budgets = {"max_prompt_length": None, "max_completion_length": None, "truncation_mode": "keep_end"} | kwargs
    return pytest.param(
        tokenizer or {},
        prompt,
        completions,
        {"joint_boundary": joint, **budgets},
        expected_prompt,
        expected_completions,
    )


SMPO = {"joint": True}
OFFLINE = {"joint": False}
POST_PROCESSOR_BOS = {"adds_bos": True}

CASES = {
    "joint-no-merge": _case("xyz", ["qq", "rr"], "xyz", ["qq</s>", "rr</s>"], **SMPO),
    "joint-merge-one-side": _case("xya", ["bz", "qq"], "xy", ["[ab]z</s>", "aqq</s>"], **SMPO),
    "joint-merge-both-sides-no-eos": _case(
        "xya", ["bz", "bq"], "xy", ["[ab]z", "[ab]q"], tokenizer={"eos_token_id": None}, **SMPO
    ),
    "joint-bos-keep-start": _case(
        "wxyz",
        ["q", "r"],
        "<s>wx",
        ["q</s>", "r</s>"],
        tokenizer=POST_PROCESSOR_BOS,
        max_prompt_length=3,
        truncation_mode="keep_start",
        **SMPO,
    ),
    "joint-bos-keep-end": _case(
        "wxyz", ["q", "r"], "xyz", ["q</s>", "r</s>"], tokenizer=POST_PROCESSOR_BOS, max_prompt_length=3, **SMPO
    ),
    "joint-eos-fills-cap": _case("p", ["qq", "rr"], "p", ["qq</s>", "rr</s>"], max_completion_length=3, **SMPO),
    "joint-terminators": _case(
        "p",
        ["qq</s>\n", "rr<end>"],
        "p",
        ["qq</s>", "rr<end>"],
        max_completion_length=3,
        eos_token_ids=frozenset({EOS, END}),
        **SMPO,
    ),
    "joint-over-cap": _case("p", ["qrst", "u"], "p", ["qrs", "u</s>"], max_completion_length=3, **SMPO),
    "joint-content-fills-cap": _case("p", ["qqq", "u"], "p", ["qqq", "u</s>"], max_completion_length=3, **SMPO),
    "joint-zero-caps": _case(
        "pp", ["qrst", "u"], "pp", ["qrst</s>", "u</s>"], max_prompt_length=0, max_completion_length=0, **SMPO
    ),
    "joint-empty-prompt-bos": _case("", ["q", "r"], "<s>", ["q</s>", "r</s>"], tokenizer=POST_PROCESSOR_BOS, **SMPO),
    "alone-no-merge": _case("xya", ["bz"], "xya", ["bz</s>"], **OFFLINE),
    "alone-bos-within-budget": _case(
        "xy", ["q"], "<s>xy", ["q</s>"], tokenizer=POST_PROCESSOR_BOS, max_prompt_length=3, **OFFLINE
    ),
    "alone-bos-at-full-budget": _case(
        "xyz", ["q"], "xyz", ["q</s>"], tokenizer=POST_PROCESSOR_BOS, max_prompt_length=3, **OFFLINE
    ),
    "alone-prompt-keeps-end": _case("vwxyz", ["q"], "xyz", ["q</s>"], max_prompt_length=3, **OFFLINE),
    "alone-cuts": _case("p", ["qq", "qqq", "qrst"], "p", ["qq</s>", "qqq", "qrs"], max_completion_length=3, **OFFLINE),
    "alone-terminators": _case(
        "p", ["qq</s>\n", "rr<end>"], "p", ["qq</s>\n", "rr<end>"], eos_token_ids=frozenset({EOS, END}), **OFFLINE
    ),
    "alone-terminator-cut": _case("p", ["qq</s>\n"], "p", ["qq</s>"], max_completion_length=3, **OFFLINE),
    "alone-left-truncation-side": _case(
        "p", ["qrst"], "p", ["rst"], tokenizer={"truncation_side": "left"}, max_completion_length=3, **OFFLINE
    ),
    "alone-unset-caps": _case("vwxyz", ["qrst"], "vwxyz", ["qrst</s>"], max_prompt_length=0, **OFFLINE),
    "alone-empty-prompt-bos": _case("", ["q"], "<s>", ["q</s>"], tokenizer=POST_PROCESSOR_BOS, **OFFLINE),
}


def _tokenize_stub(tokenizer_kwargs, prompt, completions, kwargs):
    tokenizer = CharTokenizer(**tokenizer_kwargs)
    prompt_ids, completion_ids = tokenize_prompt_completions(
        prompt, completions, tokenizer, **{"eos_token_ids": frozenset(), **kwargs}
    )
    return tokenizer.decode(prompt_ids), [tokenizer.decode(ids) for ids in completion_ids]


@pytest.mark.parametrize(
    ("tokenizer_kwargs", "prompt", "completions", "kwargs", "expected_prompt", "expected_completions"),
    list(CASES.values()),
    ids=list(CASES),
)
def test_stub_fixture(tokenizer_kwargs, prompt, completions, kwargs, expected_prompt, expected_completions):
    assert _tokenize_stub(tokenizer_kwargs, prompt, completions, kwargs) == (expected_prompt, expected_completions)


# The joint cases whose boundary merges, so the split moves one prompt token into the completions.
MERGING_JOINT_CASES = {"joint-merge-one-side", "joint-merge-both-sides-no-eos"}


@pytest.mark.parametrize("case", [name for name in CASES if name.startswith("joint-")])
def test_jointly_tokenized_completions_count_the_moved_prompt_tokens(case):
    """A prompt tokenized elsewhere (the VLM collator's) gets the joint completions unchanged, and the count of
    prompt tokens they start with, which that prompt must drop."""
    tokenizer_kwargs, prompt, completions, kwargs, _, expected_completions = CASES[case].values
    tokenizer = CharTokenizer(**tokenizer_kwargs)
    completion_ids, overlap = tokenize_completions_jointly(
        prompt,
        completions,
        tokenizer,
        max_completion_length=kwargs["max_completion_length"],
        eos_token_ids=kwargs.get("eos_token_ids", frozenset()),
    )
    assert [tokenizer.decode(ids) for ids in completion_ids] == expected_completions
    assert overlap == int(case in MERGING_JOINT_CASES)


QWEN_CASES = {
    "joint-merge": (
        {"joint_boundary": True, "max_prompt_length": None, "max_completion_length": None},
        [151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091],
        [[271, 26972, 151645, 198], [198, 37020, 374, 279, 4226, 151645]],
    ),
    "joint-cuts": (
        {"joint_boundary": True, "max_prompt_length": 4, "max_completion_length": 3},
        [151645, 198, 151644, 77091],
        [[271, 26972, 151645], [198, 37020, 374]],
    ),
    "alone-cuts": (
        {"joint_boundary": False, "max_prompt_length": 6, "max_completion_length": 3},
        [30, 151645, 198, 151644, 77091, 198],
        [[198, 26972, 151645], [37020, 374, 279]],
    ),
    "alone-cuts-left-truncation-side": (
        {"joint_boundary": False, "max_prompt_length": 6, "max_completion_length": 3, "truncation_side": "left"},
        [30, 151645, 198, 151644, 77091, 198],
        [[26972, 151645, 198], [374, 279, 4226]],
    ),
}


@pytest.mark.parametrize(
    ("kwargs", "expected_prompt", "expected_completions"), list(QWEN_CASES.values()), ids=list(QWEN_CASES)
)
def test_hf_tokenizer_fixture(kwargs, expected_prompt, expected_completions):
    kwargs = {"truncation_side": "right", "truncation_mode": "keep_end"} | kwargs
    tokenizer = copy.deepcopy(load_cached_tokenizer(QWEN3_0_6B))
    tokenizer.truncation_side = kwargs.pop("truncation_side")
    prompt_ids, completion_ids = tokenize_prompt_completions(
        QWEN_PROMPT, QWEN_COMPLETIONS, tokenizer, eos_token_ids=frozenset({QWEN_IM_END}), **kwargs
    )
    assert (prompt_ids, completion_ids) == (expected_prompt, expected_completions)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
