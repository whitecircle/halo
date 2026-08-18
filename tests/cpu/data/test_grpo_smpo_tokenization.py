#!/usr/bin/env python3
"""Offline-GRPO / SMPO tokenization contracts.

1. Offline-GRPO completion render: completions are rendered as ``template(prompt + completion)``
   minus the rendered-prompt prefix — never standalone. Standalone rendering raises on strict
   templates (Qwen3.5 rejects assistant-only message lists) and injects a mid-sequence BOS on
   BOS-emitting templates.
2. Forced-BOS sites: ``tokenize_preference_row`` (SMPO) and ``tokenize_prompt_completion``
   (offline GRPO) prepend BOS only
   when the tokenizer's own post-processor would — gpt-oss/Bailing define a nominal ``bos_token``
   the served policy never emits, and force-prepending it trains on a phantom token.

Usage:
    python tests/cpu/data/test_grpo_smpo_tokenization.py
"""

import sys

import pytest
from accelerate import PartialState

PartialState()

from scripts.training.offline_grpo import build_chat_template_row_fn
from src.data.pipeline.row_processors import prepare_generative_row
from src.trainers.grpo.offline import tokenize_prompt_completion
from src.trainers.preference.smpo import tokenize_preference_row

BOS_ID = 1
NOMINAL_BOS_ID = 999
EOS_ID = 900


class StrictTemplateTokenizer:
    """Qwen3.5-style STRICT template: raises on a conversation not opening with system/user."""

    bos_token = None

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        if not messages or messages[0]["role"] not in ("system", "user"):
            raise ValueError("Conversation must start with a system or user turn (strict template)")
        return "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)


class BosTemplateTokenizer:
    """Lenient BOS-emitting template: standalone completion renders gain a leading BOS."""

    bos_token = "<bos>"

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        return self.bos_token + "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)


ROW = {
    "prompt": [{"role": "user", "content": "What is 2+2?"}],
    "completions": [
        [{"role": "assistant", "content": "4"}],
        [{"role": "assistant", "content": "five"}],
    ],
    "rewards": [1.0, 0.0],
}


def test_offline_grpo_render_survives_strict_template():
    """A bare ``apply_chat_template([assistant-msg])`` raises on strict templates."""
    fn = build_chat_template_row_fn(StrictTemplateTokenizer(), tools_field=None)
    out = fn({**ROW, "completions": [list(c) for c in ROW["completions"]]})
    assert out["prompt"] == "<|user|>What is 2+2?<|end|>"
    assert out["completions"] == ["<|assistant|>4<|end|>", "<|assistant|>five<|end|>"]


def test_offline_grpo_render_no_mid_sequence_bos():
    """A standalone completion render carries the template's leading BOS, which would put a BOS
    mid-sequence once prompt and completion are concatenated."""
    tok = BosTemplateTokenizer()
    fn = build_chat_template_row_fn(tok, tools_field=None)
    out = fn({**ROW, "completions": [list(c) for c in ROW["completions"]]})
    for completion in out["completions"]:
        assert tok.bos_token not in completion, f"mid-sequence BOS in completion: {completion!r}"
        # The prefix-strip invariant: prompt + completion reconstructs the full rendered conversation.
    full = tok.apply_chat_template(ROW["prompt"] + ROW["completions"][0])
    assert out["prompt"] + out["completions"][0] == full


class NominalBosTokenizer:
    """gpt-oss/Bailing-style: bos_token exists but the post-processor never emits it."""

    bos_token = "<|startoftext|>"
    bos_token_id = NOMINAL_BOS_ID
    eos_token_id = EOS_ID
    _adds_bos = False

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None, padding=False):
        ids = [100 + i for i, _ in enumerate(text.split())]
        if add_special_tokens and self._adds_bos:
            ids = [self.bos_token_id] + ids
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids):
        # The terminator walk decodes id-by-id to separate template whitespace from content.
        return " ".join(f"w{i}" for i in ids)


class PostProcBosTokenizer(NominalBosTokenizer):
    """Llama-style: the add_special_tokens post-processor really prepends BOS."""

    bos_token = "<bos>"
    bos_token_id = BOS_ID
    _adds_bos = True


def _offline_tokenize(tokenizer, *, max_prompt_length=None, max_completion_length=None, prompt=None, completion=None):
    return tokenize_prompt_completion(
        prompt if prompt is not None else "user prompt text",
        completion if completion is not None else "the completion",
        tokenizer,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        is_encoder_decoder=False,
    )


def test_offline_tokenize_no_forced_bos_for_nominal_bos():
    """gpt-oss-style: the nominal BOS must NOT be injected (the served policy never emits it)."""
    batch = _offline_tokenize(NominalBosTokenizer())
    assert NOMINAL_BOS_ID not in batch["prompt_input_ids"], f"phantom BOS injected: {batch['prompt_input_ids']}"
    assert batch["completion_input_ids"][-1] == EOS_ID, "terminal EOS on the completion must remain"


def test_offline_tokenize_bos_kept_for_postprocessing_tokenizer():
    """Post-processor-owned BOS is still prepended to a BOS-less rendered prompt."""
    batch = _offline_tokenize(PostProcBosTokenizer())
    assert batch["prompt_input_ids"][0] == BOS_ID
    assert batch["prompt_input_ids"].count(BOS_ID) == 1


def test_offline_tokenize_caps_are_true_bounds():
    """The caps are token BUDGETS: a completion truncated AT the cap must not overflow to cap+1
    via the EOS append (the stored rollout did not end there — EOS at the cut would teach premature
    stopping), and BOS must not push a cap-full prompt to cap+1. The pipeline's fixed P2P shape is
    derived as max_prompt_length + max_completion_length, so an overflow crashes PP mid-epoch."""
    batch = _offline_tokenize(
        PostProcBosTokenizer(),
        max_prompt_length=3,
        max_completion_length=2,
        prompt="one two three four five",
        completion="three completion words",
    )
    assert len(batch["prompt_input_ids"]) == 3, f"prompt over budget: {batch['prompt_input_ids']}"
    assert BOS_ID not in batch["prompt_input_ids"], "BOS must not displace budgeted prompt tokens"
    assert len(batch["completion_input_ids"]) == 2, f"completion over budget: {batch['completion_input_ids']}"
    assert batch["completion_input_ids"][-1] != EOS_ID, "truncated rollout must stay EOS-free at the cut"


class HFTruncationTokenizer(NominalBosTokenizer):
    """Models HF's real truncation contract: ``truncation=True`` with ``max_length=None`` resolves
    the bound against ``model_max_length`` rather than meaning "unbounded"."""

    model_max_length = 4

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None, padding=False):
        out = super().__call__(text, add_special_tokens=add_special_tokens)
        if truncation:
            bound = self.model_max_length if max_length is None else max_length
            return {key: value[:bound] for key, value in out.items()}
        return out


def test_offline_tokenize_null_completion_cap_is_no_cap():
    """``max_completion_length: null`` is documented as NO cap, but it reached the tokenizer as
    ``truncation=True, max_length=None`` — which HF resolves against ``model_max_length``, i.e. the
    window the launch script pinned from the *prompt* budget. Every stored completion was therefore
    cut at a number nobody chose, and because the EOS-at-the-cut guard keys off the cap being set,
    an EOS was still appended at the cut — teaching premature stopping exactly where the data was
    truncated."""
    batch = _offline_tokenize(
        HFTruncationTokenizer(), max_completion_length=None, completion="one two three four five six"
    )
    assert len(batch["completion_input_ids"]) == 7, (
        f"uncapped completion truncated at model_max_length: {batch['completion_input_ids']}"
    )
    assert batch["completion_input_ids"][-1] == EOS_ID


def test_offline_tokenize_explicit_completion_cap_still_truncates():
    """The converse guard: honouring 'null = no cap' must not disable truncation when a cap IS set."""
    batch = _offline_tokenize(
        HFTruncationTokenizer(), max_completion_length=2, completion="one two three four five six"
    )
    assert len(batch["completion_input_ids"]) == 2, f"explicit cap not applied: {batch['completion_input_ids']}"


def test_generative_row_null_prompt_cap_is_no_cap():
    """``prepare_generative_row`` builds the eval-time generation prompts for offline GRPO / DPO /
    SMPO and carried the same defect: ``truncation=True`` with a nullable cap, so an uncapped prompt
    was cut at ``model_max_length`` — silently shortening the very prompts the eval generations are
    judged on."""

    class _Tok(HFTruncationTokenizer):
        pad_token_id = 0
        padding_side = "right"

        def apply_chat_template(self, messages, tokenize=False, **kwargs):
            return " ".join(m["content"] for m in messages)

        def pad(self, encoded, **kwargs):
            return encoded

    row = {"prompt": [{"role": "user", "content": "one two three four five six"}]}
    out = prepare_generative_row(row, _Tok(), max_length=None)
    assert len(out["input_ids"]) == 6, f"uncapped generation prompt truncated: {out['input_ids']}"


def test_offline_tokenize_eos_and_bos_within_budget():
    """Under the caps, the terminal EOS and the post-processor BOS are kept as before."""
    batch = _offline_tokenize(
        PostProcBosTokenizer(), max_prompt_length=3, max_completion_length=8, prompt="one two", completion="short one"
    )
    assert batch["prompt_input_ids"][0] == BOS_ID and len(batch["prompt_input_ids"]) == 3
    assert batch["completion_input_ids"][-1] == EOS_ID and len(batch["completion_input_ids"]) == 3


def _smpo_tokenize(tokenizer):
    features = {"prompt": "user prompt text", "chosen": " good answer", "rejected": " bad"}
    return tokenize_preference_row(
        features, tokenizer, max_prompt_length=None, max_completion_length=None, truncation_mode="keep_end"
    )


def test_smpo_tokenize_no_forced_bos_for_nominal_bos():
    out = _smpo_tokenize(NominalBosTokenizer())
    assert NOMINAL_BOS_ID not in out["prompt_input_ids"], f"phantom BOS injected: {out['prompt_input_ids']}"
    assert out["chosen_input_ids"][-1] == EOS_ID


def test_smpo_tokenize_bos_kept_for_postprocessing_tokenizer():
    out = _smpo_tokenize(PostProcBosTokenizer())
    assert out["prompt_input_ids"][0] == BOS_ID
    assert out["prompt_input_ids"].count(BOS_ID) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
