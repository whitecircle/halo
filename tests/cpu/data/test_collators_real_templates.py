#!/usr/bin/env python3
"""Completion-only masking against real tokenizers and original chat templates, where the turn terminator is often
not ``tokenizer.eos_token_id`` (Qwen3's ``<|im_end|>``, GLM-4's role markers, Gemma-3's ``<end_of_turn>``).
Requires the model tokenizers and configs in the HF cache."""

import pytest

from halo.data.collators.packing import (
    DataCollatorForCompletionOnlyLMWithPacking,
    DataCollatorWithFlatteningAndCompletionMask,
)
from halo.data.collators.padded import DataCollatorForCompletionOnlyLM
from halo.data.completion_spans import build_completion_only_labels
from halo.data.turn_markers import resolve_eos_token_ids
from tests.common.models import GEMMA3_4B_IT, GLM4_FLASH, QWEN3_0_6B, QWEN3_5_2B
from tests.common.repo import REPO_ROOT
from tests.common.tokenizers import load_cached_config, load_cached_tokenizer

U1, A1 = "Question alpha numero uno", "Response bravo the first answer"
U2, A2 = "Question charlie numero dos", "Response delta the second answer"
ASST_MARKERS = {  # the assistant generation marker per family (matches the production YAML configs)
    QWEN3_0_6B: "<|im_start|>assistant\n",
    QWEN3_5_2B: "<|im_start|>assistant\n",
    GLM4_FLASH: "<|assistant|>",
    GEMMA3_4B_IT: "<start_of_turn>model\n",
}
# Families whose native template leaves the FINAL assistant turn unterminated need the SFT template,
# or the completion collators cannot train that turn (Qwen3/Gemma terminate every turn natively).
SFT_TEMPLATE_FILES = {
    GLM4_FLASH: "jinja-templates/glm/glm-chat.jinja",
}
# Families whose assistant turns end in a token other than ``tokenizer.eos_token_id``, the ones a single-eos search
# gets wrong.
NON_EOS_TERMINATED = (GLM4_FLASH, GEMMA3_4B_IT)
# One family per template shape (im_end / role marker / end_of_turn); Qwen3.5 shares Qwen3's.
COLLATOR_FAMILIES = (QWEN3_0_6B, GLM4_FLASH, GEMMA3_4B_IT)


def _apply_sft_template(tok, model_id):
    """Use the framework SFT jinja template when the native template lacks a final-turn terminator."""
    path = SFT_TEMPLATE_FILES.get(model_id)
    if path:
        tok.chat_template = (REPO_ROOT / path).read_text()


CONVO = [
    {"role": "user", "content": U1},
    {"role": "assistant", "content": A1},
    {"role": "user", "content": U2},
    {"role": "assistant", "content": A2},
]


def _load(model_id):
    tok = load_cached_tokenizer(model_id, trust_remote_code=True)
    cfg = load_cached_config(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # GLM-4.7-Flash/Gemma default to left padding, which the packing collator rejects.
    tok.padding_side = "right"
    return tok, cfg


def _render_ids(tok, model_id):
    text = tok.apply_chat_template(CONVO, tokenize=False, add_generation_prompt=False)
    marker = ASST_MARKERS[model_id]
    assert marker in text, f"{model_id}: assistant marker {marker!r} not in rendered original template"
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    return ids, marker


def _unmasked_text(tok, ids, labels):
    keep = ids[0][labels[0] != -100]
    return tok.decode(keep)


def _check_masking(tok, cfg, model_id):
    ids, marker = _render_ids(tok, model_id)
    eos_set = resolve_eos_token_ids(tok, cfg)
    labels = build_completion_only_labels(ids, tok, marker, train_on_completions_only=True, eos_token_ids=eos_set)
    n_unmasked = int((labels != -100).sum())
    assert n_unmasked > 0, f"{model_id}: completion masking masked the WHOLE instance"
    decoded = _unmasked_text(tok, ids, labels)
    assert "bravo" in decoded and "delta" in decoded, f"{model_id}: assistant content not unmasked: {decoded!r}"
    assert "alpha" not in decoded and "charlie" not in decoded, (
        f"{model_id}: user content leaked into labels: {decoded!r}"
    )
    return ids, marker, eos_set, n_unmasked


@pytest.mark.parametrize("model_id", list(ASST_MARKERS))
def test_builder_masks_every_family(model_id):
    tok, cfg = _load(model_id)
    _apply_sft_template(tok, model_id)
    _check_masking(tok, cfg, model_id)


@pytest.mark.parametrize("model_id", NON_EOS_TERMINATED)
def test_single_eos_search_would_be_wrong(model_id):
    tok, cfg = _load(model_id)
    ids, marker = _render_ids(tok, model_id)
    old_set = frozenset({tok.eos_token_id})
    old_labels = build_completion_only_labels(ids, tok, marker, train_on_completions_only=True, eos_token_ids=old_set)
    new_set = resolve_eos_token_ids(tok, cfg)
    new_labels = build_completion_only_labels(ids, tok, marker, train_on_completions_only=True, eos_token_ids=new_set)
    old_n = int((old_labels != -100).sum())
    new_n = int((new_labels != -100).sum())
    assert new_n > 0, f"{model_id}: resolved-set masking unexpectedly empty"
    decoded_new = _unmasked_text(tok, ids, new_labels)
    assert "bravo" in decoded_new, f"{model_id}: assistant content missing under resolved set"
    assert "alpha" not in decoded_new and "charlie" not in decoded_new
    # A single-eos search therefore masks everything or runs past into user content.
    decoded_old = _unmasked_text(tok, ids, old_labels)
    old_is_wrong = (old_n == 0) or ("alpha" in decoded_old) or ("charlie" in decoded_old)
    assert old_is_wrong, (
        f"{model_id}: expected a single-eos search to be wrong (empty or user-leaking); "
        f"old_n={old_n} decoded_old={decoded_old!r}"
    )


@pytest.mark.parametrize("model_id", COLLATOR_FAMILIES)
def test_every_completion_collator_agrees_with_the_builder(model_id):
    tok, cfg = _load(model_id)
    _apply_sft_template(tok, model_id)
    ids, marker, eos_set, _ = _check_masking(tok, cfg, model_id)
    example = {"input_ids": ids[0].tolist(), "attention_mask": [1] * ids.shape[1]}

    std = DataCollatorForCompletionOnlyLM(response_prompt_template=marker, tokenizer=tok, eos_token_ids=eos_set)
    std_batch = std([example])
    std_decoded = _unmasked_text(tok, std_batch["input_ids"], std_batch["labels"])
    assert "bravo" in std_decoded and "delta" in std_decoded
    assert "alpha" not in std_decoded and "charlie" not in std_decoded

    pk = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=marker, tokenizer=tok, eos_token_ids=eos_set
    )
    pk_batch = pk([example])
    pk_keep = pk_batch["input_ids"][pk_batch["labels"] != -100]
    pk_decoded = tok.decode(pk_keep)
    assert "bravo" in pk_decoded and "delta" in pk_decoded
    assert "alpha" not in pk_decoded and "charlie" not in pk_decoded

    fl = DataCollatorWithFlatteningAndCompletionMask(
        response_prompt_template=marker, tokenizer=tok, eos_token_ids=eos_set
    )
    fl_batch = fl([example])
    fl_keep = fl_batch["input_ids"][fl_batch["labels"] != -100]
    fl_decoded = tok.decode(fl_keep)
    assert "bravo" in fl_decoded and "delta" in fl_decoded
    assert "alpha" not in fl_decoded and "charlie" not in fl_decoded


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
