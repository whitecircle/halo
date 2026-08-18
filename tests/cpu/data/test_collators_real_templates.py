#!/usr/bin/env python3
"""Completion-only masking against REAL tokenizers + ORIGINAL chat templates.

The synthetic tests in ``test_collators.py`` use a fake tokenizer. This file exercises the production
masking on the actual shipped chat templates of several model families, where the assistant-turn
terminator is frequently NOT ``tokenizer.eos_token_id``:

  - Qwen3 / Qwen3.5 — turns end with ``<|im_end|>`` (Qwen3.5 exposes eos only via ``text_config``);
  - GLM-4 — no per-turn ``<|endoftext|>``; turns delimited by role markers in ``config.eos_token_id``;
  - Gemma-3 — turns end with ``<end_of_turn>`` (106), distinct from ``tokenizer.eos_token_id`` (1).

For each model it asserts the shared builder and every completion collator (standard, packing,
flattening/padding-free) unmask the assistant content and mask the user content, and that a
single-eos search masks the whole instance for GLM/Gemma.

Run: python tests/cpu/data/test_collators_real_templates.py
Requires the model tokenizers/configs in the HF cache (offline-friendly).
"""

import os

import pytest
from transformers import AutoConfig

from src.data.collators.completions_only import DataCollatorForCompletionOnlyLM
from src.data.collators.packing import (
    DataCollatorForCompletionOnlyLMWithPacking,
    DataCollatorWithFlatteningAndCompletionMask,
)
from src.data.spans import build_completion_only_labels, resolve_eos_token_ids
from tests.common.tokenizers import load_cached_tokenizer

# Distinctive content so decode-based assertions are unambiguous after templating/tokenization.
U1, A1 = "Question alpha numero uno", "Response bravo the first answer"
U2, A2 = "Question charlie numero dos", "Response delta the second answer"
ASST_MARKERS = {  # the assistant generation marker per family (matches the production YAML configs)
    "Qwen/Qwen3-0.6B": "<|im_start|>assistant\n",
    "Qwen/Qwen3.5-2B": "<|im_start|>assistant\n",
    "zai-org/GLM-4.7-Flash": "<|assistant|>",
    "google/gemma-3-4b-it": "<start_of_turn>model\n",
}
# Families whose native template leaves the FINAL assistant turn unterminated need the SFT template,
# or the completion collators cannot train that turn (Qwen3/Gemma terminate every turn natively).
SFT_TEMPLATE_FILES = {
    "zai-org/GLM-4.7-Flash": "jinja_templates/glm-chat.jinja",
}
# Families whose assistant turns end in a token that is NOT ``tokenizer.eos_token_id`` — exactly the
# ones a single-eos search gets wrong.
NON_EOS_TERMINATED = ("zai-org/GLM-4.7-Flash", "google/gemma-3-4b-it")
# One family per template shape (im_end / role marker / end_of_turn); Qwen3.5 shares Qwen3's.
COLLATOR_FAMILIES = ("Qwen/Qwen3-0.6B", "zai-org/GLM-4.7-Flash", "google/gemma-3-4b-it")


def _apply_sft_template(tok, model_id):
    """Use the framework SFT jinja template when the native template lacks a final-turn terminator."""
    path = SFT_TEMPLATE_FILES.get(model_id)
    if path:
        with open(os.path.join(os.path.dirname(__file__), "../../..", path)) as f:
            tok.chat_template = f.read()


CONVO = [
    {"role": "user", "content": U1},
    {"role": "assistant", "content": A1},
    {"role": "user", "content": U2},
    {"role": "assistant", "content": A2},
]


def _load(model_id):
    """Tokenizer + config for ``model_id``, skipping the test when the HF cache lacks them."""
    tok = load_cached_tokenizer(model_id, trust_remote_code=True)
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
    except OSError:
        pytest.skip(f"config for {model_id} is not in the local HF cache")
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
    """Shared builder: assistant content unmasked, user content masked, not-all-masked."""
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
    # These families terminate turns with a token that is NOT tokenizer.eos.
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
