#!/usr/bin/env python3
"""The data pipeline's shared seams, pinned so a second copy cannot reappear beside them.

Each twin below used to exist twice, and every duplicate was silent: a change landing in one arm
left the other rendering, splitting or masking a row differently. The tests are mutation-shaped —
they break the SHARED implementation and require every consumer to see it, which is exactly what a
re-inlined copy would survive.

Run: pytest tests/cpu/data/test_pipeline_single_sourcing.py
"""

import ast
import base64
import inspect
import io
import logging
import pathlib
import sys

import numpy as np
import pytest
import torch
from datasets import Dataset
from PIL import Image

from src.data.collators.self_distill import SelfDistillTextCollator
from src.data.collators.vlm import VLMDataCollator
from src.data.pipeline import preferences, preprocessing, rendered, row_processors
from src.data.pipeline.preferences import render_vlm_preference_row, split_vlm_preference_row
from src.data.pipeline.preprocessed_metadata import PreprocessingConfig
from src.data.pipeline.preprocessing import tokenize_vlm_dataset
from src.data.pipeline.processing import filter_by_length
from src.data.pipeline.rendered import render_conversation
from src.data.pipeline.row_processors import (
    apply_chat_template_to_conversations,
    build_vlm_history,
    create_llm_processor,
    create_vlm_processor,
)
from src.data.pipeline.vlm_dataset import _VLM_SIGNATURE_COLUMNS, vlm_map_features
from src.data.vlm import VLM_OUTPUT_COLUMNS
from src.trainers.preference.smpo import tokenize_vlm_preference_row
from tests.common.vlm_fakes import FakeVLMProcessorBase, FakeVLMTokenizer


@pytest.fixture(autouse=True)
def _isolated_datasets_cache(tmp_path, monkeypatch):
    """A fresh coordinated-op cache per test: the map fingerprint cannot see a fake processor, so a
    stale hit would replay another test's bake instead of running the one under test."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets"))


_PROMPT = [{"role": "user", "content": "describe"}]
_CHOSEN = [{"role": "assistant", "content": "a cat"}]
_REJECTED = [{"role": "assistant", "content": "a dog"}]


def _image() -> Image.Image:
    return Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))


def _image_data_uri() -> str:
    """A PNG as a base64 data URI — the image spelling Arrow can carry inside a conversation column
    (a bare PIL object makes ``Dataset.from_list`` refuse the mixed content struct)."""
    buffer = io.BytesIO()
    _image().save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class _RenderTokenizer:
    """Records what the shared renderer handed the chat template."""

    bos_token = "<bos>"
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, conversation, **kwargs):
        self.calls.append({"conversation": conversation, **kwargs})
        return "<bos>" + " ".join(str(message["content"]) for message in conversation)

    def get_vocab(self):
        return {}

    def __call__(self, text, **kwargs):
        tokens = text.split()
        return {"input_ids": list(range(2, 2 + len(tokens))), "attention_mask": [1] * len(tokens)}


# --- the VLM preference prologue (SMPO ⧸ reward reward-map) ------------------------------------


def test_both_vlm_preference_routes_go_through_the_shared_prologue(monkeypatch):
    """Break the ONE image-extraction call the shared splitter makes; both routes must fail on it.

    A route that kept its own prologue would import ``process_vlm_conversation`` from the VLM leaf
    itself and sail past this patch — which is precisely how the two copies drifted apart before.
    """

    class _SharedPrologueReached(Exception):
        pass

    def _boom(*_args, **_kwargs):
        raise _SharedPrologueReached

    monkeypatch.setattr(preferences, "process_vlm_conversation", _boom)
    row = {"prompt": _PROMPT, "chosen": _CHOSEN, "rejected": _REJECTED}
    processor = FakeVLMProcessorBase()

    with pytest.raises(_SharedPrologueReached):
        render_vlm_preference_row(dict(row), processor)
    with pytest.raises(_SharedPrologueReached):
        tokenize_vlm_preference_row(
            dict(row),
            processor,
            max_prompt_length=None,
            max_completion_length=None,
            truncation_mode="keep_end",
        )


@pytest.mark.parametrize("column", ["images", "image"])
def test_the_shared_prologue_reads_both_raw_image_column_spellings(column):
    """``images`` and its singular alias both carry a run's pixels; the pair is read off
    VLM_RAW_IMAGE_COLUMNS, so neither route can support one spelling and not the other."""
    row = {"prompt": _PROMPT, "chosen": _CHOSEN, "rejected": _REJECTED, column: [_image()]}
    _prompt_history, images, completions = split_vlm_preference_row(row, "probe")
    assert len(images) == 1, f"the {column!r} column did not reach the prompt conversation"
    assert set(completions) == {"chosen", "rejected"}


def test_the_shared_prologue_refuses_images_inside_a_completion():
    """Images live in the prompt both sides share; one inside a completion would give the two halves
    of a pair different pixels."""
    row = {
        "prompt": _PROMPT,
        "chosen": [{"role": "assistant", "content": [{"type": "image", "image": _image()}]}],
        "rejected": _REJECTED,
    }
    with pytest.raises(ValueError, match="images inside the 'chosen' completion"):
        split_vlm_preference_row(row, "VLM probe row")


# --- the text chat render (SFT row map ⧸ SDPG text collator) ------------------------------------


def test_both_text_renderers_go_through_the_shared_render(monkeypatch):
    """Break the image backstop inside ``render_conversation``; both text renderers must fail on it.

    The SDPG collator used to re-implement the render (fold + template kwargs + backstop) line for
    line, so a fold-policy change reached SFT alone. It now has to route here.
    """

    class _SharedRenderReached(Exception):
        pass

    def _boom(*_args, **_kwargs):
        raise _SharedRenderReached

    monkeypatch.setattr(rendered, "reject_image_content", _boom)
    tokenizer = _RenderTokenizer()
    row = {"messages": _PROMPT}

    with pytest.raises(_SharedRenderReached):
        apply_chat_template_to_conversations(row, tokenizer, conversation_field="messages")
    collator = SelfDistillTextCollator(tokenizer=tokenizer, hint_template="hint {answer}")
    with pytest.raises(_SharedRenderReached):
        collator._render(_PROMPT, row)


def test_the_shared_render_applies_the_system_fold_to_both_renderers():
    """The fold is the policy that used to live twice: prove it reaches the collator's own render."""
    tokenizer = _RenderTokenizer()
    row = {"messages": _PROMPT}
    collator = SelfDistillTextCollator(
        tokenizer=tokenizer,
        hint_template="hint {answer}",
        system_prompt="be terse",
        model_supports_system_role=False,
    )
    collator._render(list(_PROMPT), row)
    folded = tokenizer.calls[-1]["conversation"]
    assert [m["role"] for m in folded] == ["user"], "the collator skipped the system fold"
    assert folded[0]["content"].startswith("be terse"), folded[0]["content"]


def test_the_shared_render_keeps_a_template_emitted_bos():
    """No unconditional BOS strip: ``tokenize_rendered`` owns the specials contract, and stripping
    here silently deletes BOS for families whose template emits it while their post-processor adds
    nothing (gemma-4) — nothing downstream puts it back."""
    tokenizer = _RenderTokenizer()
    text = render_conversation(tokenizer, list(_PROMPT), {}, conversation_field="messages")
    assert text.startswith("<bos>"), text
    assert apply_chat_template_to_conversations({"messages": _PROMPT}, tokenizer).startswith("<bos>")


# --- one declaration per schema ------------------------------------------------------------------


def test_the_vlm_map_output_is_one_declaration():
    """Row processor, pinned Arrow schema and collator signature list must name the SAME columns.

    They were three hand-kept lists: a column added to the row processor had to be edited into two
    more or the map stripped it before collation.
    """
    row = {"messages": [{"role": "user", "content": "hi"}]}
    emitted = set(create_vlm_processor()(row))
    assert emitted == set(vlm_map_features()), "the row processor and the pinned schema disagree"
    assert emitted <= set(_VLM_SIGNATURE_COLUMNS), "the map writes a column the signature list strips"
    assert "text" not in emitted, "the dead 'text' payload is back"


def test_the_preprocessed_vision_key_refusal_reads_the_declared_schema():
    """The offline bake refuses processor keys its stored schema cannot hold — the set is read off
    VLM_OUTPUT_COLUMNS, not hand-typed beside it.

    ``pixel_values_shape`` is the proof: the schema stores it, the hand-typed copy did not name it,
    so a processor emitting it used to be refused as unstorable. Widening the schema must widen the
    refusal with it, and a re-inlined literal would not move.
    """

    def _emitting(extra_key):
        class _ExtraKeyProcessor(FakeVLMProcessorBase):
            def __call__(self, text, images=None, **kwargs):
                batch = dict(super().__call__(text, images=images, **kwargs))
                batch[extra_key] = torch.ones_like(batch["input_ids"])
                return batch

        return _ExtraKeyProcessor(FakeVLMTokenizer())

    config = PreprocessingConfig(
        model_name_or_path="fake/vlm",
        max_length=64,
        conversation_field="conversation",
        is_vlm=True,
        num_proc=1,
    )
    row = {
        "conversation": [
            {
                "role": "user",
                "content": [{"type": "image", "image": _image_data_uri()}, {"type": "text", "text": "hi"}],
            },
            # Parts-form on BOTH turns: Arrow refuses a content column mixing a list and a string.
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        ]
    }
    assert "pixel_values_shape" in VLM_OUTPUT_COLUMNS and "pixel_attention_mask" not in VLM_OUTPUT_COLUMNS

    with pytest.raises(NotImplementedError, match="pixel_attention_mask"):
        tokenize_vlm_dataset(Dataset.from_list([row]), _emitting("pixel_attention_mask"), config, split_name="train")

    tokenized = tokenize_vlm_dataset(
        Dataset.from_list([row]), _emitting("pixel_values_shape"), config, split_name="train"
    )
    assert len(tokenized) == 1, "a key the stored schema DOES hold was refused as unstorable"


def test_the_non_completions_vlm_bake_routes_through_the_shared_label_builder(monkeypatch):
    """The bake's no-completion-masking arm re-implemented "mask these token ids out of labels"
    inline. Replacing the shared builder must change what it bakes; an inline copy would not see it."""

    def _sentinel_labels(input_ids, *_args, **_kwargs):
        return torch.full_like(input_ids, -7)

    monkeypatch.setattr(preprocessing, "build_completion_only_labels", _sentinel_labels)
    config = PreprocessingConfig(
        model_name_or_path="fake/vlm",
        max_length=64,
        conversation_field="conversation",
        is_vlm=True,
        num_proc=1,
    )
    row = {"conversation": [{"role": "user", "content": "hi there"}, {"role": "assistant", "content": "ok"}]}
    tokenized = tokenize_vlm_dataset(
        Dataset.from_list([row]), FakeVLMProcessorBase(FakeVLMTokenizer()), config, split_name="train"
    )
    assert set(tokenized[0]["labels"]) == {-7}, "the bake built its labels somewhere other than the shared builder"


def test_the_vlm_bake_masks_image_tokens_without_erasing_a_real_eos():
    """The non-completions bake routes through the shared ``extra_ignore_token_ids`` path, with the
    row's own mask passed explicitly: its value-based pad fallback would otherwise erase a real EOS
    on every family whose pad_token_id IS its eos_token_id (Qwen's default)."""

    class _EosTokenizer(FakeVLMTokenizer):
        pad_token_id = 1  # pad == eos, the trap
        eos_token_id = 1

    class _EosProcessor(FakeVLMProcessorBase):
        def encode_batch(self, text, max_length=None):
            batch = super().encode_batch(text, max_length=max_length)
            # last real position is the turn terminator, and 99 is the image pad id
            batch["input_ids"][:, 0] = 99
            batch["input_ids"][:, -1] = self.tokenizer.eos_token_id
            return batch

    config = PreprocessingConfig(
        model_name_or_path="fake/vlm",
        max_length=64,
        conversation_field="conversation",
        is_vlm=True,
        num_proc=1,
    )
    row = {"conversation": [{"role": "user", "content": "hi there"}, {"role": "assistant", "content": "ok"}]}
    tokenized = tokenize_vlm_dataset(
        Dataset.from_list([row]), _EosProcessor(_EosTokenizer()), config, split_name="train"
    )
    labels = tokenized[0]["labels"]
    input_ids = tokenized[0]["input_ids"]
    assert labels[0] == -100, "the image placeholder token was trained as text"
    assert labels[-1] == input_ids[-1] == 1, "the real EOS was erased by a pad-value mask"


# --- the layering the moved sentinels bought -----------------------------------------------------


def test_row_processors_imports_nothing_from_the_coordinated_map_module():
    """The row-shape module is a leaf under the coordinated-map module, not a peer of it: the
    sentinel it emits and the predicate that drops it live with the rows, so ``processing`` can
    import the predicate without either module reaching back."""
    tree = ast.parse(pathlib.Path(row_processors.__file__).read_text())
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert "src.data.pipeline.processing" not in imported, sorted(imported)


def test_the_row_map_factories_expose_no_dead_injection_knobs():
    """``none_example`` and ``process_vlm_conversation_fn`` were caller-supplied overrides no
    production caller ever set — one handed back the factory's own default, the other a stub only
    tests passed. A knob nothing sets cannot be verified by a run, so it must not exist."""
    assert "none_example" not in inspect.signature(create_llm_processor).parameters
    for fn in (create_vlm_processor, build_vlm_history):
        assert "process_vlm_conversation_fn" not in inspect.signature(fn).parameters


# --- one drop-rate reporter ----------------------------------------------------------------------


def test_filter_by_length_reports_through_the_shared_rejection_reporter(caplog):
    """It is a drop-and-continue filter like every other, so it owes the high-rejection WARNING:
    a max_length that removes most of the corpus is usually a config bug, and its own private INFO
    line let that scroll by. It also returns the dataset alone — both callers dropped the stats."""

    class _CharTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(len(text))), "attention_mask": [1] * len(text)}

    dataset = Dataset.from_dict({"text": ["ab", *["a very long document" for _ in range(9)]]})
    with caplog.at_level(logging.WARNING, logger="src.data.pipeline.processing"):
        filtered = filter_by_length(dataset, max_length=4, tokenizer=_CharTokenizer(), num_proc=1)

    assert isinstance(filtered, Dataset), "filter_by_length must return the dataset, not a stats tuple"
    assert len(filtered) == 1
    assert "rejection rate this high" in caplog.text, caplog.text


# --- the completion marker is required where the pair is accepted --------------------------------


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: VLMDataCollator(FakeVLMProcessorBase(), FakeVLMTokenizer(), 64, train_on_completions_only=True),
            id="vlm-collator",
        ),
        pytest.param(
            lambda: SelfDistillTextCollator(
                tokenizer=_RenderTokenizer(), hint_template="h {answer}", train_on_completions_only=True
            ),
            id="self-distill-collator",
        ),
    ],
)
def test_completion_masking_without_a_marker_is_refused_at_construction(build):
    """The refusal used to fire from the label builder — at the FIRST BATCH, after the model load and
    the whole dataset map. Nothing about it needs a batch: the pair is knowably unserviceable where
    it is accepted."""
    with pytest.raises(ValueError, match="requires assistant_message_template"):
        build()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
