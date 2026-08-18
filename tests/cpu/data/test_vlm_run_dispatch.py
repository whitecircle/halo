#!/usr/bin/env python
"""CPU tests: modality is a property of the RUN, not of the checkpoint.

Every natively-multimodal family (Gemma 4, Qwen3.5/3.6) is a VLM by config while its text-only
recipes are ordinary text runs. ``is_vlm_run`` is the seam that separates the two: the checkpoint
must be multimodal AND the run must declare image data — ``images_field``, an image column
(:data:`VLM_IMAGE_COLUMNS`), or images embedded in the conversation's own content parts. Anything
else takes the text path, where packing and padding-free are legal.

The embedded-parts declaration is read twice: off the conversation column's Arrow schema (exact and
row-independent — a mixed dataset's image rows are in the schema of every row) and, for a column of
JSON strings that has no schema to read, off the first row. Behind both, every text renderer refuses
an image content part rather than rendering it as an unbacked placeholder.

Run: python tests/cpu/data/test_vlm_run_dispatch.py  (or pytest)
"""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset, DatasetDict, Features, Value
from datasets import Image as ImageFeature
from PIL import Image as PILImage
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from src.data import vlm as vlm_module
from src.data.collators.self_distill import SelfDistillTextCollator
from src.data.pipeline import conversation as conversation_module
from src.data.pipeline import preprocessing as preprocessing_module
from src.data.pipeline import row_processors as row_processors_module
from src.data.pipeline.conversation import (
    IMAGE_PART_PAYLOAD_KEY,
    IMAGE_PART_TYPE,
    conversation_carries_images,
)
from src.data.pipeline.preferences import apply_chat_template_to_preference_data, build_reward_preprocess_fn
from src.data.pipeline.rendered import render_generation_prompt
from src.data.pipeline.row_processors import (
    apply_chat_template_to_conversations,
    normalize_vlm_conversation,
    prepare_generative_row,
)
from src.data.vlm import (
    VLM_IMAGE_COLUMNS,
    dataset_declares_images,
    dataset_image_evidence,
    is_vlm_run,
    process_vlm_conversation,
)
from src.training.script_runner import enforce_text_path_padding_side
from tests.common.utils import load_script_module

# Real configs, not stand-ins: the verdict must follow what transformers says about the architecture.
VLM_CONFIG = CONFIG_MAPPING["gemma3"]()
TEXT_CONFIG = CONFIG_MAPPING["qwen3"]()

TEXT_TURNS = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
IMAGE_TURNS = [
    {"role": "user", "content": [{"type": "image", "image": "b64"}, {"type": "text", "text": "what is this?"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "a cat"}]},
]
# Parts-form text, the shape a mixed dataset's text rows carry (the same struct, image field unset).
TEXT_PARTS_TURNS = [
    {"role": "user", "content": [{"type": "text", "text": "what is this?"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "a cat"}]},
]


def _args(**overrides):
    return SimpleNamespace(**{"conversation_field": "prompt", "images_field": None, **overrides})


def _dataset(rows, extra_columns: dict | None = None) -> DatasetDict:
    data = {"prompt": rows, **(extra_columns or {})}
    return DatasetDict({"train": Dataset.from_dict(data), "test": Dataset.from_dict(data)})


def _script_module(relative_path: str):
    name = f"halo_test_{Path(relative_path).stem}_render"
    return load_script_module(f"scripts/training/{relative_path}", name)


def _mixed_dataset(image_rows: int = 1, text_rows: int = 9) -> DatasetDict:
    """A dataset whose image rows all sit past the first row — the shape the row probe cannot see."""
    rows = [{"prompt": TEXT_PARTS_TURNS}] * text_rows + [{"prompt": IMAGE_TURNS}] * image_rows
    split = Dataset.from_list(rows)
    return DatasetDict({"train": split, "test": split})


class _UnreachableTokenizer:
    """A tokenizer whose chat template must never run: reaching it is the bug under test.

    The renderers guard BEFORE templating, so every image-content rejection here is proof the row
    never became placeholder tokens — not merely that some later stage complained.
    """

    bos_token = None

    def apply_chat_template(self, messages, **kwargs):
        raise AssertionError("the text renderer templated an image-carrying conversation")

    def __call__(self, text, **kwargs):
        raise AssertionError("the text renderer tokenized an image-carrying conversation")


# --- the run verdict -----------------------------------------------------------------------------


def test_multimodal_checkpoint_with_text_data_is_a_text_run():
    """The regression: keying the dispatch on the checkpoint alone sent every Gemma 4 / Qwen3.5
    text recipe down the VLM path, which refuses packing outright."""
    assert not is_vlm_run(_args(), "google/gemma-4-26B-A4B-it", _dataset([TEXT_TURNS]), config=VLM_CONFIG)


def test_images_field_declares_a_vlm_run():
    args = _args(images_field="images")
    assert is_vlm_run(args, "Qwen/Qwen3.5-9B", _dataset([TEXT_TURNS], {"images": [[]]}), config=VLM_CONFIG)


def test_images_field_is_a_declaration_even_before_the_dataset_is_known():
    """The column guard in ``prepare_vlm_dataset`` owns a mistyped ``images_field`` — the dispatch
    must not quietly demote it to a text run and skip that guard entirely."""
    assert is_vlm_run(_args(images_field="pictures"), "Qwen/Qwen3.5-9B", _dataset([TEXT_TURNS]), config=VLM_CONFIG)


@pytest.mark.parametrize("column", VLM_IMAGE_COLUMNS)
def test_each_image_column_declares_a_vlm_run(column):
    """Every column the VLM path consumes as image payload — the raw ``images``/``image`` spellings
    and the preprocessed artifact's ``pixel_values`` — is a declaration on its own."""
    dataset = _dataset([TEXT_TURNS], {column: [[]]})
    assert is_vlm_run(_args(), "Qwen/Qwen3.5-9B", dataset, config=VLM_CONFIG)


def test_embedded_image_content_declares_a_vlm_run():
    """The SFT-VLM format with no separate column: images ride in the message content parts."""
    assert is_vlm_run(_args(), "Qwen/Qwen3.5-9B", _dataset([IMAGE_TURNS]), config=VLM_CONFIG)


def test_text_checkpoint_never_becomes_a_vlm_run():
    """The model class is the gate the data cannot open: a text-only architecture has no processor
    to run, so image data must fail at the VLM loader, not silently switch the data path."""
    dataset = _dataset([IMAGE_TURNS], {"images": [[]]})
    assert not is_vlm_run(_args(images_field="images"), "Qwen/Qwen3-8B", dataset, config=TEXT_CONFIG)


def test_no_dataset_yet_is_not_a_declaration():
    assert not is_vlm_run(_args(), "Qwen/Qwen3.5-9B", None, config=VLM_CONFIG)


def test_args_without_the_modality_knobs_still_resolve():
    """DPO's script args declare neither knob; ``getattr`` defaults must not turn that into a crash
    or into a spurious VLM verdict."""
    bare = SimpleNamespace()
    assert not is_vlm_run(bare, "Qwen/Qwen3.5-9B", _dataset([TEXT_TURNS]), config=VLM_CONFIG)
    assert is_vlm_run(bare, "Qwen/Qwen3.5-9B", _dataset([TEXT_TURNS], {"image": [None]}), config=VLM_CONFIG)


# --- the dataset-side declaration ----------------------------------------------------------------


def test_dataset_declares_images_accepts_a_single_split():
    assert dataset_declares_images(Dataset.from_dict({"images": [[]], "prompt": [TEXT_TURNS]}))
    assert not dataset_declares_images(Dataset.from_dict({"prompt": [TEXT_TURNS]}), "prompt")


def test_dataset_declares_images_needs_the_conversation_field_to_see_embedded_images():
    """Without the field name there is no column to probe — the caller must pass it."""
    dataset = _dataset([IMAGE_TURNS])
    assert not dataset_declares_images(dataset)
    assert dataset_declares_images(dataset, "prompt")


def test_empty_split_does_not_crash_the_probe():
    empty = DatasetDict({"train": Dataset.from_dict({"prompt": []})})
    assert not dataset_declares_images(empty, "prompt")
    assert dataset_image_evidence(empty) is None


def test_dataset_image_evidence_probes_every_column_and_names_the_declaration():
    """The field-less probe (for a consumer that never learns ``conversation_field``): an image
    column, or embedded parts under ANY column name, each named in the verdict; text is ``None``."""
    assert dataset_image_evidence(None) is None
    assert dataset_image_evidence(_dataset([TEXT_TURNS])) is None
    assert dataset_image_evidence(Dataset.from_dict({"input_ids": [[1, 2, 3]], "labels": [[1, 2, 3]]})) is None
    assert "['images']" in dataset_image_evidence(_dataset([TEXT_TURNS], {"images": [[]]}))
    assert "'conversation'" in dataset_image_evidence(Dataset.from_dict({"conversation": [IMAGE_TURNS]}))
    assert "'prompt'" in dataset_image_evidence(_mixed_dataset())
    assert "'prompt'" in dataset_image_evidence(Dataset.from_dict({"prompt": [json.dumps(IMAGE_TURNS)]}))


def test_json_string_conversations_are_probed_too():
    """Conversations stored as JSON strings render the same as native ones, so they must be read
    the same way here."""
    dataset = DatasetDict({"train": Dataset.from_dict({"prompt": [json.dumps(IMAGE_TURNS)]})})
    assert dataset_declares_images(dataset, "prompt")


# --- the mixed dataset: the schema, not a row window -----------------------------------------------


def test_mixed_dataset_whose_first_rows_are_text_is_still_a_vlm_run():
    """The gap a bigger row window cannot close: on a 10%-image dataset a first-row probe misses the
    image rows ~81% of the time (0.9**16 for TRL's own window), resolving to a TEXT run that crashes
    mid-map once the model is already on the GPUs. Arrow types the parts struct over the WHOLE split,
    so the image field is in the schema of every row — including the text ones."""
    assert dataset_declares_images(_mixed_dataset(), "prompt")
    assert is_vlm_run(_args(), "Qwen/Qwen3.5-9B", _mixed_dataset(), config=VLM_CONFIG)


def test_one_image_row_in_a_thousand_still_declares_the_run():
    """Row-independent means row-independent: the verdict must not thin out with the image ratio."""
    assert dataset_declares_images(_mixed_dataset(image_rows=1, text_rows=999), "prompt")


def test_parts_form_text_schema_is_not_a_declaration():
    """The other half of the contract: a ``type``/``text``-only parts struct is TEXT. Flipping it
    would route every parts-form text dataset to the VLM path, which refuses packing outright."""
    dataset = _dataset([TEXT_PARTS_TURNS, TEXT_PARTS_TURNS])
    assert not dataset_declares_images(dataset, "prompt")
    assert not is_vlm_run(_args(), "Qwen/Qwen3.5-9B", dataset, config=VLM_CONFIG)


def test_string_content_schema_is_not_a_declaration():
    """Plain-string content carries no parts struct at all."""
    assert not dataset_declares_images(_dataset([TEXT_TURNS]), "prompt")


def test_cast_image_feature_declares_images_under_any_name():
    """A dataset that casts its image payload to ``datasets.Image`` (the shape a hub VLM dataset
    ships) declares images even under a field name the pipeline does not read — the feature is the
    declaration."""
    image = PILImage.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
    features = Features(
        {
            "prompt": [
                {
                    "role": Value("string"),
                    "content": [{"type": Value("string"), "text": Value("string"), "picture": ImageFeature()}],
                }
            ]
        }
    )
    # Mixed, image row last: the row probe cannot reach it, and it does not read this field anyway.
    rows = [
        {"prompt": [{"role": "user", "content": [{"type": "text", "picture": None, "text": "hi"}]}]},
        {"prompt": [{"role": "user", "content": [{"type": "image", "picture": image, "text": ""}]}]},
    ]
    dataset = DatasetDict({"train": Dataset.from_list(rows, features=features)})
    assert dataset_declares_images(dataset, "prompt")


def test_empty_split_with_an_image_schema_declares_images():
    """The schema stands on its own: an empty split has no row to probe, and its declaration must
    still route the run (the split fills on the next shard)."""
    features = Features(
        {
            "prompt": [
                {
                    "role": Value("string"),
                    "content": [{"type": Value("string"), "text": Value("string"), "image": Value("string")}],
                }
            ]
        }
    )
    empty = DatasetDict({"train": Dataset.from_dict({"prompt": []}, features=features)})
    assert dataset_declares_images(empty, "prompt")


# --- the per-row backstop ------------------------------------------------------------------------


def test_conversation_carries_images_predicate():
    assert conversation_carries_images(IMAGE_TURNS)
    assert not conversation_carries_images(TEXT_TURNS)
    # Parts-form text is not image data.
    assert not conversation_carries_images([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
    assert not conversation_carries_images("not a conversation")
    assert not conversation_carries_images(None)


def test_text_renderer_refuses_an_image_content_part():
    """The shape neither dispatch line sees — a JSON-string conversation column, which has no Arrow
    struct to read and only its first row probed — must not become placeholder tokens with no pixels
    behind them."""
    with pytest.raises(ValueError, match="image content part"):
        apply_chat_template_to_conversations({"prompt": IMAGE_TURNS}, tokenizer=None, conversation_field="prompt")


def test_self_distill_text_collator_refuses_an_image_content_part():
    """``SelfDistillTextCollator`` renders directly instead of through the shared renderer, so the
    backstop has to be on it too — its shipped VLM configs declare no images column and reach the
    dispatch on the conversation column alone."""
    collator = SelfDistillTextCollator(
        tokenizer=SimpleNamespace(eos_token_id=0),
        conversation_field="prompt",
        hint_template="{answer}",
        model_config=None,
    )
    with pytest.raises(ValueError, match="image content part"):
        collator._render(IMAGE_TURNS, {})


@pytest.mark.parametrize("field", ["prompt", "chosen", "rejected"])
def test_preference_renderer_refuses_an_image_content_part(field):
    """DPO/SMPO/KTO text prep. DPO's script args declare no ``conversation_field``, so an
    image-carrying preference row reaches this renderer with the dispatch none the wiser — the exact
    failure class the backstop exists for."""
    image_content = [{"type": "image", "image": "b64"}, {"type": "text", "text": "q"}]
    row = {
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "a"}],
        "rejected": [{"role": "assistant", "content": "b"}],
    }
    row[field] = [{"role": row[field][0]["role"], "content": image_content}]
    with pytest.raises(ValueError, match="image content part"):
        apply_chat_template_to_preference_data(row, _UnreachableTokenizer())


def test_reward_renderer_refuses_an_image_content_part():
    """Bradley-Terry reward prep renders prompt+chosen and prompt+rejected itself."""
    fn = build_reward_preprocess_fn(_UnreachableTokenizer(), max_length=64)
    with pytest.raises(ValueError, match="image content part"):
        fn(
            {
                "prompt": [IMAGE_TURNS[:1]],
                "chosen": [[{"role": "assistant", "content": "a"}]],
                "rejected": [[{"role": "assistant", "content": "b"}]],
            }
        )


def test_generation_prompt_renderer_refuses_an_image_content_part():
    """The GRPO family's shared prompt renderer (RLVR, environmental): a rollout has no pixels."""
    with pytest.raises(ValueError, match="image content part"):
        render_generation_prompt(_UnreachableTokenizer(), IMAGE_TURNS)


def test_generative_row_renderer_refuses_an_image_content_part():
    """The eval-generation prep shared by offline GRPO / DPO / SMPO."""
    with pytest.raises(ValueError, match="image content part"):
        prepare_generative_row({"prompt": IMAGE_TURNS}, _UnreachableTokenizer(), max_length=64)


@pytest.mark.parametrize("field", ["prompt", "completions"])
def test_offline_grpo_renderer_refuses_an_image_content_part(field):
    """``scripts/training/offline_grpo.py`` templates prompt and completions itself."""
    module = _script_module("offline_grpo.py")
    row = {"prompt": TEXT_TURNS[:1], "completions": [[{"role": "assistant", "content": "a"}]]}
    if field == "prompt":
        row["prompt"] = IMAGE_TURNS[:1]
    else:
        row["completions"] = [[{"role": "assistant", "content": [{"type": "image", "image": "b64"}]}]]
    with pytest.raises(ValueError, match="image content part"):
        module.build_chat_template_row_fn(_UnreachableTokenizer(), None)(row)


def test_classification_renderer_refuses_an_image_content_part():
    """``scripts/training/classification.py`` templates its ``prompt`` conversation itself."""
    module = _script_module("classification.py")
    with pytest.raises(ValueError, match="image content part"):
        module.tokenize_classification_row(
            {"prompt": IMAGE_TURNS},
            tokenizer=_UnreachableTokenizer(),
            max_length=64,
            label_to_id={},
            is_multi_label=False,
        )


# --- one spelling of the image part ----------------------------------------------------------------


def test_every_image_part_site_reads_the_shared_constants(monkeypatch):
    """Detector, extractor, images-column merge and schema probe must key on ONE spelling.

    Re-spelling the constants in every module that imports them moves all four together — a site
    that re-inlined the literal instead would keep matching the old spelling and fail here. That is
    the failure that matters: a detector and an extractor which disagree read as "this row has no
    images", which is exactly the silent placeholder-training the guard exists to prevent.
    """
    alt_type, alt_key = "picture", "picture_payload"
    for module in (conversation_module, vlm_module, row_processors_module):
        monkeypatch.setattr(module, "IMAGE_PART_TYPE", alt_type)
        monkeypatch.setattr(module, "IMAGE_PART_PAYLOAD_KEY", alt_key)
    # The offline preprocessor extracts through process_vlm_conversation rather than walking the
    # parts itself, so it has no spelling of its own; a re-inlined extractor would re-import these.
    assert not hasattr(preprocessing_module, "IMAGE_PART_TYPE"), (
        "preprocessing re-imported the image-part constants — it must extract via process_vlm_conversation"
    )

    image = PILImage.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
    respelled = [{"role": "user", "content": [{"type": alt_type, alt_key: image}]}]

    assert conversation_carries_images(respelled), "the guard predicate ignored the constants"
    assert not conversation_carries_images(IMAGE_TURNS), "the guard predicate hardcodes the old spelling"

    history, images = process_vlm_conversation([dict(m) for m in respelled])
    assert len(images) == 1, "the extractor ignored the constants"
    assert history[0]["content"] == [{"type": alt_type}], "the placeholder hardcodes the old spelling"
    shipped = [{"role": "user", "content": [{"type": IMAGE_PART_TYPE, IMAGE_PART_PAYLOAD_KEY: image}]}]
    assert process_vlm_conversation(shipped)[1] == [], "the extractor hardcodes the old spelling"

    merged = normalize_vlm_conversation([{"role": "user", "content": [{"type": alt_type}]}], images=[image])
    assert merged[0]["content"][0][alt_key] is image, "the images-column merge ignored the constants"

    schema_split = Dataset.from_list([{"prompt": [{"role": "user", "content": [{"type": alt_type, alt_key: "b64"}]}]}])
    assert dataset_declares_images(DatasetDict({"train": schema_split}), "prompt"), (
        "the Arrow-schema probe ignored the constants"
    )
    old_split = Dataset.from_list([{"prompt": TEXT_PARTS_TURNS}, {"prompt": IMAGE_TURNS}])
    assert not dataset_declares_images(DatasetDict({"train": old_split}), "prompt"), (
        "the Arrow-schema probe hardcodes the old spelling"
    )


def test_no_module_re_inlines_the_image_part_spelling():
    """The one image-part site a unit test cannot reach — the offline VLM preprocessor's extractor,
    which needs a live processor — plus every future one. Scoped to the pipeline package, which owns
    the part shape; the ``image``/``images`` DATASET COLUMN names are a different contract
    (:data:`VLM_IMAGE_COLUMNS`) and are spelled there."""
    inlined = (
        re.compile(r'"type"\s*\)\s*==\s*"image"'),
        re.compile(r'"type"\s*:\s*"image"'),
        re.compile(r'\.get\(\s*"image"\s*\)'),
        re.compile(r'\[\s*"image"\s*\]\s*='),
    )
    package = Path(conversation_module.__file__).parent
    offenders = [
        f"{path.name}:{index + 1}: {line.strip()}"
        for path in sorted(package.glob("*.py"))
        if path.name != Path(conversation_module.__file__).name  # the constants' home
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines())
        if any(pattern.search(line) for pattern in inlined)
    ]
    assert not offenders, "image-part shape re-inlined instead of read from the constants:\n" + "\n".join(offenders)


# --- the text path's padding side ----------------------------------------------------------------


def test_text_path_forces_right_padding():
    """A VLM processor's tokenizer keeps the checkpoint's side (left for Gemma 4 / GLM-4.7-Flash);
    the text load normalizes it and this path never did. The packing collator refuses a
    left-padding tokenizer outright, so the newly-reachable text run would die at collator build."""
    tokenizer = SimpleNamespace(padding_side="left")
    enforce_text_path_padding_side(tokenizer, False)
    assert tokenizer.padding_side == "right"


def test_vlm_run_keeps_the_processor_padding_side():
    """VLM batches pad through the processor, not the text collators — flipping its side would
    change every existing VLM run."""
    tokenizer = SimpleNamespace(padding_side="left")
    enforce_text_path_padding_side(tokenizer, True)
    assert tokenizer.padding_side == "left"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
