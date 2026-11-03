#!/usr/bin/env python
"""Modality is a property of the run: the checkpoint must be multimodal and the run must declare image data
(``images_field``, an image column, or embedded content parts), read off the Arrow schema or, for JSON strings,
the first row. Every text renderer refuses an image part, so it cannot become an unbacked placeholder."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset, DatasetDict, Features, Value
from datasets import Image as ImageFeature
from PIL import Image as PILImage
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from halo.data import conversation as conversation_module
from halo.data import vlm_detection as vlm_detection_module
from halo.data import vlm_render as vlm_render_module
from halo.data.chat_render import render_generation_prompt
from halo.data.collators.self_distill import SelfDistillTextCollator
from halo.data.conversation import (
    IMAGE_PART_PAYLOAD_KEY,
    IMAGE_PART_TYPE,
    conversation_carries_images,
)
from halo.data.pipeline import vlm_rows as vlm_rows_module
from halo.data.pipeline.preferences import render_preference_row
from halo.data.pipeline.text_rows import prepare_generative_row, render_row_conversation
from halo.data.pipeline.vlm_rows import normalize_vlm_conversation
from halo.data.preprocessed import tokenization as tokenization_module
from halo.data.vlm_detection import (
    VLM_IMAGE_COLUMNS,
    dataset_declares_images_across_ranks,
    dataset_image_evidence,
    is_vlm_run_across_ranks,
)
from halo.data.vlm_render import extract_conversation_images
from halo.entrypoints import data_setup
from halo.entrypoints.data_setup import resolve_vlm_run
from halo.entrypoints.model_setup import apply_text_path_padding_side
from tests.common.repo import load_script_module

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
    """A dataset whose image rows all sit past the first row, which a row probe cannot see."""
    rows = [{"prompt": TEXT_PARTS_TURNS}] * text_rows + [{"prompt": IMAGE_TURNS}] * image_rows
    split = Dataset.from_list(rows)
    return DatasetDict({"train": split, "test": split})


class _UnreachableTokenizer:
    """Reaching its chat template is the bug: the renderers guard before templating."""

    bos_token = None

    def apply_chat_template(self, messages, **kwargs):
        raise AssertionError("the text renderer templated an image-carrying conversation")

    def __call__(self, text, **kwargs):
        raise AssertionError("the text renderer tokenized an image-carrying conversation")


def test_multimodal_checkpoint_with_text_data_is_a_text_run():
    """Keying on the checkpoint alone would send every Gemma 4 / Qwen3.5 text recipe down the VLM path, which
    refuses packing."""
    assert not is_vlm_run_across_ranks(_args(), "google/gemma-4-26B-A4B-it", _dataset([TEXT_TURNS]), config=VLM_CONFIG)


def test_images_field_declares_a_vlm_run():
    args = _args(images_field="images")
    assert is_vlm_run_across_ranks(
        args, "Qwen/Qwen3.5-9B", _dataset([TEXT_TURNS], {"images": [[]]}), config=VLM_CONFIG
    )


def test_images_field_is_a_declaration_even_before_the_dataset_is_known():
    """``prepare_vlm_dataset``'s column guard owns a mistyped ``images_field``, so the dispatch must not demote it
    to a text run."""
    assert is_vlm_run_across_ranks(
        _args(images_field="pictures"), "Qwen/Qwen3.5-9B", _dataset([TEXT_TURNS]), config=VLM_CONFIG
    )


@pytest.mark.parametrize("column", VLM_IMAGE_COLUMNS)
def test_each_image_column_declares_a_vlm_run(column):
    """The raw ``images``/``image`` spellings and the preprocessed artifact's ``pixel_values``."""
    dataset = _dataset([TEXT_TURNS], {column: [[]]})
    assert is_vlm_run_across_ranks(_args(), "Qwen/Qwen3.5-9B", dataset, config=VLM_CONFIG)


def test_embedded_image_content_declares_a_vlm_run():
    assert is_vlm_run_across_ranks(_args(), "Qwen/Qwen3.5-9B", _dataset([IMAGE_TURNS]), config=VLM_CONFIG)


def test_text_checkpoint_never_becomes_a_vlm_run():
    """A text-only architecture has no processor, so image data fails at the VLM loader and does not switch the data
    path."""
    dataset = _dataset([IMAGE_TURNS], {"images": [[]]})
    assert not is_vlm_run_across_ranks(_args(images_field="images"), "Qwen/Qwen3-8B", dataset, config=TEXT_CONFIG)


def test_no_dataset_yet_is_not_a_declaration():
    assert not is_vlm_run_across_ranks(_args(), "Qwen/Qwen3.5-9B", None, config=VLM_CONFIG)


def test_args_without_the_modality_knobs_still_resolve():
    """DPO's script args declare neither knob."""
    bare = SimpleNamespace()
    assert not is_vlm_run_across_ranks(bare, "Qwen/Qwen3.5-9B", _dataset([TEXT_TURNS]), config=VLM_CONFIG)
    assert is_vlm_run_across_ranks(
        bare, "Qwen/Qwen3.5-9B", _dataset([TEXT_TURNS], {"image": [None]}), config=VLM_CONFIG
    )


def test_the_script_seam_probes_the_checkpoint_the_run_loads(monkeypatch):
    """Hub ``main`` can name a different modality, and a forced trust would execute code the run never trusted."""
    seen = []
    monkeypatch.setattr(
        data_setup, "probe_checkpoint", lambda *args, **kwargs: seen.append((args, kwargs)) or (None, True)
    )
    model_config = SimpleNamespace(model_name_or_path="org/ckpt", model_revision="abc123", trust_remote_code=True)

    assert resolve_vlm_run(_args(images_field="images"), model_config, None, text_only_model=False)
    assert seen == [(("org/ckpt",), {"revision": "abc123", "trust_remote_code": True})]


def test_a_run_verdict_needs_the_checkpoint_config_or_its_probe_verdict():
    with pytest.raises(ValueError, match="needs the checkpoint's config or its probe_checkpoint verdict"):
        is_vlm_run_across_ranks(_args(images_field="images"), "org/ckpt", None)


@pytest.mark.parametrize("vlm_checkpoint", [True, False])
def test_a_checkpoint_verdict_already_taken_is_not_probed_again(monkeypatch, vlm_checkpoint):
    monkeypatch.setattr(data_setup, "probe_checkpoint", lambda *a, **k: pytest.fail("the checkpoint was probed again"))
    model_config = SimpleNamespace(model_name_or_path="org/ckpt", model_revision=None, trust_remote_code=False)

    verdict = resolve_vlm_run(
        _args(images_field="images"), model_config, None, text_only_model=False, vlm_checkpoint=vlm_checkpoint
    )
    assert verdict is vlm_checkpoint


def test_dataset_declares_images_across_ranks_accepts_a_single_split():
    assert dataset_declares_images_across_ranks(Dataset.from_dict({"images": [[]], "prompt": [TEXT_TURNS]}))
    assert not dataset_declares_images_across_ranks(Dataset.from_dict({"prompt": [TEXT_TURNS]}), "prompt")


def test_dataset_declares_images_across_ranks_needs_the_conversation_field_to_see_embedded_images():
    """Without the field name there is no column to probe."""
    dataset = _dataset([IMAGE_TURNS])
    assert not dataset_declares_images_across_ranks(dataset)
    assert dataset_declares_images_across_ranks(dataset, "prompt")


def test_empty_split_does_not_crash_the_probe():
    empty = DatasetDict({"train": Dataset.from_dict({"prompt": []})})
    assert not dataset_declares_images_across_ranks(empty, "prompt")
    assert dataset_image_evidence(empty) is None


def test_dataset_image_evidence_probes_every_column_and_names_the_declaration():
    """For a consumer that never learns ``conversation_field``: each declaration is named in the verdict, and text
    is ``None``."""
    assert dataset_image_evidence(None) is None
    assert dataset_image_evidence(_dataset([TEXT_TURNS])) is None
    assert dataset_image_evidence(Dataset.from_dict({"input_ids": [[1, 2, 3]], "labels": [[1, 2, 3]]})) is None
    assert "['images']" in dataset_image_evidence(_dataset([TEXT_TURNS], {"images": [[]]}))
    assert "'conversation'" in dataset_image_evidence(Dataset.from_dict({"conversation": [IMAGE_TURNS]}))
    assert "'prompt'" in dataset_image_evidence(_mixed_dataset())
    assert "'prompt'" in dataset_image_evidence(Dataset.from_dict({"prompt": [json.dumps(IMAGE_TURNS)]}))


def test_json_string_conversations_are_probed_too():
    dataset = DatasetDict({"train": Dataset.from_dict({"prompt": [json.dumps(IMAGE_TURNS)]})})
    assert dataset_declares_images_across_ranks(dataset, "prompt")


def test_mixed_dataset_whose_first_rows_are_text_is_still_a_vlm_run():
    """On a 10%-image dataset a first-row probe misses ~81% of the time (0.9**16 for TRL's window), crashing
    mid-map. Arrow types the parts struct over the whole split, so the image field is in every row's schema."""
    assert dataset_declares_images_across_ranks(_mixed_dataset(), "prompt")
    assert is_vlm_run_across_ranks(_args(), "Qwen/Qwen3.5-9B", _mixed_dataset(), config=VLM_CONFIG)


def test_one_image_row_in_a_thousand_still_declares_the_run():
    assert dataset_declares_images_across_ranks(_mixed_dataset(image_rows=1, text_rows=999), "prompt")


def test_parts_form_text_schema_is_not_a_declaration():
    """Flipping this would route every parts-form text dataset to the VLM path, which refuses packing."""
    dataset = _dataset([TEXT_PARTS_TURNS, TEXT_PARTS_TURNS])
    assert not dataset_declares_images_across_ranks(dataset, "prompt")
    assert not is_vlm_run_across_ranks(_args(), "Qwen/Qwen3.5-9B", dataset, config=VLM_CONFIG)


def test_string_content_schema_is_not_a_declaration():
    assert not dataset_declares_images_across_ranks(_dataset([TEXT_TURNS]), "prompt")


def test_cast_image_feature_declares_images_under_any_name():
    """The feature is the declaration, even under a field name the pipeline does not read."""
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
    assert dataset_declares_images_across_ranks(dataset, "prompt")


def test_empty_split_with_an_image_schema_declares_images():
    """An empty split has no row to probe, and its declaration still routes the run."""
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
    assert dataset_declares_images_across_ranks(empty, "prompt")


def test_conversation_carries_images_predicate():
    assert conversation_carries_images(IMAGE_TURNS)
    assert not conversation_carries_images(TEXT_TURNS)
    assert not conversation_carries_images([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
    assert not conversation_carries_images("not a conversation")
    assert not conversation_carries_images(None)


def test_text_renderer_refuses_an_image_content_part():
    """A JSON-string column has no Arrow struct and only its first row probed, so this shape reaches the renderer."""
    with pytest.raises(ValueError, match="image content part"):
        render_row_conversation({"prompt": IMAGE_TURNS}, tokenizer=None, conversation_field="prompt")


def test_self_distill_text_collator_refuses_an_image_content_part():
    """The collator encodes its rows in its own map, and its VLM configs declare no images column."""
    collator = SelfDistillTextCollator(
        tokenizer=SimpleNamespace(eos_token_id=0),
        max_length=256,
        conversation_field="prompt",
        hint_template="{answer}",
        answer_field="answer",
        solution_field="solution",
        confidence_power=1.0,
        train_on_completions_only=False,
        model_config=None,
    )
    with pytest.raises(ValueError, match="image content part"):
        collator.encode_row({"prompt": IMAGE_TURNS})


@pytest.mark.parametrize("field", ["prompt", "chosen", "rejected"])
def test_preference_renderer_refuses_an_image_content_part(field):
    """DPO's script args declare no ``conversation_field``, so an image row reaches this renderer with the dispatch
    none the wiser."""
    image_content = [{"type": "image", "image": "b64"}, {"type": "text", "text": "q"}]
    row = {
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "a"}],
        "rejected": [{"role": "assistant", "content": "b"}],
    }
    row[field] = [{"role": row[field][0]["role"], "content": image_content}]
    with pytest.raises(ValueError, match="image content part"):
        render_preference_row(row, _UnreachableTokenizer())


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
    """``scripts/training/grpo/offline.py`` templates prompt and completions itself."""
    module = _script_module("grpo/offline.py")
    row = {"prompt": TEXT_TURNS[:1], "completions": [[{"role": "assistant", "content": "a"}]]}
    if field == "prompt":
        row["prompt"] = IMAGE_TURNS[:1]
    else:
        row["completions"] = [[{"role": "assistant", "content": [{"type": "image", "image": "b64"}]}]]
    with pytest.raises(ValueError, match="image content part"):
        module.render_offline_grpo_row(row, _UnreachableTokenizer(), None)


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


def test_every_image_part_site_reads_the_shared_constants(monkeypatch):
    """Re-spelling the constants moves all four sites together; a re-inlined literal keeps the old spelling and
    fails here. A detector and extractor that disagree read as no images, training placeholders."""
    alt_type, alt_key = "picture", "picture_payload"
    for module in (conversation_module, vlm_render_module, vlm_rows_module):
        monkeypatch.setattr(module, "IMAGE_PART_TYPE", alt_type)
        monkeypatch.setattr(module, "IMAGE_PART_PAYLOAD_KEY", alt_key)
    monkeypatch.setattr(vlm_detection_module, "IMAGE_PART_PAYLOAD_KEY", alt_key)
    # The offline preprocessor extracts through extract_conversation_images rather than walking the
    # parts itself, so it has no spelling of its own; a re-inlined extractor would re-import these.
    assert not hasattr(tokenization_module, "IMAGE_PART_TYPE"), (
        "tokenization re-imported the image-part constants — it must extract via extract_conversation_images"
    )

    image = PILImage.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
    respelled = [{"role": "user", "content": [{"type": alt_type, alt_key: image}]}]

    assert conversation_carries_images(respelled), "the guard predicate ignored the constants"
    assert not conversation_carries_images(IMAGE_TURNS), "the guard predicate hardcodes the old spelling"

    history, images = extract_conversation_images([dict(m) for m in respelled])
    assert len(images) == 1, "the extractor ignored the constants"
    assert history[0]["content"] == [{"type": alt_type}], "the placeholder hardcodes the old spelling"
    shipped = [{"role": "user", "content": [{"type": IMAGE_PART_TYPE, IMAGE_PART_PAYLOAD_KEY: image}]}]
    assert extract_conversation_images(shipped)[1] == [], "the extractor hardcodes the old spelling"

    merged = normalize_vlm_conversation([{"role": "user", "content": [{"type": alt_type}]}], images=[image])
    assert merged[0]["content"][0][alt_key] is image, "the images-column merge ignored the constants"

    schema_split = Dataset.from_list([{"prompt": [{"role": "user", "content": [{"type": alt_type, alt_key: "b64"}]}]}])
    assert dataset_declares_images_across_ranks(DatasetDict({"train": schema_split}), "prompt"), (
        "the Arrow-schema probe ignored the constants"
    )
    old_split = Dataset.from_list([{"prompt": TEXT_PARTS_TURNS}, {"prompt": IMAGE_TURNS}])
    assert not dataset_declares_images_across_ranks(DatasetDict({"train": old_split}), "prompt"), (
        "the Arrow-schema probe hardcodes the old spelling"
    )


def test_no_module_re_inlines_the_image_part_spelling():
    """Scoped to the pipeline package, which owns the part shape; the dataset column names are another contract
    (:data:`VLM_IMAGE_COLUMNS`)."""
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


def test_text_path_forces_right_padding():
    """A VLM processor's tokenizer keeps the checkpoint's side (left for Gemma 4 / GLM-4.7-Flash), and the packing
    collator refuses left at build."""
    tokenizer = SimpleNamespace(padding_side="left")
    apply_text_path_padding_side(tokenizer, False)
    assert tokenizer.padding_side == "right"


def test_vlm_run_keeps_the_processor_padding_side():
    """VLM batches pad through the processor, so flipping its side would change every VLM run."""
    tokenizer = SimpleNamespace(padding_side="left")
    apply_text_path_padding_side(tokenizer, True)
    assert tokenizer.padding_side == "left"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
