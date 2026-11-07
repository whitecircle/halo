#!/usr/bin/env python
"""Nemotron-H's image rows on its own processor and composite: its ``image_grid_hw`` through the bake, the collators and
the forward, the placeholder tokens it adds, and the media and methods its rows do not take."""

import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict
from PIL import Image
from transformers import (
    FeatureExtractionMixin,
    NemotronH_Omni_Reasoning_V3ImageProcessor,
    NemotronH_Omni_Reasoning_V3Processor,
    NemotronH_Omni_Reasoning_V3VideoProcessor,
)
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMamba2Mixer

from halo.data.collators.smpo import DataCollatorForVLMSMPO, concatenate_pairs
from halo.data.collators.vlm import PreprocessedVLMDataCollator, VLMDataCollator
from halo.data.pipeline.preferences import tokenize_vlm_preference_row
from halo.data.pipeline.vlm_rows import build_vlm_row_processor
from halo.data.preprocessed.metadata import PreprocessingConfig
from halo.data.preprocessed.tokenization import tokenize_vlm_dataset
from halo.data.vlm_render import resolve_image_token_ids
from halo.entrypoints import data_setup
from halo.entrypoints.data_setup import resolve_preference_vlm_run
from halo.models.families.nemotron_h import apply_nemotron_h_patches
from halo.models.moe.family_specs import MOE_FAMILY_SPECS, NEMOTRON_H
from halo.trainers.distillation.self_distillation import _teacher_inputs
from tests.common.tiny_models import TINY_MOE_VLM_FAMILIES, tiny_family_model, word_level_tokenizer
from tests.common.tiny_processors import tiny_processor

_MODEL_TYPE = "nemotron_h_omni"
_TURNS = [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the image ."}]},
    {"role": "assistant", "content": [{"type": "text", "text": "A square of noise ."}]},
]
_TEXT_TURNS = [
    {"role": "user", "content": [{"type": "text", "text": "Describe the noise ."}]},
    {"role": "assistant", "content": [{"type": "text", "text": "A square ."}]},
]
_VIDEO_TURNS = [
    {"role": "user", "content": [{"type": "video", "video": "a.mp4"}, {"type": "text", "text": "Describe ."}]},
    {"role": "assistant", "content": [{"type": "text", "text": "A ."}]},
]
_IMAGE_SIDE = 56
# ``in_proj`` scaled up where a row must cross: at stock init a crossed boundary moves the logits by rounding alone.
_IN_PROJ_SCALE = 50.0
# fp32 rounding of the scan, against a row alone.
_PADDED_EXACT = 1e-5
_CROSSED = 1e-2


@pytest.fixture(scope="module", autouse=True)
def _patched():
    apply_nemotron_h_patches()


def _image(seed: int = 0) -> Image.Image:
    return Image.fromarray((np.random.RandomState(seed).rand(_IMAGE_SIDE, _IMAGE_SIDE, 3) * 255).astype("uint8"))


def _composite(processor):
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_VLM_FAMILIES[_MODEL_TYPE], processor.tokenizer)
    model.image_token_id = model.config.image_token_id = processor.image_token_id
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, NemotronHMamba2Mixer):
                module.in_proj.weight.mul_(_IN_PROJ_SCALE)
    return model.eval()


def _collator(processor) -> VLMDataCollator:
    return VLMDataCollator(processor, processor.tokenizer, max_length=256, train_on_completions_only=False)


def _bake_config() -> PreprocessingConfig:
    return PreprocessingConfig(
        model_name_or_path="fake/vlm", max_length=256, conversation_field="conversation", is_vlm=True, num_proc=1
    )


def _baked_rows(processor) -> Dataset:
    rows = [{"conversation": _TURNS, "images": [_image()]}, {"conversation": _TEXT_TURNS, "images": []}]
    config = _bake_config()
    config.images_field = "images"
    return tokenize_vlm_dataset(Dataset.from_list(rows), processor, config, "train")


def test_the_runtime_collator_hands_the_forward_its_grid_and_the_pixels_reach_the_loss():
    processor = tiny_processor(_MODEL_TYPE)
    batch = _collator(processor)([{"history": _TURNS, "images": [_image()]}])
    assert batch["image_grid_hw"].tolist() == [[4, 4]]
    model = _composite(processor)
    with torch.no_grad():
        loss = model(**batch, use_cache=False).loss
        blank = model(**batch | {"pixel_values": torch.zeros_like(batch["pixel_values"])}, use_cache=False).loss
    assert torch.isfinite(loss) and not torch.equal(loss, blank)


def test_the_bake_stores_the_grid_and_its_collator_hands_it_back_as_the_processor_wrote_it(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets"))
    processor = tiny_processor(_MODEL_TYPE)
    baked = _baked_rows(processor)
    assert baked[0]["image_grid_hw"] == [[4, 4]] and baked[0]["image_grid_thw"] is None
    assert baked[1]["image_grid_hw"] is None, "a text row holds no grid"

    baked_batch = PreprocessedVLMDataCollator(processor.tokenizer, max_length=256)([baked[0], baked[1]])
    assert baked_batch["image_grid_hw"].tolist() == [[4, 4]] and "image_grid_thw" not in baked_batch

    runtime = _collator(processor)([{"history": _TURNS, "images": [_image()]}])
    assert baked_batch["input_ids"][0][: runtime["input_ids"].shape[1]].tolist() == runtime["input_ids"][0].tolist()
    torch.testing.assert_close(
        baked_batch["pixel_values"], runtime["pixel_values"].float(), rtol=1e-2, atol=1e-2, msg="baked at float16"
    )


def test_a_baked_batch_runs_the_composite_as_the_runtime_batch_does(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets"))
    processor = tiny_processor(_MODEL_TYPE)
    model = _composite(processor)
    baked = PreprocessedVLMDataCollator(processor.tokenizer, max_length=256)([_baked_rows(processor)[0]])
    runtime = _collator(processor)([{"history": _TURNS, "images": [_image()]}])
    with torch.no_grad():
        baked_loss = model(**baked, use_cache=False).loss
        runtime_loss = model(**runtime, use_cache=False).loss
        blank = model(**baked | {"pixel_values": torch.zeros_like(baked["pixel_values"])}, use_cache=False).loss
    torch.testing.assert_close(baked_loss, runtime_loss, rtol=1e-3, atol=1e-3)
    assert not torch.equal(baked_loss, blank), "the baked pixels never reached the loss"


def test_a_baked_batch_without_its_grid_does_not_run():
    processor = tiny_processor(_MODEL_TYPE)
    batch = _collator(processor)([{"history": _TURNS, "images": [_image()]}])
    del batch["image_grid_hw"]
    with pytest.raises((TypeError, AttributeError)):
        _composite(processor)(**batch, use_cache=False)


def test_both_markers_of_an_image_span_are_masked_from_the_labels():
    """Its closing marker follows every image, so it would train as text at each one."""
    processor = tiny_processor(_MODEL_TYPE)
    batch = _collator(processor)([{"history": _TURNS, "images": [_image()]}])
    tokenizer = processor.tokenizer
    for token in ("<image>", "<img>", "</img>"):
        positions = batch["input_ids"] == tokenizer.convert_tokens_to_ids(token)
        assert positions.any(), f"the row holds no {token}"
        assert (batch["labels"][positions] == -100).all(), token


def test_a_placeholder_the_tokenizer_lacks_does_not_mask_the_unknown_token():
    """The release's tokenizer has no ``<video>``, so its processor's video id is the unknown token's."""
    tokenizer = word_level_tokenizer(("Describe", "the", "image"))
    tokenizer.add_special_tokens({"additional_special_tokens": ["<image>", "<so_embedding>", "<img>", "</img>"]})
    processor = NemotronH_Omni_Reasoning_V3Processor(
        image_processor=NemotronH_Omni_Reasoning_V3ImageProcessor(min_num_patches=16, max_num_patches=16),
        video_processor=NemotronH_Omni_Reasoning_V3VideoProcessor(),
        feature_extractor=FeatureExtractionMixin(),
        tokenizer=tokenizer,
    )
    assert processor.video_token_id == tokenizer.unk_token_id, "the fixture stands for the release"
    masked = resolve_image_token_ids(tokenizer, processor)
    assert tokenizer.unk_token_id not in masked
    assert {tokenizer.convert_tokens_to_ids(token) for token in ("<image>", "<img>", "</img>")} <= masked


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_padded_image_batch_scores_each_row_as_it_runs_alone(side):
    """The release pads left. The collator hands one attention mask, and the zeroed padding keeps the scan's state."""
    processor = tiny_processor(_MODEL_TYPE)
    processor.tokenizer.padding_side = side
    model = _composite(processor)
    collator = _collator(processor)
    rows = [{"history": _TURNS, "images": [_image()]}, {"history": _TEXT_TURNS, "images": []}]
    batch = collator(rows)
    assert int(batch["attention_mask"][1].sum()) < batch["input_ids"].shape[1], "the batch must pad its short row"
    with torch.no_grad():
        logits = model(**{k: v for k, v in batch.items() if k != "labels"}, use_cache=False).logits
        for row, example in enumerate(rows):
            alone_batch = collator([example])
            alone = model(**{k: v for k, v in alone_batch.items() if k != "labels"}, use_cache=False).logits[0]
            held = logits[row][batch["attention_mask"][row].bool()]
            assert (held - alone).abs().max() / alone.abs().max() < _PADDED_EXACT, row


def test_a_left_padded_row_with_no_mask_would_cross_its_padding():
    """The control on the row above: the scan reads the mask, or the pads move the row."""
    processor = tiny_processor(_MODEL_TYPE)
    processor.tokenizer.padding_side = "left"
    model = _composite(processor)
    collator = _collator(processor)
    rows = [{"history": _TURNS, "images": [_image()]}, {"history": _TEXT_TURNS, "images": []}]
    batch = collator(rows)
    with torch.no_grad():
        kept = {k: v for k, v in batch.items() if k not in ("labels", "attention_mask")}
        unmasked = model(**kept, use_cache=False).logits[1]
        alone_batch = collator([rows[1]])
        alone = model(**{k: v for k, v in alone_batch.items() if k != "labels"}, use_cache=False).logits[0]
    held = unmasked[batch["attention_mask"][1].bool()]
    assert (held - alone).abs().max() / alone.abs().max() > _CROSSED


def _smpo_batch(processor) -> dict[str, torch.Tensor]:
    row = tokenize_vlm_preference_row(
        {"prompt": _TURNS[:1], "chosen": _TURNS[1:], "rejected": _TEXT_TURNS[1:], "images": [_image()]},
        processor,
        max_prompt_length=None,
        max_completion_length=None,
        truncation_mode="keep_start",
    )
    pad_token_id = processor.tokenizer.pad_token_id
    collator = DataCollatorForVLMSMPO(pad_token_id=pad_token_id, processor=processor)
    return concatenate_pairs(collator.torch_call([row]), pad_token_id)


def test_the_smpo_collator_doubles_the_grid_with_its_pixels_for_the_rejected_rows():
    processor = tiny_processor(_MODEL_TYPE)
    batch = _smpo_batch(processor)
    assert batch["image_grid_hw"].tolist() == [[4, 4], [4, 4]]
    assert batch["pixel_values"].shape[0] == 2 * 4 * 4
    model = _composite(processor)
    kept = {key: batch[key] for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_hw")}
    with torch.no_grad():
        logits = model(**kept, use_cache=False).logits
        blank = model(**kept | {"pixel_values": torch.zeros_like(kept["pixel_values"])}, use_cache=False).logits
    assert logits.shape[0] == 2 and not torch.equal(logits, blank)


def test_the_self_distill_teacher_keeps_the_students_grid():
    """The hint leaves the image alone, so the student's pixels and grid serve the hinted row."""
    processor = tiny_processor(_MODEL_TYPE)
    student = _collator(processor)([{"history": _TURNS, "images": [_image()]}])
    teacher = {"input_ids": student["input_ids"], "attention_mask": student["attention_mask"]}
    inputs = _teacher_inputs({k: v for k, v in student.items() if k != "labels"}, teacher)
    assert torch.equal(inputs["image_grid_hw"], student["image_grid_hw"])
    assert torch.equal(inputs["pixel_values"], student["pixel_values"])


def test_a_video_part_is_refused_on_the_runtime_path():
    row = {"conversation": _VIDEO_TURNS}
    with pytest.raises(ValueError, match="video"):
        build_vlm_row_processor("conversation")(row)


def test_a_video_part_is_refused_by_the_bake_instead_of_dropping_its_row(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets"))
    rows = Dataset.from_list([{"conversation": _TEXT_TURNS}, {"conversation": _VIDEO_TURNS}])
    with pytest.raises(ValueError, match="video"):
        tokenize_vlm_dataset(rows, tiny_processor(_MODEL_TYPE), _bake_config(), "train")


def _image_dataset() -> DatasetDict:
    data = {"prompt": [_TEXT_TURNS[:1]], "chosen": [_TEXT_TURNS[1:]], "rejected": [_TEXT_TURNS[1:]], "images": [[]]}
    return DatasetDict({"train": Dataset.from_dict(data), "test": Dataset.from_dict(data)})


def _text_dataset() -> DatasetDict:
    data = {"prompt": [_TEXT_TURNS[:1]], "chosen": [_TEXT_TURNS[1:]], "rejected": [_TEXT_TURNS[1:]]}
    return DatasetDict({"train": Dataset.from_dict(data), "test": Dataset.from_dict(data)})


def _resolve(monkeypatch, family, dataset, **kwargs) -> bool:
    config = tiny_family_model(family).config
    monkeypatch.setattr(data_setup, "probe_checkpoint", lambda *_args, **_kwargs: (config, True))
    args = type("Args", (), {"images_field": None, "conversation_field": None})
    model_config = type(
        "ModelConfig", (), {"model_name_or_path": "local", "model_revision": None, "trust_remote_code": False}
    )
    return resolve_preference_vlm_run(args, model_config, dataset, text_only_model=False, **kwargs)


@pytest.mark.parametrize("method", ["DPO", "KTO"])
def test_trls_vision_path_is_refused_on_image_rows(monkeypatch, method):
    family = TINY_MOE_VLM_FAMILIES[_MODEL_TYPE]
    with pytest.raises(ValueError, match=rf"{method} on image data cannot train"):
        _resolve(monkeypatch, family, _image_dataset(), method=method, trl_renders_rows=True)


def test_text_pairs_stay_on_trls_text_path(monkeypatch):
    family = TINY_MOE_VLM_FAMILIES[_MODEL_TYPE]
    assert not _resolve(monkeypatch, family, _text_dataset(), method="DPO", trl_renders_rows=True)


def test_smpo_keeps_its_own_collator_on_image_rows(monkeypatch):
    assert _resolve(monkeypatch, TINY_MOE_VLM_FAMILIES[_MODEL_TYPE], _image_dataset(), method="SMPO")


def test_another_family_keeps_trls_vision_path(monkeypatch):
    family = TINY_MOE_VLM_FAMILIES["qwen3_5_moe"]
    assert _resolve(monkeypatch, family, _image_dataset(), method="DPO", trl_renders_rows=True)


def test_the_roster_refuses_trls_image_rows_on_nemotron_alone():
    assert [spec for spec in MOE_FAMILY_SPECS if spec.refuses_trl_image_rows] == [NEMOTRON_H]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
