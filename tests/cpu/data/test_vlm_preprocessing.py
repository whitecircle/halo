#!/usr/bin/env python3
"""
Test VLM dataset preprocessing with Qwen2.5-VL-3B-Instruct.

This test verifies the VLM preprocessing pipeline:
1. Preprocessing with full tokenization (input_ids, pixel_values, etc.)
2. Loading preprocessed dataset
3. Using PreprocessedVLMDataCollator for batching

Usage:
    python tests/data/test_vlm_preprocessing.py
"""

import base64
import io
import logging
import sys
import tempfile

import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict
from PIL import Image
from transformers import AutoProcessor

from src.data.collators.vlm import PreprocessedVLMDataCollator, SelfDistillVLMDataCollator, VLMDataCollator
from src.data.pipeline.preprocessed_metadata import PreprocessingConfig
from src.data.pipeline.preprocessing import preprocess_dataset, tokenize_vlm_dataset
from src.data.vlm import VLM_IMAGE_COLUMNS
from src.models import modality
from src.models.modality import is_vlm_model
from tests.common.vlm_fakes import FakeVLMProcessorBase, FakeVLMTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_vlm_preprocessing_without_images():
    """Test VLM preprocessing with text-only conversations."""
    logger.info("=" * 60)
    logger.info("Testing VLM Preprocessing (text-only conversations)")
    logger.info("=" * 60)

    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:  # offline / uncached
        pytest.skip(f"VLM processor unavailable offline: {e}")

    examples = [
        {
            "conversation": [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 equals 4."},
            ]
        },
        {
            "conversation": [
                {"role": "user", "content": "Explain machine learning briefly."},
                {"role": "assistant", "content": "Machine learning is a subset of AI where systems learn from data."},
            ]
        },
    ]

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(examples),
            "test": Dataset.from_list(examples[:1]),
        }
    )

    config = PreprocessingConfig(
        model_name_or_path=model_name,
        max_length=512,
        conversation_field="conversation",
        is_vlm=True,
        num_proc=1,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        result = preprocess_dataset(
            dataset=dataset,
            tokenizer_or_processor=processor,
            config=config,
            output_dir=temp_dir,
        )

        metadata = result["metadata"]
        logger.info(f"Text-only VLM preprocessing: {metadata.total_train_examples} examples")

        train_ds = result["train"]
        assert "input_ids" in train_ds.column_names
        assert metadata.total_train_examples == 2

    logger.info("Text-only VLM preprocessing test passed!")


# PreprocessedVLMDataCollator tests run without a model download: only pad_token_id is read.


class _FakeTokenizer:
    """Minimal stand-in: the collator only reads pad_token_id."""

    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id


def test_the_preprocessed_vlm_schema_is_the_declared_column_set():
    """The offline writer's Arrow schema and the collator's ``required_dataset_columns`` are one
    declaration, checked here against an independent literal rather than against each other.

    Both sides matter: the writer pins the schema on the map (an all-None batch would otherwise
    infer null Arrow columns), and the trainer's dataloader mixin unions the collator's declaration
    into HF's signature-column set — without it TRL's SFT signature list prunes the stored columns
    (pixel_values_shape, attention_mask, …) before collation and the first batch crashes. Changing
    the stored schema means changing this literal, deliberately.
    """
    from src.data.vlm import VLM_OUTPUT_COLUMNS, VLM_OUTPUT_FEATURES

    expected = {
        "input_ids",
        "attention_mask",
        "labels",
        "pixel_values",
        "pixel_values_shape",
        "image_grid_thw",
    }
    assert set(VLM_OUTPUT_FEATURES) == expected, "the offline writer's Arrow schema changed"
    assert set(VLM_OUTPUT_COLUMNS) == expected
    assert set(PreprocessedVLMDataCollator.required_dataset_columns) == expected, (
        "column pruning would drop every stored column the collator does not declare"
    )


def test_collator_right_pads_to_batch_max():
    """Variable-length examples right-pad to the batch max; pad positions get
    pad_token_id (input_ids), 0 (attention_mask), -100 (labels)."""
    collator = PreprocessedVLMDataCollator(_FakeTokenizer(pad_token_id=0), max_length=2048)

    ex_short = {"input_ids": [5, 6], "attention_mask": [1, 1], "labels": [5, 6]}
    ex_long = {"input_ids": [7, 8, 9, 10], "attention_mask": [1, 1, 1, 1], "labels": [7, 8, 9, 10]}

    batch = collator([ex_short, ex_long])

    assert batch["input_ids"].shape == (2, 4)
    assert batch["input_ids"][0].tolist() == [5, 6, 0, 0]
    assert batch["attention_mask"][0].tolist() == [1, 1, 0, 0]
    assert batch["labels"][0].tolist() == [5, 6, -100, -100]
    assert batch["input_ids"][1].tolist() == [7, 8, 9, 10]
    assert batch["labels"][1].tolist() == [7, 8, 9, 10]


def test_collator_does_not_truncate_preprocessed_data():
    """Preprocessed VLM data is NOT re-truncated to max_length here.

    Capping the batch at self.max_length truncates input_ids while pixel_values / image_grid_thw pass
    through whole — desyncing the image-placeholder count from the vision patches (a vision-merge
    shape mismatch) whenever a preprocessed example is longer than the collator's max_length. The
    data is already length-bounded (image-token-aware) at preprocessing, so the collator must only
    pad. A shorter budget comes from re-preprocessing.
    """
    collator = PreprocessedVLMDataCollator(_FakeTokenizer(pad_token_id=0), max_length=3)
    ex = {"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1] * 5, "labels": [1, 2, 3, 4, 5]}
    batch = collator([ex])
    assert batch["input_ids"].shape == (1, 5)
    assert batch["input_ids"][0].tolist() == [1, 2, 3, 4, 5]
    assert batch["labels"][0].tolist() == [1, 2, 3, 4, 5]


def test_collator_keeps_image_tokens_aligned_when_over_max_length():
    """The core safety property: an over-max_length example with pixel_values keeps ALL its image
    placeholder tokens, so the count still matches the (whole, untruncated) pixel_values."""
    image_token = 99
    collator = PreprocessedVLMDataCollator(_FakeTokenizer(pad_token_id=0), max_length=4)
    pv = np.arange(6, dtype=np.float16).reshape(3, 2)  # 3 patches; one placeholder token each
    ex = {
        "input_ids": [1, image_token, image_token, image_token, 2, 3],  # len 6 > max_length 4
        "attention_mask": [1] * 6,
        "labels": [1, image_token, image_token, image_token, 2, 3],
        "pixel_values": pv.tobytes(),
        "pixel_values_shape": [3, 2],
        "image_grid_thw": [[1, 1, 3]],
    }
    batch = collator([ex])
    num_image_tokens = int((batch["input_ids"][0] == image_token).sum())
    num_patches = batch["pixel_values"].shape[0]
    assert num_image_tokens == 3, "image placeholder tokens were truncated → desynced from pixel_values"
    assert num_image_tokens == num_patches, "image-token count must equal the patch count"


def test_collator_respects_nonzero_pad_id():
    """A non-zero pad_token_id is used for input padding (mask/labels unaffected)."""
    collator = PreprocessedVLMDataCollator(_FakeTokenizer(pad_token_id=42), max_length=2048)
    batch = collator(
        [
            {"input_ids": [1], "attention_mask": [1], "labels": [1]},
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2]},
        ]
    )
    assert batch["input_ids"][0].tolist() == [1, 42]
    assert batch["attention_mask"][0].tolist() == [1, 0]
    assert batch["labels"][0].tolist() == [1, -100]


def test_collator_deserializes_pixel_values_and_grid():
    """pixel_values bytes are reconstructed by shape and concatenated; grids stacked."""
    collator = PreprocessedVLMDataCollator(_FakeTokenizer(pad_token_id=0), max_length=2048)

    pv = np.arange(6, dtype=np.float16).reshape(2, 3)  # one image: 2 patches x 3 dims
    ex = {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "labels": [1, 2, 3],
        "pixel_values": pv.tobytes(),
        "pixel_values_shape": [2, 3],
        "image_grid_thw": [[1, 2, 3]],
    }
    batch = collator([ex, ex])

    assert batch["pixel_values"].shape == (4, 3)
    assert torch.allclose(batch["pixel_values"][:2], torch.tensor(pv.astype(np.float32)))
    assert batch["image_grid_thw"].tolist() == [[1, 2, 3], [1, 2, 3]]


def test_collator_text_only_has_no_pixel_keys():
    """Text-only stored examples (no pixel_values) collate without image keys."""
    collator = PreprocessedVLMDataCollator(_FakeTokenizer(pad_token_id=0), max_length=2048)
    batch = collator([{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2]}])
    assert "pixel_values" not in batch
    assert "image_grid_thw" not in batch


# Truncation desyncs image placeholders from pixel_values/image_grid_thw: over-length rows are dropped
# at preprocessing and raise at the collator (a runtime batch cannot drop rows without desyncing DP).


class _FakeVLMProcessor(FakeVLMProcessorBase):
    """Declares the legacy truncation/max_length kwargs explicitly and accepts NO ``**kwargs`` — so
    a caller that reintroduces ``truncation=True`` silently cuts sequences here and these
    regression tests see it, instead of the drop/raise the collator owes."""

    def __call__(
        self,
        text,
        return_tensors="pt",
        padding=False,
        truncation=False,
        max_length=None,
        images=None,
        videos=None,
        add_special_tokens=True,  # accepted like real processors; the fake render carries no BOS
    ):
        return self.encode_batch(text, max_length=max_length if truncation else None)


def test_vlm_collator_raises_on_over_length_instead_of_truncating():
    """VLMDataCollator must refuse to truncate: cutting expanded image placeholder tokens while
    pixel_values/image_grid_thw keep every patch desyncs the two silently. It must raise."""
    tok = FakeVLMTokenizer()
    collator = VLMDataCollator(_FakeVLMProcessor(tok), tok, max_length=4)
    ex = {"history": [{"role": "user", "content": "one two three four five six"}], "images": []}
    with pytest.raises(ValueError, match="cannot be truncated at collation"):
        collator([ex])


def test_vlm_collator_within_budget_pads_and_builds_labels():
    """Under the budget the collator behaves as before: right-padded batch + pad-masked labels."""
    tok = FakeVLMTokenizer()
    collator = VLMDataCollator(_FakeVLMProcessor(tok), tok, max_length=16)
    examples = [
        {
            "history": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there"}],
            "images": [],
        },
        {"history": [{"role": "user", "content": "a"}], "images": []},
    ]
    batch = collator(examples)
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape == batch["labels"].shape
    assert int(batch["attention_mask"][1].sum()) < batch["input_ids"].shape[1]
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()


def test_self_distill_vlm_teacher_overflow_raises():
    """The teacher branch (student history + privileged hint) must refuse truncation too: right-
    truncating it cuts trailing response tokens the student keeps, breaking the byte-identical
    response invariant OPD row alignment relies on. It must raise, naming the teacher branch."""
    tok = FakeVLMTokenizer()
    collator = SelfDistillVLMDataCollator(
        _FakeVLMProcessor(tok),
        tok,
        max_length=8,
        hint_template=" hint hint hint hint hint {answer}",
        answer_field="answer",
        solution_field=None,
    )
    ex = {
        # Student renders to exactly 8 tokens (fits); the teacher adds the 6-token hint (overflows).
        "history": [
            {"role": "user", "content": "q q"},
            {"role": "assistant", "content": "a a"},
        ],
        "images": [],
        "answer": "42",
    }
    with pytest.raises(ValueError, match="teacher branch"):
        collator([ex])


def test_vlm_preprocessing_drops_over_length_rows_instead_of_truncating():
    """process_vlm_example must DROP an over-length row, never truncate it: input_ids cut to
    max_length beside FULL pixel_values/image_grid_thw writes desynced rows to disk. Dropping
    mirrors the text path's none-example filtering."""
    tok = FakeVLMTokenizer()
    processor = _FakeVLMProcessor(tok)
    short_row = {
        "conversation": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
    }
    long_row = {
        "conversation": [
            {"role": "user", "content": " ".join(["w"] * 64)},
            {"role": "assistant", "content": "ok"},
        ]
    }
    dataset = Dataset.from_list([short_row, long_row])
    config = PreprocessingConfig(
        model_name_or_path="fake/vlm",
        max_length=16,
        conversation_field="conversation",
        is_vlm=True,
        num_proc=1,
    )
    tokenized = tokenize_vlm_dataset(dataset, processor, config, split_name="train")
    assert len(tokenized) == 1
    assert all(len(row["input_ids"]) <= config.max_length for row in tokenized)


def test_vlm_preprocessing_error_row_alongside_success_row():
    """Every row — successful, dropped or failed — must carry ONE fixed schema, the error row being
    filtered out afterwards. The Arrow writer requires uniform keys within a write batch, so an
    error path returning a bare {"input_ids": None} crashes the map (KeyError: 'attention_mask') as
    soon as a failing row shares a batch with a good one."""
    tok = FakeVLMTokenizer()
    processor = _FakeVLMProcessor(tok)
    good_row = {
        "conversation": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
    }
    error_row = {"conversation": []}  # hits the empty-conversation none path
    dataset = Dataset.from_list([good_row, error_row])
    config = PreprocessingConfig(
        model_name_or_path="fake/vlm",
        max_length=64,
        conversation_field="conversation",
        is_vlm=True,
        num_proc=1,
    )
    tokenized = tokenize_vlm_dataset(dataset, processor, config, split_name="train")
    assert len(tokenized) == 1
    assert len(tokenized[0]["input_ids"]) > 0


class _AllImageTokenVLMProcessor(_FakeVLMProcessor):
    """Renders every attended position as the image-pad id, so the unconditional image masking
    leaves no trainable token — the same all-ignore artifact a mistyped assistant_message_template
    bakes on this path."""

    def encode_batch(self, text, max_length=None):
        batch = super().encode_batch(text, max_length=max_length)
        batch["input_ids"] = torch.where(
            batch["attention_mask"].bool(), torch.full_like(batch["input_ids"], 99), batch["input_ids"]
        )
        return batch


def test_vlm_preprocessing_refuses_a_bake_that_trains_zero_tokens():
    """The VLM bake owes the same zero-trainable-tokens refusal as the text bake.

    Its row filter keys on ``input_ids``/``attention_mask``, never on labels, so a fully-masked
    artifact passes every other check and ships: the run then trains on nothing at full cost.
    """
    processor = _AllImageTokenVLMProcessor(FakeVLMTokenizer())
    row = {"conversation": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]}
    config = PreprocessingConfig(
        model_name_or_path="fake/vlm",
        max_length=64,
        conversation_field="conversation",
        is_vlm=True,
        num_proc=1,
    )
    with pytest.raises(ValueError, match="ZERO tokens"):
        tokenize_vlm_dataset(Dataset.from_list([row] * 3), processor, config, split_name="train")


class _ImageCountingVLMProcessor(_FakeVLMProcessor):
    """Fake processor whose OUTPUT length encodes how many images it was handed.

    The merge is asserted on the baked row rather than on a recorded call, because
    ``coordinated_map`` caches its result to disk and a cache hit replays the row while a side
    channel would come back empty.
    """

    def __call__(self, text, images=None, **kwargs):
        marker = " ".join(["<image>"] * len(images or []))
        return super().__call__([f"{marker} {t}" for t in text], images=images, **kwargs)


def _tiny_image_data_uri() -> str:
    """A 64x64 PNG as a base64 data URI — an image spelling ``process_image`` accepts and Arrow can
    store (a bare path could not survive the Arrow round trip)."""
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _vlm_config(**overrides) -> PreprocessingConfig:
    return PreprocessingConfig(
        model_name_or_path="fake/vlm",
        max_length=64,
        conversation_field="conversation",
        is_vlm=True,
        num_proc=1,
        **overrides,
    )


_IMAGELESS_TURNS = [{"role": "user", "content": "describe"}, {"role": "assistant", "content": "ok"}]


class _VocabCountingVLMTokenizer(FakeVLMTokenizer):
    """Records every full-vocab read — the cost the image-token resolution must pay once, not per row."""

    def __init__(self):
        self.vocab_reads = 0

    def get_vocab(self) -> dict[str, int]:
        self.vocab_reads += 1
        return super().get_vocab()


def test_image_token_ids_are_resolved_once_per_collator_not_per_batch():
    """``get_image_token_ids`` unions the tokenizer's WHOLE vocab dict. Resolved per batch it is a
    per-step cost of the vocabulary's size (~150k entries) for a verdict fixed at construction."""
    tok = _VocabCountingVLMTokenizer()
    collator = VLMDataCollator(_FakeVLMProcessor(tok), tok, max_length=32)
    example = {"history": _IMAGELESS_TURNS, "images": []}

    for _ in range(3):
        collator([example])

    assert tok.vocab_reads == 1, f"the vocab was read {tok.vocab_reads}x for 1 construction + 3 batches"


def test_image_token_ids_are_resolved_once_per_bake_not_per_row(tmp_path, monkeypatch):
    """Same read, hoisted out of the offline map's row closure — there it was paid once per ROW."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets"))  # a cache hit would skip the map
    tok = _VocabCountingVLMTokenizer()

    tokenize_vlm_dataset(
        Dataset.from_list([{"conversation": _IMAGELESS_TURNS}] * 8), _FakeVLMProcessor(tok), _vlm_config(), "train"
    )

    assert tok.vocab_reads == 1, f"the vocab was read {tok.vocab_reads}x while baking 8 rows"


def test_vlm_preprocessing_merges_a_separate_images_column():
    """Offline ``--vlm`` must read the images column, not conversation_field alone: a dataset keeping
    its images in their own column — the hub VLM shape the runtime twin (``create_vlm_processor`` →
    ``normalize_vlm_conversation``) supports — otherwise reaches the processor with ZERO images,
    baking text rows into an artifact stamped is_vlm=True that the training side accepts as
    multimodal.
    """
    processor = _ImageCountingVLMProcessor(FakeVLMTokenizer())

    text_only = tokenize_vlm_dataset(
        Dataset.from_list([{"conversation": _IMAGELESS_TURNS}]), processor, _vlm_config(), "train"
    )
    merged = tokenize_vlm_dataset(
        Dataset.from_list([{"conversation": _IMAGELESS_TURNS, "images": _tiny_image_data_uri()}]),
        processor,
        _vlm_config(images_field="images"),
        "train",
    )

    assert len(merged) == 1, "the image-carrying row was dropped instead of being processed"
    assert len(merged[0]["input_ids"]) == len(text_only[0]["input_ids"]) + 1, (
        "the images column never reached the processor — the row baked text only"
    )


@pytest.mark.parametrize("column", VLM_IMAGE_COLUMNS)
@pytest.mark.parametrize("is_vlm", [True, False])
def test_preprocessing_refuses_an_image_column_nothing_consumes(column, is_vlm):
    """The backstop for the same silent drop without the new flag: tokenization removes every source
    column, so an image column no ``images_field`` names is deleted without a word."""
    dataset = DatasetDict({"train": Dataset.from_list([{"conversation": _IMAGELESS_TURNS, column: "img"}])})
    config = _vlm_config() if is_vlm else PreprocessingConfig(model_name_or_path="fake/vlm", num_proc=1)

    with pytest.raises(ValueError, match=f"carries the image column\\(s\\) \\['{column}'\\]"):
        preprocess_dataset(dataset, _FakeVLMProcessor(FakeVLMTokenizer()), config)


def test_a_consumed_images_column_passes_the_backstop():
    """Anti-vacuity: the refusal is about columns nothing consumes, not about image data."""
    row = {"conversation": _IMAGELESS_TURNS, "images": _tiny_image_data_uri()}
    dataset = DatasetDict({"train": Dataset.from_list([row])})
    result = preprocess_dataset(
        dataset, _ImageCountingVLMProcessor(FakeVLMTokenizer()), _vlm_config(images_field="images")
    )
    assert result["metadata"].config["images_field"] == "images"


def test_images_field_without_vlm_mode_is_refused():
    """The text tokenization merges no image column, so naming one there would drop it in silence."""
    with pytest.raises(ValueError, match="images_field='images' names an image column"):
        PreprocessingConfig(model_name_or_path="fake/vlm", images_field="images")


def test_is_vlm_model_detects_via_config_mapping():
    """A model_type registered under AutoModelForImageTextToText is a VLM regardless of its name."""
    from types import SimpleNamespace

    from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

    mt = next(iter(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES))
    assert is_vlm_model("org/some-text-named-checkpoint", config=SimpleNamespace(model_type=mt))


def test_is_vlm_model_detects_via_vision_config():
    """A config carrying a vision_config is a VLM even if its model_type isn't in the mapping."""
    from types import SimpleNamespace

    cfg = SimpleNamespace(model_type="custom_thing", vision_config=SimpleNamespace(hidden_size=8))
    assert is_vlm_model("org/whatever", config=cfg)


def test_is_vlm_model_text_config_is_not_vlm():
    """A plainly text-only config with no VLM name hint is not a VLM."""
    from types import SimpleNamespace

    assert not is_vlm_model("Qwen/Qwen3-4B", config=SimpleNamespace(model_type="qwen3"))


def test_is_vlm_model_name_fallback_for_unmapped_vlm():
    """Name heuristic still catches a VLM whose (text-looking) config isn't in the mapping — e.g. a
    remote-code VLM — so config-miss does not force a false negative."""
    from types import SimpleNamespace

    assert is_vlm_model("org/My-Custom-VL-7B", config=SimpleNamespace(model_type="custom"))
    assert is_vlm_model("llava-hf/llava-1.5-7b-hf", config=SimpleNamespace(model_type="custom"))


def test_a_registered_text_only_config_beats_a_name_hint_matching_mid_word():
    """A path substring must not override a config transformers can vouch for as text-only.

    The hints are unanchored, so ``vision`` matches ``revision-8472618`` and ``-vl`` matches
    ``-vllm``. While a text-only verdict fell through to them, such a path routed a registered
    text-only checkpoint into the VLM path — where the image collator and the packing /
    padding-free rejection make it fail — and forced text-only re-publishes of natively-multimodal
    MoEs into directories named to dodge the list.
    """
    from types import SimpleNamespace

    text_only = SimpleNamespace(model_type="qwen3_5_moe_text", vision_config=None)
    for path in (
        "/ckpt/revision-8472618112abcbd45acbcdc58436aff4233c23f7",
        "/data/-vllm/qwen3_5_moe_397b",
        "/models/qwen3.5-397b-a17b-text",
    ):
        assert not is_vlm_model(path, config=text_only), f"{path} must follow its config, not its name"


def test_an_unregistered_model_type_still_defers_to_the_name_hint():
    """The text-only shortcut is gated on transformers KNOWING the model_type.

    For a remote-code architecture, "no ITT entry and no vision_config" carries no information, so
    silence there must not be read as text-only — that would drop a VLM's images silently, where a
    false positive from the name heuristic fails loud.
    """
    from types import SimpleNamespace

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    assert "custom_remote_vlm" not in CONFIG_MAPPING_NAMES
    unregistered = SimpleNamespace(model_type="custom_remote_vlm", vision_config=None)
    assert is_vlm_model("org/Custom-VL-7B", config=unregistered)
    assert not is_vlm_model("org/Custom-Text-7B", config=unregistered)


def test_a_coordination_failure_propagates_instead_of_degrading_to_the_name_heuristic(monkeypatch):
    """The store phase around the probe is collective-equivalent; its failures are not a missing
    config. Swallowed into the name-heuristic fallback, a rank whose ``vlm_probe`` wait timed out
    answers off the path while its peers answer off the real config — the two can disagree, and
    ranks that build different model classes hang rather than fail. Only the hub read may degrade.
    """

    def _timed_out(tag, fetch):
        raise RuntimeError(f"Store coordination 'main_first/hub_meta/{tag}/shared' timed out")

    monkeypatch.setattr(modality, "hub_metadata_main_first", _timed_out)
    with pytest.raises(RuntimeError, match="timed out"):
        is_vlm_model("org/My-Custom-VL-7B")


def test_an_unreadable_config_still_falls_back_to_the_name_heuristic(monkeypatch):
    """The other half: an absent or unreachable config is a real fallback case, and it must resolve
    the same on every rank — each runs the same fetch against the same path, so the verdict is a
    pure function of ``model_name_or_path``."""

    def _unreachable(*_args, **_kwargs):
        raise OSError("hub unreachable")

    monkeypatch.setattr(modality.AutoConfig, "from_pretrained", _unreachable)
    assert is_vlm_model("org/My-Custom-VL-7B")
    assert not is_vlm_model("Qwen/Qwen3-4B")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
