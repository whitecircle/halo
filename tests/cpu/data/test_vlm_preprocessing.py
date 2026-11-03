#!/usr/bin/env python3
"""VLM dataset preprocessing and ``PreprocessedVLMDataCollator`` batching."""

import base64
import io
import logging
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict
from PIL import Image
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from halo.data.collators.self_distill import SelfDistillVLMDataCollator
from halo.data.collators.vlm import PreprocessedVLMDataCollator, VLMDataCollator
from halo.data.preprocessed.metadata import PreprocessingConfig
from halo.data.preprocessed.preparation import preprocess_dataset
from halo.data.preprocessed.tokenization import tokenize_vlm_dataset
from halo.data.vlm_detection import VLM_IMAGE_COLUMNS
from halo.data.vlm_render import VLM_OUTPUT_FEATURES
from halo.model_loading.parallel import model_source
from halo.model_loading.parallel.model_source import probe_checkpoint
from halo.models.modality import is_vlm_model
from tests.common.models import QWEN2_5_VL_3B
from tests.common.tokenizers import load_cached_processor
from tests.common.vlm_fakes import FakeVLMProcessorBase, FakeVLMTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_vlm_preprocessing_without_images():
    logger.info("=" * 60)
    logger.info("Testing VLM Preprocessing (text-only conversations)")
    logger.info("=" * 60)

    model_name = QWEN2_5_VL_3B
    processor = load_cached_processor(model_name, trust_remote_code=True)

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


class _FakeTokenizer:
    """Minimal stand-in: the collator only reads pad_token_id."""

    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id


def test_the_preprocessed_vlm_schema_is_the_declared_column_set():
    """The writer pins the schema on the map, and the dataloader mixin unions the collator's declaration into HF's
    signature columns, or TRL prunes stored columns before collation. Changing the schema means changing this
    literal."""
    expected = {
        "input_ids",
        "attention_mask",
        "labels",
        "pixel_values",
        "pixel_values_shape",
        "image_grid_thw",
    }
    assert set(VLM_OUTPUT_FEATURES) == expected, "the offline writer's Arrow schema changed"
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
    """Capping at max_length truncates input_ids while pixel_values pass through whole, desyncing placeholders from
    patches. The data is already length-bounded at preprocessing."""
    collator = PreprocessedVLMDataCollator(_FakeTokenizer(pad_token_id=0), max_length=3)
    ex = {"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1] * 5, "labels": [1, 2, 3, 4, 5]}
    batch = collator([ex])
    assert batch["input_ids"].shape == (1, 5)
    assert batch["input_ids"][0].tolist() == [1, 2, 3, 4, 5]
    assert batch["labels"][0].tolist() == [1, 2, 3, 4, 5]


def test_collator_keeps_image_tokens_aligned_when_over_max_length():
    """An over-length example with pixel_values keeps all its placeholders, matching its whole pixel_values."""
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
    collator = PreprocessedVLMDataCollator(_FakeTokenizer(pad_token_id=0), max_length=2048)
    batch = collator([{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2]}])
    assert "pixel_values" not in batch
    assert "image_grid_thw" not in batch


# Truncation desyncs image placeholders from pixel_values/image_grid_thw: over-length rows are dropped
# at preprocessing and raise at the collator (a runtime batch cannot drop rows without desyncing DP).


class _FakeVLMProcessor(FakeVLMProcessorBase):
    """Accepts no ``**kwargs``, so a caller passing ``truncation=True`` silently cuts sequences here and these tests
    see it."""

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
    """Cutting placeholders while pixel_values keep every patch desyncs the two silently."""
    tok = FakeVLMTokenizer()
    collator = VLMDataCollator(_FakeVLMProcessor(tok), tok, max_length=4, train_on_completions_only=False)
    ex = {"history": [{"role": "user", "content": "one two three four five six"}], "images": []}
    with pytest.raises(ValueError, match="cannot be truncated at collation"):
        collator([ex])


def test_vlm_collator_within_budget_pads_and_builds_labels():
    tok = FakeVLMTokenizer()
    collator = VLMDataCollator(_FakeVLMProcessor(tok), tok, max_length=16, train_on_completions_only=False)
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
    """Right-truncating the teacher cuts response tokens the student keeps, breaking the byte-identical response
    invariant OPD alignment relies on."""
    tok = FakeVLMTokenizer()
    collator = SelfDistillVLMDataCollator(
        _FakeVLMProcessor(tok),
        tok,
        max_length=8,
        train_on_completions_only=False,
        hint_template=" hint hint hint hint hint {answer}",
        answer_field="answer",
        solution_field=None,
        confidence_power=1.0,
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
    """input_ids cut beside full pixel_values would write desynced rows to disk; dropping mirrors the text path."""
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
    """The Arrow writer requires uniform keys within a write batch, so an error path returning a bare
    ``{"input_ids": None}`` crashes the map beside a good row."""
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
    """Renders every attended position as the image-pad id, so masking leaves no trainable token, the all-ignore
    artifact a mistyped template bakes."""

    def encode_batch(self, text, max_length=None):
        batch = super().encode_batch(text, max_length=max_length)
        batch["input_ids"] = torch.where(
            batch["attention_mask"].bool(), torch.full_like(batch["input_ids"], 99), batch["input_ids"]
        )
        return batch


def test_vlm_preprocessing_refuses_a_bake_that_trains_zero_tokens():
    """The row filter keys on ``input_ids``/``attention_mask``, so a fully-masked artifact would otherwise ship and
    train on nothing."""
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
    """Output length encodes how many images it was handed. Asserted on the baked row, since a ``coordinated_map``
    cache hit replays the row and a side channel would come back empty."""

    def __call__(self, text, images=None, **kwargs):
        row_images = images or [[] for _ in text]
        marked = [f"{' '.join(['<image>'] * len(one))} {t}" for t, one in zip(text, row_images, strict=True)]
        return super().__call__(marked, images=images, **kwargs)


def _tiny_image_data_uri() -> str:
    """A 64x64 PNG as a base64 data URI, which ``decode_image`` accepts and Arrow can store."""
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
    """Records every full-vocab read, a cost the image-token resolution pays once and not per row."""

    def __init__(self):
        self.vocab_reads = 0

    def get_vocab(self) -> dict[str, int]:
        self.vocab_reads += 1
        return super().get_vocab()


def test_image_token_ids_are_resolved_once_per_collator_not_per_batch():
    """``resolve_image_token_ids`` unions the whole vocab dict, a per-step cost of ~150k entries for a verdict fixed
    at construction."""
    tok = _VocabCountingVLMTokenizer()
    collator = VLMDataCollator(_FakeVLMProcessor(tok), tok, max_length=32, train_on_completions_only=False)
    example = {"history": _IMAGELESS_TURNS, "images": []}

    for _ in range(3):
        collator([example])

    assert tok.vocab_reads == 1, f"the vocab was read {tok.vocab_reads}x for 1 construction + 3 batches"


def test_image_token_ids_are_resolved_once_per_bake_not_per_row(tmp_path, monkeypatch):
    """Same read, hoisted out of the offline map's row closure, where it would be paid once per ROW."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets"))  # a cache hit would skip the map
    tok = _VocabCountingVLMTokenizer()

    tokenize_vlm_dataset(
        Dataset.from_list([{"conversation": _IMAGELESS_TURNS}] * 8), _FakeVLMProcessor(tok), _vlm_config(), "train"
    )

    assert tok.vocab_reads == 1, f"the vocab was read {tok.vocab_reads}x while baking 8 rows"


def test_vlm_preprocessing_merges_a_separate_images_column():
    """A dataset keeping images in their own column otherwise reaches the processor with zero images, baking text
    rows into an artifact stamped is_vlm=True."""
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
    """Tokenization removes every source column, so an image column no ``images_field`` names would be deleted
    without a word."""
    dataset = DatasetDict({"train": Dataset.from_list([{"conversation": _IMAGELESS_TURNS, column: "img"}])})
    config = _vlm_config() if is_vlm else PreprocessingConfig(model_name_or_path="fake/vlm", num_proc=1)

    with pytest.raises(ValueError, match=f"carries the image column\\(s\\) \\['{column}'\\]"):
        preprocess_dataset(dataset, _FakeVLMProcessor(FakeVLMTokenizer()), config)


def test_a_consumed_images_column_passes_the_backstop():
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
    mt = next(iter(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES))
    assert is_vlm_model("org/some-text-named-checkpoint", config=SimpleNamespace(model_type=mt))


def test_is_vlm_model_detects_via_vision_config():
    cfg = SimpleNamespace(model_type="custom_thing", vision_config=SimpleNamespace(hidden_size=8))
    assert is_vlm_model("org/whatever", config=cfg)


def test_is_vlm_model_text_config_is_not_vlm():
    assert not is_vlm_model("Qwen/Qwen3-4B", config=SimpleNamespace(model_type="qwen3"))


def test_is_vlm_model_name_fallback_for_unmapped_vlm():
    """A remote-code VLM whose config is unmapped still gets caught, so a config miss does not force a false
    negative."""
    assert is_vlm_model("org/My-Custom-VL-7B", config=SimpleNamespace(model_type="custom"))
    assert is_vlm_model("llava-hf/llava-1.5-7b-hf", config=SimpleNamespace(model_type="custom"))


def test_is_vlm_model_refuses_a_missing_config_naming_the_probe():
    """The verdict reads no hub: a caller without a config takes the collective probe instead."""
    with pytest.raises(TypeError, match="probe_checkpoint"):
        is_vlm_model("org/My-Custom-VL-7B", None)


def test_a_registered_text_only_config_beats_a_name_hint_matching_mid_word():
    """The hints are unanchored: ``vision`` matches ``revision-8472618`` and ``-vl`` matches ``-vllm``, which would
    route a registered text-only checkpoint into the VLM path."""
    text_only = SimpleNamespace(model_type="qwen3_5_moe_text", vision_config=None)
    for path in (
        "/ckpt/revision-8472618112abcbd45acbcdc58436aff4233c23f7",
        "/data/-vllm/qwen3_5_moe_397b",
        "/models/qwen3.5-397b-a17b-text",
    ):
        assert not is_vlm_model(path, config=text_only), f"{path} must follow its config, not its name"


def test_an_unregistered_model_type_still_defers_to_the_name_hint():
    """For a remote-code architecture no ITT entry and no vision_config carries no information; reading it as
    text-only would silently drop a VLM's images."""
    assert "custom_remote_vlm" not in CONFIG_MAPPING_NAMES
    unregistered = SimpleNamespace(model_type="custom_remote_vlm", vision_config=None)
    assert is_vlm_model("org/Custom-VL-7B", config=unregistered)
    assert not is_vlm_model("org/Custom-Text-7B", config=unregistered)


def test_a_coordination_failure_propagates_instead_of_degrading_to_the_name_heuristic(monkeypatch):
    """A rank whose ``vlm_probe`` wait timed out would answer off the path while peers answer off the config, and
    ranks building different model classes hang. Only the hub read may degrade."""

    def _timed_out(tag, fetch):
        raise RuntimeError(f"Store coordination 'main_first/hub_meta/{tag}/shared' timed out")

    monkeypatch.setattr(model_source, "hub_metadata_main_first", _timed_out)
    with pytest.raises(RuntimeError, match="timed out"):
        probe_checkpoint("org/My-Custom-VL-7B")


def test_an_unreadable_config_still_falls_back_to_the_name_heuristic(monkeypatch):
    """It resolves the same on every rank, since each runs the same fetch against the same path."""

    def _unreachable(*_args, **_kwargs):
        raise OSError("hub unreachable")

    monkeypatch.setattr(model_source.AutoConfig, "from_pretrained", _unreachable)
    assert probe_checkpoint("org/My-Custom-VL-7B")[1]
    assert not probe_checkpoint("Qwen/Qwen3-4B")[1]


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_the_probe_executes_remote_code_only_when_the_run_trusts_it(monkeypatch, trust_remote_code):
    """The probe is a run's first hub contact, so it must not run a checkpoint's config code under
    ``trust_remote_code: false``, nor refuse a config the run trusted."""
    seen = []

    def _record(*_args, **kwargs):
        seen.append(kwargs["trust_remote_code"])
        return SimpleNamespace(model_type="qwen3")

    monkeypatch.setattr(model_source.AutoConfig, "from_pretrained", _record)
    probe_checkpoint("org/checkpoint", trust_remote_code=trust_remote_code)
    probe_checkpoint("org/checkpoint")
    assert seen == [trust_remote_code, False]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
