#!/usr/bin/env python3
"""
Tests for dataset preprocessing utilities.

These tests verify the tokenization, packing, and sharding functionality.

Usage:
    python tests/data/test_preprocessing.py
"""

import json
import logging
import os
import shutil
import sys
import tempfile

import pytest
from datasets import Dataset, load_from_disk

from src.data.pipeline.preprocessed_metadata import (
    PreprocessedDatasetMetadata,
    PreprocessingConfig,
    validate_preprocessing_compatibility,
)
from src.data.pipeline.preprocessing import _warn_on_shard_count_ceiling, shard_dataset, tokenize_dataset
from src.data.shard_index import SHARD_INDEX_FILE, ShardIndex, ShardInfo


def test_shard_info_and_index():
    """Test ShardInfo and ShardIndex dataclasses."""
    print("Testing ShardInfo and ShardIndex...")

    shard1 = ShardInfo(id=0, path="train/shard_0000", num_examples=100, byte_size=10000)
    shard2 = ShardInfo(id=1, path="train/shard_0001", num_examples=150, byte_size=15000)

    assert shard1.id == 0
    assert shard1.num_examples == 100

    index = ShardIndex(
        version="1.0",
        split="train",
        num_shards=2,
        total_examples=250,
        shards=[shard1, shard2],
    )

    assert index.num_shards == 2
    assert index.total_examples == 250
    assert len(index.shards) == 2

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        index.save(temp_path)
        loaded_index = ShardIndex.load(temp_path)

        assert loaded_index.num_shards == 2
        assert loaded_index.total_examples == 250
        assert loaded_index.shards[0].id == 0
        assert loaded_index.shards[1].num_examples == 150
    finally:
        os.unlink(temp_path)

    print("  ShardInfo and ShardIndex: PASSED")


def test_tokenize_dataset():
    """Test dataset tokenization."""
    print("Testing tokenize_dataset...")

    dataset = Dataset.from_dict(
        {
            "conversation": [
                [
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well, thank you!"},
                ],
                [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "2+2 equals 4."},
                ],
            ]
        }
    )

    config = PreprocessingConfig(
        model_name_or_path="Qwen/Qwen3-0.6B",
        max_length=512,
        pack_sequences=False,
        conversation_field="conversation",
    )

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    except Exception as e:  # offline / no cached snapshot
        pytest.skip(f"tokenizer unavailable offline: {e}")
    tokenized = tokenize_dataset(dataset, tokenizer, config)

    assert "input_ids" in tokenized.column_names
    assert "labels" in tokenized.column_names, "tokenize_dataset must add a labels column"
    assert len(tokenized) == 2

    # Default (train_on_completions_only=False): labels copy input_ids, so nothing is masked out.
    row = tokenized[0]
    assert row["labels"] == row["input_ids"], "labels must equal input_ids for full-sequence loss"
    assert len(row["input_ids"]) == len(row["attention_mask"])

    print("  tokenize_dataset: PASSED")


def test_tokenize_dataset_completion_only():
    """train_on_completions_only must mask prompt tokens to -100 and keep assistant completions.

    Regression: the flag (exposed by PreprocessingConfig + prepare_dataset.py --train-only-on-
    completions) was a silent no-op — tokenize_dataset always wrote labels = input_ids.copy(), so a
    preprocessed dataset trained on the full prompt regardless.
    """
    print("Testing tokenize_dataset completion-only...")

    from transformers import AutoTokenizer

    dataset = Dataset.from_dict(
        {
            "conversation": [
                [
                    {"role": "user", "content": "Hello, how are you today friend?"},
                    {"role": "assistant", "content": "I'm doing well, thank you!"},
                ],
            ]
        }
    )
    config = PreprocessingConfig(
        model_name_or_path="Qwen/Qwen3-0.6B",
        max_length=512,
        pack_sequences=False,
        conversation_field="conversation",
        train_on_completions_only=True,
        assistant_message_template="<|im_start|>assistant\n",
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    except Exception as e:  # offline / no cached snapshot
        pytest.skip(f"tokenizer unavailable offline: {e}")

    tokenized = tokenize_dataset(dataset, tokenizer, config)
    row = tokenized[0]
    ids, labels = row["input_ids"], row["labels"]

    assert len(ids) == len(labels)
    assert labels != ids, "completion-only labels must differ from input_ids (prompt is masked)"
    assert any(t == -100 for t in labels), "prompt tokens must be masked to -100"
    assert any(t != -100 for t in labels), "assistant completion tokens must be kept"
    assert all(t == -100 or t == ids[i] for i, t in enumerate(labels))
    first_kept = next(i for i, t in enumerate(labels) if t != -100)
    assert first_kept > 0, "at least the leading prompt tokens must be masked"

    print("  tokenize_dataset completion-only: PASSED")


def test_never_matching_template_raises_at_preprocessing_time():
    """A template that never matches must fail HERE, not as loss=nan hours into training.

    Regression: running the all-masked guard before the completion bake, where the chat processor
    has not emitted labels yet, leaves it dead on the exact path it exists for."""
    from transformers import AutoTokenizer

    dataset = Dataset.from_dict(
        {
            "conversation": [
                [
                    {"role": "user", "content": "Hello, how are you today friend?"},
                    {"role": "assistant", "content": "I'm doing well, thank you!"},
                ],
            ]
        }
    )
    config = PreprocessingConfig(
        model_name_or_path="Qwen/Qwen3-0.6B",
        max_length=512,
        pack_sequences=False,
        conversation_field="conversation",
        train_on_completions_only=True,
        assistant_message_template="<|NEVER_MATCHES|>assistant:",
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    except Exception as e:  # offline / no cached snapshot
        pytest.skip(f"tokenizer unavailable offline: {e}")

    with pytest.raises(ValueError, match="ignore index"):
        tokenize_dataset(dataset, tokenizer, config)


def test_completion_only_validation():
    """Completion-only preprocessing fails fast without a marker and for mode='text'.

    Both refusals live in ``PreprocessingConfig.__post_init__``, so they fire at CONFIG time — before
    a tokenizer, a dataset or an hours-long map, and for every caller rather than only the ones that
    reach ``tokenize_dataset``.
    """
    print("Testing completion-only validation...")

    with pytest.raises(ValueError, match="assistant_message_template"):
        PreprocessingConfig(
            model_name_or_path="Qwen/Qwen3-0.6B",
            max_length=512,
            pack_sequences=False,
            train_on_completions_only=True,
        )

    with pytest.raises(ValueError, match="mode='text'"):
        PreprocessingConfig(
            model_name_or_path="Qwen/Qwen3-0.6B",
            max_length=512,
            pack_sequences=False,
            mode="text",
            train_on_completions_only=True,
            assistant_message_template="x",
        )

    print("  completion-only validation: PASSED")


def test_invalid_mode_rejected_by_the_config():
    """`mode` was unvalidated outside the CLI's choices=: a typo'd 'txt' fell through every
    `mode == "text"` branch and silently CHAT-TEMPLATED a raw pretraining corpus."""
    with pytest.raises(ValueError, match="Invalid preprocessing mode 'txt'"):
        PreprocessingConfig(model_name_or_path="m", mode="txt")


def test_mode_inapplicable_knobs_rejected_both_directions():
    """A knob set outside the mode that consumes it is silently ignored — the config must say so.
    Applicability is declared on the fields, so this covers new knobs automatically."""
    with pytest.raises(ValueError, match="system_prompt"):
        PreprocessingConfig(model_name_or_path="m", mode="text", system_prompt="you are helpful")
    with pytest.raises(ValueError, match="tools_field"):
        PreprocessingConfig(model_name_or_path="m", mode="text", tools_field="tools")
    with pytest.raises(ValueError, match="text_field"):
        PreprocessingConfig(model_name_or_path="m", mode="chat", text_field="body")

    # Anti-over-rejection: the mode's own knobs, and knobs left at their defaults, must pass.
    PreprocessingConfig(model_name_or_path="m", mode="text", text_field="body", append_eos=False)
    PreprocessingConfig(model_name_or_path="m", mode="chat", system_prompt="s", tools_field="tools")


def test_shard_dataset():
    """Test dataset sharding."""
    print("Testing shard_dataset...")

    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3]] * 100,
            "attention_mask": [[1, 1, 1]] * 100,
        }
    )

    temp_dir = tempfile.mkdtemp()

    try:
        index = shard_dataset(dataset, output_dir=temp_dir, split_name="train", num_shards=4)

        assert index.num_shards == 4
        assert index.total_examples == 100

        for shard in index.shards:
            shard_path = os.path.join(temp_dir, shard.path)
            assert os.path.exists(shard_path), f"Shard {shard.path} not found"

        # shard_dataset returns the index but does not save it.
        index_path = os.path.join(temp_dir, "train", SHARD_INDEX_FILE)
        index.save(index_path)

        assert os.path.exists(index_path), "Shard index file not found"

        total_examples = sum(s.num_examples for s in index.shards)
        assert total_examples == 100

        for shard in index.shards:
            assert 20 <= shard.num_examples <= 30, f"Shard {shard.id} has {shard.num_examples} examples"

    finally:
        shutil.rmtree(temp_dir)

    print("  shard_dataset: PASSED")


def test_shard_dataset_single_shard():
    """Test sharding with num_shards=1 (no sharding)."""
    print("Testing shard_dataset (single shard)...")

    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3]] * 50,
            "attention_mask": [[1, 1, 1]] * 50,
        }
    )

    temp_dir = tempfile.mkdtemp()

    try:
        index = shard_dataset(dataset, output_dir=temp_dir, split_name="train", num_shards=1)

        assert index.num_shards == 1
        assert index.total_examples == 50
        assert len(index.shards) == 1
        assert index.shards[0].num_examples == 50

    finally:
        shutil.rmtree(temp_dir)

    print("  shard_dataset (single shard): PASSED")


def test_shard_remainder_distribution():
    """Test that remainder shards are distributed correctly."""
    print("Testing shard remainder distribution...")

    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3]] * 10,
            "attention_mask": [[1, 1, 1]] * 10,
        }
    )

    temp_dir = tempfile.mkdtemp()

    try:
        index = shard_dataset(dataset, output_dir=temp_dir, split_name="train", num_shards=3)

        assert index.num_shards == 3
        assert index.total_examples == 10

        # Remainder (10 % 3 == 1) goes to the FIRST shard → exactly [4, 3, 3].
        example_counts = [s.num_examples for s in index.shards]
        assert example_counts == [4, 3, 3], example_counts

        # Concatenating the shards must reconstruct the original: no gaps, no duplicates, in order.
        reconstructed = []
        for shard in index.shards:
            shard_ds = load_from_disk(os.path.join(temp_dir, shard.path))
            reconstructed.extend(shard_ds["input_ids"])
        assert reconstructed == dataset["input_ids"]

    finally:
        shutil.rmtree(temp_dir)

    print("  shard remainder distribution: PASSED")


def test_shard_more_shards_than_examples():
    """num_shards > num_examples: empty shards are skipped, all examples preserved."""
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3]] * 3,
            "attention_mask": [[1, 1, 1]] * 3,
        }
    )
    temp_dir = tempfile.mkdtemp()
    try:
        index = shard_dataset(dataset, output_dir=temp_dir, split_name="train", num_shards=5)
        assert index.total_examples == 3
        # num_shards reflects the shards actually written, not the 5 requested.
        assert index.num_shards == 3
        assert all(s.num_examples == 1 for s in index.shards)
        assert sum(s.num_examples for s in index.shards) == 3
    finally:
        shutil.rmtree(temp_dir)


def test_shard_count_ceiling_warns_with_the_usable_dp_degree(caplog):
    """A split too small to fill --num-shards caps the trainable DP degree — say so at prep time.

    ``shard_dataset`` skips empty shards, so a small test split yields fewer shards than train. At
    runtime a rank without a shard is a hard failure (train) or a rejected eval (metrics-gather hang),
    discovered only after a full training startup. The warning must name the actual ceiling so the
    operator can re-preprocess correctly on the first try.
    """
    indices = {
        "train": ShardIndex(split="train", num_shards=8, total_examples=800),
        "test": ShardIndex(split="test", num_shards=3, total_examples=3),
    }
    with caplog.at_level(logging.WARNING):
        _warn_on_shard_count_ceiling(indices, requested_shards=8)
    assert "data_parallel_size <= 3" in caplog.text, caplog.text
    assert "test=3" in caplog.text and "train=8" in caplog.text

    # Every split filled its shards: nothing to warn about.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _warn_on_shard_count_ceiling(
            {"train": ShardIndex(split="train", num_shards=8), "test": ShardIndex(split="test", num_shards=8)},
            requested_shards=8,
        )
    assert caplog.text == ""


def test_max_length_compatibility_raises_both_directions():
    """A preprocessed/configured max_length mismatch must be loud in BOTH directions: smaller
    preprocessed rows waste the budget (pre-existing raise); larger ones exceed the activation
    budget because the runtime preprocessed path never re-truncates (the dangerous direction)."""
    metadata = PreprocessedDatasetMetadata(
        model_name="Qwen/Qwen3-8B",
        max_length=4096,
        packed=True,
        num_shards=1,
        total_train_examples=10,
        total_test_examples=1,
    )

    validate_preprocessing_compatibility(metadata, required_max_length=4096)

    with pytest.raises(ValueError, match="less than"):
        validate_preprocessing_compatibility(metadata, required_max_length=8192)

    with pytest.raises(ValueError, match="exceeds the configured"):
        validate_preprocessing_compatibility(metadata, required_max_length=2048)


# Render-knob compatibility: preprocessed rows are baked, so the run's render settings must match
# the recorded ones or the YAML claims a render the data does not have.


class _RenderArgs:
    """The slice of the SFT script args the render check reads (defaults match PreprocessingConfig)."""

    def __init__(self, **overrides):
        defaults = {
            "conversation_field": "prompt",
            "system_prompt": None,
            "model_supports_system_role": True,
            "tools_field": None,
            "interleaved_thinking": False,
            "assistant_message_template": None,
            "pad_token": None,
            "eos_token": None,
            "bos_token": None,
            "chat_template": None,
        }
        self.__dict__.update({**defaults, **overrides})


def _metadata_with_recorded_config(**config_overrides) -> PreprocessedDatasetMetadata:
    from dataclasses import asdict

    config = PreprocessingConfig(model_name_or_path="Qwen/Qwen3-8B", max_length=4096, **config_overrides)
    return PreprocessedDatasetMetadata(
        model_name="Qwen/Qwen3-8B",
        max_length=4096,
        train_on_completions_only=config.train_on_completions_only,
        config=asdict(config),
    )


def test_render_knob_mismatch_raises_naming_the_knob():
    metadata = _metadata_with_recorded_config(system_prompt=None)
    with pytest.raises(ValueError, match="system_prompt"):
        validate_preprocessing_compatibility(
            metadata, required_max_length=4096, render_args=_RenderArgs(system_prompt="You are helpful.")
        )
    with pytest.raises(ValueError, match="conversation_field"):
        validate_preprocessing_compatibility(
            metadata, required_max_length=4096, render_args=_RenderArgs(conversation_field="messages")
        )


def test_matching_render_knobs_pass():
    metadata = _metadata_with_recorded_config(system_prompt="sys", tools_field="tools")
    validate_preprocessing_compatibility(
        metadata,
        required_max_length=4096,
        render_args=_RenderArgs(system_prompt="sys", tools_field="tools"),
    )


def test_assistant_template_only_checked_when_labels_are_masked():
    """Without completion masking the template is inert on both sides — a mismatch must pass. With
    masking, the baked labels were built from the recorded template, so a mismatch must raise."""
    unmasked = _metadata_with_recorded_config(assistant_message_template="<A>")
    validate_preprocessing_compatibility(
        unmasked,
        required_max_length=4096,
        required_train_on_completions_only=False,
        render_args=_RenderArgs(assistant_message_template="<B>"),
    )

    masked = _metadata_with_recorded_config(train_on_completions_only=True, assistant_message_template="<A>")
    with pytest.raises(ValueError, match="assistant_message_template"):
        validate_preprocessing_compatibility(
            masked,
            required_max_length=4096,
            required_train_on_completions_only=True,
            render_args=_RenderArgs(assistant_message_template="<B>"),
        )


def test_text_mode_artifact_skips_the_render_check():
    """A raw-text pretraining artifact renders no chat template — every render knob was inert at
    prep time, so a differing run value has nothing baked to disagree with and must pass."""
    metadata = _metadata_with_recorded_config(mode="text", text_field="text")
    validate_preprocessing_compatibility(
        metadata,
        required_max_length=4096,
        render_args=_RenderArgs(system_prompt="anything", conversation_field="messages"),
    )


def test_legacy_metadata_without_recorded_render_knobs_warns_not_raises(caplog):
    metadata = PreprocessedDatasetMetadata(max_length=4096, config={})
    with caplog.at_level(logging.WARNING):
        validate_preprocessing_compatibility(
            metadata, required_max_length=4096, render_args=_RenderArgs(system_prompt="anything")
        )
    assert "predates recording" in caplog.text


def test_packing_against_unpacked_artifact_warns_with_the_consequence(caplog):
    unpacked = _metadata_with_recorded_config(pack_sequences=False)
    with caplog.at_level(logging.WARNING):
        validate_preprocessing_compatibility(
            unpacked, required_max_length=4096, render_args=_RenderArgs(), required_packing=True
        )
    assert "pack-sequences" in caplog.text, caplog.text

    caplog.clear()
    packed = _metadata_with_recorded_config(pack_sequences=True)
    packed.packed = True
    with caplog.at_level(logging.WARNING):
        validate_preprocessing_compatibility(
            packed, required_max_length=4096, render_args=_RenderArgs(), required_packing=True
        )
    assert caplog.text == ""


def test_render_check_set_is_derived_from_the_config_dataclass():
    """The checked set is read off each field's own ``render_check`` metadata, so a new render knob
    is compared by default and an exemption has to be declared where the field is defined. Pinned
    against an independent literal: a knob that silently stops being compared fails here."""
    from src.data.pipeline.preprocessed_metadata import _RENDER_CHECKED_FIELDS

    assert set(_RENDER_CHECKED_FIELDS) == {
        "conversation_field",
        "system_prompt",
        "model_supports_system_role",
        "tools_field",
        "interleaved_thinking",
        # The images column merged into the conversation: it expands vision placeholders into the
        # baked ids, so an artifact prepared without it is a different tokenization.
        "images_field",
        "assistant_message_template",
        # Tokenizer mutations: they change the baked ids, so they are compared like render knobs.
        "pad_token",
        "eos_token",
        "bos_token",
        "chat_template",
    }


# Tokenizer-mutating knobs: recorded, so the render check can see them


def test_tokenizer_override_mismatch_raises():
    """--pad/--eos/--bos/--chat-template mutate the tokenizer that bakes the ids, but were NOT
    recorded — the artifact was byte-indistinguishable from an unmutated one and the check passed.
    An eos override in particular moves the completion-mask boundaries."""
    metadata = _metadata_with_recorded_config(eos_token="<|im_end|>")
    with pytest.raises(ValueError, match="eos_token"):
        validate_preprocessing_compatibility(
            metadata, required_max_length=4096, render_args=_RenderArgs(eos_token="</s>")
        )

    metadata = _metadata_with_recorded_config(pad_token="<pad>")
    with pytest.raises(ValueError, match="pad_token"):
        validate_preprocessing_compatibility(metadata, required_max_length=4096, render_args=_RenderArgs())

    matching = _metadata_with_recorded_config(eos_token="</s>", bos_token="<s>")
    validate_preprocessing_compatibility(
        matching, required_max_length=4096, render_args=_RenderArgs(eos_token="</s>", bos_token="<s>")
    )


def test_chat_template_compared_as_resolved_text_not_as_a_path(tmp_path):
    """The config records the RESOLVED template; the run may spell the same template as a file path.
    Comparing the raw values would report every path-spelled template as a mismatch."""
    from accelerate import PartialState

    # The shared path→text resolver logs through accelerate's rank-aware logger; every caller of this
    # check runs after the toolkit has initialized the state.
    PartialState()

    template = "{% for m in messages %}{{ m.content }}{% endfor %}"
    template_file = tmp_path / "tpl.jinja"
    template_file.write_text(template)

    metadata = _metadata_with_recorded_config(chat_template=template)
    validate_preprocessing_compatibility(
        metadata, required_max_length=4096, render_args=_RenderArgs(chat_template=str(template_file))
    )

    with pytest.raises(ValueError, match="chat_template"):
        validate_preprocessing_compatibility(
            metadata, required_max_length=4096, render_args=_RenderArgs(chat_template="{{ 'different' }}")
        )


def test_source_labels_column_is_not_baked_as_loss_targets():
    """A source column named `labels` (classification/reward corpora carry one) survived the
    tokenization map, and the labels step only fills in a MISSING column — so those source values
    were baked as this dataset's loss targets."""
    from transformers import AutoTokenizer

    dataset = Dataset.from_dict(
        {
            "conversation": [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]],
            "labels": ["positive"],
        }
    )
    config = PreprocessingConfig(
        model_name_or_path="Qwen/Qwen3-0.6B",
        max_length=512,
        pack_sequences=False,
        conversation_field="conversation",
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    except Exception as e:
        pytest.skip(f"tokenizer unavailable offline: {e}")

    tokenized = tokenize_dataset(dataset, tokenizer, config)
    row = tokenized[0]
    assert row["labels"] == row["input_ids"], f"source labels leaked into the baked targets: {row['labels']!r}"


# metadata.json version stamp: compared, and never downgraded into a silent raw-path fallback


def test_metadata_version_mismatch_raises_a_version_message():
    """The stamp was written and never compared: a diverged schema surfaced as a bare TypeError
    from cls(**data), and the detection probe swallowed it into a silent raw-path downgrade that
    re-tokenizes pre-tokenized rows."""
    from src.data.shard_index import IncompatiblePreprocessedDataset

    payload = PreprocessedDatasetMetadata(max_length=4096).to_dict()
    payload["version"] = "0.9"
    with pytest.raises(IncompatiblePreprocessedDataset, match="version"):
        PreprocessedDatasetMetadata.from_dict(payload)

    payload = PreprocessedDatasetMetadata(max_length=4096).to_dict()
    payload["invented_by_a_future_build"] = 1
    with pytest.raises(IncompatiblePreprocessedDataset, match="invented_by_a_future_build"):
        PreprocessedDatasetMetadata.from_dict(payload)


def test_metadata_written_before_the_render_knob_renames_is_refused_by_name():
    """A retired spelling is refused, not migrated. Re-spelling one would claim the current knob was
    recorded when nothing verified it — the artifact's labels were baked under the OLD semantics.
    The message has to carry the offending key and the re-prep command, because a preprocessed
    dataset is an expensive S3/Hub artifact and the only fix is re-running the prep script."""
    from src.data.shard_index import IncompatiblePreprocessedDataset

    payload = PreprocessedDatasetMetadata(max_length=4096, train_on_completions_only=True).to_dict()
    # Re-spell the top-level key the way a pre-rename writer stamped it.
    payload["train_only_on_completions"] = payload.pop("train_on_completions_only")

    with pytest.raises(IncompatiblePreprocessedDataset) as excinfo:
        PreprocessedDatasetMetadata.from_dict(payload)

    message = str(excinfo.value)
    assert "train_only_on_completions" in message
    assert "scripts/before_training/prepare_dataset.py" in message


# HF Hub is a documented prepare_dataset OUTPUT, so it must be readable back as one


def test_hub_repo_id_strips_config_and_split_suffixes():
    """Hub file APIs address the repo; the loader's `org/name:config@split` spelling does not."""
    from src.data.sources.paths import hub_repo_id

    assert hub_repo_id("org/name") == "org/name"
    assert hub_repo_id("org/name@train_sft") == "org/name"
    assert hub_repo_id("org/name:subset") == "org/name"
    assert hub_repo_id("org/name:subset@train_sft") == "org/name"


def test_hub_preprocessed_dataset_is_detected_and_its_metadata_loads(tmp_path, monkeypatch):
    """An `hf://`-published preprocessed dataset could NEVER be detected (the Hub branch returned
    False unconditionally) and its metadata load raised outright — so training took the raw path and
    KeyError'd on pre-tokenized rows."""
    from src.data.pipeline import preprocessed_metadata as mod

    payload = PreprocessedDatasetMetadata(max_length=4096, model_name="Qwen/Qwen3-8B").to_dict()
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(json.dumps(payload))

    seen = {}

    def _fake_download(repo_id, filename, repo_type=None, **kwargs):
        seen["repo_id"], seen["filename"], seen["repo_type"] = repo_id, filename, repo_type
        return str(metadata_file)

    monkeypatch.setattr(mod, "hf_hub_download", _fake_download)

    assert mod.is_preprocessed_dataset("org/name@train_sft") is True
    assert seen == {"repo_id": "org/name", "filename": "metadata.json", "repo_type": "dataset"}
    assert mod.load_preprocessed_metadata("org/name").max_length == 4096


def test_hub_dataset_without_metadata_is_raw_not_an_error(monkeypatch):
    """Anti-over-rejection: the overwhelmingly common case is a RAW hub dataset, which must probe to
    False silently rather than warn or raise on every run."""
    from huggingface_hub.errors import EntryNotFoundError

    from src.data.pipeline import preprocessed_metadata as mod

    def _missing(repo_id, filename, repo_type=None, **kwargs):
        raise EntryNotFoundError("no metadata.json")

    monkeypatch.setattr(mod, "hf_hub_download", _missing)
    assert mod.is_preprocessed_dataset("HuggingFaceH4/ultrachat_200k") is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
