#!/usr/bin/env python3
import json
import logging
import os
import shutil
import tempfile
from dataclasses import asdict

import pytest
from datasets import Dataset, load_from_disk
from huggingface_hub.errors import EntryNotFoundError

from halo.data.preprocessed import metadata as mod
from halo.data.preprocessed.metadata import (
    _RENDER_CHECKED_FIELDS,
    PreprocessedDatasetMetadata,
    PreprocessingConfig,
    verify_preprocessed_dataset,
)
from halo.data.preprocessed.preparation import _warn_on_shard_count_ceiling, shard_dataset
from halo.data.preprocessed.shard_index import (
    SHARD_INDEX_FILE,
    IncompatiblePreprocessedDataset,
    ShardIndex,
    ShardInfo,
    write_stamped_sidecar,
)
from halo.data.preprocessed.tokenization import tokenize_dataset
from halo.data.sources.paths import hub_repo_id
from tests.common.tokenizers import load_cached_tokenizer


def test_shard_info_and_index():
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
        write_stamped_sidecar(temp_path, index)
        loaded_index = ShardIndex.load(temp_path)

        assert loaded_index.num_shards == 2
        assert loaded_index.total_examples == 250
        assert loaded_index.shards[0].id == 0
        assert loaded_index.shards[1].num_examples == 150
    finally:
        os.unlink(temp_path)

    print("  ShardInfo and ShardIndex: PASSED")


def test_tokenize_dataset():
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

    tokenizer = load_cached_tokenizer(config.model_name_or_path)
    tokenized = tokenize_dataset(dataset, tokenizer, config)

    assert "input_ids" in tokenized.column_names
    assert len(tokenized) == 2

    # train_on_completions_only=False: no labels are baked; the collator derives them from input_ids.
    assert "labels" not in tokenized.column_names, "a full-sequence bake must not store a labels copy"
    row = tokenized[0]
    assert len(row["input_ids"]) == len(row["attention_mask"])

    print("  tokenize_dataset: PASSED")


def test_tokenize_dataset_completion_only():
    """Must mask prompt tokens to -100 and keep assistant completions, not copy labels from input_ids."""
    print("Testing tokenize_dataset completion-only...")

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
    tokenizer = load_cached_tokenizer(config.model_name_or_path)

    tokenized = tokenize_dataset(dataset, tokenizer, config)
    # At input width: labels inferred as int64 beside int32 input_ids would be most of the artifact's bytes.
    assert tokenized.features["labels"] == tokenized.features["input_ids"], tokenized.features
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
    """A template that never matches must fail here, not as loss=nan hours into training."""
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
    tokenizer = load_cached_tokenizer(config.model_name_or_path)

    with pytest.raises(ValueError, match="ignore index"):
        tokenize_dataset(dataset, tokenizer, config)


@pytest.mark.parametrize("knob", ["min_pixels", "max_pixels"])
def test_a_pixel_bound_needs_the_vlm_bake(knob):
    with pytest.raises(ValueError, match=rf"{knob}.*is_vlm=False"):
        PreprocessingConfig(model_name_or_path="m", **{knob: 1024})
    assert getattr(PreprocessingConfig(model_name_or_path="m", is_vlm=True, **{knob: 1024}), knob) == 1024


def test_a_packing_strategy_needs_packing():
    with pytest.raises(ValueError, match="packing_strategy='wrapped'.*pack_sequences=False"):
        PreprocessingConfig(model_name_or_path="m", packing_strategy="wrapped")
    config = PreprocessingConfig(model_name_or_path="m", pack_sequences=True, packing_strategy="wrapped")
    assert config.packing_strategy == "wrapped"


def test_completion_only_validation():
    """Both refusals live in ``PreprocessingConfig.__post_init__``, so they fire at config time, for every caller."""
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
    """A typo'd 'txt' would fall through every ``mode == "text"`` branch and silently chat-template a raw corpus."""
    with pytest.raises(ValueError, match="Invalid preprocessing mode 'txt'"):
        PreprocessingConfig(model_name_or_path="m", mode="txt")


def test_mode_inapplicable_knobs_rejected_both_directions():
    """A knob set outside the mode that consumes it is silently ignored. Applicability is declared on the fields, so
    this covers new knobs."""
    with pytest.raises(ValueError, match="system_prompt"):
        PreprocessingConfig(model_name_or_path="m", mode="text", system_prompt="you are helpful")
    with pytest.raises(ValueError, match="tools_field"):
        PreprocessingConfig(model_name_or_path="m", mode="text", tools_field="tools")
    with pytest.raises(ValueError, match="text_field"):
        PreprocessingConfig(model_name_or_path="m", mode="chat", text_field="body")

    PreprocessingConfig(model_name_or_path="m", mode="text", text_field="body", append_eos=False)
    PreprocessingConfig(model_name_or_path="m", mode="chat", system_prompt="s", tools_field="tools")


def test_shard_dataset():
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
        write_stamped_sidecar(index_path, index)

        assert os.path.exists(index_path), "Shard index file not found"

        total_examples = sum(s.num_examples for s in index.shards)
        assert total_examples == 100

        for shard in index.shards:
            assert 20 <= shard.num_examples <= 30, f"Shard {shard.id} has {shard.num_examples} examples"

    finally:
        shutil.rmtree(temp_dir)

    print("  shard_dataset: PASSED")


def test_shard_dataset_single_shard():
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
    """``shard_dataset`` skips empty shards, so a small split caps the trainable DP degree; at runtime a rank
    without a shard fails after a full startup, so the warning names the ceiling."""
    indices = {
        "train": ShardIndex(split="train", num_shards=8, total_examples=800),
        "test": ShardIndex(split="test", num_shards=3, total_examples=3),
    }
    with caplog.at_level(logging.WARNING):
        _warn_on_shard_count_ceiling(indices, requested_shards=8)
    assert "data_parallel_size <= 3" in caplog.text, caplog.text
    assert "test=3" in caplog.text and "train=8" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _warn_on_shard_count_ceiling(
            {"train": ShardIndex(split="train", num_shards=8), "test": ShardIndex(split="test", num_shards=8)},
            requested_shards=8,
        )
    assert caplog.text == ""


def test_max_length_compatibility_raises_both_directions():
    """Smaller preprocessed rows waste the budget; larger ones exceed it, since the runtime preprocessed path never
    re-truncates."""
    metadata = PreprocessedDatasetMetadata(
        model_name="Qwen/Qwen3-8B",
        max_length=4096,
        packed=True,
        num_shards=1,
        total_train_examples=10,
        total_test_examples=1,
    )

    verify_preprocessed_dataset(metadata, required_max_length=4096)

    with pytest.raises(ValueError, match="less than"):
        verify_preprocessed_dataset(metadata, required_max_length=8192)

    with pytest.raises(ValueError, match="exceeds the configured"):
        verify_preprocessed_dataset(metadata, required_max_length=2048)


# Preprocessed rows are baked, so the run's render settings must match the recorded ones.


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
        verify_preprocessed_dataset(
            metadata, required_max_length=4096, render_args=_RenderArgs(system_prompt="You are helpful.")
        )
    with pytest.raises(ValueError, match="conversation_field"):
        verify_preprocessed_dataset(
            metadata, required_max_length=4096, render_args=_RenderArgs(conversation_field="messages")
        )


def test_matching_render_knobs_pass():
    metadata = _metadata_with_recorded_config(system_prompt="sys", tools_field="tools")
    verify_preprocessed_dataset(
        metadata,
        required_max_length=4096,
        render_args=_RenderArgs(system_prompt="sys", tools_field="tools"),
    )


def test_assistant_template_only_checked_when_labels_are_masked():
    """Without completion masking the template is inert; with it, the baked labels were built from the recorded
    template."""
    unmasked = _metadata_with_recorded_config(assistant_message_template="<A>")
    verify_preprocessed_dataset(
        unmasked,
        required_max_length=4096,
        required_train_on_completions_only=False,
        render_args=_RenderArgs(assistant_message_template="<B>"),
    )

    masked = _metadata_with_recorded_config(train_on_completions_only=True, assistant_message_template="<A>")
    with pytest.raises(ValueError, match="assistant_message_template"):
        verify_preprocessed_dataset(
            masked,
            required_max_length=4096,
            required_train_on_completions_only=True,
            render_args=_RenderArgs(assistant_message_template="<B>"),
        )


def test_text_mode_artifact_skips_the_render_check():
    """A raw-text artifact renders no chat template, so a differing run value has nothing baked to disagree with."""
    metadata = _metadata_with_recorded_config(mode="text", text_field="text")
    verify_preprocessed_dataset(
        metadata,
        required_max_length=4096,
        render_args=_RenderArgs(system_prompt="anything", conversation_field="messages"),
    )


def test_legacy_metadata_without_recorded_render_knobs_warns_not_raises(caplog):
    metadata = PreprocessedDatasetMetadata(max_length=4096, config={})
    with caplog.at_level(logging.WARNING):
        verify_preprocessed_dataset(
            metadata, required_max_length=4096, render_args=_RenderArgs(system_prompt="anything")
        )
    assert "predates recording" in caplog.text


def test_packing_against_unpacked_artifact_warns_with_the_consequence(caplog):
    unpacked = _metadata_with_recorded_config(pack_sequences=False)
    with caplog.at_level(logging.WARNING):
        verify_preprocessed_dataset(
            unpacked, required_max_length=4096, render_args=_RenderArgs(), required_packing=True
        )
    assert "pack_sequences" in caplog.text, caplog.text

    caplog.clear()
    packed = _metadata_with_recorded_config(pack_sequences=True)
    packed.packed = True
    with caplog.at_level(logging.WARNING):
        verify_preprocessed_dataset(packed, required_max_length=4096, render_args=_RenderArgs(), required_packing=True)
    assert caplog.text == ""


def test_render_check_set_is_derived_from_the_config_dataclass():
    """Read off each field's ``render_check`` metadata, so a new knob is compared by default. Pinned against an
    independent literal."""
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


def test_tokenizer_override_mismatch_raises():
    """These knobs mutate the tokenizer that bakes the ids; unrecorded, the artifact looks unmutated. An eos
    override moves the completion-mask boundaries."""
    metadata = _metadata_with_recorded_config(eos_token="<|im_end|>")
    with pytest.raises(ValueError, match="eos_token"):
        verify_preprocessed_dataset(metadata, required_max_length=4096, render_args=_RenderArgs(eos_token="</s>"))

    metadata = _metadata_with_recorded_config(pad_token="<pad>")
    with pytest.raises(ValueError, match="pad_token"):
        verify_preprocessed_dataset(metadata, required_max_length=4096, render_args=_RenderArgs())

    matching = _metadata_with_recorded_config(eos_token="</s>", bos_token="<s>")
    verify_preprocessed_dataset(
        matching, required_max_length=4096, render_args=_RenderArgs(eos_token="</s>", bos_token="<s>")
    )


def test_chat_template_compared_as_resolved_text_not_as_a_path(tmp_path):
    """The config records the resolved template while the run may spell it as a file path."""
    template = "{% for m in messages %}{{ m.content }}{% endfor %}"
    template_file = tmp_path / "tpl.jinja"
    template_file.write_text(template)

    metadata = _metadata_with_recorded_config(chat_template=template)
    verify_preprocessed_dataset(
        metadata, required_max_length=4096, render_args=_RenderArgs(chat_template=str(template_file))
    )

    with pytest.raises(ValueError, match="chat_template"):
        verify_preprocessed_dataset(
            metadata, required_max_length=4096, render_args=_RenderArgs(chat_template="{{ 'different' }}")
        )


def test_source_labels_column_is_not_baked_as_loss_targets():
    """A full-sequence bake stores no labels, so a surviving source ``labels`` column would be read as loss targets."""
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
    tokenizer = load_cached_tokenizer(config.model_name_or_path)

    tokenized = tokenize_dataset(dataset, tokenizer, config)
    assert "labels" not in tokenized.column_names, f"source labels leaked into the baked rows: {tokenized[0]!r}"


def test_metadata_version_mismatch_raises_a_version_message():
    """An uncompared stamp lets a diverged schema surface as a TypeError the detection probe swallows into a silent
    raw-path downgrade."""
    payload = asdict(PreprocessedDatasetMetadata(max_length=4096))
    payload["version"] = "0.9"
    with pytest.raises(IncompatiblePreprocessedDataset, match="version"):
        PreprocessedDatasetMetadata.from_dict(payload)

    payload = asdict(PreprocessedDatasetMetadata(max_length=4096))
    payload["invented_by_a_future_build"] = 1
    with pytest.raises(IncompatiblePreprocessedDataset, match="invented_by_a_future_build"):
        PreprocessedDatasetMetadata.from_dict(payload)


def test_metadata_written_before_the_render_knob_renames_is_refused_by_name():
    """A retired spelling is refused: the labels were baked under the old semantics. The message carries the key and
    the re-prep command, the only fix."""
    payload = asdict(PreprocessedDatasetMetadata(max_length=4096, train_on_completions_only=True))
    # Re-spell the top-level key the way a pre-rename writer stamped it.
    payload["train_only_on_completions"] = payload.pop("train_on_completions_only")

    with pytest.raises(IncompatiblePreprocessedDataset) as excinfo:
        PreprocessedDatasetMetadata.from_dict(payload)

    message = str(excinfo.value)
    assert "train_only_on_completions" in message
    assert "scripts/before_training/prepare_dataset.py" in message


def test_hub_repo_id_strips_config_and_split_suffixes():
    """Hub file APIs address the repo; the loader's `org/name:config@split` spelling does not."""
    assert hub_repo_id("org/name") == "org/name"
    assert hub_repo_id("org/name@train_sft") == "org/name"
    assert hub_repo_id("org/name:subset") == "org/name"
    assert hub_repo_id("org/name:subset@train_sft") == "org/name"


def test_hub_preprocessed_dataset_is_detected_and_its_metadata_loads(tmp_path, monkeypatch):
    """Otherwise training takes the raw path and KeyErrors on pre-tokenized rows."""
    payload = asdict(PreprocessedDatasetMetadata(max_length=4096, model_name="Qwen/Qwen3-8B"))
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
    """A raw hub dataset, the common case, probes to False silently."""

    def _missing(repo_id, filename, repo_type=None, **kwargs):
        raise EntryNotFoundError("no metadata.json")

    monkeypatch.setattr(mod, "hf_hub_download", _missing)
    assert mod.is_preprocessed_dataset("HuggingFaceH4/ultrachat_200k") is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
