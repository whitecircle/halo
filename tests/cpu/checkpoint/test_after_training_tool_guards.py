#!/usr/bin/env python
"""Guards on the after-training tools at the seams where they otherwise warn, default, or drop in
silence.

* ``merge_models.py`` printing "the merged checkpoint has none" about a missing tokenizer and
  exiting 0 hands the next stage an artifact no ``from_pretrained`` consumer can build a tokenizer
  for.
* ``reset_sinks.py`` defaulting ``--output_dir`` to its input lets a mistyped command rewrite the
  only copy of a checkpoint with no undo. In-place is a flag you ask for.
* ``convert_to_bf16.py --peft --merge_adapter`` copying the ADAPTER directory's aux files into the
  merged checkpoint lands ``adapter_config.json`` beside full merged weights, and every
  from_pretrained-based tool downstream then reads the result as an unmerged adapter — and
  re-exports the bare base.
* ``dataset_deduplication.py`` silently ignoring an ``--additional_fields`` name the dataset does not
  carry writes an output missing exactly the column the caller asked to keep.

Run: pytest tests/cpu/checkpoint/test_after_training_tool_guards.py
"""

import json
import sys
from pathlib import Path

import pytest
from datasets import Dataset

from scripts.after_training import convert_to_bf16, merge_models, reset_sinks
from scripts.inference.generation import dataset_deduplication
from src.checkpoint import adapters

# --- merge_models: a tokenizer-less artifact is not a deliverable ---------------------------------


def _source_without_tokenizer(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}))
    return source


def test_a_merge_whose_source_ships_no_tokenizer_raises_and_names_the_flag(tmp_path):
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(RuntimeError) as excinfo:
        merge_models._copy_aux_files(
            str(_source_without_tokenizer(tmp_path)),
            str(out),
            "bfloat16",
            verbose=False,
            allow_missing_tokenizer=False,
            trust_remote_code=True,
        )

    message = str(excinfo.value)
    assert "--tokenizer_source" in message, "the refusal must name the knob that fixes it"
    assert "--allow_missing_tokenizer" in message, "the refusal must name its opt-out"


def test_the_opt_out_accepts_a_tokenizer_less_artifact(tmp_path):
    """The other side of the boundary — the guard must be escapable on purpose, or a legitimate
    tokenizer-less export becomes a dead end."""
    out = tmp_path / "out"
    out.mkdir()

    merge_models._copy_aux_files(
        str(_source_without_tokenizer(tmp_path)),
        str(out),
        "bfloat16",
        verbose=False,
        allow_missing_tokenizer=True,
        trust_remote_code=True,
    )

    assert (out / "config.json").is_file()


# --- reset_sinks: in-place is asked for, never defaulted ------------------------------------------


def test_reset_sinks_refuses_to_pick_an_output_directory_for_you(tmp_path):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()

    with pytest.raises(ValueError) as excinfo:
        reset_sinks.reset_sinks(str(checkpoint))

    message = str(excinfo.value)
    assert "--output_dir is required" in message
    assert "--in_place" in message, "the refusal must name the deliberate in-place path"


def test_reset_sinks_refuses_an_output_dir_aimed_at_its_own_input(tmp_path):
    """The same mistake through the other flag: --output_dir == --checkpoint_dir IS the in-place run
    the explicit flag exists to make deliberate."""
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()

    with pytest.raises(ValueError, match="input and output directory are the same path"):
        reset_sinks.reset_sinks(str(checkpoint), output_dir=str(checkpoint))


def test_in_place_and_output_dir_together_are_contradictory(tmp_path):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()

    with pytest.raises(ValueError, match="cannot be combined with --output_dir"):
        reset_sinks.reset_sinks(str(checkpoint), output_dir=str(tmp_path / "other"), in_place=True)


def test_in_place_cannot_target_a_hub_repo_id(tmp_path):
    with pytest.raises(ValueError, match="HuggingFace repo ID"):
        reset_sinks.reset_sinks("org/model", in_place=True)


def test_one_spelling_decides_what_a_sink_key_is():
    """The scan, the reset and the verification all read this predicate; three literals could
    verify tensors they never reset."""
    assert reset_sinks.is_sink_key("model.layers.0.self_attn.sinks")
    assert not reset_sinks.is_sink_key("model.layers.0.self_attn.q_proj.weight")


# --- convert_to_bf16: the merged branch takes the BASE's aux files --------------------------------


def test_a_merged_peft_conversion_copies_the_base_directory_not_the_adapter(tmp_path, monkeypatch):
    """THE regression: with the adapter directory as ``source_dir`` the save copied
    ``adapter_config.json`` next to full merged weights, and downstream tools re-exported the base.

    Driven through the tool's own entry point, so it also pins that ``--peft --merge_adapter`` still
    reaches the shared merge rather than growing a second copy of the sequence.
    """
    adapter = tmp_path / "adapter"
    base = tmp_path / "base"
    for directory in (adapter, base):
        directory.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    captured: dict = {}

    class _FakePeftConfig:
        base_model_name_or_path = str(base)
        peft_type = "LORA"

        @classmethod
        def from_pretrained(cls, _path):
            return cls()

        def save_pretrained(self, path):
            captured["adapter_config_written_to"] = path

    class _Merged:
        """The merged model, reduced to what the conversion actually touches."""

        def merge_and_unload(self):
            return self

        def modules(self):
            return []

    class _FakePeftModel:
        @staticmethod
        def from_pretrained(base_model, _adapter_dir):
            return base_model

    monkeypatch.setattr(adapters, "PeftConfig", _FakePeftConfig)
    monkeypatch.setattr(adapters, "PeftModel", _FakePeftModel)
    monkeypatch.setattr(adapters, "assert_no_expert_lora_adapter", lambda _path: None)
    monkeypatch.setattr(adapters, "resolve_peft_processing_class", lambda *a, **k: object())
    monkeypatch.setattr(adapters, "preflight_model_load_resources", lambda *a, **k: None)
    monkeypatch.setattr(adapters, "reject_sharded_checkpoint", lambda _path: None)
    monkeypatch.setattr(adapters, "apply_training_sidecars", lambda *a, **k: [])
    monkeypatch.setattr(adapters, "copy_training_sidecars", lambda *a, **k: None)
    monkeypatch.setattr(
        adapters,
        "save_full_checkpoint",
        lambda model, output_dir, **kwargs: captured.update(source_dir=kwargs.get("source_dir")),
    )
    monkeypatch.setattr(convert_to_bf16, "_load_verified", lambda *a, **k: _Merged())

    convert_to_bf16.convert_to_bf16(str(adapter), str(tmp_path / "out"), "causal_lm", is_peft=True, merge_adapter=True)

    assert captured["source_dir"] == str(base), "the merged checkpoint must inherit the BASE's aux files"
    assert captured["adapter_config_written_to"].endswith("original_adapter_config"), (
        "the adapter's own config must be relocated, not dropped beside the merged weights"
    )


# --- dataset_deduplication: an unknown keep-column is a typo, not an opt-out ----------------------


def test_an_unknown_additional_field_raises(tmp_path):
    dataset = Dataset.from_dict({"text": ["a", "b"], "id": [1, 2]})

    with pytest.raises(ValueError) as excinfo:
        dataset_deduplication.save_deduplicated_dataset(
            dataset,
            __import__("numpy").array([0, 1]),
            str(tmp_path / "out.jsonl"),
            additional_fields=["text", "sorce"],
        )

    assert "'sorce'" in str(excinfo.value) or "sorce" in str(excinfo.value)


def test_known_additional_fields_still_narrow_the_output(tmp_path):
    """Anti-vacuity: the guard must not refuse the working case it exists around."""
    import numpy as np

    dataset = Dataset.from_dict({"text": ["a", "b"], "id": [1, 2]})
    output = tmp_path / "out.jsonl"

    dataset_deduplication.save_deduplicated_dataset(dataset, np.array([0]), str(output), additional_fields=["text"])

    written = json.loads(output.read_text().splitlines()[0])
    assert written == {"text": "a"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
