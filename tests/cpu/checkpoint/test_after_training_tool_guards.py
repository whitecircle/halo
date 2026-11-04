#!/usr/bin/env python
"""Guards on the after-training tools at the seams where they would otherwise warn, default, or drop in silence."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset

from halo.checkpoint.tools import adapter_merge, model_merge, sink_reset
from halo.checkpoint.tools.adapter_merge import DEFAULT_NUM_LABELS
from halo.data import deduplication
from tests.common.repo import load_script_module


def _source_without_tokenizer(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}))
    return source


def test_a_merge_whose_source_ships_no_tokenizer_raises_and_names_the_flag(tmp_path):
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(RuntimeError) as excinfo:
        model_merge._copy_aux_files(
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
    """A legitimate tokenizer-less export must not become a dead end."""
    out = tmp_path / "out"
    out.mkdir()

    model_merge._copy_aux_files(
        str(_source_without_tokenizer(tmp_path)),
        str(out),
        "bfloat16",
        verbose=False,
        allow_missing_tokenizer=True,
        trust_remote_code=True,
    )

    assert (out / "config.json").is_file()


def test_merge_models_refuses_a_hub_tokenizer_source(tmp_path):
    """A Hub id here would run a third-party repo's config code under the local default."""
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="--tokenizer_source 'org/third-party' is not a local directory.*hf download"):
        model_merge.merge_models(["/nope/a", "/nope/b"], str(out), method="linear", tokenizer_source="org/third-party")
    assert not out.exists(), "a refused merge left an output directory behind"


def test_reset_sinks_refuses_to_pick_an_output_directory_for_you(tmp_path):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()

    with pytest.raises(ValueError) as excinfo:
        sink_reset.reset_sinks(str(checkpoint))

    message = str(excinfo.value)
    assert "--output_dir is required" in message
    assert "--in_place" in message, "the refusal must name the deliberate in-place path"


def test_reset_sinks_refuses_an_output_dir_aimed_at_its_own_input(tmp_path):
    """--output_dir == --model_id is the in-place run the explicit flag exists to make deliberate."""
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()

    with pytest.raises(ValueError, match="input and output directory are the same path"):
        sink_reset.reset_sinks(str(checkpoint), output_dir=str(checkpoint))


def test_in_place_and_output_dir_together_are_contradictory(tmp_path):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()

    with pytest.raises(ValueError, match="cannot be combined with --output_dir"):
        sink_reset.reset_sinks(str(checkpoint), output_dir=str(tmp_path / "other"), in_place=True)


@pytest.mark.parametrize("source", ["org/model", "missing-checkpoint"], ids=["hub-repo-id", "missing-path"])
def test_in_place_needs_a_local_directory(source, tmp_path, monkeypatch):
    """Refused before anything is fetched."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sink_reset, "resolve_checkpoint_source", lambda *a, **k: pytest.fail(f"--in_place fetched {source}")
    )

    with pytest.raises(ValueError, match="not a local directory"):
        sink_reset.reset_sinks(source, in_place=True)


def test_one_spelling_decides_what_a_sink_key_is():
    """The scan, the reset and the verification all read this predicate; three literals could verify tensors they
    never reset."""
    assert sink_reset.is_sink_key("model.layers.0.self_attn.sinks")
    assert not sink_reset.is_sink_key("model.layers.0.self_attn.q_proj.weight")


def test_a_peft_merge_copies_the_base_directory_not_the_adapter(tmp_path, monkeypatch):
    """The adapter's ``adapter_config.json`` next to merged weights makes downstream tools re-export the base."""
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
        """What the conversion actually touches."""

        def merge_and_unload(self):
            return self

        def modules(self):
            return []

    class _FakePeftModel:
        @staticmethod
        def from_pretrained(base_model, _adapter_dir):
            return base_model

    monkeypatch.setattr(adapter_merge, "PeftConfig", _FakePeftConfig)
    monkeypatch.setattr(adapter_merge, "PeftModel", _FakePeftModel)
    monkeypatch.setattr(adapter_merge, "reject_expert_lora_adapter", lambda _path: None)
    monkeypatch.setattr(adapter_merge, "resolve_peft_processing_class", lambda *a, **k: object())
    monkeypatch.setattr(adapter_merge, "preflight_model_load_resources", lambda *a, **k: None)
    monkeypatch.setattr(adapter_merge, "reject_partial_checkpoint", lambda _path: None)
    monkeypatch.setattr(adapter_merge, "apply_training_sidecars", lambda *a, **k: [])
    monkeypatch.setattr(adapter_merge, "copy_training_sidecars", lambda *a, **k: None)
    monkeypatch.setattr(
        adapter_merge,
        "save_full_checkpoint",
        lambda model, output_dir, **kwargs: captured.update(source_dir=kwargs.get("source_dir")),
    )
    monkeypatch.setattr(adapter_merge, "load_task_base_model", lambda *a, **k: _Merged())

    adapter_merge.merge_peft_adapter(str(adapter), str(tmp_path / "out"), fp32_norms=True, verbose=False)

    assert captured["source_dir"] == str(base), "the merged checkpoint must inherit the BASE's aux files"
    assert captured["adapter_config_written_to"].endswith("original_adapter_config"), (
        "the adapter's own config must be relocated, not dropped beside the merged weights"
    )


def test_an_unknown_keep_field_raises(tmp_path):
    dataset = Dataset.from_dict({"text": ["a", "b"], "id": [1, 2]})

    with pytest.raises(ValueError) as excinfo:
        deduplication.save_deduplicated_dataset(
            dataset,
            np.array([0, 1]),
            str(tmp_path / "out.jsonl"),
            keep_fields=["text", "sorce"],
        )

    assert "'sorce'" in str(excinfo.value) or "sorce" in str(excinfo.value)


def test_known_keep_fields_still_narrow_the_output(tmp_path):
    dataset = Dataset.from_dict({"text": ["a", "b"], "id": [1, 2]})
    output = tmp_path / "out.jsonl"

    deduplication.save_deduplicated_dataset(dataset, np.array([0]), str(output), keep_fields=["text"])

    written = json.loads(output.read_text().splitlines()[0])
    assert written == {"text": "a"}


_DEDUP_ARGV = ["deduplicate_dataset.py", "--dataset", "d", "--output_path", "/out/rows.jsonl", "--text_field", "text"]


def _parse_dedup(monkeypatch, *flags: str) -> argparse.Namespace:
    module = load_script_module("scripts/before_training/deduplicate_dataset.py", "halo_test_dedup_embedding_flags")
    monkeypatch.setattr(sys, "argv", [*_DEDUP_ARGV, *flags])
    return module.parse_arguments()


def test_dedup_takes_one_embeddings_output_flag(monkeypatch):
    assert _parse_dedup(monkeypatch).save_embeddings is None
    assert _parse_dedup(monkeypatch, "--save_embeddings").save_embeddings == "/out/rows.jsonl.embeddings.npy"
    assert _parse_dedup(monkeypatch, "--save_embeddings", "/x.npy").save_embeddings == "/x.npy"
    with pytest.raises(SystemExit):
        _parse_dedup(monkeypatch, "--embeddings_output_path", "/x.npy")


def test_dedup_hands_the_embedding_step_the_parsed_path(monkeypatch, tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"text": "a"}\n{"text": "b"}\n')
    save_paths = []

    def compute(**kwargs):
        save_paths.append(kwargs["save_path"])
        return np.eye(2, dtype=np.float32)

    monkeypatch.setattr(deduplication, "compute_embeddings", compute)
    args = _parse_dedup(
        monkeypatch, "--save_embeddings", "/x.npy", "--dry_run", "--device", "cpu", "--dedup_steps", "1"
    )
    args.dataset = str(rows)
    assert deduplication.run_deduplication(args) == 0
    assert save_paths == ["/x.npy"]


@pytest.mark.parametrize("save", [["--save_embeddings"], ["--save_embeddings", "/x.npy"]], ids=["bare", "path"])
def test_dedup_refuses_saving_embeddings_it_only_loads(monkeypatch, capsys, save):
    with pytest.raises(SystemExit):
        _parse_dedup(monkeypatch, "--load_embeddings", "/y.npy", *save)
    assert "--load_embeddings computes none" in capsys.readouterr().err


_MERGE_ARGV = ["merge_peft_adapters.py", "--adapter_dir", "a", "--output_dir", "o"]


def _merge_labels(monkeypatch, *flags: str) -> list[int]:
    seen: list[int] = []
    module = load_script_module("scripts/after_training/merge_peft_adapters.py", "halo_test_merge_num_labels")
    monkeypatch.setattr(module, "merge_peft_adapter", lambda **kwargs: seen.append(kwargs["num_labels"]))
    monkeypatch.setattr(sys, "argv", [*_MERGE_ARGV, *flags])
    module.main()
    return seen


def test_merge_refuses_a_label_count_without_a_classification_head(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _merge_labels(monkeypatch, "--num_labels", "3")
    assert "--task causal_lm loads none" in capsys.readouterr().err


def test_merge_keeps_its_label_default_on_a_classification_head(monkeypatch):
    assert _merge_labels(monkeypatch) == [DEFAULT_NUM_LABELS]
    assert _merge_labels(monkeypatch, "--task", "classification") == [DEFAULT_NUM_LABELS]
    assert _merge_labels(monkeypatch, "--task", "classification", "--num_labels", "3") == [3]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
