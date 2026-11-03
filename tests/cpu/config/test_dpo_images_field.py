#!/usr/bin/env python
"""DPO's ``images_field`` reaches TRL's vision route under the spelling TRL probes for. ``is_vlm_run`` calls a run
multimodal off the field alone while TRL 1.14 probes for an ``image``/``images`` key, so the alias must land
before the dispatch, or the column is pruned to text."""

import sys
import types
from unittest import mock

import pytest
from datasets import Dataset, DatasetDict
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from halo.entrypoints.model_setup import PreferenceModels
from tests.common.repo import load_script_module

# The name heuristic reads this as multimodal, so the checkpoint verdict holds offline too.
_VLM_MODEL_ID = "stub/qwen3.5-9b"

_TURNS = [{"role": "user", "content": "hi"}]


class _TextPathReached(Exception):
    """Raised by the stubbed preference prep: the run dispatched to the text branch."""


class _VisionPathReached(Exception):
    """Raised once the vision branch has handed its datasets on."""


@pytest.fixture(autouse=True)
def _hub_offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _dpo_module():
    return load_script_module("scripts/training/preference/dpo.py", "halo_test_dpo_images_field")


def _dataset(image_column: str | None) -> DatasetDict:
    data = {"prompt": [_TURNS], "chosen": [_TURNS], "rejected": [_TURNS]}
    if image_column is not None:
        data[image_column] = [[]]
    return DatasetDict({"train": Dataset.from_dict(data), "test": Dataset.from_dict(data)})


def _run_dpo(tmp_path, dataset: DatasetDict, yaml_body: str = "") -> dict:
    """Run ``preference/dpo.py:main()`` up to its data dispatch on a multimodal checkpoint; returns the ``{split:
    dataset}`` the vision branch handed on, and raises ``_TextPathReached`` on the text branch."""
    module = _dpo_module()
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: {_VLM_MODEL_ID}\n"
        f"dataset:\n- dummy/dataset\n"
        f"output_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\nmax_length: 512\n{yaml_body}"
    )

    model = types.SimpleNamespace(config=CONFIG_MAPPING["gemma3"]())
    tokenizer = types.SimpleNamespace(padding_side="left")
    runtime = types.SimpleNamespace(
        parallelism_config=types.SimpleNamespace(cp_size=1, is_cp_mode=False, pp_size=1),
        model_source=_VLM_MODEL_ID,
        mode_suffix="",
        local_rank=0,
        resume_checkpoint=None,
        policy_from_checkpoint=False,
    )
    reached: dict = {}

    def capture(log_examples, **_kwargs):
        reached.update(log_examples)
        raise _VisionPathReached

    def load_models(context, *_args, **kwargs):
        assert context is runtime, "the policy load must read the checkpoint identity off the script context"
        return PreferenceModels(model, None, None, tokenizer, types.SimpleNamespace(tokenizer=None))

    patches = [
        mock.patch.object(module, "init_training_script", return_value=runtime),
        mock.patch.object(module, "load_preference_models", side_effect=load_models),
        mock.patch.object(module, "load_script_datasets", return_value=(dataset, False)),
        # Both live in the shared data_setup helper the script calls, so patch its globals.
        mock.patch("halo.entrypoints.data_setup.prepare_preference_datasets", side_effect=_TextPathReached),
        mock.patch("halo.entrypoints.data_setup.log_dataset_examples", side_effect=capture),
        mock.patch("halo.entrypoints.script.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
    ]

    with mock.patch.object(module, "with_distributed_teardown", lambda fn: fn):
        for patch in patches:
            patch.start()
        try:
            module.main()
        except _VisionPathReached:
            return reached
        finally:
            for patch in reversed(patches):
                patch.stop()
    raise AssertionError("main() returned without reaching either data branch")


def _trl_sees_a_vision_dataset(dataset: Dataset) -> bool:
    """TRL 1.14's own probe, verbatim (``DPOTrainer.__init__``: ``_is_vision_dataset``)."""
    sample = next(iter(dataset))
    return "image" in sample or "images" in sample


def test_the_dispatch_and_trls_probe_agree_on_the_declared_column(tmp_path):
    """``is_vlm_run`` and TRL's probe must agree on the one dataset the trainer receives; without the alias TRL
    reads the rows as text and prunes the images."""
    reached = _run_dpo(tmp_path, _dataset("image_bytes"), "images_field: image_bytes\n")

    # The shared data_setup helper logs every non-None split, generate included.
    assert set(reached) == {"train", "test", "generate"}
    assert "images" in reached["train"].column_names
    assert "image_bytes" not in reached["train"].column_names
    assert _trl_sees_a_vision_dataset(reached["train"])
    assert _trl_sees_a_vision_dataset(reached["test"])


def test_images_column_is_left_exactly_as_it_is(tmp_path):
    reached = _run_dpo(tmp_path, _dataset("images"), "images_field: images\n")

    assert reached["train"].column_names == ["prompt", "chosen", "rejected", "images"]


def test_text_dataset_without_the_knob_still_takes_the_text_path(tmp_path):
    with pytest.raises(_TextPathReached):
        _run_dpo(tmp_path, _dataset(None))


def test_mistyped_column_raises_naming_the_knob(tmp_path):
    """A typo would otherwise drop every image and train text on a run declared multimodal."""
    with pytest.raises(ValueError, match="images_field='pictures' names a column"):
        _run_dpo(tmp_path, _dataset("image_bytes"), "images_field: pictures\n")


def test_alias_onto_an_occupied_images_column_raises(tmp_path):
    """Two image columns: renaming would clobber one of them, so say which is in the way."""
    dataset = _dataset("images")
    dataset = DatasetDict({name: split.add_column("image_bytes", [[]]) for name, split in dataset.items()})

    with pytest.raises(ValueError, match="already carry one"):
        _run_dpo(tmp_path, dataset, "images_field: image_bytes\n")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
