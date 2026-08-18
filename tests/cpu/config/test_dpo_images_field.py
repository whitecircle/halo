#!/usr/bin/env python
"""DPO's ``images_field`` must reach TRL's vision route under the spelling TRL probes for.

``is_vlm_run`` calls a run multimodal as soon as ``images_field`` is set, while TRL 1.6's
``DPOTrainer`` decides its own vision branch by probing the first sample for an ``image``/``images``
key and lists only those two among its signature columns. A hub dataset storing its images under any
other name therefore has no way in without the alias: the knob does not parse, and parsed alone the
column would still be pruned to text on a run the toolkit already calls multimodal.

Driven through ``main()``: the point is that the alias lands BEFORE the dispatch, so the script's
verdict and TRL's probe read the same dataset.

Run: pytest tests/cpu/config/test_dpo_images_field.py
"""

import sys
import types
from unittest import mock

import pytest
from datasets import Dataset, DatasetDict
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from tests.common.utils import load_script_module

# The name heuristic reads this as multimodal, so the checkpoint verdict holds offline too.
_VLM_MODEL_ID = "stub/qwen3.5-9b"

_TURNS = [{"role": "user", "content": "hi"}]


class _TextPathReached(Exception):
    """Raised by the stubbed preference prep — the run dispatched to the TEXT branch."""


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
    """Run ``preference/dpo.py:main()`` up to its data dispatch on a multimodal checkpoint.

    Returns the ``{split: dataset}`` the vision branch handed on; raises ``_TextPathReached`` when
    the run took the text branch instead, and any config-time guard error unchanged.
    """
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
    )
    reached: dict = {}

    def capture(log_examples, **_kwargs):
        reached.update(log_examples)
        raise _VisionPathReached

    patches = [
        mock.patch.object(module, "init_training_script", return_value=runtime),
        mock.patch.object(
            module,
            "load_model_for_training",
            return_value=(model, types.SimpleNamespace(tokenizer=None), tokenizer, True),
        ),
        mock.patch.object(module, "load_reference_model_for_preference", return_value=None),
        mock.patch.object(module, "apply_max_length", side_effect=lambda cfg, args, model, tok: tok),
        mock.patch.object(module, "install_resolved_tokenizer", side_effect=lambda pc, tok, is_vlm: pc),
        mock.patch.object(module, "setup_peft_model", return_value=None),
        mock.patch.object(module, "log_model_info"),
        mock.patch.object(module, "load_script_datasets", return_value=(dataset, False)),
        # Both moved into the shared script_runner helper the script now calls — patch its globals.
        mock.patch("src.training.script_runner.prepare_preference_datasets", side_effect=_TextPathReached),
        mock.patch("src.training.script_runner.log_dataset_examples", side_effect=capture),
        mock.patch("src.training.parser.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
    ]

    with mock.patch.object(module, "run_training", lambda fn: fn):
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
    """TRL 1.6's own probe, verbatim (``DPOTrainer.__init__``: ``_is_vision_dataset``)."""
    sample = next(iter(dataset))
    return "image" in sample or "images" in sample


def test_the_dispatch_and_trls_probe_agree_on_the_declared_column(tmp_path):
    """The failure mode is a DISAGREEMENT. ``is_vlm_run`` calls the run multimodal off
    ``images_field`` alone, so reaching the vision branch at all is the first verdict; TRL then takes
    its own from the column names, and without the alias it reads the same rows as text and prunes
    the images away. Both verdicts, on the one dataset the trainer receives."""
    reached = _run_dpo(tmp_path, _dataset("image_bytes"), "images_field: image_bytes\n")

    # The shared script_runner helper logs every non-None split, generate included.
    assert set(reached) == {"train", "test", "generate"}
    assert "images" in reached["train"].column_names
    assert "image_bytes" not in reached["train"].column_names
    assert _trl_sees_a_vision_dataset(reached["train"])
    assert _trl_sees_a_vision_dataset(reached["test"])


def test_images_column_is_left_exactly_as_it_is(tmp_path):
    """``images_field: images`` is already TRL's spelling — no rename, no copy."""
    reached = _run_dpo(tmp_path, _dataset("images"), "images_field: images\n")

    assert reached["train"].column_names == ["prompt", "chosen", "rejected", "images"]


def test_text_dataset_without_the_knob_still_takes_the_text_path(tmp_path):
    """Anti-vacuity: the alias is a no-op for the ordinary text preference run."""
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
    sys.exit(pytest.main([__file__, "-v"]))
