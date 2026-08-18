#!/usr/bin/env python
"""The training scripts must route their DATA path on the run, not on the checkpoint class.

``is_vlm_model`` answers "is this checkpoint multimodal", and every natively-multimodal family
(Gemma 4, Qwen3.5/3.6) answers yes — including for the text-only SFT recipes shipped against them.
The VLM data path refuses ``packing``/``padding_free``/``train_on_last_assistant_only``, so keying
the dispatch on the checkpoint makes those recipes raise on the very path they intend. ``is_vlm_run``
adds the run's own declaration of image data; the model still loads through its multimodal class.

Driven through ``main()`` rather than a restatement of the branch: the seam is only correct if the
script reaches it with the dataset in hand, after the model load.

Run: pytest tests/cpu/config/test_sft_modality_dispatch.py
"""

import ast
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
from datasets import Dataset, DatasetDict
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from tests.common.utils import load_script_module

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRAINING_DIR = _REPO_ROOT / "scripts" / "training"

# Every script whose data path branches on modality. One seam, not four copies of it.
_DISPATCHING_SCRIPTS = [
    "sft.py",
    "distillation/self_distill.py",
    "distillation/teacher_distill.py",
    "preference/dpo.py",
]

# A model id the name heuristic reads as multimodal, so the checkpoint verdict is the same offline
# as it is against the hub — otherwise this test would prove nothing about the failure mode it targets.
_VLM_MODEL_ID = "stub/qwen3.5-9b"

_TEXT_TURNS = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


class _TextPathReached(Exception):
    """Raised by the stubbed text prep: main() runs only as far as the dispatch."""


class _VLMPathReached(Exception):
    """Raised by the stubbed VLM prep."""


@pytest.fixture(autouse=True)
def _hub_offline(monkeypatch):
    """No network from this tier: the modality probe must resolve from the name heuristic and the
    already-loaded config, never a hub round-trip."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _script_module(relative_path: str):
    name = f"halo_test_{Path(relative_path).stem}_dispatch"
    return load_script_module(f"scripts/training/{relative_path}", name)


def _dataset(extra_columns: dict | None = None) -> DatasetDict:
    data = {"prompt": [_TEXT_TURNS], **(extra_columns or {})}
    return DatasetDict({"train": Dataset.from_dict(data), "test": Dataset.from_dict(data)})


def _run_sft(tmp_path, yaml_body: str, dataset: DatasetDict, *, stub_vlm_prep: bool, tokenizer=None):
    """Run ``sft.py:main()`` up to the data dispatch against a multimodal checkpoint.

    The VLM prep is left REAL unless ``stub_vlm_prep``: its packing rejection is the exact error a
    misrouted text recipe hits, so that failure mode is what this asserts on. ``tokenizer``
    stands in for the processor's own, whose padding side the run must settle.
    """
    tokenizer = tokenizer if tokenizer is not None else types.SimpleNamespace(padding_side="left")
    module = _script_module("sft.py")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: {_VLM_MODEL_ID}\n"
        f"dataset:\n- dummy/dataset\n"
        f"output_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\nmax_length: 512\n{yaml_body}"
    )

    model = types.SimpleNamespace(config=CONFIG_MAPPING["gemma3"]())
    processing_class = types.SimpleNamespace(tokenizer=None)
    runtime = types.SimpleNamespace(
        parallelism_config=types.SimpleNamespace(cp_size=1, is_cp_mode=False, pp_size=1),
        model_source=_VLM_MODEL_ID,
        mode_suffix="",
        local_rank=0,
        resume_checkpoint=None,
    )

    def fail_text(*_args, **_kwargs):
        raise _TextPathReached

    def fail_vlm(*_args, **_kwargs):
        raise _VLMPathReached

    patches = [
        mock.patch.object(module, "init_training_script", return_value=runtime),
        mock.patch.object(module, "load_model_for_training", return_value=(model, processing_class, tokenizer, True)),
        mock.patch.object(module, "apply_max_length", side_effect=lambda cfg, args, model, tok: tok),
        mock.patch.object(module, "install_resolved_tokenizer", side_effect=lambda pc, tok, is_vlm: pc),
        mock.patch.object(module, "setup_peft_model", return_value=None),
        mock.patch.object(module, "log_model_info"),
        mock.patch.object(module, "load_script_datasets", return_value=((dataset, False), False)),
        mock.patch.object(module, "_prepare_text_data", side_effect=fail_text),
        mock.patch("src.training.parser.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
    ]
    if stub_vlm_prep:
        patches.append(mock.patch.object(module, "_prepare_vlm_data", side_effect=fail_vlm))

    with mock.patch.object(module, "run_training", lambda fn: fn):
        for patch in patches:
            patch.start()
        try:
            module.main()
        finally:
            for patch in reversed(patches):
                patch.stop()


def _run_dpo(tmp_path, dataset: DatasetDict, tokenizer):
    """Run ``preference/dpo.py:main()`` up to its data dispatch against a multimodal checkpoint.

    Same shape as ``_run_sft``: the text branch stops at the preference prep, the VLM branch (which
    has no prep of its own — TRL tokenizes the vision dataset) at the first call past the branch.
    """
    module = _script_module("preference/dpo.py")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: {_VLM_MODEL_ID}\n"
        f"dataset:\n- dummy/dataset\n"
        f"output_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\nmax_length: 512\n"
    )

    model = types.SimpleNamespace(config=CONFIG_MAPPING["gemma3"]())
    processing_class = types.SimpleNamespace(tokenizer=None)
    runtime = types.SimpleNamespace(
        parallelism_config=types.SimpleNamespace(cp_size=1, is_cp_mode=False, pp_size=1),
        model_source=_VLM_MODEL_ID,
        mode_suffix="",
        local_rank=0,
        resume_checkpoint=None,
    )

    def fail_text(*_args, **_kwargs):
        raise _TextPathReached

    def fail_vlm(*_args, **_kwargs):
        raise _VLMPathReached

    patches = [
        mock.patch.object(module, "init_training_script", return_value=runtime),
        mock.patch.object(module, "load_model_for_training", return_value=(model, processing_class, tokenizer, True)),
        mock.patch.object(module, "load_reference_model_for_preference", return_value=None),
        mock.patch.object(module, "apply_max_length", side_effect=lambda cfg, args, model, tok: tok),
        mock.patch.object(module, "install_resolved_tokenizer", side_effect=lambda pc, tok, is_vlm: pc),
        mock.patch.object(module, "setup_peft_model", return_value=None),
        mock.patch.object(module, "log_model_info"),
        mock.patch.object(module, "load_script_datasets", return_value=(dataset, False)),
        # The preference prep and the example log both live in the shared script_runner helper the
        # script now calls, so the two arms are distinguished on ITS globals: the text arm reaches
        # the tokenizing prep first, the VLM arm (which has no prep) reaches the log.
        mock.patch("src.training.script_runner.prepare_preference_datasets", side_effect=fail_text),
        mock.patch("src.training.script_runner.log_dataset_examples", side_effect=fail_vlm),
        mock.patch("src.training.parser.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
    ]

    with mock.patch.object(module, "run_training", lambda fn: fn):
        for patch in patches:
            patch.start()
        try:
            module.main()
        finally:
            for patch in reversed(patches):
                patch.stop()


# --- text recipe on a multimodal checkpoint -------------------------------------------------------


def test_text_dataset_on_a_multimodal_checkpoint_takes_the_text_path_with_packing(tmp_path):
    """End to end: a checkpoint-keyed dispatch raises "packing / padding_free are not supported for
    VLM training" here, and the shipped Gemma 4 / Qwen3.5 text SFT recipes cannot run."""
    with pytest.raises(_TextPathReached):
        _run_sft(tmp_path, "packing: true\n", _dataset(), stub_vlm_prep=False)


def test_text_path_settles_the_padding_side_the_text_collators_require(tmp_path):
    """A multimodal checkpoint hands back the processor's tokenizer, which keeps the checkpoint's own
    side — left for Gemma 4. The packing collator refuses a left-padding tokenizer outright, so
    without this the recipes the dispatch fix unlocks would still not run."""
    tokenizer = types.SimpleNamespace(padding_side="left")
    with pytest.raises(_TextPathReached):
        _run_sft(tmp_path, "packing: true\n", _dataset(), stub_vlm_prep=False, tokenizer=tokenizer)
    assert tokenizer.padding_side == "right"


def test_vlm_run_leaves_the_processor_padding_side_alone(tmp_path):
    """VLM batches pad through the processor — flipping its side would change every VLM run."""
    tokenizer = types.SimpleNamespace(padding_side="left")
    with pytest.raises(_VLMPathReached):
        _run_sft(tmp_path, "", _dataset({"images": [[]]}), stub_vlm_prep=True, tokenizer=tokenizer)
    assert tokenizer.padding_side == "left"


def test_padding_free_text_run_on_a_multimodal_checkpoint_also_reaches_the_text_path(tmp_path):
    with pytest.raises(_TextPathReached):
        _run_sft(tmp_path, "padding_free: true\n", _dataset(), stub_vlm_prep=False)


def test_train_on_last_assistant_only_reaches_the_text_path(tmp_path):
    with pytest.raises(_TextPathReached):
        _run_sft(tmp_path, "train_on_last_assistant_only: true\n", _dataset(), stub_vlm_prep=False)


# --- genuine VLM runs -------------------------------------------------------------------------------


def test_images_field_still_dispatches_to_the_vlm_path(tmp_path):
    with pytest.raises(_VLMPathReached):
        _run_sft(tmp_path, "images_field: images\n", _dataset({"images": [[]]}), stub_vlm_prep=True)


def test_image_column_still_dispatches_to_the_vlm_path(tmp_path):
    """No ``images_field``: the dataset's own column is the declaration."""
    with pytest.raises(_VLMPathReached):
        _run_sft(tmp_path, "", _dataset({"images": [[]]}), stub_vlm_prep=True)


def test_vlm_run_still_rejects_packing(tmp_path):
    """The VLM path's refusals must be intact for runs that really do carry images — the dispatch
    decides which runs reach them, not what they do."""
    with pytest.raises(ValueError, match="packing / padding_free are not supported for VLM"):
        _run_sft(tmp_path, "packing: true\nimages_field: images\n", _dataset({"images": [[]]}), stub_vlm_prep=False)


def test_embedded_image_conversation_still_dispatches_to_the_vlm_path(tmp_path):
    """The SFT-VLM format that ships no images column: the images ride in the content parts."""
    rows = [[{"role": "user", "content": [{"type": "image", "image": "b64"}]}]]
    dataset = DatasetDict({"train": Dataset.from_dict({"prompt": rows}), "test": Dataset.from_dict({"prompt": rows})})
    with pytest.raises(_VLMPathReached):
        _run_sft(tmp_path, "", dataset, stub_vlm_prep=True)


# --- one seam, every script ------------------------------------------------------------------------


def test_dpo_text_path_settles_the_padding_side_too(tmp_path):
    """DPO's text branch runs on the processor's tokenizer whenever the checkpoint is multimodal —
    Gemma 4 / GLM-4.7-Flash keep the checkpoint's LEFT side there, while every text load normalizes
    to right. The text collators inherit that side (the packing collator refuses left outright), so
    the branch owes the same normalization SFT and distillation already apply."""
    tokenizer = types.SimpleNamespace(padding_side="left")
    with pytest.raises(_TextPathReached):
        _run_dpo(tmp_path, _dataset(), tokenizer)
    assert tokenizer.padding_side == "right"


def test_dpo_vision_run_keeps_the_processor_padding_side(tmp_path):
    """An image-carrying preference dataset takes TRL's vision path, which pads through the
    processor — its side must survive untouched."""
    tokenizer = types.SimpleNamespace(padding_side="left")
    with pytest.raises(_VLMPathReached):
        _run_dpo(tmp_path, _dataset({"images": [[]]}), tokenizer)
    assert tokenizer.padding_side == "left"


@pytest.mark.parametrize("script", _DISPATCHING_SCRIPTS)
def test_every_dispatching_script_settles_the_text_path_padding_side(script):
    """The seam is one call, and every script that branches on modality owes it: a script that
    dispatches without it hands the text collators a left-padding tokenizer."""
    tree = ast.parse((_TRAINING_DIR / script).read_text(encoding="utf-8"))
    called = {
        node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "enforce_text_path_padding_side" in called, (
        f"{script} dispatches on modality without settling the text path's padding side"
    )


@pytest.mark.parametrize("script", _DISPATCHING_SCRIPTS)
def test_data_dispatch_goes_through_the_shared_seam(script):
    """Each script that branches its data path on modality must ask ``is_vlm_run``. A local
    ``is_vlm_model`` / column check here is the copy that drifts: it is exactly what made the SFT
    text recipes unrunnable while DPO's own dataset-keyed copy kept working."""
    tree = ast.parse((_TRAINING_DIR / script).read_text(encoding="utf-8"))
    called = {
        node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "is_vlm_run" in called, f"{script} decides its data path without the shared is_vlm_run seam"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
