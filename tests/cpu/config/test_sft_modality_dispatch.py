#!/usr/bin/env python
"""The training scripts route their data path on the run, not the checkpoint class. ``is_vlm_model`` says yes for
every natively multimodal family, and the VLM data path refuses ``packing``, so keying on the checkpoint breaks
the text-only recipes; ``is_vlm_run_across_ranks`` adds the run's declaration of image data."""

import ast
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
from datasets import Dataset, DatasetDict
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from halo.entrypoints import model_setup
from halo.entrypoints.model_setup import PolicyLoad
from halo.model_loading.parallel import model_source
from tests.common.repo import REPO_ROOT, load_script_module

_TRAINING_DIR = REPO_ROOT / "scripts" / "training"

# The run verdict's seam, and the preference scripts' form of it, which also reads the checkpoint config they refuse
# a prefix LM on.
_VLM_RUN_SEAMS = {"resolve_vlm_run", "resolve_preference_vlm_run"}

_DISPATCHING_SCRIPTS = [
    "sft.py",
    "distillation/self_distill.py",
    "distillation/teacher_distill.py",
    "preference/dpo.py",
]

# An id the name heuristic reads as multimodal, so the checkpoint verdict is the same offline as against the hub.
_VLM_MODEL_ID = "stub/qwen3.5-9b"

_TEXT_TURNS = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


class _TextPathReached(Exception):
    """Raised by the stubbed text prep: main() runs only as far as the dispatch."""


class _VLMPathReached(Exception):
    """Raised by the stubbed VLM prep."""


@pytest.fixture(autouse=True)
def _hub_offline(monkeypatch):
    """The modality probe must resolve from the name heuristic, never the hub."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _script_module(relative_path: str):
    name = f"halo_test_{Path(relative_path).stem}_dispatch"
    return load_script_module(f"scripts/training/{relative_path}", name)


def _dataset(extra_columns: dict | None = None) -> DatasetDict:
    data = {"prompt": [_TEXT_TURNS], **(extra_columns or {})}
    return DatasetDict({"train": Dataset.from_dict(data), "test": Dataset.from_dict(data)})


def _run_sft(tmp_path, yaml_body: str, dataset: DatasetDict, *, stub_vlm_prep: bool, tokenizer=None, loads=None):
    """Run ``sft.py:main()`` up to the data dispatch against a multimodal checkpoint. ``tokenizer`` stands in for
    the processor's own, and ``loads`` collects every model load's keyword arguments."""
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
        policy_from_checkpoint=False,
    )

    def fail_text(*_args, **_kwargs):
        raise _TextPathReached

    def fail_vlm(*_args, **_kwargs):
        raise _VLMPathReached

    def load_model(context, *_args, **kwargs):
        assert context is runtime, "the policy load must read the checkpoint identity off the script context"
        if loads is not None:
            loads.append(kwargs)
        return PolicyLoad(model, processing_class, tokenizer, True)

    patches = [
        mock.patch.object(module, "init_training_script", return_value=runtime),
        mock.patch.object(module, "load_policy_model", side_effect=load_model),
        mock.patch.object(module, "apply_max_length", side_effect=lambda cfg, args, model, tok: tok),
        mock.patch.object(module, "install_resolved_tokenizer", side_effect=lambda pc, tok: pc),
        mock.patch.object(module, "configure_adapter_training", return_value=None),
        mock.patch.object(module, "log_model_info"),
        mock.patch.object(module, "load_script_datasets", return_value=((dataset, False), False)),
        mock.patch.object(module, "_prepare_text_data", side_effect=fail_text),
        mock.patch("halo.entrypoints.script.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
    ]
    if stub_vlm_prep:
        patches.append(mock.patch.object(module, "_prepare_vlm_data", side_effect=fail_vlm))

    with mock.patch.object(module, "with_distributed_teardown", lambda fn: fn):
        for patch in patches:
            patch.start()
        try:
            module.main()
        finally:
            for patch in reversed(patches):
                patch.stop()


def _run_dpo(tmp_path, dataset: DatasetDict, tokenizer):
    """Run ``preference/dpo.py:main()`` up to its data dispatch: the text branch stops at the preference prep, the
    VLM branch at the first call past it."""
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
        policy_from_checkpoint=False,
    )

    def fail_text(*_args, **_kwargs):
        raise _TextPathReached

    def fail_vlm(*_args, **_kwargs):
        raise _VLMPathReached

    def load_model(context, *_args, **kwargs):
        assert context is runtime, "the policy load must read the checkpoint identity off the script context"
        return PolicyLoad(model, processing_class, tokenizer, True)

    patches = [
        mock.patch.object(module, "init_training_script", return_value=runtime),
        # The policy-and-reference load DPO shares with KTO runs for real around these stubs.
        mock.patch.object(model_setup, "load_policy_model", side_effect=load_model),
        mock.patch.object(model_setup, "load_reference_model_for_preference", return_value=None),
        mock.patch.object(model_setup, "apply_max_length", side_effect=lambda cfg, args, model, tok: tok),
        mock.patch.object(model_setup, "install_resolved_tokenizer", side_effect=lambda pc, tok: pc),
        mock.patch.object(model_setup, "configure_adapter_training", return_value=None),
        mock.patch.object(model_setup, "log_model_info"),
        mock.patch.object(module, "load_script_datasets", return_value=(dataset, False)),
        # Both live in the shared data_setup helper, so the arms are distinguished on its globals: the text arm reaches
        # the prep first, the VLM arm the log.
        mock.patch("halo.entrypoints.data_setup.prepare_preference_datasets", side_effect=fail_text),
        mock.patch("halo.entrypoints.data_setup.log_dataset_examples", side_effect=fail_vlm),
        mock.patch("halo.entrypoints.script.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
    ]

    with mock.patch.object(module, "with_distributed_teardown", lambda fn: fn):
        for patch in patches:
            patch.start()
        try:
            module.main()
        finally:
            for patch in reversed(patches):
                patch.stop()


def test_text_dataset_on_a_multimodal_checkpoint_takes_the_text_path_with_packing(tmp_path):
    """A checkpoint-keyed dispatch raises the VLM packing refusal here, and the shipped Gemma 4 / Qwen3.5 text
    recipes cannot run."""
    with pytest.raises(_TextPathReached):
        _run_sft(tmp_path, "packing: true\n", _dataset(), stub_vlm_prep=False)


def test_text_path_settles_the_padding_side_the_text_collators_require(tmp_path):
    """A multimodal checkpoint hands back the processor's tokenizer with the checkpoint's own side, left for Gemma
    4, which the packing collator refuses."""
    tokenizer = types.SimpleNamespace(padding_side="left")
    with pytest.raises(_TextPathReached):
        _run_sft(tmp_path, "packing: true\n", _dataset(), stub_vlm_prep=False, tokenizer=tokenizer)
    assert tokenizer.padding_side == "right"


def test_vlm_run_leaves_the_processor_padding_side_alone(tmp_path):
    """VLM batches pad through the processor, so flipping its side would change every VLM run."""
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


def test_images_field_still_dispatches_to_the_vlm_path(tmp_path):
    with pytest.raises(_VLMPathReached):
        _run_sft(tmp_path, "images_field: images\n", _dataset({"images": [[]]}), stub_vlm_prep=True)


def test_image_column_still_dispatches_to_the_vlm_path(tmp_path):
    """No ``images_field``: the dataset's own column is the declaration."""
    with pytest.raises(_VLMPathReached):
        _run_sft(tmp_path, "", _dataset({"images": [[]]}), stub_vlm_prep=True)


def test_vlm_run_still_rejects_packing(tmp_path):
    """The dispatch decides which runs reach the refusals, not what they do. They are pure config, so they land
    before the model load."""
    loads = []
    with pytest.raises(ValueError, match="packing / padding_free are not supported for VLM"):
        _run_sft(
            tmp_path,
            "packing: true\nimages_field: images\n",
            _dataset({"images": [[]]}),
            stub_vlm_prep=False,
            loads=loads,
        )
    assert not loads


def test_embedded_image_conversation_still_dispatches_to_the_vlm_path(tmp_path):
    """The SFT-VLM format that ships no images column: the images ride in the content parts."""
    rows = [[{"role": "user", "content": [{"type": "image", "image": "b64"}]}]]
    dataset = DatasetDict({"train": Dataset.from_dict({"prompt": rows}), "test": Dataset.from_dict({"prompt": rows})})
    with pytest.raises(_VLMPathReached):
        _run_sft(tmp_path, "", dataset, stub_vlm_prep=True)


def test_text_only_model_refuses_embedded_images_before_the_model_load(tmp_path):
    """Embedded parts would route a text-only CausalLM load to the VLM data path, where a bare tokenizer stands in
    for the processor and the pixels are dropped."""
    rows = [[{"role": "user", "content": [{"type": "image", "image": "b64"}]}]]
    dataset = DatasetDict({"train": Dataset.from_dict({"prompt": rows}), "test": Dataset.from_dict({"prompt": rows})})
    loads = []
    with pytest.raises(ValueError, match="text_only_model=True loads the text-only CausalLM"):
        _run_sft(tmp_path, "text_only_model: true\n", dataset, stub_vlm_prep=True, loads=loads)
    assert not loads


def test_the_checkpoint_config_is_read_once(tmp_path):
    """The run prefix and the data dispatch both need the modality; the dispatch takes the verdict the prefix
    already probed."""
    probes = []
    with (
        mock.patch.object(
            model_source, "_probe_checkpoint_config", side_effect=lambda *args, **kwargs: probes.append(args)
        ),
        pytest.raises(_TextPathReached),
    ):
        _run_sft(tmp_path, "", _dataset(), stub_vlm_prep=False)
    assert len(probes) == 1, probes


def test_the_run_verdict_reaches_the_model_load(tmp_path):
    """Only an image run requires the processor, and a multimodal checkpoint may ship none (Step-3.7 Flash)."""
    text_loads, image_loads = [], []
    with pytest.raises(_TextPathReached):
        _run_sft(tmp_path, "", _dataset(), stub_vlm_prep=False, loads=text_loads)
    with pytest.raises(_VLMPathReached):
        _run_sft(tmp_path, "", _dataset({"images": [[]]}), stub_vlm_prep=True, loads=image_loads)

    assert [load["vlm_run"] for load in text_loads] == [False]
    assert [load["vlm_run"] for load in image_loads] == [True]


def test_dpo_text_path_settles_the_padding_side_too(tmp_path):
    """DPO's text branch runs on the processor's tokenizer on a multimodal checkpoint, whose left side the text
    collators inherit (the packing collator refuses left)."""
    tokenizer = types.SimpleNamespace(padding_side="left")
    with pytest.raises(_TextPathReached):
        _run_dpo(tmp_path, _dataset(), tokenizer)
    assert tokenizer.padding_side == "right"


def test_dpo_vision_run_keeps_the_processor_padding_side(tmp_path):
    """TRL's vision path pads through the processor, so its side must survive."""
    tokenizer = types.SimpleNamespace(padding_side="left")
    with pytest.raises(_VLMPathReached):
        _run_dpo(tmp_path, _dataset({"images": [[]]}), tokenizer)
    assert tokenizer.padding_side == "left"


@pytest.mark.parametrize("script", _DISPATCHING_SCRIPTS)
def test_every_dispatching_script_settles_the_text_path_padding_side(script):
    """A script dispatching without the seam hands the text collators a left-padding tokenizer."""
    tree = ast.parse((_TRAINING_DIR / script).read_text(encoding="utf-8"))
    called = {
        node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "apply_text_path_padding_side" in called, (
        f"{script} dispatches on modality without settling the text path's padding side"
    )


@pytest.mark.parametrize("script", _DISPATCHING_SCRIPTS)
def test_data_dispatch_goes_through_the_shared_seam(script):
    """A local ``is_vlm_model`` or column check is a copy that drifts: a checkpoint-keyed one makes the script's
    text recipes unrunnable."""
    tree = ast.parse((_TRAINING_DIR / script).read_text(encoding="utf-8"))
    called = {
        node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert called & _VLM_RUN_SEAMS, f"{script} decides its data path without the shared resolve_vlm_run seam"


# The modality-aware loader and the preference load built on it, each taking the run's verdict.
_POLICY_LOADS = ("load_policy_model", "load_preference_models")
_LOADING_SCRIPTS = sorted(
    str(path.relative_to(_TRAINING_DIR))
    for path in _TRAINING_DIR.rglob("*.py")
    if any(f"{load}(" in path.read_text(encoding="utf-8") for load in _POLICY_LOADS)
)


def test_the_loading_script_roster_is_not_vacuous():
    assert set(_DISPATCHING_SCRIPTS) <= set(_LOADING_SCRIPTS), _LOADING_SCRIPTS


@pytest.mark.parametrize("script", _LOADING_SCRIPTS)
def test_every_model_load_takes_the_run_verdict(script):
    """A constant or the checkpoint's verdict either refuses the text recipes on a checkpoint shipping no processor
    or hands an image run a bare tokenizer."""
    tree = ast.parse((_TRAINING_DIR / script).read_text(encoding="utf-8"))
    verdicts = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) in _VLM_RUN_SEAMS
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) in _POLICY_LOADS
    ]
    assert verdicts and loads, f"{script} no longer both decides the run and loads through the shared loader"
    for call in loads:
        passed = [kw.value for kw in call.keywords if kw.arg == "vlm_run"]
        assert len(passed) == 1 and isinstance(passed[0], ast.Name) and passed[0].id in verdicts, (
            f"{script}: {call.func.id} is not handed the resolve_vlm_run verdict as vlm_run"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
