#!/usr/bin/env python
"""The pipeline-parallel multimodal gate must be given the model's NAME, not an empty string.

``is_vlm_model``'s config verdict is authoritative only for an architecture transformers KNOWS: an
UNREGISTERED ``model_type`` (remote code, a custom class) is in no ``AutoModelForImageTextToText``
mapping and often declares no ``vision_config``, so the name-substring heuristic is the only signal
left. Called with ``""`` that heuristic could never fire, and such a VLM fed images under PP would
train with its vision tower dropped from every stage while the saved config still declared it —
silent on both counts. A multimodal wrapper is admitted under PP only for a run that feeds it no
images (its tower is dead weight the save re-emits), so the refusal is driven with an image-bearing
dataset in hand: it fires exactly when the gate saw a VLM.

The gate is driven through ``_maybe_prepare_pipeline_model`` itself: asserting on ``is_vlm_model``
would pass with the whole PP call site deleted.

Run: python tests/cpu/trainers/test_pp_vlm_name_gate.py
"""

from types import SimpleNamespace
from unittest import mock

import pytest
from datasets import Dataset

import src.models.modality as modality
from src.trainers.mixins.pipeline import PipelineTrainerMixin

_VLM_REJECTION = "Vision-language training is not supported under pipeline parallelism"


def _mixin():
    """A trainer-shaped stub: the split dispatches the ``_validate_pp_mode`` hook through ``self``."""
    stub = object.__new__(PipelineTrainerMixin)
    stub.parallelism_config = SimpleNamespace(is_pp_mode=True)
    stub.save_sharded_ep = False
    return stub


def _args():
    return SimpleNamespace(
        max_length=128,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        eval_strategy="no",
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        torch_compile=False,
        activation_offloading=False,
    )


def _image_dataset() -> Dataset:
    return Dataset.from_dict({"prompt": ["describe"], "images": [[]]})


def _drive(model_type, name, *, vision_config=None):
    """Run the PP preparation over a model whose config declares ``model_type`` and ``name``, with an
    image-bearing train dataset. Returns the gate's own exception, or None when the model got past
    it (anything raised further down — the split needs a real module — counts as "past")."""
    config = SimpleNamespace(model_type=model_type, _name_or_path=name)
    if vision_config is not None:
        config.vision_config = vision_config
    kwargs = {"model": SimpleNamespace(config=config), "train_dataset": _image_dataset()}
    try:
        PipelineTrainerMixin._maybe_prepare_pipeline_model(_mixin(), kwargs, _args())
    except Exception as exc:  # the gate's own raise is what is under test
        if _VLM_REJECTION in str(exc):
            return exc
    return None


def test_unregistered_vlm_name_is_rejected():
    """A custom/remote-code VLM: no ITT registration, no vision_config — only the name says VLM."""
    assert _drive("acme_mystery_vlm", "acme/mystery-vl-8b") is not None, (
        "an unregistered VLM fed images reached the pipeline split; its vision tower would be dropped silently"
    )


def test_unregistered_text_model_is_not_rejected():
    """Anti-vacuity: the same unregistered architecture under a text-only name must run."""
    assert _drive("acme_mystery_lm", "acme/mystery-8b") is None


def test_registered_text_config_vetoes_a_matching_name():
    """The hints match mid-word ('re**vision**-8472618'), so a config transformers knows must win —
    otherwise a checkpoint path alone would make every text run un-pipelineable."""
    assert _drive("qwen3", "/ckpt/revision-8472618/qwen3-8b") is None


def test_registered_multimodal_config_is_still_rejected():
    """The config route must keep deciding on its own: a registered VLM under a name carrying no
    hint at all (an output_dir, a local path) is still a VLM."""
    assert _drive("qwen3", "/runs/stage1", vision_config={"hidden_size": 8}) is not None


def test_a_missing_name_field_is_not_an_error():
    """A config built in-process carries no ``_name_or_path``; the gate must degrade to the config
    verdict, not raise an AttributeError before it."""
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3"))
    try:
        PipelineTrainerMixin._maybe_prepare_pipeline_model(_mixin(), {"model": model}, _args())
    except Exception as exc:  # later failures are fine, faulting on the name is not
        assert "_name_or_path" not in str(exc), f"the name read must tolerate an absent field: {exc}"
        assert _VLM_REJECTION not in str(exc)


def test_the_probe_never_reaches_the_hub():
    """The live ``model.config`` is passed, so the modality probe must not fetch a config: a
    per-rank hub call inside a trainer ctor is a rank-divergence and an offline-run failure."""

    def _forbidden(*args, **kwargs):
        raise AssertionError("is_vlm_model fetched a config from the hub despite being passed one")

    with mock.patch.object(modality.AutoConfig, "from_pretrained", _forbidden):
        assert _drive("acme_mystery_vlm", "acme/mystery-vl-8b") is not None
        assert _drive("qwen3", "/ckpt/revision-8472618/qwen3-8b") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
