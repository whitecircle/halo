#!/usr/bin/env python
"""The PP multimodal gate must get the model's name: for an unregistered ``model_type`` only the name says VLM.
Such a VLM fed images would train with its vision tower dropped from every stage while the saved config declared it.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
from datasets import Dataset

import halo.model_loading.parallel.model_source as model_source
from halo.trainers.parallelism.pipeline_trainer import PipelineTrainerMixin
from tests.common.tiny_models import config_only_module

_VLM_REJECTION = "Vision-language training is not supported under pipeline parallelism"


def _mixin():
    """The split dispatches the ``_reject_unsupported_pp_options`` hook through ``self``."""
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
        eval_on_start=False,
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        torch_compile=False,
        activation_offloading=False,
    )


def _image_dataset() -> Dataset:
    return Dataset.from_dict({"prompt": ["describe"], "images": [[]]})


def _drive(model_type, name, *, vision_config=None):
    """Returns the gate's exception, or None past it; a later raise counts as past, since the split needs a real model.
    Drives the split itself: asserting on ``is_vlm_model`` would pass with the whole PP call site deleted."""
    config = SimpleNamespace(model_type=model_type, _name_or_path=name)
    if vision_config is not None:
        config.vision_config = vision_config
    kwargs = {"model": config_only_module(config), "train_dataset": _image_dataset()}
    try:
        PipelineTrainerMixin._split_pipeline_stage(_mixin(), kwargs, _args())
    except Exception as exc:
        if _VLM_REJECTION in str(exc):
            return exc
    return None


def test_unregistered_vlm_name_is_rejected():
    assert _drive("acme_mystery_vlm", "acme/mystery-vl-8b") is not None, (
        "an unregistered VLM fed images reached the pipeline split; its vision tower would be dropped silently"
    )


def test_unregistered_text_model_is_not_rejected():
    assert _drive("acme_mystery_lm", "acme/mystery-8b") is None


def test_registered_text_config_vetoes_a_matching_name():
    """The hints match mid-word ('re**vision**-8472618'), so a config transformers knows must win over the path."""
    assert _drive("qwen3", "/ckpt/revision-8472618/qwen3-8b") is None


def test_registered_multimodal_config_is_still_rejected():
    """A registered VLM under a name carrying no hint (an output_dir, a local path) is still a VLM."""
    assert _drive("qwen3", "/runs/stage1", vision_config={"hidden_size": 8}) is not None


def test_a_missing_name_field_is_not_an_error():
    """A config built in-process carries no ``_name_or_path``; the gate falls back to the config verdict."""
    model = config_only_module(SimpleNamespace(model_type="qwen3"))
    try:
        PipelineTrainerMixin._split_pipeline_stage(_mixin(), {"model": model}, _args())
    except Exception as exc:
        assert "_name_or_path" not in str(exc), f"the name read must tolerate an absent field: {exc}"
        assert _VLM_REJECTION not in str(exc)


def test_the_probe_never_reaches_the_hub():
    """A per-rank hub call inside a trainer ctor is a rank divergence and an offline-run failure."""

    def _forbidden(*args, **kwargs):
        raise AssertionError("is_vlm_model fetched a config from the hub despite being passed one")

    with mock.patch.object(model_source.AutoConfig, "from_pretrained", _forbidden):
        assert _drive("acme_mystery_vlm", "acme/mystery-vl-8b") is not None
        assert _drive("qwen3", "/ckpt/revision-8472618/qwen3-8b") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
