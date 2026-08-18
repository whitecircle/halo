#!/usr/bin/env python
"""Test that trainer-side ``dataset.map`` passes honour ``dataset_num_proc`` the toolkit way.

An UNSET ``dataset_num_proc`` means "toolkit default" (``DATASET_NUM_PROC``), not "one
worker": handing the raw ``None`` to ``coordinated_map`` overrides that default and runs
the heaviest tokenize single-process. EP gets the same count as every other mode — which only holds
because the mapped callables are module-level functions that take their state through ``fn_kwargs``.
A bound method would ship its ``self`` (the model, and under EP the DeepEP/NCCL process groups) to
every worker; ``reject_self_capturing_fn`` rejects that at the map seam, and the tokenize functions
are exercised here through a real multi-worker ``Dataset.map``.

Run: python tests/cpu/trainers/test_dataset_num_proc_resolution.py
"""

import inspect
import types
from functools import partial

import pytest
from datasets import Dataset

import src.trainers.preference.smpo as smpo_module
from src.data.pipeline.processing import (
    DATASET_NUM_PROC,
    coordinated_filter,
    coordinated_map,
    reject_self_capturing_fn,
)
from src.trainers.grpo.offline import tokenize_offline_grpo_rows
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.preference.smpo import SmoothMarginPOTrainer, tokenize_preference_row

_resolve = DistributedTrainerMixin._dataset_map_num_proc

# >1 is what makes datasets pickle the mapped callable.
_WORKERS = 2


def _fake_self(*, is_ep_mode=False):
    return types.SimpleNamespace(parallelism_config=types.SimpleNamespace(is_ep_mode=is_ep_mode))


def test_unset_resolves_to_toolkit_default():
    assert _resolve(_fake_self(), None) == DATASET_NUM_PROC
    assert DATASET_NUM_PROC > 1  # else this test proves nothing about the None bug


def test_explicit_value_wins():
    assert _resolve(_fake_self(), 3) == 3


def test_ep_is_not_pinned_to_one_worker():
    assert _resolve(_fake_self(is_ep_mode=True), None) == DATASET_NUM_PROC
    assert _resolve(_fake_self(is_ep_mode=True), 8) == 8


class _Unpicklable:
    """Stands in for a trainer holding a ProcessGroup: dill cannot send it to a worker."""

    def __reduce__(self):
        raise TypeError("cannot pickle 'ProcessGroup' object")

    def row_fn(self, row):
        return row


def test_bound_method_rejected_for_multi_worker_map():
    with pytest.raises(TypeError, match="bound method _Unpicklable.row_fn"):
        coordinated_map(Dataset.from_dict({"a": [1]}), _Unpicklable().row_fn, num_proc=_WORKERS)


def test_bound_method_rejected_for_multi_worker_filter():
    with pytest.raises(TypeError, match="bound method _Unpicklable.row_fn"):
        coordinated_filter(Dataset.from_dict({"a": [1]}), _Unpicklable().row_fn, num_proc=_WORKERS)


def test_single_worker_map_leaves_bound_methods_alone():
    # num_proc<=1 pickles nothing, so firing the guard would break every serial path.
    for num_proc in (None, 0, 1):
        reject_self_capturing_fn(_Unpicklable().row_fn, num_proc, "map operation")


def test_guard_sees_through_partial():
    with pytest.raises(TypeError, match="bound method"):
        reject_self_capturing_fn(partial(_Unpicklable().row_fn, row=None), _WORKERS, "map operation")


class _FakeDataset:
    column_names = ["prompt", "chosen", "rejected"]

    def cast_column(self, *_args, **_kwargs):  # the VLM branch recasts the images feature
        return self


def _smpo_self(*, dataset_num_proc, is_ep_mode=False, is_vlm=False):
    me = types.SimpleNamespace(
        is_vlm=is_vlm,
        dataset_num_proc=dataset_num_proc,
        max_prompt_length=17,
        max_completion_length=23,
        truncation_mode="keep_end",
        _eos_token_ids=frozenset({2}),
        parallelism_config=types.SimpleNamespace(is_ep_mode=is_ep_mode),
    )
    me._dataset_map_num_proc = types.MethodType(_resolve, me)
    return me


def _run_smpo_prepare(me):
    captured = {}

    def _fake_map(dataset, fn, **kwargs):
        captured["fn"] = fn
        captured["num_proc"] = kwargs["num_proc"]
        captured["fn_kwargs"] = kwargs["fn_kwargs"]
        return dataset

    original = smpo_module.coordinated_map
    smpo_module.coordinated_map = _fake_map
    try:
        SmoothMarginPOTrainer._prepare_dataset(me, _FakeDataset(), None, "train")
    finally:
        smpo_module.coordinated_map = original
    return captured


def test_smpo_tokenize_uses_toolkit_default_when_unset():
    assert _run_smpo_prepare(_smpo_self(dataset_num_proc=None))["num_proc"] == DATASET_NUM_PROC


def test_smpo_tokenize_honours_explicit_value():
    assert _run_smpo_prepare(_smpo_self(dataset_num_proc=2))["num_proc"] == 2


def test_smpo_tokenize_uses_default_under_ep():
    assert _run_smpo_prepare(_smpo_self(dataset_num_proc=8, is_ep_mode=True))["num_proc"] == 8


def test_smpo_passes_a_self_free_callable_with_state_in_fn_kwargs():
    # Tunables must ride fn_kwargs: the map cache key fingerprints those, never values off self.
    for is_vlm in (False, True):
        captured = _run_smpo_prepare(_smpo_self(dataset_num_proc=4, is_vlm=is_vlm))
        assert not inspect.ismethod(captured["fn"]), f"is_vlm={is_vlm}: bound method reached the map"
        assert captured["fn_kwargs"]["max_prompt_length"] == 17
        assert captured["fn_kwargs"]["max_completion_length"] == 23
        assert captured["fn_kwargs"]["truncation_mode"] == "keep_end"
        # Terminators too, else a model swap silently reuses a stale cached map.
        assert captured["fn_kwargs"]["eos_token_ids"] == frozenset({2})


class PicklableTokenizer:
    """Whitespace tokenizer with a BOS-emitting post-processor (module-level so workers can load it)."""

    bos_token_id = 1
    eos_token_id = 2
    bos_token = "<bos>"

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None, **kwargs):
        ids = [10 + (len(word) % 50) for word in text.split()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids)


def test_smpo_tokenize_row_runs_in_worker_processes():
    ds = Dataset.from_dict(
        {
            "prompt": [f"prompt number {i} here" for i in range(16)],
            "chosen": [f" good answer {i}" for i in range(16)],
            "rejected": [f" bad {i}" for i in range(16)],
        }
    )
    kwargs = {
        "processing_class": PicklableTokenizer(),
        "max_prompt_length": None,
        "max_completion_length": None,
        "truncation_mode": "keep_end",
    }
    parallel = ds.map(
        tokenize_preference_row,
        fn_kwargs=kwargs,
        remove_columns=ds.column_names,
        num_proc=_WORKERS,
        load_from_cache_file=False,
    )
    serial = ds.map(
        tokenize_preference_row,
        fn_kwargs=kwargs,
        remove_columns=ds.column_names,
        load_from_cache_file=False,
    )
    assert parallel.to_dict() == serial.to_dict(), "multi-worker tokenize diverged from serial"
    assert parallel[0]["chosen_input_ids"][-1] == PicklableTokenizer.eos_token_id


def test_offline_grpo_tokenize_rows_runs_in_worker_processes():
    ds = Dataset.from_dict(
        {
            "prompt": [f"question {i} text" for i in range(16)],
            "completions": [[f"answer {i} a", f"answer {i} b"] for i in range(16)],
            "rewards": [[1.0, 0.0] for _ in range(16)],
        }
    )
    kwargs = {
        "processing_class": PicklableTokenizer(),
        "max_prompt_length": 8,
        "max_completion_length": 8,
        "advantage_method": "z_norm",
        "best_completion_emphasis": 0.0,
        "is_encoder_decoder": False,
    }
    map_kwargs = {
        "batched": True,
        "batch_size": 4,
        "with_indices": True,
        "fn_kwargs": kwargs,
        "remove_columns": ds.column_names,
        "load_from_cache_file": False,
    }
    parallel = ds.map(tokenize_offline_grpo_rows, num_proc=_WORKERS, **map_kwargs)
    serial = ds.map(tokenize_offline_grpo_rows, **map_kwargs)
    assert parallel.to_dict() == serial.to_dict(), "multi-worker tokenize diverged from serial"
    assert len(parallel) == 2 * len(ds)  # one row per completion
    assert parallel[0]["advantage"] > 0 > parallel[1]["advantage"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
