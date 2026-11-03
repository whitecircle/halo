#!/usr/bin/env python
"""The reward-model inference scripts key their output file on the generating model: both resume from the ids in it,
so a name built from the prompt source alone makes a second model's run read the first's rows as its own. An
over-long ``--model`` must not raise ``OSError(ENAMETOOLONG)`` once the reward model is on the GPU."""

from __future__ import annotations

import pytest

from halo.entrypoints.reward_scoring import _MAX_OUTPUT_BASENAME, build_output_path


def _path(model_name, suffix="rs", dataset="/data/prompts.jsonl"):
    return build_output_path("/out", dataset, model_name, suffix)


def test_two_models_over_the_same_prompts_get_separate_files():
    assert _path("model-a") != _path("model-b")


def test_output_name_carries_the_model_and_the_prompt_source():
    path = _path("model-a")
    assert path.name == "prompts_model-a_rs.jsonl"
    assert path.parent.name == "out"


def test_each_suffix_keeps_its_own_file():
    """The record shapes are not interchangeable, so the formats must not share a resume file."""
    assert _path("model-a", suffix="offline_grpo").name == "prompts_model-a_offline_grpo.jsonl"
    assert _path("model-a", suffix="reward_scoring").name == "prompts_model-a_reward_scoring.jsonl"
    assert _path("model-a", suffix="offline_grpo") != _path("model-a")


def test_slashed_model_name_stays_one_path_component():
    """A hub id would otherwise open a subdirectory that does not exist."""
    path = _path("org/model-a")
    assert path.name == "prompts_org__model-a_rs.jsonl"
    assert path.parent.name == "out"


@pytest.mark.parametrize(
    ("model_name", "dataset", "suffix"),
    [
        ("/models/" + "a" * 400, "/data/prompts.jsonl", "rs"),  # vLLM's --served-model-name default
        ("m", "/data/" + "p" * 400 + ".jsonl", "reward_scoring"),  # a long prompt path on its own
        ("/models/" + "a" * 200, "/data/" + "p" * 200 + ".jsonl", "offline_grpo"),  # both, each under the cap
        ("m", "/data/" + "я" * 200 + ".jsonl", "rs"),  # multi-byte: 200 chars is 400 bytes
    ],
)
def test_every_over_long_name_fits_a_filesystem_name(model_name, dataset, suffix):
    """The cap is in bytes on the basename, and either half can overrun it alone."""
    path = build_output_path("/out", dataset, model_name, suffix)
    assert len(path.name.encode()) <= _MAX_OUTPUT_BASENAME, path.name


def test_two_long_model_names_still_get_separate_files():
    """Truncation alone would collide two long paths sharing a prefix."""
    prefix = "/models/checkpoints/" + "a" * 400
    assert _path(prefix + "/run-one") != _path(prefix + "/run-two")


def test_two_long_prompt_sources_still_get_separate_files():
    prefix = "/data/" + "p" * 400
    assert _path("m", dataset=prefix + "-one.jsonl") != _path("m", dataset=prefix + "-two.jsonl")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
