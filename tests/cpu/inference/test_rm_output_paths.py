#!/usr/bin/env python
"""CPU tests: the reward-model inference scripts must key their output file on the GENERATING model.

Both scripts resume from the ids already present in their output file. A name built only from the
prompt source therefore makes a second model's run read the first model's rows as its own and
generate nothing — while appending its results to a file that claims a different ``gen_model``.

The name is also the first thing touched after the reward model is resident on the GPU
(``output_path.exists()``), so an over-long ``--model_name`` — vLLM's ``--served-model-name`` defaults
to the served model's PATH — must not surface as an uncaught ``OSError(ENAMETOOLONG)`` there.

Run: ``python tests/cpu/inference/test_rm_output_paths.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest

from scripts.inference.reward_model._common import _MAX_OUTPUT_BASENAME, build_output_path


def _path(model_name, suffix="rs", prompts_source="/data/prompts.jsonl"):
    return build_output_path("/out", prompts_source, model_name, suffix)


def test_two_models_over_the_same_prompts_get_separate_files():
    assert _path("model-a") != _path("model-b")


def test_output_name_carries_the_model_and_the_prompt_source():
    path = _path("model-a")
    assert path.name == "prompts_model-a_rs.jsonl"
    assert path.parent.name == "out"


def test_each_suffix_keeps_its_own_file():
    """The record shapes are not interchangeable, so the formats must not share a resume file; the
    shared helper spells each script's own suffix rather than one name for all of them."""
    assert _path("model-a", suffix="offline_grpo").name == "prompts_model-a_offline_grpo.jsonl"
    assert _path("model-a", suffix="rm_scoring").name == "prompts_model-a_rm_scoring.jsonl"
    assert _path("model-a", suffix="offline_grpo") != _path("model-a")


def test_slashed_model_name_stays_one_path_component():
    """A hub id would otherwise open a subdirectory that does not exist."""
    path = _path("org/model-a")
    assert path.name == "prompts_org__model-a_rs.jsonl"
    assert path.parent.name == "out"


@pytest.mark.parametrize(
    ("model_name", "prompts_source", "suffix"),
    [
        ("/models/" + "a" * 400, "/data/prompts.jsonl", "rs"),  # vLLM's --served-model-name default
        ("m", "/data/" + "p" * 400 + ".jsonl", "rm_scoring"),  # a long prompt path on its own
        ("/models/" + "a" * 200, "/data/" + "p" * 200 + ".jsonl", "offline_grpo"),  # both, each under the cap
        ("m", "/data/" + "я" * 200 + ".jsonl", "rs"),  # multi-byte: 200 chars is 400 bytes
    ],
)
def test_every_over_long_name_fits_a_filesystem_name(model_name, prompts_source, suffix):
    """The cap is a BYTE cap on the basename, and either half can overrun it alone. The resulting
    OSError(ENAMETOOLONG) lands only after the reward model is already on the GPU."""
    path = build_output_path("/out", prompts_source, model_name, suffix)
    assert len(path.name.encode()) <= _MAX_OUTPUT_BASENAME, path.name


def test_two_long_model_names_still_get_separate_files():
    """Truncation alone would collide two long paths sharing a prefix, re-creating the very bug the
    model name is in the filename to prevent."""
    prefix = "/models/checkpoints/" + "a" * 400
    assert _path(prefix + "/run-one") != _path(prefix + "/run-two")


def test_two_long_prompt_sources_still_get_separate_files():
    """Same collision, from the other half of the name."""
    prefix = "/data/" + "p" * 400
    assert _path("m", prompts_source=prefix + "-one.jsonl") != _path("m", prompts_source=prefix + "-two.jsonl")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
