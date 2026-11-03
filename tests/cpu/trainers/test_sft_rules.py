#!/usr/bin/env python
"""The SFT script's data-option refusals: each fires on its own condition, and the default run passes them all."""

import re
from types import SimpleNamespace

import pytest

from halo.configs.parallelism_arguments import ParallelismArguments
from halo.configs.sft import SFTScriptArguments
from halo.trainers.sft_rules import (
    reject_unsupported_sft_data_options,
    reject_unsupported_tree_packing,
    reject_unsupported_vlm_sft_options,
)


def _data_options(**overrides) -> SimpleNamespace:
    return SimpleNamespace(
        **{"packing": False, "padding_free": False, "max_length": 1024, "eval_packing": None, **overrides}
    )


def _parallelism(**overrides) -> SimpleNamespace:
    return SimpleNamespace(**{"context_parallel_size": 1, "pipeline_parallel_size": 1, **overrides})


def test_the_default_run_passes_every_data_option_refusal():
    reject_unsupported_sft_data_options(_data_options(), ParallelismArguments())
    reject_unsupported_sft_data_options(_data_options(packing=True, eval_packing=False), _parallelism())
    reject_unsupported_vlm_sft_options(SFTScriptArguments(), _data_options(), preprocessed=False)


@pytest.mark.parametrize(
    ("options", "parallelism", "opening"),
    [
        ({"padding_free": True}, {"context_parallel_size": 2}, "Context Parallelism (CP) is NOT compatible with"),
        ({"padding_free": True}, {"pipeline_parallel_size": 2}, "Pipeline Parallelism (PP) is NOT compatible with"),
        ({"packing": True, "padding_free": True}, {}, "Cannot use both 'packing' and 'padding_free' simultaneously."),
        ({"packing": True, "max_length": None}, {}, "packing=True requires an explicit max_length"),
        ({"packing": True, "max_length": 0}, {}, "packing=True requires an explicit max_length"),
        ({"eval_packing": True}, {}, "eval_packing=True packs nothing without packing=True"),
    ],
    ids=["cp", "pp", "packing", "null-length", "zero-length", "eval-packing"],
)
def test_each_data_option_conflict_raises_its_message(options, parallelism, opening):
    with pytest.raises(ValueError, match="^" + re.escape(opening)):
        reject_unsupported_sft_data_options(_data_options(**options), _parallelism(**parallelism))


def _tree_run() -> dict[str, SimpleNamespace]:
    """The arguments of a run tree packing supports, by parameter of the refusal."""
    return {
        "args": SimpleNamespace(
            tree_share_markers=["\nQuestion:\n"], train_on_completions_only=True, train_sinks=False
        ),
        "sft_config": SimpleNamespace(packing=True, packing_strategy="bfd", eval_packing=None, torch_compile=False),
        "dist_args": SimpleNamespace(context_parallel_size=1, pipeline_parallel_size=1, tensor_parallel_size=1),
    }


@pytest.mark.parametrize(
    ("holder", "field", "value", "match"),
    [
        ("args", "tree_share_markers", [], "tree_share_markers"),
        ("sft_config", "packing", False, "packing: true"),
        ("sft_config", "packing_strategy", "wrapped", "packing_strategy: bfd"),
        ("sft_config", "eval_packing", False, "eval_packing left on"),
        ("args", "train_on_completions_only", False, "train_on_completions_only: true"),
        ("dist_args", "context_parallel_size", 2, "no context or pipeline parallelism"),
        ("dist_args", "pipeline_parallel_size", 2, "no context or pipeline parallelism"),
        ("dist_args", "tensor_parallel_size", 2, "no tensor parallelism"),
        ("sft_config", "torch_compile", True, "torch_compile off"),
        ("args", "train_sinks", True, "train_sinks off"),
    ],
)
def test_tree_packing_refuses_every_option_that_would_reshape_a_tree_row(holder, field, value, match):
    run = _tree_run()
    reject_unsupported_tree_packing(**run)
    setattr(run[holder], field, value)
    with pytest.raises(ValueError, match=f"^tree_packing needs: {match}") as refusal:
        reject_unsupported_tree_packing(**run)
    assert str(refusal.value).count(";") == 0, "only the option that was changed is named"


@pytest.mark.parametrize(
    ("options", "args", "opening"),
    [
        ({"packing": True}, {}, "packing / padding_free are not supported for VLM training"),
        ({"padding_free": True}, {}, "packing / padding_free are not supported for VLM training"),
        (
            {},
            {"train_on_last_assistant_only": True},
            "train_on_last_assistant_only is not implemented on the VLM path",
        ),
    ],
    ids=["packing", "padding-free", "last-assistant-only"],
)
def test_the_raw_vlm_path_refuses_what_it_does_not_implement(options, args, opening):
    with pytest.raises(ValueError, match="^" + re.escape(opening)):
        reject_unsupported_vlm_sft_options(SFTScriptArguments(**args), _data_options(**options), preprocessed=False)


def test_a_preprocessed_vlm_dataset_trains_as_baked():
    """Its packing was decided by the bake, so only tree packing, which no VLM row can take, is refused."""
    baked = SFTScriptArguments(train_on_last_assistant_only=True)
    reject_unsupported_vlm_sft_options(baked, _data_options(packing=True), preprocessed=True)
    with pytest.raises(ValueError, match="Halo SFT on the VLM path .*tree_packing"):
        reject_unsupported_vlm_sft_options(SFTScriptArguments(tree_packing=True), _data_options(), preprocessed=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
