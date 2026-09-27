"""A code-contest eval reads, and records, the split its dataset actually ships.

LiveCodeBench ships only ``test`` and HLCE only ``train``; their loaders read that split whatever
they are passed. A run naming another split would record it in the trajectory meta while scoring
the source's own, so the adapter resolves the split and refuses a mismatch before anything loads.
"""

import sys

import pytest

from scripts.environments._common import DEFAULT_SPLIT
from scripts.environments.inference import run_code_contests
from src.environments.envs.tasks.coding.datasets import CODE_DATASET_ADAPTERS


@pytest.mark.parametrize(("adapter", "shipped"), [("livecodebench", "test"), ("hlce", "train")])
def test_a_single_split_benchmark_resolves_to_its_own_split(adapter, shipped):
    resolve = CODE_DATASET_ADAPTERS[adapter].resolve_split
    assert resolve(None, DEFAULT_SPLIT) == shipped
    assert resolve(shipped, DEFAULT_SPLIT) == shipped


@pytest.mark.parametrize(("adapter", "other"), [("livecodebench", "train"), ("hlce", "test")])
def test_a_single_split_benchmark_refuses_another_split(adapter, other):
    with pytest.raises(ValueError, match=f"ships only the .* split, not '{other}'"):
        CODE_DATASET_ADAPTERS[adapter].resolve_split(other, DEFAULT_SPLIT)


def test_a_multi_split_source_takes_the_requested_split_else_the_default():
    resolve = CODE_DATASET_ADAPTERS["codeforces"].resolve_split
    assert resolve(None, DEFAULT_SPLIT) == DEFAULT_SPLIT
    assert resolve("train", DEFAULT_SPLIT) == "train"


def test_the_eval_script_exits_naming_the_adapter_before_loading(monkeypatch):
    argv = ["run_code_contests.py", "--dataset", "org/hlce", "--model", "m", "--adapter", "hlce", "--split", "test"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exited:
        run_code_contests.main()
    assert str(exited.value.code) == "--adapter hlce: this dataset ships only the 'train' split, not 'test'"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
