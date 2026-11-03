"""A code-contest eval reads and records the split its dataset ships: LiveCodeBench ships only ``test``, HLCE only
``train``, so the adapter resolves the split and refuses a mismatch before anything loads."""

import sys

import pytest

from halo.entrypoints.eval_contract import DEFAULT_SPLIT
from halo.environments.tasks.coding.contest_datasets import CODE_DATASET_ADAPTERS
from scripts.environments.inference import eval_code_contests, eval_env


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
    argv = ["eval_code_contests.py", "--dataset", "org/hlce", "--model", "m", "--adapter", "hlce", "--split", "test"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exited:
        eval_code_contests.main()
    assert str(exited.value.code) == "--split on --adapter hlce: this dataset ships only the 'train' split, not 'test'"


def _help(module, monkeypatch, capsys) -> str:
    monkeypatch.setattr(sys, "argv", [f"{module.__name__}.py", "--help"])
    with pytest.raises(SystemExit):
        module.parse_args()
    return " ".join(capsys.readouterr().out.split())


def test_only_the_runner_that_refuses_a_foreign_split_says_it_does(monkeypatch, capsys):
    """``eval_env.py`` reads the split it is given, whatever the source ships; its help must not
    promise the single-split refusal only the code-contest adapters make."""
    refusal = "A source that ships a single split of its own reads that split and refuses another."
    assert refusal in _help(eval_code_contests, monkeypatch, capsys)
    assert refusal not in _help(eval_env, monkeypatch, capsys)

    monkeypatch.setattr(sys, "argv", ["eval_env.py", "--dataset", "d", "--model", "m"])
    assert eval_env.parse_args().split == DEFAULT_SPLIT


@pytest.mark.parametrize("runner", [eval_code_contests, eval_env], ids=["eval_code_contests", "eval_env"])
def test_a_trajectory_path_and_a_trajectory_dir_are_refused_together(runner, monkeypatch, capsys):
    """One would silently beat the other."""
    argv = ["eval.py", "--dataset", "d", "--model", "m", "--adapter", "codeforces"]
    argv = argv if runner is eval_code_contests else argv[:-2]
    monkeypatch.setattr(sys, "argv", [*argv, "--save_trajectories", "t.jsonl", "--trajectory_dir", "runs"])
    with pytest.raises(SystemExit):
        runner.parse_args()
    assert "not allowed with argument" in capsys.readouterr().err
    for flags in (["--save_trajectories", "t.jsonl"], ["--trajectory_dir", "runs"]):
        monkeypatch.setattr(sys, "argv", [*argv, *flags])
        assert runner.parse_args()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
