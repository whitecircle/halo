import os
import sys
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from src.cli import app, command, resolve_config, tool_commands, training_methods

runner = CliRunner()


def stub_exec(monkeypatch) -> tuple[list[Path], list[list[str]]]:
    """Replace the process handover so a non-dry-run invocation is observable.

    Returns ``(chdirs, argvs)`` — what the CLI would have chdir'd to, and the argv it would exec.
    """
    chdirs: list[Path] = []
    argvs: list[list[str]] = []
    monkeypatch.setattr(os, "chdir", chdirs.append)
    monkeypatch.setattr(os, "execvp", lambda file, argv: argvs.append(argv))
    return chdirs, argvs


def write_script(path: Path, *, runnable: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = 'def main():\n    pass\n\n\nif __name__ == "__main__":\n    main()\n' if runnable else "VALUE = 1\n"
    path.write_text(body, encoding="utf-8")


def write_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("output_dir: /tmp/out\n", encoding="utf-8")


def test_methods_are_indexed_from_training_file_paths(tmp_path):
    write_script(tmp_path / "scripts/training/sft.py")
    write_script(tmp_path / "scripts/training/preference/smpo.py")
    write_script(tmp_path / "scripts/training/online_grpo/rlvr.py")

    index = training_methods(tmp_path)

    assert index["sft"] == tmp_path / "scripts/training/sft.py"
    assert index["preference/smpo"] == tmp_path / "scripts/training/preference/smpo.py"
    assert index["smpo"] == tmp_path / "scripts/training/preference/smpo.py"
    assert index["online-grpo/rlvr"] == tmp_path / "scripts/training/online_grpo/rlvr.py"
    assert index["rlvr"] == tmp_path / "scripts/training/online_grpo/rlvr.py"


def test_ambiguous_short_names_are_not_indexed(tmp_path):
    write_script(tmp_path / "scripts/training/a/train.py")
    write_script(tmp_path / "scripts/training/b/train.py")

    index = training_methods(tmp_path)

    assert "train" not in index
    assert "a/train" in index
    assert "b/train" in index


def test_nested_methods_index_without_shadowing_flat_methods(tmp_path):
    write_script(tmp_path / "scripts/training/sft.py")
    write_script(tmp_path / "scripts/training/nested/sft.py")
    write_script(tmp_path / "scripts/training/nested/reward.py")

    index = training_methods(tmp_path)

    # the flat method keeps its short name even though nested/sft.py shares the stem
    assert index["sft"] == tmp_path / "scripts/training/sft.py"
    assert index["nested/sft"] == tmp_path / "scripts/training/nested/sft.py"
    assert index["nested/reward"] == tmp_path / "scripts/training/nested/reward.py"
    # ambiguous stem gets no alias; the unambiguous one does
    assert index["reward"] == tmp_path / "scripts/training/nested/reward.py"


def test_singular_plural_stem_variants_get_no_alias(tmp_path):
    """`reward` (nested) vs `rewards` (preference) differ only by pluralization; a bare-stem
    alias would silently launch the wrong trainer, so neither stem is aliased."""
    write_script(tmp_path / "scripts/training/preference/rewards.py")
    write_script(tmp_path / "scripts/training/nested/reward.py")

    index = training_methods(tmp_path)

    assert "reward" not in index, "'reward' would silently resolve to the nested trainer"
    assert "rewards" not in index
    assert index["preference/rewards"] == tmp_path / "scripts/training/preference/rewards.py"
    assert index["nested/reward"] == tmp_path / "scripts/training/nested/reward.py"


def test_launch_ambiguous_stem_reports_qualified_names(tmp_path):
    write_script(tmp_path / "scripts/training/preference/rewards.py")
    write_script(tmp_path / "scripts/training/nested/reward.py")
    write_config(tmp_path / "config.yaml")

    result = runner.invoke(app, ["launch", "reward", "config.yaml", "--dry-run", "--root", str(tmp_path)])

    assert result.exit_code != 0
    assert "ambiguous" in result.output
    assert "preference/rewards" in result.output
    assert "nested/reward" in result.output


def test_underscored_method_name_resolves_to_the_hyphenated_entry(tmp_path):
    """The index is hyphenated, but `scripts/training/offline_grpo.py`, its examples directory and
    half the docs spell the method `offline_grpo` — that spelling must launch it, not be rejected as
    unknown. The listing stays hyphenated: one canonical entry, no alias."""
    write_script(tmp_path / "scripts/training/offline_grpo.py")
    write_config(tmp_path / "config.yaml")

    commands = set()
    for spelling in ("offline-grpo", "offline_grpo", "Offline_GRPO"):
        result = runner.invoke(app, ["launch", spelling, "config.yaml", "--dry-run", "--root", str(tmp_path)])
        assert result.exit_code == 0, f"{spelling}: {result.output}"
        commands.add(result.stdout.strip())

    assert len(commands) == 1, f"spellings must resolve to one command, got {commands}"
    assert str(tmp_path / "scripts/training/offline_grpo.py") in commands.pop()

    listed = runner.invoke(app, ["launch", "--list", "--root", str(tmp_path)])
    assert [line.split()[0] for line in listed.stdout.splitlines() if line.strip()] == ["offline-grpo"]


def test_unknown_method_still_fails_loud(tmp_path):
    """Normalization must not swallow a typo into a near neighbour: an unindexed name raises, naming
    the known methods and the flag that lists the rest."""
    write_script(tmp_path / "scripts/training/offline_grpo.py")
    write_config(tmp_path / "config.yaml")

    result = runner.invoke(app, ["launch", "offline_grpoo", "config.yaml", "--dry-run", "--root", str(tmp_path)])

    assert result.exit_code != 0
    # Substrings, not the whole sentence: the error renders through a width-wrapped rich panel.
    assert "unknown method" in result.output
    assert "offline_grpoo" in result.output
    assert "offline-grpo" in result.output and "--list" in result.output


def test_a_root_with_no_scripts_tree_names_the_root(tmp_path):
    """An index that came back empty is a wrong ``--root`` (or an install without the tree). Calling
    the typed name unknown would name the wrong problem and send the user hunting for a typo."""
    write_config(tmp_path / "config.yaml")

    results = {
        "method": runner.invoke(app, ["launch", "sft", "config.yaml", "--dry-run", "--root", str(tmp_path)]),
        "tool": runner.invoke(app, ["run", "merge-ep-shards", "--dry-run", "--root", str(tmp_path)]),
    }

    for kind, result in results.items():
        # The error renders through a width-wrapped rich panel, so collapse it before matching.
        output = " ".join(result.output.split())
        assert result.exit_code != 0, output
        assert f"no {kind}s found under" in output, output
        assert "unknown" not in output, output


def test_helpers_without_main_guard_are_not_indexed(tmp_path):
    write_script(tmp_path / "scripts/training/preference/constants.py", runnable=False)

    assert training_methods(tmp_path) == {}


def test_tool_commands_cover_after_training_inference_and_preparation(tmp_path):
    write_script(tmp_path / "scripts/after_training/merge_ep_shards.py")
    write_script(tmp_path / "scripts/inference/reward_model/rm_scoring.py")
    write_script(tmp_path / "scripts/before_training/prepare_dataset.py")
    write_script(tmp_path / "scripts/environments/inference/run_env.py")
    # training scripts must NOT leak into the tool index
    write_script(tmp_path / "scripts/training/sft.py")

    index = tool_commands(tmp_path)

    assert index["after-training/merge-ep-shards"] == tmp_path / "scripts/after_training/merge_ep_shards.py"
    assert index["merge-ep-shards"] == tmp_path / "scripts/after_training/merge_ep_shards.py"
    assert index["inference/reward-model/rm-scoring"] == tmp_path / "scripts/inference/reward_model/rm_scoring.py"
    assert index["before-training/prepare-dataset"] == tmp_path / "scripts/before_training/prepare_dataset.py"
    assert index["environments/inference/run-env"] == tmp_path / "scripts/environments/inference/run_env.py"
    assert all(not name.startswith("training/") for name in index)
    assert "sft" not in index


def test_tool_subtrees_are_derived_from_the_scripts_tree(tmp_path):
    # a new scripts/ subtree (profiling, inference, ...) must index without a hand-maintained list
    write_script(tmp_path / "scripts/profiling/trace_report.py")
    write_script(tmp_path / "scripts/after_training/merge_models.py")
    write_script(tmp_path / "scripts/diagrams/gen_roofline.py")
    write_script(tmp_path / "scripts/training/sft.py")

    index = tool_commands(tmp_path)

    assert index["profiling/trace-report"] == tmp_path / "scripts/profiling/trace_report.py"
    assert index["after-training/merge-models"] == tmp_path / "scripts/after_training/merge_models.py"
    # training stays owned by `halo launch`
    assert all(not name.startswith("training/") for name in index)
    assert "sft" not in index
    # diagrams regenerates committed doc assets through generators that rely on being run as a
    # script (`from _theory_style import *`), so advertising them as tools would only mislead.
    assert all(not name.startswith("diagrams/") for name in index)


def test_real_repo_profiling_tools_are_indexed():
    # CLAUDE.md advertises the profiling CLIs; the real tree must expose them via `halo run`
    index = tool_commands()

    assert "profiling/trace-report" in index
    assert "profiling/py-spy-diag" in index


def test_tools_index_scripts_without_def_main(tmp_path):
    # quantize_to_lowp / convert_to_bf16 style scripts have a __main__ guard but no def main()
    path = tmp_path / "scripts/after_training/quantize_to_lowp.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('import sys\n\nif __name__ == "__main__":\n    sys.exit(0)\n', encoding="utf-8")

    assert tool_commands(tmp_path)["quantize-to-lowp"] == path


def test_command_uses_python_for_single_process():
    run = command(Path("scripts/training/sft.py"), Path("config.yaml"), nproc=1, args=["--lr=1e-5"])

    assert run == [sys.executable, "scripts/training/sft.py", "config.yaml", "--lr=1e-5"]


def test_command_uses_torchrun_when_nproc_is_greater_than_one():
    run = command(Path("scripts/training/sft.py"), Path("config.yaml"), nproc=8, args=[])

    assert run == ["torchrun", "--nproc_per_node=8", "scripts/training/sft.py", "config.yaml"]


def test_command_omits_config_for_tools():
    run = command(Path("scripts/after_training/merge_ep_shards.py"), None, nproc=1, args=["--input_dir=x"])

    assert run == [sys.executable, "scripts/after_training/merge_ep_shards.py", "--input_dir=x"]


def test_command_uses_accelerate_launch_with_config():
    run = command(
        Path("scripts/training/sft.py"),
        Path("config.yaml"),
        nproc=1,
        args=["--lr=1e-5"],
        accelerate_config=Path("launcher-configs/accelerate/fsdp2_gradop_config.yaml"),
    )

    assert run == [
        "accelerate",
        "launch",
        "--config_file",
        "launcher-configs/accelerate/fsdp2_gradop_config.yaml",
        "scripts/training/sft.py",
        "config.yaml",
        "--lr=1e-5",
    ]


def test_command_accelerate_passes_num_processes_when_nproc_set():
    run = command(
        Path("scripts/training/sft.py"),
        Path("config.yaml"),
        nproc=4,
        args=[],
        accelerate_config=Path("launcher-configs/accelerate/fsdp2_gradop_config.yaml"),
    )

    assert run[:6] == [
        "accelerate",
        "launch",
        "--config_file",
        "launcher-configs/accelerate/fsdp2_gradop_config.yaml",
        "--num_processes",
        "4",
    ]
    # accelerate takes precedence over torchrun even when nproc > 1
    assert "torchrun" not in run


def test_repo_relative_config_resolves_from_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "repo"
    config = Path("examples/sft/example.yaml")
    write_config(root / config)

    assert resolve_config(config, root) == root / config


def test_resolve_config_fails_loud_on_missing_file(tmp_path):
    with pytest.raises(typer.BadParameter, match="config not found"):
        resolve_config(Path("examples/sft/missing.yaml"), tmp_path)


def test_launch_command_keeps_launch_as_subcommand(tmp_path):
    write_script(tmp_path / "scripts/training/sft.py")
    write_config(tmp_path / "config.yaml")

    result = runner.invoke(
        app,
        ["launch", "sft", "config.yaml", "--nproc", "8", "--dry-run", "--root", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "torchrun --nproc_per_node=8" in result.stdout


def test_launch_rejects_missing_config(tmp_path):
    write_script(tmp_path / "scripts/training/sft.py")

    result = runner.invoke(app, ["launch", "sft", "missing.yaml", "--dry-run", "--root", str(tmp_path)])

    assert result.exit_code != 0
    assert "config not found" in result.output


def test_launch_accelerate_option_resolves_config_from_root(tmp_path):
    write_script(tmp_path / "scripts/training/sft.py")
    write_config(tmp_path / "config.yaml")
    write_config(tmp_path / "accelerate/_cli_test_only.yaml")

    # use a path that does not exist in the cwd so resolve_config falls back to root deterministically
    result = runner.invoke(
        app,
        [
            "launch",
            "sft",
            "config.yaml",
            "--accelerate",
            "accelerate/_cli_test_only.yaml",
            "--dry-run",
            "--root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "accelerate launch --config_file" in result.stdout
    assert str(tmp_path / "accelerate/_cli_test_only.yaml") in result.stdout


def test_run_command_passes_flags_through(tmp_path):
    write_script(tmp_path / "scripts/after_training/merge_ep_shards.py")

    result = runner.invoke(
        app,
        ["run", "merge-ep-shards", "--dry-run", "--root", str(tmp_path), "--", "--input_dir", "ckpt"],
    )

    assert result.exit_code == 0
    assert "merge_ep_shards.py --input_dir ckpt" in result.stdout


def test_run_keeps_the_caller_cwd_so_relative_path_flags_mean_what_they_say(tmp_path, monkeypatch):
    """`halo run <tool> --output_dir ./out` must write under the CALLER's cwd.

    Chdir'ing to the repo root re-roots every relative path flag — a multi-GB checkpoint/dataset
    write silently lands on the small root FS instead of the scratch volume the user is standing in.
    """
    write_script(tmp_path / "scripts/after_training/merge_ep_shards.py")
    caller_cwd = tmp_path / "caller"
    caller_cwd.mkdir()
    monkeypatch.chdir(caller_cwd)
    chdirs, argvs = stub_exec(monkeypatch)

    result = runner.invoke(app, ["run", "merge-ep-shards", "--root", str(tmp_path), "--", "--output_dir", "./out"])

    assert result.exit_code == 0, result.output
    assert chdirs == [], "halo run must not chdir — './out' would resolve under the repo root"
    assert Path.cwd() == caller_cwd
    # The script is addressed absolutely, so nothing depended on the chdir.
    assert argvs[0][1] == str(tmp_path / "scripts/after_training/merge_ep_shards.py")
    assert argvs[0][-2:] == ["--output_dir", "./out"]


def test_launch_still_runs_the_training_script_from_the_repo_root(tmp_path, monkeypatch):
    """The launch side is unchanged: relative paths in a training config resolve at the root."""
    write_script(tmp_path / "scripts/training/sft.py")
    write_config(tmp_path / "config.yaml")
    monkeypatch.chdir(tmp_path)
    chdirs, argvs = stub_exec(monkeypatch)

    result = runner.invoke(app, ["launch", "sft", "config.yaml", "--root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert chdirs == [tmp_path.resolve()]
    assert argvs[0][1:] == [
        str(tmp_path / "scripts/training/sft.py"),
        str((tmp_path / "config.yaml").resolve()),
    ]


def test_run_rejects_unknown_tool(tmp_path):
    write_script(tmp_path / "scripts/after_training/merge_ep_shards.py")

    result = runner.invoke(app, ["run", "does-not-exist", "--root", str(tmp_path)])

    assert result.exit_code != 0


def test_command_torchrun_gets_master_port():
    run = command(Path("scripts/training/sft.py"), Path("config.yaml"), nproc=2, args=[], port=29750)

    assert run[:3] == ["torchrun", "--nproc_per_node=2", "--master_port=29750"]


def test_command_accelerate_gets_main_process_port():
    run = command(
        Path("scripts/training/sft.py"),
        Path("config.yaml"),
        nproc=2,
        args=[],
        accelerate_config=Path("launcher-configs/accelerate/fsdp2_gradop_config.yaml"),
        port=29750,
    )

    assert run[:8] == [
        "accelerate",
        "launch",
        "--config_file",
        "launcher-configs/accelerate/fsdp2_gradop_config.yaml",
        "--num_processes",
        "2",
        "--main_process_port",
        "29750",
    ]


def test_command_rejects_port_for_single_process():
    with pytest.raises(typer.BadParameter, match="--port only applies"):
        command(Path("scripts/training/sft.py"), Path("config.yaml"), nproc=1, args=[], port=29750)


@pytest.mark.parametrize("flag", ["--nnodes=2", "--node_rank", "--master-addr=10.0.0.1", "--rdzv_endpoint=h:1"])
def test_launch_refuses_torchrun_multinode_flags_by_name(tmp_path, flag):
    """``halo launch`` is single-node: a torchrun rendezvous flag among the pass-through args would
    otherwise reach the script's parser and be reported as an unknown config field."""
    write_script(tmp_path / "scripts/training/sft.py")
    write_config(tmp_path / "config.yaml")

    result = runner.invoke(
        app, ["launch", "sft", "config.yaml", "--nproc", "8", "--dry-run", "--root", str(tmp_path), flag, "0"]
    )

    assert result.exit_code != 0
    assert "torchrun" in result.output and "single-node" in result.output


def test_launch_port_option_reaches_torchrun(tmp_path):
    write_script(tmp_path / "scripts/training/sft.py")
    write_config(tmp_path / "config.yaml")

    result = runner.invoke(
        app,
        ["launch", "sft", "config.yaml", "--nproc", "2", "--port", "29751", "--dry-run", "--root", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "--master_port=29751" in result.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
