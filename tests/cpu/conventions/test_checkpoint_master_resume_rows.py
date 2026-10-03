"""Requested FP32-master GPU rows use explicit layouts and strict checkpoint-resume provenance."""

import ast
import shlex
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tests.gpu.parallelism.ep.test_ep_optimizer_resume as resume_suite
from src.distributed.parallelism_config import ParallelismConfig
from tests.gpu.manifest import MANIFEST

SUITE = "parallelism/ep/test_ep_optimizer_resume.py"
MASTER_ROWS = {
    ("ep", False, False): (2, 1, True),
    ("ep_cp", False, False): (2, 2, True),
    ("ep", True, False): (2, 1, False),
    ("ep1", False, True): (1, 1, True),
    ("cp", False, False): (1, 2, True),
    ("fsdp", False, False): (1, 1, True),
}


class _LoaderReached(Exception):
    """Stop the GPU row after its public loader receives the construction options."""


def _rows():
    return [resume_suite.optimizer_resume_parser().parse_args(shlex.split(row)) for row in MANIFEST[SUITE].args_matrix]


def _configured_row(row):
    # Exercise the script's actual config wiring without resolving the host's GPU topology.
    with patch.object(resume_suite, "ParallelismConfig", side_effect=lambda **kwargs: SimpleNamespace(**kwargs)):
        config = resume_suite._parallelism_config(
            row.mode,
            MANIFEST[SUITE].nproc,
            fp32_masters=row.fp32_masters,
            eager_loading=row.eager_loading,
            unsharded_ep1_experts=row.unsharded_ep1_experts,
        )
    assert set(vars(config)) <= {field.name for field in fields(ParallelismConfig)}
    return config


def test_all_master_paths_are_explicitly_registered_next_to_the_original_rows():
    spec = MANIFEST[SUITE]
    assert spec.nproc == 2 and "2gpu" in spec.markers
    rows = _rows()
    original = [row for row in rows if not row.fp32_masters]
    assert sorted(row.mode for row in original) == ["cp", "ep", "ep1"]
    assert all(not row.eager_loading and not row.unsharded_ep1_experts for row in original)
    masters = [(row.mode, row.eager_loading, row.unsharded_ep1_experts) for row in rows if row.fp32_masters]
    assert set(masters) == set(MASTER_ROWS), "a checkpoint master layout or loader has no registered GPU row"
    assert len(masters) == len(set(masters)), "duplicate checkpoint master rows"


@pytest.mark.parametrize("mode,eager,unsharded", MASTER_ROWS)
def test_master_rows_reach_all_three_real_config_flags_and_the_declared_axes(mode, eager, unsharded):
    row = next(
        row
        for row in _rows()
        if row.fp32_masters and (row.mode, row.eager_loading, row.unsharded_ep1_experts) == (mode, eager, unsharded)
    )
    config = _configured_row(row)
    assert config.ep_fp32_router and config.ep_fp32_experts and config.fp32_non_ep_params
    assert (config.ep_size, config.cp_size, config.ep_lazy_loading) == MASTER_ROWS[(mode, eager, unsharded)]
    assert config.fsdp_shard_ep1_experts is not unsharded
    assert (mode in resume_suite.MOE_MODES) == (mode in {"ep", "ep1", "ep_cp"}), (
        "pure CP and FSDP2 master rows must load dense models, not the refused managed-EP1 MoE shape"
    )


def test_original_rows_keep_their_default_master_and_loading_policy():
    for row in _rows():
        if not row.fp32_masters:
            config = _configured_row(row)
            assert not config.ep_fp32_router and not config.ep_fp32_experts and not config.fp32_non_ep_params
            assert config.ep_lazy_loading and config.fsdp_shard_ep1_experts


@pytest.mark.parametrize("mode", ["cp", "ep", "ep_cp", "fsdp"])
def test_an_unsharded_ep1_flag_cannot_silently_change_another_axis(mode):
    with pytest.raises(ValueError, match="applies only to --mode ep1"):
        resume_suite._parallelism_config(mode, 2, fp32_masters=True, unsharded_ep1_experts=True)


@pytest.mark.parametrize("preserve", [None, False, True], ids=["fresh-default", "fresh-explicit", "checkpoint"])
def test_resume_construction_provenance_reaches_the_public_loader_before_the_trainer(preserve):
    kwargs = {} if preserve is None else {"preserve_checkpoint_precision": preserve}
    with (
        patch.object(resume_suite, "load_distributed_model", side_effect=_LoaderReached) as loader,
        patch.object(resume_suite, "DistributedSFTTrainer", side_effect=AssertionError("loader must run first")),
        pytest.raises(_LoaderReached),
    ):
        resume_suite._make_trainer("source", SimpleNamespace(is_cp_mode=False), object(), object(), object(), **kwargs)
    assert loader.call_count == 1
    assert loader.call_args.kwargs["preserve_checkpoint_precision"] is (False if preserve is None else preserve)


def test_checkpoint_phases_request_strict_loading_and_fresh_phases_do_not():
    tree = ast.parse(Path(resume_suite.__file__).read_text(encoding="utf-8"))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    phases = {"tiny_dir": [], "ckpt_dir": []}
    for call in ast.walk(run):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "_make_trainer":
            assert isinstance(call.args[0], ast.Name) and call.args[0].id in phases
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            phases[call.args[0].id].append(keywords.get("preserve_checkpoint_precision"))
    assert len(phases["tiny_dir"]) == len(phases["ckpt_dir"]) == 2, "the real four-phase row lost a construction"
    assert all(
        value is None or isinstance(value, ast.Constant) and value.value is False for value in phases["tiny_dir"]
    )
    assert all(isinstance(value, ast.Constant) and value.value is True for value in phases["ckpt_dir"]), (
        "checkpoint and mismatch resumes must both request strict checkpoint coverage"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
