#!/usr/bin/env python
"""The entry-script runs (``trainers/other/test_entry_scripts.py``) are held to the training scripts: every method
under ``scripts/training`` has a case, a tiny recipe and a GPU row, and each run config parses with its script's
own arguments under the overrides its launches pass, its environment building where it names one. A renamed
config field or a new script fails here, not first on the GPU tier.

Run: python tests/cpu/config/test_entry_script_recipes.py
"""

import ast
import dataclasses
import inspect
import textwrap

import pytest
from trl import GRPOConfig

from halo.configs.grpo_environmental import EnvironmentConfig
from halo.configs.parser import HaloArgumentParser
from halo.environments.registry import build_environment
from halo.trainers.grpo.environmental.environmental_rules import reject_unread_grpo_knobs
from tests.common.entry_scripts import (
    CASES,
    FIRST_STEPS,
    RECIPES_DIR,
    RESUMED_STEPS,
    first_launch_overrides,
    method_scripts,
    recipe_path,
    resume_overrides,
    write_run_config,
)
from tests.common.repo import REPO_ROOT, load_script_module
from tests.gpu.manifest import MANIFEST

_ENTRY_SCRIPT_ROWS = "trainers/other/test_entry_scripts.py"
_MODEL = "org/tiny-model"
_SERVER_URL = "http://localhost:8000"
# A launch passes a weight-transfer port of its own (the GPU runs draw a free one), never a field's default.
_GROUP_PORT = 29517
# What a CUDA-less host needs to build the training configs; the GPU runs pass neither.
_CPU_ONLY = ["--use_cpu=true", "--bf16=false"]


def _row_methods() -> dict[str, tuple]:
    """Each entry-script row's method and markers."""
    return {args.removeprefix("--method "): spec.markers for args, spec in MANIFEST[_ENTRY_SCRIPT_ROWS].launches()}


def test_every_training_method_has_a_case_a_recipe_and_a_gpu_row():
    methods = set(method_scripts())
    assert methods, "the method index came back empty, so nothing below would be checked"
    assert set(CASES) == methods
    assert set(RECIPES_DIR.rglob("*.yaml")) == {recipe_path(method) for method in methods}
    assert set(_row_methods()) == methods


def test_the_engine_driven_rows_need_the_live_server_and_the_rest_gate_merges():
    for method, markers in _row_methods().items():
        served = CASES[method].server_fields is not None
        assert ("vllm_server" in markers) is served, method
        assert ("full" if served else "core") in markers, method


def _script_arguments(method: str) -> tuple[type, ...]:
    """The dataclasses ``method``'s ``main`` parses its command line into, read off its ``parse_script_args``
    call in the module that defines it."""
    script = method_scripts()[method]
    main = load_script_module(str(script.relative_to(REPO_ROOT)), f"entry_recipe_{script.stem}").main
    call = next(
        node
        for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(main))))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "parse_script_args"
    )
    return tuple(main.__globals__[argument.id] for argument in call.args)


@pytest.mark.parametrize("method", sorted(CASES))
@pytest.mark.parametrize(
    ("overrides", "max_steps", "resumes"),
    [(first_launch_overrides(), FIRST_STEPS, None), (resume_overrides(), RESUMED_STEPS, "true")],
    ids=["first-launch", "resume"],
)
def test_each_run_config_parses_with_its_scripts_own_arguments(tmp_path, method, overrides, max_steps, resumes):
    dataset = str(tmp_path / "data")
    config = write_run_config(method, tmp_path, model=_MODEL, dataset_dir=dataset, server_url=_SERVER_URL)
    server_fields = CASES[method].server_fields
    port = [] if server_fields is None else [f"--{server_fields[1]}={_GROUP_PORT}"]
    parser = HaloArgumentParser(_script_arguments(method))

    parsed = parser.parse_dict(parser.load_values([str(config), *overrides, *port, *_CPU_ONLY]))

    values = {field.name: getattr(obj, field.name) for obj in parsed for field in dataclasses.fields(obj)}
    assert values["max_steps"] == max_steps and values["save_steps"] == 1
    assert values["resume_from_checkpoint"] == resumes
    assert values["output_dir"] == str(tmp_path / "output")
    assert values["dataset"] == dataset
    for field in CASES[method].model_fields:
        assert values[field] == _MODEL, field
    if server_fields is not None:
        assert (values[server_fields[0]], values[server_fields[1]]) == (_SERVER_URL, _GROUP_PORT)
    # environment_kwargs reach only the environment's constructor, and a GRPO knob environmental GRPO leaves unread
    # (the launch's port among them) only its trainer's refusal: no parse reaches either.
    for environment in (obj for obj in parsed if isinstance(obj, EnvironmentConfig)):
        build_environment(environment.environment_type, environment.to_env_config())
        reject_unread_grpo_knobs(next(obj for obj in parsed if isinstance(obj, GRPOConfig)))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
