#!/usr/bin/env python
"""CLI-override casting and dispatch in ``HaloArgumentParser``: Optional numeric fields are cast, bools parse
strictly, and launch commands use ``--key=value``."""

import re
import tempfile
from dataclasses import dataclass

import pytest

from halo.configs.parser import HaloArgumentParser
from tests.common.repo import REPO_ROOT


@dataclass
class _ScriptArgs:
    num_eval_examples: int | None = 50
    ratio: float | None = None
    verbose: bool = False


@dataclass
class _TrainConfig:
    lr: float = 1e-4


def _parse(overrides):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("lr: 0.001\n")
        yaml_path = f.name
    parser = HaloArgumentParser((_ScriptArgs, _TrainConfig))
    return parser.parse_dict(parser.load_values([yaml_path, *overrides]))  # returns [_ScriptArgs, _TrainConfig]


def test_optional_int_and_float_cast():
    script, _ = _parse(["--num_eval_examples=100", "--ratio=0.5"])
    assert script.num_eval_examples == 100 and isinstance(script.num_eval_examples, int)
    assert script.ratio == 0.5 and isinstance(script.ratio, float)


def test_bool_override_parses_truthy_and_falsey():
    script, _ = _parse(["--verbose=1"])
    assert script.verbose is True
    script, _ = _parse(["--verbose=false"])
    assert script.verbose is False


def test_bool_override_rejects_garbage():
    try:
        _parse(["--verbose=maybe"])
    except ValueError as e:
        assert "boolean" in str(e).lower()
    else:
        raise AssertionError("an unparseable boolean override must raise, not silently become False")


def test_example_launch_commands_use_the_equals_form():
    """``load_values`` only accepts ``--key=value``, and example headers are copy-pasted verbatim. Scans the tree."""
    launch_line = re.compile(r"scripts/training/\S+\.py.*")
    # A torchrun/accelerate flag consumed by the launcher before the script path is no override.
    space_form = re.compile(
        r"--(?!nproc_per_node|master_port|rdzv|config_file|num_processes|main_process_port)[a-z_]+\s+[^-\s]"
    )
    offenders = []
    for config in sorted((REPO_ROOT / "examples").rglob("*.yaml")):
        for lineno, line in enumerate(config.read_text().splitlines(), 1):
            match = launch_line.search(line)
            if match and space_form.search(match.group()):
                offenders.append(f"{config.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "example launch commands must pass overrides as --key=value; these use the space form, "
        "which load_values rejects outright:\n  " + "\n  ".join(offenders)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
