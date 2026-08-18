#!/usr/bin/env python
"""CLI-override edge cases the naive setattr path gets wrong.

Two classes of bug live here: a ``float | str`` field whose string sentinel (``"auto"``) cannot
survive a blind ``float(val)`` cast, and a ``__post_init__`` cross-field guard that setattr bypasses
because it never re-runs. Both are exercised through the real ``H4ArgumentParser`` override path.

Run: ``python tests/cpu/config/test_cli_override_edge_cases.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import pytest

from src.configs.classification_config import ClassificationConfig
from src.training.parser import _union_permits_str


@dataclass
class _ScalarUnionArgs:
    """A field whose string sentinel a consumer resolves at read time."""

    best_completion_emphasis: float | str = 1.0
    plain_float: float = 0.0


def test_union_permits_str_detects_the_string_member():
    import typing

    hints = typing.get_type_hints(_ScalarUnionArgs)
    assert _union_permits_str(hints["best_completion_emphasis"])
    assert not _union_permits_str(hints["plain_float"])
    assert not _union_permits_str(float)


def _override(parser, args):
    """Drive the YAML+CLI override path with an empty YAML file so only the CLI values apply."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("{}\n")
        yaml_path = f.name
    try:
        return parser.parse_yaml_and_args(yaml_path, args)
    finally:
        os.unlink(yaml_path)


def test_float_or_str_field_keeps_a_non_numeric_cli_override():
    from src.training.parser import H4ArgumentParser

    parser = H4ArgumentParser((_ScalarUnionArgs,))
    (parsed,) = _override(parser, ["--best_completion_emphasis=auto"])
    assert parsed.best_completion_emphasis == "auto", "string sentinel must survive, not crash on float('auto')"


def test_float_or_str_field_still_casts_a_numeric_cli_override():
    from src.training.parser import H4ArgumentParser

    parser = H4ArgumentParser((_ScalarUnionArgs,))
    (parsed,) = _override(parser, ["--best_completion_emphasis=1.5"])
    assert parsed.best_completion_emphasis == pytest.approx(1.5)
    assert isinstance(parsed.best_completion_emphasis, float)


def test_plain_float_still_rejects_a_non_numeric_override():
    from src.training.parser import H4ArgumentParser

    parser = H4ArgumentParser((_ScalarUnionArgs,))
    with pytest.raises(ValueError):
        _override(parser, ["--plain_float=auto"])


def test_classification_mutual_exclusion_guard_fires_on_cli_override():
    """The YAML path already rejects setting both; the CLI setattr path must too (__post_override__)."""
    from src.training.parser import H4ArgumentParser

    parser = H4ArgumentParser((ClassificationConfig,))
    with pytest.raises(ValueError, match="mutually exclusive"):
        _override(
            parser,
            ["--output_dir=/tmp/x", "--derive_class_weights=true", "--class_weights=1.0,2.0"],
        )


def test_classification_single_class_weight_override_is_accepted():
    from src.training.parser import H4ArgumentParser

    parser = H4ArgumentParser((ClassificationConfig,))
    (parsed,) = _override(parser, ["--output_dir=/tmp/x", "--derive_class_weights=true"])
    assert parsed.derive_class_weights is True
    assert parsed.class_weights is None


def test_post_override_guard_is_a_noop_when_neither_field_overridden():
    cfg = ClassificationConfig(output_dir="/tmp/x", derive_class_weights=True)
    cfg.__post_override__({"learning_rate"})  # must not raise on an unrelated override


@dataclass
class _OptionalUnionArgs:
    """Optional fields across the cast spectrum, including the report_to-shaped container union."""

    report_to: None | str | list[str] = "wandb"
    max_len: int | None = 512
    mode: Literal["none", "all"] | None = "all"
    bf16: bool | None = None


@pytest.mark.parametrize("spelling", ["none", "None", "null"])
def test_optional_container_union_clears_to_none(spelling):
    """--report_to=none must clear the field: it is the standard smoke-run override, and the
    None | str | list[str] union has no confident value cast to hide behind."""
    from src.training.parser import H4ArgumentParser

    parser = H4ArgumentParser((_OptionalUnionArgs,))
    (parsed,) = _override(parser, [f"--report_to={spelling}"])
    assert parsed.report_to is None


def test_optional_container_union_still_rejects_a_value_override():
    """Setting a VALUE on the container union stays unsupported — a raw string in a list field
    is iterated char-wise downstream, so the guard must survive the clearing feature."""
    from src.training.parser import H4ArgumentParser

    parser = H4ArgumentParser((_OptionalUnionArgs,))
    with pytest.raises(ValueError, match="not supported"):
        _override(parser, ["--report_to=wandb"])


@pytest.mark.parametrize("spelling", ["none", "None", "null"])
def test_optional_bool_refuses_none_clearing(spelling):
    """--bf16=none must RAISE, not clear: an optional bool reads any none-spelling as a mistyped
    boolean, and clearing it would silently flip the run's precision or re-arm an auto-default."""
    from src.training.parser import H4ArgumentParser

    parser = H4ArgumentParser((_OptionalUnionArgs,))
    with pytest.raises(ValueError, match="boolean"):
        _override(parser, [f"--bf16={spelling}"])


def test_optional_bool_still_parses_true_false():
    from src.training.parser import H4ArgumentParser

    parser = H4ArgumentParser((_OptionalUnionArgs,))
    (parsed,) = _override(parser, ["--bf16=false"])
    assert parsed.bf16 is False


def test_optional_int_clears_to_none():
    from src.training.parser import H4ArgumentParser

    parser = H4ArgumentParser((_OptionalUnionArgs,))
    (parsed,) = _override(parser, ["--max_len=None"])
    assert parsed.max_len is None


def test_literal_choice_spelled_none_outranks_clearing():
    """A field whose Literal includes the string "none" must get the string from --mode=none;
    the capitalized spellings still clear the Optional."""
    from src.training.parser import H4ArgumentParser

    parser = H4ArgumentParser((_OptionalUnionArgs,))
    (parsed,) = _override(parser, ["--mode=none"])
    assert parsed.mode == "none"
    (parsed,) = _override(parser, ["--mode=None"])
    assert parsed.mode is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
