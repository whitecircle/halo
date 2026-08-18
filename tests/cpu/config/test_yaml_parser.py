#!/usr/bin/env python
"""
Tests for H4ArgumentParser YAML parsing, unknown-key rejection, toolkit defaults, and CLI
override type casting.

Run: python tests/cpu/config/test_yaml_parser.py
"""

import os
import sys
import tempfile
from dataclasses import dataclass, field, make_dataclass
from typing import Literal

import pytest
from transformers import TrainingArguments

from src.training.parser import (
    _TOOLKIT_DEFAULTS,
    H4ArgumentParser,
)

# Simple target dataclasses for parsing


@dataclass
class SimpleConfig:
    max_length: int = 512
    learning_rate: float = 1e-4
    use_liger_kernel: bool = False
    bf16: bool = False
    name: str = "default"


@dataclass
class OutputDirConfig:
    output_dir: str = "output/default"
    name: str = "default"


@dataclass
class ExtraConfig:
    batch_size: int = 8
    tags: list[str] = field(default_factory=list)
    verbose: bool = True


# Helpers


def _write_yaml(content: str, suffix: str = ".yaml") -> str:
    """Write YAML content to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


# parse_yaml_file: nothing is migrated, nothing is stripped


def test_unknown_field_is_not_silently_stripped():
    """Nothing is stripped and nothing is migrated: a key no config declares — a retired knob like
    `use_unsloth`, or a spelling this repo retired — must fail loud rather than parse and do
    nothing."""
    path = _write_yaml("name: unknown_test\nuse_unsloth: true\n")
    try:
        parser = H4ArgumentParser((SimpleConfig,))
        with pytest.raises(ValueError, match="use_unsloth"):
            parser.parse_yaml_file(path, allow_extra_keys=False)
    finally:
        os.unlink(path)


def test_retired_ecosystem_spelling_is_not_migrated():
    """No rename table: TRL's own retired ``max_seq_length`` reaches the strict check and raises,
    naming the field, instead of being quietly rewritten onto ``max_length`` forever."""
    path = _write_yaml("max_seq_length: 2048\nname: retired\n")
    try:
        parser = H4ArgumentParser((SimpleConfig,))
        with pytest.raises(ValueError, match="max_seq_length"):
            parser.parse_yaml_file(path, allow_extra_keys=False)
    finally:
        os.unlink(path)


def test_plain_config_parses_unchanged():
    """A YAML using only current spellings parses with nothing rewritten."""
    path = _write_yaml("name: clean\n")
    try:
        parser = H4ArgumentParser((SimpleConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert result.name == "clean"
    finally:
        os.unlink(path)


# _apply_toolkit_defaults


def test_toolkit_defaults_applied():
    """use_liger_kernel and bf16 should default to True when not explicitly set."""
    path = _write_yaml("name: defaults_test\n")
    try:
        parser = H4ArgumentParser((SimpleConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        # Before applying defaults
        assert result.use_liger_kernel is False
        assert result.bf16 is False
        # Apply toolkit defaults (no fields explicitly set)
        H4ArgumentParser._apply_toolkit_defaults([result], set())
        assert result.use_liger_kernel is True, "use_liger_kernel should be True"
        assert result.bf16 is True, "bf16 should be True"
    finally:
        os.unlink(path)


def test_every_declared_toolkit_default_is_applied():
    """Derived from the registry, so a default added later cannot ship unexercised.

    Each entry is a knob the toolkit overrides upstream on — ``logging_nan_inf_filter: False``
    exists because upstream's ``True`` reads a device scalar per micro-batch AND replaces a NaN with
    the running average. A default that silently stops being applied restores the upstream behavior
    with nothing said.
    """
    opposites = {name: (not value if isinstance(value, bool) else None) for name, value in _TOOLKIT_DEFAULTS.items()}
    config = make_dataclass("AllDefaults", [(name, bool, field(default=v)) for name, v in opposites.items()])()

    H4ArgumentParser._apply_toolkit_defaults([config], set())

    for name, expected in _TOOLKIT_DEFAULTS.items():
        assert getattr(config, name) == expected, f"toolkit default {name}={expected} was not applied"


def test_toolkit_defaults_not_overridden():
    """Explicitly set fields should not be overridden by toolkit defaults."""
    path = _write_yaml("use_liger_kernel: false\nbf16: false\nname: explicit\n")
    try:
        parser = H4ArgumentParser((SimpleConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        # Simulate user explicitly setting these
        explicitly_set = {"use_liger_kernel", "bf16"}
        H4ArgumentParser._apply_toolkit_defaults([result], explicitly_set)
        assert result.use_liger_kernel is False, "Should remain False when explicitly set"
        assert result.bf16 is False, "Should remain False when explicitly set"
    finally:
        os.unlink(path)


def test_toolkit_defaults_partial_override():
    """Only non-explicitly-set toolkit defaults should be applied."""
    path = _write_yaml("use_liger_kernel: false\nname: partial\n")
    try:
        parser = H4ArgumentParser((SimpleConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        # User explicitly set use_liger_kernel only
        explicitly_set = {"use_liger_kernel"}
        H4ArgumentParser._apply_toolkit_defaults([result], explicitly_set)
        assert result.use_liger_kernel is False, "Explicitly set, should stay False"
        assert result.bf16 is True, "Not explicitly set, should become True"
    finally:
        os.unlink(path)


def test_toolkit_defaults_set_on_first_matching_dataclass_only():
    """Each toolkit default is applied to exactly ONE object — the first that has the field.

    The loop ``break``s after the first match, so when two parsed dataclasses both expose
    bf16, only the first receives the default. Pinning this prevents a future refactor from
    silently scattering the flag onto every config.
    """

    @dataclass
    class First:
        bf16: bool = False

    @dataclass
    class Second:
        bf16: bool = False

    a, b = First(), Second()
    H4ArgumentParser._apply_toolkit_defaults([a, b], set())
    assert a.bf16 is True, "first matching dataclass gets the default"
    assert b.bf16 is False, "second matching dataclass is left untouched (break after first)"


def test_toolkit_defaults_skip_objects_without_field():
    """Objects lacking a toolkit field are skipped; the default lands on the one that has it."""

    @dataclass
    class NoFlags:
        x: int = 1

    @dataclass
    class HasFlag:
        use_liger_kernel: bool = False

    no_flags, has_flag = NoFlags(), HasFlag()
    H4ArgumentParser._apply_toolkit_defaults([no_flags, has_flag], set())
    assert has_flag.use_liger_kernel is True
    assert not hasattr(no_flags, "use_liger_kernel")


# parse_yaml_and_args: CLI override type casting


def test_cli_override_int():
    """CLI args should cast int fields properly."""
    path = _write_yaml("name: cli_int\n")
    try:
        parser = H4ArgumentParser((SimpleConfig,))
        (result,) = parser.parse_yaml_and_args(path, ["--max_length=2048"])
        assert result.max_length == 2048
        assert isinstance(result.max_length, int)
    finally:
        os.unlink(path)


def test_cli_override_float():
    """CLI args should cast float fields properly."""
    path = _write_yaml("name: cli_float\n")
    try:
        parser = H4ArgumentParser((SimpleConfig,))
        (result,) = parser.parse_yaml_and_args(path, ["--learning_rate=0.001"])
        assert abs(result.learning_rate - 0.001) < 1e-10
        assert isinstance(result.learning_rate, float)
    finally:
        os.unlink(path)


def test_cli_override_bool_true():
    """CLI args should cast bool fields: 'true'/'True' -> True."""
    path = _write_yaml("name: cli_bool\n")
    try:
        parser = H4ArgumentParser((SimpleConfig,))
        (result,) = parser.parse_yaml_and_args(path, ["--use_liger_kernel=True"])
        assert result.use_liger_kernel is True
    finally:
        os.unlink(path)


def test_cli_override_bool_false():
    """CLI args should cast bool fields: anything else -> False."""
    path = _write_yaml("use_liger_kernel: true\nname: cli_bool_f\n")
    try:
        parser = H4ArgumentParser((SimpleConfig,))
        (result,) = parser.parse_yaml_and_args(path, ["--use_liger_kernel=false"])
        assert result.use_liger_kernel is False
    finally:
        os.unlink(path)


def test_cli_override_list_str():
    """CLI args should split comma-separated values into List[str]."""
    path = _write_yaml("batch_size: 16\n")
    try:
        parser = H4ArgumentParser((ExtraConfig,))
        (result,) = parser.parse_yaml_and_args(path, ["--tags=a,b,c"])
        assert result.tags == ["a", "b", "c"]
    finally:
        os.unlink(path)


def test_cli_override_list_str_replaces_yaml_list():
    """A list override must REPLACE the YAML list, not extend it — the run must see exactly the
    values the CLI named."""
    path = _write_yaml("tags:\n- from yaml\n")
    try:
        parser = H4ArgumentParser((ExtraConfig,))
        (result,) = parser.parse_yaml_and_args(path, ["--tags=first,second"])
        assert result.tags == ["first", "second"], f"--tags override did not replace the YAML list: {result.tags}"
    finally:
        os.unlink(path)


def test_cli_override_unknown_arg_raises():
    """A CLI arg that matches no dataclass field fails loudly (a silently-dropped override would
    run the job with the un-overridden value)."""
    path = _write_yaml("name: ignore_unknown\n")
    try:
        parser = H4ArgumentParser((SimpleConfig,))
        with pytest.raises(ValueError, match="not_a_field"):
            parser.parse_yaml_and_args(path, ["--not_a_field=123", "--max_length=64"])
    finally:
        os.unlink(path)


def test_cli_override_shared_field_sets_on_both_dataclasses():
    """A field owned by two dataclasses (e.g. pad_token on SFTScriptArguments + SFTConfig) is set
    on BOTH — the override must not raise 'duplicate' (token-sync relies on this)."""

    @dataclass
    class DupA:
        shared: int = 0

    @dataclass
    class DupB:
        shared: int = 0

    parser = H4ArgumentParser((DupA, DupB))
    path = _write_yaml("shared: 1\n")
    try:
        a, b = parser.parse_yaml_and_args(path, ["--shared=7"])
        assert a.shared == 7 and b.shared == 7
    finally:
        os.unlink(path)


def test_cli_override_genuinely_repeated_flag_raises():
    """The same flag passed twice on the CLI is a real duplicate and must raise (not last-wins)."""
    parser = H4ArgumentParser((SimpleConfig,))
    path = _write_yaml("max_length: 100\n")
    try:
        parser.parse_yaml_and_args(path, ["--max_length=1", "--max_length=2"])
        raise AssertionError("Should have raised ValueError for a repeated CLI flag")
    except ValueError as e:
        assert "duplicate" in str(e).lower()
    finally:
        os.unlink(path)


def test_cli_override_applies_after_yaml_value():
    """A CLI override beats the YAML-loaded value for the same field."""
    path = _write_yaml("max_length: 100\nname: override_me\n")
    try:
        parser = H4ArgumentParser((SimpleConfig,))
        (result,) = parser.parse_yaml_and_args(path, ["--max_length=999"])
        assert result.max_length == 999
    finally:
        os.unlink(path)


def test_overlapping_fields_resolved():
    """Overlapping fields across dataclasses should be resolved (last wins) without error."""

    @dataclass
    class OverlapA:
        shared_field: int = 0
        only_a: str = "a"

    @dataclass
    class OverlapB:
        shared_field: int = 99
        only_b: str = "b"

    # Should not raise — conflict_handler='resolve' allows duplicate fields
    parser = H4ArgumentParser((OverlapA, OverlapB))

    # Parse with a YAML that sets the shared field
    path = _write_yaml("shared_field: 42\nonly_a: hello\nonly_b: world\n")
    try:
        result_a, result_b = parser.parse_yaml_file(path, allow_extra_keys=False)
        # Last dataclass (OverlapB) wins for the shared field
        assert result_b.shared_field == 42, f"Expected 42, got {result_b.shared_field}"
        assert result_a.only_a == "hello"
        assert result_b.only_b == "world"
    finally:
        os.unlink(path)


# parse_yaml_file: empty YAML


def test_empty_yaml():
    """Empty YAML file should produce dataclass with all defaults."""
    path = _write_yaml("")
    try:
        parser = H4ArgumentParser((SimpleConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert result.max_length == 512
        assert result.name == "default"
    finally:
        os.unlink(path)


# _format_output_dir: strftime expansion


def test_format_output_dir_with_strftime():
    """output_dir containing strftime codes should be expanded."""
    from datetime import datetime

    path = _write_yaml('output_dir: "output/run-%Y-%m-%d"\nname: fmt\n')
    try:
        parser = H4ArgumentParser((OutputDirConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        H4ArgumentParser._format_output_dir([result])
        expected = datetime.now().strftime("output/run-%Y-%m-%d")
        assert result.output_dir == expected, f"Expected {expected}, got {result.output_dir}"
    finally:
        os.unlink(path)


def test_format_output_dir_no_strftime():
    """output_dir without strftime codes should remain unchanged."""
    path = _write_yaml('output_dir: "output/plain-run"\nname: nofmt\n')
    try:
        parser = H4ArgumentParser((OutputDirConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        H4ArgumentParser._format_output_dir([result])
        assert result.output_dir == "output/plain-run"
    finally:
        os.unlink(path)


def test_format_output_dir_full_datetime():
    """output_dir with full datetime pattern should expand correctly."""
    template = "output/sft-%Y-%m-%dT%H-%M-%S"
    path = _write_yaml(f'output_dir: "{template}"\n')
    try:
        parser = H4ArgumentParser((OutputDirConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        H4ArgumentParser._format_output_dir([result])
        assert "%" not in result.output_dir, f"Unexpanded codes in: {result.output_dir}"
        assert result.output_dir.startswith("output/sft-")
    finally:
        os.unlink(path)


def test_format_output_dir_percent_prose_survives():
    """Non-directive percent sequences are prose and must survive byte-identical — a whole-string
    strftime lets glibc expand `%-d` (no-padding day), turning `sft-100%-data` into `sft-10014ata`."""
    for raw in ("output/sft-100%-data", "output/50%_subset", "output/run-100%"):
        path = _write_yaml(f'output_dir: "{raw}"\n')
        try:
            parser = H4ArgumentParser((OutputDirConfig,))
            (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
            H4ArgumentParser._format_output_dir([result])
            assert result.output_dir == raw, f"prose percent mangled: {raw!r} -> {result.output_dir!r}"
        finally:
            os.unlink(path)


def test_format_output_dir_mixed_prose_and_directives():
    """Real directives expand while adjacent prose percents stay intact."""
    from datetime import datetime

    path = _write_yaml('output_dir: "output/run-50%_subset-%Y%m%d"\n')
    try:
        parser = H4ArgumentParser((OutputDirConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        H4ArgumentParser._format_output_dir([result])
        expected = "output/run-50%_subset-" + datetime.now().strftime("%Y%m%d")
        assert result.output_dir == expected, f"Expected {expected}, got {result.output_dir}"
    finally:
        os.unlink(path)


def test_format_output_dir_percent_escape():
    """%% keeps its strftime escape meaning: a literal percent."""
    path = _write_yaml('output_dir: "output/100%%-data"\n')
    try:
        parser = H4ArgumentParser((OutputDirConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        H4ArgumentParser._format_output_dir([result])
        assert result.output_dir == "output/100%-data"
    finally:
        os.unlink(path)


def test_format_output_dir_no_output_dir_field():
    """Objects without output_dir should be silently skipped."""
    path = _write_yaml("name: no_outdir\n")
    try:
        parser = H4ArgumentParser((SimpleConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        H4ArgumentParser._format_output_dir([result])
        assert not hasattr(result, "output_dir")
    finally:
        os.unlink(path)


# YAML 1.2 numeric parsing (ruamel.yaml)


def test_scientific_notation_no_decimal_parsed_as_float():
    """YAML 1.2 should parse '3e-5' as float (PyYAML 1.1 returns string)."""
    path = _write_yaml("learning_rate: 3e-5\nname: sci\n")
    try:
        parser = H4ArgumentParser((SimpleConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert isinstance(result.learning_rate, float), f"Expected float, got {type(result.learning_rate)}"
        assert abs(result.learning_rate - 3e-5) < 1e-10
    finally:
        os.unlink(path)


def test_scientific_notation_with_decimal_parsed_as_float():
    """YAML 1.2 should also parse '1.5e-5' as float (regression check)."""
    path = _write_yaml("learning_rate: 1.5e-5\nname: sci_dec\n")
    try:
        parser = H4ArgumentParser((SimpleConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert isinstance(result.learning_rate, float)
        assert abs(result.learning_rate - 1.5e-5) < 1e-10
    finally:
        os.unlink(path)


# __post_override__: re-deriving __post_init__ state after CLI overrides


@dataclass
class DerivedStateConfig:
    base: int = 1
    other: str = "x"

    def __post_init__(self):
        self.doubled = self.base * 2

    def __post_override__(self, overridden_fields: set):
        self.seen_overrides = set(overridden_fields)
        if "base" in overridden_fields:
            self.doubled = self.base * 2


def test_post_override_hook_rederives_state():
    """A CLI override bypasses __init__; the hook must re-derive __post_init__-derived state."""
    path = _write_yaml("base: 3\n")
    try:
        parser = H4ArgumentParser((DerivedStateConfig,))
        (result,) = parser.parse_yaml_and_args(path, ["--base=5"])
        assert result.base == 5
        assert result.doubled == 10, f"derived state stale after override: {result.doubled}"
        assert result.seen_overrides == {"base"}
    finally:
        os.unlink(path)


def test_post_override_hook_skipped_without_overrides():
    path = _write_yaml("base: 3\n")
    try:
        parser = H4ArgumentParser((DerivedStateConfig,))
        (result,) = parser.parse_yaml_and_args(path, [])
        assert result.doubled == 6
        assert not hasattr(result, "seen_overrides"), "__post_override__ must not run without overrides"
    finally:
        os.unlink(path)


def test_list_cli_override_replaces_yaml_list():
    """A CLI list override must fully replace the YAML list, never merge with it."""
    from src.args.environmental_grpo_args import EnvironmentalGRPOScriptArguments

    path = _write_yaml("context_fields:\n- old_field\n")
    try:
        parser = H4ArgumentParser((EnvironmentalGRPOScriptArguments,))
        (result,) = parser.parse_yaml_and_args(path, ["--context_fields=first_field,second_field"])
        assert result.context_fields == ["first_field", "second_field"], (
            f"--context_fields override did not replace the YAML list: {result.context_fields}"
        )
    finally:
        os.unlink(path)


# parse(): .yml suffix + explicit first-position CLI flag


def _parse_with_argv(parser, argv):
    old_argv = sys.argv
    sys.argv = argv
    try:
        return parser.parse()
    finally:
        sys.argv = old_argv


def test_parse_accepts_yml_suffix():
    """A .yml config must go down the YAML branch, not be mistaken for argparse flags."""
    path = _write_yaml("max_length: 777\nname: yml\n", suffix=".yml")
    try:
        parser = H4ArgumentParser((SimpleConfig,))
        result = _parse_with_argv(parser, ["prog", path])
        assert result.max_length == 777
        assert result.name == "yml"
    finally:
        os.unlink(path)


def test_first_cli_flag_counts_as_explicit_without_yaml():
    """In the pure-CLI launch form the first flag is sys.argv[1]; it must still block the
    toolkit default from overwriting it (bf16 defaults to True when not explicitly set)."""
    parser = H4ArgumentParser((SimpleConfig,))
    result = _parse_with_argv(parser, ["prog", "--bf16", "false", "--name", "cli"])
    assert result.name == "cli"
    assert result.bf16 is False, "explicit first-position --bf16 false was clobbered by the toolkit default"


def test_dashed_cli_flag_counts_as_explicit():
    """argparse accepts `--use-liger-kernel=false`, so the explicit-set scan must normalize the
    dashed spelling to the underscore field name the toolkit-default check reads — recording it
    dashed lets the default REVERSE the user's false back to True."""
    parser = H4ArgumentParser((SimpleConfig,))
    result = _parse_with_argv(parser, ["prog", "--use-liger-kernel=false", "--name", "cli"])
    assert result.use_liger_kernel is False, "toolkit default reversed the dashed-spelled --use-liger-kernel=false"


def test_explicit_set_scan_normalizes_dashed_flags():
    old_argv = sys.argv
    sys.argv = ["prog", "--use-liger-kernel=false", "--per-device-train-batch-size", "2"]
    try:
        explicitly_set = H4ArgumentParser._get_explicitly_set_fields()
    finally:
        sys.argv = old_argv
    assert "use_liger_kernel" in explicitly_set
    assert "per_device_train_batch_size" in explicitly_set


def test_dashed_cli_override_with_yaml():
    """The YAML+override path normalizes dashed flags to the field spelling too — same convention
    as argparse — instead of rejecting them as unknown."""
    path = _write_yaml("use_liger_kernel: true\nname: dash\n")
    try:
        parser = H4ArgumentParser((SimpleConfig,))
        (result,) = parser.parse_yaml_and_args(path, ["--use-liger-kernel=false"])
        assert result.use_liger_kernel is False
    finally:
        os.unlink(path)


def test_dashed_and_underscored_spellings_are_one_flag():
    """The duplicate-flag guard must see through the spelling difference."""
    path = _write_yaml("name: dash_dup\n")
    try:
        parser = H4ArgumentParser((SimpleConfig,))
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            parser.parse_yaml_and_args(path, ["--use-liger-kernel=false", "--use_liger_kernel=true"])
    finally:
        os.unlink(path)


# mixed_precision stays in sync with post-parse fp16/bf16 mutation (F1-A)


def test_toolkit_bf16_default_syncs_mixed_precision():
    """TrainingArguments.__post_init__ derives mixed_precision BEFORE the toolkit bf16 default is
    applied; parse() must re-derive it or the Accelerator autocasts 'no' while bf16 is True."""
    path = _write_yaml("output_dir: /tmp/h4_mp_default\n")
    try:
        parser = H4ArgumentParser((TrainingArguments,))
        result = _parse_with_argv(parser, ["prog", path])
        assert result.bf16 is True
        assert result.mixed_precision == "bf16", f"stale mixed_precision: {result.mixed_precision}"
    finally:
        os.unlink(path)


def test_toolkit_bf16_default_yields_to_explicit_fp16():
    """An explicit fp16 must win over the toolkit bf16 default — applying both would form the
    fp16+bf16 pair TrainingArguments rejects (and an fp16 GradScaler over a bf16 autocast)."""
    path = _write_yaml("output_dir: /tmp/h4_mp_fp16\nfp16: true\n")
    try:
        parser = H4ArgumentParser((TrainingArguments,))
        result = _parse_with_argv(parser, ["prog", path])
        assert result.fp16 is True
        assert result.bf16 is False, "toolkit bf16 default must not apply on top of explicit fp16"
        assert result.mixed_precision == "fp16"
    finally:
        os.unlink(path)


def test_cli_bf16_false_override_rederives_mixed_precision():
    """--bf16=false after a bf16 YAML must reset mixed_precision to 'no' — otherwise the model
    loads fp32 while the Accelerator still autocasts bf16."""
    # use_cpu keeps TrainingArguments' bf16-support validation off GPU-less test machines.
    path = _write_yaml("output_dir: /tmp/h4_mp_cli\nbf16: true\nuse_cpu: true\n")
    try:
        parser = H4ArgumentParser((TrainingArguments,))
        result = _parse_with_argv(parser, ["prog", path, "--bf16=false"])
        assert result.bf16 is False
        assert result.mixed_precision == "no", f"stale mixed_precision: {result.mixed_precision}"
    finally:
        os.unlink(path)


def test_cli_fp16_override_conflicting_with_yaml_bf16_raises():
    """--fp16=true on top of bf16: true escapes __post_init__'s at-most-one check via setattr;
    the re-derivation must fail loud instead of training with both flags set."""
    path = _write_yaml("output_dir: /tmp/h4_mp_conflict\nbf16: true\nuse_cpu: true\n")
    try:
        parser = H4ArgumentParser((TrainingArguments,))
        with pytest.raises(ValueError, match="At most one of fp16 and bf16"):
            _parse_with_argv(parser, ["prog", path, "--fp16=true"])
    finally:
        os.unlink(path)


# Literal-annotated fields are validated at parse time (F1-D)


@dataclass
class LiteralConfig:
    mode: Literal["alpha", "beta"] = "alpha"
    level: Literal[1, 2, 3] = 1
    opt_mode: Literal["x", "y"] | None = None
    hybrid: Literal["auto"] | str = "auto"
    name: str = "default"


def test_literal_field_invalid_yaml_value_raises():
    """parse_dict bypasses argparse choices; an invalid Literal value must fail at parse time,
    not deep in training."""
    path = _write_yaml("mode: banana\n")
    try:
        parser = H4ArgumentParser((LiteralConfig,))
        with pytest.raises(ValueError, match="banana.*mode.*alpha"):
            parser.parse_yaml_file(path, allow_extra_keys=False)
    finally:
        os.unlink(path)


def test_literal_field_valid_values_pass_through():
    path = _write_yaml("mode: beta\nlevel: 2\nopt_mode: y\n")
    try:
        parser = H4ArgumentParser((LiteralConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert result.mode == "beta"
        assert result.level == 2
        assert result.opt_mode == "y"
    finally:
        os.unlink(path)


def test_optional_literal_field_validates_and_allows_none():
    path = _write_yaml("opt_mode: banana\n")
    try:
        parser = H4ArgumentParser((LiteralConfig,))
        with pytest.raises(ValueError, match="opt_mode"):
            parser.parse_yaml_file(path, allow_extra_keys=False)
    finally:
        os.unlink(path)
    path = _write_yaml("opt_mode: null\n")
    try:
        parser = H4ArgumentParser((LiteralConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert result.opt_mode is None, "Optional[Literal] must keep accepting None"
    finally:
        os.unlink(path)


def test_mixed_union_literal_is_not_validated():
    """Literal['auto'] | str admits values outside the literal set — no confident validation."""
    path = _write_yaml("hybrid: custom-value\n")
    try:
        parser = H4ArgumentParser((LiteralConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert result.hybrid == "custom-value"
    finally:
        os.unlink(path)


def test_literal_cli_override_validates_and_casts():
    """CLI overrides bypass argparse too: invalid values raise, valid str and int literals cast."""
    path = _write_yaml("name: lit_cli\n")
    try:
        parser = H4ArgumentParser((LiteralConfig,))
        with pytest.raises(ValueError, match="banana.*--mode|--mode.*banana"):
            parser.parse_yaml_and_args(path, ["--mode=banana"])
        (result,) = parser.parse_yaml_and_args(path, ["--mode=beta", "--level=3"])
        assert result.mode == "beta"
        assert result.level == 3, f"int literal must cast to int, got {result.level!r}"
    finally:
        os.unlink(path)


def test_real_config_literal_field_validated():
    """Pin the real seam: OfflineGRPOConfig.advantage_method is Literal-annotated and must reject
    a typo at parse time."""
    from src.configs.offline_grpo_config import OfflineGRPOConfig

    path = _write_yaml("output_dir: /tmp/h4_lit_real\nadvantage_method: banana\n")
    try:
        parser = H4ArgumentParser((OfflineGRPOConfig,))
        with pytest.raises(ValueError, match="advantage_method"):
            parser.parse_yaml_file(path, allow_extra_keys=False)
    finally:
        os.unlink(path)


# YAML 1.1 bool spellings on bool-typed fields fail loud instead of parsing as truthy strings


@dataclass
class BoolUnionConfig:
    maybe: bool | None = None
    mode: bool | str = False


def test_yaml_11_bool_spelling_on_bool_field_raises():
    """YAML 1.2 parses `no`/`off`/`yes`/`on` as STRINGS; on a bool field every non-empty string is
    truthy, so `packing: no` would silently ENABLE packing. Must fail loud, naming field and fix."""
    for spelling in ("no", "off", "yes", "on", "No", "OFF"):
        path = _write_yaml(f"use_liger_kernel: {spelling}\nname: y11\n")
        try:
            parser = H4ArgumentParser((SimpleConfig,))
            with pytest.raises(ValueError, match=rf"use_liger_kernel.*{spelling}"):
                parser.parse_yaml_file(path, allow_extra_keys=False)
        finally:
            os.unlink(path)


def test_quoted_bool_string_raises():
    """A quoted "false" is a string too — same silent inversion, same loud failure."""
    path = _write_yaml('bf16: "false"\nname: quoted\n')
    try:
        parser = H4ArgumentParser((SimpleConfig,))
        with pytest.raises(ValueError, match="bf16"):
            parser.parse_yaml_file(path, allow_extra_keys=False)
    finally:
        os.unlink(path)


def test_real_config_bool_string_rejected():
    """Pin the real seam: `gradient_checkpointing: no` on TrainingArguments must raise, not
    silently enable gradient checkpointing."""
    path = _write_yaml("output_dir: /tmp/h4_bool_no\ngradient_checkpointing: no\n")
    try:
        parser = H4ArgumentParser((TrainingArguments,))
        with pytest.raises(ValueError, match="gradient_checkpointing.*'no'"):
            parser.parse_yaml_file(path, allow_extra_keys=False)
    finally:
        os.unlink(path)


def test_optional_bool_keeps_real_booleans_and_null():
    """The rejection targets strings only: real YAML 1.2 booleans and null still parse."""
    path = _write_yaml("maybe: true\n")
    try:
        parser = H4ArgumentParser((BoolUnionConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert result.maybe is True
    finally:
        os.unlink(path)
    path = _write_yaml("maybe: null\n")
    try:
        parser = H4ArgumentParser((BoolUnionConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert result.maybe is None
    finally:
        os.unlink(path)
    path = _write_yaml("maybe: no\n")
    try:
        parser = H4ArgumentParser((BoolUnionConfig,))
        with pytest.raises(ValueError, match="maybe"):
            parser.parse_yaml_file(path, allow_extra_keys=False)
    finally:
        os.unlink(path)


def test_bool_union_with_string_member_admits_strings():
    """bool | str legitimately carries strings — no rejection."""
    path = _write_yaml("mode: balanced\n")
    try:
        parser = H4ArgumentParser((BoolUnionConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert result.mode == "balanced"
    finally:
        os.unlink(path)


# Un-castable CLI overrides fail loud instead of setattr-ing a raw string (F1-E)


@dataclass
class UncastableConfig:
    trackers: None | str | list[str] = None
    options: dict | None = None
    threshold: float | str = 1.0
    name: str = "default"


def test_cli_override_container_union_rejected():
    """A raw string VALUE in a str|list[str] field is later indexed/iterated as a list (char-wise);
    there is no confident cast, so a value override must fail loud. The none-spellings are the one
    exception: they clear an optional field instead."""
    path = _write_yaml("name: uncast\n")
    try:
        parser = H4ArgumentParser((UncastableConfig,))
        with pytest.raises(ValueError, match="trackers.*YAML|YAML.*trackers"):
            parser.parse_yaml_and_args(path, ["--trackers=wandb"])
        (parsed,) = parser.parse_yaml_and_args(path, ["--trackers=none"])
        assert parsed.trackers is None
    finally:
        os.unlink(path)


def test_cli_override_dict_field_rejected():
    path = _write_yaml("name: uncast_dict\n")
    try:
        parser = H4ArgumentParser((UncastableConfig,))
        with pytest.raises(ValueError, match="options"):
            parser.parse_yaml_and_args(path, ["--options=a:1"])
    finally:
        os.unlink(path)


def test_cli_override_report_to_rejected():
    """Pin the real seam: --report_to=none clears to None (never the raw string, which
    report_to[0]-style consumers would index char-wise), and a value override still fails loud."""
    path = _write_yaml("output_dir: /tmp/h4_report_to\n")
    try:
        parser = H4ArgumentParser((TrainingArguments,))
        (parsed,) = parser.parse_yaml_and_args(path, ["--report_to=none"])
        assert parsed.report_to is None
        with pytest.raises(ValueError, match="report_to"):
            parser.parse_yaml_and_args(path, ["--report_to=wandb"])
    finally:
        os.unlink(path)


def test_cli_override_scalar_union_keeps_current_cast():
    """float | str unions keep the existing numeric cast (consumers handle both)."""
    path = _write_yaml("name: scalar_union\n")
    try:
        parser = H4ArgumentParser((UncastableConfig,))
        (result,) = parser.parse_yaml_and_args(path, ["--threshold=0.25"])
        assert result.threshold == 0.25
    finally:
        os.unlink(path)


@dataclass
class ListDictConfig:
    servers: list[dict] | None = None
    name: str = "default"


def test_cli_override_list_of_dict_rejected():
    """A comma-split CLI string in a list[dict] field becomes list[str] and TypeErrors deep in its
    consumer (Ray setup); the override must fail at parse time and point at YAML."""
    path = _write_yaml("name: listdict\n")
    try:
        parser = H4ArgumentParser((ListDictConfig,))
        with pytest.raises(ValueError, match="servers.*YAML|YAML.*servers"):
            parser.parse_yaml_and_args(path, ["--servers=http://localhost:8000"])
    finally:
        os.unlink(path)


def test_cli_override_rollout_server_configs_rejected():
    """Pin the real seam: --rollout_server_configs=... must raise, not ship a list[str] into Ray."""
    from src.configs.async_training_config import AsyncTrainingConfig

    path = _write_yaml("{}\n")
    try:
        parser = H4ArgumentParser((AsyncTrainingConfig,))
        with pytest.raises(ValueError, match="rollout_server_configs"):
            parser.parse_yaml_and_args(path, ["--rollout_server_configs=http://localhost:8000"])
    finally:
        os.unlink(path)


# Optional[str] CLI overrides: None/null clear the field instead of setting the literal string


@dataclass
class OptionalStrConfig:
    note: str | None = "keep"
    label: str = "x"


def test_cli_none_clears_optional_str():
    """--note=None / --note=null must set real None (symmetric with YAML's null), not the string
    "None" a consumer then treats as a real value."""
    for spelling in ("None", "null"):
        path = _write_yaml("note: from-yaml\n")
        try:
            parser = H4ArgumentParser((OptionalStrConfig,))
            (result,) = parser.parse_yaml_and_args(path, [f"--note={spelling}"])
            assert result.note is None, f"--note={spelling} set {result.note!r} instead of None"
        finally:
            os.unlink(path)


def test_cli_none_on_plain_str_stays_literal():
    """A non-Optional str field cannot hold None; the value stays the literal string."""
    path = _write_yaml("note: plain_str\n")
    try:
        parser = H4ArgumentParser((OptionalStrConfig,))
        (result,) = parser.parse_yaml_and_args(path, ["--label=None"])
        assert result.label == "None"
    finally:
        os.unlink(path)


# --help renders with percent signs in field help


@dataclass
class PercentHelpConfig:
    shard_experts: bool = field(
        default=True,
        metadata={"help": "Frees DP-growing memory (gpt-oss-20b -19%/-37% at 2/8 GPU), throughput-neutral."},
    )
    ratio: float = field(default=0.5, metadata={"help": "Keeps 50% of rows."})
    speedup: bool = field(default=False, metadata={"help": "Throughput +20%% at 60%% of the memory."})


def test_help_renders_percent_in_field_help():
    """argparse expands help via ``help % params``; a bare percent in prose must not blow up --help."""
    parser = H4ArgumentParser((PercentHelpConfig,))
    text = parser.format_help()
    # Assertions stay inside one help line — argparse hard-wraps the rendered text.
    assert "-19%/-37%" in text, f"percent prose must render literally, got:\n{text}"
    assert "Keeps 50% of rows" in text
    # Help already escaped the argparse way (upstream transformers/TRL style) still renders one `%`.
    assert "Throughput +20% at 60% of the memory" in text, f"pre-escaped help must not double up:\n{text}"


def test_help_keeps_argparse_default_placeholder():
    """Escaping percents must not break argparse's own ``%(default)s`` expansion."""
    parser = H4ArgumentParser((PercentHelpConfig,))
    text = parser.format_help()
    assert "(default: 0.5)" in text, f"default expansion must survive, got:\n{text}"
    assert "%(default)s" not in text


def test_help_renders_for_real_script_dataclasses():
    """Pin the real seam: every training script parses DistributedArguments, whose help carries a
    bare percent — ``--help`` must render for the shipped dataclasses, not just synthetic ones."""
    from src.args.distributed_args import DistributedArguments
    from src.configs.offline_grpo_config import OfflineGRPOConfig

    parser = H4ArgumentParser((DistributedArguments, OfflineGRPOConfig))
    text = parser.format_help()
    assert "--fsdp_shard_ep1_experts" in text
    assert "--advantage_method" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
