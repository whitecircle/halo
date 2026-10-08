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
from datetime import datetime
from typing import Literal
from unittest import mock

import pytest
from transformers import TrainingArguments
from trl import DPOConfig, GRPOConfig, ModelConfig, RewardConfig, SFTConfig

from src.args.distributed_args import DistributedArguments
from src.args.dpo_args import DPOScriptArguments
from src.args.environmental_grpo_args import EnvironmentalGRPOScriptArguments
from src.args.reward_args import RMScriptArguments
from src.args.self_distill_args import SelfDistillationArguments
from src.args.sft_args import SFTScriptArguments
from src.configs.async_training_config import AsyncTrainingConfig
from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.configs.smpo_config import SmoothMarginPOConfig
from src.training.parser import (
    _TOOLKIT_DEFAULTS,
    H4ArgumentParser,
    _expand_output_dir,
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


def _parse_with_argv(parser, argv):
    old_argv = sys.argv
    sys.argv = argv
    try:
        return parser.parse()
    finally:
        sys.argv = old_argv


def _parse_yaml_launch(parser, yaml_body: str, *overrides: str):
    """``parse()`` as a ``<script> config.yaml [--overrides]`` launch does it."""
    path = _write_yaml(yaml_body)
    try:
        return _parse_with_argv(parser, ["prog", path, *overrides])
    finally:
        os.unlink(path)


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


# Toolkit defaults: a value the YAML did not write, applied before the dataclasses are built


def test_toolkit_defaults_applied():
    """use_liger_kernel and bf16 default to True when not explicitly set."""
    result = _parse_yaml_launch(H4ArgumentParser((SimpleConfig,)), "name: defaults_test\n")
    assert result.use_liger_kernel is True, "use_liger_kernel should be True"
    assert result.bf16 is True, "bf16 should be True"


def test_every_declared_toolkit_default_is_applied():
    """Derived from the registry, so a default added later cannot ship unexercised.

    Each entry is a knob the toolkit overrides upstream on — ``logging_nan_inf_filter: False``
    exists because upstream's ``True`` reads a device scalar per micro-batch AND replaces a NaN with
    the running average. A default that silently stops being applied restores the upstream behavior
    with nothing said.
    """
    opposites = {name: (not value if isinstance(value, bool) else None) for name, value in _TOOLKIT_DEFAULTS.items()}
    all_defaults = make_dataclass("AllDefaults", [(name, bool, field(default=v)) for name, v in opposites.items()])

    config = _parse_yaml_launch(H4ArgumentParser((all_defaults,)), "{}\n")

    for name, expected in _TOOLKIT_DEFAULTS.items():
        assert getattr(config, name) == expected, f"toolkit default {name}={expected} was not applied"


def test_toolkit_defaults_not_overridden():
    """Explicitly set fields are not overridden by toolkit defaults."""
    result = _parse_yaml_launch(
        H4ArgumentParser((SimpleConfig,)), "use_liger_kernel: false\nbf16: false\nname: explicit\n"
    )
    assert result.use_liger_kernel is False, "Should remain False when explicitly set"
    assert result.bf16 is False, "Should remain False when explicitly set"


def test_toolkit_defaults_partial_override():
    """Only the toolkit defaults the user left unset are applied."""
    result = _parse_yaml_launch(H4ArgumentParser((SimpleConfig,)), "use_liger_kernel: false\nname: partial\n")
    assert result.use_liger_kernel is False, "Explicitly set, should stay False"
    assert result.bf16 is True, "Not explicitly set, should become True"


def test_cli_override_counts_as_explicitly_set():
    """A CLI value, dashed spelling included, is explicit: the default must not reverse it."""
    result = _parse_yaml_launch(H4ArgumentParser((SimpleConfig,)), "name: cli\n", "--use-liger-kernel=false")
    assert result.use_liger_kernel is False, "toolkit default reversed --use-liger-kernel=false"
    assert result.bf16 is True


def test_toolkit_default_reaches_every_declarer_like_a_yaml_value():
    """A toolkit default is the value the YAML would carry, so it reaches every dataclass declaring
    the field, as a YAML key does."""

    @dataclass
    class First:
        bf16: bool = False

    @dataclass
    class Second:
        bf16: bool = False

    a, b = _parse_yaml_launch(H4ArgumentParser((First, Second)), "{}\n")
    assert (a.bf16, b.bf16) == (True, True)


def test_toolkit_defaults_skip_dataclasses_without_the_field():
    """A default lands only where a dataclass declares the field: injected into a parser none of whose
    dataclasses does, it would be refused as an unknown key."""

    @dataclass
    class NoFlags:
        x: int = 1

    @dataclass
    class HasFlag:
        use_liger_kernel: bool = False

    no_flags, has_flag = _parse_yaml_launch(H4ArgumentParser((NoFlags, HasFlag)), "{}\n")
    assert has_flag.use_liger_kernel is True
    assert not hasattr(no_flags, "use_liger_kernel")
    assert _parse_yaml_launch(H4ArgumentParser((NoFlags,)), "{}\n").x == 1


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
    """A field two dataclasses declare reaches BOTH, from the YAML and from a CLI override alike.

    The training scripts rely on it: ``pad_token``/``eos_token`` sit on the script args and on TRL's
    configs, and the tokenizer setup reads the script-args copy.
    """

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

    path = _write_yaml("shared_field: 42\nonly_a: hello\nonly_b: world\n")
    try:
        result_a, result_b = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert (result_a.shared_field, result_b.shared_field) == (42, 42)
        assert result_a.only_a == "hello"
        assert result_b.only_b == "world"

        result_a, result_b = parser.parse_yaml_and_args(path, ["--shared_field=7"])
        assert (result_a.shared_field, result_b.shared_field) == (7, 7)
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


# _expand_output_dir: strftime expansion


def _expanded(output_dir: str) -> str:
    return _expand_output_dir({"output_dir": output_dir, "name": "fmt"})["output_dir"]


def test_format_output_dir_with_strftime():
    """output_dir containing strftime codes is expanded."""
    expected = datetime.now().strftime("output/run-%Y-%m-%d")
    assert _expanded("output/run-%Y-%m-%d") == expected


def test_format_output_dir_no_strftime():
    """output_dir without strftime codes remains unchanged."""
    assert _expanded("output/plain-run") == "output/plain-run"


def test_format_output_dir_full_datetime():
    """output_dir with a full datetime pattern expands completely."""
    expanded = _expanded("output/sft-%Y-%m-%dT%H-%M-%S")
    assert "%" not in expanded, f"Unexpanded codes in: {expanded}"
    assert expanded.startswith("output/sft-")


def test_format_output_dir_percent_prose_survives():
    """Non-directive percent sequences are prose and must survive byte-identical — a whole-string
    strftime lets glibc expand `%-d` (no-padding day), turning `sft-100%-data` into `sft-10014ata`."""
    for raw in ("output/sft-100%-data", "output/50%_subset", "output/run-100%"):
        assert _expanded(raw) == raw, f"prose percent mangled: {raw!r} -> {_expanded(raw)!r}"


def test_format_output_dir_mixed_prose_and_directives():
    """Real directives expand while adjacent prose percents stay intact."""
    expected = "output/run-50%_subset-" + datetime.now().strftime("%Y%m%d")
    assert _expanded("output/run-50%_subset-%Y%m%d") == expected


def test_format_output_dir_percent_escape():
    """%% keeps its strftime escape meaning: a literal percent."""
    assert _expanded("output/100%%-data") == "output/100%-data"


def test_format_output_dir_leaves_other_values_alone():
    """Only output_dir expands, and only when it is a string."""
    values = {"name": "run-%Y", "output_dir": None}
    assert _expand_output_dir(values) == values
    assert _expand_output_dir({"name": "run-%Y"}) == {"name": "run-%Y"}


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


def test_list_cli_override_replaces_yaml_list():
    """A CLI list override must fully replace the YAML list, never merge with it."""
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


@pytest.mark.parametrize(
    "script_args_cls, trl_config_cls",
    [
        (SFTScriptArguments, SFTConfig),
        (SelfDistillationArguments, SFTConfig),
        (DPOScriptArguments, DPOConfig),
        (RMScriptArguments, RewardConfig),
    ],
    ids=["sft", "self_distill", "dpo", "reward"],
)
def test_flags_only_launch_hands_shared_field_to_every_declarer(script_args_cls, trl_config_cls, tmp_path):
    """A launch with no YAML, on a script whose tuple declares ``pad_token`` twice (its script args
    and TRL's config), parses and gives the value to both — the tokenizer setup reads the
    script-args copy, TRL its own."""
    parser = H4ArgumentParser((script_args_cls, trl_config_cls, ModelConfig, DistributedArguments))
    argv = [
        "prog",
        "--model_name_or_path=dummy/model",
        f"--output_dir={tmp_path}",
        "--pad_token=<|pad|>",
        "--use_cpu=true",
        "--bf16=false",
    ]
    with mock.patch("src.training.parser.install_log_tee"):
        script_args, trl_config, model_config, _ = _parse_with_argv(parser, argv)
    assert (script_args.pad_token, trl_config.pad_token) == ("<|pad|>", "<|pad|>")
    assert script_args.eos_token is None
    assert model_config.model_name_or_path == "dummy/model"
    assert trl_config.output_dir == str(tmp_path)


def test_flags_only_launch_leaves_unset_shared_field_at_each_default():
    """Only the flags given are handed out, as a YAML hands out only its keys: a shared field left
    off the command line keeps each declarer's own default, not the last declarer's for all."""

    @dataclass
    class OverlapA:
        shared_field: int = 0
        only_a: str = "a"

    @dataclass
    class OverlapB:
        shared_field: int = 99

    parser = H4ArgumentParser((OverlapA, OverlapB))
    result_a, result_b = _parse_with_argv(parser, ["prog", "--only_a=hello"])
    assert (result_a.only_a, result_a.shared_field, result_b.shared_field) == ("hello", 0, 99)

    result_a, result_b = _parse_with_argv(parser, ["prog", "--shared_field=7"])
    assert (result_a.shared_field, result_b.shared_field) == (7, 7)


def test_flags_only_launch_rejects_an_unknown_flag(capsys):
    parser = H4ArgumentParser((SimpleConfig,))
    with pytest.raises(SystemExit):
        _parse_with_argv(parser, ["prog", "--name=cli", "--no_such_field=1"])
    assert "unrecognized arguments: --no_such_field=1" in capsys.readouterr().err


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


# mixed_precision follows the final fp16/bf16 flags


def test_toolkit_bf16_default_syncs_mixed_precision():
    """TrainingArguments.__post_init__ derives mixed_precision from bf16, so the toolkit default must
    reach it, or the Accelerator autocasts 'no' while bf16 is True."""
    # use_cpu keeps TrainingArguments' bf16-support validation off GPU-less test machines.
    path = _write_yaml("output_dir: /tmp/h4_mp_default\nuse_cpu: true\n")
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
    path = _write_yaml("output_dir: /tmp/h4_mp_fp16\nfp16: true\nuse_cpu: true\n")
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
    """--fp16=true on top of bf16: true must meet __post_init__'s at-most-one check and fail loud
    instead of training with both flags set."""
    path = _write_yaml("output_dir: /tmp/h4_mp_conflict\nbf16: true\nuse_cpu: true\n")
    try:
        parser = H4ArgumentParser((TrainingArguments,))
        with pytest.raises(ValueError, match="At most one of fp16 and bf16"):
            _parse_with_argv(parser, ["prog", path, "--fp16=true"])
    finally:
        os.unlink(path)


# Literal-annotated fields are validated at parse time


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


# Un-castable CLI overrides fail loud instead of passing a raw string on


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


@pytest.mark.parametrize("spelling", ["none", "None", "null"])
def test_cli_override_report_to_none_silences_reporting(spelling):
    """Pin the real seam: every none-spelling reaches transformers as its own "none", which
    ``__post_init__`` turns into no integrations. None would come out as [None], which the Trainer
    refuses; a value override still fails loud."""
    path = _write_yaml("output_dir: /tmp/h4_report_to\nreport_to: wandb\n")
    try:
        parser = H4ArgumentParser((TrainingArguments,))
        (parsed,) = parser.parse_yaml_and_args(path, [f"--report_to={spelling}"])
        assert parsed.report_to == []
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
    parser = H4ArgumentParser((DistributedArguments, OfflineGRPOConfig))
    text = parser.format_help()
    assert "--fsdp_shard_ep1_experts" in text
    assert "--advantage_method" in text


# CLI overrides are construction values: every __post_init__ derives from the final configuration


@dataclass
class _DerivedStateConfig:
    base: int = 1

    def __post_init__(self):
        self.doubled = self.base * 2


@dataclass
class _GuardedConfig:
    count: int = 1

    def __post_init__(self):
        if self.count < 1:
            raise ValueError(f"count must be >= 1, got {self.count}")


def test_cli_override_reaches_post_init_derived_state():
    """State ``__post_init__`` derives from a field follows the CLI value, not the YAML one."""
    path = _write_yaml("base: 3\n")
    try:
        (result,) = H4ArgumentParser((_DerivedStateConfig,)).parse_yaml_and_args(path, ["--base=5"])
    finally:
        os.unlink(path)
    assert (result.base, result.doubled) == (5, 10), f"derived state kept the YAML value: {result.doubled}"


def test_cli_override_meets_every_post_init_guard():
    """A guard in any ``__post_init__`` holds a CLI value to the bound a YAML value meets, with no
    opt-in base class on the config."""
    path = _write_yaml("count: 2\n")
    try:
        with pytest.raises(ValueError, match="count must be >= 1"):
            H4ArgumentParser((_GuardedConfig,)).parse_yaml_and_args(path, ["--count=0"])
    finally:
        os.unlink(path)


@pytest.mark.parametrize(
    ("override", "steps_per_generation", "generation_batch_size"),
    [("--gradient_accumulation_steps=1", 1, 2), ("--per_device_train_batch_size=4", 8, 32)],
)
def test_cli_override_rederives_the_grpo_generation_geometry(
    override, steps_per_generation, generation_batch_size, tmp_path
):
    """TRL's ``GRPOConfig.__post_init__`` derives ``steps_per_generation`` from
    ``gradient_accumulation_steps`` and ``generation_batch_size`` from it and the per-device batch. Kept at
    the YAML's 8, ``--gradient_accumulation_steps=1`` would still generate once every 8 steps."""
    path = tmp_path / "grpo.yaml"
    path.write_text(
        f"output_dir: {tmp_path / 'out'}\nuse_cpu: true\nbf16: false\nnum_generations: 2\n"
        "per_device_train_batch_size: 2\ngradient_accumulation_steps: 8\n"
    )
    (config,) = H4ArgumentParser((GRPOConfig,)).parse_yaml_and_args(str(path), [override])
    assert config.world_size == 1
    assert (config.steps_per_generation, config.generation_batch_size) == (
        steps_per_generation,
        generation_batch_size,
    )


def test_cli_fp16_override_reaches_the_toolkit_bf16_derivation(tmp_path):
    """``SmoothMarginPOConfig.__post_init__`` derives an unset ``bf16`` from ``fp16``. Derived from the
    YAML alone it comes out true, and ``--fp16=true`` then forms the fp16+bf16 pair the run never asked
    for."""
    path = tmp_path / "smpo.yaml"
    path.write_text(f"output_dir: {tmp_path / 'out'}\nuse_cpu: true\n")
    with mock.patch("src.training.parser.install_log_tee"):
        config = _parse_with_argv(H4ArgumentParser((SmoothMarginPOConfig,)), ["prog", str(path), "--fp16=true"])
    assert (config.fp16, config.bf16, config.mixed_precision) == (True, False, "fp16")


def test_output_dir_is_rank0s_expansion_broadcast_once(tmp_path):
    """Every rank builds the configs from rank 0's expansion: a per-rank ``datetime.now()`` can straddle a
    second, and each node would then tee ``run.log`` into its own directory."""
    path = tmp_path / "out.yaml"
    path.write_text('output_dir: "run-%Y%m%d-%H%M%S"\n')
    broadcasts = []

    def rank0_value(value):
        broadcasts.append(value)
        return str(tmp_path / "rank0")

    with (
        mock.patch("src.training.parser.broadcast_from_rank0", side_effect=rank0_value),
        mock.patch("src.training.parser.install_log_tee"),
    ):
        config = _parse_with_argv(H4ArgumentParser((OutputDirConfig,)), ["prog", str(path)])
    assert config.output_dir == str(tmp_path / "rank0")
    assert len(broadcasts) == 1 and "%" not in broadcasts[0], broadcasts


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
