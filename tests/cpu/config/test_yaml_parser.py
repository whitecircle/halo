#!/usr/bin/env python
import os
import sys
import tempfile
from dataclasses import dataclass, field, make_dataclass
from datetime import datetime
from typing import Literal
from unittest import mock

import pytest
from transformers import TrainingArguments
from trl import DPOConfig, GRPOConfig, RewardConfig, SFTConfig

from halo.configs.distillation import SelfDistillScriptArguments
from halo.configs.grpo_environmental import AsyncRolloutConfig, EnvironmentalGRPOScriptArguments
from halo.configs.grpo_offline import OfflineGRPOConfig
from halo.configs.parallelism_arguments import ParallelismArguments
from halo.configs.parser import _TOOLKIT_DEFAULTS, HaloArgumentParser, expand_strftime_directives
from halo.configs.preference import DPOScriptArguments, SMPOConfig
from halo.configs.reward import RewardScriptArguments
from halo.configs.script_arguments import ModelArguments
from halo.configs.sft import SFTScriptArguments
from halo.entrypoints.script import parse_script_args


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


def _write_yaml(content: str, suffix: str = ".yaml") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def _parse_with_argv(parser, argv):
    """The dataclasses ``argv`` (``sys.argv`` form) parses into, the lone instance for one type."""
    output = parser.parse_dict(parser.load_values(argv[1:]))
    return output[0] if len(output) == 1 else output


def _parse_yaml_launch(parser, yaml_body: str, *overrides: str):
    """The parse a ``<script> config.yaml [--overrides]`` launch does."""
    path = _write_yaml(yaml_body)
    try:
        return _parse_with_argv(parser, ["prog", path, *overrides])
    finally:
        os.unlink(path)


def test_unknown_field_is_not_silently_stripped():
    path = _write_yaml("name: unknown_test\nuse_unsloth: true\n")
    try:
        parser = HaloArgumentParser((SimpleConfig,))
        with pytest.raises(ValueError, match="use_unsloth"):
            parser.parse_yaml_file(path, allow_extra_keys=False)
    finally:
        os.unlink(path)


def test_retired_ecosystem_spelling_is_not_migrated():
    """No rename table: TRL's retired ``max_seq_length`` reaches the strict check and raises."""
    path = _write_yaml("max_seq_length: 2048\nname: retired\n")
    try:
        parser = HaloArgumentParser((SimpleConfig,))
        with pytest.raises(ValueError, match="max_seq_length"):
            parser.parse_yaml_file(path, allow_extra_keys=False)
    finally:
        os.unlink(path)


def test_plain_config_parses_unchanged():
    path = _write_yaml("name: clean\n")
    try:
        parser = HaloArgumentParser((SimpleConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert result.name == "clean"
    finally:
        os.unlink(path)


def test_toolkit_defaults_applied():
    result = _parse_yaml_launch(HaloArgumentParser((SimpleConfig,)), "name: defaults_test\n")
    assert result.use_liger_kernel is True, "use_liger_kernel should be True"
    assert result.bf16 is True, "bf16 should be True"


def test_every_declared_toolkit_default_is_applied():
    """Derived from the registry. ``logging_nan_inf_filter: False`` exists because upstream's ``True`` reads a
    device scalar per micro-batch and replaces a NaN with the running average."""
    opposites = {
        name: (not value if isinstance(value, bool) else None) for name, (value, _) in _TOOLKIT_DEFAULTS.items()
    }
    all_defaults = make_dataclass("AllDefaults", [(name, bool, field(default=v)) for name, v in opposites.items()])

    config = _parse_yaml_launch(HaloArgumentParser((all_defaults,)), "{}\n")

    for name, (expected, _) in _TOOLKIT_DEFAULTS.items():
        assert getattr(config, name) == expected, f"toolkit default {name}={expected} was not applied"


def test_toolkit_defaults_not_overridden():
    result = _parse_yaml_launch(
        HaloArgumentParser((SimpleConfig,)), "use_liger_kernel: false\nbf16: false\nname: explicit\n"
    )
    assert result.use_liger_kernel is False, "Should remain False when explicitly set"
    assert result.bf16 is False, "Should remain False when explicitly set"


def test_toolkit_defaults_partial_override():
    result = _parse_yaml_launch(HaloArgumentParser((SimpleConfig,)), "use_liger_kernel: false\nname: partial\n")
    assert result.use_liger_kernel is False, "Explicitly set, should stay False"
    assert result.bf16 is True, "Not explicitly set, should become True"


def test_cli_override_counts_as_explicitly_set():
    """A CLI value, dashed spelling included, is explicit: the default must not reverse it."""
    result = _parse_yaml_launch(HaloArgumentParser((SimpleConfig,)), "name: cli\n", "--use-liger-kernel=false")
    assert result.use_liger_kernel is False, "toolkit default reversed --use-liger-kernel=false"
    assert result.bf16 is True


def test_toolkit_default_reaches_every_declarer_like_a_yaml_value():
    @dataclass
    class First:
        bf16: bool = False

    @dataclass
    class Second:
        bf16: bool = False

    a, b = _parse_yaml_launch(HaloArgumentParser((First, Second)), "{}\n")
    assert (a.bf16, b.bf16) == (True, True)


def test_toolkit_defaults_skip_dataclasses_without_the_field():
    """Injected into a parser none of whose dataclasses declares the field, it would be refused as an unknown key."""

    @dataclass
    class NoFlags:
        x: int = 1

    @dataclass
    class HasFlag:
        use_liger_kernel: bool = False

    no_flags, has_flag = _parse_yaml_launch(HaloArgumentParser((NoFlags, HasFlag)), "{}\n")
    assert has_flag.use_liger_kernel is True
    assert not hasattr(no_flags, "use_liger_kernel")
    assert _parse_yaml_launch(HaloArgumentParser((NoFlags,)), "{}\n").x == 1


def test_cli_override_int():
    path = _write_yaml("name: cli_int\n")
    try:
        parser = HaloArgumentParser((SimpleConfig,))
        (result,) = parser.parse_dict(parser.load_values([path, "--max_length=2048"]))
        assert result.max_length == 2048
        assert isinstance(result.max_length, int)
    finally:
        os.unlink(path)


def test_cli_override_float():
    path = _write_yaml("name: cli_float\n")
    try:
        parser = HaloArgumentParser((SimpleConfig,))
        (result,) = parser.parse_dict(parser.load_values([path, "--learning_rate=0.001"]))
        assert abs(result.learning_rate - 0.001) < 1e-10
        assert isinstance(result.learning_rate, float)
    finally:
        os.unlink(path)


def test_cli_override_bool_true():
    path = _write_yaml("name: cli_bool\n")
    try:
        parser = HaloArgumentParser((SimpleConfig,))
        (result,) = parser.parse_dict(parser.load_values([path, "--use_liger_kernel=True"]))
        assert result.use_liger_kernel is True
    finally:
        os.unlink(path)


def test_cli_override_bool_false():
    path = _write_yaml("use_liger_kernel: true\nname: cli_bool_f\n")
    try:
        parser = HaloArgumentParser((SimpleConfig,))
        (result,) = parser.parse_dict(parser.load_values([path, "--use_liger_kernel=false"]))
        assert result.use_liger_kernel is False
    finally:
        os.unlink(path)


def test_cli_override_list_str():
    path = _write_yaml("batch_size: 16\n")
    try:
        parser = HaloArgumentParser((ExtraConfig,))
        (result,) = parser.parse_dict(parser.load_values([path, "--tags=a,b,c"]))
        assert result.tags == ["a", "b", "c"]
    finally:
        os.unlink(path)


def test_cli_override_list_str_replaces_yaml_list():
    path = _write_yaml("tags:\n- from yaml\n")
    try:
        parser = HaloArgumentParser((ExtraConfig,))
        (result,) = parser.parse_dict(parser.load_values([path, "--tags=first,second"]))
        assert result.tags == ["first", "second"], f"--tags override did not replace the YAML list: {result.tags}"
    finally:
        os.unlink(path)


def test_cli_override_unknown_arg_raises():
    path = _write_yaml("name: ignore_unknown\n")
    try:
        parser = HaloArgumentParser((SimpleConfig,))
        with pytest.raises(ValueError, match="not_a_field"):
            parser.parse_dict(parser.load_values([path, "--not_a_field=123", "--max_length=64"]))
    finally:
        os.unlink(path)


def test_cli_override_shared_field_sets_on_both_dataclasses():
    """A field two dataclasses own (pad_token on SFTScriptArguments + SFTConfig) is set on both; token sync relies
    on this."""

    @dataclass
    class DupA:
        shared: int = 0

    @dataclass
    class DupB:
        shared: int = 0

    parser = HaloArgumentParser((DupA, DupB))
    path = _write_yaml("shared: 1\n")
    try:
        a, b = parser.parse_dict(parser.load_values([path, "--shared=7"]))
        assert a.shared == 7 and b.shared == 7
    finally:
        os.unlink(path)


def test_cli_override_genuinely_repeated_flag_raises():
    parser = HaloArgumentParser((SimpleConfig,))
    path = _write_yaml("max_length: 100\n")
    try:
        parser.parse_dict(parser.load_values([path, "--max_length=1", "--max_length=2"]))
        raise AssertionError("Should have raised ValueError for a repeated CLI flag")
    except ValueError as e:
        assert "duplicate" in str(e).lower()
    finally:
        os.unlink(path)


def test_cli_override_applies_after_yaml_value():
    path = _write_yaml("max_length: 100\nname: override_me\n")
    try:
        parser = HaloArgumentParser((SimpleConfig,))
        (result,) = parser.parse_dict(parser.load_values([path, "--max_length=999"]))
        assert result.max_length == 999
    finally:
        os.unlink(path)


def test_overlapping_fields_resolved():
    """``pad_token``/``eos_token`` sit on the script args and on TRL's configs, and the tokenizer setup reads the
    script-args copy."""

    @dataclass
    class OverlapA:
        shared_field: int = 0
        only_a: str = "a"

    @dataclass
    class OverlapB:
        shared_field: int = 99
        only_b: str = "b"

    parser = HaloArgumentParser((OverlapA, OverlapB))

    path = _write_yaml("shared_field: 42\nonly_a: hello\nonly_b: world\n")
    try:
        result_a, result_b = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert (result_a.shared_field, result_b.shared_field) == (42, 42)
        assert result_a.only_a == "hello"
        assert result_b.only_b == "world"

        result_a, result_b = parser.parse_dict(parser.load_values([path, "--shared_field=7"]))
        assert (result_a.shared_field, result_b.shared_field) == (7, 7)
    finally:
        os.unlink(path)


def test_empty_yaml():
    path = _write_yaml("")
    try:
        parser = HaloArgumentParser((SimpleConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert result.max_length == 512
        assert result.name == "default"
    finally:
        os.unlink(path)


def _expanded(output_dir: str) -> str:
    return expand_strftime_directives(output_dir)


def test_format_output_dir_with_strftime():
    expected = datetime.now().strftime("output/run-%Y-%m-%d")
    assert _expanded("output/run-%Y-%m-%d") == expected


def test_format_output_dir_no_strftime():
    assert _expanded("output/plain-run") == "output/plain-run"


def test_format_output_dir_full_datetime():
    expanded = _expanded("output/sft-%Y-%m-%dT%H-%M-%S")
    assert "%" not in expanded, f"Unexpanded codes in: {expanded}"
    assert expanded.startswith("output/sft-")


def test_format_output_dir_percent_prose_survives():
    """A whole-string strftime lets glibc expand ``%-d``, turning ``sft-100%-data`` into ``sft-10014ata``."""
    for raw in ("output/sft-100%-data", "output/50%_subset", "output/run-100%"):
        assert _expanded(raw) == raw, f"prose percent mangled: {raw!r} -> {_expanded(raw)!r}"


def test_format_output_dir_mixed_prose_and_directives():
    expected = "output/run-50%_subset-" + datetime.now().strftime("%Y%m%d")
    assert _expanded("output/run-50%_subset-%Y%m%d") == expected


def test_format_output_dir_percent_escape():
    assert _expanded("output/100%%-data") == "output/100%-data"


def test_parse_script_args_expands_output_dir_alone(tmp_path):
    """Only output_dir expands: a directive in any other string value reaches its field verbatim."""
    path = tmp_path / "out.yaml"
    path.write_text(f'output_dir: "{tmp_path}/run-%Y"\nname: "run-%Y"\n')
    with mock.patch.object(sys, "argv", ["prog", str(path)]), mock.patch("halo.entrypoints.script.install_log_tee"):
        config = parse_script_args(OutputDirConfig)
    assert config.output_dir == f"{tmp_path}/run-{datetime.now():%Y}"
    assert config.name == "run-%Y"


def test_scientific_notation_no_decimal_parsed_as_float():
    """PyYAML 1.1 returns ``3e-5`` as a string."""
    path = _write_yaml("learning_rate: 3e-5\nname: sci\n")
    try:
        parser = HaloArgumentParser((SimpleConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert isinstance(result.learning_rate, float), f"Expected float, got {type(result.learning_rate)}"
        assert abs(result.learning_rate - 3e-5) < 1e-10
    finally:
        os.unlink(path)


def test_scientific_notation_with_decimal_parsed_as_float():
    path = _write_yaml("learning_rate: 1.5e-5\nname: sci_dec\n")
    try:
        parser = HaloArgumentParser((SimpleConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert isinstance(result.learning_rate, float)
        assert abs(result.learning_rate - 1.5e-5) < 1e-10
    finally:
        os.unlink(path)


def test_list_cli_override_replaces_yaml_list():
    path = _write_yaml("context_fields:\n- old_field\n")
    try:
        parser = HaloArgumentParser((EnvironmentalGRPOScriptArguments,))
        (result,) = parser.parse_dict(parser.load_values([path, "--context_fields=first_field,second_field"]))
        assert result.context_fields == ["first_field", "second_field"], (
            f"--context_fields override did not replace the YAML list: {result.context_fields}"
        )
    finally:
        os.unlink(path)


def test_parse_accepts_yml_suffix():
    path = _write_yaml("max_length: 777\nname: yml\n", suffix=".yml")
    try:
        parser = HaloArgumentParser((SimpleConfig,))
        result = _parse_with_argv(parser, ["prog", path])
        assert result.max_length == 777
        assert result.name == "yml"
    finally:
        os.unlink(path)


def test_first_cli_flag_counts_as_explicit_without_yaml():
    """In the pure-CLI form the first flag is sys.argv[1]; it must still block the toolkit default."""
    parser = HaloArgumentParser((SimpleConfig,))
    result = _parse_with_argv(parser, ["prog", "--bf16", "false", "--name", "cli"])
    assert result.name == "cli"
    assert result.bf16 is False, "explicit first-position --bf16 false was clobbered by the toolkit default"


def test_dashed_cli_flag_counts_as_explicit():
    """The explicit-set scan normalizes the dashed spelling; recording it dashed lets the default reverse the user's
    false back to True."""
    parser = HaloArgumentParser((SimpleConfig,))
    result = _parse_with_argv(parser, ["prog", "--use-liger-kernel=false", "--name", "cli"])
    assert result.use_liger_kernel is False, "toolkit default reversed the dashed-spelled --use-liger-kernel=false"


@pytest.mark.parametrize(
    "script_args_cls, trl_config_cls",
    [
        (SFTScriptArguments, SFTConfig),
        (SelfDistillScriptArguments, SFTConfig),
        (DPOScriptArguments, DPOConfig),
        (RewardScriptArguments, RewardConfig),
    ],
    ids=["sft", "self_distill", "dpo", "reward"],
)
def test_flags_only_launch_hands_shared_field_to_every_declarer(script_args_cls, trl_config_cls, tmp_path):
    """The tokenizer setup reads the script-args copy, TRL its own."""
    parser = HaloArgumentParser((script_args_cls, trl_config_cls, ModelArguments, ParallelismArguments))
    argv = [
        "prog",
        "--model_name_or_path=dummy/model",
        f"--output_dir={tmp_path}",
        "--pad_token=<|pad|>",
        "--use_cpu=true",
        "--bf16=false",
    ]
    script_args, trl_config, model_config, _ = _parse_with_argv(parser, argv)
    assert (script_args.pad_token, trl_config.pad_token) == ("<|pad|>", "<|pad|>")
    assert script_args.eos_token is None
    assert model_config.model_name_or_path == "dummy/model"
    assert trl_config.output_dir == str(tmp_path)


def test_flags_only_launch_leaves_unset_shared_field_at_each_default():
    """Only the flags given are handed out, so a shared field left off keeps each declarer's own default."""

    @dataclass
    class OverlapA:
        shared_field: int = 0
        only_a: str = "a"

    @dataclass
    class OverlapB:
        shared_field: int = 99

    parser = HaloArgumentParser((OverlapA, OverlapB))
    result_a, result_b = _parse_with_argv(parser, ["prog", "--only_a=hello"])
    assert (result_a.only_a, result_a.shared_field, result_b.shared_field) == ("hello", 0, 99)

    result_a, result_b = _parse_with_argv(parser, ["prog", "--shared_field=7"])
    assert (result_a.shared_field, result_b.shared_field) == (7, 7)


def test_flags_only_launch_rejects_an_unknown_flag(capsys):
    parser = HaloArgumentParser((SimpleConfig,))
    with pytest.raises(SystemExit):
        _parse_with_argv(parser, ["prog", "--name=cli", "--no_such_field=1"])
    assert "unrecognized arguments: --no_such_field=1" in capsys.readouterr().err


def test_dashed_cli_override_with_yaml():
    path = _write_yaml("use_liger_kernel: true\nname: dash\n")
    try:
        parser = HaloArgumentParser((SimpleConfig,))
        (result,) = parser.parse_dict(parser.load_values([path, "--use-liger-kernel=false"]))
        assert result.use_liger_kernel is False
    finally:
        os.unlink(path)


def test_dashed_and_underscored_spellings_are_one_flag():
    path = _write_yaml("name: dash_dup\n")
    try:
        parser = HaloArgumentParser((SimpleConfig,))
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            parser.parse_dict(parser.load_values([path, "--use-liger-kernel=false", "--use_liger_kernel=true"]))
    finally:
        os.unlink(path)


def test_toolkit_bf16_default_syncs_mixed_precision():
    """``__post_init__`` derives mixed_precision from bf16, so the toolkit default must reach it, or the Accelerator
    autocasts 'no' while bf16 is True."""
    # use_cpu keeps TrainingArguments' bf16-support validation off GPU-less test machines.
    path = _write_yaml("output_dir: /tmp/h4_mp_default\nuse_cpu: true\n")
    try:
        parser = HaloArgumentParser((TrainingArguments,))
        result = _parse_with_argv(parser, ["prog", path])
        assert result.bf16 is True
        assert result.mixed_precision == "bf16", f"stale mixed_precision: {result.mixed_precision}"
    finally:
        os.unlink(path)


def test_toolkit_bf16_default_yields_to_explicit_fp16():
    """Applying both would form the fp16+bf16 pair TrainingArguments rejects."""
    path = _write_yaml("output_dir: /tmp/h4_mp_fp16\nfp16: true\nuse_cpu: true\n")
    try:
        parser = HaloArgumentParser((TrainingArguments,))
        result = _parse_with_argv(parser, ["prog", path])
        assert result.fp16 is True
        assert result.bf16 is False, "toolkit bf16 default must not apply on top of explicit fp16"
        assert result.mixed_precision == "fp16"
    finally:
        os.unlink(path)


def test_cli_bf16_false_override_rederives_mixed_precision():
    """Otherwise the model loads fp32 while the Accelerator still autocasts bf16."""
    path = _write_yaml("output_dir: /tmp/h4_mp_cli\nbf16: true\nuse_cpu: true\n")
    try:
        parser = HaloArgumentParser((TrainingArguments,))
        result = _parse_with_argv(parser, ["prog", path, "--bf16=false"])
        assert result.bf16 is False
        assert result.mixed_precision == "no", f"stale mixed_precision: {result.mixed_precision}"
    finally:
        os.unlink(path)


def test_cli_fp16_override_conflicting_with_yaml_bf16_raises():
    path = _write_yaml("output_dir: /tmp/h4_mp_conflict\nbf16: true\nuse_cpu: true\n")
    try:
        parser = HaloArgumentParser((TrainingArguments,))
        with pytest.raises(ValueError, match="At most one of fp16 and bf16"):
            _parse_with_argv(parser, ["prog", path, "--fp16=true"])
    finally:
        os.unlink(path)


@dataclass
class LiteralConfig:
    mode: Literal["alpha", "beta"] = "alpha"
    level: Literal[1, 2, 3] = 1
    opt_mode: Literal["x", "y"] | None = None
    hybrid: Literal["auto"] | str = "auto"
    name: str = "default"


def test_literal_field_invalid_yaml_value_raises():
    """parse_dict bypasses argparse choices, so an invalid value must fail at parse time."""
    path = _write_yaml("mode: banana\n")
    try:
        parser = HaloArgumentParser((LiteralConfig,))
        with pytest.raises(ValueError, match="banana.*mode.*alpha"):
            parser.parse_yaml_file(path, allow_extra_keys=False)
    finally:
        os.unlink(path)


def test_literal_field_valid_values_pass_through():
    path = _write_yaml("mode: beta\nlevel: 2\nopt_mode: y\n")
    try:
        parser = HaloArgumentParser((LiteralConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert result.mode == "beta"
        assert result.level == 2
        assert result.opt_mode == "y"
    finally:
        os.unlink(path)


def test_optional_literal_field_validates_and_allows_none():
    path = _write_yaml("opt_mode: banana\n")
    try:
        parser = HaloArgumentParser((LiteralConfig,))
        with pytest.raises(ValueError, match="opt_mode"):
            parser.parse_yaml_file(path, allow_extra_keys=False)
    finally:
        os.unlink(path)
    path = _write_yaml("opt_mode: null\n")
    try:
        parser = HaloArgumentParser((LiteralConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert result.opt_mode is None, "Optional[Literal] must keep accepting None"
    finally:
        os.unlink(path)


def test_mixed_union_literal_is_not_validated():
    """Literal['auto'] | str admits values outside the literal set, so there is no confident validation."""
    path = _write_yaml("hybrid: custom-value\n")
    try:
        parser = HaloArgumentParser((LiteralConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert result.hybrid == "custom-value"
    finally:
        os.unlink(path)


def test_literal_cli_override_validates_and_casts():
    """CLI overrides bypass argparse too: invalid values raise, valid str and int literals cast."""
    path = _write_yaml("name: lit_cli\n")
    try:
        parser = HaloArgumentParser((LiteralConfig,))
        with pytest.raises(ValueError, match="banana.*--mode|--mode.*banana"):
            parser.parse_dict(parser.load_values([path, "--mode=banana"]))
        (result,) = parser.parse_dict(parser.load_values([path, "--mode=beta", "--level=3"]))
        assert result.mode == "beta"
        assert result.level == 3, f"int literal must cast to int, got {result.level!r}"
    finally:
        os.unlink(path)


def test_real_config_literal_field_validated():
    path = _write_yaml("output_dir: /tmp/h4_lit_real\nadvantage_method: banana\n")
    try:
        parser = HaloArgumentParser((OfflineGRPOConfig,))
        with pytest.raises(ValueError, match="advantage_method"):
            parser.parse_yaml_file(path, allow_extra_keys=False)
    finally:
        os.unlink(path)


@dataclass
class BoolUnionConfig:
    maybe: bool | None = None
    mode: bool | str = False


def test_yaml_11_bool_spelling_on_bool_field_raises():
    """YAML 1.2 parses ``no``/``off``/``yes``/``on`` as strings, and every non-empty string is truthy, so ``packing:
    no`` would silently enable packing."""
    for spelling in ("no", "off", "yes", "on", "No", "OFF"):
        path = _write_yaml(f"use_liger_kernel: {spelling}\nname: y11\n")
        try:
            parser = HaloArgumentParser((SimpleConfig,))
            with pytest.raises(ValueError, match=rf"use_liger_kernel.*{spelling}"):
                parser.parse_yaml_file(path, allow_extra_keys=False)
        finally:
            os.unlink(path)


def test_quoted_bool_string_raises():
    path = _write_yaml('bf16: "false"\nname: quoted\n')
    try:
        parser = HaloArgumentParser((SimpleConfig,))
        with pytest.raises(ValueError, match="bf16"):
            parser.parse_yaml_file(path, allow_extra_keys=False)
    finally:
        os.unlink(path)


def test_real_config_bool_string_rejected():
    path = _write_yaml("output_dir: /tmp/h4_bool_no\ngradient_checkpointing: no\n")
    try:
        parser = HaloArgumentParser((TrainingArguments,))
        with pytest.raises(ValueError, match="gradient_checkpointing.*'no'"):
            parser.parse_yaml_file(path, allow_extra_keys=False)
    finally:
        os.unlink(path)


def test_optional_bool_keeps_real_booleans_and_null():
    """The rejection targets strings only: real YAML 1.2 booleans and null still parse."""
    path = _write_yaml("maybe: true\n")
    try:
        parser = HaloArgumentParser((BoolUnionConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert result.maybe is True
    finally:
        os.unlink(path)
    path = _write_yaml("maybe: null\n")
    try:
        parser = HaloArgumentParser((BoolUnionConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert result.maybe is None
    finally:
        os.unlink(path)
    path = _write_yaml("maybe: no\n")
    try:
        parser = HaloArgumentParser((BoolUnionConfig,))
        with pytest.raises(ValueError, match="maybe"):
            parser.parse_yaml_file(path, allow_extra_keys=False)
    finally:
        os.unlink(path)


def test_bool_union_with_string_member_admits_strings():
    path = _write_yaml("mode: balanced\n")
    try:
        parser = HaloArgumentParser((BoolUnionConfig,))
        (result,) = parser.parse_yaml_file(path, allow_extra_keys=False)
        assert result.mode == "balanced"
    finally:
        os.unlink(path)


@dataclass
class UncastableConfig:
    trackers: None | str | list[str] = None
    options: dict | None = None
    threshold: float | str = 1.0
    name: str = "default"


def test_cli_override_container_union_rejected():
    """A raw string in a str|list[str] field is later iterated char-wise, so a value override fails loud; the
    none-spellings clear an optional field."""
    path = _write_yaml("name: uncast\n")
    try:
        parser = HaloArgumentParser((UncastableConfig,))
        with pytest.raises(ValueError, match="trackers.*YAML|YAML.*trackers"):
            parser.parse_dict(parser.load_values([path, "--trackers=wandb"]))
        (parsed,) = parser.parse_dict(parser.load_values([path, "--trackers=none"]))
        assert parsed.trackers is None
    finally:
        os.unlink(path)


def test_cli_override_dict_field_rejected():
    path = _write_yaml("name: uncast_dict\n")
    try:
        parser = HaloArgumentParser((UncastableConfig,))
        with pytest.raises(ValueError, match="options"):
            parser.parse_dict(parser.load_values([path, "--options=a:1"]))
    finally:
        os.unlink(path)


@pytest.mark.parametrize("spelling", ["none", "None", "null"])
def test_cli_override_report_to_none_silences_reporting(spelling):
    """Every none-spelling reaches transformers as its own ``none``; None would come out as [None], which the
    Trainer refuses."""
    path = _write_yaml("output_dir: /tmp/h4_report_to\nbf16: false\nreport_to: wandb\n")
    try:
        parser = HaloArgumentParser((TrainingArguments,))
        (parsed,) = parser.parse_dict(parser.load_values([path, f"--report_to={spelling}"]))
        assert parsed.report_to == []
        with pytest.raises(ValueError, match="report_to"):
            parser.parse_dict(parser.load_values([path, "--report_to=wandb"]))
    finally:
        os.unlink(path)


def test_cli_override_scalar_union_keeps_current_cast():
    """float | str unions keep the existing numeric cast (consumers handle both)."""
    path = _write_yaml("name: scalar_union\n")
    try:
        parser = HaloArgumentParser((UncastableConfig,))
        (result,) = parser.parse_dict(parser.load_values([path, "--threshold=0.25"]))
        assert result.threshold == 0.25
    finally:
        os.unlink(path)


@dataclass
class ListDictConfig:
    servers: list[dict] | None = None
    name: str = "default"


def test_cli_override_list_of_dict_rejected():
    """A comma-split string in a list[dict] field becomes list[str] and TypeErrors deep in its consumer (Ray setup)."""
    path = _write_yaml("name: listdict\n")
    try:
        parser = HaloArgumentParser((ListDictConfig,))
        with pytest.raises(ValueError, match="servers.*YAML|YAML.*servers"):
            parser.parse_dict(parser.load_values([path, "--servers=http://localhost:8000"]))
    finally:
        os.unlink(path)


def test_cli_override_rollout_server_configs_rejected():
    path = _write_yaml("{}\n")
    try:
        parser = HaloArgumentParser((AsyncRolloutConfig,))
        with pytest.raises(ValueError, match="rollout_server_configs"):
            parser.parse_dict(parser.load_values([path, "--rollout_server_configs=http://localhost:8000"]))
    finally:
        os.unlink(path)


@dataclass
class OptionalStrConfig:
    note: str | None = "keep"
    label: str = "x"


def test_cli_none_clears_optional_str():
    """Symmetric with YAML's null; the string ``None`` would reach a consumer as a real value."""
    for spelling in ("None", "null"):
        path = _write_yaml("note: from-yaml\n")
        try:
            parser = HaloArgumentParser((OptionalStrConfig,))
            (result,) = parser.parse_dict(parser.load_values([path, f"--note={spelling}"]))
            assert result.note is None, f"--note={spelling} set {result.note!r} instead of None"
        finally:
            os.unlink(path)


def test_cli_none_on_plain_str_stays_literal():
    path = _write_yaml("note: plain_str\n")
    try:
        parser = HaloArgumentParser((OptionalStrConfig,))
        (result,) = parser.parse_dict(parser.load_values([path, "--label=None"]))
        assert result.label == "None"
    finally:
        os.unlink(path)


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
    parser = HaloArgumentParser((PercentHelpConfig,))
    text = parser.format_help()
    # Assertions stay inside one help line, since argparse hard-wraps the rendered text.
    assert "-19%/-37%" in text, f"percent prose must render literally, got:\n{text}"
    assert "Keeps 50% of rows" in text
    # Help already escaped the argparse way (upstream transformers/TRL style) still renders one `%`.
    assert "Throughput +20% at 60% of the memory" in text, f"pre-escaped help must not double up:\n{text}"


def test_help_keeps_argparse_default_placeholder():
    parser = HaloArgumentParser((PercentHelpConfig,))
    text = parser.format_help()
    assert "(default: 0.5)" in text, f"default expansion must survive, got:\n{text}"
    assert "%(default)s" not in text


def test_help_renders_for_real_script_dataclasses():
    """ParallelismArguments' help carries a bare percent."""
    parser = HaloArgumentParser((ParallelismArguments, OfflineGRPOConfig))
    text = parser.format_help()
    assert "--fsdp_shard_ep1_experts" in text
    assert "--advantage_method" in text


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
    path = _write_yaml("base: 3\n")
    try:
        parser = HaloArgumentParser((_DerivedStateConfig,))
        (result,) = parser.parse_dict(parser.load_values([path, "--base=5"]))
    finally:
        os.unlink(path)
    assert (result.base, result.doubled) == (5, 10), f"derived state kept the YAML value: {result.doubled}"


def test_cli_override_meets_every_post_init_guard():
    path = _write_yaml("count: 2\n")
    parser = HaloArgumentParser((_GuardedConfig,))
    try:
        with pytest.raises(ValueError, match="count must be >= 1"):
            parser.parse_dict(parser.load_values([path, "--count=0"]))
    finally:
        os.unlink(path)


@pytest.mark.parametrize(
    ("override", "steps_per_generation", "generation_batch_size"),
    [("--gradient_accumulation_steps=1", 1, 2), ("--per_device_train_batch_size=4", 8, 32)],
)
def test_cli_override_rederives_the_grpo_generation_geometry(
    override, steps_per_generation, generation_batch_size, tmp_path
):
    """TRL's ``GRPOConfig.__post_init__`` derives ``steps_per_generation`` from ``gradient_accumulation_steps``;
    kept at the YAML's 8, ``--gradient_accumulation_steps=1`` would still generate once every 8 steps."""
    path = tmp_path / "grpo.yaml"
    path.write_text(
        f"output_dir: {tmp_path / 'out'}\nuse_cpu: true\nbf16: false\nnum_generations: 2\n"
        "per_device_train_batch_size: 2\ngradient_accumulation_steps: 8\n"
    )
    parser = HaloArgumentParser((GRPOConfig,))
    (config,) = parser.parse_dict(parser.load_values([str(path), override]))
    assert config.world_size == 1
    assert (config.steps_per_generation, config.generation_batch_size) == (
        steps_per_generation,
        generation_batch_size,
    )


def test_cli_fp16_override_reaches_the_toolkit_bf16_derivation(tmp_path):
    """``SMPOConfig.__post_init__`` derives an unset ``bf16`` from ``fp16``; derived from the YAML alone,
    ``--fp16=true`` forms the fp16+bf16 pair."""
    path = tmp_path / "smpo.yaml"
    path.write_text(f"output_dir: {tmp_path / 'out'}\nuse_cpu: true\n")
    config = _parse_with_argv(HaloArgumentParser((SMPOConfig,)), ["prog", str(path), "--fp16=true"])
    assert (config.fp16, config.bf16, config.mixed_precision) == (True, False, "fp16")


def test_output_dir_is_rank0s_expansion_broadcast_once(tmp_path):
    """A per-rank ``datetime.now()`` can straddle a second, and each node would tee ``run.log`` into its own
    directory."""
    path = tmp_path / "out.yaml"
    path.write_text('output_dir: "run-%Y%m%d-%H%M%S"\n')
    broadcasts = []

    def rank0_value(value):
        broadcasts.append(value)
        return str(tmp_path / "rank0")

    with (
        mock.patch.object(sys, "argv", ["prog", str(path)]),
        mock.patch("halo.entrypoints.script.broadcast_from_rank0", side_effect=rank0_value),
        mock.patch("halo.entrypoints.script.install_log_tee") as tee,
    ):
        config = parse_script_args(OutputDirConfig)
    assert config.output_dir == str(tmp_path / "rank0")
    assert len(broadcasts) == 1 and "%" not in broadcasts[0], broadcasts
    tee.assert_called_once_with(str(tmp_path / "rank0"))


def test_parse_leaves_output_dir_verbatim(tmp_path):
    """The expansion is the script entry's, so a launcher-free caller reads the YAML's value as written."""
    path = tmp_path / "out.yaml"
    path.write_text('output_dir: "run-%Y"\n')
    config = _parse_with_argv(HaloArgumentParser((OutputDirConfig,)), ["prog", str(path)])
    assert config.output_dir == "run-%Y"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
