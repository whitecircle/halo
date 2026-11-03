"""The config validation seam: the guard hook every validated config runs, the declared-default comparison and the
argument refusals."""

import re
from dataclasses import dataclass, field

import pytest

from halo.config_validation import (
    ValidatedConfig,
    differs_from_default,
    is_finite_number,
    reject_non_default_args,
    reject_unsupported_args,
    require_bool,
    require_finite,
    require_int,
    require_text,
)
from halo.configs.distillation import SelfDistillScriptArguments


@dataclass(kw_only=True)
class _Knobs:
    path: str
    count: int = 50
    names: list[str] = field(default_factory=list)


@pytest.mark.parametrize(
    ("bounds", "value", "message"),
    [
        ({"gt": 0}, 0, "Cfg: knob must be > 0, got 0"),
        ({"ge": 0, "why": " (0 = off)"}, -1.5, "Cfg: knob must be >= 0 (0 = off), got -1.5"),
        ({"gt": 0, "le": 1}, 1.5, "Cfg: knob must be in (0, 1], got 1.5"),
        ({"ge": 0, "lt": 1}, 1, "Cfg: knob must be in [0, 1), got 1"),
        ({}, float("nan"), "Cfg: knob must be a finite number, got nan"),
        ({"gt": 0}, True, "Cfg: knob must be a finite number, got True"),
    ],
)
def test_require_finite_refuses_with_the_rendered_bound(bounds, value, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        require_finite("Cfg", **bounds, knob=value)


def test_require_finite_checks_every_value_finite_before_any_bound():
    with pytest.raises(ValueError, match="b must be a finite number"):
        require_finite("Cfg", gt=0, a=0, b=float("inf"))


@pytest.mark.parametrize(
    ("bounds", "value", "message"),
    [
        ({"ge": 1}, 0, "Cfg: knob must be an integer >= 1, got 0"),
        ({"ge": 1}, 2.0, "Cfg: knob must be an integer >= 1, got 2.0"),
        ({}, True, "Cfg: knob must be an integer, got True"),
    ],
)
def test_require_int_refuses_a_non_int_and_an_out_of_bound_int_alike(bounds, value, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        require_int("Cfg", **bounds, knob=value)


def test_in_bound_values_pass():
    require_finite("Cfg", gt=0, le=1, a=1, b=0.5)
    require_int("Cfg", ge=0, a=0, b=7)


def test_is_finite_number_admits_only_a_finite_int_or_float():
    assert all(is_finite_number(value) for value in (0, -2.5))
    assert not any(is_finite_number(value) for value in (True, float("nan"), float("-inf"), "1", None))


@pytest.mark.parametrize("value", ["no", 0, None])
def test_require_bool_refuses_anything_but_a_bool(value):
    with pytest.raises(ValueError, match=f"^{re.escape(f'Cfg: flag must be true or false, got {value!r}')}$"):
        require_bool("Cfg", flag=value)


@pytest.mark.parametrize("value", ["", "  ", 3, None])
def test_require_text_refuses_anything_but_a_non_blank_string(value):
    with pytest.raises(ValueError, match=f"^{re.escape(f'Cfg: name must be a non-blank string, got {value!r}')}$"):
        require_text("Cfg", name=value)


def test_type_guards_pass_valid_values():
    require_bool("Cfg", a=True, b=False)
    require_text("Cfg", a="x", b=" y ")


def test_validated_config_guards_before_the_next_post_init():
    """TrainingArguments derives state in its own ``__post_init__``; the guards must see the parsed values first."""
    calls = []

    @dataclass
    class _Upstream:
        def __post_init__(self):
            calls.append("upstream")

    @dataclass
    class _Guarded(ValidatedConfig, _Upstream):
        value: int = 1

        def _validate_fields(self):
            super()._validate_fields()
            calls.append("guard")

    _Guarded()
    assert calls == ["guard", "upstream"]


def test_differs_from_default_reads_literal_and_factory_defaults():
    assert not differs_from_default(_Knobs(path="p"), "count")
    assert differs_from_default(_Knobs(path="p", count=8), "count")
    assert not differs_from_default(_Knobs(path="p"), "names")
    assert differs_from_default(_Knobs(path="p", names=["a"]), "names")


@pytest.mark.parametrize("name", ["path", "cuont"])
def test_differs_from_default_refuses_a_name_with_no_default(name):
    """Reading a required field or a typo as unchanged would hide a request."""
    with pytest.raises(ValueError, match=f"'{name}' is not a defaulted field of _Knobs"):
        differs_from_default(_Knobs(path="p"), name)


def test_reject_unsupported_args_passes_when_nothing_set():
    reject_unsupported_args("Some script", tools_field=None, images_field="", train_flag=False)


def test_reject_unsupported_args_names_every_set_field():
    with pytest.raises(ValueError) as excinfo:
        reject_unsupported_args("Some script", tools_field="tools", images_field=None, train_flag=True)
    message = str(excinfo.value)
    assert "Some script" in message
    assert "tools_field" in message
    assert "train_flag" in message
    assert "images_field" not in message  # unset fields are not a request


def test_reject_non_default_args_passes_at_the_declared_defaults():
    """``num_eval_examples`` defaults to 50, which is truthy, so the truthiness gate could not guard it without
    rejecting every run."""
    reject_non_default_args("Some script", SelfDistillScriptArguments(), "generate_eval_examples", "num_eval_examples")


def test_reject_non_default_args_rejects_a_changed_truthy_default():
    args = SelfDistillScriptArguments()
    args.num_eval_examples = 8
    with pytest.raises(ValueError, match="num_eval_examples"):
        reject_non_default_args("Self-distillation", args, "generate_eval_examples", "num_eval_examples")


def test_reject_non_default_args_still_rejects_a_flipped_flag():
    args = SelfDistillScriptArguments()
    args.generate_eval_examples = True
    with pytest.raises(ValueError, match="generate_eval_examples"):
        reject_non_default_args("Self-distillation", args, "generate_eval_examples", "num_eval_examples")


def test_reject_non_default_args_refuses_a_field_the_dataclass_does_not_declare():
    """A typo'd name is a guard that can never fire."""
    with pytest.raises(ValueError, match="num_gen_exmaples"):
        reject_non_default_args("Some script", SelfDistillScriptArguments(), "num_gen_exmaples")


def test_reject_non_default_args_judges_a_factory_default_by_its_value():
    reject_non_default_args("Some script", _Knobs(path="p"), "names")
    with pytest.raises(ValueError, match="names"):
        reject_non_default_args("Some script", _Knobs(path="p", names=["a"]), "names")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
