"""The declared-default comparison and the argument refusals: a parsed knob a script cannot honor raises
instead of being dropped, read off its truthiness or against the field's declared default.

python tests/cpu/test_config_validation.py
"""

from dataclasses import dataclass, field

import pytest

from src.config_validation import differs_from_default, reject_non_default_args, reject_unsupported_args
from src.configs.distillation import SelfDistillScriptArguments


@dataclass(kw_only=True)
class _Knobs:
    path: str
    count: int = 50
    names: list[str] = field(default_factory=list)


# differs_from_default: whether a config set a field, read off its declared default


def test_differs_from_default_reads_literal_and_factory_defaults():
    assert not differs_from_default(_Knobs(path="p"), "count")
    assert differs_from_default(_Knobs(path="p", count=8), "count")
    assert not differs_from_default(_Knobs(path="p"), "names")
    assert differs_from_default(_Knobs(path="p", names=["a"]), "names")


@pytest.mark.parametrize("name", ["path", "cuont"])
def test_differs_from_default_refuses_a_name_with_no_default(name):
    """A required field and a typo have no default to compare against; reading either as unchanged
    would hide a request."""
    with pytest.raises(ValueError, match=f"'{name}' is not a defaulted field of _Knobs"):
        differs_from_default(_Knobs(path="p"), name)


# reject_unsupported_args: the shared "would be silently ignored" gate


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


# reject_non_default_args: the same gate for knobs whose own default is truthy


def test_reject_non_default_args_passes_at_the_declared_defaults():
    """The whole point: ``num_eval_examples`` defaults to 50 — truthy — so the truthiness gate could
    never guard it without rejecting every run."""
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
    """A typo'd name is a guard that can never fire — the silent no-op this family exists to stop."""
    with pytest.raises(ValueError, match="num_gen_exmaples"):
        reject_non_default_args("Some script", SelfDistillScriptArguments(), "num_gen_exmaples")


def test_reject_non_default_args_judges_a_factory_default_by_its_value():
    """A ``default_factory`` field's default is the value the factory builds, not the factory."""
    reject_non_default_args("Some script", _Knobs(path="p"), "names")
    with pytest.raises(ValueError, match="names"):
        reject_non_default_args("Some script", _Knobs(path="p", names=["a"]), "names")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
