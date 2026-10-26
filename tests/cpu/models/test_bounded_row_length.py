"""The one row-length rule: a bounded ``max_length`` wins, else a prompt + completion budget only when
both halves are bounded. A one-sided budget is no row cap: the unbounded half can be any length, unless
the caller takes a lone bounded half (``one_sided``)."""

from types import SimpleNamespace

import pytest

from halo.models.context_window import bounded_row_length

_UNSET = (None, 0, -1)


@pytest.mark.parametrize("max_length", [*_UNSET, 512])
@pytest.mark.parametrize("prompt", [*_UNSET, 100])
@pytest.mark.parametrize("completion", [*_UNSET, 28])
def test_one_source_follows_the_rule(max_length, prompt, completion):
    source = SimpleNamespace(max_length=max_length, max_prompt_length=prompt, max_completion_length=completion)
    if max_length == 512:
        expected = 512
    elif prompt == 100 and completion == 28:
        expected = 128
    else:
        expected = None
    assert bounded_row_length(source) == expected


@pytest.mark.parametrize(("prompt", "completion"), [(100, None), (None, 28), (100, 0), (-1, 28)])
def test_a_one_sided_budget_is_no_row_cap(prompt, completion):
    source = SimpleNamespace(max_length=None, max_prompt_length=prompt, max_completion_length=completion)
    assert bounded_row_length(source) is None


@pytest.mark.parametrize(
    ("prompt", "completion", "expected"),
    [(100, None, 100), (None, 28, 28), (100, 0, 100), (-1, 28, 28), (100, 28, 128), (None, None, None), (-1, 0, None)],
)
def test_one_sided_takes_a_lone_bounded_half_and_never_a_non_positive_knob(prompt, completion, expected):
    source = SimpleNamespace(max_length=-1, max_prompt_length=prompt, max_completion_length=completion)
    assert bounded_row_length(source, one_sided=True) == expected


def test_each_field_is_read_from_the_first_source_that_bounds_it():
    trainer_args = SimpleNamespace(max_length=0, max_completion_length=28)
    script_args = SimpleNamespace(max_length=None, max_prompt_length=100, max_completion_length=4096)
    assert bounded_row_length(trainer_args, script_args) == 128
    assert bounded_row_length(SimpleNamespace(), SimpleNamespace(max_length=64)) == 64
    assert bounded_row_length(SimpleNamespace(max_length=-1), SimpleNamespace(max_length=64)) == 64
    assert bounded_row_length(SimpleNamespace(), object()) is None
    assert bounded_row_length() is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
