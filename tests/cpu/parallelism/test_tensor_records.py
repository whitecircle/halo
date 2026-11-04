"""The per-case records the EP and CP harnesses compare, and the verdict pieces every harness shares."""

import pytest
import torch

from tests.common.equivalence.revisions import (
    OUTPUT_ROOT_MARK,
    RAISED,
    compare_outputs,
    split_accepted,
    unseen_controls,
)
from tests.common.equivalence.tensor_records import (
    bitwise_equal,
    case_of,
    compare_records,
    load_record,
    raised_records,
    record_case,
    record_deltas,
)


def _record(tensors: dict, orders: dict | None = None) -> dict:
    return {"tensors": tensors, "orders": orders or {}}


def test_equal_values_with_different_bits_differ():
    assert not bitwise_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not bitwise_equal(torch.tensor([1.0]), torch.tensor([1.0], dtype=torch.float64))
    assert not bitwise_equal(torch.zeros(2, 3), torch.zeros(3, 2))


def test_the_same_nan_matches_itself():
    nan = torch.tensor([float("nan"), 1.0])
    assert bitwise_equal(nan, nan.clone())


def test_every_kind_of_difference_is_named():
    base = _record(
        {"loss": torch.tensor(1.0), "ids": torch.tensor([1, 2]), "gone": torch.ones(1)}, {"keys": ["a", "b"]}
    )
    head = _record(
        {"loss": torch.tensor(1.5), "ids": torch.tensor([1, 3]), "new": torch.ones(1)}, {"keys": ["a", "c"]}
    )

    assert record_deltas(base, head) == [
        "orders/keys[1]: 'b' != 'c' (2 vs 2 entries)",
        "tensors/gone: only in base",
        "tensors/ids: 1 of 2 values differ",
        "tensors/loss: bits differ, max |delta| 5.000e-01",
        "tensors/new: only in head",
    ]
    assert record_deltas(base, base) == []


def test_a_raising_case_is_recorded_as_what_it_raised(tmp_path):
    def refuse():
        raise ValueError(f"refused under {tmp_path}")

    record_case(tmp_path, "case", 0, refuse)
    record_case(tmp_path, "other", 0, lambda: ({"x": torch.ones(1)}, {}))

    assert raised_records(tmp_path) == {
        "case/rank0": {"type": "ValueError", "message": f"refused under {OUTPUT_ROOT_MARK}"}
    }
    raised, recorded = (load_record(tmp_path / case / "rank0.pt") for case in ("case", "other"))
    assert record_deltas(raised, recorded)[0].startswith(f"{RAISED}:")


def test_a_record_lands_under_its_case_and_rank(tmp_path):
    record_case(tmp_path, "cpu/qwen3_moe/defaults", 3, lambda: ({}, {}))

    assert (tmp_path / "cpu" / "qwen3_moe" / "defaults" / "rank3.pt").is_file()
    assert case_of("cpu/qwen3_moe/defaults/rank3") == "cpu/qwen3_moe/defaults"


def test_trees_differ_by_record_and_by_content(tmp_path):
    for root, value in (("base", 1.0), ("head", 2.0)):
        record_case(tmp_path / root, "shared", 0, lambda value=value: ({"x": torch.tensor(value)}, {}))
    record_case(tmp_path / "base", "lost", 0, lambda: ({}, {}))

    assert compare_records(tmp_path / "base", tmp_path / "head") == [
        {"path": "lost/rank0", "delta": "only in base"},
        {"path": "shared/rank0", "delta": "differs", "entries": ["tensors/x: bits differ, max |delta| 1.000e+00"]},
    ]


def test_described_outputs_differ_by_leaf():
    base = {"kept": {"a": 1}, "changed": {"a": 1, "b": {"c": [1]}}, "gone": {}}
    head = {"kept": {"a": 1}, "changed": {"a": 2, "b": {"c": [1], "d": 0}}, "new": {}}

    assert compare_outputs(base, head) == [
        {"path": "changed", "delta": "differs", "entries": ["a: 1 != 2", "b/d: only in head"]},
        {"path": "gone", "delta": "only in base"},
        {"path": "new", "delta": "only in head"},
    ]


def test_an_accepted_case_covers_every_path_under_it():
    deltas = [{"path": path, "delta": "differs"} for path in ("cpu/a/x/rank0", "cpu/a/x/rank1", "cpu/ab/x/rank0")]

    failing, accepted = split_accepted(deltas, ["cpu/a"])

    assert (failing, accepted) == (deltas[2:], deltas[:2])
    assert split_accepted(deltas, ["cpu/ab/x/rank0"]) == (deltas[:2], deltas[2:])


def test_a_control_that_raised_or_matched_goes_unseen():
    control_deltas = [{"path": "moved", "delta": "differs"}, {"path": "raised", "delta": "differs"}]

    unseen = unseen_controls(control_deltas, ["moved", "raised", "unmoved"], ["raised"])

    assert unseen == ["raised", "unmoved"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
