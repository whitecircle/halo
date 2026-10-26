#!/usr/bin/env python
"""Every golden over the shape grid, pinned against today's code: the process groups and rank
coordinates each shape builds (``rank_layout``), and the ``new_group`` calls its ``EPConfig`` makes
(``ep_group_order``). A golden holds a digest per shape, and the full record of its literal shapes so
a failure reads as a diff.

Run: python tests/cpu/parallelism/test_grid_goldens.py
"""

import json

import pytest

from tests.common.goldens import ep_group_order, rank_layout
from tests.common.goldens.shape_grid import GridGolden, LayoutShape, layout_shapes, record_digest

GOLDENS = (rank_layout.GOLDEN, ep_group_order.GOLDEN)
COMMITTED = {golden.name: json.loads(golden.path.read_text(encoding="utf-8")) for golden in GOLDENS}


def _case(golden: GridGolden, shape: LayoutShape):
    return pytest.param(golden, shape, id=f"{golden.name}-{shape.key}")


@pytest.mark.parametrize("golden", GOLDENS, ids=lambda golden: golden.name)
def test_the_golden_covers_exactly_the_grid(golden):
    assert [shape.key for shape in layout_shapes()] == list(COMMITTED[golden.name]["digest"])


@pytest.mark.parametrize("golden", GOLDENS, ids=lambda golden: golden.name)
def test_the_literal_rows_are_the_literal_shapes(golden):
    assert [shape.key for shape in golden.literal_shapes] == list(COMMITTED[golden.name]["literal"])
    assert set(golden.literal_shapes) <= set(layout_shapes())


@pytest.mark.parametrize(
    ("golden", "shape"), [_case(golden, shape) for golden in GOLDENS for shape in layout_shapes()]
)
def test_the_record_matches_the_golden_digest(golden, shape):
    assert record_digest(golden.record(shape)) == COMMITTED[golden.name]["digest"][shape.key]


@pytest.mark.parametrize(
    ("golden", "shape"), [_case(golden, shape) for golden in GOLDENS for shape in golden.literal_shapes]
)
def test_the_record_matches_the_literal_row(golden, shape):
    assert golden.record(shape) == COMMITTED[golden.name]["literal"][shape.key]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
