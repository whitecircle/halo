#!/usr/bin/env python
"""CPU test: every topology a shipped diagram depicts must be one ``ParallelismConfig`` accepts.

The EP diagrams are what a user copies a launch from, so a drawing of a shape the toolkit rejects at
config time costs a whole multi-node allocation before the error appears. The generator declares
its shapes in ``TOPOLOGIES`` and formats each subtitle from them; the literal is read out of the
source (via ``ast``, so nothing is drawn) and replayed through the real validator.

Run: ``python tests/cpu/parallelism/test_shipped_diagram_topology.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import ast
import os

import pytest

from src.distributed.parallelism_config import ParallelismConfig

_GENERATOR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "diagrams", "gen_ep_group_tree.py")
_FIGURES = ("ep_group_hierarchy", "ep_multi_node_layout")


def _source() -> str:
    with open(os.path.abspath(_GENERATOR), encoding="utf-8") as f:
        return f.read()


def _topologies() -> dict[str, dict[str, int | str]]:
    """The ``TOPOLOGIES`` literal, without importing the module (it renders both figures at import
    and would overwrite the tracked PNGs)."""
    for node in ast.walk(ast.parse(_source())):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "TOPOLOGIES" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("gen_ep_group_tree.py declares no TOPOLOGIES literal")


def test_generator_declares_a_topology_per_figure_and_titles_from_it():
    """Anti-vacuity: a subtitle typed by hand instead of formatted from the literal would let the
    drawing drift from the shape this test validates."""
    topologies = _topologies()
    assert set(topologies) == set(_FIGURES), sorted(topologies)
    source = _source()
    for figure in _FIGURES:
        assert f'TOPOLOGIES["{figure}"]' in source, f"{figure}'s subtitle is not formatted from TOPOLOGIES"


@pytest.mark.parametrize("figure", _FIGURES)
def test_depicted_topology_is_accepted_by_parallelism_config(figure):
    shape = _topologies()[figure]
    gpus_per_node = int(shape["gpus"])
    config = ParallelismConfig(
        ep_size=int(shape["ep"]),
        tp_size=int(shape["tp"]),
        ep_scope=str(shape["scope"]),
        world_size=gpus_per_node * int(shape["nodes"]),
        gpus_per_node=gpus_per_node,
        nvlink_domain_size=gpus_per_node,
    )
    assert config.data_parallel_size == int(shape["dp"]), f"{figure} claims a DP the config does not give"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
