#!/usr/bin/env python
"""Every topology a shipped diagram depicts is one ``ParallelismConfig`` accepts. The ``TOPOLOGIES`` literal is read
via ``ast``, since the module renders both figures at import."""

from __future__ import annotations

import ast

import pytest

from halo.distributed.parallelism_config import ParallelismConfig
from tests.common.repo import REPO_ROOT

_GENERATOR = REPO_ROOT / "scripts/diagrams/gen_ep_group_tree.py"
_FIGURES = ("ep_group_hierarchy", "ep_multi_node_layout")


def _source() -> str:
    return _GENERATOR.read_text(encoding="utf-8")


def _topologies() -> dict[str, dict[str, int | str]]:
    """Importing would overwrite the tracked PNGs."""
    for node in ast.walk(ast.parse(_source())):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "TOPOLOGIES" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("gen_ep_group_tree.py declares no TOPOLOGIES literal")


def test_generator_declares_a_topology_per_figure_and_titles_from_it():
    """A hand-typed subtitle would let the drawing drift from the validated shape."""
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
