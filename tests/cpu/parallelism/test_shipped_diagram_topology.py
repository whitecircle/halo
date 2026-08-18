#!/usr/bin/env python
"""CPU test: every topology a shipped diagram depicts must be one ``ParallelismConfig`` accepts.

The EP diagrams are what a user copies a launch from, so a drawing of a shape the toolkit rejects at
config time costs a whole multi-node allocation before the error appears. The subtitles are parsed out
of the generator source (via ``ast``, so nothing is drawn) and replayed through the real validator.

Run: ``python tests/cpu/parallelism/test_shipped_diagram_topology.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import ast
import os
import re

import pytest

from src.distributed.parallelism_config import ParallelismConfig

_GENERATOR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "diagrams", "gen_ep_group_tree.py")

# Diagram subtitles read "[<N> Nodes × ]<G> GPUs · EP = <e>[ (global)] · [TP = <t> · ]DP = <d>".
_SUBTITLE = re.compile(
    r"(?:(?P<nodes>\d+)\s*Nodes\s*×\s*)?(?P<gpus>\d+)\s*GPUs?\s*·\s*"
    r"EP\s*=\s*(?P<ep>\d+)(?P<scope>\s*\(global\))?\s*·\s*"
    r"(?:TP\s*=\s*(?P<tp>\d+)\s*·\s*)?DP\s*=\s*(?P<dp>\d+)"
)


def _subtitles():
    """Every topology subtitle string literal in the generator, without importing it (the module
    renders both figures at import and would overwrite the tracked PNGs)."""
    with open(os.path.abspath(_GENERATOR), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    literals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    return [(text, match) for text in literals if (match := _SUBTITLE.search(text))]


def test_generator_declares_the_topologies_this_test_checks():
    """Anti-vacuity: a subtitle reworded past the regex would make every assertion below vacuous."""
    found = _subtitles()
    assert len(found) == 2, [text for text, _ in found]


@pytest.mark.parametrize("index", [0, 1])
def test_depicted_topology_is_accepted_by_parallelism_config(index):
    text, match = _subtitles()[index]
    gpus_per_node = int(match["gpus"])
    world_size = gpus_per_node * int(match["nodes"] or 1)
    tp_size = int(match["tp"] or 1)

    config = ParallelismConfig(
        ep_size=int(match["ep"]),
        tp_size=tp_size,
        ep_scope="global" if match["scope"] else "node",
        world_size=world_size,
        gpus_per_node=gpus_per_node,
        nvlink_domain_size=gpus_per_node,
    )

    assert config.data_parallel_size == int(match["dp"]), f"{text!r} claims a DP the config does not give"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
