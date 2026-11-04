#!/usr/bin/env python
"""HF's ``save_on_each_node`` names the same writers as the toolkit's saves. Forced on for per-node storage, or nodes
1..N cannot resume; forced off for shared, or two writers race one namespace."""

from types import SimpleNamespace

import pytest

import halo.trainers.training_args_overrides as base_mod
from halo.trainers.training_args_overrides import align_save_on_each_node


@pytest.mark.parametrize(
    ("shared", "nodes", "requested", "expected"),
    [
        (False, 2, False, True),
        (False, 1, False, True),
        (False, 2, True, True),
        (True, 2, True, False),
        (True, 2, False, False),
        (True, 1, True, True),
    ],
    ids=[
        "per-node-forced-on",
        "per-node-single-node-forced-on",
        "per-node-kept-on",
        "shared-multi-node-forced-off",
        "shared-kept-off",
        "shared-single-node-left-as-set",
    ],
)
def test_save_on_each_node_follows_the_output_filesystem(monkeypatch, shared, nodes, requested, expected):
    monkeypatch.setattr(base_mod, "is_output_shared_filesystem", lambda: shared)
    monkeypatch.setattr(base_mod, "get_num_nodes", lambda: nodes)
    args = SimpleNamespace(save_on_each_node=requested)

    align_save_on_each_node(args)

    assert args.save_on_each_node is expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
