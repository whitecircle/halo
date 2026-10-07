#!/usr/bin/env python
"""HF's ``save_on_each_node`` must name the same writers as the toolkit's own saves.

The base Trainer writes ``trainer_state.json``, ``scheduler.pt`` and ``optimizer.pt`` and rotates old
checkpoints on ``args.should_save``; the toolkit's writers follow the output filesystem. On per-node
storage the flag is forced on, or no checkpoint resumes on nodes 1..N. On storage shared across nodes it is
forced off, or every node's local rank 0 writes and rotates the same files at once: two writers per
file in one filesystem namespace. A single node has one local rank 0, so the flag is left as set.

    python tests/cpu/checkpoint/test_save_on_each_node_alignment.py
"""

from types import SimpleNamespace

import pytest
from accelerate import PartialState

import src.trainers.mixins.base as base_mod
from src.trainers.mixins.base import align_save_on_each_node

PartialState()  # the mixin's accelerate logger requires an initialized state


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
