#!/usr/bin/env python
"""``EPBufferBackend`` is the one vocabulary of DeepEP buffer backends, and the factory builds every name in it.

``ParallelismConfig`` validates ``ep_buffer_backend`` against the Literal, and ``build_ep_backend`` builds
the transport the dispatcher claims buffers through. A name one accepts and the other refuses fails a run
only once a model has loaded on every rank, so every name must reach a backend.

    python tests/cpu/parallelism/test_ep_buffer_backend_vocabulary.py
"""

from typing import get_args

import pytest

from src.distributed.expert_parallel.config import EPBufferBackend
from src.distributed.expert_parallel.deepep import backends as backends_mod

# What each name selects; "auto" is the elastic default.
_SELECTED = {
    "auto": backends_mod._ElasticBackend,
    "elastic": backends_mod._ElasticBackend,
    "legacy": backends_mod._LegacyBackend,
}


class _Dispatcher:
    """The fields a backend reads at construction: a node-local ep2 dispatch group."""

    hidden_dim = 256
    ep_size = 2
    spans_domains = False


def test_the_vocabulary_is_the_three_backend_names():
    assert set(get_args(EPBufferBackend)) == {"auto", "elastic", "legacy"}


@pytest.mark.parametrize("name", get_args(EPBufferBackend))
def test_the_factory_builds_every_name_in_the_vocabulary(monkeypatch, name):
    monkeypatch.setattr(backends_mod, "get_local_world_size", lambda: 8)
    assert type(backends_mod.build_ep_backend(_Dispatcher(), name)) is _SELECTED[name]


def test_an_unknown_name_is_refused_with_the_vocabulary():
    with pytest.raises(ValueError, match="expected one of 'auto', 'elastic', 'legacy'"):
        backends_mod.build_ep_backend(_Dispatcher(), "nvshmem")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
