#!/usr/bin/env python
"""``EPBufferBackend`` is the one vocabulary of DeepEP backends, since a name one side refuses fails only after a
model loaded on every rank."""

from typing import get_args

import pytest

from halo.distributed.expert_parallel.config import EPBufferBackend
from halo.distributed.expert_parallel.deepep import backends as backends_mod

# "auto" is the elastic default.
_SELECTED = {
    "auto": backends_mod._ElasticBackend,
    "elastic": backends_mod._ElasticBackend,
    "legacy": backends_mod._LegacyBackend,
}


class _Dispatcher:
    """A node-local ep2 dispatch group."""

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
