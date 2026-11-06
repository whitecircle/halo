#!/usr/bin/env python
"""The elastic DeepEP arena is built and grown with headroom, held to the limits the dispatch check enforces, so a
rising high-water mark rebuilds it a few times per run instead of on every step that sets a new maximum."""

from types import SimpleNamespace

import pytest

from halo.distributed.expert_parallel.deepep import backends as backends_mod
from halo.distributed.expert_parallel.deepep import capacity as capacity_mod
from halo.distributed.expert_parallel.deepep.backends import _ElasticBackend
from halo.distributed.expert_parallel.deepep.capacity import (
    ARENA_HEADROOM,
    EP_CAPACITY_ALIGN,
    aligned_dispatch_ceiling,
    arena_capacity,
    capacity_scope,
    max_dispatch_capacity,
    reject_oversized_dispatch,
)

# Qwen3.5-35B-A3B wire shape; the logged grow sequence of a batch-32 run on it.
HIDDEN, TOP_K = 2048, 8
LOGGED_GROWS = (36096, 37120, 37632)
# Hidden 8192 x top-8 puts the 32-bit index ceiling at 32767 tokens/rank.
WIDE = {"num_topk": 8, "padded_hidden": 8192}


def _capacity(needed: int, *, spans_domains: bool = False, **wire) -> int:
    return arena_capacity(
        needed, spans_domains=spans_domains, **(wire or {"num_topk": TOP_K, "padded_hidden": HIDDEN})
    )


@pytest.mark.parametrize("needed", [256, 1280, 4608, 9216, 36096])
def test_headroom_is_the_factor_rounded_up_to_the_alignment(needed):
    got = _capacity(needed)
    assert got % EP_CAPACITY_ALIGN == 0
    assert needed * ARENA_HEADROOM <= got < needed * ARENA_HEADROOM + EP_CAPACITY_ALIGN
    assert got > needed


@pytest.mark.parametrize("needed", [256, 8192, 30000, 32512])
def test_the_wire_index_ceiling_clamps_the_headroom(needed):
    got = _capacity(needed, **WIDE)
    assert got >= needed
    assert got <= max(needed, max_dispatch_capacity(spans_domains=False, **WIDE))
    reject_oversized_dispatch(got, spans_domains=False, **WIDE)
    if needed == 32512:
        assert got == needed, "an arena already at the aligned ceiling takes no headroom"
    if needed == 30000:
        assert got == 32512, "the headroom stops at the largest aligned capacity under the index limit"


def test_the_gin_ceiling_clamps_across_domains_only(monkeypatch):
    monkeypatch.setattr(capacity_mod, "GIN_MAX_TOKENS_PER_RANK", 8192)
    assert _capacity(8192, spans_domains=True) == 8192
    assert _capacity(6144, spans_domains=True) == 8192
    assert _capacity(4096, spans_domains=True) == 6144
    assert _capacity(8192, spans_domains=False) == 12288
    monkeypatch.setattr(capacity_mod, "GIN_MAX_TOKENS_PER_RANK", 0)
    assert _capacity(8192, spans_domains=True) == 12288, "0 disables the ceiling, as for the dispatch check"


@pytest.mark.parametrize("spans_domains", [False, True])
@pytest.mark.parametrize("needed", range(256, 40000, 1792))
def test_headroom_never_exceeds_what_the_dispatch_check_allows(needed, spans_domains):
    wire = {"num_topk": 4, "padded_hidden": 3072, "spans_domains": spans_domains}
    limit = max_dispatch_capacity(**wire)
    if needed > limit:
        with pytest.raises(ValueError):
            reject_oversized_dispatch(needed, **wire)
        return
    reject_oversized_dispatch(arena_capacity(needed, **wire), **wire)


@pytest.mark.parametrize("spans_domains", [False, True])
def test_the_aligned_ceiling_is_the_limit_rounded_down_to_the_block(spans_domains):
    wire = {"num_topk": 8, "padded_hidden": 2048, "spans_domains": spans_domains}
    limit = max_dispatch_capacity(**wire)
    ceiling = aligned_dispatch_ceiling(**wire)
    assert ceiling % EP_CAPACITY_ALIGN == 0 and limit - EP_CAPACITY_ALIGN < ceiling <= limit
    reject_oversized_dispatch(ceiling, **wire)


def test_every_rank_of_a_group_builds_the_same_size():
    sizes = {_capacity(LOGGED_GROWS[0], spans_domains=False) for _ in range(8)}
    assert len(sizes) == 1


class _Buffer:
    built: list = []

    def __init__(self, group, *, num_max_tokens_per_rank: int, **kwargs):
        self.capacity = num_max_tokens_per_rank
        _Buffer.built.append(num_max_tokens_per_rank)

    def get_theoretical_num_sms(self, num_experts, num_topk):
        return 24


@pytest.fixture
def backend(monkeypatch):
    _Buffer.built = []
    monkeypatch.setattr(backends_mod, "deep_ep", lambda: SimpleNamespace(ElasticBuffer=_Buffer))
    monkeypatch.setattr(backends_mod._Arena, "_registry", {})
    monkeypatch.setattr(backends_mod, "EP_CAPACITY_DEDUP", True)
    monkeypatch.setattr(backends_mod, "has_process_group", lambda: False)
    monkeypatch.setattr(backends_mod, "_deepep_env_configured", True)
    dispatcher = SimpleNamespace(
        hidden_dim=HIDDEN,
        dispatch_group=object(),
        spans_domains=False,
        ep_size=8,
        num_experts=256,
        ep_config=SimpleNamespace(expert_tp_size=1),
    )
    return _ElasticBackend(dispatcher)


def _replay(backend, needs) -> None:
    """One forward per entry: the first layer's all-reduce result, then the arena."""
    for needed in needs:
        capacity_scope.bump()
        capacity_scope.store(backend._d.dispatch_group, id(backend._d.dispatch_group), TOP_K, needed)
        backend.ensure(needed, TOP_K, deterministic=False)


def test_the_logged_grow_sequence_builds_the_arena_once(backend):
    _replay(backend, LOGGED_GROWS)
    assert _Buffer.built == [
        arena_capacity(LOGGED_GROWS[0], num_topk=TOP_K, padded_hidden=HIDDEN, spans_domains=False)
    ]
    assert max(LOGGED_GROWS) <= backend._arena.capacity


def test_a_batch_past_the_headroom_grows_it_once_more(backend):
    _replay(backend, (LOGGED_GROWS[0], 60000))
    assert len(_Buffer.built) == 2 and _Buffer.built[1] >= 60000


def test_exact_fit_regrows_on_the_logged_sequence(backend, monkeypatch):
    """Negative control: without headroom the same sequence rebuilds the arena at every new high-water mark."""
    monkeypatch.setattr(backends_mod, "arena_capacity", lambda needed, **wire: needed)
    _replay(backend, LOGGED_GROWS)
    assert _Buffer.built == list(LOGGED_GROWS)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
