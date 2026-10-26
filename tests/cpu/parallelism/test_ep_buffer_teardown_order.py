#!/usr/bin/env python
"""Which DeepEP buffers a rank frees, and in what order, is decided by state every rank of the group shares.

Freeing a buffer is a collective: ``buffer.destroy()`` spins on that buffer's device-side NVLink barrier, so
the ranks of a group must free the same buffers in the same order, or each spins in the barrier of a
different buffer until the device timeout aborts the job (``DeepEP NVLink barrier timeout``, then
``cudaErrorLaunchFailure``). Which dispatchers are still alive is not shared: a dispatcher dropped with its
model gives up its claim when this rank's garbage collector reclaims it. Each test impersonates one rank
whose buffers were built in the same order as on its peers, differing only in what its collector has
reclaimed:

- teardown frees every buffer in build order, whatever was reclaimed;
- the unclaimed sweep frees a buffer only once no rank of that buffer's own group claims it;
- destroying the live layers (``cleanup_ep``), or moving one to a buffer of a new determinism mode, frees
  the old buffer only once no rank of the group claims it.

The last test runs real ranks on gloo, two EP groups of two, where only one rank of each group has collected
a dropped model, and gives each buffer a destroy that is a collective over its group: the ranks must free
the same buffers in the same order, and a rank that skips a collective its peers enter times its group out.

DeepEP is replaced by a recorder, so the dispatchers run their real sizing and arena logic on a CPU box.
``tests/gpu/parallelism/ep/test_ep_buffer_gc_safety.py`` runs the sweep on real DeepEP buffers.

Run: python tests/cpu/parallelism/test_ep_buffer_teardown_order.py
"""

import datetime
import gc
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from src.distributed.expert_parallel.deepep import backends as backends_mod
from src.distributed.expert_parallel.deepep import dispatcher as dispatcher_mod
from src.distributed.expert_parallel.deepep.capacity import capacity_scope
from src.runtime.consensus import RankConsensus
from tests.common.gloo import new_groups, run_gloo_ranks

# One EP group per model, each serving its own buffer, built in this order on every rank.
MODELS = ("first", "second")
LAYERS_PER_MODEL = 2
NUM_EXPERTS, HIDDEN, TOP_K, TOKENS = 8, 256, 2, 64
# What a rank's collector may have reclaimed by teardown: nothing, or the layers of one dropped model.
RECLAIMED = {"none": (), "first": ("first",), "second": ("second",)}
# The gloo run: two EP groups of two ranks, each model on its own buffer of the group through its hidden size.
GLOO_WORLD_SIZE = 4
GLOO_EP_GROUPS = [[0, 1], [2, 3]]
GLOO_HIDDEN = {"first": 256, "second": 512}
GLOO_PG_TIMEOUT = datetime.timedelta(seconds=60)


class _Group:
    """An EP group, named so the destroy log reads which buffer was freed."""

    def __init__(self, name: str):
        self.name = name


class _RecordingBuffer:
    """Stands in for DeepEP's ElasticBuffer: logs the group of each buffer it destroys."""

    destroyed: list = []

    def __init__(self, group: _Group, **kwargs):
        self.group = group

    def get_theoretical_num_sms(self, num_experts: int, num_topk: int) -> int:
        return 24

    def destroy(self) -> None:
        _RecordingBuffer.destroyed.append(self.group.name)


class _GatheringBuffer(_RecordingBuffer):
    """A buffer on a real group whose destroy is a collective over it, as DeepEP's is: it logs which buffer
    every rank of the group was destroying in that collective."""

    def __init__(self, group: dist.ProcessGroup, *, hidden: int, **kwargs):
        self.group = group
        self.model = next(name for name, size in GLOO_HIDDEN.items() if size == hidden)

    def destroy(self) -> None:
        models = [None] * dist.get_world_size(self.group)
        dist.all_gather_object(models, self.model, group=self.group)
        _RecordingBuffer.destroyed.append(models)


def _record_deep_ep(patcher: pytest.MonkeyPatch, buffer: type = _RecordingBuffer, collectives: bool = False) -> None:
    """DeepEP replaced by ``buffer`` and registries of the test's own; without ``collectives`` the capacity
    all-reduce is a no-op too."""
    _RecordingBuffer.destroyed = []
    patcher.setattr(backends_mod, "deep_ep", lambda: SimpleNamespace(ElasticBuffer=buffer))
    patcher.setattr(dispatcher_mod, "deep_ep_available", lambda: True)
    patcher.setattr(backends_mod._Arena, "_registry", {})
    # Every test relies on the layers of a group sharing one buffer.
    patcher.setattr(backends_mod, "EP_CAPACITY_DEDUP", True)
    capacity_scope.clear()
    patcher.setattr(dispatcher_mod, "_live_dispatchers", weakref.WeakValueDictionary())
    real_tensor = torch.tensor
    patcher.setattr(backends_mod.torch, "tensor", lambda data, **kw: real_tensor(data, **{**kw, "device": "cpu"}))
    if not collectives:
        patcher.setattr(backends_mod.dist, "all_reduce", lambda *a, **kw: None)
    patcher.setenv("EP_DISABLE_GIN", "1")


@pytest.fixture(autouse=True)
def recorded_deep_ep(monkeypatch):
    """Every in-process test runs on recorded DeepEP with no collectives."""
    _record_deep_ep(monkeypatch)


def _layer(group, hidden: int = HIDDEN) -> dispatcher_mod.DeepEPDispatcher:
    """A layer's dispatcher on ``group`` that has dispatched once, claiming that group's buffer."""
    ep_config = SimpleNamespace(
        expert_tp_size=1, dispatch_ep_group=group, ep_size=2, spans_domains=False, ep_buffer_backend="elastic"
    )
    dispatcher = dispatcher_mod.DeepEPDispatcher(ep_config, NUM_EXPERTS, hidden)
    dispatcher._ensure_buffer(TOKENS, TOP_K)
    return dispatcher


def _rank_holding(groups: dict[str, _Group], reclaimed: tuple[str, ...]) -> dict[str, list]:
    """Every model's layers built in :data:`MODELS` order, then the layers of ``reclaimed`` dropped and
    collected: the live layers of a rank whose collector has reclaimed those models."""
    layers = {name: [_layer(groups[name]) for _ in range(LAYERS_PER_MODEL)] for name in MODELS}
    for name in reclaimed:
        del layers[name]
    gc.collect()
    return layers


def _registered() -> list[str]:
    return [arena.group.name for arena in backends_mod._Arena._registry.values()]


def _consensus(peer_unclaimed: dict[str, bool], asked: list[tuple[str, bool]]):
    """``rank_consensus`` against a peer whose claims on each group's buffer are ``peer_unclaimed``,
    logging each question as ``(group, this rank's answer)``."""

    def consensus(local_ok: bool, group=None) -> RankConsensus:
        asked.append((group.name, local_ok))
        peer_ok = peer_unclaimed[group.name]
        return RankConsensus(local_ok and peer_ok, local_ok or peer_ok)

    return consensus


@pytest.mark.parametrize("reclaimed", RECLAIMED.values(), ids=RECLAIMED.keys())
def test_teardown_frees_every_buffer_in_build_order(reclaimed):
    live = _rank_holding({name: _Group(name) for name in MODELS}, reclaimed)
    assert _registered() == list(MODELS), "a reclaimed layer freed its buffer"

    dispatcher_mod.destroy_all_dispatchers()

    assert _RecordingBuffer.destroyed == list(MODELS)
    assert _registered() == []
    assert not any(layer._live for layers in live.values() for layer in layers)


def test_the_sweep_frees_only_what_no_rank_of_its_group_claims(monkeypatch):
    """This rank has reclaimed every layer; a peer on the first group still claims that group's buffer."""
    asked = []
    monkeypatch.setattr(backends_mod, "rank_consensus", _consensus({"first": False, "second": True}, asked))
    _rank_holding({name: _Group(name) for name in MODELS}, reclaimed=MODELS)

    assert backends_mod.free_unclaimed_ep_buffers() == 1
    assert asked == [("first", True), ("second", True)], "each buffer is agreed on by its own group, in build order"
    assert _RecordingBuffer.destroyed == ["second"]
    assert _registered() == ["first"]

    dispatcher_mod.destroy_all_dispatchers()

    assert _RecordingBuffer.destroyed == ["second", "first"]


def test_destroying_the_live_layers_frees_their_buffer_only_once_no_rank_claims_it(monkeypatch):
    """A dropped model and the live one share a buffer on one group; this rank has reclaimed the dropped
    model, a peer has not, so destroying the live layers leaves the buffer to teardown."""
    asked = []
    monkeypatch.setattr(backends_mod, "rank_consensus", _consensus({"shared": False}, asked))
    group = _Group("shared")
    dropped = [_layer(group) for _ in range(LAYERS_PER_MODEL)]
    live = [_layer(group) for _ in range(LAYERS_PER_MODEL)]
    del dropped
    gc.collect()

    for layer in live:
        layer.destroy()

    assert asked == [("shared", False), ("shared", True)], "each destroy asks the group, with this rank's claims"
    assert _RecordingBuffer.destroyed == []
    assert _registered() == ["shared"]

    dispatcher_mod.destroy_all_dispatchers()

    assert _RecordingBuffer.destroyed == ["shared"]


def test_a_mode_change_frees_the_old_buffer_only_once_no_rank_claims_it(monkeypatch):
    """The live layer moves to a deterministic buffer while a peer still claims the old one for a model it
    has not collected: the old buffer stays registered until teardown."""
    asked = []
    monkeypatch.setattr(backends_mod, "rank_consensus", _consensus({"shared": False}, asked))
    group = _Group("shared")
    dropped = [_layer(group) for _ in range(LAYERS_PER_MODEL)]
    live = _layer(group)
    del dropped
    gc.collect()

    monkeypatch.setattr(dispatcher_mod, "_deterministic_dispatch", lambda: True)
    live._ensure_buffer(TOKENS, TOP_K)

    assert asked == [("shared", True)]
    assert _RecordingBuffer.destroyed == []
    assert _registered() == ["shared", "shared"], "the old buffer and the deterministic one"

    dispatcher_mod.destroy_all_dispatchers()

    assert _RecordingBuffer.destroyed == ["shared", "shared"]


def _teardown_rank(rank: int, out_dir: str) -> None:
    """Build both models, drop the first, destroy the second's layers, sweep, tear down. Only the second
    rank of each group collects the dropped model; the first still holds its claims at teardown."""
    group = new_groups(rank, GLOO_EP_GROUPS, GLOO_PG_TIMEOUT)
    with pytest.MonkeyPatch.context() as patcher:
        _record_deep_ep(patcher, buffer=_GatheringBuffer, collectives=True)
        gc.disable()
        try:
            models = {
                name: [_layer(group, hidden) for _ in range(LAYERS_PER_MODEL)] for name, hidden in GLOO_HIDDEN.items()
            }
            dropped = models.pop("first")
            # A cycle, so the model goes only when this rank's collector runs.
            dropped.append(dropped)
            del dropped
            if dist.get_rank(group) == 1:
                gc.collect()
            for layer in models["second"]:
                layer.destroy()
            backends_mod.free_unclaimed_ep_buffers()
            dispatcher_mod.destroy_all_dispatchers()
        finally:
            gc.enable()
        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(_RecordingBuffer.destroyed))


def test_ranks_that_collected_differently_free_the_same_buffers_in_the_same_order(tmp_path):
    run_gloo_ranks(_teardown_rank, GLOO_WORLD_SIZE, str(tmp_path), pg_timeout=GLOO_PG_TIMEOUT)

    # The second model's buffer goes as its last live layer is destroyed; the first's, which one rank of the
    # group still claims, is left by the sweep to teardown.
    expected = [["second", "second"], ["first", "first"]]
    for rank in range(GLOO_WORLD_SIZE):
        assert json.loads((tmp_path / f"rank{rank}.json").read_text()) == expected, f"rank {rank}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
