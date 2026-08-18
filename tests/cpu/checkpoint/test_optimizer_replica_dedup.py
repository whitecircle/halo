#!/usr/bin/env python
"""CPU tests for the expert-replica dedup of the per-rank optimizer shards.

Under multi-group EP the EP groups are DP replicas: the FSDP-IGNORED expert params (the whole EP
module, router included) hold identical bytes on every rank of an ``expert_replica_group``, and
their moments are identical too — the grads are averaged over that group and the optimizer is
deterministic. Only the group's lowest rank keeps them in its shard; its peers strip them and read
them back from that shard on resume.

Covered here, on a fake 4-rank ``ep2`` x 2-replica topology (replica sets ``{0,2}`` and ``{1,3}``):
the writer election, which FQNs count as replicated, the save-side strip, the non-shared-filesystem
refusal to dedup, the resume merge restoring every rank to identical state, the loud raise when the
writer's shard is not reachable, and the fingerprint field that refuses a drifted replica layout.

    python tests/cpu/checkpoint/test_optimizer_replica_dedup.py
"""

import os

import pytest
import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

import src.distributed.checkpoint.optimizer as optimizer_mod
from src.distributed.checkpoint.context import CheckpointLoadContext
from src.distributed.checkpoint.fingerprint import OptimizerStateFingerprint
from src.distributed.checkpoint.optimizer import OptimizerShardStore, expert_replica_writer
from src.distributed.expert_parallel.layers.qwen3 import EPQwen3MoELayer
from tests.common.parallelism import make_parallelism_config

# ep2 over 4 ranks = 2 EP groups; the same ep_rank in each group is a replica of the other.
REPLICA_SETS = {0: [0, 2], 1: [1, 3], 2: [0, 2], 3: [1, 3]}
REPLICATED_KEYS = {"mlp.experts", "mlp.router.weight"}
SHARDED_KEY = "fc.weight"


class _StubEpConfig:
    def __init__(self, replica_ranks, *, experts_fsdp_managed=False):
        self.expert_replica_ranks = replica_ranks
        self.needs_expert_grad_sync = len(replica_ranks) > 1
        self.experts_fsdp_managed = experts_fsdp_managed


def _ep_model(replica_ranks, *, experts_fsdp_managed=False) -> nn.Module:
    """A model whose ``mlp`` is a real EP wrapper instance (class only, as the seam reads nothing
    else) carrying the FSDP-ignored expert + router params, beside one FSDP-sharded param."""
    model = nn.Module()
    model.fc = nn.Linear(4, 4, bias=False)
    layer = object.__new__(EPQwen3MoELayer)
    nn.Module.__init__(layer)
    layer.ep_config = _StubEpConfig(replica_ranks, experts_fsdp_managed=experts_fsdp_managed)
    layer.experts = nn.Parameter(torch.zeros(2, 4))
    layer.router = nn.Linear(4, 2, bias=False)
    model.mlp = layer
    return model


def _osd(scale: float) -> dict:
    """An optimizer state dict shaped like ``get_optimizer_state_dict``'s: state by FQN + groups."""
    keys = [SHARDED_KEY, *sorted(REPLICATED_KEYS)]
    return {
        "state": {key: {"exp_avg": torch.full((2,), scale)} for key in keys},
        "param_groups": [{"params": keys, "lr": 0.1}],
    }


def _store(model, rank, monkeypatch, *, shared_fs=True) -> OptimizerShardStore:
    monkeypatch.setattr(optimizer_mod, "get_global_rank", lambda: rank)
    monkeypatch.setattr(optimizer_mod, "is_output_shared_filesystem", lambda: shared_fs)
    monkeypatch.setattr(optimizer_mod, "is_global_main_process", lambda: rank == 0)
    ctx = CheckpointLoadContext(
        model=model,
        optimizer=None,
        lr_scheduler=None,
        parallelism_config=None,
        is_pp_mode=False,
        is_cp_mode=False,
        is_tp_mode=False,
        has_ep_layers=True,
        fsdp_wrapped=True,
        tp_rank=0,
        tp_size=1,
        super_load_from_checkpoint=lambda *a, **k: None,
        super_load_optimizer_and_scheduler=lambda *a, **k: None,
    )
    return OptimizerShardStore(ctx)


@pytest.mark.parametrize("rank", sorted(REPLICA_SETS))
def test_writer_is_the_lowest_rank_of_the_replica_group(rank, monkeypatch):
    monkeypatch.setattr(optimizer_mod, "get_global_rank", lambda: rank)
    writer, keys = expert_replica_writer(_ep_model(REPLICA_SETS[rank]))
    assert writer == min(REPLICA_SETS[rank])
    assert keys == REPLICATED_KEYS, "only the FSDP-ignored EP-module params are replica-duplicated"


def test_writer_election_does_not_depend_on_group_member_order(monkeypatch):
    """The group list is built per rank; a different member order must not move the writer, or two
    replicas would each strip their state expecting the other to have kept it."""
    monkeypatch.setattr(optimizer_mod, "get_global_rank", lambda: 2)
    assert expert_replica_writer(_ep_model([2, 0]))[0] == 0


@pytest.mark.parametrize(
    "replica_ranks,experts_fsdp_managed",
    [([3], False), ([0, 2], True)],
    ids=["single-ep-group", "ep1-experts-fsdp-sharded"],
)
def test_nothing_is_replicated_without_multi_group_ep(replica_ranks, experts_fsdp_managed, monkeypatch):
    """One EP group replicates nothing, and FSDP-managed (ep1) experts are already reduce-scattered
    — deduping either would drop state no other shard holds."""
    monkeypatch.setattr(optimizer_mod, "get_global_rank", lambda: 3)
    assert expert_replica_writer(_ep_model(replica_ranks, experts_fsdp_managed=experts_fsdp_managed)) == (
        3,
        frozenset(),
    )


def test_dense_model_is_untouched(monkeypatch):
    monkeypatch.setattr(optimizer_mod, "get_global_rank", lambda: 1)
    assert expert_replica_writer(nn.Linear(4, 4)) == (1, frozenset())


@pytest.mark.parametrize("rank", sorted(REPLICA_SETS))
def test_only_the_replica_writer_keeps_the_replicated_state(rank, monkeypatch):
    store = _store(_ep_model(REPLICA_SETS[rank]), rank, monkeypatch)
    osd = _osd(1.0)
    store._drop_replicated_state(osd)

    kept = set(osd["state"])
    if rank == min(REPLICA_SETS[rank]):
        assert kept == {SHARDED_KEY} | REPLICATED_KEYS
    else:
        assert kept == {SHARDED_KEY}, "a follower still writes the bytes the writer already holds"
    assert osd["param_groups"][0]["params"] == [SHARDED_KEY, *sorted(REPLICATED_KEYS)], (
        "param_groups must stay whole — the resume's missing-FQN gate reads that key space"
    )


def test_non_shared_filesystem_keeps_every_copy(monkeypatch, caplog):
    """The follower could not read the writer's shard off another node, so the duplication stands —
    loudly, never as a shard set that resumes on one node only."""
    store = _store(_ep_model(REPLICA_SETS[1]), 1, monkeypatch, shared_fs=False)
    osd = _osd(1.0)
    with caplog.at_level("WARNING"):
        store._drop_replicated_state(osd)
    assert set(osd["state"]) == {SHARDED_KEY} | REPLICATED_KEYS
    monkeypatch.setattr(optimizer_mod, "is_global_main_process", lambda: True)
    with caplog.at_level("WARNING"):
        store._drop_replicated_state(_osd(1.0))
    assert "shared output directory" in caplog.text


def _write_deduped_checkpoint(tmp_path, monkeypatch) -> dict[int, dict]:
    """Save all four ranks' shards the way ``save`` does, and return what each rank wrote."""
    written = {}
    for rank in sorted(REPLICA_SETS):
        store = _store(_ep_model(REPLICA_SETS[rank]), rank, monkeypatch)
        osd = _osd(1.0)
        store._drop_replicated_state(osd)
        torch.save(osd, os.path.join(tmp_path, f"optimizer_shard_{rank:05d}.pt"))
        written[rank] = osd
    return written


def test_every_rank_restores_identical_state(tmp_path, monkeypatch):
    """The point of the dedup: after the merge each rank holds exactly what a per-rank save gave it."""
    written = _write_deduped_checkpoint(str(tmp_path), monkeypatch)
    assert sum(len(osd["state"]) for osd in written.values()) == 4 + len(REPLICATED_KEYS) * 2, (
        "premise: 2 replica sets keep one copy each, not four"
    )

    restored = {}
    for rank in sorted(REPLICA_SETS):
        store = _store(_ep_model(REPLICA_SETS[rank]), rank, monkeypatch)
        osd = torch.load(os.path.join(tmp_path, f"optimizer_shard_{rank:05d}.pt"), weights_only=False)
        assert store._merge_replicated_state(str(tmp_path), osd) is True
        restored[rank] = osd

    expected = _osd(1.0)["state"]
    for rank, osd in restored.items():
        assert set(osd["state"]) == set(expected), rank
        for key, state in osd["state"].items():
            assert torch.equal(state["exp_avg"], expected[key]["exp_avg"]), (rank, key)


def test_writer_reads_nothing_extra(tmp_path, monkeypatch):
    """A rank that IS the writer must not go looking for another shard — there is none to find."""
    _write_deduped_checkpoint(str(tmp_path), monkeypatch)
    os.remove(os.path.join(str(tmp_path), "optimizer_shard_00002.pt"))
    store = _store(_ep_model(REPLICA_SETS[0]), 0, monkeypatch)
    osd = torch.load(os.path.join(str(tmp_path), "optimizer_shard_00000.pt"), weights_only=False)
    assert store._merge_replicated_state(str(tmp_path), osd) is True


def test_pre_dedup_checkpoint_needs_no_writer_shard(tmp_path, monkeypatch):
    """A shard set written before the dedup (every rank self-contained) must resume as it always
    did, without reaching for a peer's file."""
    store = _store(_ep_model(REPLICA_SETS[2]), 2, monkeypatch)
    assert store._merge_replicated_state(str(tmp_path), _osd(1.0)) is True


def test_a_stateless_replicated_param_is_not_read_as_a_dedup(tmp_path, monkeypatch):
    """A tracked param that never received a gradient is absent from EVERY shard, so "some
    replicated key is missing" would send a self-contained shard chasing a writer's file that a
    non-shared filesystem does not have. The test is all-or-nothing for exactly this reason."""
    store = _store(_ep_model(REPLICA_SETS[2]), 2, monkeypatch)
    osd = _osd(1.0)
    del osd["state"]["mlp.router.weight"]
    assert store._merge_replicated_state(str(tmp_path), osd) is True
    assert "mlp.experts" in osd["state"], "the surviving replicated state must not be disturbed"


def test_pre_first_step_checkpoint_needs_no_writer_shard(tmp_path, monkeypatch):
    """An empty optimizer state is a checkpoint saved before the first step, not a deduplicated one."""
    store = _store(_ep_model(REPLICA_SETS[1]), 1, monkeypatch)
    assert store._merge_replicated_state(str(tmp_path), {"state": {}, "param_groups": []}) is True


def test_missing_writer_shard_fails_the_read_verdict(tmp_path, monkeypatch, caplog):
    """A follower missing its replicated state with no writer shard in reach would otherwise restore
    a partial optimizer and log success. The verdict rides the shard-readability consensus that
    follows, so the whole world warm-restarts (raises under PP) instead of this rank alone."""
    _write_deduped_checkpoint(str(tmp_path), monkeypatch)
    os.remove(os.path.join(str(tmp_path), "optimizer_shard_00000.pt"))
    store = _store(_ep_model(REPLICA_SETS[2]), 2, monkeypatch)
    osd = torch.load(os.path.join(str(tmp_path), "optimizer_shard_00002.pt"), weights_only=False)
    with caplog.at_level("WARNING"):
        assert store._merge_replicated_state(str(tmp_path), osd) is False
    assert "optimizer_shard_00000.pt is not in" in caplog.text


def test_a_writer_shard_spelling_the_experts_differently_fails_the_read(tmp_path, monkeypatch, caplog):
    """The merge's own gate: restoring NONE of the FQNs this shard was deduplicated against means the
    writer's state is keyed differently from this run's expert parameter names.

    The missing-FQN gate downstream cannot see it — it subtracts every ``param_groups`` FQN out, and
    the dedup keeps ``param_groups`` whole — so without this verdict the run resumes every expert
    param at zero moments and step 0 while logging '✓ Restored per-rank optimizer state'.
    """
    drifted = _osd(1.0)
    drifted["state"] = {f"{key}_gmm": value for key, value in drifted["state"].items()}
    torch.save(drifted, os.path.join(str(tmp_path), "optimizer_shard_00000.pt"))

    follower = _osd(1.0)
    store = _store(_ep_model(REPLICA_SETS[2]), 2, monkeypatch)
    store._drop_replicated_state(follower)
    with caplog.at_level("WARNING"):
        assert store._merge_replicated_state(str(tmp_path), follower) is False
    assert "keyed differently" in caplog.text


def test_untracked_replicated_params_merge_nothing_without_failing(tmp_path, monkeypatch):
    """The legitimate empty merge: with the base experts frozen (expert-LoRA), the saving optimizer
    tracked none of the replicated FQNs, so neither shard carries state for them and there is nothing
    to restore — a False here would warm-restart the whole world over a healthy checkpoint."""
    frozen = {"state": {SHARDED_KEY: {"exp_avg": torch.ones(2)}}, "param_groups": [{"params": [SHARDED_KEY]}]}
    torch.save(frozen, os.path.join(str(tmp_path), "optimizer_shard_00000.pt"))
    store = _store(_ep_model(REPLICA_SETS[2]), 2, monkeypatch)
    assert store._merge_replicated_state(str(tmp_path), dict(frozen)) is True


def test_the_writer_fqns_match_a_real_optimizer_state_dict(monkeypatch):
    """The alignment the merge assumes, against ``get_optimizer_state_dict`` itself.

    ``expert_replica_writer`` derives the replicated key set from ``named_parameters()`` while the
    shard is keyed by torch's own state-dict FQNs; a test that builds its fixture from the same
    derivation asserts nothing about that pairing. Here the state dict is the real one, so a spelling
    that drifts on either side (an FSDP2/compile wrapper the save does not peel, a torch FQN change)
    fails here rather than at 512 ranks on resume.
    """
    monkeypatch.setattr(optimizer_mod, "get_global_rank", lambda: 2)
    model = _ep_model(REPLICA_SETS[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    optimizer.step()

    _writer, keys = expert_replica_writer(model)
    osd = get_optimizer_state_dict(model, optimizer, options=StateDictOptions(full_state_dict=False))
    assert keys and keys <= set(osd["state"]), sorted(set(osd["state"]))


class _StubParallelismConfig:
    def __init__(self, num_ep_groups):
        self.ep_size = 2
        self.experts_fsdp_managed = False  # ep2: the experts are FSDP-ignored and replica-duplicated
        self.num_ep_groups = num_ep_groups


def _fingerprint(num_ep_groups) -> OptimizerStateFingerprint:
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.1)
    return OptimizerStateFingerprint.capture(_StubParallelismConfig(num_ep_groups), optimizer, 4)


def test_replica_layout_is_fingerprinted():
    """Which rank's shard carries the replicated state follows from the replica layout, so a run
    resuming under a different one must be refused, not silently pointed at another set's writer."""
    live = _fingerprint(num_ep_groups=2)
    assert live.expert_replica_size == 2
    mismatches = live.mismatches(_fingerprint(num_ep_groups=4))
    assert any(field.startswith("expert_replica_size") for field in mismatches), mismatches
    assert not live.mismatches(_fingerprint(num_ep_groups=2))


@pytest.mark.parametrize(
    ("topology", "expected"),
    [
        ({"world_size": 8}, 1),
        ({"world_size": 8, "ep_size": 1}, 1),
        ({"world_size": 8, "ep_size": 1, "fsdp_shard_ep1_experts": False}, 8),
        ({"world_size": 8, "ep_size": 8}, 1),
        ({"world_size": 32, "ep_size": 8}, 4),
    ],
    ids=["dense", "ep1-experts-sharded", "ep1-experts-replicated", "ep8-one-group", "ep8-four-groups"],
)
def test_the_fingerprinted_replica_size_is_what_the_dedup_actually_writes(topology, expected):
    """The field records how many shards the dedup collapses into one, so it must read 1 wherever no
    writer exists.

    ``num_ep_groups`` is a derived count that is meaningful only where the expert params are
    FSDP-ignored plain tensors: at ``ep_group_size == 1`` it degenerates to one singleton group per
    rank (the layout ``EPConfig`` builds), which describes no dedup at all. Stamping that on a dense
    or default-ep1 run makes every existing checkpoint report ``saved=1 current=<world>`` on resume
    and warm-restart its optimizer — a hard raise under PP — for a topology that never changed.
    """
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.1)
    config = make_parallelism_config(**topology)
    live = OptimizerStateFingerprint.capture(config, optimizer, topology["world_size"])
    assert live.expert_replica_size == expected


@pytest.mark.parametrize("live_groups", [1, 2, 64])
def test_a_pre_dedup_meta_is_compatible_with_every_replica_size(live_groups):
    """A meta written before the field existed must resume under ANY replica layout, not just one.

    Those shards duplicate the expert moments on every rank (nothing was stripped), so they restore
    whatever the live replica size is. Reading the absent field as a VALUE — 1, say — reports
    ``saved=1 current=64`` on every rank of a multi-node EP campaign and discards the optimizer state
    world-wide (a hard raise under PP), for a topology that never changed.
    """
    saved = {k: v for k, v in _fingerprint(2).to_dict().items() if k != "expert_replica_size"}
    parsed = OptimizerStateFingerprint.from_dict(saved)
    assert parsed is not None and parsed.expert_replica_size is None
    assert not _fingerprint(live_groups).mismatches(parsed)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
