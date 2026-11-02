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
The same topology then runs on four gloo ranks through the real ``save`` and ``load``, where the restore
gate and ``set_optimizer_state_dict`` follow the merge: an expert-only run (every trainable param a
replicated EP-layer one, as expert-only LoRA trains) resumes every rank bit for bit though its followers'
shards hold no state at all, and refuses on every rank where a follower's node lacks the writer's shard;
a run with every EP-layer param frozen resumes from per-node copies without reaching for a writer's shard.

    python tests/cpu/checkpoint/test_optimizer_replica_dedup.py
"""

import copy
import datetime
import json
import os
import shutil

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

import halo.checkpoint.parallel.optimizer as optimizer_mod
from halo.checkpoint.parallel.optimizer import OptimizerShardStore, expert_replica_writer
from halo.checkpoint.parallel.optimizer_meta import OptimizerStateFingerprint
from halo.distributed.expert_parallel.families.qwen3 import EPQwen3MoELayer
from tests.common.checkpoint_io import load_context, shard_store_context
from tests.common.gloo import run_gloo_ranks
from tests.common.optimizer_state import optimizer_state_matches
from tests.common.parallelism import make_parallelism_config

# ep2 over 4 ranks = 2 EP groups; the same ep_rank in each group is a replica of the other.
REPLICA_SETS = {0: [0, 2], 1: [1, 3], 2: [0, 2], 3: [1, 3]}
REPLICATED_KEYS = {"mlp.experts", "mlp.router.weight"}
SHARDED_KEY = "fc.weight"
WORLD_SIZE = len(REPLICA_SETS)
# Each EP group on a node of its own: the followers (ranks 2, 3) sit on the node without the writers' shards.
NODE_OF_RANK = {0: "a", 1: "a", 2: "b", 3: "b"}
TRAIN_STEPS = 3
SEED = 20261101
# A rank that leaves a collective records gloo's error at this bound instead of gloo's 30-minute default.
PG_TIMEOUT = datetime.timedelta(seconds=120)
# Which params each gloo run trains, by FQN, and whether its resume reads per-node copies of the checkpoint.
RESUME_RUNS = {
    "expert_only": (lambda name: name.startswith("mlp."), False),
    "expert_only_per_node_copies": (lambda name: name.startswith("mlp."), True),
    "frozen_ep_per_node_copies": (lambda name: not name.startswith("mlp."), True),
}


class _StubEpConfig:
    def __init__(self, replica_ranks, *, experts_fsdp_managed=False):
        self.expert_replica_ranks = replica_ranks
        self.needs_expert_grad_sync = len(replica_ranks) > 1
        self.experts_fsdp_managed = experts_fsdp_managed


def _ep_model(replica_ranks, *, experts_fsdp_managed=False) -> nn.Module:
    """A model whose ``mlp`` is a real EP layer instance (class only, as the seam reads nothing
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
    return OptimizerShardStore(load_context(model, has_ep_layers=True, fsdp_wrapped=True))


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
        assert store._merge_replicated_state(str(tmp_path), osd) is None
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
    assert store._merge_replicated_state(str(tmp_path), osd) is None


def test_pre_dedup_checkpoint_needs_no_writer_shard(tmp_path, monkeypatch):
    """A shard set written before the dedup (every rank self-contained) must resume as it always
    did, without reaching for a peer's file."""
    store = _store(_ep_model(REPLICA_SETS[2]), 2, monkeypatch)
    assert store._merge_replicated_state(str(tmp_path), _osd(1.0)) is None


def test_a_stateless_replicated_param_is_not_read_as_a_dedup(tmp_path, monkeypatch):
    """A tracked param that never received a gradient is absent from EVERY shard, so "some
    replicated key is missing" would send a self-contained shard chasing a writer's file that a
    non-shared filesystem does not have. The test is all-or-nothing for exactly this reason."""
    store = _store(_ep_model(REPLICA_SETS[2]), 2, monkeypatch)
    osd = _osd(1.0)
    del osd["state"]["mlp.router.weight"]
    assert store._merge_replicated_state(str(tmp_path), osd) is None
    assert "mlp.experts" in osd["state"], "the surviving replicated state must not be disturbed"


def _expert_only_shards(tmp_path: str, monkeypatch, *, stepped: bool) -> dict:
    """The writer's (rank 0) and a follower's (rank 2) shards of a run that trains only the replicated EP-layer
    params, as expert-only LoRA does, saved the way ``save`` writes them from a real ``get_optimizer_state_dict``;
    returns the state every rank held before the dedup."""
    model = _ep_model(REPLICA_SETS[0])
    model.fc.weight.requires_grad_(False)
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=0.1)
    if stepped:
        for param in optimizer.param_groups[0]["params"]:
            param.grad = torch.ones_like(param)
        optimizer.step()
    full = get_optimizer_state_dict(model, optimizer, options=StateDictOptions(full_state_dict=False))
    for rank in (0, 2):
        osd = copy.deepcopy(full)
        _store(_ep_model(REPLICA_SETS[rank]), rank, monkeypatch)._drop_replicated_state(osd)
        torch.save(osd, os.path.join(tmp_path, f"optimizer_shard_{rank:05d}.pt"))
    return full["state"]


@pytest.mark.parametrize("stepped", [True, False], ids=["trained", "pre-first-step"])
def test_a_follower_whose_whole_state_is_replicated_restores_the_writers(tmp_path, monkeypatch, stepped):
    """Expert-only LoRA under multi-group EP: every trainable param is a replicated EP-layer one, so the dedup
    leaves the follower's shard no state at all. That empty state is no pre-first-step checkpoint:
    ``get_optimizer_state_dict`` zero-initializes a never-stepped optimizer before reading it, so the writer holds
    moments and a step count even then. Unmerged, the follower resumes at fresh moments while its replica
    restores, and the replicas' expert weights part from the first resumed step."""
    writer_state = _expert_only_shards(str(tmp_path), monkeypatch, stepped=stepped)
    assert set(writer_state) == REPLICATED_KEYS, "premise: the read leaves no trainable param without state"
    follower = torch.load(os.path.join(str(tmp_path), "optimizer_shard_00002.pt"), weights_only=False)
    assert follower["state"] == {}, "premise: the dedup stripped the follower's whole state"

    store = _store(_ep_model(REPLICA_SETS[2]), 2, monkeypatch)
    assert store._merge_replicated_state(str(tmp_path), follower) is None
    assert set(follower["state"]) == REPLICATED_KEYS
    for key, state in writer_state.items():
        for name, value in state.items():
            assert torch.equal(follower["state"][key][name], value), (key, name)


def test_an_empty_follower_shard_without_its_writer_shard_fails_the_read(tmp_path, monkeypatch):
    """The loud half of reading an empty shard as deduplicated: an expert-only follower whose writer's shard is
    out of reach holds no moments to resume from, and resuming anyway would reset every one of them."""
    _expert_only_shards(str(tmp_path), monkeypatch, stepped=True)
    os.remove(os.path.join(str(tmp_path), "optimizer_shard_00000.pt"))
    follower = torch.load(os.path.join(str(tmp_path), "optimizer_shard_00002.pt"), weights_only=False)
    store = _store(_ep_model(REPLICA_SETS[2]), 2, monkeypatch)
    assert "optimizer_shard_00000.pt is not in" in (store._merge_replicated_state(str(tmp_path), follower) or "")


def test_a_shard_tracking_no_replicated_param_needs_no_writer_shard(tmp_path, monkeypatch):
    """Attention-only LoRA under multi-group EP: every EP-layer param is frozen, so no shard tracks a replicated
    FQN and the save stripped nothing. On a non-shared filesystem the follower's node holds no writer shard, and
    reaching for one would refuse a healthy checkpoint's resume on every rank."""
    osd = {"state": {SHARDED_KEY: {"exp_avg": torch.ones(2)}}, "param_groups": [{"params": [SHARDED_KEY]}]}
    store = _store(_ep_model(REPLICA_SETS[2]), 2, monkeypatch, shared_fs=False)
    assert store._merge_replicated_state(str(tmp_path), osd) is None
    assert set(osd["state"]) == {SHARDED_KEY}


def test_a_stateless_optimizer_merges_nothing_without_failing(tmp_path, monkeypatch):
    """Plain SGD keeps no per-param state, so writer and follower both hold none: there is nothing to merge, and
    a writer holding no state at all is not one that spells the experts differently."""
    model = _ep_model(REPLICA_SETS[0])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    optimizer.step()
    osd = get_optimizer_state_dict(model, optimizer, options=StateDictOptions(full_state_dict=False))
    assert osd["state"] == {}, "premise: SGD without momentum keeps no state"
    assert set(osd["param_groups"][0]["params"]) >= REPLICATED_KEYS, "premise: the replicated params are tracked"
    torch.save(osd, os.path.join(str(tmp_path), "optimizer_shard_00000.pt"))
    store = _store(_ep_model(REPLICA_SETS[2]), 2, monkeypatch)
    assert store._merge_replicated_state(str(tmp_path), copy.deepcopy(osd)) is None


def test_missing_writer_shard_fails_the_read_verdict(tmp_path, monkeypatch):
    """A follower missing its replicated state with no writer shard in reach would otherwise restore
    a partial optimizer and log success. The reason rides the shard-read verdict that follows, so the
    whole world's resume fails instead of this rank alone."""
    _write_deduped_checkpoint(str(tmp_path), monkeypatch)
    os.remove(os.path.join(str(tmp_path), "optimizer_shard_00000.pt"))
    store = _store(_ep_model(REPLICA_SETS[2]), 2, monkeypatch)
    osd = torch.load(os.path.join(str(tmp_path), "optimizer_shard_00002.pt"), weights_only=False)
    assert "optimizer_shard_00000.pt is not in" in (store._merge_replicated_state(str(tmp_path), osd) or "")


def test_a_writer_shard_spelling_the_experts_differently_fails_the_read(tmp_path, monkeypatch):
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
    refusal = store._merge_replicated_state(str(tmp_path), follower) or ""
    assert "keyed differently" in refusal
    # Both sides of the mismatch are shown: this run's names and the writer's own spelling.
    assert any(f"'{key}'" in refusal for key in drifted["state"]), refusal


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


def _seeded_step(model, optimizer, step: int) -> None:
    """One optimizer step on gradients seeded by step and parameter, the same on every rank, as the replica
    group's gradient average leaves them."""
    for index, param in enumerate(model.parameters()):
        if param.requires_grad:
            param.grad = torch.randn(param.shape, generator=torch.Generator().manual_seed(1000 * step + index))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _trained_pair(rank: int, trains) -> tuple[nn.Module, torch.optim.Optimizer]:
    """This rank's model, identical on every rank, with only the params ``trains`` names trainable, and its
    optimizer."""
    torch.manual_seed(SEED)
    model = _ep_model(REPLICA_SETS[rank])
    for name, param in model.named_parameters():
        param.requires_grad_(trains(name))
    return model, torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=0.1)


def _resume_run(rank: int, root: str, trains, per_node_copies: bool) -> dict:
    """Train, save through the store, and resume into a fresh optimizer; what this rank saw."""
    config = make_parallelism_config(world_size=WORLD_SIZE, gpus_per_node=WORLD_SIZE, ep_size=2, rank=rank)
    save_dir = os.path.join(root, "checkpoint")
    if rank == 0:
        os.makedirs(save_dir)
    dist.barrier()
    model, optimizer = _trained_pair(rank, trains)
    for step in range(TRAIN_STEPS):
        _seeded_step(model, optimizer, step)
    saved = copy.deepcopy(optimizer.state_dict())
    OptimizerShardStore(shard_store_context(model, optimizer, config)).save(save_dir)
    shard = torch.load(os.path.join(save_dir, f"optimizer_shard_{rank:05d}.pt"), weights_only=False)

    load_dir = save_dir
    if per_node_copies:
        # Each node's disk holds its own ranks' shards and the meta, as a checkpoint copied out per node.
        if rank == 0:
            for node in sorted(set(NODE_OF_RANK.values())):
                node_dir = os.path.join(root, f"node_{node}")
                os.makedirs(node_dir)
                shutil.copy(os.path.join(save_dir, "optimizer_meta.pt"), node_dir)
                for peer in (peer for peer, home in NODE_OF_RANK.items() if home == node):
                    shutil.copy(os.path.join(save_dir, f"optimizer_shard_{peer:05d}.pt"), node_dir)
        dist.barrier()
        load_dir = os.path.join(root, f"node_{NODE_OF_RANK[rank]}")

    resumed, resumed_optimizer = _trained_pair(rank, trains)
    resumed.load_state_dict(model.state_dict())
    raised = None
    try:
        OptimizerShardStore(shard_store_context(resumed, resumed_optimizer, config)).load(load_dir)
    except RuntimeError as error:
        raised = str(error)
    exact, why = optimizer_state_matches(saved, resumed_optimizer.state_dict())

    _seeded_step(resumed, resumed_optimizer, TRAIN_STEPS)
    experts = [torch.empty_like(resumed.mlp.experts) for _ in range(WORLD_SIZE)]
    dist.all_gather(experts, resumed.mlp.experts.detach().contiguous())
    return {
        "shard_state_keys": sorted(shard["state"]),
        "raised": raised,
        "restored_exact": exact,
        "why": why,
        "replicas_equal_after_a_resumed_step": all(
            torch.equal(experts[peer], experts[min(REPLICA_SETS[peer])]) for peer in range(WORLD_SIZE)
        ),
    }


def _resume_worker(rank: int, root: str) -> None:
    for name, (trains, per_node_copies) in RESUME_RUNS.items():
        try:
            outcome = _resume_run(rank, os.path.join(root, name), trains, per_node_copies)
        except Exception as error:  # every rank still writes a verdict for the assertions to read
            outcome = {"setup_error": f"{type(error).__name__}: {error}"}
        with open(os.path.join(root, f"{name}_{rank}.json"), "w") as fh:
            json.dump(outcome, fh)


@pytest.fixture(scope="module")
def resume_runs(tmp_path_factory):
    """Every :data:`RESUME_RUNS` entry on four gloo ranks; returns each run's per-rank outcomes."""
    root = tmp_path_factory.mktemp("replica_dedup_resume")
    for name in RESUME_RUNS:
        (root / name).mkdir()
    run_gloo_ranks(_resume_worker, WORLD_SIZE, str(root), pg_timeout=PG_TIMEOUT, env={"DIST_SHARED_FILESYSTEM": "1"})
    return {
        name: [json.loads((root / f"{name}_{rank}.json").read_text()) for rank in range(WORLD_SIZE)]
        for name in RESUME_RUNS
    }


def test_an_expert_only_run_resumes_every_rank_bit_for_bit(resume_runs):
    """The followers' shards carry no state at all after the dedup; each follower must still resume the
    moments it saved, read from its writer's shard, or its replicas part from the first resumed step."""
    outcomes = resume_runs["expert_only"]
    assert not [outcome for outcome in outcomes if "setup_error" in outcome], outcomes
    followers = [rank for rank in range(WORLD_SIZE) if rank != min(REPLICA_SETS[rank])]
    assert all(outcomes[rank]["shard_state_keys"] == [] for rank in followers), (
        "premise: a follower's shard kept state"
    )
    unrestored = {
        rank: outcome["raised"] or outcome["why"]
        for rank, outcome in enumerate(outcomes)
        if outcome["raised"] is not None or not outcome["restored_exact"]
    }
    assert not unrestored, f"ranks that did not resume the moments they saved: {unrestored}"
    assert all(outcome["replicas_equal_after_a_resumed_step"] for outcome in outcomes), "the replicas parted"


def test_an_expert_only_follower_without_its_writer_shard_refuses_on_every_rank(resume_runs):
    """Resumed from per-node copies, a follower's node lacks its writer's shard: it has no moments to resume
    from, so the whole world refuses rather than the follower resetting them while its replica restores."""
    for rank, outcome in enumerate(resume_runs["expert_only_per_node_copies"]):
        assert "setup_error" not in outcome, (rank, outcome)
        assert outcome["raised"] is not None and "optimizer_shard_00000.pt is not in" in outcome["raised"], (
            rank,
            outcome["raised"],
        )


def test_a_run_with_frozen_ep_layers_resumes_from_per_node_copies(resume_runs):
    """No shard tracks a replicated param, so nothing was stripped and no follower may reach for a writer's
    shard its node does not hold: every rank resumes its own shard bit for bit."""
    for rank, outcome in enumerate(resume_runs["frozen_ep_per_node_copies"]):
        assert "setup_error" not in outcome, (rank, outcome)
        assert outcome["shard_state_keys"] == [SHARDED_KEY], (rank, outcome["shard_state_keys"])
        assert outcome["raised"] is None, (rank, outcome["raised"])
        assert outcome["restored_exact"], (rank, outcome["why"])


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
    mismatches = live.mismatches(_fingerprint(num_ep_groups=4).to_dict())
    assert any(field.startswith("expert_replica_size") for field in mismatches), mismatches
    assert not live.mismatches(_fingerprint(num_ep_groups=2).to_dict())


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

    Those shards hold the expert moments on every rank (nothing was stripped), so they restore whatever
    the live replica size is. Reading the absent field as a value would discard the optimizer state
    world-wide (a raise under PP) for a topology that never changed.
    """
    saved = {k: v for k, v in _fingerprint(2).to_dict().items() if k != "expert_replica_size"}
    assert not _fingerprint(live_groups).mismatches(saved)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
