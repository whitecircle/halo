#!/usr/bin/env python
"""``resolve_model_source`` — every rank must load the same checkpoint, fetched once per scope.

Real gloo ranks simulate a 2-node job on one host: each "node" gets its own Hub cache (and its own
working directory, for a relative checkpoint path), so the per-node filesystem the coordination
reasons about is genuinely different per rank. The fetch itself is stood in by staging a snapshot
into the fetching rank's cache; every other lookup is huggingface_hub's own cache resolution.

1. per-node input: each node's local rank 0 fetches exactly once, and every rank loads one commit;
2. a shared input declaration over node-local caches (only global rank 0 fetched) fails EVERY rank,
   naming the flag to fix — the peers would otherwise each re-download against the NCCL watchdog —
   including a node whose cache holds the weightless snapshot the metadata reads leave there;
3. per-node caches holding different commits (staged at different times) fail every rank instead of
   training different weights on different nodes;
4. a checkpoint directory present on one node only fails every rank instead of stranding the ranks
   that found it in the next collective;
5. a trainer handed a path string loads the agreed commit, inside a world-joined load.

    python tests/cpu/parallelism/test_model_source_agreement.py
"""

import contextlib
import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest
import torch
from huggingface_hub import constants as hub_constants
from transformers import Qwen3Config, Qwen3ForCausalLM

import src.distributed.loading.model_loading as model_loading
from src.distributed.loading import model_source
from src.distributed.loading.model_source import resolve_model_source
from tests.common.gloo import run_gloo_ranks
from tests.common.models import TINY_QWEN3_CONFIG

REPO = "org/model"
COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
RANKS_PER_NODE = 2
TWO_NODES = 2 * RANKS_PER_NODE
PG_TIMEOUT = datetime.timedelta(seconds=60)


def _stage_snapshot(cache: Path, commit: str, *, weights: bool = True) -> str:
    """What a finished ``snapshot_download`` leaves in a Hub cache: the ref, and the snapshot dir —
    without its weights for the config and tokenizer reads that precede the load."""
    storage = cache / f"models--{REPO.replace('/', '--')}"
    (storage / "refs").mkdir(parents=True, exist_ok=True)
    (storage / "refs" / "main").write_text(commit)
    snapshot = storage / "snapshots" / commit
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text(json.dumps({"model_type": "llama"}))
    if weights:
        weight_map = {f"layers.{i}.weight": shard for i, shard in enumerate(SHARDS)}
        (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        for shard in SHARDS:
            (snapshot / shard).write_bytes(b"")
    return str(snapshot)


def _enter_node(rank: int, tmp_dir: str, *, ranks_per_node: int) -> Path:
    """Make this rank one of ``ranks_per_node`` on its node, with that node's own cache and cwd."""
    node_dir = Path(tmp_dir) / f"node{rank // ranks_per_node}"
    node_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(LOCAL_RANK=str(rank % ranks_per_node), LOCAL_WORLD_SIZE=str(ranks_per_node))
    hub_constants.HF_HUB_CACHE = str(node_dir / "hub")
    os.chdir(node_dir)
    return node_dir


def _fetching_stand_in(tmp_dir: str, rank: int, commit: str):
    """``snapshot_download`` whose online call stages ``commit`` into this rank's cache (and records
    that it fetched); a cache-only call is huggingface_hub's own resolution."""

    def snapshot_download(repo_id, *, revision=None, local_files_only=False):
        if not local_files_only:
            Path(tmp_dir, f"fetched_by_rank{rank}").touch()
            _stage_snapshot(Path(hub_constants.HF_HUB_CACHE), commit)
        return huggingface_hub.snapshot_download(repo_id, revision=revision, local_files_only=True)

    return snapshot_download


def _record(tmp_dir: str, rank: int, path: str) -> None:
    try:
        result = f"RETURNED {resolve_model_source(path, None, tag='policy')}"
    except Exception as exc:  # the raise itself is the outcome under test
        result = f"{type(exc).__name__}: {exc}"
    Path(tmp_dir, f"rank{rank}.txt").write_text(result)


def _fetch_worker(rank: int, tmp_dir: str) -> None:
    _enter_node(rank, tmp_dir, ranks_per_node=RANKS_PER_NODE)
    model_source.snapshot_download = _fetching_stand_in(tmp_dir, rank, COMMIT_A)
    _record(tmp_dir, rank, REPO)


def _diverged_caches_worker(rank: int, tmp_dir: str) -> None:
    node_dir = _enter_node(rank, tmp_dir, ranks_per_node=RANKS_PER_NODE)
    if rank % RANKS_PER_NODE == 0:
        _stage_snapshot(node_dir / "hub", COMMIT_A if rank == 0 else COMMIT_B)
    # An offline fetch: the cache answers, whatever commit it holds.
    model_source.snapshot_download = lambda repo_id, *, revision=None, local_files_only=False: (
        huggingface_hub.snapshot_download(repo_id, revision=revision, local_files_only=True)
    )
    _record(tmp_dir, rank, REPO)


def _directory_on_one_node_worker(rank: int, tmp_dir: str) -> None:
    node_dir = _enter_node(rank, tmp_dir, ranks_per_node=1)
    if rank == 0:
        (node_dir / "ckpt").mkdir(exist_ok=True)
    _record(tmp_dir, rank, "ckpt")


def _outcomes(tmp_path: Path, world: int) -> list[str]:
    return [(tmp_path / f"rank{rank}.txt").read_text() for rank in range(world)]


def test_per_node_input_fetches_once_per_node_and_agrees_the_commit(tmp_path):
    run_gloo_ranks(_fetch_worker, TWO_NODES, str(tmp_path), pg_timeout=PG_TIMEOUT, env={"DIST_SHARED_FILESYSTEM": "0"})
    assert _outcomes(tmp_path, TWO_NODES) == [f"RETURNED {COMMIT_A}"] * TWO_NODES
    fetchers = sorted(path.name for path in tmp_path.glob("fetched_by_rank*"))
    assert fetchers == ["fetched_by_rank0", "fetched_by_rank2"], f"each node's local rank 0 fetches once: {fetchers}"


def test_a_shared_declaration_over_node_local_caches_fails_every_rank(tmp_path):
    run_gloo_ranks(_fetch_worker, TWO_NODES, str(tmp_path), pg_timeout=PG_TIMEOUT, env={"DIST_SHARED_FILESYSTEM": "1"})
    outcomes = _outcomes(tmp_path, TWO_NODES)
    for rank in (0, 1):
        assert outcomes[rank].startswith(
            "RuntimeError: Resolving the checkpoint 'org/model' failed on 2 of 4 rank(s) [2, 3]"
        ), f"rank {rank} found the snapshot and must still raise, naming the ranks that did not: {outcomes[rank]}"
    for rank in (2, 3):
        assert outcomes[rank].startswith("FileNotFoundError: 'org/model' is not a directory"), outcomes[rank]
    assert all("DIST_INPUT_SHARED_FILESYSTEM=0" in outcome for outcome in outcomes), outcomes
    assert [path.name for path in tmp_path.glob("fetched_by_rank*")] == ["fetched_by_rank0"]


def test_a_weightless_snapshot_in_a_node_local_cache_fails_every_rank(tmp_path):
    """The config and tokenizer reads ahead of the load leave a snapshot of the commit in every node's
    own cache, and huggingface_hub's cache-only resolution returns it without checking its files: the
    shared declaration over node-local caches must still fail every rank, naming the missing weights
    and the flag to fix, rather than send node 1 to the hub or to a missing-shard error."""
    _stage_snapshot(tmp_path / "node1" / "hub", COMMIT_A, weights=False)
    run_gloo_ranks(_fetch_worker, TWO_NODES, str(tmp_path), pg_timeout=PG_TIMEOUT, env={"DIST_SHARED_FILESYSTEM": "1"})
    for rank, outcome in enumerate(_outcomes(tmp_path, TWO_NODES)):
        assert outcome.startswith(
            "FileNotFoundError: Resolving the checkpoint 'org/model' failed on 2 of 4 rank(s) [2, 3]"
        ), f"rank {rank} must refuse a node whose snapshot lacks the weights: {outcome}"
        assert f"at commit {COMMIT_A} without 3 of the 3 weight file(s)" in outcome, outcome
        assert all(name in outcome for name in ("model.safetensors.index.json", *SHARDS)), outcome
        assert "DIST_INPUT_SHARED_FILESYSTEM=0" in outcome, outcome
    assert [path.name for path in tmp_path.glob("fetched_by_rank*")] == ["fetched_by_rank0"]


def test_per_node_caches_holding_different_commits_fail_every_rank(tmp_path):
    run_gloo_ranks(
        _diverged_caches_worker, TWO_NODES, str(tmp_path), pg_timeout=PG_TIMEOUT, env={"DIST_SHARED_FILESYSTEM": "0"}
    )
    for rank, outcome in enumerate(_outcomes(tmp_path, TWO_NODES)):
        assert outcome.startswith("ValueError: The checkpoint 'org/model' each rank resolved differs"), (
            f"rank {rank} must refuse to train on a commit its peers do not hold: {outcome}"
        )
        assert COMMIT_A in outcome and COMMIT_B in outcome, outcome


def test_a_checkpoint_directory_on_one_node_only_fails_every_rank(tmp_path):
    run_gloo_ranks(
        _directory_on_one_node_worker, 2, str(tmp_path), pg_timeout=PG_TIMEOUT, env={"DIST_SHARED_FILESYSTEM": "1"}
    )
    holder, missing = _outcomes(tmp_path, 2)
    assert holder.startswith("RuntimeError: Resolving the checkpoint 'ckpt' failed on 1 of 2 rank(s) [1]"), holder
    assert missing.startswith("FileNotFoundError: 'ckpt' is not a directory"), missing


def test_a_local_directory_keeps_the_callers_revision(tmp_path):
    assert resolve_model_source(str(tmp_path), "v1", tag="policy") == "v1"


def test_a_trainer_model_path_loads_the_agreed_commit_inside_the_joined_load(tmp_path, monkeypatch):
    """A trainer handed a path string (SMPO, offline GRPO, teacher distillation) loads it like the
    scripts' loaders do: the source agreed across ranks first, then the agreed commit read on every
    rank inside a load whose failure joins the world."""
    checkpoint = tmp_path / "tiny"
    Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG)).save_pretrained(checkpoint)
    events: list[tuple] = []

    def agreed_source(path, revision, *, tag):
        events.append(("agree", path, revision, tag))
        return COMMIT_A

    @contextlib.contextmanager
    def joined_load(what, max_concurrent):
        events.append(("enter", max_concurrent))
        yield
        events.append(("exit",))

    real_auto_load = model_loading.auto_load_model

    def recording_auto_load(path, **kwargs):
        events.append(("load", kwargs["revision"]))
        return real_auto_load(path, **kwargs)

    monkeypatch.setattr(model_loading, "resolve_model_source", agreed_source)
    monkeypatch.setattr(model_loading, "joined_node_load", joined_load)
    monkeypatch.setattr(model_loading, "auto_load_model", recording_auto_load)
    args = SimpleNamespace(model_init_kwargs={"revision": "v1", "dtype": torch.float32})

    model_loading.load_model_from_pretrained(str(checkpoint), args, model_cls=Qwen3ForCausalLM, keep_fp32=False)

    assert events == [
        ("agree", str(checkpoint), "v1", "trainer_model"),
        ("enter", 0),
        ("load", COMMIT_A),
        ("exit",),
    ], events


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
