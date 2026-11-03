#!/usr/bin/env python
"""``resolve_model_source``: every rank loads one checkpoint, fetched once per scope. Real gloo ranks simulate 2 nodes,
each with its own Hub cache and cwd; the fetch is a stand-in staging a snapshot."""

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
from huggingface_hub.utils import filter_repo_objects
from transformers import Qwen3Config, Qwen3ForCausalLM

import halo.model_loading.parallel.whole_model as whole_model
from halo.model_loading.parallel import model_source
from halo.model_loading.parallel.model_source import HUB_SUBFOLDER_IGNORE_PATTERNS, ModelSource, resolve_model_source
from tests.common.gloo import run_gloo_ranks
from tests.common.models import TINY_QWEN3_CONFIG

REPO = "org/model"
COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
# What from_pretrained reads at the top level, and the same weights again in vendor formats under subfolders.
TOP_LEVEL_FILES = (
    ".gitattributes",
    "README.md",
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "configuration_custom.py",
    "modeling_custom.py",
    "model.safetensors.index.json",
    *SHARDS,
)
NESTED_FILES = (
    "original/config.json",
    "original/dtypes.json",
    "original/model--00001-of-00002.safetensors",
    "metal/model.bin",
)
RANKS_PER_NODE = 2
TWO_NODES = 2 * RANKS_PER_NODE
PG_TIMEOUT = datetime.timedelta(seconds=60)


def _storage(cache: Path) -> Path:
    return cache / f"models--{REPO.replace('/', '--')}"


def _stage_snapshot(cache: Path, commit: str, *, weights: bool = True, nested: tuple[str, ...] = ()) -> str:
    """Weightless for the config and tokenizer reads that precede the load. No tree listing, as an older
    huggingface_hub's cache lacks one."""
    storage = _storage(cache)
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
    for name in nested:
        (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
        (snapshot / name).write_bytes(b"")
    return str(snapshot)


def _enter_node(rank: int, tmp_dir: str, *, ranks_per_node: int) -> Path:
    node_dir = Path(tmp_dir) / f"node{rank // ranks_per_node}"
    node_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(LOCAL_RANK=str(rank % ranks_per_node), LOCAL_WORLD_SIZE=str(ranks_per_node))
    hub_constants.HF_HUB_CACHE = str(node_dir / "hub")
    os.chdir(node_dir)
    return node_dir


def _fetching_stand_in(tmp_dir: str, rank: int, commit: str):
    """The hub must read as online, as the CPU tier runs offline."""

    def snapshot_download(repo_id, *, revision=None, local_files_only=False, ignore_patterns=None):
        if not local_files_only:
            Path(tmp_dir, f"fetched_by_rank{rank}").touch()
            nested = tuple(filter_repo_objects(NESTED_FILES, ignore_patterns=ignore_patterns))
            _stage_snapshot(Path(hub_constants.HF_HUB_CACHE), commit, nested=nested)
        return huggingface_hub.snapshot_download(repo_id, revision=revision, local_files_only=True)

    return snapshot_download


def _record(tmp_dir: str, rank: int, path: str) -> None:
    try:
        result = f"RETURNED {resolve_model_source(path, None, tag='policy').revision}"
    except Exception as exc:  # The raise itself is the outcome under test.
        result = f"{type(exc).__name__}: {exc}"
    Path(tmp_dir, f"rank{rank}.txt").write_text(result)


def _fetch_worker(rank: int, tmp_dir: str) -> None:
    _enter_node(rank, tmp_dir, ranks_per_node=RANKS_PER_NODE)
    hub_constants.HF_HUB_OFFLINE = False
    model_source.snapshot_download = _fetching_stand_in(tmp_dir, rank, COMMIT_A)
    _record(tmp_dir, rank, REPO)


def _diverged_caches_worker(rank: int, tmp_dir: str) -> None:
    node_dir = _enter_node(rank, tmp_dir, ranks_per_node=RANKS_PER_NODE)
    if rank % RANKS_PER_NODE == 0:
        _stage_snapshot(node_dir / "hub", COMMIT_A if rank == 0 else COMMIT_B)
    # The cache answers, whatever commit it holds.
    model_source.snapshot_download = lambda repo_id, *, revision=None, local_files_only=False, ignore_patterns=None: (
        huggingface_hub.snapshot_download(repo_id, revision=revision, local_files_only=True)
    )
    _record(tmp_dir, rank, REPO)


def _directory_on_one_node_worker(rank: int, tmp_dir: str, offline: bool) -> None:
    node_dir = _enter_node(rank, tmp_dir, ranks_per_node=1)
    hub_constants.HF_HUB_OFFLINE = offline
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
    """Cache-only resolution returns a weightless snapshot without checking files, which must still fail every rank."""
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


@pytest.mark.parametrize(
    ("offline", "remedy"),
    [
        (False, "a local checkpoint must sit on a mount every node sees"),
        (True, "a local checkpoint must exist at this path on this node"),
    ],
    ids=["online", "offline"],
)
def test_a_checkpoint_directory_on_one_node_only_fails_every_rank(tmp_path, offline, remedy):
    run_gloo_ranks(
        _directory_on_one_node_worker,
        2,
        str(tmp_path),
        offline,
        pg_timeout=PG_TIMEOUT,
        env={"DIST_SHARED_FILESYSTEM": "1"},
    )
    holder, missing = _outcomes(tmp_path, 2)
    assert holder.startswith("RuntimeError: Resolving the checkpoint 'ckpt' failed on 1 of 2 rank(s) [1]"), holder
    assert missing.startswith("FileNotFoundError: 'ckpt' is not a directory"), missing
    assert remedy in holder and remedy in missing, f"the rank missing the checkpoint must name its remedy: {missing}"


def test_a_local_directory_keeps_the_callers_revision(tmp_path):
    assert resolve_model_source(str(tmp_path), "v1", tag="policy") == ModelSource(str(tmp_path), "v1", str(tmp_path))


def test_the_model_fetch_rule_keeps_the_top_level_files_alone():
    """huggingface_hub's fnmatch ``*`` crosses ``/``."""
    listing = [*TOP_LEVEL_FILES, *NESTED_FILES]
    kept = list(filter_repo_objects(listing, ignore_patterns=list(HUB_SUBFOLDER_IGNORE_PATTERNS)))
    assert kept == list(TOP_LEVEL_FILES)


@pytest.mark.parametrize("whole_repo", [False, True], ids=["model-load", "whole-repo"])
def test_the_fetch_leaves_the_subfolders_unless_the_loader_reads_them(tmp_path, monkeypatch, whole_repo):
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(model_source, "snapshot_download", _fetching_stand_in(str(tmp_path), 0, COMMIT_A))

    source = resolve_model_source(REPO, None, tag="policy", whole_repo=whole_repo)

    snapshot = tmp_path / "hub" / f"models--{REPO.replace('/', '--')}" / "snapshots" / COMMIT_A
    assert source == ModelSource(REPO, COMMIT_A, str(snapshot)), "every rank reads its own verified snapshot"
    fetched = {str(path.relative_to(snapshot)) for path in snapshot.rglob("*") if path.is_file()}
    assert set(SHARDS) <= fetched
    assert sorted(name for name in fetched if "/" in name) == (sorted(NESTED_FILES) if whole_repo else [])


@pytest.mark.parametrize("revision", [COMMIT_A, None], ids=["pinned", "main"])
def test_offline_the_fetching_rank_resolves_from_its_cache_alone(tmp_path, monkeypatch, revision):
    """A call that may fetch lists a pinned commit's files through the Hub unless cached, and offline that raises
    although the snapshot is complete."""
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    snapshot = _stage_snapshot(tmp_path / "hub", COMMIT_A)
    assert not (_storage(tmp_path / "hub") / "trees").exists()

    assert resolve_model_source(REPO, revision, tag="policy") == ModelSource(REPO, COMMIT_A, snapshot)


def test_offline_a_snapshot_missing_from_the_cache_names_the_offline_hub(tmp_path, monkeypatch):
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    with pytest.raises(FileNotFoundError) as raised:
        resolve_model_source(REPO, COMMIT_A, tag="policy")
    message = str(raised.value)
    assert f"holds no snapshot of it at revision {COMMIT_A}" in message, message
    assert "The hub is offline (HF_HUB_OFFLINE)" in message, message
    assert "a Hub snapshot must be staged complete into this rank's cache" in message, message


def test_a_trainer_model_path_loads_the_agreed_commit_inside_the_joined_load(tmp_path, monkeypatch):
    checkpoint = tmp_path / "tiny"
    Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG)).save_pretrained(checkpoint)
    events: list[tuple] = []

    def agreed_source(path, revision, *, tag):
        events.append(("agree", path, revision, tag))
        return ModelSource(path, COMMIT_A, path)

    @contextlib.contextmanager
    def joined_load(what, max_concurrent):
        events.append(("enter", max_concurrent))
        yield
        events.append(("exit",))

    real_auto_load = whole_model.auto_load_model

    def recording_auto_load(path, **kwargs):
        events.append(("load", kwargs["revision"]))
        return real_auto_load(path, **kwargs)

    monkeypatch.setattr(whole_model, "resolve_model_source", agreed_source)
    monkeypatch.setattr(whole_model, "joined_node_load", joined_load)
    monkeypatch.setattr(whole_model, "auto_load_model", recording_auto_load)
    args = SimpleNamespace(model_init_kwargs={"revision": "v1", "dtype": torch.float32})

    whole_model.resolve_trainer_model(str(checkpoint), args, parallelism_config=None, model_class=Qwen3ForCausalLM)

    assert events == [
        ("agree", str(checkpoint), "v1", "trainer_model"),
        ("enter", 0),
        ("load", COMMIT_A),
        ("exit",),
    ], events


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
