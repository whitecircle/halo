#!/usr/bin/env python
"""Pre-download hub reads are ordered main-rank-first, and stay free without a process group.

Three sites contact the hub before the coordinated source resolution (``resolve_model_source``) ever
runs: the text-vs-multimodal probe (``probe_checkpoint``), the expert-LoRA peel
(``split_expert_lora_targets``), and the VLM processor/tokenizer load. Each is one small
``from_pretrained``; at 512 ranks it is 512 simultaneous requests for the same files (HTTP 429, after
which the probe falls back to a name heuristic that can DISAGREE with the config answer its peers read)
— and under ``trust_remote_code`` every rank also races to populate transformers' dynamic-module
cache.

``hub_metadata_main_first`` is the one seam they share. What these tests pin: the ordering actually
happens (the fetch runs inside the coordination, under a tag namespaced away from the other
main-first phases), the result is returned unchanged, a failing fetch still releases the waiters,
and a single process is untouched.

Run: pytest tests/cpu/runtime/test_hub_metadata_coordination.py
"""

import socketserver
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.distributed as dist
import transformers.utils.hub as transformers_hub
from huggingface_hub import constants as hub_constants
from huggingface_hub.file_download import repo_folder_name
from transformers import Qwen3Config

from halo.env import store_timeout
from halo.model_loading.parallel.model_source import fetch_checkpoint_config
from halo.runtime import consensus, filesystem
from tests.common.distributed import FakeStore, reset_shared_filesystem_consensus

_SHARED_FILESYSTEM_VARS = (
    filesystem._SHARED_FILESYSTEM_VAR,
    filesystem._INPUT_SHARED_FILESYSTEM_VAR,
    filesystem._OUTPUT_SHARED_FILESYSTEM_VAR,
)


@pytest.fixture()
def fake_job(monkeypatch):
    """One rank of a 2-rank shared-filesystem job with a fake coordination store."""
    store = FakeStore()
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(consensus.c10d, "_get_default_store", lambda: store)
    monkeypatch.setattr(filesystem, "is_input_shared_filesystem", lambda: True)
    monkeypatch.setattr(filesystem, "get_global_world_size", lambda: 2)
    # Consensus already agreed, as it is after init_distributed; the pre-init case is its own test.
    monkeypatch.setattr(filesystem, "_shared_filesystem_consensus", dict.fromkeys(_SHARED_FILESYSTEM_VARS, True))
    return store


def _as_rank(monkeypatch, rank: int) -> None:
    monkeypatch.setattr(filesystem, "get_global_rank", lambda: rank)
    monkeypatch.setattr(filesystem, "is_global_main_process", lambda: rank == 0)


def test_a_peer_waits_for_the_main_rank_s_fetch(fake_job, monkeypatch):
    """The whole point: rank 1 must not reach the hub before rank 0 has populated the cache.

    ``FakeStore.wait`` raises instead of blocking, so "would have waited" is observable — and a
    fetch that ran anyway would leave ``fetched`` set.
    """
    _as_rank(monkeypatch, 1)
    fetched = []
    with pytest.raises(RuntimeError, match="hub_meta/vlm_probe"):
        filesystem.hub_metadata_main_first("vlm_probe", lambda **_: fetched.append("hub"))
    assert fetched == [], "the peer fetched before the main rank had written the cache"


def test_the_main_rank_fetches_and_releases_the_peers(fake_job, monkeypatch):
    """Rank 0 runs the fetch, gets its value back, and publishes the done-key its peers wait on."""
    _as_rank(monkeypatch, 0)
    result = filesystem.hub_metadata_main_first("vlm_probe", lambda **_: "config-object")
    assert result == "config-object", "the fetch's result must pass through unchanged"
    assert any(key.endswith("/done") for key in fake_job.set_keys), "peers were never released"

    _as_rank(monkeypatch, 1)
    assert filesystem.hub_metadata_main_first("vlm_probe", lambda **_: "peer-read") == "peer-read"


@pytest.mark.parametrize("shared", [True, False], ids=["shared", "per-node"])
def test_only_each_scopes_main_rank_may_query_the_hub(fake_job, monkeypatch, shared):
    """A follower reads the cache its scope's main rank filled, never the Hub: at 1,024 ranks a follower
    query per read is a burst of metadata requests against a per-account rate limit, whose 429 the
    probe would read as a missing config. Per-node, each node's local rank 0 leads its own scope."""
    monkeypatch.setattr(filesystem, "is_input_shared_filesystem", lambda: shared)
    monkeypatch.setattr(filesystem, "get_local_world_size", lambda: 1 if not shared else 2)
    asked = []
    for rank, local_main in ((0, True), (1, not shared)):
        _as_rank(monkeypatch, rank)
        # Per-node, each rank leads its own one-rank node.
        monkeypatch.setattr(filesystem, "get_node_rank", lambda rank=rank: rank)
        monkeypatch.setattr(filesystem, "get_local_rank", lambda local_main=local_main: 0 if local_main else 1)
        monkeypatch.setattr(filesystem, "is_local_main_process", lambda local_main=local_main: local_main)
        filesystem.hub_metadata_main_first("vlm_probe", lambda local_files_only: asked.append(local_files_only))
    assert asked == [False, shared], f"cache-only per rank: {asked}"


class _CountingEndpoint(socketserver.ThreadingTCPServer):
    """A Hub endpoint that counts the connections made to it and answers none."""

    daemon_threads = True

    def __init__(self):
        self.connections = 0
        super().__init__(("127.0.0.1", 0), socketserver.BaseRequestHandler)

    def process_request(self, request, client_address):
        self.connections += 1
        self.shutdown_request(request)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


def _cached_tiny_config(cache: Path, repo_id: str) -> None:
    """``repo_id``'s ``main`` in a Hub cache at ``cache``: a config and nothing else, as a metadata read leaves it."""
    repo = cache / repo_folder_name(repo_id=repo_id, repo_type="model")
    commit = "0" * 40
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir(parents=True)
    Qwen3Config(hidden_size=16, num_hidden_layers=1, num_attention_heads=2).to_json_file(snapshot / "config.json")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(commit)


def test_a_follower_reads_its_scopes_cache_without_a_hub_request(monkeypatch, tmp_path):
    """Through a real call site: the scope's main rank contacts the Hub, and its follower then reads the
    config from the cache that fetch left, with no request of its own. The group is faked inside the
    coordination only, since transformers itself asks ``torch.distributed`` for a rank."""
    monkeypatch.setattr(filesystem, "has_process_group", lambda: True)
    store = FakeStore()
    monkeypatch.setattr(consensus.c10d, "_get_default_store", lambda: store)
    monkeypatch.setattr(filesystem, "is_input_shared_filesystem", lambda: True)
    monkeypatch.setattr(filesystem, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(filesystem, "_shared_filesystem_consensus", dict.fromkeys(_SHARED_FILESYSTEM_VARS, True))
    _cached_tiny_config(tmp_path, "org/tiny")
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    model_config = SimpleNamespace(model_name_or_path="org/tiny", model_revision=None, trust_remote_code=False)
    endpoint = _CountingEndpoint()
    threading.Thread(target=endpoint.serve_forever, daemon=True).start()
    monkeypatch.setattr(hub_constants, "ENDPOINT", endpoint.url)
    # transformers' shared HfApi latches the endpoint when first built, possibly by an earlier test.
    monkeypatch.setattr(transformers_hub, "_hf_api", None)
    try:
        _as_rank(monkeypatch, 0)
        assert fetch_checkpoint_config(model_config, tag="config").hidden_size == 16
        by_main = endpoint.connections
        _as_rank(monkeypatch, 1)
        assert fetch_checkpoint_config(model_config, tag="config").hidden_size == 16
    finally:
        endpoint.shutdown()
        endpoint.server_close()
    assert by_main > 0, "anti-vacuity: the main rank's read must reach the endpoint"
    assert endpoint.connections == by_main, f"the follower made {endpoint.connections - by_main} Hub request(s)"


def test_a_failing_fetch_still_releases_the_peers(fake_job, monkeypatch):
    """``probe_checkpoint`` catches its own fetch failure and falls back to the name heuristic; if the
    main rank's failure skipped the done-key the peers would wait out the whole store timeout."""
    _as_rank(monkeypatch, 0)

    def boom(**_):
        raise OSError("hub unreachable")

    with pytest.raises(OSError):
        filesystem.hub_metadata_main_first("vlm_probe", boom)
    assert any(key.endswith("/done") for key in fake_job.set_keys), "a failed fetch stranded the peers"


def test_distinct_call_sites_do_not_share_a_phase(fake_job, monkeypatch):
    """Tags are per call site because the call COUNTS differ per script; sharing one tag would make
    a script that probes twice permanently off-by-one against one that probes once."""
    _as_rank(monkeypatch, 0)
    filesystem.hub_metadata_main_first("vlm_probe", lambda **_: None)
    filesystem.hub_metadata_main_first("expert_lora_probe", lambda **_: None)
    tags = {key.rsplit("/p", 1)[0] for key in fake_job.set_keys}
    assert tags == {
        "_halo_phase/main_first/hub_meta/vlm_probe/shared",
        "_halo_phase/main_first/hub_meta/expert_lora_probe/shared",
    }, f"call sites shared a coordination phase: {tags}"


def test_the_wait_is_bounded_far_below_the_store_default(fake_job, monkeypatch):
    """These phases cover a handful of small config files, not a 100B snapshot. Inheriting the
    hours-scale store budget would turn a call site entered from a branch that is not in fact
    rank-uniform into a silent multi-hour stall instead of an error naming the tag."""
    _as_rank(monkeypatch, 1)
    waited = []
    monkeypatch.setattr(fake_job, "wait", lambda keys, timeout=None: waited.append(timeout))
    filesystem.hub_metadata_main_first("vlm_processor", lambda **_: None)
    assert waited and waited[0] < store_timeout(), f"the hub-metadata wait inherited the store default: {waited}"


def test_a_single_process_is_untouched(monkeypatch):
    """No process group (dataset prep tools, a laptop run): the fetch must run inline, no store."""
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    reset_shared_filesystem_consensus()
    assert filesystem.hub_metadata_main_first("vlm_probe", lambda **_: "inline") == "inline"


def test_a_follower_without_a_process_group_still_fetches(monkeypatch):
    """Before ``init_distributed`` nobody waits for the scope's main rank, so a per-node follower (LOCAL_RANK 1)
    cannot count on a cache it filled: read cache-only, a cold cache would fail where the Hub would answer."""
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setenv("DIST_INPUT_SHARED_FILESYSTEM", "0")
    monkeypatch.setenv("LOCAL_RANK", "1")
    reset_shared_filesystem_consensus()
    assert not filesystem.is_fs_load_rank(), "anti-vacuity: this rank must be a follower of its scope"
    asked = []
    filesystem.hub_metadata_main_first("vlm_probe", lambda local_files_only: asked.append(local_files_only))
    assert asked == [False], "a follower with no main rank ahead of it read the cache alone"


def test_the_scope_is_agreed_before_the_first_coordinated_phase(fake_job, monkeypatch):
    """This helper can run BEFORE ``init_distributed`` (the modality probe names the run), so it may
    be the run's first coordinated phase. The scope comes from the shared-filesystem flags: if a
    per-node override left them un-agreed, this tag's participants split across two scopes and the
    ones whose scope has no main rank wait out the whole store timeout."""
    reset_shared_filesystem_consensus()
    resolved = []
    monkeypatch.setattr(
        filesystem,
        "resolve_shared_filesystem_consensus",
        lambda: resolved.append(True),
    )
    _as_rank(monkeypatch, 0)
    filesystem.hub_metadata_main_first("vlm_probe", lambda **_: None)
    assert resolved == [True], "the coordination scope was used before the ranks agreed on it"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
