#!/usr/bin/env python
"""Pre-download hub reads are ordered main-rank-first, and stay free without a process group.

Three sites contact the hub before ``fs_aware_main_first("model_download")`` ever runs: the
text-vs-multimodal probe (``is_vlm_model``), the expert-LoRA peel (``split_expert_lora_targets``),
and the VLM processor/tokenizer load. Each is one small ``from_pretrained``; at 512 ranks it is 512
simultaneous requests for the same files (HTTP 429, after which the probe silently falls back to a
name heuristic that can DISAGREE with the config answer and route ranks to different model classes)
— and under ``trust_remote_code`` every rank also races to populate transformers' dynamic-module
cache, which is written with no lock, so a peer can import a truncated module.

``hub_metadata_main_first`` is the one seam they share. What these tests pin: the ordering actually
happens (the fetch runs inside the coordination, under a tag namespaced away from the other
main-first phases), the result is returned unchanged, a failing fetch still releases the waiters,
and a single process is untouched.

Run: pytest tests/cpu/parallelism/test_hub_metadata_coordination.py
"""

import sys

import pytest
import torch.distributed as dist

from src.distributed import filesystem, runtime
from tests.common.distributed import FakeStore

_SHARED_FILESYSTEM_VARS = (
    runtime._SHARED_FILESYSTEM_VAR,
    runtime._INPUT_SHARED_FILESYSTEM_VAR,
    runtime._OUTPUT_SHARED_FILESYSTEM_VAR,
)


@pytest.fixture()
def fake_job(monkeypatch):
    """One rank of a 2-rank shared-filesystem job with a fake coordination store."""
    store = FakeStore()
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runtime.c10d, "_get_default_store", lambda: store)
    monkeypatch.setattr(filesystem, "is_input_shared_filesystem", lambda: True)
    monkeypatch.setattr(filesystem, "get_global_world_size", lambda: 2)
    # Consensus already agreed, as it is after init_distributed; the pre-init case is its own test.
    monkeypatch.setattr(runtime, "_SHARED_FILESYSTEM_CONSENSUS", dict.fromkeys(_SHARED_FILESYSTEM_VARS, True))
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
        filesystem.hub_metadata_main_first("vlm_probe", lambda: fetched.append("hub"))
    assert fetched == [], "the peer fetched before the main rank had written the cache"


def test_the_main_rank_fetches_and_releases_the_peers(fake_job, monkeypatch):
    """Rank 0 runs the fetch, gets its value back, and publishes the done-key its peers wait on."""
    _as_rank(monkeypatch, 0)
    result = filesystem.hub_metadata_main_first("vlm_probe", lambda: "config-object")
    assert result == "config-object", "the fetch's result must pass through unchanged"
    assert any(key.endswith("/done") for key in fake_job.set_keys), "peers were never released"

    _as_rank(monkeypatch, 1)
    assert filesystem.hub_metadata_main_first("vlm_probe", lambda: "peer-read") == "peer-read"


def test_a_failing_fetch_still_releases_the_peers(fake_job, monkeypatch):
    """``is_vlm_model`` catches its own fetch failure and falls back to the name heuristic; if the
    main rank's failure skipped the done-key the peers would wait out the whole store timeout."""
    _as_rank(monkeypatch, 0)

    def boom():
        raise OSError("hub unreachable")

    with pytest.raises(OSError):
        filesystem.hub_metadata_main_first("vlm_probe", boom)
    assert any(key.endswith("/done") for key in fake_job.set_keys), "a failed fetch stranded the peers"


def test_distinct_call_sites_do_not_share_a_phase(fake_job, monkeypatch):
    """Tags are per call site because the call COUNTS differ per script; sharing one tag would make
    a script that probes twice permanently off-by-one against one that probes once."""
    _as_rank(monkeypatch, 0)
    filesystem.hub_metadata_main_first("vlm_probe", lambda: None)
    filesystem.hub_metadata_main_first("expert_lora_probe", lambda: None)
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
    filesystem.hub_metadata_main_first("vlm_processor", lambda: None)
    assert waited and waited[0] < runtime.get_store_timeout(), (
        f"the hub-metadata wait inherited the store default: {waited}"
    )


def test_a_single_process_is_untouched(monkeypatch):
    """No process group (dataset prep tools, a laptop run): the fetch must run inline, no store."""
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(runtime, "_SHARED_FILESYSTEM_CONSENSUS", None)
    assert filesystem.hub_metadata_main_first("vlm_probe", lambda: "inline") == "inline"


def test_the_scope_is_agreed_before_the_first_coordinated_phase(fake_job, monkeypatch):
    """This helper can run BEFORE ``init_distributed`` (the modality probe names the run), so it may
    be the run's first coordinated phase. The scope comes from the shared-filesystem flags: if a
    per-node override left them un-agreed, this tag's participants split across two scopes and the
    ones whose scope has no main rank wait out the whole store timeout."""
    monkeypatch.setattr(runtime, "_SHARED_FILESYSTEM_CONSENSUS", None)
    resolved = []
    monkeypatch.setattr(
        runtime,
        "resolve_shared_filesystem_consensus",
        lambda: resolved.append(True),
    )
    _as_rank(monkeypatch, 0)
    filesystem.hub_metadata_main_first("vlm_probe", lambda: None)
    assert resolved == [True], "the coordination scope was used before the ranks agreed on it"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
