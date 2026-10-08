#!/usr/bin/env python
"""The preprocessed-dataset probe runs once per input-filesystem scope, not once per rank.

``load_datasets_auto`` branches every rank on whether the source is a preprocessed artifact. For a
Hub-named dataset the probe is a ``metadata.json`` request to the Hub, and a per-rank probe sends one
per rank — a storm the Hub rate-limits on a large job. The verdict is the same on every rank of a
filesystem scope, so only the scope's load rank asks and the rest take the agreed answer: one
request on a shared input filesystem, one per node on per-node storage. Proven on a real gloo group
by counting the requests each rank makes, and by every rank — probing or not — taking the branch the
verdict decides. A probe that raises on a probing rank raises its cause on every rank, rather than
leaving the peers in the consensus all-reduce until the process-group timeout. Since one rank's
answer is the whole scope's, a Hub it cannot reach must raise rather than read as a raw dataset;
offline, a cache without the file is the absence it reads as.

    python tests/cpu/data/test_input_probe_scope.py
"""

import datetime
import json
import os

import pytest
from accelerate import PartialState
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

from src.data.pipeline import preprocessed_metadata
from src.data.pipeline.preprocessed_metadata import PreprocessedDatasetMetadata, is_preprocessed_dataset
from src.data.probe_consensus import agree_input_probe_across_ranks
from src.data.sources.loading import load_datasets_auto
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 4
HUB_PATH = "org/name"
# A peer left in the consensus all-reduce fails at this bound instead of hanging the suite.
PG_TIMEOUT = datetime.timedelta(seconds=30)


def _worker(rank: int, tmp_dir: str, ranks_per_node: int, stamped: bool) -> None:
    """Count this rank's Hub requests while it resolves the probe; record what it decided."""
    os.environ["LOCAL_RANK"] = str(rank % ranks_per_node)
    PartialState()
    requests = []

    def _download(repo_id, filename, repo_type=None, **kwargs):
        requests.append(repo_id)
        if not stamped:
            raise EntryNotFoundError("no metadata.json")
        return os.path.join(tmp_dir, "metadata.json")

    preprocessed_metadata.hf_hub_download = _download
    if stamped:
        try:
            load_datasets_auto(HUB_PATH, test_size=None, dataset_ratio=None)
            outcome = "NO RAISE"
        except ValueError as e:
            outcome = "hub refusal" if "on the Hub" in str(e) else f"ValueError: {e}"
    else:
        verdict = agree_input_probe_across_ranks(
            lambda: is_preprocessed_dataset(HUB_PATH), HUB_PATH, "is_preprocessed_dataset"
        )
        outcome = f"verdict {verdict}"
    with open(os.path.join(tmp_dir, f"rank_{rank}.json"), "w") as f:
        json.dump({"requests": len(requests), "outcome": outcome}, f)


def _raising_probe_worker(rank: int, tmp_dir: str, ranks_per_node: int) -> None:
    """Resolve a probe that raises on global rank 0 alone; record what this rank raised."""
    os.environ["LOCAL_RANK"] = str(rank % ranks_per_node)

    def probe() -> bool:
        if rank == 0:
            raise ConnectionError("hub unreachable")
        return False

    try:
        outcome = f"verdict {agree_input_probe_across_ranks(probe, HUB_PATH, 'is_preprocessed_dataset')}"
    except Exception as exc:  # the raise itself is the outcome under test
        outcome = f"{type(exc).__name__}: {exc}"
    with open(os.path.join(tmp_dir, f"rank_{rank}.json"), "w") as f:
        json.dump({"outcome": outcome}, f)


def _uncached_hub_worker(rank: int, tmp_dir: str, offline: bool) -> None:
    """Resolve the real probe against a Hub whose ``metadata.json`` download finds nothing in the cache,
    which huggingface_hub raises online when the Hub is unreachable and offline when the file is uncached."""
    requests = []

    def _download(repo_id, filename, repo_type=None, **kwargs):
        requests.append(repo_id)
        raise LocalEntryNotFoundError("cannot find the requested files in the local cache")

    preprocessed_metadata.hf_hub_download = _download
    preprocessed_metadata.is_offline_mode = lambda: offline
    try:
        verdict = agree_input_probe_across_ranks(
            lambda: is_preprocessed_dataset(HUB_PATH), HUB_PATH, "is_preprocessed_dataset"
        )
        outcome = f"verdict {verdict}"
    except Exception as exc:  # the raise itself is the outcome under test
        outcome = f"{type(exc).__name__}: {exc}"
    with open(os.path.join(tmp_dir, f"rank_{rank}.json"), "w") as f:
        json.dump({"requests": len(requests), "outcome": outcome}, f)


def _run(tmp_path, *, shared: bool, ranks_per_node: int, stamped: bool) -> list[dict]:
    (tmp_path / "metadata.json").write_text(json.dumps(PreprocessedDatasetMetadata().to_dict()))
    run_gloo_ranks(
        _worker,
        WORLD_SIZE,
        str(tmp_path),
        ranks_per_node,
        stamped,
        env={"DIST_SHARED_FILESYSTEM": "1" if shared else "0", "HF_DATASETS_CACHE": str(tmp_path / "hf")},
    )
    return [json.loads((tmp_path / f"rank_{rank}.json").read_text()) for rank in range(WORLD_SIZE)]


@pytest.mark.parametrize(
    ("shared", "ranks_per_node", "expected_requests"),
    [(True, WORLD_SIZE, 1), (True, 2, 1), (False, 2, 2), (False, 1, WORLD_SIZE)],
    ids=["shared-one-node", "shared-two-nodes", "per-node-two-nodes", "per-node-one-rank-each"],
)
def test_hub_probe_runs_once_per_filesystem_scope_and_every_rank_follows_it(
    tmp_path, shared, ranks_per_node, expected_requests
):
    """A shared input filesystem is one scope (global rank 0 asks); per-node storage makes each node
    its own (its local rank 0 asks). Every rank, including the ones that never asked, must take the
    preprocessed branch — whose Hub refusal is the observable here."""
    results = _run(tmp_path, shared=shared, ranks_per_node=ranks_per_node, stamped=True)

    assert sum(r["requests"] for r in results) == expected_requests, results
    assert [r["outcome"] for r in results] == ["hub refusal"] * WORLD_SIZE, results


def test_a_raw_hub_dataset_reads_raw_on_every_rank_from_one_request(tmp_path):
    """The common case — a hub dataset without a stamp — must agree on False from the single probe,
    not leave the abstaining ranks on a verdict they never read."""
    results = _run(tmp_path, shared=True, ranks_per_node=WORLD_SIZE, stamped=False)

    assert sum(r["requests"] for r in results) == 1, results
    assert [r["outcome"] for r in results] == ["verdict False"] * WORLD_SIZE, results


@pytest.mark.parametrize(("shared", "ranks_per_node"), [(True, 2), (False, 2)], ids=["shared", "per-node"])
def test_a_probe_raising_on_a_probing_rank_raises_its_cause_on_every_rank(tmp_path, shared, ranks_per_node):
    """Rank 0 probes for its scope and raises: under a shared declaration its peers only abstain, and
    per-node rank 2 probes its own node cleanly. Every rank must still raise naming rank 0's cause."""
    run_gloo_ranks(
        _raising_probe_worker,
        WORLD_SIZE,
        str(tmp_path),
        ranks_per_node,
        pg_timeout=PG_TIMEOUT,
        env={"DIST_SHARED_FILESYSTEM": "1" if shared else "0"},
    )
    outcomes = [json.loads((tmp_path / f"rank_{rank}.json").read_text())["outcome"] for rank in range(WORLD_SIZE)]

    assert outcomes[0] == "ConnectionError: hub unreachable", outcomes
    for rank in range(1, WORLD_SIZE):
        assert outcomes[rank] == (
            f"RuntimeError: Probing {HUB_PATH} (is_preprocessed_dataset) failed on 1 of {WORLD_SIZE} rank(s) [0]. "
            "First (rank 0): ConnectionError: hub unreachable"
        ), f"rank {rank}: {outcomes[rank]}"


def test_an_unreachable_hub_raises_on_every_rank_instead_of_reading_raw(tmp_path):
    """Only rank 0 asks, so a transient Hub failure there is the whole job's answer: read as absence,
    every rank would take the raw path for a dataset that may be preprocessed."""
    run_gloo_ranks(
        _uncached_hub_worker,
        WORLD_SIZE,
        str(tmp_path),
        False,
        pg_timeout=PG_TIMEOUT,
        env={"DIST_SHARED_FILESYSTEM": "1"},
    )
    results = [json.loads((tmp_path / f"rank_{rank}.json").read_text()) for rank in range(WORLD_SIZE)]

    assert sum(r["requests"] for r in results) == 1, results
    for rank, result in enumerate(results):
        assert not result["outcome"].startswith("verdict"), f"rank {rank} read an unreachable Hub as raw: {result}"
        assert "The Hub could not be reached" in result["outcome"], f"rank {rank}: {result}"
        assert "HF_HUB_OFFLINE=1" in result["outcome"], f"rank {rank}: {result}"


def test_an_offline_cache_without_the_stamp_reads_raw_on_every_rank(tmp_path):
    """Offline, the cache stands for the repo: a raw dataset cached without a ``metadata.json`` must still
    train, so the same cache miss is a clean absence there."""
    run_gloo_ranks(
        _uncached_hub_worker,
        WORLD_SIZE,
        str(tmp_path),
        True,
        pg_timeout=PG_TIMEOUT,
        env={"DIST_SHARED_FILESYSTEM": "1"},
    )
    outcomes = [json.loads((tmp_path / f"rank_{rank}.json").read_text())["outcome"] for rank in range(WORLD_SIZE)]

    assert outcomes == ["verdict False"] * WORLD_SIZE, outcomes


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
