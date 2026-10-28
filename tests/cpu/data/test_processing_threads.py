#!/usr/bin/env python
"""Dataset map worker counts: HALO_DATASET_NUM_PROC resolution and the share a follower rank maps with.

Unset, the worker count comes from the CPUs the process may use: its affinity set, not the host count a container
cpuset hides, capped by the cgroup CPU quotas over the process, which the affinity set does not show. The load rank
runs a coordinated map alone while its node's other ranks wait, so it takes most of them, capped. Those ranks compute
only on a per-rank key (a pre-sharded load), all at once, so each maps with its share of the count; on a shared key
they read the load rank's cache back at their own count and spawn nothing.
Falsy-override guard: applying the override as ``env_int(...) or derived`` makes an explicit ``0`` falsy and
silently replaces it with the derived value. The resolver fails loud on an override below 1 (``num_proc``
counts worker processes; 1 disables multiprocessing) and honors any valid explicit value.

Run: pytest tests/cpu/data/test_processing_threads.py
"""

import os

import multiprocess
import pytest
from datasets import Dataset

from halo import hardware
from halo.data.pipeline import coordinated_map
from halo.data.pipeline.coordinated_map import _resolve_dataset_num_proc

# Set while a map must be served from the load rank's cache: a recompute raises, in a forked worker too.
_FORBID_RECOMPUTE = "TEST_FORBID_MAP_RECOMPUTE"

_ROWS = Dataset.from_dict({"x": list(range(16))})

# A private cgroup namespace: the process's cgroup is the mount's root.
_V2_PRIVATE = "0::/\n"

# A host cgroup namespace: the mount shows the whole hierarchy, the container's cgroup deep inside it.
_V2_HOST = "0::/system.slice/docker-c0.scope\n"

# cgroup v1 Docker: each controller's mount is the container's own cgroup, which /proc/self/cgroup names by host path.
_V1_DOCKER = "12:cpuset:/docker/c0\n4:cpu,cpuacct:/docker/c0\n1:name=systemd:/docker/c0\n"


def _double(row):
    if os.environ.get(_FORBID_RECOMPUTE):
        raise AssertionError("the map ran again instead of reading the load rank's cache")
    return {"x": 2 * row["x"]}


@pytest.fixture(autouse=True)
def _clear_pin(monkeypatch):
    """Unset unless a test sets it."""
    monkeypatch.delenv("HALO_DATASET_NUM_PROC", raising=False)


def _cgroups(monkeypatch, root, files: dict[str, str], membership: str = _V2_PRIVATE) -> None:
    """Serve the CPU count's cgroup reads from ``root``: ``membership`` as ``/proc/self/cgroup`` and ``files`` under
    the cgroup mount, where none means no quota."""
    mount = root / "sys-fs-cgroup"
    for relative_path, content in files.items():
        path = mount / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (root / "proc-self-cgroup").write_text(membership)
    monkeypatch.setattr(hardware, "_CGROUP_ROOT", str(mount))
    monkeypatch.setattr(hardware, "_PROC_SELF_CGROUP", str(root / "proc-self-cgroup"))


@pytest.mark.parametrize(("usable_cpus", "expected"), [(192, 64), (64, 48), (8, 6), (1, 1)])
def test_unset_takes_most_of_the_cpus_the_process_may_use(monkeypatch, tmp_path, usable_cpus, expected):
    monkeypatch.setattr(os, "cpu_count", lambda: 384)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(usable_cpus)))
    _cgroups(monkeypatch, tmp_path, {})
    assert _resolve_dataset_num_proc() == expected


@pytest.mark.parametrize(
    ("usable_cpus", "membership", "cgroup", "expected"),
    [
        (192, _V2_PRIVATE, {"cpu.max": "800000 100000\n"}, 6),
        (192, _V2_PRIVATE, {"cpu.max": "250000 100000\n"}, 2),
        (8, _V2_PRIVATE, {"cpu.max": "1600000 100000\n"}, 6),
        (192, _V2_PRIVATE, {"cpu.max": "max 100000\n"}, 64),
        (
            192,
            _V2_HOST,
            {"system.slice/cpu.max": "max 100000\n", "system.slice/docker-c0.scope/cpu.max": "400000 100000\n"},
            3,
        ),
        (
            192,
            "0::/kubepods/pod0/c0\n",
            {"kubepods/pod0/cpu.max": "800000 100000\n", "kubepods/pod0/c0/cpu.max": "1600000 100000\n"},
            6,
        ),
        (192, _V2_HOST, {"system.slice/docker-c1.scope/cpu.max": "100000 100000\n"}, 64),
        (192, _V1_DOCKER, {"cpu/cpu.cfs_quota_us": "1600000\n", "cpu/cpu.cfs_period_us": "100000\n"}, 12),
        (
            192,
            _V1_DOCKER,
            {"cpu/docker/c0/cpu.cfs_quota_us": "400000\n", "cpu/docker/c0/cpu.cfs_period_us": "100000\n"},
            3,
        ),
        (192, _V1_DOCKER, {"cpu/cpu.cfs_quota_us": "-1\n", "cpu/cpu.cfs_period_us": "100000\n"}, 64),
    ],
    ids=[
        "v2-quota",
        "v2-fractional-quota-rounds-up",
        "v2-quota-above-the-affinity-set",
        "v2-unlimited",
        "v2-host-cgroup-namespace",
        "v2-ancestor-quota-binds-a-looser-child",
        "v2-a-sibling-quota-does-not-apply",
        "v1-mount-of-the-own-cgroup",
        "v1-host-hierarchy",
        "v1-unlimited",
    ],
)
def test_unset_takes_most_of_the_tightest_cgroup_cpu_quota(
    monkeypatch, tmp_path, usable_cpus, membership, cgroup, expected
):
    """``docker --cpus``, a Kubernetes CPU limit or a systemd ``CPUQuota`` leaves the affinity set whole: sized off it
    alone, a map runs 64 workers on the quota's few CPUs. The quota sits on the process's own cgroup or an ancestor,
    and a host cgroup namespace mounts the whole hierarchy, so reading the mount's root alone misses it."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(usable_cpus)))
    _cgroups(monkeypatch, tmp_path, cgroup, membership)
    assert _resolve_dataset_num_proc() == expected


def test_explicit_override_wins_over_derived_value(monkeypatch):
    """The override is honored verbatim, including values above the derived cap."""
    monkeypatch.setenv("HALO_DATASET_NUM_PROC", "97")
    assert _resolve_dataset_num_proc() == 97


def test_pin_of_one_disables_multiprocessing_but_is_valid(monkeypatch):
    monkeypatch.setenv("HALO_DATASET_NUM_PROC", "1")
    assert _resolve_dataset_num_proc() == 1


def test_zero_pin_fails_loud_not_silently_ignored(monkeypatch):
    """An explicit 0 is falsy → a truthiness fallback silently swaps in the derived value; it must raise."""
    monkeypatch.setenv("HALO_DATASET_NUM_PROC", "0")
    with pytest.raises(ValueError, match="HALO_DATASET_NUM_PROC"):
        _resolve_dataset_num_proc()


def test_negative_pin_fails_loud(monkeypatch):
    monkeypatch.setenv("HALO_DATASET_NUM_PROC", "-2")
    with pytest.raises(ValueError, match="must be >= 1"):
        _resolve_dataset_num_proc()


def _map_as(monkeypatch, *, load_rank: bool, local_world_size: int, num_proc: int) -> list:
    """Run a coordinated map over :data:`_ROWS` as the load rank or a follower; the ``num_proc`` each
    ``Dataset.map`` call was handed."""
    monkeypatch.setattr(coordinated_map, "is_fs_load_rank", lambda: load_rank)
    monkeypatch.setattr(coordinated_map, "get_local_world_size", lambda: local_world_size)
    handed, real_map = [], Dataset.map

    def _recording_map(self, function, *args, num_proc=None, **kwargs):
        handed.append(num_proc)
        return real_map(self, function, *args, num_proc=num_proc, **kwargs)

    monkeypatch.setattr(Dataset, "map", _recording_map)
    mapped = coordinated_map.coordinated_map(_ROWS, _double, desc="worker share", num_proc=num_proc)
    assert mapped["x"] == [2 * x for x in _ROWS["x"]]
    monkeypatch.setattr(Dataset, "map", real_map)
    return handed


@pytest.mark.parametrize(
    ("load_rank", "local_world_size", "handed"),
    [(True, 4, 4), (False, 2, 2), (False, 8, None)],
    ids=["load-rank-takes-the-count", "follower-takes-its-share", "a-share-below-two-runs-in-process"],
)
def test_a_follower_maps_with_its_share_of_the_node(monkeypatch, tmp_path, load_rank, local_world_size, handed):
    """A follower computes only on a per-rank key, when every follower of the node maps its own shard at once:
    at the whole count each, a node of 8 ranks would run 8x the workers the count names."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path))
    assert _map_as(monkeypatch, load_rank=load_rank, local_world_size=local_world_size, num_proc=4) == [handed]


def test_a_follower_reads_the_load_ranks_cache_at_its_own_count_without_workers(monkeypatch, tmp_path):
    """The premise of the share: on a shared key the follower's smaller count still finds the load rank's
    per-worker cache files, so it neither recomputes nor spawns a worker."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path))
    _map_as(monkeypatch, load_rank=True, local_world_size=2, num_proc=4)
    monkeypatch.setenv(_FORBID_RECOMPUTE, "1")
    monkeypatch.setattr(multiprocess, "Pool", lambda *a, **k: pytest.fail("a cache hit spawned map workers"))
    assert _map_as(monkeypatch, load_rank=False, local_world_size=2, num_proc=4) == [2]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
