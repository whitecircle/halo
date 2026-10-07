#!/usr/bin/env python
"""A launch's offline GRPO reference scratch is removed only once no live launch holds it.

Every rank of a launch maps its scores from ``output_dir/_reference_cache/<launch>/`` for the whole run,
and on NFS a removed file stays readable only on the host that removed it. Another job on the same
``output_dir`` — a separate evaluation, a second launch — must therefore never remove a live launch's
scratch: each launch's writer holds a lock on ``<launch>.lock`` while it runs, and a launch removes only
the scratch whose lock it can take. Where the mount takes no locks, nothing is removed.

Separate processes, as separate launches are: a lock one process holds is what another must respect.

    python tests/cpu/grpo/test_reference_scratch_launch_locks.py
"""

import errno
import fcntl
import multiprocessing
import os

import pytest

import src.trainers.grpo.reference_cache as cache_module
from src.checkpoint.format import REFERENCE_CACHE_DIR_NAME

# Bounds every wait on a spawned launch: spawn and import, then the step it signals.
LAUNCH_TIMEOUT_S = 180.0


def _launch(output_dir: str, created, release=None, locks_supported: bool = True) -> None:
    """One launch: create its first cache, record a mapped-file stand-in, report, then run until released."""
    if not locks_supported:

        def no_locks(descriptor, operation):
            raise OSError(errno.ENOLCK, "No locks available")

        fcntl.flock = no_locks
    directory = cache_module._create_launch_cache_directory(output_dir)
    with open(os.path.join(directory, "merged.values"), "wb") as fh:
        fh.write(b"scores a rank maps for the whole launch")
    created.put(directory)
    if release is not None and not release.wait(LAUNCH_TIMEOUT_S):
        raise TimeoutError("the test never released this launch")


def _run_launch(context, output_dir: str, *, locks_supported: bool = True) -> str:
    """A launch that creates its cache and exits; returns its cache directory."""
    created = context.Queue()
    process = context.Process(target=_launch, args=(output_dir, created, None, locks_supported))
    process.start()
    directory = created.get(timeout=LAUNCH_TIMEOUT_S)
    process.join(LAUNCH_TIMEOUT_S)
    assert process.exitcode == 0, f"launch exited {process.exitcode}"
    return directory


def _launch_dir(cache_directory: str) -> str:
    return os.path.dirname(cache_directory)


def test_a_live_launchs_scratch_survives_another_launch_and_goes_once_it_exits(tmp_path):
    context = multiprocessing.get_context("spawn")
    output_dir = str(tmp_path)
    created, release = context.Queue(), context.Event()
    live = context.Process(target=_launch, args=(output_dir, created, release))
    live.start()
    try:
        live_cache = created.get(timeout=LAUNCH_TIMEOUT_S)
        concurrent_cache = _run_launch(context, output_dir)

        assert os.path.isfile(os.path.join(live_cache, "merged.values")), (
            "a concurrent launch removed the scratch a live launch still maps"
        )
        assert os.path.isdir(concurrent_cache)
    finally:
        release.set()
        live.join(LAUNCH_TIMEOUT_S)
    assert live.exitcode == 0, f"the live launch exited {live.exitcode}"

    next_cache = _run_launch(context, output_dir)

    remaining = sorted(os.listdir(tmp_path / REFERENCE_CACHE_DIR_NAME))
    next_launch = os.path.basename(_launch_dir(next_cache))
    assert remaining == [next_launch, f"{next_launch}.lock"], (
        f"finished launches' scratch or locks outlived them: {remaining}"
    )


def test_a_mount_without_locks_keeps_every_other_launchs_scratch(tmp_path):
    context = multiprocessing.get_context("spawn")
    output_dir = str(tmp_path)
    finished_cache = _run_launch(context, output_dir)

    _run_launch(context, output_dir, locks_supported=False)

    assert os.path.isfile(os.path.join(finished_cache, "merged.values")), (
        "a launch that could not prove the other launch finished removed its scratch"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
