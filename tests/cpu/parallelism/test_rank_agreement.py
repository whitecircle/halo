#!/usr/bin/env python
"""Cross-rank agreement and rejection are one fixed-size all-reduce on a real 3-rank gloo world.

``agree_across_ranks`` compares a canonical blake2b digest with one int64 MAX all-reduce of
``[d, -d, -rank, abstained]``, so the common case of a check that runs every log step, or once per
setting at startup, costs nothing proportional to the world: no object gather, no per-rank
unpickle. Only a disagreement gathers the objects, to name them. The pins:

* agreement issues exactly one all-reduce and no object collective, for ``reject_divergent_settings``
  and for the stored-metric key check alike;
* the digest is canonical across processes (a mapping's insertion order does not matter, and
  Python's per-process ``hash`` salt never enters);
* a disagreement still raises on every rank with the caller's message;
* an abstaining rank leaves the verdict unchanged and learns which rank to read the object from;
* ``reject_across_ranks`` (and so every ``DeferredRankFailure.reject``) with no failure anywhere is
  one all-reduce of a has-reason flag and no object gather, and one rank's failure still raises on
  every rank with the gathered message.

    python tests/cpu/parallelism/test_rank_agreement.py
"""

import datetime
import json
from contextlib import contextmanager

import pytest
import torch.distributed as dist

from src.distributed.runtime import (
    DeferredRankFailure,
    agree_across_ranks,
    reject_across_ranks,
    reject_divergent_settings,
)
from src.trainers.mixins.stored_metrics import StoredMetricsMixin
from tests.common.gloo import run_gloo_ranks

WORLD = 3


@contextmanager
def _counted_collectives(counts: dict[str, int]):
    """Count the collectives the block issues, by name, delegating to the real ones."""
    names = ("all_reduce", "all_gather_object", "broadcast_object_list")
    originals = {name: getattr(dist, name) for name in names}

    def counting(name):
        def call(*args, **kwargs):
            counts[name] = counts.get(name, 0) + 1
            return originals[name](*args, **kwargs)

        return call

    for name in names:
        setattr(dist, name, counting(name))
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(dist, name, original)


class _Recorder:
    def log(self, logs, start_time=None):
        return logs


class _Trainer(StoredMetricsMixin, _Recorder):
    pass


def _worker(rank: int, out: str) -> None:
    results = {}

    counts: dict[str, int] = {}
    with _counted_collectives(counts):
        # Built in a different insertion order on every rank: the digest must not see it.
        items = [("b", 2), ("a", 1), ("c", (3, None))]
        agreement = agree_across_ranks(dict(items[rank:] + items[:rank]))
    results["agree"] = [agreement.agreed, agreement.first_present, agreement.any_abstained, counts]

    counts = {}
    with _counted_collectives(counts):
        reject_divergent_settings({"HALO_GRAD_BUCKET_MB": 256}, "Toolkit environment", "Align it.")
    results["settings_fast_path"] = counts

    try:
        reject_divergent_settings(
            {"HALO_GRAD_BUCKET_MB": 64 if rank == 2 else 256}, "Toolkit environment", "Align it."
        )
        results["settings_mismatch"] = "agreed"
    except ValueError as exc:
        results["settings_mismatch"] = str(exc)

    # Equal under == but spelled differently per rank: the digest misses, the gathered comparison agrees.
    for name, values in {
        "set": {"names": {f"layer_{i}.mlp" for i in range(16)}},
        "nested_dict": {"cfg": dict([("a", 1), ("b", 2)][:: 1 if rank == 0 else -1])},
        "int_and_float": {"timeout": 30 if rank == 0 else 30.0},
        "none_and_absent": {"x": None} if rank == 0 else {},
    }.items():
        try:
            reject_divergent_settings(values, f"Equal {name}", "Align it.")
            results[f"equal_{name}"] = "agreed"
        except ValueError as exc:
            results[f"equal_{name}"] = str(exc)

    # Rank 2's mapping mixes a str key with an int one, which the canonical digest cannot sort.
    try:
        reject_divergent_settings({"x": 1, 2: 3} if rank == 2 else {"x": 1}, "Mixed-key settings", "Align it.")
        results["undigestible"] = "agreed"
    except ValueError as exc:
        results["undigestible"] = str(exc)
    # Every rank undigestible, the values still differing: no shared fallback digest may agree them.
    try:
        reject_divergent_settings({"x": 1, 2: rank}, "All mixed-key settings", "Align it.")
        results["all_undigestible"] = "agreed"
    except ValueError as exc:
        results["all_undigestible"] = str(exc)

    agreement = agree_across_ranks(["m"], abstain=rank == 0)
    results["abstain"] = [agreement.agreed, agreement.first_present, agreement.any_abstained]
    agreement = agree_across_ranks([], abstain=True)
    results["all_abstain"] = [agreement.agreed, agreement.first_present, agreement.any_abstained]

    counts = {}
    with _counted_collectives(counts):
        reject_across_ranks(None, "checkpoint write")
        guard = DeferredRankFailure("checkpoint finalize")
        guard.run(lambda: None)
        guard.reject()
    results["reject_fast_path"] = counts

    try:
        reject_across_ranks("disk full" if rank == 1 else None, "checkpoint write")
        results["reject_one_rank"] = "passed"
    except RuntimeError as exc:
        results["reject_one_rank"] = str(exc)

    counts = {}
    trainer = _Trainer()
    trainer.store_metrics({"m": float(rank)})
    with _counted_collectives(counts):
        trainer.log({"loss": 0.1})
    results["log_fast_path"] = counts

    with open(f"{out}.{rank}", "w") as fh:
        json.dump(results, fh)


@pytest.fixture(scope="module")
def per_rank(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("agreement") / "results")
    run_gloo_ranks(_worker, WORLD, out, pg_timeout=datetime.timedelta(seconds=60))
    results = []
    for rank in range(WORLD):
        with open(f"{out}.{rank}") as fh:
            results.append(json.load(fh))
    return results


def test_agreement_is_one_all_reduce_whatever_the_insertion_order(per_rank):
    for rank, results in enumerate(per_rank):
        assert results["agree"] == [True, 0, False, {"all_reduce": 1}], f"rank {rank}: {results['agree']}"


def test_the_settings_check_agrees_without_an_object_collective(per_rank):
    for rank, results in enumerate(per_rank):
        assert results["settings_fast_path"] == {"all_reduce": 1}, f"rank {rank}"


def test_a_settings_mismatch_raises_on_every_rank_with_the_callers_message(per_rank):
    for rank, results in enumerate(per_rank):
        message = results["settings_mismatch"]
        assert "Toolkit environment differs across ranks" in message, f"rank {rank}: {message}"
        assert "Align it." in message and "rank 0 has {'HALO_GRAD_BUCKET_MB': 256}" in message, message


@pytest.mark.parametrize("name", ["set", "nested_dict", "int_and_float", "none_and_absent"])
def test_values_equal_under_eq_but_spelled_differently_agree(per_rank, name):
    """The digest only fast-paths agreement; a set iterated in salted order, a dict built in another
    order, ``30`` beside ``30.0`` and ``None`` beside an absent key are the same setting."""
    for rank, results in enumerate(per_rank):
        assert results[f"equal_{name}"] == "agreed", f"rank {rank}: {results[f'equal_{name}']}"


def test_an_object_one_rank_cannot_digest_reaches_the_gather_and_is_named(per_rank):
    """A raise inside the digest on rank 2 alone would leave its peers in the agreement reduce."""
    for rank, results in enumerate(per_rank):
        assert "Mixed-key settings differs across ranks: {2: ['3', 'None']}" in results["undigestible"], (
            f"rank {rank}: {results['undigestible']}"
        )
        assert "All mixed-key settings differs across ranks: {2: ['0', '1', '2']}" in results["all_undigestible"], (
            f"rank {rank}: {results['all_undigestible']}"
        )


def test_an_abstaining_rank_leaves_the_verdict_and_names_where_to_read(per_rank):
    for rank, results in enumerate(per_rank):
        assert results["abstain"] == [True, 1, True], f"rank {rank}: {results['abstain']}"
        assert results["all_abstain"] == [True, None, True], f"rank {rank}: {results['all_abstain']}"


def test_the_stored_metric_log_is_two_all_reduces_and_no_object_collective(per_rank):
    """One for the key agreement, one for the sums — the cost of a logging step at any world size."""
    for rank, results in enumerate(per_rank):
        assert results["log_fast_path"] == {"all_reduce": 2}, f"rank {rank}: {results['log_fast_path']}"


def test_a_rejection_nobody_needs_is_one_all_reduce(per_rank):
    """Two rejections here (a bare one and a deferred guard's), so two all-reduces and no gather."""
    for rank, results in enumerate(per_rank):
        assert results["reject_fast_path"] == {"all_reduce": 2}, f"rank {rank}: {results['reject_fast_path']}"


def test_one_ranks_failure_raises_on_every_rank_with_the_gathered_message(per_rank):
    for rank, results in enumerate(per_rank):
        assert results["reject_one_rank"] == (
            f"checkpoint write failed on 1 of {WORLD} rank(s) [1]. First (rank 1): disk full"
        ), f"rank {rank}: {results['reject_one_rank']}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
