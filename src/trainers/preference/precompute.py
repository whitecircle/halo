"""Rank-consistent reference-log-prob precompute cache for distributed preference trainers.

TRL's ``precompute_ref_log_probs`` keys its disk cache on
``Hasher.hash((dataset._fingerprint, hash_module(model)))``. Both inputs diverge across ranks:
``hash_module`` hashes rank-sharded parameter values under EP/TP, and datasets' ``map`` emits a
random per-process ``_fingerprint`` when it cannot hash the transform. TRL writes the file on the
main process only and has every rank read it back, so with a divergent key the non-main ranks block
on a path that was never written and the next collective forward deadlocks.

The cached values are the full gathered log-probs, so one rank-0-authoritative cache is correct: this
mixin broadcasts both key inputs from rank 0 for the duration of the precompute call.

TRL's sweep also shards its loader and gathers by global rank, which is incorrect whenever DP <
world_size: the model already carries its TP attention shards and EP/ETP expert wrappers, so siblings
forwarding different rows hit the TP and ``ReduceFromExpertTP`` collectives with mismatched token
counts. ``data_parallel_sweep`` puts both ends back on the DP axis.
"""

from __future__ import annotations

import contextlib
import sys
from types import ModuleType

from accelerate.logging import get_logger

from src.distributed.runtime import broadcast_from_rank0

logger = get_logger(__name__, log_level="info")

_METHOD = "_precompute_ref_logps"


def _defining_module(instance: object) -> ModuleType | None:
    """Module of the first MRO class below this mixin that defines ``_precompute_ref_logps``.

    That is the concrete TRL trainer (``DPOTrainer`` / ``KTOTrainer``), whose module defines the
    ``hash_module`` symbol used to build the cache key. Classes above the mixin — the mixin itself
    and any trainer subclass layering more logic over it — are skipped: their modules define no
    ``hash_module``, and matching one would disable the rank-consistent patch.
    """
    for klass in type(instance).__mro__:
        if issubclass(klass, PrecomputeRefLogpsRankConsistentMixin):
            continue
        if _METHOD in klass.__dict__:
            return sys.modules.get(klass.__module__)
    return None


@contextlib.contextmanager
def _rank0_authoritative_module_hash(module: ModuleType | None):
    """Temporarily replace ``module.hash_module`` so it returns rank 0's value on every rank.

    ``hash_module`` hashes parameter values; under EP/TP those are rank-sharded, so the raw hash
    diverges. The hash only keys the precompute cache (detecting a changed model across runs), so
    rank 0's value is a correct, rank-consistent representative.
    """
    original = getattr(module, "hash_module", None) if module is not None else None
    if original is None:
        yield
        return

    def _consistent(mod):
        return broadcast_from_rank0(original(mod))

    module.hash_module = _consistent
    try:
        yield
    finally:
        module.hash_module = original


class PrecomputeRefLogpsRankConsistentMixin:
    """Make TRL's ``precompute_ref_log_probs`` disk-cache path identical across ranks.

    Without it, EP/TP full-finetune DPO/KTO deadlock on the first collective forward after the main
    process writes a cache file that no other rank's (divergent) path reads. See the module
    docstring for the mechanism. It also implements the skip for a dataset that already carries the
    reference columns (``_required_ref_logps_columns``), which PP relies on.
    """

    def _required_ref_logps_columns(self) -> tuple[str, ...]:
        """Reference log-prob columns whose presence makes TRL's sweep unnecessary.

        Each subclass names the columns its own TRL base would write; an empty default would
        disable the short-circuit and leave PP without a reference, since a pipeline stage cannot
        run the sweep.
        """
        raise NotImplementedError(f"{type(self).__name__} must name its reference log-prob columns")

    def _precompute_ref_logps(self, dataset, name, batch_size):
        """Trust dataset-supplied reference log-probs; run TRL's sweep only when one is absent.

        With every needed column present the sweep would recompute values the caller supplied (then
        axis-1 concatenate duplicate columns), and under PP it cannot run at all — ``self.model`` is
        a bare pipeline stage. The check precedes the pre-sharded rejection below, whose suggested
        workaround is to precompute these columns before sharding. The PP construction gate
        guarantees the columns exist, so the sweep only runs outside PP.
        """
        needed = self._required_ref_logps_columns()
        if all(column in (dataset.column_names or []) for column in needed):
            logger.info(f"Dataset '{name}' already carries {needed}; skipping the reference sweep.")
            return dataset
        if self._dataset_presharded:
            raise ValueError(
                "precompute_ref_log_probs=True is not supported with a pre-sharded dataset: each "
                "rank holds a DIFFERENT shard, but TRL's sweep caches one rank-0-authoritative file "
                "of reference log-probs and has every rank concatenate it onto its own rows — so "
                "every non-zero data-parallel rank would train on rank 0's log-probs. Load the "
                "dataset unsharded, or precompute the ref_chosen_logps/ref_rejected_logps "
                "(ref_logps for KTO) columns into the dataset before sharding it."
            )
        # Tokenization is deterministic, so only the fingerprint diverges — pinning it is safe.
        dataset._fingerprint = broadcast_from_rank0(dataset._fingerprint)
        with _rank0_authoritative_module_hash(_defining_module(self)), self.data_parallel_sweep():
            return super()._precompute_ref_logps(dataset, name, batch_size)
