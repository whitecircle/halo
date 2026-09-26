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

With no separate reference model TRL sweeps the policy, and a Path-B resume builds the policy from
the checkpoint before the trainer exists, so a sweep there would score the trained weights. The swept
columns therefore ride every checkpoint (``REFERENCE_LOGPS_FILE``), and a resume attaches them in
place of the sweep once their row count and token digest match the dataset.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import sys
from collections.abc import Mapping, Sequence
from functools import partial
from types import ModuleType

import numpy as np
import pyarrow as pa
import torch
from accelerate.logging import get_logger
from datasets import Dataset, concatenate_datasets

from src.checkpoint.format import REFERENCE_LOGPS_FILE
from src.distributed.checkpoint.coordination import consensus_read
from src.distributed.runtime import (
    barrier_on_exit,
    broadcast_from_rank0,
    fs_aware_save_rank,
    rank_consensus,
    reject_across_ranks,
)

logger = get_logger(__name__, log_level="info")

_METHOD = "_precompute_ref_logps"
# Rows per Arrow batch the token digest reads: bounds the int64 copy of one batch's token ids.
_DIGEST_BATCH_ROWS = 4096


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


def _is_token_type(arrow_type: pa.DataType) -> bool:
    """Token ids and labels: integer or bool values, bare or in one list level."""
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        arrow_type = arrow_type.value_type
    return pa.types.is_integer(arrow_type) or pa.types.is_boolean(arrow_type)


def _token_digest(dataset: Dataset, columns: Sequence[str]) -> str:
    """SHA-256 over ``columns`` in row order, list lengths included and every value widened to int64.

    Content-derived, where ``dataset._fingerprint`` is not: TRL's tokenize map closes over the
    trainer, so the fingerprint hashes trainer state and turns random when that cannot be pickled.
    """
    digest = hashlib.sha256()
    for column in columns:
        digest.update(column.encode())
        for batch in dataset.select_columns([column]).with_format("arrow").iter(batch_size=_DIGEST_BATCH_ROWS):
            values = batch.column(column).combine_chunks()
            if pa.types.is_list(values.type) or pa.types.is_large_list(values.type):
                digest.update(values.value_lengths().to_numpy(zero_copy_only=False).astype(np.int64).tobytes())
                values = values.flatten()
            digest.update(values.to_numpy(zero_copy_only=False).astype(np.int64).tobytes())
    return digest.hexdigest()


def _attach_reference_columns(dataset: Dataset, columns: Mapping[str, torch.Tensor]) -> Dataset:
    """``dataset`` with per-row reference log-prob columns appended, held in memory on this rank."""
    appended = Dataset.from_dict({name: values.numpy() for name, values in columns.items()})
    return concatenate_datasets([dataset, appended], axis=1)


def _saved_split_mismatch(entry: dict, num_rows: int, token_digest: str, needed: Sequence[str]) -> str | None:
    """Why a saved split cannot serve this dataset, or ``None`` when it can."""
    missing = [column for column in needed if column not in entry["columns"]]
    if missing:
        return f"it lacks {missing}, carrying only {sorted(entry['columns'])} (a changed loss_type?)"
    if entry["num_rows"] != num_rows:
        return f"it was saved for {entry['num_rows']} rows and this dataset has {num_rows}"
    if entry["token_digest"] != token_digest:
        return (
            "this dataset's token ids differ from the saved run's (a changed dataset, split, chat "
            "template or tokenizer — or, for KTO's KL completions, per_device_train_batch_size)"
        )
    return None


class PrecomputeRefLogpsRankConsistentMixin:
    """Make TRL's ``precompute_ref_log_probs`` disk-cache path identical across ranks.

    Without it, EP/TP full-finetune DPO/KTO deadlock on the first collective forward after the main
    process writes a cache file that no other rank's (divergent) path reads. See the module
    docstring for the mechanism. It also implements the skip for a dataset that already carries the
    reference columns (``_required_ref_logps_columns``), which PP relies on, and carries the swept
    columns across a resume. A trainer lists it ahead of ``DistributedTrainerMixin`` in its bases,
    so its :meth:`_persist_trainer_sidecars` overrides the checkpointing default.
    """

    def _init_reference_resume(self, kwargs: dict) -> None:
        """Pop the resume context of the precompute TRL's ``__init__`` runs; call before it.

        ``resume_checkpoint`` is the checkpoint the run resumes from and ``policy_from_checkpoint``
        whether the policy's weights were built from it (a Path-B resume), as
        :class:`~src.training.script_runner.ScriptRuntime` resolves them.
        """
        self._reference_resume_checkpoint = kwargs.pop("resume_checkpoint", None)
        self._policy_from_checkpoint = kwargs.pop("policy_from_checkpoint", False)
        if self._policy_from_checkpoint and self._reference_resume_checkpoint is None:
            raise ValueError("policy_from_checkpoint=True needs the resume_checkpoint the policy was built from.")
        self._reference_logps_by_split: dict[str, dict] = {}

    def _required_ref_logps_columns(self) -> tuple[str, ...]:
        """Reference log-prob columns whose presence makes TRL's sweep unnecessary.

        Each subclass names the columns its own TRL base would write; an empty default would
        disable the short-circuit and leave PP without a reference, since a pipeline stage cannot
        run the sweep.
        """
        raise NotImplementedError(f"{type(self).__name__} must name its reference log-prob columns")

    def _precompute_ref_logps(self, dataset, name, batch_size):
        """Trust dataset-supplied reference log-probs, restore a resumed run's, and sweep otherwise.

        With every needed column present the sweep would recompute values the caller supplied (then
        axis-1 concatenate duplicate columns), and under PP it cannot run at all — ``self.model`` is
        a bare pipeline stage. The check precedes the pre-sharded rejection below, whose suggested
        workaround is to precompute these columns before sharding. The PP construction gate
        guarantees the columns exist, so the sweep only runs outside PP.

        Swept and restored columns are kept for :meth:`_persist_trainer_sidecars`; dataset-supplied
        ones are not, since the dataset carries them again on resume.
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
        if name in self._reference_logps_by_split:
            raise ValueError(
                f"Two precomputed datasets share the name '{name}' (an eval_dataset key 'train'?), so "
                f"their reference log-probs would share one {REFERENCE_LOGPS_FILE} entry. Rename it."
            )
        if self._reference_resume_checkpoint is not None:
            restored = self._restore_reference_logps(dataset, name, needed)
            if restored is not None:
                return restored
        # Tokenization is deterministic, so only the fingerprint diverges — pinning it is safe.
        dataset._fingerprint = broadcast_from_rank0(dataset._fingerprint)
        with _rank0_authoritative_module_hash(_defining_module(self)), self.data_parallel_sweep():
            prepared = super()._precompute_ref_logps(dataset, name, batch_size)
        swept = prepared.select_columns(list(needed)).with_format("arrow")[:]
        self._reference_logps_by_split[name] = {
            "num_rows": len(dataset),
            "token_digest": _token_digest(dataset, self._reference_input_columns(dataset, name)),
            "columns": {column: torch.from_numpy(swept.column(column).to_numpy().copy()) for column in needed},
        }
        return prepared

    def _reference_input_columns(self, dataset, name: str) -> list[str]:
        """The token-id and label columns among TRL's signature columns: what the reference read."""
        self._set_signature_columns_if_needed()
        schema = dataset.features.arrow_schema
        columns = sorted(
            column
            for column in self._signature_columns
            if column in dataset.column_names and _is_token_type(schema.field(column).type)
        )
        if not columns:
            raise RuntimeError(
                f"None of TRL's signature columns {self._signature_columns} is a token-id column of "
                f"the '{name}' dataset ({dataset.column_names}), so its reference log-probs could not "
                f"be matched to its rows on resume."
            )
        return columns

    def _restore_reference_logps(self, dataset, name: str, needed: tuple[str, ...]) -> Dataset | None:
        """Attach the ``name`` split's columns saved in the resume checkpoint, or ``None`` to sweep.

        Absent, a sweep is correct only when it scores untrained weights; with no separate reference
        model and the policy built from the checkpoint it would score the trained ones, which raises.
        Present, the saved split must match this dataset's row count and token digest. Every verdict
        is joined across ranks, so the raises and the sweep are taken by the whole world or none.
        """
        checkpoint = self._reference_resume_checkpoint
        saved, path = consensus_read(
            os.path.join(checkpoint, REFERENCE_LOGPS_FILE),
            partial(torch.load, map_location="cpu", weights_only=True),
            what=REFERENCE_LOGPS_FILE,
            checkpoint=checkpoint,
        )
        entry = (saved or {}).get(name)
        present_all, present_any = rank_consensus(entry is not None)
        if present_any and not present_all:
            raise RuntimeError(
                f"{REFERENCE_LOGPS_FILE} at {checkpoint} holds the '{name}' split on some ranks only — "
                f"the nodes' copies differ. Resume from a complete checkpoint."
            )
        if not present_all:
            if self._policy_from_checkpoint and self.ref_model is None:
                raise RuntimeError(
                    f"Cannot resume precompute_ref_log_probs from {checkpoint}: it carries no saved "
                    f"reference log-probs for the '{name}' dataset ({REFERENCE_LOGPS_FILE}), and they "
                    f"cannot be recomputed. With no separate reference model TRL sweeps the policy, "
                    f"and this resume built the policy from the checkpoint, so the sweep would score "
                    f"the TRAINED weights as the reference and zero every log-ratio. Resume from a "
                    f"checkpoint written with that file, supply the {list(needed)} columns computed "
                    f"on the base model in the dataset, or restart from the base model."
                )
            logger.info(
                f"No saved reference log-probs for '{name}' in {checkpoint}; sweeping, since the "
                f"reference weights are not the checkpoint's."
            )
            return None
        num_rows = len(dataset)
        token_digest = _token_digest(dataset, self._reference_input_columns(dataset, name))
        mismatch = _saved_split_mismatch(entry, num_rows, token_digest, needed)
        reject_across_ranks(
            None
            if mismatch is None
            else (
                f"{path} does not belong to this '{name}' dataset: {mismatch}. Each saved value is its "
                f"own row's reference, so attaching them would score rows against other rows' "
                f"references. Resume with the data configuration the checkpoint was written with."
            ),
            f"Restoring the '{name}' reference log-probs",
            exc_type=ValueError,
        )
        columns = {column: entry["columns"][column] for column in needed}
        self._reference_logps_by_split[name] = {"num_rows": num_rows, "token_digest": token_digest, "columns": columns}
        logger.info(f"Restored the '{name}' reference log-probs from {path}; skipping the sweep.")
        return _attach_reference_columns(dataset, columns)

    def _persist_trainer_sidecars(self, checkpoint_dir: str) -> None:
        """Write the swept or restored reference columns into every checkpoint of the run.

        The next resume attaches them rather than sweeping a policy that may by then hold trained
        weights. Written on the FS-aware save rank(s), so each node of a non-shared output
        filesystem holds its own copy, and fenced so a failed write cannot strand the peers.
        """
        super()._persist_trainer_sidecars(checkpoint_dir)
        if not self._reference_logps_by_split:
            return
        with barrier_on_exit():
            if fs_aware_save_rank():
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(self._reference_logps_by_split, os.path.join(checkpoint_dir, REFERENCE_LOGPS_FILE))
