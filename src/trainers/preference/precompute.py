"""Reference-log-prob precompute for the distributed preference trainers (DPO, KTO).

TRL's ``precompute_ref_log_probs`` sweeps the reference once inside ``__init__`` and hands the
columns to every rank through an Arrow cache file the global main process writes and every rank
reads back. Keyed on rank-sharded parameter hashes and random per-process fingerprints, that path
diverges across ranks under EP/TP, and on per-node storage the file exists on one node only. This
mixin runs the same sweep itself and attaches the gathered columns in memory on every rank, which
already holds all of them after the gather, so no rank reads a file another rank wrote.

TRL's sweep also shards its loader and gathers by global rank, which is incorrect whenever DP <
world_size: the model already carries its TP attention shards and EP/ETP expert wrappers, so siblings
forwarding different rows hit the TP and ``ReduceFromExpertTP`` collectives with mismatched token
counts. ``data_parallel_sweep`` puts both ends back on the DP axis.

With no separate reference model the sweep scores the policy, and a Path-B resume builds the policy
from the checkpoint before the trainer exists, so a sweep there would score the trained weights. The
swept columns therefore ride every checkpoint (``REFERENCE_LOGPS_FILE``), and a resume attaches them
in place of the sweep once their row count, token digests and reference settings match.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from functools import partial

import numpy as np
import pyarrow as pa
import torch
from accelerate.logging import get_logger
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.checkpoint.format import REFERENCE_LOGPS_FILE
from src.distributed.checkpoint.coordination import consensus_read
from src.distributed.runtime import barrier_on_exit, fs_aware_save_rank, rank_consensus, reject_across_ranks

logger = get_logger(__name__, log_level="info")

# Rows per Arrow batch the token digest reads, and values per int64 copy it hashes: together they
# bound its transient memory whatever the row length (a 32k-token row is 256 KiB widened).
_DIGEST_BATCH_ROWS = 256
_DIGEST_CHUNK_VALUES = 1 << 22
# What every saved split entry holds, by type; anything else is refused as a mismatch.
_ENTRY_SCHEMA = {"num_rows": int, "token_digests": Mapping, "settings": Mapping, "columns": Mapping}
# The settings key recording the precision the sweep summed a split's log-probs in.
_PRECISION_KEY = "logprob_precision"


def _is_token_type(arrow_type: pa.DataType) -> bool:
    """Token ids and labels: integer or bool values, bare or in one list level."""
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        arrow_type = arrow_type.value_type
    return pa.types.is_integer(arrow_type) or pa.types.is_boolean(arrow_type)


def _token_digest(dataset: Dataset, column: str) -> str:
    """SHA-256 of one column in row order: its list lengths and its values, each widened to int64.

    Content-derived, where ``dataset._fingerprint`` is not: TRL's tokenize map closes over the
    trainer, so the fingerprint hashes trainer state and turns random when that cannot be pickled.
    Lengths and values stream into separate hashes, so neither the batch size nor the chunking
    enters the digest.
    """
    lengths, values = hashlib.sha256(), hashlib.sha256()
    for batch in dataset.select_columns([column]).with_format("arrow").iter(batch_size=_DIGEST_BATCH_ROWS):
        array = batch.column(column).combine_chunks()
        if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
            lengths.update(array.value_lengths().to_numpy(zero_copy_only=False).astype(np.int64).tobytes())
            array = array.flatten()
        for start in range(0, len(array), _DIGEST_CHUNK_VALUES):
            chunk = array.slice(start, _DIGEST_CHUNK_VALUES)
            values.update(chunk.to_numpy(zero_copy_only=False).astype(np.int64).tobytes())
    return hashlib.sha256(lengths.digest() + values.digest()).hexdigest()


def _attach_reference_columns(dataset: Dataset, columns: Mapping[str, torch.Tensor]) -> Dataset:
    """``dataset`` with per-row reference log-prob columns appended as float32, in memory on this rank."""
    appended = Dataset.from_dict({name: values.float().numpy() for name, values in columns.items()})
    return concatenate_datasets([dataset, appended], axis=1)


def _is_saved_split(entry: object) -> bool:
    """Whether a saved entry has every :data:`_ENTRY_SCHEMA` field, each of its type."""
    return isinstance(entry, Mapping) and all(isinstance(entry.get(k), t) for k, t in _ENTRY_SCHEMA.items())


def _saved_split_mismatch(
    entry: object, num_rows: int, token_digests: Mapping[str, str], settings: Mapping, needed: Sequence[str]
) -> str | None:
    """Why a saved split cannot serve this dataset, or ``None`` when it can.

    Total over whatever the file held: a malformed entry is a mismatch, never a raise, since one
    node's bad copy must reach the verdict every rank joins rather than fail on that rank alone.
    """
    if not _is_saved_split(entry):
        return f"its entry is not a saved reference split (expected {sorted(_ENTRY_SCHEMA)})"
    columns = entry["columns"]
    missing = [column for column in needed if column not in columns]
    if missing:
        return f"it lacks {missing}, carrying only {sorted(columns)} (a changed loss_type?)"
    if entry["num_rows"] != num_rows:
        return f"it was saved for {entry['num_rows']} rows and this dataset has {num_rows}"
    malformed = [
        column
        for column in needed
        if not isinstance(columns[column], torch.Tensor) or tuple(columns[column].shape) != (num_rows,)
    ]
    if malformed:
        return f"its {malformed} do not hold one value per row"
    if entry["settings"] != settings:
        return f"it was computed under {entry['settings']} and this run sets {dict(settings)}"
    changed = sorted(
        column for column, digest in token_digests.items() if entry["token_digests"].get(column) != digest
    )
    if changed:
        return (
            f"this dataset's {changed} differ from the saved run's (a changed dataset, split, chat template "
            f"or tokenizer — or, for KTO's KL completions, per_device_train_batch_size or dataset_num_proc)"
        )
    return None


def _regeneration_steps(checkpoint: str) -> str:
    """How to give ``checkpoint`` a reference file computed from the base for this run's data and
    settings: the recovery a resume that cannot sweep names."""
    # Outside the run's own output_dir, whose rotation could otherwise delete this checkpoint.
    scratch = f"{os.path.dirname(os.path.abspath(checkpoint))}-reference-recovery"
    return (
        f"run this config for one step from the base model into a scratch directory "
        f"(--output_dir={scratch} --max_steps=1 --save_strategy=steps --save_steps=1 "
        f"--save_only_model=true --resume_from_checkpoint=null) and copy its "
        f"checkpoint-1/{REFERENCE_LOGPS_FILE} into {checkpoint}, on every node when checkpoints are "
        f"node-local"
    )


def _restore_refusal(path: str, name: str, entry: object, settings: Mapping, mismatch: str, checkpoint: str) -> str:
    """Why a resume whose sweep would score trained weights refuses the saved split, and the remedy.

    A split summed at another log-prob precision cannot be fixed by resuming with the saving run's
    settings, since no setting selects the precision, so that refusal names only the regeneration.
    """
    if _is_saved_split(entry) and entry["settings"].get(_PRECISION_KEY) != settings[_PRECISION_KEY]:
        return (
            f"Regenerate the '{name}' reference log-probs for this run: {_regeneration_steps(checkpoint)}. "
            f"The saved ones in {path} were summed at {entry['settings'].get(_PRECISION_KEY, 'unrecorded')} "
            f"precision, and this run sums them in {settings[_PRECISION_KEY]}, which no setting changes."
        )
    return (
        f"{path} does not belong to this '{name}' dataset: {mismatch}. Each saved value is one row's "
        f"reference under the saving run's data and settings, so attaching them here would score rows "
        f"against references computed otherwise. Resume with the data and reference settings the "
        f"checkpoint was written with, or give it a file computed for this run: "
        f"{_regeneration_steps(checkpoint)}."
    )


class PrecomputeRefLogpsRankConsistentMixin:
    """Run TRL's ``precompute_ref_log_probs`` sweep on the DP axis and attach its columns per rank.

    See the module docstring for the mechanism. It also implements the skip for a dataset that
    already carries the reference columns (``_required_ref_logps_columns``), which PP relies on, and
    carries the swept columns across a resume. A trainer lists it ahead of
    ``DistributedTrainerMixin`` in its bases, so its :meth:`_persist_trainer_sidecars` overrides the
    checkpointing default.
    """

    # Declared by FP32LogprobsMixin; part of every saved split's identity.
    logprob_precision: str | None = None

    def _init_reference_resume(self, kwargs: dict) -> None:
        """Pop the resume context of the precompute TRL's ``__init__`` runs; call before it.

        ``resume_checkpoint`` is the checkpoint the run resumes from and ``policy_from_checkpoint``
        whether the policy's weights were built from it (a Path-B resume), as
        :class:`~src.training.script_runner.ScriptRuntime` resolves them.
        """
        self._reference_resume_given = "resume_checkpoint" in kwargs
        self._reference_resume_checkpoint = kwargs.pop("resume_checkpoint", None)
        self._policy_from_checkpoint = kwargs.pop("policy_from_checkpoint", False)
        if self._policy_from_checkpoint and self._reference_resume_checkpoint is None:
            raise ValueError("policy_from_checkpoint=True needs the resume_checkpoint the policy was built from.")
        self._reference_logps_by_split: dict[str, dict] = {}
        # The resume checkpoint's saved splits, carried into this run's checkpoints for the splits
        # it does not precompute itself (an eval split switched off), so a later resume still has them.
        self._resumed_reference_logps: dict[str, object] = {}

    def _required_ref_logps_columns(self) -> tuple[str, ...]:
        """Reference log-prob columns, in the order TRL's ``compute_ref_log_probs`` returns them.

        Their presence in a dataset makes the sweep unnecessary. Each subclass names the columns its
        own TRL base would write; an empty default would disable the short-circuit and leave PP
        without a reference, since a pipeline stage cannot run the sweep.
        """
        raise NotImplementedError(f"{type(self).__name__} must name its reference log-prob columns")

    def _reference_settings(self) -> dict:
        """The run's knobs, besides the tokens, that shape the reference values TRL computes."""
        raise NotImplementedError(f"{type(self).__name__} must name the settings its reference depends on")

    def _reference_identity(self) -> dict:
        """The settings a saved split records and must match: the run's knobs plus the precision the
        sweep sums its log-probs in."""
        if self.logprob_precision is None:
            raise NotImplementedError(
                f"{type(self).__name__} must declare the logprob_precision its reference sweep sums in "
                f"(list FP32LogprobsMixin in its bases)"
            )
        return {**self._reference_settings(), _PRECISION_KEY: self.logprob_precision}

    def _precompute_ref_logps(self, dataset, name, batch_size):
        """Trust dataset-supplied reference log-probs, restore a resumed run's, and sweep otherwise.

        With every needed column present the sweep would recompute values the caller supplied, and
        under PP it cannot run at all — ``self.model`` is a bare pipeline stage. The check precedes
        the pre-sharded rejection below, whose suggested workaround is to precompute these columns
        before sharding. The PP construction gate guarantees the columns exist, so the sweep only
        runs outside PP.

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
                "rank holds a DIFFERENT shard, but the sweep gathers one set of reference log-probs "
                "in dataset order and every rank attaches it to its own rows — so every non-zero "
                "data-parallel rank would train on rank 0's log-probs. Load the dataset unsharded, or "
                "precompute the ref_chosen_logps/ref_rejected_logps (ref_logps for KTO) columns into "
                "the dataset before sharding it."
            )
        if name in self._reference_logps_by_split:
            raise ValueError(
                f"Two precomputed datasets share the name '{name}' (an eval_dataset key 'train'?), so "
                f"their reference log-probs would share one {REFERENCE_LOGPS_FILE} entry. Rename it."
            )
        if not self._reference_resume_given and self.ref_model is None and self.args.resume_from_checkpoint:
            raise ValueError(
                f"resume_from_checkpoint={self.args.resume_from_checkpoint!r} is set, but the trainer "
                f"was built without resume_checkpoint/policy_from_checkpoint. TRL sweeps the reference "
                f"inside __init__, before train() sees the checkpoint, so a policy loaded from it would "
                f"be scored as its own reference. Pass resume_checkpoint (the resolved checkpoint, or "
                f"None) and policy_from_checkpoint (whether the policy weights came from it)."
            )
        settings = self._reference_identity()
        if self._reference_resume_checkpoint is not None:
            restored = self._restore_reference_logps(dataset, name, needed, settings)
            if restored is not None:
                return restored
        columns = self._sweep_reference_logps(dataset, name, batch_size, needed)
        self._reference_logps_by_split[name] = {
            "num_rows": len(dataset),
            "token_digests": self._reference_input_digests(dataset, name),
            "settings": settings,
            "columns": columns,
        }
        return _attach_reference_columns(dataset, columns)

    def _sweep_reference_logps(self, dataset, name: str, batch_size: int, needed) -> dict[str, torch.Tensor]:
        """TRL's reference sweep over ``dataset``, gathered on the DP axis into dataset order.

        ``compute_ref_log_probs`` returns KTO's KL term as ``None`` when the loss has none, so the
        non-``None`` outputs line up with ``needed``.
        """
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=False,
        )
        parts: dict[str, list[torch.Tensor]] = {column: [] for column in needed}
        with self.data_parallel_sweep():
            for batch in tqdm(self.accelerator.prepare(loader), desc=f"Computing reference log probs for {name}"):
                outputs = tuple(value for value in self.compute_ref_log_probs(batch) if value is not None)
                for column, value in zip(needed, self.accelerator.gather_for_metrics(outputs), strict=True):
                    parts[column].append(value.float().cpu())
        return {column: torch.cat(chunks) for column, chunks in parts.items()}

    def _reference_input_digests(self, dataset, name: str) -> dict[str, str]:
        """Token digest per token-id and label column among TRL's signature columns: what the
        reference read."""
        self._set_signature_columns_if_needed()
        schema = dataset.features.arrow_schema
        columns = [
            column
            for column in self._signature_columns
            if column in dataset.column_names and _is_token_type(schema.field(column).type)
        ]
        if not columns:
            raise RuntimeError(
                f"None of TRL's signature columns {self._signature_columns} is a token-id column of "
                f"the '{name}' dataset ({dataset.column_names}), so its reference log-probs could not "
                f"be matched to its rows on resume."
            )
        return {column: _token_digest(dataset, column) for column in columns}

    def _restore_reference_logps(
        self, dataset, name: str, needed: tuple[str, ...], settings: Mapping
    ) -> Dataset | None:
        """Attach the ``name`` split's columns saved in the resume checkpoint, or ``None`` to sweep.

        A sweep is correct only when it scores untrained weights. With no separate reference model
        and the policy built from the checkpoint it would score the trained ones, so there a saved
        split must exist and match this dataset's row count, token digests and reference settings,
        or the resume raises. Anywhere else an absent or mismatched split is swept. Every verdict is
        joined across ranks, so the raises and the sweep are taken by the whole world or none.
        """
        checkpoint = self._reference_resume_checkpoint
        saved, path = consensus_read(
            os.path.join(checkpoint, REFERENCE_LOGPS_FILE),
            partial(torch.load, map_location="cpu", weights_only=True),
            what=REFERENCE_LOGPS_FILE,
            checkpoint=checkpoint,
        )
        if not isinstance(saved, Mapping):
            saved = {}
        self._resumed_reference_logps = dict(saved)
        entry = saved.get(name)
        sweep_scores_trained_weights = self._policy_from_checkpoint and self.ref_model is None
        present_all, present_any = rank_consensus(entry is not None)
        if present_any and not present_all:
            raise RuntimeError(
                f"{REFERENCE_LOGPS_FILE} at {checkpoint} holds the '{name}' split on some ranks only — "
                f"the nodes' copies differ. Resume from a complete checkpoint."
            )
        if not present_all:
            if sweep_scores_trained_weights:
                raise RuntimeError(
                    f"Cannot resume precompute_ref_log_probs from {checkpoint}: it holds no saved "
                    f"reference log-probs for the '{name}' dataset ({REFERENCE_LOGPS_FILE} is missing "
                    f"or lacks that split), and they cannot be recomputed here. With no separate "
                    f"reference model the sweep scores the policy, and this resume built the policy "
                    f"from the checkpoint, so the sweep would score the TRAINED weights as the "
                    f"reference and zero every log-ratio. To recover, {_regeneration_steps(checkpoint)}; "
                    f"this resume then checks it against the dataset. A checkpoint whose save stopped "
                    f"before this file can take the previous checkpoint's copy instead, which holds the "
                    f"same values. Or supply the {list(needed)} columns, computed on the base model, in "
                    f"the dataset."
                )
            logger.info(
                f"No saved reference log-probs for '{name}' in {checkpoint}; sweeping, since the "
                f"reference weights are not the checkpoint's."
            )
            return None
        num_rows = len(dataset)
        token_digests = self._reference_input_digests(dataset, name)
        mismatch = _saved_split_mismatch(entry, num_rows, token_digests, settings, needed)
        if not sweep_scores_trained_weights:
            matches_all, _ = rank_consensus(mismatch is None)
            if not matches_all:
                logger.info(
                    f"The saved '{name}' reference log-probs in {path} do not match this dataset "
                    f"({mismatch or 'on another rank'}); sweeping, since the reference weights are not "
                    f"the checkpoint's."
                )
                return None
        reject_across_ranks(
            None if mismatch is None else _restore_refusal(path, name, entry, settings, mismatch, checkpoint),
            f"Restoring the '{name}' reference log-probs",
            exc_type=ValueError,
        )
        columns = {column: entry["columns"][column] for column in needed}
        self._reference_logps_by_split[name] = {
            "num_rows": num_rows,
            "token_digests": token_digests,
            "settings": settings,
            "columns": columns,
        }
        logger.info(f"Restored the '{name}' reference log-probs from {path}; skipping the sweep.")
        return _attach_reference_columns(dataset, columns)

    def _persist_trainer_sidecars(self, checkpoint_dir: str) -> None:
        """Write the swept or restored reference columns into every checkpoint of the run.

        The next resume attaches them rather than sweeping a policy that may by then hold trained
        weights. The resume checkpoint's other splits ride along unchanged, for a later resume that
        precomputes them again. Written on the FS-aware save rank(s), so each node of a non-shared
        output filesystem holds its own copy, and fenced so a failed write cannot strand the peers.
        """
        super()._persist_trainer_sidecars(checkpoint_dir)
        splits = {**self._resumed_reference_logps, **self._reference_logps_by_split}
        if not splits:
            return
        with barrier_on_exit():
            if fs_aware_save_rank():
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(splits, os.path.join(checkpoint_dir, REFERENCE_LOGPS_FILE))
