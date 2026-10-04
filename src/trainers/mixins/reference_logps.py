"""Rank-consistent identity, resume and checkpoint lifecycle for frozen reference scores."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from functools import partial

import numpy as np
import pyarrow as pa
import torch
from accelerate.logging import get_logger
from datasets import Dataset

from src.checkpoint.atomic import atomic_torch_save
from src.checkpoint.format import REFERENCE_LOGPS_FILE
from src.distributed.checkpoint.coordination import consensus_read
from src.distributed.runtime import (
    DeferredRankFailure,
    barrier_on_exit,
    fs_aware_save_rank,
    rank_consensus,
    reject_across_ranks,
    reject_divergent_settings,
)

logger = get_logger(__name__, log_level="info")

_DIGEST_BATCH_ROWS = 256
_DIGEST_CHUNK_VALUES = 1 << 22
_IDENTITY_SCHEMA = {"num_rows": int, "token_digests": Mapping, "settings": Mapping}


def is_token_type(arrow_type: pa.DataType) -> bool:
    """Integer or boolean tokens, bare or in one list level."""
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        arrow_type = arrow_type.value_type
    return pa.types.is_integer(arrow_type) or pa.types.is_boolean(arrow_type)


def token_digest(dataset: Dataset, column: str) -> str:
    """Hash ordered row lengths and int64 values, independent of Arrow batches and integer width.

    Dataset fingerprints can hash trainer state through TRL's tokenize map and become random when
    it cannot be pickled. Reference identity must depend on token content, not that fingerprint.
    Separate length and value hashes keep batching and chunk boundaries out of the digest.
    """
    lengths, values = hashlib.sha256(), hashlib.sha256()
    for batch in dataset.select_columns([column]).with_format("arrow").iter(batch_size=_DIGEST_BATCH_ROWS):
        array = batch.column(column).combine_chunks()
        if array.null_count:
            raise ValueError(f"'{column}' contains null token IDs")
        if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
            lengths.update(array.value_lengths().to_numpy(zero_copy_only=False).astype(np.int64).tobytes())
            array = array.flatten()
        if array.null_count:
            raise ValueError(f"'{column}' contains null token IDs")
        for start in range(0, len(array), _DIGEST_CHUNK_VALUES):
            chunk = array.slice(start, _DIGEST_CHUNK_VALUES)
            values.update(chunk.to_numpy(zero_copy_only=False).astype(np.int64).tobytes())
    return hashlib.sha256(lengths.digest() + values.digest()).hexdigest()


def is_reference_entry(entry: object) -> bool:
    """Whether a saved split carries every identity field, each of its type."""
    return isinstance(entry, Mapping) and all(
        isinstance(entry.get(key), value_type) for key, value_type in _IDENTITY_SCHEMA.items()
    )


def reference_regeneration_steps(checkpoint: str) -> str:
    """How to give ``checkpoint`` a reference file computed from the base model for this run."""
    # Outside the run's own output_dir, whose rotation could otherwise delete this checkpoint.
    scratch = f"{os.path.dirname(os.path.abspath(checkpoint))}-reference-recovery"
    return (
        f"run this config for one step from the base model into a scratch directory (--output_dir={scratch} "
        "--max_steps=1 --save_strategy=steps --save_steps=1 --save_only_model=true "
        f"--resume_from_checkpoint=null) and copy its checkpoint-1/{REFERENCE_LOGPS_FILE} into "
        f"{checkpoint}, on every node when checkpoints are node-local"
    )


class ReferenceLogpsCheckpointMixin:
    """Shared frozen-reference lifecycle; subclasses own the score payload and the sweep.

    Place this mixin before DistributedTrainerMixin in the trainer's bases so its checkpoint hook
    overrides CheckpointingMixin's empty default and the reference rides every checkpoint.
    """

    def _init_reference_state(self, *, checkpoint, given: bool, policy_from_checkpoint: bool) -> None:
        self._reference_resume_given = given
        self._reference_resume_checkpoint = checkpoint
        self._policy_from_checkpoint = policy_from_checkpoint
        self._reference_logps_by_split: dict[str, dict] = {}
        self._resumed_reference_logps: dict[str, object] = {}
        self._reference_saved_loaded = False
        self._reference_state_generation = 0
        self._reference_saved_generation = -1
        self._reference_immutable_path: str | None = None

    def _reference_resume_required(self) -> bool:
        return self._policy_from_checkpoint and getattr(self, "ref_model", None) is None

    def _reference_split_identity(self, dataset: Dataset, name: str, settings: Mapping | None = None) -> dict:
        guard = DeferredRankFailure(f"Identifying the '{name}' reference dataset", exc_type=ValueError)
        identity = guard.run(
            lambda: {
                "num_rows": len(dataset),
                "token_digests": self._reference_input_digests(dataset, name),
                "settings": dict(self._reference_settings() if settings is None else settings),
            }
        )
        guard.reject()
        return identity

    def _check_reference_resume_context(self) -> None:
        missing = (
            not self._reference_resume_given
            and getattr(self, "ref_model", None) is None
            and getattr(getattr(self, "args", None), "resume_from_checkpoint", None)
        )
        reject_across_ranks(
            f"resume_from_checkpoint={self.args.resume_from_checkpoint!r} is set, but the trainer was built "
            "without resume_checkpoint/"
            "policy_from_checkpoint context. The reference sweep runs inside __init__, before train() "
            "restores weights; pass the resolved context so trained weights are not their own reference."
            if missing
            else None,
            "Validating reference resume context",
            exc_type=ValueError,
        )

    def _restore_reference_split(self, dataset: Dataset, name: str, needed: Sequence[str], identity: Mapping):
        checkpoint = self._reference_resume_checkpoint
        if checkpoint is None:
            return None
        if not self._reference_saved_loaded:
            saved, path = consensus_read(
                os.path.join(checkpoint, REFERENCE_LOGPS_FILE),
                self._read_reference_checkpoint,
                what=REFERENCE_LOGPS_FILE,
                checkpoint=checkpoint,
            )
            guard = DeferredRankFailure(f"Reading {REFERENCE_LOGPS_FILE}", exc_type=ValueError)
            self._resumed_reference_logps = guard.run(lambda: dict(saved) if isinstance(saved, Mapping) else {})
            guard.reject()
            self._reference_saved_path = path
            self._reference_saved_loaded = True
        entry = self._resumed_reference_logps.get(name)
        present_all, present_any = rank_consensus(entry is not None)
        if present_any and not present_all:
            raise RuntimeError(
                f"{REFERENCE_LOGPS_FILE} at {checkpoint} holds the '{name}' split on some ranks only — "
                "the nodes' copies differ. Resume from a complete checkpoint."
            )
        required = self._reference_resume_required()
        if not present_all:
            if required:
                raise RuntimeError(self._missing_reference_split(checkpoint, name, needed))
            logger.info(f"No saved '{name}' reference in {checkpoint}; sweeping untrained reference weights.")
            return None
        guard = DeferredRankFailure(f"Validating the '{name}' reference payload", exc_type=ValueError)
        mismatch = guard.run(lambda: self._reference_entry_mismatch(entry, dataset, needed, identity))
        guard.reject()
        if not required and not rank_consensus(mismatch is None)[0]:
            logger.info(f"Saved '{name}' reference does not match ({mismatch or 'another rank'}); sweeping.")
            return None
        reject_across_ranks(
            None if mismatch is None else self._reference_mismatch_refusal(name, entry, identity, mismatch),
            f"Restoring the '{name}' reference log-probs",
            exc_type=ValueError,
        )
        guard = DeferredRankFailure(f"Attaching the '{name}' reference log-probs", exc_type=ValueError)
        attached = guard.run(lambda: self._attach_reference_payload(dataset, entry, needed))
        guard.reject()
        self._validate_restored_reference_payload(name, entry)
        self._remember_reference_split(name, identity, entry, attached)
        logger.info(f"Restored the '{name}' reference log-probs from {self._reference_saved_path}; skipping sweep.")
        return attached

    def _read_reference_checkpoint(self, path: str):
        return torch.load(path, map_location="cpu", weights_only=True)

    def _reference_entry_mismatch(self, entry, dataset, needed, identity) -> str | None:
        if not is_reference_entry(entry):
            return f"its entry is not a saved reference split (expected {sorted(_IDENTITY_SCHEMA)})"
        if entry["num_rows"] != identity["num_rows"]:
            return f"it was saved for {entry['num_rows']} rows and this dataset has {identity['num_rows']}"
        if mismatch := self._reference_payload_mismatch(entry, dataset, needed):
            return mismatch
        if entry["settings"] != identity["settings"]:
            return f"it was computed under {entry['settings']} and this run sets {identity['settings']}"
        changed = sorted(
            column
            for column, digest in identity["token_digests"].items()
            if entry["token_digests"].get(column) != digest
        )
        if changed:
            return (
                f"this dataset's {changed} differ from the saved run's (a changed dataset, split, chat template "
                "or tokenizer — or, for KTO's KL completions, per_device_train_batch_size or dataset_num_proc)"
            )
        return None

    def _validate_restored_reference_payload(self, name: str, entry: Mapping) -> None:
        """Optional rank-consistency check after all local payload validation has succeeded."""

    def _reference_mismatch_refusal(self, name: str, entry: object, identity: Mapping, mismatch: str) -> str:
        """The refusal of a saved split that does not match, where a sweep would score trained weights.

        Total over a malformed ``entry``: every rank that sees a mismatch must reach the verdict.
        """
        return (
            f"{self._reference_saved_path} does not belong to this '{name}' dataset: {mismatch}. "
            "Each saved value is its own row's reference, so attaching it to different data would "
            "score rows against references they were not computed for. Resume with the data and "
            "reference settings the checkpoint was written with, or give it a file computed for this "
            f"run: {reference_regeneration_steps(self._reference_resume_checkpoint)}."
        )

    def _missing_reference_split(self, checkpoint: str, name: str, needed: Sequence[str]) -> str:
        return (
            f"Cannot resume precompute_ref_log_probs from {checkpoint}: it holds no saved reference "
            f"log-probs for the '{name}' dataset ({REFERENCE_LOGPS_FILE} is missing or lacks that split), "
            "and they cannot be recomputed here: this resume built the policy from the checkpoint, "
            "so the sweep would score the TRAINED weights as the reference and zero every log-ratio. "
            f"To recover, {reference_regeneration_steps(checkpoint)}. A checkpoint whose save "
            "stopped before this file can take the previous checkpoint's copy instead. Or supply the "
            f"{list(needed)} columns, computed on the base model, in the dataset."
        )

    def _remember_reference_split(self, name: str, identity: Mapping, payload: Mapping, dataset: Dataset) -> None:
        self._reference_logps_by_split[name] = {**payload, **identity}
        self._reference_state_generation += 1

    def _reference_checkpoint_payload(self) -> dict:
        return {**self._resumed_reference_logps, **self._reference_logps_by_split}

    def _persist_trainer_sidecars(self, checkpoint_dir: str) -> None:
        """Commit references before CheckpointingMixin can rotate the previous complete checkpoint.

        This hook must precede the checkpointing default in the trainer's MRO. Its cooperative
        super() preserves other sidecars, and every rank joins the filesystem-aware writers.
        """
        super()._persist_trainer_sidecars(checkpoint_dir)
        names = sorted(set(self._resumed_reference_logps) | set(self._reference_logps_by_split))
        present_all, present_any = rank_consensus(bool(names))
        if not present_any:
            return
        if not present_all:
            raise RuntimeError("Reference splits exist on only some ranks; refusing a partial checkpoint")
        reject_divergent_settings(
            {"splits": names}, "Checkpoint reference splits", "Every rank must save the same split names."
        )
        previous = (
            self._reference_immutable_path
            if self._reference_saved_generation == self._reference_state_generation
            else None
        )
        path = os.path.join(checkpoint_dir, REFERENCE_LOGPS_FILE)
        guard = DeferredRankFailure(f"Writing {REFERENCE_LOGPS_FILE} to {checkpoint_dir}")
        with barrier_on_exit():
            if fs_aware_save_rank():
                guard.run(partial(atomic_torch_save, path, self._reference_checkpoint_payload, previous))
        guard.reject()
        self._reference_immutable_path = path
        self._reference_saved_generation = self._reference_state_generation
