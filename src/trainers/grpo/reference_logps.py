"""Ragged completion-token payload hooks for offline GRPO's frozen-reference sidecar."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import pyarrow as pa
import torch
from datasets import Dataset

from src.checkpoint.format import REFERENCE_LOGPS_FILE
from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from src.distributed.runtime import (
    DeferredRankFailure,
    reject_across_ranks,
    reject_divergent_settings,
)
from src.trainers.grpo.reference_cache import (
    REFERENCE_BUFFER_VALUES,
    MappedReferenceScores,
    mapped_reference_scores,
    reference_payload_mismatch,
)
from src.trainers.mixins.reference_logps import ReferenceLogpsCheckpointMixin, token_digest

_TOKEN_COLUMNS = ("prompt_input_ids", "completion_input_ids")


def _scores_digest(lengths: torch.Tensor, values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in (lengths, values):
        for start in range(0, tensor.numel(), REFERENCE_BUFFER_VALUES):
            digest.update(memoryview(tensor[start : start + REFERENCE_BUFFER_VALUES].contiguous().numpy()).cast("B"))
    return digest.hexdigest()


def assert_replicated_reference_scores(split: str, lengths: torch.Tensor, values: torch.Tensor) -> str:
    """Require identical raw reference scores on every rank and return their content digest."""
    digest = _scores_digest(lengths, values)
    reject_divergent_settings(
        {"split": split, "reference_digest": digest},
        "Offline GRPO reference values",
        "The same unsharded rows must carry the same raw reference scores on every rank.",
    )
    return digest


def attach_reference_column(dataset: Dataset, scores: MappedReferenceScores, digest: str | None = None) -> Dataset:
    """Attach mapped scores without copying their immutable token buffer."""
    digest = _scores_digest(scores.lengths, scores.values) if digest is None else digest
    fingerprint = hashlib.sha256(f"{dataset._fingerprint}/{digest}".encode()).hexdigest()
    attached = dataset.add_column(REF_PER_TOKEN_LOGPS_COLUMN, scores.column(), new_fingerprint=fingerprint)
    attached._reference_storage_owner = scores
    return attached


class OfflineGRPOReferenceLogpsMixin(ReferenceLogpsCheckpointMixin):
    """Restore frozen token scores using the DPO/KTO checkpoint lifecycle."""

    def _reference_resume_required(self) -> bool:
        return self._policy_from_checkpoint

    def _init_reference_logps(self, *, resume_checkpoint: str | None, resume_context_given: bool = True) -> None:
        self._init_reference_state(
            checkpoint=resume_checkpoint,
            given=resume_context_given,
            policy_from_checkpoint=resume_checkpoint is not None,
        )
        self._reference_storage_by_split: dict[str, MappedReferenceScores] = {}

    def _reference_input_digests(self, dataset: Dataset, name: str) -> dict[str, str]:
        if not isinstance(dataset, Dataset):
            raise TypeError(f"'{name}' must be a finite datasets.Dataset for a reference sweep")
        for column in _TOKEN_COLUMNS:
            arrow_type = dataset.features.arrow_schema.field(column).type
            if not (pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type)) or not pa.types.is_integer(
                arrow_type.value_type
            ):
                raise ValueError(f"'{column}' must be a list of integer token IDs, got {arrow_type}")
        return {column: token_digest(dataset, column) for column in _TOKEN_COLUMNS}

    def _restore_reference_logps_or_none(
        self,
        dataset: Dataset,
        split: str,
        *,
        identity: Mapping,
    ) -> Dataset | None:
        reject_across_ranks(
            None if isinstance(dataset, Dataset) else f"'{split}' must be a finite datasets.Dataset",
            "Validating offline GRPO reference dataset",
            exc_type=ValueError,
        )
        self._check_reference_resume_context()
        reject_divergent_settings(
            {"split": split, **identity},
            "Offline GRPO reference inputs",
            "The same unsharded tokenized split and settings must reach every rank.",
        )
        return self._restore_reference_split(dataset, split, (REF_PER_TOKEN_LOGPS_COLUMN,), identity)

    def _attach_scored_reference_logps(
        self,
        dataset: Dataset,
        split: str,
        rows: MappedReferenceScores,
        *,
        identity: Mapping,
    ) -> Dataset:
        reject_across_ranks(
            f"Reference split '{split}' was already attached" if split in self._reference_logps_by_split else None,
            f"Recording the '{split}' GRPO reference",
            exc_type=ValueError,
        )
        digest = assert_replicated_reference_scores(split, rows.lengths, rows.values)
        guard = DeferredRankFailure(f"Attaching the '{split}' GRPO reference", exc_type=ValueError)
        attached = guard.run(lambda: attach_reference_column(dataset, rows, digest))
        guard.reject()
        self._reference_storage_by_split[split] = rows
        self._remember_reference_split(split, identity, {}, attached)
        return attached

    def _reference_payload_mismatch(self, entry: Mapping, dataset: Dataset, needed: Sequence[str]) -> str | None:
        del needed
        return reference_payload_mismatch(entry.get("lengths"), entry.get("values"), dataset)

    def _attach_reference_payload(self, dataset: Dataset, entry: Mapping, needed: Sequence[str]) -> Dataset:
        del needed
        scores = mapped_reference_scores(entry["lengths"], entry["values"])
        return attach_reference_column(dataset, scores)

    def _validate_restored_reference_payload(self, name: str, entry: Mapping) -> None:
        assert_replicated_reference_scores(name, entry["lengths"], entry["values"])

    def _remember_reference_split(self, name: str, identity: Mapping, payload: Mapping, dataset: Dataset) -> None:
        if payload:
            self._reference_storage_by_split[name] = mapped_reference_scores(payload["lengths"], payload["values"])
        self._reference_logps_by_split[name] = dict(identity)
        self._resumed_reference_logps.pop(name, None)
        self._reference_state_generation += 1

    def _reference_checkpoint_payload(self) -> dict:
        return {
            **self._resumed_reference_logps,
            **{
                name: {
                    **identity,
                    "lengths": self._reference_storage_by_split[name].lengths,
                    "values": self._reference_storage_by_split[name].values,
                }
                for name, identity in self._reference_logps_by_split.items()
            },
        }

    def _missing_reference_split(self, checkpoint: str, name: str, needed: Sequence[str]) -> str:
        del needed
        return (
            f"Cannot resume offline GRPO from {checkpoint}: its {REFERENCE_LOGPS_FILE} lacks '{name}'. "
            "Recomputing it would score the TRAINED checkpoint policy as its own reference. "
            "Restore the original sidecar from a complete checkpoint. If none exists, run the same "
            "config from the exact original model/revision and tokenized train/eval data into a "
            "separate scratch output (--resume_from_checkpoint=null --max_steps=1 "
            "--save_strategy=steps --save_steps=1 --save_only_model=true), then copy its "
            f"checkpoint-1/{REFERENCE_LOGPS_FILE} here on every node with node-local checkpoints. "
            "Do not use the trained checkpoint as that recovery run's model source."
        )

    def _read_reference_checkpoint(self, path: str):
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
