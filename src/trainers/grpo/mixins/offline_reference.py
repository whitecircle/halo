"""Offline GRPO's run-start KL reference: the ragged per-token payload of the shared frozen-reference
checkpoint lifecycle, the training-checkpoint contract, and evaluation's reuse of the original scores."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa
import torch
from datasets import Dataset

from src.checkpoint.format import REFERENCE_LOGPS_FILE
from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from src.distributed.runtime import (
    DeferredRankFailure,
    fs_aware_save_rank,
    rank_consensus,
    reject_across_ranks,
    reject_divergent_settings,
)
from src.models.structure import base_transformers_model
from src.trainers.grpo.reference_cache import (
    REFERENCE_BUFFER_VALUES,
    MappedReferenceScores,
    ReferenceScoreCache,
    mapped_reference_scores,
    reference_payload_mismatch,
    stage_checkpoint_file,
)
from src.trainers.mixins.reference_logps import (
    PREVIOUS_CHECKPOINT_RECOVERY,
    REFERENCE_SCAN_ROWS,
    ReferenceLogpsCheckpointMixin,
    reference_regeneration_steps,
    token_digest,
)

_TOKEN_COLUMNS = ("prompt_input_ids", "completion_input_ids")


def reject_unsupported_reference_input(
    train_dataset, eval_dataset, *, active: bool = True, presharded: bool = False
) -> None:
    """Validate only the finite splits consumed by a run-start full-finetuning sweep."""
    if not active:
        return
    reason = None
    if presharded:
        reason = (
            "Offline GRPO reference precompute is not supported with a pre-sharded dataset; load the same "
            "unsharded dataset on every rank."
        )
    elif not isinstance(train_dataset, Dataset):
        reason = (
            "Offline GRPO KL reference preparation requires a finite datasets.Dataset training split, not "
            "streaming or schema-less input."
        )
    elif isinstance(eval_dataset, Mapping):
        reason = (
            "Offline GRPO constructor KL reference preparation requires one finite evaluation Dataset. Use "
            "evaluate() for named tokenized splits after construction."
        )
    elif eval_dataset is not None and not isinstance(eval_dataset, Dataset):
        reason = "Offline GRPO KL reference preparation requires a finite datasets.Dataset evaluation split."
    elif any(
        REF_PER_TOKEN_LOGPS_COLUMN in split.column_names
        for split in (train_dataset, eval_dataset)
        if split is not None
    ):
        reason = (
            "Supplied ref_per_token_logps are not supported by offline GRPO full-finetuning KL; load an "
            "unsharded dataset and let the trainer prepare its checkpointed run-start reference."
        )
    reject_across_ranks(reason, "Validating offline GRPO reference inputs", exc_type=ValueError)


def _token_row_keys(dataset: Dataset) -> Iterator[bytes]:
    """Hash each ordered token row without converting Arrow token arrays into Python lists."""
    for batch in (
        dataset.select_columns(list(_TOKEN_COLUMNS)).with_format("arrow").iter(batch_size=REFERENCE_SCAN_ROWS)
    ):
        columns = [batch.column(name).combine_chunks() for name in _TOKEN_COLUMNS]
        for index in range(len(batch)):
            digest = hashlib.sha256()
            for column in columns:
                tokens = column[index].values.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
                digest.update(np.asarray([len(tokens)], dtype=np.int64).tobytes())
                digest.update(memoryview(tokens).cast("B"))
            yield digest.digest()


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


class OfflineGRPOReferenceMixin(ReferenceLogpsCheckpointMixin):
    """Offline GRPO's frozen per-token reference over the DPO/KTO checkpoint lifecycle.

    The host trainer sets ``_precompute_reference`` (whether this run sweeps a run-start reference)
    before it trains or evaluates.
    """

    _reference_token_settings = "max_prompt_length, max_completion_length or drop_degenerate_groups"
    _maps_saved_reference = True

    def _init_reference_logps(self, *, resume_checkpoint: str | None, resume_context_given: bool = True) -> None:
        self._init_reference_state(
            checkpoint=resume_checkpoint,
            given=resume_context_given,
            policy_from_checkpoint=resume_checkpoint is not None,
        )
        self._reference_storage_by_split: dict[str, MappedReferenceScores] = {}
        self._reference_dataset_by_split: dict[str, Dataset] = {}
        # Rows scored for evaluate() after construction, kept so the same rows are never stored twice.
        self._evaluation_references: list[tuple[dict, Dataset, MappedReferenceScores]] = []

    def _reference_resume_required(self) -> bool:
        return self._policy_from_checkpoint

    def _reference_cache_output_dir(self) -> str:
        return os.fspath(self.args.output_dir)

    def _saved_reference_file(self, checkpoint: str) -> str:
        return stage_checkpoint_file(self._reference_cache_output_dir(), super()._saved_reference_file(checkpoint))

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
        self._reference_dataset_by_split[name] = dataset
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
            f"To recover, {reference_regeneration_steps(checkpoint)}. {PREVIOUS_CHECKPOINT_RECOVERY}"
        )

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        if self._precompute_reference:
            declared = self._reference_resume_checkpoint
            requested = None if resume_from_checkpoint is False else resume_from_checkpoint
            reason = None
            if requested is True or (requested is not None and declared is None):
                reason = "Resolve the offline GRPO resume checkpoint before constructing the trainer"
            elif declared is not None:
                same_checkpoint = isinstance(requested, (str, os.PathLike)) and os.path.realpath(
                    os.fspath(requested)
                ) == os.path.realpath(os.fspath(declared))
                if not same_checkpoint:
                    reason = "Offline GRPO must train from the same checkpoint used to restore its reference"
            reject_across_ranks(reason, "Validating the offline GRPO training checkpoint", exc_type=ValueError)
        return super().train(resume_from_checkpoint, *args, **kwargs)

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
        *,
        original_reference_model=None,
    ):
        """Reuse the run-start anchor, or score unseen rows with a caller-owned original policy."""
        if not self._precompute_reference:
            if original_reference_model is not None:
                raise ValueError("original_reference_model is only used by full-finetuning KL evaluation")
        else:
            reject_divergent_settings(
                {
                    "reference_model_supplied": original_reference_model is not None,
                    "evaluation_kind": "default"
                    if eval_dataset is None
                    else "named"
                    if isinstance(eval_dataset, dict)
                    else "split",
                },
                "Offline GRPO evaluation call",
                "Every rank must supply the same evaluation/reference call layout.",
            )
            if isinstance(eval_dataset, dict):
                reject_divergent_settings(
                    {"eval_split_names": list(eval_dataset)},
                    "Offline GRPO evaluation splits",
                    "Every rank must evaluate the same named splits in the same order.",
                )
                eval_dataset = {
                    name: self._prepare_evaluation_reference(dataset, original_reference_model)
                    for name, dataset in eval_dataset.items()
                }
            elif eval_dataset is not None:
                eval_dataset = self._prepare_evaluation_reference(eval_dataset, original_reference_model)
            elif original_reference_model is not None:
                raise ValueError("Pass an evaluation dataset when supplying original_reference_model")
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def _validate_evaluation_reference_model(self, model) -> None:
        reason = None
        if not isinstance(model, torch.nn.Module):
            reason = "original_reference_model must be the exact original frozen policy model"
        elif model is self.model or base_transformers_model(model) is base_transformers_model(self.model):
            reason = "The trained/live policy cannot be original_reference_model"
        elif model.training or any(parameter.requires_grad for parameter in model.parameters()):
            reason = "original_reference_model must be frozen (requires_grad=False) and in eval mode"
        elif self.parallelism_config.is_cp_mode or self._pp_runtime is not None:
            reason = (
                "An external original_reference_model is not supported for CP/PP evaluation. "
                "Declare the evaluation split before training so the original policy precomputes it."
            )
        reject_across_ranks(reason, "Validating the original evaluation reference", exc_type=ValueError)

    def _stored_references(self) -> Iterator[tuple[Mapping, Dataset, MappedReferenceScores]]:
        """Every ``(identity, token rows, scores)`` the run holds: its checkpointed splits, then the rows
        evaluation scored since."""
        for name, identity in self._reference_logps_by_split.items():
            yield identity, self._reference_dataset_by_split[name], self._reference_storage_by_split[name]
        yield from self._evaluation_references

    def _prepare_evaluation_reference(self, dataset, original_reference_model) -> Dataset:
        if original_reference_model is not None:
            self._validate_evaluation_reference_model(original_reference_model)
        reject_across_ranks(
            None if isinstance(dataset, Dataset) else "KL evaluation requires a finite tokenized datasets.Dataset",
            "Preparing offline GRPO evaluation references",
            exc_type=ValueError,
        )
        if REF_PER_TOKEN_LOGPS_COLUMN in dataset.column_names:
            dataset = dataset.remove_columns(REF_PER_TOKEN_LOGPS_COLUMN)
        identity = self._reference_split_identity(dataset, "evaluation")
        reject_divergent_settings(identity, "Offline GRPO evaluation rows", "Every rank must evaluate the same rows.")
        stored = next((scores for known, _, scores in self._stored_references() if known == identity), None)
        # Agreed, not local: scoring enters collectives that attaching stored scores does not.
        if rank_consensus(stored is not None)[0]:
            scores = stored
        else:
            scores = self._score_evaluation_rows(dataset, identity, original_reference_model)
            self._evaluation_references.append((identity, dataset, scores))
        guard = DeferredRankFailure("Attaching original evaluation reference scores", exc_type=ValueError)
        attached = guard.run(lambda: attach_reference_column(dataset, scores))
        guard.reject()
        return attached

    def _score_evaluation_rows(self, dataset: Dataset, identity: Mapping, original_reference_model):
        """Original scores for rows no stored set holds whole: copied row by row from the stored sets, or
        swept with the caller's original policy where some row is unseen."""
        guard = DeferredRankFailure("Matching evaluation rows to the original reference", exc_type=ValueError)

        def match_rows():
            known = {}
            for stored_identity, stored, scores in self._stored_references():
                if stored_identity["settings"] != identity["settings"]:
                    continue
                for index, key in enumerate(_token_row_keys(stored)):
                    known.setdefault(key, (scores, index))
            return [known.get(key) for key in _token_row_keys(dataset)]

        matches = guard.run(match_rows)
        guard.reject()
        # Agreed, not local: the sweep and the cache reuse below enter different collectives.
        missing = rank_consensus(any(match is None for match in matches))[1]
        reject_across_ranks(
            "Evaluation contains unseen token rows whose original KL reference was not precomputed. "
            "Declare this eval split before training, or outside CP/PP call "
            "evaluate(new_dataset, original_reference_model=original_frozen_policy). "
            "Use the exact run-start weights, not the trained checkpoint or a different base revision."
            if missing and original_reference_model is None
            else None,
            "Preparing offline GRPO evaluation references",
            exc_type=ValueError,
        )
        if missing:
            scores = self._sweep_with_original_reference(dataset, original_reference_model)
        else:
            cache = ReferenceScoreCache(self._reference_cache_output_dir(), dp_size=1)
            guard = DeferredRankFailure("Reusing original evaluation reference scores")
            try:
                if fs_aware_save_rank():
                    guard.run(
                        lambda: cache.append_rows(
                            0,
                            (
                                storage.values[int(storage.offsets[index]) : int(storage.offsets[index + 1])]
                                for storage, index in matches
                            ),
                        )
                    )
                guard.reject()
                scores = cache.finish(dataset)
            except BaseException as exc:
                try:
                    cache.discard()
                except Exception as cleanup_error:
                    exc.add_note(f"Evaluation reference cache cleanup also failed: {cleanup_error}")
                raise
        assert_replicated_reference_scores("evaluation", scores.lengths, scores.values)
        return scores

    def _sweep_with_original_reference(self, dataset: Dataset, model) -> MappedReferenceScores:
        """Sweep ``dataset`` with the caller's original policy, placed on the policy's device for the sweep
        and returned to its own device and ownership after it."""
        devices = {tensor.device for tensor in (*model.parameters(), *model.buffers())}
        reject_across_ranks(
            "original_reference_model must reside on one device before temporary scoring placement"
            if len(devices) != 1
            else None,
            "Placing the original evaluation reference",
            exc_type=ValueError,
        )
        original_device = next(iter(devices))
        previous_reference = self.ref_model
        try:
            guard = DeferredRankFailure("Placing the original evaluation reference")
            guard.run(lambda: model.to(next(self.model.parameters()).device))
            guard.reject()
            self.ref_model = model
            scores = self._sweep_reference_logps(dataset, "evaluation")
        except BaseException as exc:
            # A model forward may have failed while peers remain in its collectives.
            # Restore caller ownership locally, without entering another rank consensus.
            self.ref_model = previous_reference
            try:
                model.to(original_device)
            except Exception as cleanup_error:
                exc.add_note(f"Original evaluation reference device restoration also failed: {cleanup_error}")
            raise
        self.ref_model = previous_reference
        guard = DeferredRankFailure("Restoring the caller-owned evaluation reference")
        guard.run(lambda: model.to(original_device))
        guard.reject()
        return scores
