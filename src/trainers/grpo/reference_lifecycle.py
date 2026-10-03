"""Offline GRPO's run-start reference training and evaluation lifecycle."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping

import numpy as np
import torch
from datasets import Dataset

from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from src.distributed.runtime import (
    DeferredRankFailure,
    fs_aware_save_rank,
    rank_consensus,
    reject_across_ranks,
    reject_divergent_settings,
)
from src.models.structure import base_transformers_model
from src.trainers.grpo.reference_cache import REFERENCE_BATCH_ROWS, ReferenceScoreCache
from src.trainers.grpo.reference_logps import (
    OfflineGRPOReferenceLogpsMixin,
    assert_replicated_reference_scores,
    attach_reference_column,
)


def reject_unsupported_reference_input(
    train_dataset, eval_dataset, *, active: bool = True, presharded: bool = False
) -> None:
    """Validate only the finite splits consumed by a run-start full-finetuning sweep."""
    if not active:
        return
    reason = None
    if presharded:
        reason = "Offline GRPO reference precompute is not supported with a pre-sharded dataset; load the same unsharded dataset on every rank."
    elif not isinstance(train_dataset, Dataset):
        reason = "Offline GRPO KL reference preparation requires a finite datasets.Dataset training split, not streaming or schema-less input."
    elif isinstance(eval_dataset, Mapping):
        reason = "Offline GRPO constructor KL reference preparation requires one finite evaluation Dataset. Use evaluate() for named tokenized splits after construction."
    elif eval_dataset is not None and not isinstance(eval_dataset, Dataset):
        reason = "Offline GRPO KL reference preparation requires a finite datasets.Dataset evaluation split."
    elif any(
        REF_PER_TOKEN_LOGPS_COLUMN in split.column_names
        for split in (train_dataset, eval_dataset)
        if split is not None
    ):
        reason = "Supplied ref_per_token_logps are not supported by offline GRPO full-finetuning KL; load an unsharded dataset and let the trainer prepare its checkpointed run-start reference."
    reject_across_ranks(reason, "Validating offline GRPO reference inputs", exc_type=ValueError)


def _token_row_keys(dataset: Dataset):
    """Hash each ordered token row without converting Arrow token arrays into Python lists."""
    for batch in (
        dataset.select_columns(["prompt_input_ids", "completion_input_ids"])
        .with_format("arrow")
        .iter(batch_size=REFERENCE_BATCH_ROWS)
    ):
        columns = [batch.column(name).combine_chunks() for name in ("prompt_input_ids", "completion_input_ids")]
        for index in range(len(batch)):
            digest = hashlib.sha256()
            for column in columns:
                tokens = column[index].values.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
                digest.update(np.asarray([len(tokens)], dtype=np.int64).tobytes())
                digest.update(memoryview(tokens).cast("B"))
            yield digest.digest()


class OfflineGRPOReferenceLifecycleMixin(OfflineGRPOReferenceLogpsMixin):
    """Trainer behavior over the checkpointed, immutable run-start reference store."""

    def _init_reference_logps(self, *, resume_checkpoint: str | None, resume_context_given: bool = True) -> None:
        super()._init_reference_logps(resume_checkpoint=resume_checkpoint, resume_context_given=resume_context_given)
        self._reference_dataset_by_split: dict[str, Dataset] = {}
        self._reference_evaluation_datasets: list[Dataset] = []
        self._reference_evaluation_depth = 0

    def _remember_reference_split(self, name: str, identity: Mapping, payload: Mapping, dataset: Dataset) -> None:
        super()._remember_reference_split(name, identity, payload, dataset)
        self._reference_dataset_by_split[name] = dataset

    def _reference_cache_output_dir(self) -> str:
        return os.fspath(self.args.output_dir)

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        prepared = bool(self._reference_logps_by_split) or getattr(self, "_precompute_reference", False)
        if prepared and self.beta != 0.0:
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
        outermost = self._reference_evaluation_depth == 0
        self._reference_evaluation_depth += 1
        try:
            if getattr(self, "_precompute_reference", False):
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
            if not getattr(self, "_precompute_reference", False):
                if original_reference_model is not None:
                    raise ValueError("original_reference_model is only used by full-finetuning KL evaluation")
            elif eval_dataset is not None:
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
                else:
                    eval_dataset = self._prepare_evaluation_reference(eval_dataset, original_reference_model)
            elif original_reference_model is not None:
                raise ValueError("Pass an evaluation dataset when supplying original_reference_model")
            return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        finally:
            self._reference_evaluation_depth -= 1
            if outermost:
                self._reference_evaluation_datasets.clear()

    def _validate_evaluation_reference_model(self, model) -> None:
        reason = None
        if not isinstance(model, torch.nn.Module):
            reason = "original_reference_model must be the exact original frozen policy model"
        elif model is self.model or base_transformers_model(model) is base_transformers_model(self.model):
            reason = "The trained/live policy cannot be original_reference_model"
        elif model.training or any(parameter.requires_grad for parameter in model.parameters()):
            reason = "original_reference_model must be frozen (requires_grad=False) and in eval mode"
        elif getattr(self, "_pp_runtime", None) is not None:
            reason = (
                "An external original_reference_model is not supported for PP evaluation. "
                "Declare the evaluation split before training so the original policy precomputes it."
            )
        reject_across_ranks(reason, "Validating the original evaluation reference", exc_type=ValueError)

    def _prepare_evaluation_reference(self, dataset, original_reference_model) -> Dataset:
        if original_reference_model is not None:
            self._validate_evaluation_reference_model(original_reference_model)
        reject_across_ranks(
            None if isinstance(dataset, Dataset) else "KL evaluation requires a finite tokenized datasets.Dataset",
            "Preparing offline GRPO evaluation references",
            exc_type=ValueError,
        )
        original_dataset = dataset
        if REF_PER_TOKEN_LOGPS_COLUMN in dataset.column_names:
            dataset = dataset.remove_columns(REF_PER_TOKEN_LOGPS_COLUMN)
        identity = self._reference_split_identity(dataset, "evaluation")
        reject_divergent_settings(identity, "Offline GRPO evaluation rows", "Every rank must evaluate the same rows.")
        known = any(
            original_dataset is stored and self._reference_logps_by_split[name]["settings"] == identity["settings"]
            for name, stored in self._reference_dataset_by_split.items()
        ) or any(
            original_dataset is stored and getattr(stored, "_reference_settings", None) == identity["settings"]
            for stored in self._reference_evaluation_datasets
        )
        if rank_consensus(known)[0]:
            return original_dataset
        guard = DeferredRankFailure("Matching evaluation rows to the original reference", exc_type=ValueError)

        def match_rows():
            known = {}
            for name, stored in self._reference_dataset_by_split.items():
                if self._reference_logps_by_split[name]["settings"] != identity["settings"]:
                    continue
                storage = self._reference_storage_by_split[name]
                for index, key in enumerate(_token_row_keys(stored)):
                    known.setdefault(key, (storage, index))
            return [known.get(key) for key in _token_row_keys(dataset)]

        matches = guard.run(match_rows)
        guard.reject()
        missing = any(match is None for match in matches)
        reject_across_ranks(
            "Evaluation contains unseen token rows whose original KL reference was not precomputed. "
            "Declare this eval split before training, or outside PP call "
            "evaluate(new_dataset, original_reference_model=original_frozen_policy). "
            "Use the exact run-start weights, not the trained checkpoint or a different base revision."
            if missing and original_reference_model is None
            else None,
            "Preparing offline GRPO evaluation references",
            exc_type=ValueError,
        )
        if missing:
            model = original_reference_model
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
            else:
                self.ref_model = previous_reference
                guard = DeferredRankFailure("Restoring the caller-owned evaluation reference")
                guard.run(lambda: model.to(original_device))
                guard.reject()
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
        digest = assert_replicated_reference_scores("evaluation", scores.lengths, scores.values)
        guard = DeferredRankFailure("Attaching original evaluation reference scores", exc_type=ValueError)
        attached = guard.run(lambda: attach_reference_column(dataset, scores, digest))
        guard.reject()
        attached._reference_settings = identity["settings"]
        self._reference_evaluation_datasets.append(attached)
        return attached
