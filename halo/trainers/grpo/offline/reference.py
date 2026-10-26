"""Offline GRPO's KL reference: where it comes from (:class:`ReferenceMode`), the run-start sweep that scores
it, its ragged per-token payload in the shared frozen-reference checkpoint lifecycle, and evaluation's reuse
of the original scores."""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Iterator, Mapping, Sequence
from enum import Enum

import numpy as np
import pyarrow as pa
import torch
from accelerate.utils import is_peft_model
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader

from halo.checkpoint.format import REFERENCE_LOGPS_FILE
from halo.configs.grpo_offline import OfflineGRPOConfig
from halo.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.dtype import resolve_training_dtype
from halo.models.unwrapping import base_transformers_model
from halo.runtime.consensus import (
    DeferredRankFailure,
    rank_consensus,
    rank_max,
    reject_across_ranks,
    reject_divergent_settings,
)
from halo.runtime.filesystem import is_fs_save_rank
from halo.runtime.ranks import get_global_rank, is_global_main_process
from halo.trainers.dataloaders import dp_representative_ranks
from halo.trainers.grpo.offline.reference_cache import (
    REFERENCE_BUFFER_VALUES,
    MappedReferenceScores,
    ReferenceScoreCache,
    link_checkpoint_into_scratch,
    mapped_reference_scores,
    reference_payload_mismatch,
)
from halo.trainers.reference_logps import (
    LOGPROB_PRECISION_KEY,
    PREVIOUS_CHECKPOINT_RECOVERY,
    REFERENCE_SCAN_ROWS,
    ReferenceLogpsMixin,
    reference_regeneration_steps,
    token_digest,
)

logger = logging.getLogger(__name__)

_TOKEN_COLUMNS = ("prompt_input_ids", "completion_input_ids")


class ReferenceMode(Enum):
    """Where an offline-GRPO policy's KL reference comes from; ``kl_beta`` 0 reads none of them."""

    # Full fine-tune: raw run-start scores, swept once before training and checkpointed.
    RUN_START = "run_start"
    # PEFT, native expert LoRA beside it included: the live policy with every adapter disabled.
    ADAPTERS_OFF = "adapters_off"
    # Native expert LoRA alone: its unadapted base, loaded separately and forwarded per batch.
    LIVE_BASE = "live_base"


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


class OfflineGRPOReferenceMixin(ReferenceLogpsMixin):
    """Offline GRPO's frozen per-token reference over the DPO/KTO checkpoint lifecycle.

    The host trainer sets ``_precompute_reference`` (whether this run sweeps a run-start reference) before
    it trains or evaluates, and scores one swept batch per row in ``_score_reference_batch``.
    """

    REFERENCE_TOKEN_SETTINGS = "max_prompt_length, max_completion_length or drop_degenerate_groups"
    MAPS_SAVED_REFERENCE = True

    @staticmethod
    def _reference_mode(model, parallelism_config: ParallelismConfig | None, peft_config) -> ReferenceMode:
        """The policy's KL reference source. ``model`` may still be a checkpoint id, never a PEFT model."""
        if peft_config is not None or (isinstance(model, nn.Module) and is_peft_model(model)):
            return ReferenceMode.ADAPTERS_OFF
        if getattr(parallelism_config, "expert_lora", None) is not None:
            return ReferenceMode.LIVE_BASE
        return ReferenceMode.RUN_START

    @classmethod
    def _holds_kl_reference(cls, model, kl_beta: float, parallelism_config: ParallelismConfig, peft_config) -> bool:
        """Whether this run reads an external frozen reference: the run-start sweep scores one in place
        of the policy, and native expert LoRA forwards its base per batch. CP and PP score the reference
        through the policy's own distributed path."""
        return (
            kl_beta != 0.0
            and cls._reference_mode(model, parallelism_config, peft_config) is not ReferenceMode.ADAPTERS_OFF
            and not parallelism_config.is_pp_mode
            and not parallelism_config.is_cp_mode
        )

    @classmethod
    def sweeps_run_start_reference(
        cls, model, kl_beta: float, parallelism_config: ParallelismConfig | None, *, peft: bool
    ) -> bool:
        """Whether the run scores its KL reference once at run start: a full fine-tune with ``kl_beta``
        set. ``model`` may still be a checkpoint id; ``peft`` is whether attention adapters wrap it."""
        return (
            kl_beta != 0.0
            and not peft
            and cls._reference_mode(model, parallelism_config, None) is ReferenceMode.RUN_START
        )

    @classmethod
    def requires_ref_model(
        cls, model, args: OfflineGRPOConfig, parallelism_config: ParallelismConfig, peft_config
    ) -> bool:
        """Native expert LoRA needs its unadapted base; full fine-tuning sweeps its own policy."""
        return cls._holds_kl_reference(model, args.kl_beta, parallelism_config, peft_config) and (
            cls._reference_mode(model, parallelism_config, peft_config) is ReferenceMode.LIVE_BASE
        )

    @classmethod
    def _kl_reference(cls, model, ref_model, kl_beta: float, parallelism_config: ParallelismConfig, peft_config):
        """An explicit reference to sweep, or the native-expert-LoRA base kept live."""
        if not cls._holds_kl_reference(model, kl_beta, parallelism_config, peft_config):
            if ref_model is not None:
                raise ValueError(
                    "ref_model was passed, but this run holds no KL reference model (kl_beta 0, a PEFT policy "
                    "scored with its adapters disabled, or PP, which scores the reference through the "
                    "policy's pipeline), so it would never be read. Drop ref_model."
                )
            return None
        if ref_model is not None:
            return ref_model
        if cls._reference_mode(model, parallelism_config, peft_config) is ReferenceMode.LIVE_BASE:
            raise ValueError(
                "kl_beta > 0 with native expert LoRA needs ref_model loaded through "
                "load_frozen_reference_model from the unadapted base with the policy's revision, "
                "attention and sinks settings. Or use PEFT, or set kl_beta: 0."
            )
        return None

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

    def _precompute_reference_logps(self, dataset, split: str):
        """Restore original KL scores or sweep the untrained policy once before any update."""
        settings = self._reference_settings()
        identity = self._reference_split_identity(dataset, split, settings)
        restored = self._restore_reference_logps_or_none(dataset, split, identity=identity)
        if restored is not None:
            return restored
        rows = self._sweep_reference_logps(dataset, split)
        return self._attach_scored_reference_logps(dataset, split, rows, identity=identity)

    def _reference_settings(self) -> dict:
        """Values that affect run-start raw reference scores, excluding the live KL clamp."""
        precision = resolve_training_dtype(self.args) if self._scores_full_logits else torch.float32
        return {
            "model_type": self.model.config.model_type,
            "temperature": self.temperature,
            LOGPROB_PRECISION_KEY: str(precision).removeprefix("torch."),
        }

    def _restore_reference_logps_or_none(self, dataset: Dataset, split: str, *, identity: Mapping) -> Dataset | None:
        reject_across_ranks(
            None if isinstance(dataset, Dataset) else f"'{split}' must be a finite datasets.Dataset",
            "Validating offline GRPO reference dataset",
            exc_type=ValueError,
        )
        self._verify_reference_resume_context()
        reject_divergent_settings(
            {"split": split, **identity},
            "Offline GRPO reference inputs",
            "The same unsharded tokenized split and settings must reach every rank.",
        )
        return self._restore_reference_split(dataset, split, (REF_PER_TOKEN_LOGPS_COLUMN,), identity)

    def _sweep_reference_logps(self, dataset, split: str) -> MappedReferenceScores:
        """COLLECTIVE. Score contiguous DP shards of ``dataset`` in every mode and rebuild its row order."""
        dp_size, dp_rank = self.dp_shard_geometry()
        loader = self._reference_sweep_loader(dataset, split, dp_size, dp_rank)
        # FSDP and EP peers may span DP shards; every rank must run the same number of forwards.
        (total_batches,) = rank_max([len(loader)])
        representatives = dict(enumerate(dp_representative_ranks(self._data_parallel_rank_by_global_rank())))
        cache = ReferenceScoreCache(self._reference_cache_output_dir(), dp_size=dp_size)
        was_training = self.model.training
        if is_global_main_process():
            logger.info(
                "Preparing run-start KL reference for '%s': %s rows, %s batches before training",
                split,
                len(dataset),
                total_batches,
            )
        try:
            self.model.eval()
            with torch.no_grad():
                self._score_reference_batches(
                    loader, cache, total_batches, split, representatives, get_global_rank() == representatives[dp_rank]
                )
            return cache.finish(dataset)
        except BaseException as exc:
            try:
                cache.discard()
            except Exception as cleanup_error:
                exc.add_note(f"Reference cache cleanup also failed: {cleanup_error}")
            raise
        finally:
            self.model.train(was_training)

    def _reference_sweep_loader(self, dataset, split: str, dp_size: int, dp_rank: int) -> DataLoader:
        """COLLECTIVE. This DP rank's contiguous shard of ``dataset`` in full batches of ``(batch, real rows)``;
        refuses, on every rank, a dataset that is not finite or leaves some rank no rows."""
        reject_across_ranks(
            None if isinstance(dataset, Dataset) else f"Reference sweep needs a finite Dataset for '{split}'",
            f"Preparing the '{split}' reference sweep",
            exc_type=ValueError,
        )
        rows = len(dataset)
        batch_size = self.args.per_device_train_batch_size

        def collate(batch):
            # Padded to a full batch so FSDP/EP per-row forward counts stay equal; only the real rows are cached.
            return self.data_collator(batch + [batch[-1]] * (batch_size - len(batch))), len(batch)

        guard = DeferredRankFailure(f"Preparing the '{split}' reference loader", exc_type=ValueError)
        loader = guard.run(
            lambda: DataLoader(
                dataset.select(range(dp_rank * rows // dp_size, (dp_rank + 1) * rows // dp_size)),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate,
            )
        )
        guard.reject()
        # A rank with no batch has nothing to replay, while its peers head into the batch-count
        # all-reduce: refuse on every rank before it.
        reject_across_ranks(
            None
            if len(loader)
            else f"the '{split}' dataset leaves data-parallel rank {dp_rank} no rows to score the KL reference "
            f"over ({rows} rows across {dp_size} replicas); the dataset is too small for this world size",
            f"Preparing the '{split}' reference sweep",
            exc_type=ValueError,
        )
        return loader

    def _score_reference_batches(
        self, loader, cache, total_batches: int, split: str, representatives: dict[int, int], is_representative: bool
    ) -> None:
        """COLLECTIVE. ``total_batches`` forwards on every rank, a rank past its own batches replaying its
        last; each DP rank's representative hands its real rows to ``cache``."""
        iterator = iter(loader)
        progress_interval = max(1, total_batches // 10)
        last_batch = None
        real_rows = 0
        for index in range(total_batches):
            actual_batch = index < len(loader)
            guard = DeferredRankFailure(f"Preparing '{split}' reference batch {index}")
            loaded = guard.run(lambda: next(iterator)) if actual_batch else None
            guard.reject()
            guard = DeferredRankFailure(f"Placing '{split}' reference batch {index}")
            if actual_batch:
                batch, real_rows = loaded
                last_batch = guard.run(lambda batch=batch: self._prepare_inputs(batch))
            guard.reject()
            # The forward owns collectives: swallowing a rank's exception would strand
            # peers inside them while that rank enters a different consensus collective.
            scored_rows = self._score_reference_batch(last_batch)
            cache.collect_batch(
                scored_rows[:real_rows] if actual_batch and is_representative else None, representatives
            )
            if is_global_main_process() and ((index + 1) % progress_interval == 0 or index + 1 == total_batches):
                logger.info("Run-start KL reference '%s': batch %s/%s", split, index + 1, total_batches)

    def _attach_scored_reference_logps(
        self, dataset: Dataset, split: str, rows: MappedReferenceScores, *, identity: Mapping
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

    def _reference_resume_required(self) -> bool:
        return self._policy_from_checkpoint

    def _reference_cache_output_dir(self) -> str:
        return os.fspath(self.args.output_dir)

    def _saved_reference_file(self, checkpoint: str) -> str:
        return link_checkpoint_into_scratch(
            self._reference_cache_output_dir(), super()._saved_reference_file(checkpoint)
        )

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
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", *, original_reference_model=None
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
        if rank_consensus(stored is not None).all_ok:
            scores = stored
        else:
            scores = self._score_evaluation_rows(dataset, identity, original_reference_model)
            self._evaluation_references.append((identity, dataset, scores))
        guard = DeferredRankFailure("Attaching original evaluation reference scores", exc_type=ValueError)
        attached = guard.run(lambda: attach_reference_column(dataset, scores))
        guard.reject()
        return attached

    def _validate_evaluation_reference_model(self, model) -> None:
        reason = None
        if not isinstance(model, torch.nn.Module):
            reason = "original_reference_model must be the exact original frozen policy model"
        elif model is self.model or base_transformers_model(model) is base_transformers_model(self.model):
            reason = "The trained/live policy cannot be original_reference_model"
        elif model.training or any(parameter.requires_grad for parameter in model.parameters()):
            reason = "original_reference_model must be frozen (requires_grad=False) and in eval mode"
        elif self.parallelism_config.is_cp_mode or self._pp_runner is not None:
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
        missing = rank_consensus(any(match is None for match in matches)).any_ok
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
                if is_fs_save_rank():
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
