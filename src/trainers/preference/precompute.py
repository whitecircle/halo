"""DPO/KTO reference sweep on the DP axis, with shared frozen-reference persistence.

TRL's reference cache uses rank-sharded parameter hashes and per-process dataset fingerprints,
which diverge under EP/TP. Its rank-0 Arrow file is also inaccessible on other nodes' local storage.
Each rank therefore attaches gathered columns in memory instead of reading another rank's file.

The loader and output gather both run on the DP axis: model-parallel siblings must forward the same
rows, or TP/ETP collectives see mismatched tokens. A Path-B resume builds the policy from trained
checkpoint weights before this sweep, so with no separate reference it must restore the run-start
scores from ReferenceLogpsCheckpointMixin's sidecar instead of scoring that policy again.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from accelerate.logging import get_logger
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.checkpoint.format import REFERENCE_LOGPS_FILE
from src.trainers.mixins.reference_logps import (
    ReferenceLogpsCheckpointMixin,
    is_reference_entry,
    is_token_type,
    reference_regeneration_steps,
    token_digest,
)

logger = get_logger(__name__, log_level="info")

# The settings key recording the precision the sweep summed a split's log-probs in.
_PRECISION_KEY = "logprob_precision"


def _attach_reference_columns(dataset: Dataset, columns: Mapping[str, torch.Tensor]) -> Dataset:
    appended = Dataset.from_dict({name: values.float().numpy() for name, values in columns.items()})
    return concatenate_datasets([dataset, appended], axis=1)


class PrecomputeRefLogpsRankConsistentMixin(ReferenceLogpsCheckpointMixin):
    """Run TRL's reference sweep on the DP axis and preserve each named split on resume."""

    # Declared by FP32LogprobsMixin; part of every saved split's identity.
    logprob_precision: str | None = None

    def _init_reference_resume(self, kwargs: dict) -> None:
        given = "resume_checkpoint" in kwargs
        checkpoint = kwargs.pop("resume_checkpoint", None)
        policy_from_checkpoint = kwargs.pop("policy_from_checkpoint", False)
        if policy_from_checkpoint and checkpoint is None:
            raise ValueError("policy_from_checkpoint=True needs the resume_checkpoint the policy was built from.")
        self._init_reference_state(checkpoint=checkpoint, given=given, policy_from_checkpoint=policy_from_checkpoint)

    def _required_ref_logps_columns(self) -> tuple[str, ...]:
        raise NotImplementedError(f"{type(self).__name__} must name its reference log-prob columns")

    def _reference_settings(self) -> dict:
        raise NotImplementedError(f"{type(self).__name__} must name the settings its reference depends on")

    def _reference_identity_settings(self) -> dict:
        """The settings a saved split records and must match: the run's knobs plus the precision the
        sweep sums its log-probs in."""
        if self.logprob_precision is None:
            raise NotImplementedError(
                f"{type(self).__name__} must declare the logprob_precision its reference sweep sums in "
                "(list FP32LogprobsMixin in its bases)"
            )
        return {**self._reference_settings(), _PRECISION_KEY: self.logprob_precision}

    def _precompute_ref_logps(self, dataset, name, batch_size):
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
        self._check_reference_resume_context()
        identity = self._reference_split_identity(dataset, name, self._reference_identity_settings())
        restored = self._restore_reference_split(dataset, name, needed, identity)
        if restored is not None:
            return restored
        columns = self._sweep_reference_logps(dataset, name, batch_size, needed)
        attached = _attach_reference_columns(dataset, columns)
        self._remember_reference_split(name, identity, {"columns": columns}, attached)
        return attached

    def _sweep_reference_logps(self, dataset, name: str, batch_size: int, needed) -> dict[str, torch.Tensor]:
        """TRL's sweep, gathering reference outputs into ordered rows on the DP axis."""
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
        self._set_signature_columns_if_needed()
        schema = dataset.features.arrow_schema
        columns = [
            column
            for column in self._signature_columns
            if column in dataset.column_names and is_token_type(schema.field(column).type)
        ]
        if not columns:
            raise RuntimeError(
                f"None of TRL's signature columns {self._signature_columns} is a token-id column of "
                f"the '{name}' dataset ({dataset.column_names}), so its reference log-probs could not "
                "be matched to its rows on resume."
            )
        return {column: token_digest(dataset, column) for column in columns}

    def _reference_payload_mismatch(self, entry: Mapping, dataset: Dataset, needed: Sequence[str]) -> str | None:
        columns = entry.get("columns")
        if not isinstance(columns, Mapping):
            return "its reference columns are malformed"
        missing = [column for column in needed if column not in columns]
        if missing:
            return f"it lacks {missing}, carrying only {sorted(columns)} (a changed loss_type?)"
        malformed = [
            column
            for column in needed
            if not isinstance(columns[column], torch.Tensor) or tuple(columns[column].shape) != (len(dataset),)
        ]
        return f"its {malformed} do not hold one value per row" if malformed else None

    def _reference_mismatch_refusal(self, name: str, entry: object, identity: Mapping, mismatch: str) -> str:
        """A split summed at another precision names only the regeneration: no setting selects it."""
        ours = identity["settings"][_PRECISION_KEY]
        if not is_reference_entry(entry) or entry["settings"].get(_PRECISION_KEY) == ours:
            return super()._reference_mismatch_refusal(name, entry, identity, mismatch)
        return (
            f"Regenerate the '{name}' reference log-probs for this run: "
            f"{reference_regeneration_steps(self._reference_resume_checkpoint)}. The saved ones in "
            f"{self._reference_saved_path} were summed at "
            f"{entry['settings'].get(_PRECISION_KEY, 'unrecorded')} precision, and this run sums them in "
            f"{ours}, which no setting changes."
        )

    def _attach_reference_payload(self, dataset: Dataset, entry: Mapping, needed: Sequence[str]) -> Dataset:
        return _attach_reference_columns(dataset, {column: entry["columns"][column] for column in needed})

    def _remember_reference_split(self, name: str, identity: Mapping, payload: Mapping, dataset: Dataset) -> None:
        columns = {column: payload["columns"][column] for column in self._required_ref_logps_columns()}
        super()._remember_reference_split(name, identity, {"columns": columns}, dataset)
