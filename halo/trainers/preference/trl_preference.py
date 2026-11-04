"""The TRL DPO/KTO integration both preference trainers share. TRL's reference cache keys off rank-sharded parameter
hashes and a rank-0 Arrow file, both wrong under EP/TP or per-node storage, so the sweep runs on the DP axis and every
rank attaches the columns in memory."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import ClassVar

import torch
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from trl.trainer.utils import (
    entropy_from_logits,
    hash_module,
    selective_log_softmax,
    selective_log_softmax_and_entropy,
)

from halo.checkpoint.format import REFERENCE_LOGPS_FILE
from halo.kernels.logprobs import selective_logprobs
from halo.runtime.ranks import is_global_main_process
from halo.trainers.ctor_slots import get_ctor_arg, get_ctor_config, resolve_ctor_positions
from halo.trainers.preference.preference_rules import reject_policy_from_checkpoint_without_resume
from halo.trainers.reference_logps import (
    LOGPROB_PRECISION_KEY,
    ReferenceLogpsMixin,
    is_reference_entry,
    is_token_type,
    reference_regeneration_steps,
    token_digest,
)
from halo.trainers.training_args_overrides import disable_trl_liger

logger = logging.getLogger(__name__)


def _fp32_logprobs_and_entropy(logits, index, entropy_requires_grad=True, temperature=1.0, row_mask=None):
    """TRL's ``selective_log_softmax_and_entropy`` with its log-probs from :func:`selective_logprobs`."""
    if temperature != 1.0:
        logits = logits / temperature
    logps = selective_logprobs(logits, index, row_mask=row_mask)
    with torch.set_grad_enabled(entropy_requires_grad and torch.is_grad_enabled()):
        entropy = entropy_from_logits(logits)
    if row_mask is not None:
        entropy = entropy.masked_fill(row_mask == 0, 0.0)
    return logps, entropy


# (the name TRL's trainer modules import a log-prob function under, TRL's function, its fp32 stand-in).
_FP32_REBINDS = (
    (selective_log_softmax.__name__, selective_log_softmax, selective_logprobs),
    (selective_log_softmax_and_entropy.__name__, selective_log_softmax_and_entropy, _fp32_logprobs_and_entropy),
)


@contextmanager
def _fp32_logprobs(trl_method: Callable) -> Iterator[None]:
    """Rebind TRL's log-prob functions to their fp32 stand-ins in the globals ``trl_method`` resolves
    them from, while the block runs."""
    namespace = trl_method.__func__.__globals__
    if all(namespace.get(name) is fp32 for name, _, fp32 in _FP32_REBINDS):
        # An enclosing call already rebound them and restores them on exit.
        yield
        return
    stale = [name for name, trl_fn, _ in _FP32_REBINDS if namespace.get(name) is not trl_fn]
    if stale:
        raise RuntimeError(
            f"{trl_method.__qualname__} no longer reads TRL's {', '.join(stale)} from "
            f"{namespace.get('__name__')}, so its log-probs cannot be routed through the fp32 path. "
            f"Re-audit TRLPreferenceMixin against the installed TRL."
        )
    for name, _, fp32 in _FP32_REBINDS:
        namespace[name] = fp32
    try:
        yield
    finally:
        for name, trl_fn, _ in _FP32_REBINDS:
            namespace[name] = trl_fn


def _trl_trainer_class(cls: type) -> type | None:
    return next(
        (base for base in cls.__mro__ if base.__module__.startswith("trl.") and "__init__" in vars(base)), None
    )


def _attach_reference_columns(dataset: Dataset, columns: Mapping[str, torch.Tensor]) -> Dataset:
    appended = Dataset.from_dict({name: values.float().numpy() for name, values in columns.items()})
    return concatenate_datasets([dataset, appended], axis=1)


class TRLPreferenceMixin(ReferenceLogpsMixin):
    """Listed first in the bases: ahead of ``DistributedTrainerMixin`` so its checkpoint hook wins, ahead of the TRL
    base so the fp32 rebinds wrap TRL's loss and reference pass. A subclass declares ``REFERENCE_COLUMNS``,
    ``LIGER_DISABLE_REASON`` and ``_reference_settings``."""

    # Recorded in every saved reference split, so one summed at another precision is not reused on resume.
    LOGPROB_PRECISION: ClassVar[str] = "float32"
    # The columns TRL's reference sweep writes, in its output order; present in a dataset, the sweep is skipped.
    REFERENCE_COLUMNS: ClassVar[tuple[str, ...]]
    LIGER_DISABLE_REASON: ClassVar[str]
    CTOR_POSITIONS: ClassVar[dict[str, int]]
    # TRL's loss adds the router aux term itself and neither trainer overrides it.
    ROUTER_AUX_LOSS_ADDED_BY = "trl_loss"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        trl_trainer = _trl_trainer_class(cls)
        if trl_trainer is not None:
            cls.CTOR_POSITIONS = resolve_ctor_positions(trl_trainer, "model", "args", "ref_model")

    def __init__(self, *args, **kwargs):
        # Must precede _init_distributed_config.
        disable_trl_liger(get_ctor_config(args, kwargs, self.CTOR_POSITIONS), self.LIGER_DISABLE_REASON)
        self._init_reference_resume(kwargs)
        kwargs = self._init_distributed_config(kwargs, ctor_args=args, ctor_positions=self.CTOR_POSITIONS)
        self._validate_reference_model(get_ctor_arg(args, kwargs, "ref_model", self.CTOR_POSITIONS))
        with self._trl_reference_hash_skipped():
            super().__init__(*args, **kwargs)
        # TRL builds its own reference inside __init__ when none is passed, the model is not PEFT-wrapped and
        # precompute_ref_log_probs is off.
        self._validate_held_reference_model()
        self._configure_distributed_modes()

    def compute_ref_log_probs(self, model, inputs):
        trl_compute_ref_log_probs = super().compute_ref_log_probs
        with _fp32_logprobs(trl_compute_ref_log_probs):
            return trl_compute_ref_log_probs(model, inputs)

    def _init_reference_resume(self, kwargs: dict) -> None:
        reject_policy_from_checkpoint_without_resume(kwargs)
        self._init_reference_state(
            given="resume_checkpoint" in kwargs,
            checkpoint=kwargs.pop("resume_checkpoint", None),
            policy_from_checkpoint=kwargs.pop("policy_from_checkpoint", False),
        )

    @contextmanager
    def _trl_reference_hash_skipped(self) -> Iterator[None]:
        """With ``precompute_ref_log_probs`` on, TRL hashes the whole reference in ``__init__``, a CPU copy of every
        tensor on every rank and a raise on a DTensor (TP), only to key the sweep this mixin replaces."""
        trl_module = _trl_trainer_class(type(self)).__module__
        namespace = vars(sys.modules[trl_module])
        hash_fn = hash_module.__name__
        if namespace.get(hash_fn) is not hash_module:
            raise RuntimeError(
                f"{trl_module} no longer binds TRL's {hash_fn}, so its construction-time "
                f"reference hash cannot be skipped. Re-audit TRLPreferenceMixin "
                f"against the installed TRL."
            )
        namespace[hash_fn] = lambda module: None
        try:
            yield
        finally:
            namespace[hash_fn] = hash_module

    def _reference_columns(self) -> tuple[str, ...]:
        return self.REFERENCE_COLUMNS

    def _precompute_ref_logps(self, dataset, name, batch_size):
        needed = self._reference_columns()
        if all(column in (dataset.column_names or []) for column in needed):
            if is_global_main_process():
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
        self._verify_reference_resume_context()
        identity = self._reference_split_identity(
            dataset, name, {**self._reference_settings(), LOGPROB_PRECISION_KEY: self.LOGPROB_PRECISION}
        )
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
        model = self.ref_model if self.ref_model is not None else self.model
        parts: dict[str, list[torch.Tensor]] = {column: [] for column in needed}
        with self.data_parallel_sweep():
            for batch in tqdm(self.accelerator.prepare(loader), desc=f"Computing reference log probs for {name}"):
                outputs = tuple(value for value in self.compute_ref_log_probs(model, batch) if value is not None)
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
        ours = identity["settings"][LOGPROB_PRECISION_KEY]
        if not is_reference_entry(entry) or entry["settings"].get(LOGPROB_PRECISION_KEY) == ours:
            return super()._reference_mismatch_refusal(name, entry, identity, mismatch)
        return (
            f"Regenerate the '{name}' reference log-probs for this run: "
            f"{reference_regeneration_steps(self._reference_resume_checkpoint)}. The saved ones in "
            f"{self._reference_saved_path} were summed at "
            f"{entry['settings'].get(LOGPROB_PRECISION_KEY, 'unrecorded')} precision, and this run sums them in "
            f"{ours}, which no setting changes."
        )

    def _attach_reference_payload(self, dataset: Dataset, entry: Mapping, needed: Sequence[str]) -> Dataset:
        return _attach_reference_columns(dataset, {column: entry["columns"][column] for column in needed})

    def _remember_reference_split(self, name: str, identity: Mapping, payload: Mapping, dataset: Dataset) -> None:
        columns = {column: payload["columns"][column] for column in self._reference_columns()}
        super()._remember_reference_split(name, identity, {"columns": columns}, dataset)

    def _compute_loss(self, model, inputs, return_outputs):
        trl_compute_loss = super()._compute_loss
        with _fp32_logprobs(trl_compute_loss):
            return trl_compute_loss(model, inputs, return_outputs)
