"""Packed micro-batches of the on-policy GRPO trainers: a round's plan, its slot layout over TRL's buffer, the slot's rows
gathered for the loss forward, the no-op slots that precede a window's real ones, the row-mean loss rescale and the
round-wide scoring passes run bin by bin."""

import contextlib
import logging
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from typing import ClassVar

import torch

from halo.models.attention.selection import effective_attn_implementation
from halo.runtime.ranks import is_global_main_process
from halo.trainers.grpo.microbatch_plan import MicrobatchPlan, build_slot_layout, plan_round_microbatches
from halo.trainers.grpo.objective.loss_inputs import IMPORTANCE_RATIO_KEY
from halo.trainers.grpo.packed_rules import (
    PACKED_KNOB,
    reject_undispatchable_budget,
    reject_unpackable_model,
    reject_unpackable_options,
    resolve_dispatch_ceiling,
)
from halo.trainers.grpo.row_order import ROW_MEAN_LOSS_TYPES, replacing_trl_shuffle

logger = logging.getLogger(__name__)

# Row-first tensors of the buffer TRL splits into ``steps_per_generation`` equal chunks: one entry per slot row,
# ``-1`` marking a filler row, and the slot's activity repeated down its rows.
SLOT_ROWS_KEY = "slot_rows"
SLOT_ACTIVE_KEY = "slot_active"
# Host values of the slot a micro-step trains on, read by ``compute_loss``.
SLOT_IS_ACTIVE_KEY = "packed_slot_active"
LOSS_SCALE_KEY = "packed_loss_scale"


class PackedRoundMixin:
    """Mixed in ahead of ``GenerationRoundMixin`` and ``ProtectedTokenEntropyMixin``. The host provides
    ``_packed_microbatch_tokens`` and ``pad_token_id``, calls :meth:`_configure_packed_microbatches` once the model is
    placed, wraps a train round's build in :meth:`_scoring_round_in_bins` where TRL scores the whole round, and ends the
    build in :meth:`_lay_out_packed_round`."""

    # Why this trainer cannot pack its rows; None admits it. A trainer with a budget reads it before building TRL's
    # state, through :func:`reject_trainer_refusal`.
    PACKED_ROWS_REFUSAL: ClassVar[str | None] = None
    # What a filler row holds per round key when it is not zero: one attended prompt token (no NaN) and a fully masked
    # completion, as the rows that pad a round across ranks.
    PACKED_FILLER_VALUES: ClassVar[dict[str, float]] = {IMPORTANCE_RATIO_KEY: 1}

    _packed_microbatch_tokens: int | None = None
    _packed_dispatch_ceiling: int | None = None
    _round_store: dict | None = None
    # The plan of the round being built, agreed by the first scoring call that needs it or by the layout.
    _round_plan: MicrobatchPlan | None = None
    _bins_round_scoring = False

    def _configure_packed_microbatches(self, *, length_ordered: bool | None = None) -> None:
        """Refuse at construction every setup the packed path cannot hold, then turn it on. ``length_ordered`` is the
        trainer's ``length_ordered_microbatches`` (``None`` where it has none)."""
        budget = self._packed_microbatch_tokens
        if budget is None:
            return
        reject_unpackable_options(
            self.args,
            use_chunked_logprobs=self._use_chunked_grpo_logprobs,
            length_ordered=length_ordered,
            misaligned_accumulation=self._misaligned_accumulation,
        )
        model = self.accelerator.unwrap_model(self.model)
        reject_unpackable_model(model, effective_attn_implementation(model.config))
        reject_undispatchable_budget(self.parallelism_config, budget)
        self._packed_dispatch_ceiling = resolve_dispatch_ceiling(self.parallelism_config)
        # The plan orders the round's rows, and each micro-batch's spans are read from its own mask.
        self._stashes_row_spans = False
        self._packs_microbatches = True
        self._warm_packed_row_kernels()
        if is_global_main_process():
            logger.info(
                f"Packed micro-batches: up to {budget} attended tokens per rank in each of at most "
                f"{self.args.steps_per_generation} slots."
            )

    def _warm_packed_row_kernels(self) -> None:
        """COLLECTIVE. A two-document packed forward and backward, so the varlen attention and the delta-rule and conv
        variants a packed call launches compile and load at construction (:meth:`_warm_scoring_pass`)."""
        if not torch.cuda.is_available():
            return
        keep = 4
        ids = torch.zeros(2, 2 * keep, dtype=torch.long, device=self.accelerator.device)
        self._warm_scoring_pass(ids, keep, compute_entropy=False)

    def _plan_round(self, attention_mask: torch.Tensor, num_dummy_rows: int = 0) -> MicrobatchPlan:
        """COLLECTIVE. The round's real rows planned into bins, the world's bin count and widest bin agreed."""
        return plan_round_microbatches(
            attention_mask,
            num_dummy_rows,
            budget=self._packed_microbatch_tokens,
            slots=self.args.steps_per_generation,
            ceiling=self._packed_dispatch_ceiling,
        )

    @contextlib.contextmanager
    def _scoring_round_in_bins(self) -> Iterator[None]:
        """For a train round's build where TRL scores the whole round in one call per pass: those calls plan the rows
        they are handed (COLLECTIVE, once, as TRL's gate for each pass is config-derived and every rank takes it alike)
        and score bin by bin, returning the log-probs in batch order. :meth:`_take_round_plan` hands the plan on."""
        if not (self._packs_microbatches and self.model.training):
            yield
            return
        self._round_plan, self._bins_round_scoring = None, True
        try:
            yield
        finally:
            self._bins_round_scoring = False

    @contextlib.contextmanager
    def unbinned_scoring(self) -> Iterator[None]:
        """A scoring call inside this scope is no part of the round being built (a weight-sync probe forward, which
        TRL's generate step runs mid-build): it runs as one plain call and leaves the round's plan alone."""
        binned, self._bins_round_scoring = self._bins_round_scoring, False
        try:
            yield
        finally:
            self._bins_round_scoring = binned

    def _take_round_plan(self, batch: dict) -> MicrobatchPlan:
        """COLLECTIVE when no scoring call planned the round (no pass ran). The round's plan, consumed: the next round
        plans its own."""
        plan, self._round_plan = self._round_plan, None
        if plan is None:
            plan = self._plan_round(torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1))
        return plan

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, *args, **kwargs):
        if not self._bins_round_scoring:
            return super()._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep, *args, **kwargs
            )
        carried = [
            key for key, value in kwargs.items() if key not in ("batch_size", "compute_entropy") and value is not None
        ]
        if kwargs.get("compute_entropy") or carried:
            raise RuntimeError(
                f"A packed round's scoring pass returns log-probs of text rows only, but this call asked for "
                f"{'entropies' if kwargs.get('compute_entropy') else carried}."
            )
        if self._round_plan is None:
            self._round_plan = self._plan_round(attention_mask)
        plan = self._round_plan
        if plan.round_rows != input_ids.size(0):
            raise RuntimeError(
                f"The round was planned over {plan.round_rows} rows but a scoring pass of it holds {input_ids.size(0)}."
            )
        order = torch.tensor(plan.order, dtype=torch.long, device=input_ids.device)
        # Each bin is a scoring call of its own, which must not be binned again.
        self._bins_round_scoring = False
        try:
            logps = self._score_bins(model, input_ids[order], attention_mask[order], logits_to_keep, plan)
        finally:
            self._bins_round_scoring = True
        return logps, None, None

    def _score_bins(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        plan: MicrobatchPlan,
        bin_context: Callable[[slice, torch.Tensor], AbstractContextManager[None]] | None = None,
    ) -> torch.Tensor:
        """``model``'s per-token log-probs of rows already taken in ``plan.order``, one scoring call per bin (an empty
        bin an inert row), put back in batch order; the rows no bin holds score 0. ``bin_context`` wraps a bin's call,
        given its rows and their attention mask."""
        scored, start = [], 0
        for size in plan.bin_sizes:
            if size == 0:
                self._score_inert_row(model, logits_to_keep, input_ids.device)
                continue
            bin_rows = slice(start, start + size)
            start += size
            context = (
                contextlib.nullcontext() if bin_context is None else bin_context(bin_rows, attention_mask[bin_rows])
            )
            with context:
                logps, _, _ = self._get_per_token_logps_and_entropies(
                    model, input_ids[bin_rows], attention_mask[bin_rows], logits_to_keep, compute_entropy=False
                )
            scored.append(logps)
        in_batch_order = torch.zeros(
            plan.round_rows,
            logits_to_keep,
            dtype=scored[0].dtype if scored else torch.float32,
            device=input_ids.device,
        )
        if scored:
            in_batch_order[torch.tensor(plan.order, dtype=torch.long, device=input_ids.device)] = torch.cat(scored)
        return in_batch_order

    def _score_inert_row(self, model, logits_to_keep: int, device: torch.device) -> None:
        """A bin this rank holds no row for still takes part in the forward every rank runs: one attended token, routed
        naturally and left out of any capture or replay."""
        ids = torch.full((1, 1 + logits_to_keep), self.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        mask[0, 0] = 1
        with self._routing_replay_paused():
            self._get_per_token_logps_and_entropies(model, ids, mask, logits_to_keep, compute_entropy=False)

    def _routing_replay_paused(self) -> AbstractContextManager[None]:
        """A trainer that records or replays routing pauses it here."""
        return contextlib.nullcontext()

    def _lay_out_packed_round(self, round_tensors: dict, plan: MicrobatchPlan) -> dict:
        """COLLECTIVE under TP/ETP. Keep the round's tensors, as built, in the store and hand TRL only the slot index
        tensors: gathering every slot's rows up front would materialize filler rows the widest bin multiplies."""
        num_items = round_tensors.pop("num_items_in_batch")
        self._round_store = round_tensors
        # Siblings must train on the leader's tensors; TRL's own broadcast of the batch covers only the indices.
        self._broadcast_tensors_from_tp_leader(round_tensors)
        slot_rows, slot_active = build_slot_layout(plan)
        index_device = self.accelerator.device if self._get_tp_or_etp_process_group() is not None else "cpu"
        return {
            SLOT_ROWS_KEY: torch.tensor(slot_rows, dtype=torch.long, device=index_device),
            SLOT_ACTIVE_KEY: torch.tensor(slot_active, dtype=torch.long, device=index_device),
            "num_items_in_batch": num_items,
        }

    def _prepare_inputs(self, generation_batch):
        """TRL's buffering with the identity in place of its shuffle on a train round's refill, then the slot's rows
        gathered from the round store."""
        if not (self._packs_microbatches and self.model.training):
            return super()._prepare_inputs(generation_batch)
        buffered = self._buffered_inputs
        with replacing_trl_shuffle(lambda batch: batch) as kept:
            inputs = super()._prepare_inputs(generation_batch)
        if self._buffered_inputs is not buffered and not kept:
            raise RuntimeError(
                "TRL refilled its generation buffer without calling shuffle_sequence_dict, which packed micro-batches "
                "keep the identity for: the installed TRL moved its buffer shuffle. Unset packed_microbatch_tokens "
                "until the slot layout follows it."
            )
        return self._build_slot_view(inputs)

    def _build_slot_view(self, slot: dict) -> dict:
        """The micro-step's ``[rows_per_bin, width]`` view of the round: real rows by index, filler rows inert. A no-op
        slot carries only its flag."""
        rows = slot[SLOT_ROWS_KEY].tolist()
        if not slot[SLOT_ACTIVE_KEY][0]:
            return {SLOT_IS_ACTIVE_KEY: False}
        store = self._round_store
        positions = [i for i, row in enumerate(rows) if row >= 0]
        device = store["advantages"].device
        sources = torch.tensor([rows[i] for i in positions], dtype=torch.long, device=device)
        targets = torch.tensor(positions, dtype=torch.long, device=device)
        round_rows = store["advantages"].size(0)
        view: dict = {"num_items_in_batch": slot["num_items_in_batch"], SLOT_IS_ACTIVE_KEY: True}
        for key, value in store.items():
            if value is None or (isinstance(value, torch.Tensor) and value.ndim == 0):
                view[key] = value
                continue
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"The round's {key!r} is a {type(value).__name__}, which packed micro-batches cannot slot by row. "
                    f"Unset {PACKED_KNOB}."
                )
            filler = (
                self.pad_token_id if key in ("prompt_ids", "completion_ids") else self.PACKED_FILLER_VALUES.get(key, 0)
            )
            gathered = value.new_full((len(rows), *value.shape[1:]), filler)
            gathered[targets] = value[sources]
            view[key] = gathered
        fillers = torch.tensor([i for i, row in enumerate(rows) if row < 0], dtype=torch.long, device=device)
        view["prompt_mask"][fillers, -1] = True
        steps = self.args.steps_per_generation
        view[LOSS_SCALE_KEY] = len(rows) * steps / round_rows if self.loss_type in ROW_MEAN_LOSS_TYPES else 1.0
        return view

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """A no-op slot of a window returns before any forward; a real one's loss is rescaled for its row count."""
        if self._is_noop_slot(model, inputs):
            return self._noop_loss()
        loss = super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )
        return self._scale_packed_loss(loss, inputs)

    def _is_noop_slot(self, model, inputs) -> bool:
        return model.training and isinstance(inputs, dict) and inputs.get(SLOT_IS_ACTIVE_KEY) is False

    def _noop_loss(self) -> torch.Tensor:
        """A leaf with no edge into the model: no forward, no collective, and a backward every rank still calls."""
        return torch.zeros((), device=self.accelerator.device, requires_grad=True)

    def _scale_packed_loss(self, loss: torch.Tensor, inputs) -> torch.Tensor:
        """Row-mean losses divide a slot by its own row count, which a plan changes from the legacy ``R / S``."""
        scale = inputs.get(LOSS_SCALE_KEY) if isinstance(inputs, dict) else None
        return loss if scale is None else loss * scale
