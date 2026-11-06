"""Packed micro-batches of the environmental trainer: a round's plan, its slot layout over TRL's buffer, the slot's rows
gathered for the loss forward, the no-op slots that precede a window's real ones and the row-mean loss rescale."""

import contextlib
import logging

import torch

from halo.models.attention.selection import effective_attn_implementation
from halo.models.moe.balancing import iter_balancing_routers
from halo.runtime.collectives import barrier
from halo.runtime.ranks import is_global_main_process
from halo.trainers.grpo.environmental.environmental_rules import (
    reject_undispatchable_budget,
    reject_unpackable_model,
    reject_unpackable_options,
    resolve_dispatch_ceiling,
)
from halo.trainers.grpo.environmental.row_order import ROW_MEAN_LOSS_TYPES, replacing_trl_shuffle
from halo.trainers.grpo.environmental.step_batch import ROUTING_MASKS_KEY, StepRows
from halo.trainers.grpo.microbatch_plan import (
    MIN_ROW_TOKENS,
    MicrobatchPlan,
    agree_microbatch_geometry,
    build_slot_layout,
)

logger = logging.getLogger(__name__)

# Row-first tensors of the buffer TRL splits into ``steps_per_generation`` equal chunks: one entry per slot row,
# ``-1`` marking a filler row, and the slot's activity repeated down its rows.
SLOT_ROWS_KEY = "slot_rows"
SLOT_ACTIVE_KEY = "slot_active"
# Host values of the slot a micro-step trains on, read by ``compute_loss``.
SLOT_IS_ACTIVE_KEY = "packed_slot_active"
LOSS_SCALE_KEY = "packed_loss_scale"
# What a filler row holds per round key when it is not zero: one attended prompt token (no NaN) and a fully masked
# completion, as the rows that pad a round across ranks.
_FILLER_VALUES = {"importance_sampling_ratio": 1, ROUTING_MASKS_KEY: -1}


class PackedMicrobatchMixin:
    """Mixed in ahead of ``LengthOrderedRowsMixin`` (whose deal it replaces) and ``GenerationRoundMixin``. The host
    calls :meth:`_configure_packed_microbatches` once the model is placed, :meth:`_plan_packed_round` and
    :meth:`_lay_out_packed_round` from the train round's batch build, and routes ``compute_loss`` through
    :meth:`_is_noop_slot` and :meth:`_scale_packed_loss`."""

    _round_store: dict | None = None
    _packed_dispatch_ceiling: int | None = None

    def _configure_packed_microbatches(self) -> None:
        """Refuse at construction every setup the packed path cannot hold, then turn it on."""
        budget = self.async_config.packed_microbatch_tokens
        if budget is None:
            return
        reject_unpackable_options(self.args, self.async_config, misaligned_accumulation=self._misaligned_accumulation)
        model = self.accelerator.unwrap_model(self.model)
        reject_unpackable_model(model, effective_attn_implementation(model.config))
        reject_undispatchable_budget(self.parallelism_config, model.config, budget)
        self._packed_dispatch_ceiling = resolve_dispatch_ceiling(self.parallelism_config, model.config)
        # The plan orders the round's rows, and each micro-batch's spans are read from its own mask.
        self._orders_rows_by_length = False
        self._stashes_row_spans = False
        self._packs_microbatches = True
        self._warm_packed_row_kernels()
        if is_global_main_process():
            logger.info(
                f"Packed micro-batches: up to {budget} attended tokens per rank in each of at most "
                f"{self.args.steps_per_generation} slots."
            )

    def _warm_packed_row_kernels(self) -> None:
        """COLLECTIVE. A two-document packed forward and backward through the trainer's own scorer, so the varlen
        attention and the delta-rule and conv variants a packed call launches compile and load here, on every rank,
        and not between a MoE dispatch and its combine in the first step. The gradients it leaves are dropped, and the
        expert loads it counted are reset so no bias update sees its tokens."""
        if not torch.cuda.is_available():
            return
        model, device, keep = self.model, self.accelerator.device, 4
        ids = torch.zeros(2, 2 * keep, dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        was_training = model.training
        model.train()
        try:
            logps, _, _ = self._get_per_token_logps_and_entropies(model, ids, mask, keep, compute_entropy=False)
            logps.sum().backward()
            self._finish_fsdp2_backward(model)
        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)
            for router in iter_balancing_routers(model):
                router.expert_load_counter = None
        barrier()

    def _plan_packed_round(self, step: StepRows, num_dummy_rows: int) -> MicrobatchPlan:
        """COLLECTIVE. The round's real rows planned into bins, the world's bin count and widest bin agreed."""
        attended = step.attention_mask.sum(dim=1).clamp(min=MIN_ROW_TOKENS)
        real_rows = attended.numel() - num_dummy_rows
        return agree_microbatch_geometry(
            attended[:real_rows].tolist(),
            self.async_config.packed_microbatch_tokens,
            slots=self.args.steps_per_generation,
            ceiling=self._packed_dispatch_ceiling,
            round_rows=attended.numel(),
        )

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
            if value is None or value.ndim == 0:
                view[key] = value
                continue
            filler = self.pad_token_id if key in ("prompt_ids", "completion_ids") else _FILLER_VALUES.get(key, 0)
            gathered = value.new_full((len(rows), *value.shape[1:]), filler)
            gathered[targets] = value[sources]
            view[key] = gathered
        fillers = torch.tensor([i for i, row in enumerate(rows) if row < 0], dtype=torch.long, device=device)
        view["prompt_mask"][fillers, -1] = True
        steps = self.args.steps_per_generation
        view[LOSS_SCALE_KEY] = len(rows) * steps / round_rows if self.loss_type in ROW_MEAN_LOSS_TYPES else 1.0
        return view

    def _is_noop_slot(self, model, inputs) -> bool:
        return model.training and isinstance(inputs, dict) and inputs.get(SLOT_IS_ACTIVE_KEY) is False

    def _noop_loss(self) -> torch.Tensor:
        """A leaf with no edge into the model: no forward, no collective, and a backward every rank still calls."""
        return torch.zeros((), device=self.accelerator.device, requires_grad=True)

    def _scale_packed_loss(self, loss: torch.Tensor, inputs) -> torch.Tensor:
        """Row-mean losses divide a slot by its own row count, which a plan changes from the legacy ``R / S``."""
        scale = inputs.get(LOSS_SCALE_KEY) if isinstance(inputs, dict) else None
        return loss if scale is None else loss * scale

    def _score_inert_row(self, model, logits_to_keep: int, device: torch.device) -> None:
        """A bin this rank holds no row for still takes part in the forward every rank runs: one attended token, routed
        naturally and left out of any capture or replay."""
        ids = torch.full((1, 1 + logits_to_keep), self.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        mask[0, 0] = 1
        paused = self._routing_injector.paused() if self._routing_injector is not None else contextlib.nullcontext()
        with paused:
            self._get_per_token_logps_and_entropies(model, ids, mask, logits_to_keep, compute_entropy=False)
