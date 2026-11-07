"""MoE routing replay (arXiv:2507.18071): top-k selections replayed in the update, so near-tie flips cannot train it
through other experts than the ratios saw. Gate weights re-derive from the live router, as replaying them zeroes its
gradient (RSPO, arXiv:2510.23027)."""

import base64
import contextlib
import io
import logging
import sys
from collections.abc import Iterable, Sequence
from typing import get_args

import numpy as np
import torch
from numpy.lib.format import MAGIC_PREFIX as NPY_MAGIC_PREFIX
from torch import nn

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase, named_ep_layers
from halo.models.config_access import text_config
from halo.models.module_tree import decoder_layer_index
from halo.models.unwrapping import unwrap_framework_wrappers
from halo.runtime.ranks import is_global_main_process
from halo.trainers.grpo.environmental.environmental_rules import reject_unsupported_routing_replay
from halo.trainers.grpo.environmental.rollout_routing import (
    ROLLOUT_COVERAGE_SHAPES,
    RolloutRouting,
    RowsReplay,
    build_rollout_routing,
)
from halo.trainers.grpo.environmental.step_batch import ROUTING_ROWS_KEY, StepRows
from halo.trainers.grpo.environmental.trajectory_tokenize import TurnRow
from halo.trainers.grpo.logprob_scoring import dense_row_spans, host_row_spans
from halo.trainers.grpo.row_span_stash import ROW_SPANS_KEY
from halo.trainers.world_metrics import world_sums

logger = logging.getLogger(__name__)


def decoder_layer_indices(names: Iterable[str]) -> list[int]:
    """Both engines key their per-token rows by decoder-layer index, and a family with dense first layers ships more
    rows than it has EP layers. An FQN outside ``layers.N`` raises."""
    indices = []
    for name in names:
        index = decoder_layer_index(name)
        if index is None:
            raise ValueError(f"EP layer {name!r} carries no decoder-layer index to match an engine row")
        indices.append(index)
    return indices


def decode_rollout_routing(payload: str, num_layers: int, top_k: int) -> torch.Tensor:
    """Int16 ``[tokens, L, K]``, from vLLM's base64 ``.npy`` or SGLang's base64 raw int32 rows, reshaped by the
    trainer's own shape so a disagreement raises here rather than misaligning the mask."""
    raw = base64.b64decode(payload)
    if raw.startswith(NPY_MAGIC_PREFIX):
        array = np.load(io.BytesIO(raw))
        if array.ndim != 3:
            raise ValueError(f"routed_experts payload has shape {array.shape}, expected [tokens, layers, top_k]")
    else:
        flat = np.frombuffer(raw, dtype=np.int32)
        if flat.size == 0 or flat.size % (num_layers * top_k):
            raise ValueError(
                f"raw routed_experts payload has {flat.size} int32 entries, not a positive multiple "
                f"of layers*top_k = {num_layers}*{top_k} — engine and trainer disagree on the MoE shape."
            )
        array = flat.reshape(-1, num_layers, top_k)
    return torch.from_numpy(np.ascontiguousarray(array.astype(np.int16, copy=False)))


def _require_spans_with_order(row_spans: list[tuple[int, int]] | None, row_order: torch.Tensor | None) -> None:
    if row_order is not None and row_spans is None:
        raise RuntimeError("routing replay got a row order without row spans: only per-row forwards are reordered")


class RoutingReplayInjector:
    """Capture/replay across EP MoE layers, in ``named_modules`` order (the mask's layer axis); ``engine_layers`` and
    ``layer_indices`` place them among the engine's per-decoder-layer rows."""

    def __init__(self, ep_layers: list[EPMoELayerBase], *, engine_layers: int, layer_indices: list[int]):
        if not ep_layers:
            raise ValueError("RoutingReplayInjector requires at least one EP MoE layer")
        if len(layer_indices) != len(ep_layers):
            raise ValueError(f"{len(layer_indices)} decoder-layer indices for {len(ep_layers)} EP layers")
        if len(set(layer_indices)) != len(layer_indices) or not all(0 <= i < engine_layers for i in layer_indices):
            raise ValueError(
                f"EP layer indices {layer_indices} must be distinct decoder layers below engine_layers={engine_layers}"
            )
        self._engine_layers = engine_layers
        self._layer_indices = layer_indices
        unsupported = sorted({type(m).__name__ for m in ep_layers if not m.FAMILY.supports_routing_replay})
        if unsupported:
            raise NotImplementedError(
                f"routing replay is not supported for {unsupported}: these families cannot re-derive "
                f"gate weights at a forced selection (see FAMILY.supports_routing_replay on each class)."
            )
        missing_top_k = sorted({type(m).__name__ for m in ep_layers if not hasattr(m, "top_k")})
        if missing_top_k:
            raise AttributeError(
                f"routing replay requires every EP layer to expose top_k, but {missing_top_k} never set "
                f"it — the family __init__ must call self._find_top_k(original_layer) (the injector sizes "
                f"the mask's top_k axis from it, and weight re-derivation reads it)."
            )
        self._layers = ep_layers
        self._armed = False

    @property
    def num_ep_layers(self) -> int:
        return len(self._layers)

    @property
    def top_k(self) -> int:
        return int(self._layers[0].top_k)

    @property
    def num_experts(self) -> int:
        return int(self._layers[0].num_experts)

    def decode_engine_mask(self, payload: str) -> torch.Tensor:
        """``[tokens, EP layers, top_k]`` from an engine payload carrying one row per decoder layer."""
        mask = decode_rollout_routing(payload, self._engine_layers, self.top_k)
        if mask.shape[1] != self._engine_layers:
            raise ValueError(
                f"routed_experts payload carries {mask.shape[1]} layer rows per token, the model has "
                f"{self._engine_layers} decoder layers — engine and trainer disagree on the MoE shape."
            )
        if self._layer_indices == list(range(self._engine_layers)):
            return mask
        return mask[:, self._layer_indices, :]

    @contextlib.contextmanager
    def capture(
        self,
        rows: int,
        seq_len: int,
        row_spans: list[tuple[int, int]] | None = None,
        row_order: torch.Tensor | None = None,
    ):
        """The yielded list holds the pass's :class:`RolloutRouting` on exit, ragged on the host: each forwarded row
        keeps only its tokens, and a forward pads the rows it uses. ``row_spans`` describes span-trimmed per-row
        forwards, ``row_order[p]`` the batch row the ``p``-th forward is. An error inside resets every layer and
        propagates, so the token-count check cannot shadow an OOM."""
        _require_spans_with_order(row_spans, row_order)
        targets = list(range(rows)) if row_order is None else row_order.tolist()
        for layer in self._layers:
            layer._capture_routing = True
            layer._captured_routing_chunks = []
        result: list[RolloutRouting] = []
        try:
            yield result
            expected = sum(hi - lo for lo, hi in row_spans) if row_spans is not None else rows * seq_len
            layer_chunks = [layer._captured_routing_chunks for layer in self._layers]
        finally:
            # Reset every layer on any exception path, else stale capture state leaks into the next forward.
            for layer in self._layers:
                layer._capture_routing = False
                layer._captured_routing_chunks = []
        flats = [torch.cat(chunks, dim=0) if chunks else None for chunks in layer_chunks]
        errors = [
            f"layer {index}: {0 if flat is None else flat.size(0)} tokens"
            for index, flat in enumerate(flats)
            if flat is None or flat.size(0) != expected
        ]
        if errors:
            raise RuntimeError(
                f"routing-replay capture recorded a wrong token count (expected {expected} = "
                f"{'Σ row spans' if row_spans is not None else f'rows*seq_len = {rows}*{seq_len}'}): "
                f"{'; '.join(errors)}. The capture forwards must tile exactly the declared row layout "
                f"(pass row_spans for span-trimmed per-row forwards)."
            )
        spans = row_spans if row_spans is not None else [(0, seq_len)] * rows
        segments: list[torch.Tensor | None] = [None] * rows
        starts = [0] * rows
        offset = 0
        tokens = torch.stack(flats, dim=1).cpu()  # [real tokens, layers, top_k], one device-to-host copy
        for row, (lo, hi) in zip(targets, spans, strict=True):
            segments[row] = tokens[offset : offset + hi - lo]
            starts[row] = lo
            offset += hi - lo
        result.append(RolloutRouting(segments, starts, seq_len, len(flats), flats[0].size(-1)))

    @contextlib.contextmanager
    def paused(self):
        """Neither capture nor replay for the block's forwards, which route naturally and leave the cursor where it
        was: a rank with no row in a bin still forwards an inert one, to keep every rank's dispatch count equal."""
        saved = [(layer._capture_routing, layer._forced_topk_indices) for layer in self._layers]
        for layer in self._layers:
            layer._capture_routing, layer._forced_topk_indices = False, None
        try:
            yield
        finally:
            for layer, (capturing, forced) in zip(self._layers, saved, strict=True):
                layer._capture_routing, layer._forced_topk_indices = capturing, forced

    def arm(self, masks: torch.Tensor, row_spans: list[tuple[int, int]] | None = None) -> None:
        """One microbatch's ``[rows, seq, layers, top_k]`` mask, consumed cursor-wise across row-chunked forwards (a GC
        forward saves its slice for the recompute) and armed through backward. ``row_spans`` trims each forwarded
        row."""
        if masks.dim() != 4 or masks.size(2) != len(self._layers):
            raise RuntimeError(
                f"routing mask shape {tuple(masks.shape)} does not match "
                f"[rows, seq, {len(self._layers)} EP layers, top_k]"
            )
        if row_spans is not None and len(row_spans) != masks.size(0):
            raise RuntimeError(f"row_spans has {len(row_spans)} entries for {masks.size(0)} forwarded rows")
        for i, layer in enumerate(self._layers):
            layer_masks = masks[:, :, i, :]
            if row_spans is None:
                flat = layer_masks.reshape(-1, masks.size(3))
            else:
                flat = torch.cat([layer_masks[r, lo:hi] for r, (lo, hi) in enumerate(row_spans)], dim=0)
            layer._forced_topk_indices = flat.long()
            layer._forced_cursor = 0
            layer._forced_consumed_total = 0
        self._armed = True

    def arm_rows(
        self,
        routing: RolloutRouting,
        rows: Sequence[int],
        attended: torch.Tensor,
        row_spans: list[tuple[int, int]] | None = None,
    ) -> None:
        """:meth:`arm` over the named rows of a round's routing, in the order they forward. Only that padded
        slice is built, on ``attended``'s device and outside the attended positions ``-1``; it is freed on return."""
        if attended.shape != (len(rows), routing.width):
            raise RuntimeError(
                f"attention mask {tuple(attended.shape)} does not match {len(rows)} rows of routing width {routing.width}"
            )
        masks = routing.pad_rows(rows, attended.device)
        masks.masked_fill_(~attended.bool()[:, :, None, None], -1)
        self.arm(masks, row_spans)

    @contextlib.contextmanager
    def replaying(
        self,
        routing: RolloutRouting,
        rows: Sequence[int],
        attended: torch.Tensor,
        row_spans: list[tuple[int, int]] | None = None,
    ):
        """:meth:`arm_rows` for the block's forwards, which must consume exactly what it armed."""
        self.arm_rows(routing, rows, attended, row_spans)
        try:
            yield
        finally:
            self.disarm()

    def disarm(self) -> None:
        """Raises (unless an exception is propagating) when a layer's consumed total is not a positive multiple of its
        armed size: only non-recompute forwards consume, and the cursor wraps on exact exhaustion."""
        if not self._armed:
            return
        bad = []
        for layer in self._layers:
            armed = layer._forced_topk_indices
            armed_size = 0 if armed is None else armed.size(0)
            consumed = layer._forced_consumed_total
            if armed_size > 0 and (consumed == 0 or consumed % armed_size != 0):
                bad.append(f"{type(layer).__name__} consumed={consumed} of armed={armed_size}")
        for layer in self._layers:
            layer._forced_topk_indices = None
            layer._forced_cursor = 0
            layer._forced_consumed_total = 0
        self._armed = False
        if bad and sys.exc_info()[0] is None:
            raise RuntimeError(
                f"routing-replay mask mis-consumed at disarm ({'; '.join(bad)}): the microbatch's "
                f"chunked forwards must tile the armed mask exactly (consumed-total a positive "
                f"multiple of the armed size)."
            )

    def drain_flip_rate(self) -> float | None:
        """COLLECTIVE. The share of forced (token, layer) selections whose live top-k differed since the last call,
        ``None`` when none was forced; one host sync, once per log step."""
        totals = [layer._replay_flip_counts for layer in self._layers if layer._replay_flip_counts is not None]
        for layer in self._layers:
            layer._replay_flip_counts = None
        local = torch.stack(totals).sum(dim=0) if totals else torch.zeros(2)
        flipped, forced = world_sums(local.unbind())
        return flipped / forced if forced > 0 else None


def build_routing_replay_injector(model: nn.Module) -> RoutingReplayInjector:
    layers = dict(named_ep_layers(model))
    return RoutingReplayInjector(
        list(layers.values()),
        engine_layers=int(text_config(unwrap_framework_wrappers(model).config).num_hidden_layers),
        layer_indices=decoder_layer_indices(layers.keys()),
    )


class RoutingReplayMixin:
    _rollout_routing: RolloutRouting | None = None

    def _build_routing_replay_injector(self, mode: str) -> RoutingReplayInjector | None:
        """``None`` when off; refuses an unknown mode, a model without EP layers or whose family cannot re-derive gate
        weights, and an unsupported setup."""
        modes = get_args(AsyncRolloutConfig.__annotations__["routing_replay"])
        if mode not in modes:
            raise ValueError(f"routing_replay must be one of {list(modes)}, got {mode!r}")
        self._rollout_routing_replay = mode == "rollout"
        if mode == "none":
            return None
        injector = build_routing_replay_injector(self.model)
        # The dense per-row path trims each row to its real span, and capture/arm must tile the same layout.
        self._replay_row_spans = self._trains_rows_densely
        reject_unsupported_routing_replay(
            mode,
            train_on_sampled_tokens=self.async_config.train_on_sampled_tokens,
            runs_logprob_recompute=self._runs_logprob_recompute,
        )
        if is_global_main_process():
            logger.info(f"Routing replay enabled (mode={mode}) over {injector.num_ep_layers} EP MoE layers")
        return injector

    @property
    def _runs_logprob_recompute(self) -> bool:
        return self.num_iterations > 1 or self._misaligned_accumulation or self._is_correction or self.beta != 0.0

    def _assemble_rollout_routing(self, rows: list[TurnRow], step: StepRows, mode: str) -> RolloutRouting | None:
        """``None`` off rollout replay. A batch with no engine routing fails only while its loss mask has a trainable
        token: an all-masked step has nothing to replay and trains as a no-op. A train round's routing stays on the
        host, ragged, for its forwards to pad."""
        rollout_routing = None
        if mode == "train":
            self._rollout_routing = None
        if self._rollout_routing_replay and mode == "train":
            if any(row.turn_routing is not None for row in rows):
                rollout_routing = self._rollout_routing_with_coverage(rows, step)
                self._rollout_routing = rollout_routing
            elif step.tool_mask.any():
                self._batch_errors.record(
                    "routing_replay='rollout' but no rollout in this batch returned routed_experts "
                    "— serve vLLM with --enable-return-routed-experts and --moe-backend triton, or "
                    "SGLang with --enable-return-routed-experts and --moe-runner-backend triton (the "
                    "triton_kernel/flashinfer runners bypass the capture hook)."
                )
            elif is_global_main_process():
                logger.warning(
                    "routing_replay='rollout': nothing to replay this step — every training row is "
                    "masked (no assistant turn yielded a row), so the step contributes "
                    "no gradient."
                )
            self._batch_errors.reject()
        return rollout_routing

    def _rollout_routing_with_coverage(self, rows: list[TurnRow], step: StepRows) -> RolloutRouting | None:
        """Each row's coverage is recorded; a malformed or wholly unresolved batch is recorded on the fence."""
        try:
            routing, coverage = build_rollout_routing(
                [row.turn_routing for row in rows],
                [len(row.prompt_ids) for row in rows],
                [len(row.completion_ids) for row in rows],
                step.prompt_ids.size(1),
                step.completion_ids.size(1),
                self._routing_injector.num_ep_layers,
                self._routing_injector.top_k,
                self._routing_injector.num_experts,
            )
        except ValueError as e:
            self._batch_errors.record(f"routing_replay='rollout': {e}")
            return None
        total = sum(coverage[k] for k in ROLLOUT_COVERAGE_SHAPES)
        for key, count in coverage.items():
            self._world_metrics.fraction(f"routing/rollout_{key}_frac", count, total)
        if coverage["unresolved"] == total:
            self._batch_errors.record(
                f"routing_replay='rollout': no routed row on this rank ({total} in the "
                f"batch) matched a known engine coverage convention, so every position "
                f"would replay natural routing and R3 would be inert. The engine's "
                f"routed_experts token count no longer lines up with the trainer's "
                f"prompt + completion lengths — check the engine version against the "
                f"conventions in build_rollout_routing."
            )
        return routing

    def _replaying_forwards(self, replay: RowsReplay | None, group: slice, attended: torch.Tensor):
        """Arms the rows of ``group`` among a no-grad call's forwarded rows for the block, none when ``replay`` is
        ``None``. ``attended`` holds the group's attention masks."""
        if replay is None:
            return contextlib.nullcontext()
        spans = None if replay.spans is None else replay.spans[group]
        return self._routing_injector.replaying(replay.routing, replay.rows[group], attended, spans)

    def _arm_loss_forward_replay(self, inputs: dict) -> None:
        """Over the positions the gradient forward attends; the rest route naturally. That excludes padding and the
        completion of a row dropped from the loss, which this forward reads as attention padding, in a context other
        than its mask's. A micro-batch pads its own rows of the round's routing from the host."""
        round_rows = inputs.get(ROUTING_ROWS_KEY)
        if round_rows is None:
            return
        attended = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        stashed = inputs.get(ROW_SPANS_KEY)
        if not self._replay_row_spans:
            spans = None
        elif stashed is not None:
            spans = host_row_spans(stashed)[0]
        else:
            spans = dense_row_spans(attended)[0]
        if self._rollout_routing is None:
            raise RuntimeError("the batch names routing rows but the trainer holds none for this round")
        self._routing_injector.arm_rows(self._rollout_routing, round_rows.tolist(), attended, spans)
