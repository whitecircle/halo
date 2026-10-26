"""MoE routing replay (arXiv:2507.18071): the recompute's or the engine's top-k selections replayed in the update,
so near-tie flips cannot train it through other experts than the importance ratios saw. Gate weights are
re-derived from the live router over the replayed selection: replaying weights zeroes the router gradient (RSPO,
arXiv:2510.23027)."""

import base64
import contextlib
import io
import logging
import sys
from collections.abc import Iterable
from typing import get_args

import numpy as np
import torch
from numpy.lib.format import MAGIC_PREFIX as NPY_MAGIC_PREFIX
from torch import nn

from halo.configs.grpo_environmental import AsyncTrainingConfig
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase, named_ep_layers
from halo.models.config_access import text_config
from halo.models.module_tree import decoder_layer_index
from halo.models.unwrapping import unwrap_framework_wrappers
from halo.runtime.ranks import is_global_main_process
from halo.trainers.grpo.environmental.environmental_rules import reject_unsupported_routing_replay
from halo.trainers.grpo.environmental.step_batch import StepRows
from halo.trainers.grpo.environmental.trajectory_tokenize import TurnRow
from halo.trainers.grpo.logprob_scoring import dense_row_spans, rows_forward_densely
from halo.trainers.world_metrics import world_sums

logger = logging.getLogger(__name__)

# The coverage conventions a routed row falls into, exactly one per row (``prompt_len_mismatch`` is
# counted on top of them, so it stays out of any per-row denominator).
ROLLOUT_COVERAGE_SHAPES = ("full", "engine_omits_last", "completion_only", "unresolved")


def decoder_layer_indices(names: Iterable[str]) -> list[int]:
    """The decoder-layer index of each EP layer FQN, which both engines key their per-token rows by: a family
    with dense first layers ships more rows than it has EP layers. An FQN outside ``layers.N`` raises."""
    indices = []
    for name in names:
        index = decoder_layer_index(name)
        if index is None:
            raise ValueError(f"EP layer {name!r} carries no decoder-layer index to match an engine row")
        indices.append(index)
    return indices


def decode_rollout_routing(payload: str, num_layers: int, top_k: int) -> torch.Tensor:
    """An engine's ``routed_experts`` payload as an int16 ``[tokens, L, K]`` tensor: vLLM's base64 ``.npy`` or
    SGLang's base64 raw int32 rows, reshaped by the trainer's own ``(num_layers, top_k)`` so a shape
    disagreement raises here rather than misaligning the mask."""
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


def assemble_rollout_masks(
    turn_masks: list[tuple[torch.Tensor, int | None] | None],
    prompt_lens: list[int],
    completion_lens: list[int],
    max_prompt_len: int,
    max_completion_len: int,
    num_layers: int,
    top_k: int,
    num_experts: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    """``(masks, stats)``: per-turn ``(mask, engine_prompt_tokens)`` routing in the padded ``[rows, P+C, L, K]``
    batch mask, and the row count of each coverage convention (:data:`ROLLOUT_COVERAGE_SHAPES` plus
    ``prompt_len_mismatch``). Alignment anchors on the engine's prompt token count, since a re-render can drift
    by a token. Positions without engine routing stay ``-1`` (natural selection); a row matching no convention
    is counted, not raised, so the caller can refuse uniformly."""
    full, engine_omits_last, completion_only, unresolved = ROLLOUT_COVERAGE_SHAPES
    rows = len(turn_masks)
    stats = {**dict.fromkeys(ROLLOUT_COVERAGE_SHAPES, 0), "prompt_len_mismatch": 0}
    out = torch.full((rows, max_prompt_len + max_completion_len, num_layers, top_k), -1, dtype=torch.int16)
    for i, turn in enumerate(turn_masks):
        if turn is None:
            continue
        mask, engine_p = turn
        if mask.shape[1] != num_layers or mask.shape[2] != top_k:
            raise ValueError(
                f"routed_experts mask row {i} has [layers, top_k] = {tuple(mask.shape[1:])}, the model "
                f"has ({num_layers}, {top_k}) — engine and trainer disagree on the MoE shape."
            )
        if mask.min() < 0 or mask.max() >= num_experts:
            # Out-of-range ids corrupt DeepEP dispatch device-side rather than raising cleanly.
            raise ValueError(
                f"routed_experts mask row {i} contains expert ids outside [0, {num_experts}) — the "
                f"engine is serving a different expert layout than the trainer."
            )
        p, c = prompt_lens[i], completion_lens[i]
        ep = engine_p if engine_p is not None else p
        tokens = mask.shape[0]
        if tokens in (ep + c, ep + c - 1):
            stats[full if tokens == ep + c else engine_omits_last] += 1
            covered = min(tokens - ep, c)
            out[i, max_prompt_len : max_prompt_len + covered] = mask[ep : ep + covered]
            if ep == p:
                out[i, max_prompt_len - p : max_prompt_len] = mask[:p]
            else:
                stats["prompt_len_mismatch"] += 1  # completion still replayed; prompt stays natural
        elif tokens in (c, c - 1):
            stats[completion_only] += 1
            out[i, max_prompt_len : max_prompt_len + tokens] = mask
        else:
            stats[unresolved] += 1
    return out, stats


class RoutingReplayInjector:
    """Capture/replay state across one model's EP MoE layers, in ``named_modules`` order (the mask's layer
    axis); ``engine_layers`` and ``layer_indices`` place those layers among the engine's per-decoder-layer rows."""

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
    def capture(self, rows: int, seq_len: int, row_spans: list[tuple[int, int]] | None = None):
        """Capture routing over a (possibly chunked) no-grad forward of ``rows`` sequences of padded width
        ``seq_len``; the yielded list holds the ``[rows, seq, layers, top_k]`` mask on exit. ``row_spans``
        describes span-trimmed per-row forwards (positions outside a span stay ``-1``). An error inside resets
        every layer and propagates unchanged, so the token-count check cannot shadow an OOM or NCCL failure.
        """
        for layer in self._layers:
            layer._capture_routing = True
            layer._captured_routing_chunks = []
        result: list[torch.Tensor] = []
        try:
            yield result
            expected = sum(hi - lo for lo, hi in row_spans) if row_spans is not None else rows * seq_len
            per_layer = []
            errors = []
            layer_chunks = []
            for layer in self._layers:
                layer_chunks.append(layer._captured_routing_chunks)
        finally:
            # Reset every layer on any exception path, else stale capture state leaks into the next forward.
            for layer in self._layers:
                layer._capture_routing = False
                layer._captured_routing_chunks = []
        for chunks in layer_chunks:
            flat = torch.cat(chunks, dim=0) if chunks else None
            if flat is None or flat.size(0) != expected:
                got = 0 if flat is None else flat.size(0)
                errors.append(f"layer {len(per_layer) + len(errors)}: {got} tokens")
            elif row_spans is None:
                per_layer.append(flat.view(rows, seq_len, -1))
            else:
                full = flat.new_full((rows, seq_len, flat.size(-1)), -1)
                offset = 0
                for r, (lo, hi) in enumerate(row_spans):
                    full[r, lo:hi] = flat[offset : offset + (hi - lo)]
                    offset += hi - lo
                per_layer.append(full)
        if errors:
            raise RuntimeError(
                f"routing-replay capture recorded a wrong token count (expected {expected} = "
                f"{'Σ row spans' if row_spans is not None else f'rows*seq_len = {rows}*{seq_len}'}): "
                f"{'; '.join(errors)}. The capture forwards must tile exactly the declared row layout "
                f"(pass row_spans for span-trimmed per-row forwards)."
            )
        result.append(torch.stack(per_layer, dim=2))  # [rows, seq, layers, top_k] int16

    def arm(self, masks: torch.Tensor, row_spans: list[tuple[int, int]] | None = None) -> None:
        """Arm every EP layer with one microbatch's ``[rows, seq, layers, top_k]`` mask, consumed cursor-wise
        across the row-chunked forwards (a GC forward saves its slice for the backward recompute); it stays armed
        through backward. ``row_spans`` trims each row to what its per-row forward covers."""
        if masks.dim() != 4 or masks.size(2) != len(self._layers):
            raise RuntimeError(
                f"routing mask shape {tuple(masks.shape)} does not match "
                f"[rows, seq, {len(self._layers)} EP layers, top_k]"
            )
        if row_spans is not None and len(row_spans) != masks.size(0):
            raise RuntimeError(f"row_spans has {len(row_spans)} entries for {masks.size(0)} mask rows")
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

    def disarm(self) -> None:
        """Clear the forced masks after the microbatch's backward, raising (unless an exception is propagating)
        when a layer's consumed total is not a positive multiple of its armed size: only non-recompute forwards
        consume, and the cursor wraps on exact exhaustion, so a correctly tiled microbatch ends there."""
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

    def flip_rate(self) -> float | None:
        """COLLECTIVE. The share of the world's forced (token, layer) selections whose live top-k differed from
        the replayed mask since the last call, ``None`` when none was forced; one host sync, once per log step."""
        totals = [layer._replay_flip_counts for layer in self._layers if layer._replay_flip_counts is not None]
        for layer in self._layers:
            layer._replay_flip_counts = None
        local = torch.stack(totals).sum(dim=0) if totals else torch.zeros(2)
        flipped, forced = world_sums(local.unbind())
        return flipped / forced if forced > 0 else None


def build_routing_replay_injector(model: nn.Module) -> RoutingReplayInjector:
    """An injector over ``model``'s EP MoE layers, placed on the engine's one-row-per-decoder-layer wire."""
    layers = dict(named_ep_layers(model))
    return RoutingReplayInjector(
        list(layers.values()),
        engine_layers=int(text_config(unwrap_framework_wrappers(model).config).num_hidden_layers),
        layer_indices=decoder_layer_indices(layers.keys()),
    )


class RoutingReplayMixin:
    """Builds and feeds the routing-replay injector of the environmental GRPO trainer."""

    def _build_routing_replay_injector(self, mode: str) -> RoutingReplayInjector | None:
        """The routing-replay injector for ``mode`` (``None`` when off), refusing an unknown mode, a model
        without EP MoE wrappers or whose family cannot re-derive gate weights, and an unsupported mode setup."""
        modes = get_args(AsyncTrainingConfig.__annotations__["routing_replay"])
        if mode not in modes:
            raise ValueError(f"routing_replay must be one of {list(modes)}, got {mode!r}")
        self._rollout_routing_replay = mode == "rollout"
        if mode == "none":
            return None
        injector = build_routing_replay_injector(self.model)
        # The dense per-row path trims each row to its real span, and capture/arm must tile the same layout.
        self._replay_row_spans = self._use_chunked_grpo_logprobs and rows_forward_densely(
            self.accelerator.unwrap_model(self.model), self.args.per_device_train_batch_size
        )
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
        """Whether the no-grad log-prob recompute runs: the one gate the recompute pass and the ``recompute``
        replay refusal both read."""
        return self.num_iterations > 1 or self._misaligned_accumulation or self._is_correction or self.beta != 0.0

    def _assemble_rollout_routing(
        self, rows: list[TurnRow], step: StepRows, device: torch.device, mode: str
    ) -> torch.Tensor | None:
        """The engine-supplied routing mask under ``routing_replay='rollout'``, else ``None``; the uniform raise
        it wraps sits behind a config and mode gate. A batch with no engine routing fails only while its loss
        mask still has a trainable token: an all-masked step has no selection to replay and trains as a no-op.
        """
        rollout_routing_masks = None
        if self._rollout_routing_replay and mode == "train":
            if any(row.turn_routing is not None for row in rows):
                rollout_routing_masks = self._rollout_masks_with_coverage(rows, step, device)
            elif step.tool_mask.any():
                self._batch_errors.record(
                    "routing_replay='rollout' but no rollout in this batch returned routed_experts "
                    "— serve vLLM >= 0.22 with --enable-return-routed-experts and a non-FlashInfer "
                    "MoE backend (--moe-backend triton), or SGLang with --enable-return-routed-experts "
                    "and --moe-runner-backend triton (the triton_kernel/flashinfer runners bypass "
                    "the capture hook)."
                )
            elif is_global_main_process():
                logger.warning(
                    "routing_replay='rollout': nothing to replay this step — every training row is "
                    "masked (no assistant turn yielded a row), so the step contributes "
                    "no gradient."
                )
            self._batch_errors.reject()
        return rollout_routing_masks

    def _rollout_masks_with_coverage(
        self, rows: list[TurnRow], step: StepRows, device: torch.device
    ) -> torch.Tensor | None:
        """The step's engine routing as the batch mask, each row's coverage recorded; a malformed or wholly
        unresolved batch is recorded on the fence."""
        try:
            masks, coverage = assemble_rollout_masks(
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
                f"conventions in assemble_rollout_masks."
            )
        return masks.to(device)

    def _replay_spans_for(self, inputs: dict) -> list[tuple[int, int]] | None:
        """Row spans for capture/arm when the dense per-row path trims the forwards; ``None`` on the padded
        paths, where the mask tiles ``rows × seq`` directly."""
        if not self._replay_row_spans:
            return None
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        return dense_row_spans(attention_mask)
