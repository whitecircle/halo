"""Trainer-side MoE routing replay: capture top-k expert selections during the no-grad logprob-recompute
pass and replay them in the gradient forward.

Small numeric differences flip top-k selection near decision boundaries, so the update pass would
otherwise train through a different token distribution than the importance ratios were computed on
(Qwen Routing Replay, arXiv:2507.18071). Gate weights are always re-derived from the live router scores
restricted to the replayed selection; replaying weights or logits zeroes the router gradient (RSPO,
arXiv:2510.23027).

:class:`RoutingReplayInjector` holds the per-layer capture/arm state on
:class:`~src.distributed.expert_parallel.base_layer.EPMoELayerBase` as plain attributes (following
``balancing_biases``); its methods document the capture → arm → disarm → ``flip_rate`` cycle.
"""

import base64
import contextlib
import io
import sys

import numpy as np
import torch
from torch import nn

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.trainers.mixins.ep_introspection import named_ep_layers

ROUTING_MASKS_KEY = "routing_masks"


_NPY_MAGIC = b"\x93NUMPY"


def decode_rollout_routing(payload: str, num_layers: int, top_k: int) -> torch.Tensor:
    """Decode an engine's ``routed_experts`` payload to an int16 ``[tokens, L, K]`` tensor.

    Two wire formats exist: vLLM ships a base64 ``.npy`` (shape in the header), SGLang ships base64
    raw little-endian int32 rows with the shape implied. The npy magic disambiguates; the raw branch
    reshapes by the trainer's own ``(num_layers, top_k)`` and rejects a byte count they do not divide,
    so a shape disagreement surfaces here rather than as a misaligned mask.
    """
    raw = base64.b64decode(payload)
    if raw[: len(_NPY_MAGIC)] == _NPY_MAGIC:
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
    """Assemble per-turn rollout routing into the padded ``[rows, P+C, L, K]`` batch mask.

    Each turn entry is ``(mask, engine_prompt_tokens)``. Alignment is anchored on the engine's prompt
    token count (``usage.prompt_tokens``): the trainer's re-render can drift by a token, which would
    shift every completion position by one. The trainer-side prompt span is only filled when the two
    counts agree.

    Positions without engine routing stay ``-1`` (natural selection): padding, whole rows without a
    mask, the prompt span on a count mismatch, and rows whose coverage matches no known engine
    convention. Those last are counted rather than raised, since a per-rank raise would desync ranks
    into a collective hang; the caller decides uniformly. Returns ``(masks, stats)`` with
    per-convention row counts, the observable for engine coverage drift.
    """
    rows = len(turn_masks)
    stats = {"full": 0, "engine_omits_last": 0, "completion_only": 0, "prompt_len_mismatch": 0, "unresolved": 0}
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
            stats["full" if tokens == ep + c else "engine_omits_last"] += 1
            covered = min(tokens - ep, c)
            out[i, max_prompt_len : max_prompt_len + covered] = mask[ep : ep + covered]
            if ep == p:
                out[i, max_prompt_len - p : max_prompt_len] = mask[:p]
            else:
                stats["prompt_len_mismatch"] += 1  # completion still replayed; prompt stays natural
        elif tokens in (c, c - 1):
            stats["completion_only"] += 1
            out[i, max_prompt_len : max_prompt_len + tokens] = mask
        else:
            stats["unresolved"] += 1
    return out, stats


class RoutingReplayInjector:
    """Holds capture/replay state across the EP MoE layers of one model.

    The layer list is in ``named_modules`` order (identical between capture and replay); the mask's
    layer axis indexes this list.
    """

    def __init__(self, ep_layers: list[EPMoELayerBase]):
        if not ep_layers:
            raise ValueError("RoutingReplayInjector requires at least one EP MoE layer")
        unsupported = sorted({type(m).__name__ for m in ep_layers if not m._supports_routing_replay})
        if unsupported:
            raise NotImplementedError(
                f"routing replay is not supported for {unsupported}: these families cannot re-derive "
                f"gate weights at a forced selection (see _supports_routing_replay on each class)."
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
    def num_layers(self) -> int:
        return len(self._layers)

    @property
    def top_k(self) -> int:
        return int(self._layers[0].top_k)

    @property
    def num_experts(self) -> int:
        return int(self._layers[0].num_experts)

    @contextlib.contextmanager
    def capture(self, rows: int, seq_len: int, row_spans: list[tuple[int, int]] | None = None):
        """Capture routing over a (possibly chunked) no-grad forward covering ``rows`` sequences of
        padded width ``seq_len``; yields a list that holds the assembled ``[rows, seq, layers, top_k]``
        mask on exit.

        ``row_spans`` describes span-trimmed per-row forwards (the FA4 dense path): row ``i`` forwards
        only ``[lo, hi)`` of its padded width, so the capture expects ``Σ(hi-lo)`` tokens per layer and
        scatters each row's segment back to its span; positions outside a span stay ``-1``. ``None``
        means full-width forwards.

        An error inside resets every layer's capture state and propagates unchanged, so the assembly
        check cannot shadow an OOM or NCCL failure.
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
        """Arm every EP layer with one microbatch's mask ``[rows, seq, layers, top_k]``.

        The loss forward is row-chunked, so each layer consumes the mask cursor-wise across the chunk
        forwards; under GC each forward saves its slice in its checkpoint scope for that frame's
        backward recompute (see ``EPMoELayerBase._maybe_replace_selection``). Stays armed through backward; re-arming resets
        consumption. ``row_spans`` (the FA4 dense path) trims each row to the ``[lo, hi)`` its per-row
        forward actually covers, so the cursor tiles the span-trimmed token stream.
        """
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
        """Clear the forced masks after the microbatch's backward.

        Consumption model: only non-recompute forwards advance the cursor and consumed-total; the
        cursor wraps to zero on exact exhaustion so whole-pass reuse re-consumes cleanly, and GC
        backward recomputes re-read their frame's saved slice without consuming. A correctly-tiled
        microbatch therefore ends with consumed-total a positive multiple of the armed size; cursor==0
        alone would also match a never-consumed mask. Raises on any other end state, unless an
        exception is already propagating."""
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
        """Mean fraction of forced (token, layer) selections whose live top-k differed from the replayed
        mask since the last call. One host sync per call — call once per logging step."""
        totals = [layer._replay_flip_counts for layer in self._layers if layer._replay_flip_counts is not None]
        for layer in self._layers:
            layer._replay_flip_counts = None
        if not totals:
            return None
        flipped, forced = torch.stack(totals).sum(dim=0).tolist()
        return flipped / forced if forced > 0 else None


def build_routing_replay_injector(model: nn.Module) -> RoutingReplayInjector:
    """Build an injector over ``model``'s EP MoE layers (named_modules order)."""
    return RoutingReplayInjector(list(named_ep_layers(model).values()))
