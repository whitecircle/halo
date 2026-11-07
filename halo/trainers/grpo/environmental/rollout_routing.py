"""A train round's routing (the engine's under R3, the recompute pass's under R2), kept ragged on the host: each row
holds only the positions covered, and the padded ``[rows, width, layers, top_k]`` slice a forward needs is built on
demand."""

from collections.abc import Sequence
from typing import NamedTuple

import torch

# The coverage conventions a routed row falls into, exactly one per row (``prompt_len_mismatch`` is
# counted on top of them, so it stays out of any per-row denominator).
ROLLOUT_COVERAGE_SHAPES = ("full", "engine_omits_last", "completion_only", "unresolved")


class RolloutRouting:
    """Per row, the covered run of an int16 ``[tokens, layers, top_k]`` mask, laid at ``starts[row]`` of a row
    ``width`` positions wide. A row without routing is ``None``."""

    def __init__(
        self,
        segments: list[torch.Tensor | None],
        starts: list[int],
        width: int,
        num_layers: int,
        top_k: int,
    ):
        self._segments = segments
        self._starts = starts
        self.width = width
        self.num_layers = num_layers
        self.top_k = top_k

    def __len__(self) -> int:
        return len(self._segments)

    def pad_rows(self, rows: Sequence[int], device: torch.device) -> torch.Tensor:
        """Int16 ``[len(rows), width, layers, top_k]`` on ``device`` for the named rows, in that order; ``-1`` marks
        a position without routing and a negative row id is a row of them."""
        out = torch.full((len(rows), self.width, self.num_layers, self.top_k), -1, dtype=torch.int16, device=device)
        for i, row in enumerate(rows):
            if row >= len(self):
                raise IndexError(f"routing row {row} out of range for a round of {len(self)} rows")
            segment = None if row < 0 else self._segments[row]
            if segment is None:
                continue
            start = self._starts[row]
            if start + segment.size(0) > self.width:
                raise ValueError(f"routing row {row} runs {start}+{segment.size(0)} past a width of {self.width}")
            out[i, start : start + segment.size(0)] = segment.to(device)
        return out


class RowsReplay(NamedTuple):
    """The rollout routing a scoring call replays: ``rows[p]`` is the round row the ``p``-th forwarded row is, and
    ``spans`` its attended span where rows forward trimmed."""

    routing: RolloutRouting
    rows: list[int]
    spans: list[tuple[int, int]] | None


def build_rollout_routing(
    turn_masks: list[tuple[torch.Tensor, int | None] | None],
    prompt_lens: list[int],
    completion_lens: list[int],
    max_prompt_len: int,
    max_completion_len: int,
    num_layers: int,
    top_k: int,
    num_experts: int,
) -> tuple[RolloutRouting, dict[str, int]]:
    """``(routing, stats)``. Alignment anchors on the engine's prompt token count, since a re-render can drift by a
    token. Positions without routing stay ``-1``; a row matching no convention is counted, so the caller can refuse
    uniformly."""
    full, engine_omits_last, completion_only, unresolved = ROLLOUT_COVERAGE_SHAPES
    stats = {**dict.fromkeys(ROLLOUT_COVERAGE_SHAPES, 0), "prompt_len_mismatch": 0}
    segments: list[torch.Tensor | None] = []
    starts: list[int] = []
    for i, turn in enumerate(turn_masks):
        segments.append(None)
        starts.append(max_prompt_len)
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
        picks = mask.sort(dim=-1).values
        if (picks[..., 1:] == picks[..., :-1]).any():
            # A router never picks an expert twice for a token; DeepEP dispatch faults on a repeated id.
            raise ValueError(
                f"routed_experts mask row {i} names one expert twice in a token's top-{top_k} — the engine's "
                f"routing is not a top-k selection."
            )
        p, c = prompt_lens[i], completion_lens[i]
        ep = engine_p if engine_p is not None else p
        tokens = mask.shape[0]
        if tokens in (ep + c, ep + c - 1):
            stats[full if tokens == ep + c else engine_omits_last] += 1
            covered = min(tokens - ep, c)
            if ep == p:
                segments[i], starts[i] = mask[: p + covered], max_prompt_len - p
            else:
                segments[i] = mask[ep : ep + covered]
                stats["prompt_len_mismatch"] += 1  # completion still replayed; prompt stays natural
        elif tokens in (c, c - 1):
            stats[completion_only] += 1
            segments[i] = mask
        else:
            stats[unresolved] += 1
    routing = RolloutRouting(segments, starts, max_prompt_len + max_completion_len, num_layers, top_k)
    return routing, stats
