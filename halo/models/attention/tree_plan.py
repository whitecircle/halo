"""The tree-attention plan of a tree-packed row: its nodes recovered from the two per-token layout columns (path
position, node depth) and the index tensors that give every token exactly its own root-to-leaf path in each
sequence mixer. Nodes are laid out depth-first, so a node's ancestors always precede it in the row."""

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, get_args

import numpy as np
import torch

# The batch key the collator emits the plan under and the model's sequence mixers read it from.
TREE_ATTENTION_KEY = "tree_attention"

# The two forms full-attention layers read a tree in (:mod:`halo.models.attention.tree`).
FullAttentionForm = Literal["split", "gather"]
FULL_ATTENTION_FORMS = get_args(FullAttentionForm)

# Gathered keys one attention call takes. The gather form repeats a prefix once per node below it, so a wide
# tree gathers fan-out x prefix keys, and a GQA backward expands its key and value gradients to every query head.
GATHER_KEYS_PER_CALL = 1 << 19


def resolve_node_depths(parents: Iterable[int]) -> list[int]:
    """Depth of every node of a depth-first layout, from each node's parent index (-1 for a root)."""
    depths: list[int] = []
    for parent in parents:
        depths.append(0 if parent == -1 else depths[parent] + 1)
    return depths


@dataclass(frozen=True)
class NodeTable:
    """Nodes of a flattened batch, in row order: first token, token count and parent of each."""

    starts: np.ndarray
    lengths: np.ndarray
    parents: np.ndarray

    @property
    def has_branches(self) -> bool:
        return bool((self.parents >= 0).any())


def build_node_table(position_ids: torch.Tensor, depth: torch.Tensor) -> NodeTable:
    """Recover the nodes of a row from its two per-token columns, refusing anything but a depth-first layout.

    A node starts where the depth changes or the position does not continue. A sibling restarts at
    its parent's end, at or before the previous sibling's last position, so the second test never
    misses one. Depth-first order makes a node's parent the nearest earlier node one level up, and a
    child must continue that parent's last position: a row that was truncated, split or reordered
    below tree granularity fails this check instead of attending across trees.
    """
    position_ids, depth = position_ids.flatten().numpy(), depth.flatten().numpy()
    boundary = np.ones(len(position_ids), dtype=bool)
    boundary[1:] = (depth[1:] != depth[:-1]) | (position_ids[1:] != position_ids[:-1] + 1)
    starts = np.flatnonzero(boundary)
    lengths = np.diff(np.append(starts, len(position_ids)))
    node_depth = depth[starts]
    parents = np.full(len(starts), -1, dtype=np.int64)
    expected_start = np.zeros(len(starts), dtype=position_ids.dtype)
    intact = node_depth[0] == 0
    for level in range(1, int(node_depth.max(initial=0)) + 1):
        above, here = np.flatnonzero(node_depth == level - 1), np.flatnonzero(node_depth == level)
        rank = np.searchsorted(above, here)
        if not rank.all():  # a node with no earlier node one level up
            intact = False
            break
        parents[here] = above[rank - 1]
        expected_start[here] = position_ids[starts[parents[here]] + lengths[parents[here]] - 1] + 1
    if not intact or (position_ids[starts] != expected_start).any():
        raise ValueError(
            "the row is not a depth-first tree layout: a node does not continue its parent's last "
            "position, or a root does not start at position 0. Tree rows must reach the collator "
            "whole (packing_strategy 'bfd', no truncation or splitting below a tree)."
        )
    return NodeTable(starts, lengths, parents)


def _expand(runs: list[tuple[int, int]]) -> np.ndarray:
    """Token indices of ``(start, length)`` runs, concatenated."""
    if not runs:
        return np.empty(0, dtype=np.int64)
    starts, lengths = np.asarray(runs, dtype=np.int64).T
    return np.repeat(starts - (np.cumsum(lengths) - lengths), lengths) + np.arange(lengths.sum())


def _varlen(sequences: list[list[tuple[int, int]]], side: str = "k") -> dict[str, torch.Tensor | int]:
    """Gather index and cumulative lengths of varlen sequences given as lists of token runs, named
    for the side of the kernel call they feed (``q`` or ``k``)."""
    lengths = [sum(length for _, length in runs) for runs in sequences]
    return {
        f"{side}_index": torch.from_numpy(_expand([run for runs in sequences for run in runs])),
        f"cu_seqlens_{side}": torch.tensor([0, *np.cumsum(lengths)], dtype=torch.int32),
        f"max_seqlen_{side}": max(lengths),
    }


def _tail(runs: list[tuple[int, int]], budget: int) -> list[tuple[int, int]]:
    """The last ``budget`` tokens of consecutive runs, as runs."""
    kept: list[tuple[int, int]] = []
    for start, length in reversed(runs):
        if budget <= 0:
            break
        take = min(length, budget)
        kept.append((start + length - take, take))
        budget -= take
    return kept[::-1]


def _behind_tail(
    ancestors: list[list[tuple[int, int]]], runs: list[tuple[int, int]], budget: int | None
) -> dict[str, torch.Tensor | int]:
    """One gathered sequence per node: the last ``budget`` tokens above it on its path (all of them
    for ``None``), then the node."""
    return _varlen(
        [
            [*(above if budget is None else _tail(above, budget)), run]
            for above, run in zip(ancestors, runs, strict=True)
        ]
    )


def _with_call_chunks(part: dict, node_lengths: np.ndarray) -> dict:
    """``part`` plus, when its gathered keys exceed :data:`GATHER_KEYS_PER_CALL`, the runs of whole
    nodes that each fit (a node over the budget runs alone) under ``chunks``: per run, its query and
    key token spans and the varlen bounds and maxima of one call over it."""
    key_bounds = part["cu_seqlens_k"].numpy().astype(np.int64)
    if key_bounds[-1] <= GATHER_KEYS_PER_CALL:
        return part
    query_bounds = np.concatenate([[0], np.cumsum(node_lengths)])
    cuts = [0]
    for node in range(1, len(node_lengths)):
        if key_bounds[node + 1] - key_bounds[cuts[-1]] > GATHER_KEYS_PER_CALL:
            cuts.append(node)
    cuts.append(len(node_lengths))
    chunks = []
    for first, last in pairwise(cuts):
        q_bounds, k_bounds = query_bounds[first : last + 1], key_bounds[first : last + 1]
        chunks.append(
            {
                "q_span": (int(q_bounds[0]), int(q_bounds[-1])),
                "k_span": (int(k_bounds[0]), int(k_bounds[-1])),
                "cu_seqlens_q": torch.from_numpy((q_bounds - q_bounds[0]).astype(np.int32)),
                "cu_seqlens_k": torch.from_numpy((k_bounds - k_bounds[0]).astype(np.int32)),
                "max_seqlen_q": int(np.diff(q_bounds).max()),
                "max_seqlen_k": int(np.diff(k_bounds).max()),
            }
        )
    return {**part, "chunks": chunks}


def _levels(runs: list[tuple[int, int]], parents: list[int]) -> tuple[list[dict], torch.Tensor, torch.Tensor]:
    """The nodes one depth at a time: the row's tokens in that order, the permutation from that order back to
    row order, and per depth its nodes' varlen bounds (``cu_seqlens``, and as python ints ``cu_seqlens_host``,
    which a batch transport leaves on the host) and, below the roots, every node's ``parent`` slot in the
    level above.
    """
    depths = resolve_node_depths(parents)
    levels: list[dict] = []
    in_depth_order: list[tuple[int, int]] = []
    slots: dict[int, int] = {}
    for depth in range(max(depths) + 1):
        nodes = [node for node, node_depth in enumerate(depths) if node_depth == depth]
        in_depth_order += [runs[node] for node in nodes]
        bounds = (0, *np.cumsum([runs[node][1] for node in nodes]).tolist())
        level = {"cu_seqlens": torch.tensor(bounds, dtype=torch.int32), "cu_seqlens_host": bounds}
        if depth:
            level["parent"] = torch.tensor([slots[parents[node]] for node in nodes])
        levels.append(level)
        slots = {node: slot for slot, node in enumerate(nodes)}
    order = torch.from_numpy(_expand(in_depth_order))
    return levels, order, torch.argsort(order)


def _path_links(table: NodeTable, parents: list[int]) -> dict[str, torch.Tensor]:
    """Per token: its predecessor on its own path (``-1`` at a root), the row end of its node's subtree,
    its path position and the row start of its tree. The subtree of a depth-first layout is contiguous,
    so a key at ``e`` precedes a query at ``q`` on the query's path exactly when ``e <= q < subtree_end[e]``."""
    starts, lengths = table.starts.astype(np.int64), table.lengths.astype(np.int64)
    ends = starts + lengths
    subtree_end, path_start, tree_start = ends.copy(), np.zeros(len(starts), dtype=np.int64), starts.copy()
    for node in reversed(range(len(parents))):
        if parents[node] != -1:
            subtree_end[parents[node]] = max(subtree_end[parents[node]], subtree_end[node])
    for node, parent in enumerate(parents):
        if parent != -1:
            path_start[node] = path_start[parent] + lengths[parent]
            tree_start[node] = tree_start[parent]
    total = int(lengths.sum())
    predecessor = np.arange(total, dtype=np.int64) - 1
    predecessor[starts] = [-1 if parent == -1 else ends[parent] - 1 for parent in parents]
    within = np.arange(total, dtype=np.int64) - np.repeat(starts, lengths)
    return {
        "predecessor": torch.from_numpy(predecessor),
        "subtree_end": torch.from_numpy(np.repeat(subtree_end, lengths)),
        "position": torch.from_numpy(np.repeat(path_start, lengths) + within),
        "tree_start": torch.from_numpy(np.repeat(tree_start, lengths)),
    }


def build_tree_attention_plan(
    table: NodeTable,
    *,
    sliding_window: int | None,
    full_form: FullAttentionForm,
    conv_kernel: int | None = None,
    sparse_attention: bool = False,
) -> dict:
    """The index tensors a model's sequence mixers need to give every token its own path as context.

    Always present: ``cu_seqlens_q`` / ``max_seqlen_q`` (the nodes as varlen query sequences) and
    ``leaf_count``, per token the number of examples its node belongs to — how often the token would
    occur with every example laid out on its own, which per-token statistics weight by.

    Softmax attention, only where the row holds a branch (plain documents need nothing beyond their
    position resets):

    - ``sliding``: per node, the keys of one gathered sequence (``k_index``, ``cu_seqlens_k``,
      ``max_seqlen_k``) — the last ``sliding_window - 1`` tokens above it, then the node.
    - ``full``, form ``gather``: the same with every token above it.
    - Either gather part over more than :data:`GATHER_KEYS_PER_CALL` keys also carries ``chunks``,
      runs of whole nodes that each fit, which the layer attends one call at a time.
    - ``full``, form ``split``: per parent, its children's tokens as queries (``q_index``,
      ``cu_seqlens_q``, ``max_seqlen_q``) and the path down to it as keys (``k_index``, ...).

    A family with stateful linear-attention layers (``conv_kernel``, the width of their causal
    convolution), on every row:

    - ``conv``: each node behind the last ``conv_kernel - 1`` tokens above it (``index``), the
      gathered sequences kept apart by ``seq_idx``; ``own`` picks the nodes' outputs back out.
    - ``levels``: per tree depth, its nodes' varlen bounds (``cu_seqlens``, and ``cu_seqlens_host``
      as python ints for the kernel's host-side chunk indices) and each node's ``parent`` slot in the
      level above; ``depth_order`` lists the row's tokens depth by depth, and ``restore`` returns that
      order to row order.

    A family whose softmax layers select their keys per query (``sparse_attention``, GLM-5 Next's
    DeepSeek sparse attention), on every row and in place of the softmax parts above: ``sparse``, the
    per-token path links of :func:`_path_links`, from which those layers build their candidates on the
    device.

    Maxima are python ints: a tensor there enters FA4's backward compile key and recompiles it per call.
    """
    runs = list(zip(table.starts.tolist(), table.lengths.tolist(), strict=True))
    parents = table.parents.tolist()
    ancestors: list[list[tuple[int, int]]] = []
    for parent in parents:
        ancestors.append([] if parent == -1 else [*ancestors[parent], runs[parent]])

    leaves = np.zeros(len(runs), dtype=np.int64)  # examples below each node; children come after their parent
    for node in reversed(range(len(runs))):
        leaves[node] = max(leaves[node], 1)
        if parents[node] != -1:
            leaves[parents[node]] += leaves[node]
    plan: dict = {
        "cu_seqlens_q": torch.tensor([0, *np.cumsum(table.lengths)], dtype=torch.int32),
        "max_seqlen_q": int(table.lengths.max()),
        "leaf_count": torch.from_numpy(np.repeat(leaves, table.lengths)),
    }
    if sparse_attention:
        plan["sparse"] = _path_links(table, parents)
    elif table.has_branches:
        if sliding_window is not None:
            plan["sliding"] = _with_call_chunks(_behind_tail(ancestors, runs, sliding_window - 1), table.lengths)
        if full_form == "gather":
            plan["full"] = _with_call_chunks(_behind_tail(ancestors, runs, None), table.lengths)
        else:
            children: dict[int, list[tuple[int, int]]] = {}
            for node, parent in enumerate(parents):
                if parent != -1:
                    children.setdefault(parent, []).append(runs[node])
            plan["full"] = {
                **_varlen(list(children.values()), side="q"),
                **_varlen([[*ancestors[parent], runs[parent]] for parent in children]),
            }
    if conv_kernel is not None:
        behind = _behind_tail(ancestors, runs, conv_kernel - 1)
        bounds = behind["cu_seqlens_k"].long()
        plan["conv"] = {
            "index": behind["k_index"],
            "seq_idx": torch.arange(len(runs), dtype=torch.int32).repeat_interleave(bounds.diff())[None],
            "own": torch.from_numpy(
                _expand([(end - length, length) for end, (_, length) in zip(bounds[1:].tolist(), runs, strict=True)])
            ),
        }
        plan["levels"], plan["depth_order"], plan["restore"] = _levels(runs, parents)
    return plan
