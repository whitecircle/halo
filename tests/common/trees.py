"""Reference pieces the tree-packing tests share.

The dense keep-masks state what tree attention means — a token attends its own root-to-leaf path and
nothing else — independently of any kernel plan, so every plan and every forward is tested against
them. :func:`reference_varlen` is the kernel contract tree attention relies on, in plain torch: it
stands in for the CUDA kernels on the CPU tier and is pinned to them on the GPU tier. The other
helpers turn token paths into a tree row and a recovered node table back into the examples it holds.
"""

from collections.abc import Sequence

import numpy as np
import torch

from src.data.tree_layout import DEPTH_COLUMN, POSITION_COLUMN, TRAINED_COLUMN, build_trie, serialize_tree, share_limit
from src.kernels.logprobs import LABEL_IGNORE_INDEX
from src.models.attention.tree_plan import NodeTable, build_node_table


def tree_attention_masks(
    position_ids: torch.Tensor, depth: torch.Tensor, sliding_window: int | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Dense ``[N, N]`` keep-masks (full, sliding) of a row.

    A query keeps a key on its own path at or before its own position; the window is measured in
    path positions, not in row order.
    """
    table = build_node_table(position_ids, depth)
    on_path = np.eye(len(table.starts), dtype=bool)
    for node, parent in enumerate(table.parents.tolist()):
        if parent != -1:
            on_path[node] |= on_path[parent]
    node_of = torch.from_numpy(np.repeat(np.arange(len(table.starts)), table.lengths))
    position = position_ids.flatten()
    full = torch.from_numpy(on_path)[node_of][:, node_of] & (position[None, :] <= position[:, None])
    if sliding_window is None:
        return full, None
    return full, full & (position[:, None] - position[None, :] < sliding_window)


def additive_mask(keep: torch.Tensor) -> torch.Tensor:
    """A keep-mask as the ``[1, 1, N, N]`` additive fp32 mask an eager forward takes."""
    return torch.zeros(keep.shape).masked_fill(~keep, torch.finfo(torch.float32).min)[None, None]


def leaf_chains(parents: Sequence[int]) -> dict[int, list[int]]:
    """Leaf index -> node indices from the root down to it, leaves in row order."""
    has_child = set(parents)
    chains = {}
    for leaf in range(len(parents)):
        if leaf in has_child:
            continue
        chain = [leaf]
        while parents[chain[-1]] != -1:
            chain.append(parents[chain[-1]])
        chains[leaf] = chain[::-1]
    return chains


def example_rows(table: NodeTable) -> list[torch.Tensor]:
    """Row indices along every root-to-leaf path, one tensor per example, leaves in row order."""
    return [
        torch.cat([torch.arange(table.starts[node], table.starts[node] + table.lengths[node]) for node in chain])
        for chain in leaf_chains(table.parents.tolist()).values()
    ]


def tree_columns(paths: list[list[int]], trained: list[list[bool]]) -> dict[str, torch.Tensor]:
    """The per-token columns of the one tree over ``paths``, plus the ``labels`` a collator derives."""
    nodes = build_trie(paths, [share_limit(mask) for mask in trained], 1)
    columns = {name: torch.tensor(values) for name, values in serialize_tree(nodes, paths, trained).items()}
    columns["labels"] = columns["input_ids"].masked_fill(~columns[TRAINED_COLUMN], LABEL_IGNORE_INDEX)
    return columns


def reference_varlen(calls: list[dict]):
    """A flash varlen kernel in plain torch: per sequence a bottom-right aligned causal/window softmax
    with the sink in the normalizer, returning ``(out, lse)`` in the kernel's layouts. Every call's
    shape facts are appended to ``calls``."""

    def varlen(
        q,
        k,
        v,
        *,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
        softmax_scale,
        window_size=(None, None),
        learnable_sink=None,
        dropout_p=0.0,
        deterministic=False,
        return_lse=False,
    ):
        calls.append(
            {
                "causal": causal,
                "window": window_size,
                "sink": learnable_sink is not None,
                "queries": q.shape[0],
                "keys": k.shape[0],
                "max_seqlen": (max_seqlen_q, max_seqlen_k),
            }
        )
        group = q.shape[1] // k.shape[1]
        outs, lses = [], []
        for b in range(len(cu_seqlens_q) - 1):
            queries = q[cu_seqlens_q[b] : cu_seqlens_q[b + 1]]
            keys = k[cu_seqlens_k[b] : cu_seqlens_k[b + 1]].repeat_interleave(group, dim=1)
            values = v[cu_seqlens_k[b] : cu_seqlens_k[b + 1]].repeat_interleave(group, dim=1)
            scores = torch.einsum("qhd,khd->hqk", queries, keys) * softmax_scale
            i = torch.arange(len(queries), device=q.device)[:, None] + len(keys) - len(queries)
            j = torch.arange(len(keys), device=q.device)[None, :]
            keep = torch.ones(len(queries), len(keys), dtype=torch.bool, device=q.device)
            if causal:
                keep &= j <= i
            if window_size[0] is not None:
                keep &= j >= i - window_size[0]
            scores = scores.masked_fill(~keep, float("-inf"))
            if learnable_sink is not None:
                scores = torch.cat([scores, learnable_sink.view(-1, 1, 1).expand(-1, len(queries), 1)], dim=-1)
            lses.append(torch.logsumexp(scores, dim=-1))
            outs.append(torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1)[..., : len(keys)], values))
        return torch.cat(outs), torch.cat(lses, dim=1)

    return varlen


def randomize_sinks(model: torch.nn.Module) -> None:
    """Give every attention sink a value of its own, so a path that drops the sink moves the output."""
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "sinks"):
                module.sinks.normal_(0.0, 1.0)


def concat_columns(trees: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Several trees as one packed row."""
    return {
        name: torch.cat([tree[name] for tree in trees])
        for name in ("input_ids", "labels", POSITION_COLUMN, DEPTH_COLUMN)
    }
