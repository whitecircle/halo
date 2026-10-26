"""Tree layout of prefix-sharing examples: token paths to the trie over them, split to a row budget and laid out
depth-first as per-token columns. A per-token packer concatenates trees untouched, and the collator recovers every
node from the flattened batch (:func:`~src.models.attention.tree_plan.build_node_table`). Runs in the dataset maps."""

from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np

from src.models.attention.tree_plan import resolve_node_depths

# Per-token dataset columns of a tree row beside ``input_ids``: the position along the token's own root-to-leaf
# path, the depth of its node, and whether it is trained (all a completion-only label adds to ``input_ids``).
POSITION_COLUMN = "position_ids"
DEPTH_COLUMN = "tree_depth"
TRAINED_COLUMN = "trained"


@dataclass(frozen=True)
class _TreeNode:
    """A run of tokens shared by every path below it: ``length`` tokens of ``path`` from position ``start``."""

    parent: int
    start: int
    path: int
    length: int


def share_limit(trained: list[bool]) -> int:
    """How many leading tokens of a path other paths may share.

    Sharing stops one token before the first trained one: a node boundary breaks the next-token
    pairing of the loss, so the token a completion is first predicted from must sit in the leaf with it.
    """
    first = trained.index(True) if True in trained else len(trained)
    return max(first - 1, 0)


def _common_prefix(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    differs = a[:n] != b[:n]
    return int(differs.argmax()) if differs.any() else n


def build_trie(paths: list[list[int]], share_limits: list[int], min_shared: int) -> list[_TreeNode]:
    """Depth-first nodes of the trie over ``paths``; a shared span under ``min_shared`` (at least 1)
    tokens is not split off.

    Paths are sorted, so every subtree is a contiguous run and the shared length of a run is the
    minimum over its adjacent pairs. Each adjacent value is capped by both paths' ``share_limits``;
    a run then never shares past the limit of any path inside it.
    """
    order = sorted(range(len(paths)), key=paths.__getitem__)
    arrays = [np.asarray(paths[index]) for index in order]
    shared = [
        min(_common_prefix(a, b), share_limits[i], share_limits[j])
        for (a, i), (b, j) in pairwise(zip(arrays, order, strict=True))
    ]
    nodes: list[_TreeNode] = []
    pending = [(0, len(order), -1, 0)]  # paths order[lo:hi] share their first `start` tokens, held by `parent`
    while pending:
        lo, hi, parent, start = pending.pop()
        if hi - lo == 1:
            nodes.append(_TreeNode(parent=parent, start=start, path=order[lo], length=len(arrays[lo]) - start))
            continue
        common = min(shared[lo : hi - 1])
        if common - start >= min_shared:
            nodes.append(_TreeNode(parent=parent, start=start, path=order[lo], length=common - start))
            parent, start = len(nodes) - 1, common
        cuts = [lo, *(i + 1 for i in range(lo, hi - 1) if shared[i] <= common), hi]
        pending.extend((a, b, parent, start) for a, b in reversed(list(pairwise(cuts))))
    return nodes


def split_to_budget(nodes: list[_TreeNode], max_tokens: int) -> tuple[list[list[_TreeNode]], int]:
    """Partition a trie into trees of at most ``max_tokens`` tokens, each repeating the ancestors it needs.

    Leaves are taken in order and a tree is closed when the next leaf's unshared tokens would
    overflow it. A leaf whose own path exceeds the budget is dropped and counted, as an over-length
    row is. Returned trees are depth-first with parents re-indexed.
    """
    has_child = {node.parent for node in nodes}
    trees: list[list[_TreeNode]] = []
    members: dict[int, int] = {}  # node index in `nodes` -> index in the open tree
    size = dropped = 0

    def close() -> None:
        if members:
            trees.append([replace(nodes[i], parent=members.get(nodes[i].parent, -1)) for i in members])

    for leaf, node in enumerate(nodes):
        if leaf in has_child:
            continue
        if node.start + node.length > max_tokens:
            dropped += 1
            continue
        chain = [leaf]
        while nodes[chain[-1]].parent != -1:
            chain.append(nodes[chain[-1]].parent)
        fresh = [i for i in reversed(chain) if i not in members]
        if size + sum(nodes[i].length for i in fresh) > max_tokens:
            close()
            members, size, fresh = {}, 0, chain[::-1]
        for i in fresh:
            members[i] = len(members)
        size += sum(nodes[i].length for i in fresh)
    close()
    return trees, dropped


def serialize_tree(nodes: list[_TreeNode], paths: list[list[int]], trained: list[list[bool]]) -> dict[str, list]:
    """The per-token columns of one tree. Only a leaf's tokens are trained; shared tokens never are."""
    has_child = {node.parent for node in nodes}
    depths = resolve_node_depths(node.parent for node in nodes)
    row: dict[str, list] = {"input_ids": [], TRAINED_COLUMN: [], POSITION_COLUMN: [], DEPTH_COLUMN: []}
    for index, node in enumerate(nodes):
        span = slice(node.start, node.start + node.length)
        row["input_ids"] += paths[node.path][span]
        row[TRAINED_COLUMN] += [False] * node.length if index in has_child else trained[node.path][span]
        row[POSITION_COLUMN] += range(span.start, span.stop)
        row[DEPTH_COLUMN] += [depths[index]] * node.length
    return row
