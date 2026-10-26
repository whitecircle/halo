#!/usr/bin/env python
"""CPU tests: the tree layout stores prefix-sharing examples once without changing any of them.

A tree row is only worth having if every root-to-leaf path is still exactly the flat example it came
from, and if the attention plan the collator derives gives each token that path and nothing else. A
shared span cut one token too far trains a completion from another branch's token; a plan that keeps
one sibling key is a plausible loss on leaked context. Both are checked here against brute-force
references over random tries: path reconstruction against the input paths, and each kernel plan
against the dense keep-mask.

Run: python tests/cpu/data/test_tree_layout.py  (or pytest)
"""

import random
from itertools import accumulate

import pytest
import torch

from src.data.tree_layout import (
    DEPTH_COLUMN,
    POSITION_COLUMN,
    TRAINED_COLUMN,
    build_trie,
    serialize_tree,
    share_limit,
    split_to_budget,
)
from src.models.attention import tree_plan
from src.models.attention.tree_plan import build_node_table, build_tree_attention_plan
from tests.common.trees import leaf_chains, tree_attention_masks

VOCAB = 50
UNBOUNDED = 10**9


def _random_paths(rng: random.Random) -> tuple[list[list[int]], list[list[bool]]]:
    """Paths sharing a trunk, some sharing a second level, each ending in a trained answer."""
    trunk = [rng.randrange(VOCAB) for _ in range(rng.randrange(0, 40))]
    paths, trained = [], []
    for _ in range(rng.randrange(1, 9)):
        middle = [rng.randrange(VOCAB) for _ in range(rng.randrange(0, 12))] if rng.random() < 0.5 else []
        tail = [rng.randrange(VOCAB) for _ in range(rng.randrange(1, 15))]
        answer = [rng.randrange(VOCAB) for _ in range(rng.randrange(1, 6))]
        path = trunk + middle + tail + answer
        paths.append(path)
        trained.append([False] * (len(path) - len(answer)) + [True] * len(answer))
    if rng.random() < 0.3:
        # The same prompt with another answer: sharing must stop before the first answer token.
        paths.append(paths[0][:-1] + [VOCAB + 1])
        trained.append(list(trained[0]))
    return paths, trained


def _row(trees, paths, trained) -> dict[str, list]:
    row = {"input_ids": [], TRAINED_COLUMN: [], POSITION_COLUMN: [], DEPTH_COLUMN: []}
    for tree in trees:
        for column, values in serialize_tree(tree, paths, trained).items():
            row[column] += values
    return row


def _plan_keep_mask(plan: dict, size: int, part: str, window: int | None) -> torch.Tensor:
    """The keep-mask a kernel computes from ``plan``: bottom-right causal over each node's keys, plus
    the rectangular ancestor pass of the split form."""
    keep = torch.zeros(size, size, dtype=torch.bool)
    bounds = plan["cu_seqlens_q"].tolist()
    section = plan[part]
    if "q_index" in section:
        for start, end in zip(bounds, bounds[1:], strict=False):
            for query in range(start, end):
                keep[query, start : query + 1] = True
        queries, keys = section["q_index"].tolist(), section["k_index"].tolist()
        q_bounds, k_bounds = section["cu_seqlens_q"].tolist(), section["cu_seqlens_k"].tolist()
        for sequence in range(len(q_bounds) - 1):
            ancestors = keys[k_bounds[sequence] : k_bounds[sequence + 1]]
            for query in queries[q_bounds[sequence] : q_bounds[sequence + 1]]:
                assert not keep[query, ancestors].any(), "an ancestor key is attended twice"
                keep[query, ancestors] = True
        return keep
    keys, k_bounds = section["k_index"].tolist(), section["cu_seqlens_k"].tolist()
    for node, (start, end) in enumerate(zip(bounds, bounds[1:], strict=False)):
        node_keys = keys[k_bounds[node] : k_bounds[node + 1]]
        offset = len(node_keys) - (end - start)
        for i in range(end - start):
            first = max(0, i + offset - (window - 1)) if window else 0
            keep[start + i, node_keys[first : i + offset + 1]] = True
    return keep


def _assert_stateful_parts(plan: dict, table, kernel: int) -> None:
    """The convolution context and the recurrence levels against the nodes' own root-to-leaf paths."""
    starts, lengths, parents = table.starts.tolist(), table.lengths.tolist(), table.parents.tolist()
    size = sum(lengths)
    path_rows: list[list[int]] = []  # row indices from the root down to each node's last token
    depths: list[int] = []
    for start, length, parent in zip(starts, lengths, parents, strict=True):
        path_rows.append([*(path_rows[parent] if parent != -1 else []), *range(start, start + length)])
        depths.append(depths[parent] + 1 if parent != -1 else 0)

    conv = plan["conv"]
    index, sequence = conv["index"].tolist(), conv["seq_idx"][0].tolist()
    for node, length in enumerate(lengths):
        behind = [row for row, owner in zip(index, sequence, strict=True) if owner == node]
        assert behind == path_rows[node][-(length + kernel - 1) :], "a node sits behind the last tokens of its path"
    assert conv["index"][conv["own"]].tolist() == list(range(size)), "every token's own output, once, in row order"

    above: list[int] = []
    for depth, level in enumerate(plan["levels"]):
        nodes = [node for node, node_depth in enumerate(depths) if node_depth == depth]
        assert level["index"].tolist() == [
            row for node in nodes for row in range(starts[node], starts[node] + lengths[node])
        ]
        assert level["cu_seqlens"].diff().tolist() == [lengths[node] for node in nodes]
        if depth:
            assert [above[slot] for slot in level["parent"].tolist()] == [parents[node] for node in nodes]
        else:
            assert "parent" not in level
        above = nodes
    assert len(plan["levels"]) == max(depths) + 1
    in_level_order = torch.cat([level["index"] for level in plan["levels"]])
    assert in_level_order[plan["restore"]].tolist() == list(range(size))


@pytest.mark.parametrize("seed", range(120))
def test_every_leaf_is_its_flat_example_and_plans_match_the_dense_mask(seed):
    rng = random.Random(seed)
    paths, trained = _random_paths(rng)
    nodes = build_trie(paths, [share_limit(mask) for mask in trained], rng.choice([1, 1, 4, 8]))
    budget = rng.choice([UNBOUNDED, UNBOUNDED, 60, 35])
    trees, dropped = split_to_budget(nodes, budget)

    kept = []
    for tree in trees:
        assert sum(node.length for node in tree) <= budget
        columns = serialize_tree(tree, paths, trained)
        ends = list(accumulate(node.length for node in tree))
        for leaf, chain in leaf_chains([node.parent for node in tree]).items():
            index = [i for node in chain for i in range(ends[node] - tree[node].length, ends[node])]
            assert [columns["input_ids"][i] for i in index] == paths[tree[leaf].path]
            assert [columns[TRAINED_COLUMN][i] for i in index] == trained[tree[leaf].path]
            kept.append(tree[leaf].path)
    fitting = [index for index, path in enumerate(paths) if len(path) <= budget]
    assert sorted(kept) == fitting
    assert dropped == len(paths) - len(fitting)
    if not kept:
        return

    row = _row(trees, paths, trained)
    positions, depths = torch.tensor(row[POSITION_COLUMN]), torch.tensor(row[DEPTH_COLUMN])
    table = build_node_table(positions, depths)
    # The nodes recovered from the two columns are the trie's own, parent for parent.
    offsets = [0, *accumulate(len(tree) for tree in trees)]
    assert table.lengths.tolist() == [node.length for tree in trees for node in tree]
    assert table.parents.tolist() == [
        -1 if node.parent == -1 else node.parent + offset
        for tree, offset in zip(trees, offsets, strict=False)
        for node in tree
    ]
    assert not any(row[TRAINED_COLUMN][start] for start in table.starts)

    window = rng.choice([None, 4, 9])
    full, sliding = tree_attention_masks(positions, depths, window)
    size = len(row["input_ids"])
    for form in ("gather", "split"):
        plan = build_tree_attention_plan(table, sliding_window=window, full_form=form)
        assert "conv" not in plan and "levels" not in plan
        # Counting every token once per example it belongs to gives back the flat token count.
        assert int(plan["leaf_count"].sum()) == sum(len(paths[index]) for index in kept)
        assert plan["max_seqlen_q"] == int(plan["cu_seqlens_q"].diff().max())
        if not table.has_branches:
            # Plain documents carry no attention parts: their position resets say everything.
            assert "full" not in plan and "sliding" not in plan
            continue
        assert torch.equal(_plan_keep_mask(plan, size, "full", None), full), form
        for part in (plan["full"], plan.get("sliding", {})):
            for side in ("q", "k"):
                if f"cu_seqlens_{side}" in part:
                    assert part[f"max_seqlen_{side}"] == int(part[f"cu_seqlens_{side}"].diff().max())
        if window:
            # A node's ancestors hold as many tokens as its first position; a window reaches window - 1 of them.
            reach = positions[table.starts].clamp(max=window - 1) + torch.from_numpy(table.lengths)
            assert plan["sliding"]["cu_seqlens_k"].diff().tolist() == reach.tolist()
            assert torch.equal(_plan_keep_mask(plan, size, "sliding", window), sliding)

    kernel = rng.choice([2, 4])
    stateful = build_tree_attention_plan(table, sliding_window=window, full_form="gather", conv_kernel=kernel)
    _assert_stateful_parts(stateful, table, kernel)


@pytest.mark.parametrize("seed", range(60))
def test_a_gather_over_the_call_budget_splits_into_runs_of_whole_nodes(seed, monkeypatch):
    """Each run is as many whole nodes as fit the budget (one node alone may exceed it), the runs tile
    the row's queries and gathered keys in order, and their bounds are the part's own, rebased."""
    rng = random.Random(seed)
    paths, trained = _random_paths(rng)
    trees, _ = split_to_budget(build_trie(paths, [share_limit(mask) for mask in trained], 1), UNBOUNDED)
    row = _row(trees, paths, trained)
    table = build_node_table(torch.tensor(row[POSITION_COLUMN]), torch.tensor(row[DEPTH_COLUMN]))
    if not table.has_branches:
        return
    window = rng.choice([None, 4])
    whole = build_tree_attention_plan(table, sliding_window=window, full_form="gather")
    assert "chunks" not in whole["full"], "a row this small fits one call at the default budget"
    budget = rng.randrange(1, int(whole["full"]["cu_seqlens_k"][-1]))
    monkeypatch.setattr(tree_plan, "GATHER_KEYS_PER_CALL", budget)
    plan = build_tree_attention_plan(table, sliding_window=window, full_form="gather")

    q_bounds = plan["cu_seqlens_q"].tolist()
    for name in ("full", "sliding") if window else ("full",):
        part, k_bounds = plan[name], plan[name]["cu_seqlens_k"].tolist()
        assert torch.equal(part["k_index"], whole[name]["k_index"]), "the gathered keys do not change"
        if k_bounds[-1] <= budget:
            assert "chunks" not in part
            continue
        first = 0
        for chunk in part["chunks"]:
            last = first + len(chunk["cu_seqlens_q"]) - 1
            assert chunk["q_span"] == (q_bounds[first], q_bounds[last])
            assert chunk["k_span"] == (k_bounds[first], k_bounds[last])
            assert chunk["cu_seqlens_q"].tolist() == [bound - q_bounds[first] for bound in q_bounds[first : last + 1]]
            assert chunk["cu_seqlens_k"].tolist() == [bound - k_bounds[first] for bound in k_bounds[first : last + 1]]
            for side in ("q", "k"):
                assert chunk[f"max_seqlen_{side}"] == int(chunk[f"cu_seqlens_{side}"].diff().max())
            keys = k_bounds[last] - k_bounds[first]
            assert keys <= budget or last == first + 1, "only a node over the budget on its own exceeds it"
            if last < len(q_bounds) - 1:
                assert k_bounds[last + 1] - k_bounds[first] > budget, "a run takes every node that fits"
            first = last
        assert first == len(q_bounds) - 1


def test_the_dense_mask_is_the_per_example_causal_mask():
    """The reference itself: a query keeps exactly the tokens of its own example at or before it."""
    paths = [[1, 2, 3, 4, 9], [1, 2, 3, 5, 6, 9], [1, 2, 7, 9]]
    trained = [[False] * (len(path) - 1) + [True] for path in paths]
    nodes = build_trie(paths, [share_limit(mask) for mask in trained], 1)
    row = serialize_tree(nodes, paths, trained)
    full, sliding = tree_attention_masks(torch.tensor(row[POSITION_COLUMN]), torch.tensor(row[DEPTH_COLUMN]), 3)
    # tokens in row order: [1 2] [3] [4 9] [5 6 9] [7 9]
    assert row["input_ids"] == [1, 2, 3, 4, 9, 5, 6, 9, 7, 9]
    assert full[4].tolist() == [True, True, True, True, True, False, False, False, False, False]
    assert full[7].tolist() == [True, True, True, False, False, True, True, True, False, False]
    assert full[9].tolist() == [True, True, False, False, False, False, False, False, True, True]
    # window 3 in PATH positions: the last token of the second example sees positions 3, 4, 5 only
    assert sliding[7].tolist() == [False, False, False, False, False, True, True, True, False, False]
    assert sliding[5].tolist() == [False, True, True, False, False, True, False, False, False, False]


def test_sharing_stops_one_token_before_a_completion():
    """Two examples with one prompt: the last prompt token is repeated, so each answer's first
    token is predicted from a token of its own leaf."""
    prompt = [3, 4, 5, 6]
    paths = [prompt + [10, 11], prompt + [20]]
    trained = [[False] * 4 + [True, True], [False] * 4 + [True]]
    assert [share_limit(mask) for mask in trained] == [3, 3]
    assert share_limit([False] * 4) == 3, "a path with nothing trained shares all but its last token"
    nodes = build_trie(paths, [3, 3], 1)
    assert [(node.parent, node.start, node.length) for node in nodes] == [(-1, 0, 3), (0, 3, 3), (0, 3, 2)]
    row = serialize_tree(nodes, paths, trained)
    assert row["input_ids"] == [3, 4, 5, 6, 10, 11, 6, 20]
    assert row[TRAINED_COLUMN] == [False] * 4 + [True, True, False, True]


def test_sharing_is_capped_by_the_limit_of_every_path_in_a_run():
    """A row whose prompt holds another row's completion (the same conversation one turn further)
    shares only up to that completion, whichever of the two sorts first."""
    prompt = [3, 4, 5, 6]
    cases = [
        ([prompt + [10, 11], prompt + [10, 11, 7, 20]], [[False] * 4 + [True] * 2, [False] * 7 + [True]]),
        ([prompt + [10, 1, 7, 20], prompt + [10, 11]], [[False] * 7 + [True], [False] * 4 + [True] * 2]),
    ]
    for paths, trained in cases:
        assert paths == sorted(paths)
        nodes = build_trie(paths, [share_limit(mask) for mask in trained], 1)
        assert (nodes[0].parent, nodes[0].start, nodes[0].length) == (-1, 0, 3)
        row = serialize_tree(nodes, paths, trained)
        assert sum(row[TRAINED_COLUMN]) == sum(map(sum, trained)), "a completion token lost its label to a shared node"
        table = build_node_table(torch.tensor(row[POSITION_COLUMN]), torch.tensor(row[DEPTH_COLUMN]))
        assert not any(row[TRAINED_COLUMN][start] for start in table.starts)


def test_paths_sharing_a_prefix_meet_in_whatever_order_they_arrive():
    paths = [[1, 2, 3, 10, 9], [7, 8, 9, 9], [1, 2, 3, 11, 9]]
    trained = [[False] * (len(path) - 1) + [True] for path in paths]
    nodes = build_trie(paths, [share_limit(mask) for mask in trained], 1)
    assert sum(node.length for node in nodes) == 3 + 2 + 2 + 4, "the two rows over [1, 2, 3] store it once"


def test_a_short_shared_span_is_repeated_not_split_off():
    paths = [[1, 2, 3, 30, 31], [1, 2, 3, 40, 41]]
    limits = [4, 4]
    assert len(build_trie(paths, limits, 3)) == 3
    roots = build_trie(paths, limits, 4)
    assert [(node.parent, node.start, node.length) for node in roots] == [(-1, 0, 5), (-1, 0, 5)]


def test_a_tree_over_budget_repeats_its_trunk_in_each_part():
    trunk = list(range(10))
    paths = [trunk + [20 + branch] * 6 for branch in range(4)]
    nodes = build_trie(paths, [15] * 4, 1)
    trees, dropped = split_to_budget(nodes, 25)
    assert dropped == 0
    assert [[(node.parent, node.length) for node in tree] for tree in trees] == [[(-1, 10), (0, 6), (0, 6)]] * 2
    assert split_to_budget(nodes, 15) == ([], 4)


@pytest.mark.parametrize(
    ("positions", "depths"),
    [
        pytest.param([0, 1, 2, 5, 6], [0, 0, 0, 1, 1], id="child-does-not-continue-its-parent"),
        pytest.param([0, 1, 2, 3, 4, 2, 3], [0, 0, 0, 0, 0, 1, 1], id="child-starts-inside-its-parent"),
        pytest.param([3, 4, 5], [1, 1, 1], id="row-starts-inside-a-tree"),
        pytest.param([1, 2, 3], [0, 0, 0], id="root-does-not-start-at-zero"),
        pytest.param([0, 1, 2, 3], [0, 0, 2, 2], id="depth-skips-a-level"),
    ],
)
def test_a_row_that_is_not_a_whole_depth_first_tree_is_refused(positions, depths):
    """A truncated or split row would attach a branch to the wrong trunk; it must not collate."""
    with pytest.raises(ValueError, match="not a depth-first tree layout"):
        build_node_table(torch.tensor(positions), torch.tensor(depths))


def test_consecutive_plain_documents_are_separate_roots():
    table = build_node_table(torch.tensor([0, 1, 2, 0, 0, 1]), torch.zeros(6, dtype=torch.long))
    assert table.starts.tolist() == [0, 3, 4]
    assert table.lengths.tolist() == [3, 1, 2]
    assert not table.has_branches


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
