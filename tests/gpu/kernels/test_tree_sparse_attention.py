#!/usr/bin/env python
"""A GLM-5 Next tree row on the real kernels matches every example run alone.

The CPU tier runs the KDA layers through the toolkit's torch fallbacks and the sparse-attention
products at fp32; this pins what only the CUDA path can break. The KDA scan runs fla's ``chunk_kda``
with one initial state per varlen sequence and a gradient through both states, its convolution runs
``causal_conv1d`` over a gathered row with ``seq_idx``, and the sparse attention runs its products on
the tensor cores with fp32 results.

A tiny GLM-5 Next runs a row of trees of several shapes, in bf16 and in fp32, against the fp32 stock
forward of each example on its own: hidden states and mixer gradients, the same row without its plan
as a negative control, and the KDA warm-up running every state variant a training row compiles. Its
top-k exceeds every path, so each query attends all its visible pools: a selection cut ranks scores
that the tree and the example alone reduce in different orders, and a near-tie swapped there moves a
token by a whole pool — that would measure rounding, not the mixers. The selection itself is pinned
on an fp64 model, on paths that outgrow a 16-key top-k: the tree's key sets against the stock indexer's
on every example. Both score in fp32 there, from inputs that agree to fp64 rounding, so only two pools
within fp32 rounding of each other at a cut could split them. Every expert is active: top-k routing
flips on bf16 noise, which would measure the router.
"""

import copy
import functools

import torch

from halo.data.tree_layout import DEPTH_COLUMN, POSITION_COLUMN
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.models.attention.tree import (
    TREE_FAMILIES,
    _on_device,
    install_tree_attention,
    resolve_tree_plan_options,
    warm_recurrent_mixer_kernels,
)
from halo.models.attention.tree_plan import TREE_ATTENTION_KEY, build_node_table, build_tree_attention_plan
from halo.models.attention.tree_sparse import tree_key_selection
from tests.common.harness import gpu_test_main
from tests.common.numerics import max_abs_rel_err
from tests.common.rank_log import log
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model
from tests.common.trees import concat_columns, example_rows, randomize_indexer_pools, tree_columns

VOCAB = 1024
BF16_REL_TOL = 4e-2  # bf16 through four layers against fp32
FP32_REL_TOL = 1e-3
GRAD_COSINE_FLOOR = 0.999
LEAK_REL_FLOOR = 0.3  # a row without its plan must be far from the reference, or the checks compare nothing
FAMILY = "glm5_next_text"
# Every expert active; a top-k past every path of the row (the longest is 406 tokens); the tail on.
OVERRIDES = {"index_topk": 512, "index_kpool_always_select_tail": True, "num_experts_per_tok": 8}
# The selection check: eight indexer heads (fewer score many pools exactly zero, a tie the stock
# top-k breaks arbitrarily) and sixteen keys, so every path past twenty tokens selects.
SELECTING = {"index_n_heads": 8, "index_topk": 16}
# (takes a state, returns one, normalizes query and key itself) of the recurrence calls along a
# chain: its root, its middle, its leaf. A tree row is normalized once, ahead of the depths.
RECURRENCE_STATE_VARIANTS = {(False, True, False), (True, True, False), (True, False, False)}


def _tree_row(generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Trees of several shapes in one row: trunks shorter and longer than a pool's reach, a second
    level, branches from 2 to 70 tokens, and a plain document."""

    def tokens(n: int) -> list[int]:
        return torch.randint(0, VOCAB, (n,), generator=generator).tolist()

    trees = []
    for trunk_len, fan_out in ((5, 3), (41, 5), (301, 6), (90, 1)):
        trunk, paths, trained = tokens(trunk_len), [], []
        for branch in range(fan_out):
            middle = tokens(30) if branch % 2 else []
            for tail_len in (2, 21, 70)[: 1 + branch % 3]:
                answer = 1 + branch % 4
                path = trunk + middle + tokens(tail_len + answer)
                paths.append(path)
                trained.append([False] * (len(path) - answer) + [True] * answer)
        trees.append(tree_columns(paths, trained))
    return {name: column.cuda() for name, column in concat_columns(trees).items()}


def _hidden(model, row, **kwargs) -> torch.Tensor:
    return model.model(
        input_ids=row["input_ids"][None], position_ids=row[POSITION_COLUMN][None], use_cache=False, **kwargs
    ).last_hidden_state[0]


def _loss(model, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    targets = torch.nonzero(labels != LABEL_IGNORE_INDEX).flatten()
    logits = model.lm_head(hidden[targets - 1]).float()
    return torch.nn.functional.cross_entropy(logits, labels[targets], reduction="sum")


def _mixer_gradients(model) -> torch.Tensor:
    return torch.cat(
        [
            parameter.grad.float().flatten()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and "self_attn" in name
        ]
    )


def _alone(model, row, examples) -> tuple[torch.Tensor, torch.Tensor]:
    """Hidden states of every example run on its own by the stock forward, laid back on the row, and
    the mixer gradients of the summed loss over their trained tokens."""
    model.zero_grad(set_to_none=True)
    hidden = torch.zeros(len(row["input_ids"]), model.config.get_text_config().hidden_size, device="cuda")
    for rows in examples:
        states = model.model(input_ids=row["input_ids"][rows][None], use_cache=False).last_hidden_state[0]
        hidden[rows] = states.detach().float()
        _loss(model, states, row["labels"][rows]).backward()
    return hidden, _mixer_gradients(model)


def _record_recurrence_states(module, name: str) -> set[tuple[bool, bool, bool]]:
    """Note, for every call of the recurrence from here on, whether it took a state, whether it
    returned one and whether it normalized. Installed before tree attention, so it sees what the
    kernel is asked."""
    seen: set[tuple[bool, bool, bool]] = set()
    stock = getattr(module, name)

    @functools.wraps(stock)
    def recording(*args, initial_state=None, output_final_state=False, **kwargs):
        seen.add((initial_state is not None, output_final_state, kwargs["use_qk_l2norm_in_kernel"]))
        return stock(*args, initial_state=initial_state, output_final_state=output_final_state, **kwargs)

    setattr(module, name, recording)
    return seen


def _selection_matches_alone(row, table) -> bool:
    """In fp64 on the GPU, the tree selection of every query is the stock indexer's on its example alone."""
    torch.manual_seed(1)
    model = tiny_family_model(TINY_MOE_FAMILIES["glm5_next"], overrides={**OVERRIDES, **SELECTING}).double().cuda()
    randomize_indexer_pools(model)
    attention = next(
        layer.self_attn for layer in model.model.language_model.layers if getattr(layer.self_attn, "indexer", None)
    )
    links = _on_device(
        build_tree_attention_plan(table, full_form="gather", **resolve_tree_plan_options(model.config))["sparse"],
        "cuda",
    )
    generator = torch.Generator(device="cuda").manual_seed(2)
    hidden = torch.randn(1, len(row["input_ids"]), attention.hidden_size, generator=generator, device="cuda").double()
    big = torch.iinfo(torch.long).max
    with torch.no_grad():
        q_resid = attention.q_a_layernorm(attention.q_a_proj(hidden))
        tree = tree_key_selection(attention.indexer, hidden, q_resid, links)[0].long()
        same = selecting = queries = 0
        for rows in example_rows(table):
            rows = rows.cuda()
            solo = attention.indexer(
                hidden[:, rows], q_resid[:, rows], torch.ones(1, len(rows), dtype=torch.bool, device="cuda"), None
            )[0].long()
            solo = torch.where(solo >= 0, rows[solo.clamp_min(0)], solo)
            mine, theirs = (torch.where(k >= 0, k, big).sort(-1).values for k in (tree[rows], solo))
            same += int((mine == theirs).all(-1).sum())
            queries += len(rows)
            selecting += int((torch.arange(len(rows)) >= attention.indexer.index_topk + 4).sum())
    log(f"  [fp64 selection] {same} / {queries} queries select the stock indexer's keys ({selecting} past the top-k)")
    return same == queries and selecting > 0


@gpu_test_main(exact_world_size=1, prefix="tree_sparse_attention")
def run(ctx):
    row = _tree_row(torch.Generator().manual_seed(0))
    table = build_node_table(row[POSITION_COLUMN].cpu(), row[DEPTH_COLUMN].cpu())
    log(f"  row: {len(row['input_ids'])} tokens, {len(table.starts)} nodes, depth {int(row[DEPTH_COLUMN].max())}")
    examples = [rows.cuda() for rows in example_rows(table)]

    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES["glm5_next"], overrides=OVERRIDES)
    randomize_indexer_pools(model)
    family = TREE_FAMILIES[FAMILY]
    recurrence_states = _record_recurrence_states(family.module, family.stateful.recurrence)
    reference, truth = _alone(copy.deepcopy(model).float().cuda().train(), row, examples)
    plan = {
        TREE_ATTENTION_KEY: _on_device(
            build_tree_attention_plan(table, full_form="gather", **resolve_tree_plan_options(model.config)),
            row["labels"].device,
        )
    }

    checks = {}
    for dtype, tolerance in ((torch.float32, FP32_REL_TOL), (torch.bfloat16, BF16_REL_TOL)):
        tag = str(dtype).removeprefix("torch.")
        replica = copy.deepcopy(model).to(dtype).cuda()
        install_tree_attention(replica, "split")
        replica.eval()
        with torch.no_grad():
            rel = max_abs_rel_err(_hidden(replica, row, **plan), reference)
            log(f"  [{tag}] hidden vs each example alone (fp32) rel {rel:.2e}")
            checks[f"{tag}_hidden_matches_examples_alone"] = rel < tolerance
            leak = max_abs_rel_err(_hidden(replica, row), reference)
            log(f"  [{tag}, no plan] hidden vs each example alone (fp32) rel {leak:.2e}")
            checks[f"{tag}_row_without_plan_is_far_from_examples_alone"] = leak > LEAK_REL_FLOOR
        replica.train()
        replica.zero_grad(set_to_none=True)
        _loss(replica, _hidden(replica, row, **plan), row["labels"]).backward()
        cosine = torch.nn.functional.cosine_similarity(_mixer_gradients(replica), truth, dim=0).item()
        log(f"  [{tag}] mixer gradient cosine vs examples alone (fp32) {cosine:.6f}")
        checks[f"{tag}_gradients_match_examples_alone"] = cosine > GRAD_COSINE_FLOOR

    recurrence_states.clear()
    warm_recurrent_mixer_kernels(replica, dtype=torch.bfloat16)
    checks["recurrence_warm_up_runs_every_state_variant"] = recurrence_states == RECURRENCE_STATE_VARIANTS
    checks["fp64_selection_is_the_stock_indexers_on_every_path"] = _selection_matches_alone(row, table)
    return {"checks": checks}


if __name__ == "__main__":
    run()
