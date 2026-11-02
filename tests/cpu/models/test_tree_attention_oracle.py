#!/usr/bin/env python
"""CPU tests: a tree forward is the per-example forward, through the oracle and through the flash seam.

Three layers of the same claim, on a tiny model of every family tree attention admits (GPT-OSS with
live sinks and a sliding window short enough to cross node boundaries; Qwen3 dense and MoE; Qwen3.5
dense and MoE, whose GatedDeltaNet layers carry a convolution context and a recurrent state):

* under the dense keep-masks of :func:`~tests.common.trees.tree_attention_masks` (eager attention)
  every example has the logits and gradients it has when run alone; a stateful family's
  linear-attention layers read their part of the plan there too;
* the kernel plans, run through the real transformers flash path and the registry view with a
  reference implementation standing in for the CUDA kernel, give the same in both forms;
* the custom autograd pieces of the seam are exact: the ancestor merge and the query/key
  normalization against finite differences, the key gather against a plain fp32 scatter;
* under gradient checkpointing, reentrant or not, a tree row trains to the logits and gradients it
  trains to without: the recompute reads the plan as the first pass did.

The CUDA kernels themselves are pinned by ``tests/gpu/kernels/test_fa4_tree_attention.py``.

Run: python tests/cpu/models/test_tree_attention_oracle.py  (or pytest)
"""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import modeling_flash_attention_utils as flash_utils

from halo.data.tree_layout import DEPTH_COLUMN, POSITION_COLUMN
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.models import segment_markers
from halo.models.attention import tree as tree_attention
from halo.models.attention import tree_plan
from halo.models.attention.tree import (
    TREE_FAMILIES,
    TreeAttentionRegistry,
    TreeFamily,
    _GatherTokens,
    _L2Norm,
    _MergeAncestors,
    _on_device,
    _tree_recurrence,
    install_tree_attention,
    resolve_tree_plan_options,
)
from halo.models.attention.tree_plan import TREE_ATTENTION_KEY, build_node_table, build_tree_attention_plan
from tests.common.tiny_models import (
    TINY_DENSE_FAMILY,
    TINY_MOE_FAMILIES,
    TINY_QWEN35_DENSE_FAMILY,
    tiny_family_model,
)
from tests.common.trees import (
    additive_mask,
    checkpointing_gaps,
    example_rows,
    randomize_sinks,
    reference_varlen,
    tree_attention_masks,
    tree_columns,
)

WINDOW = 6
VOCAB = 1024
# A tiny model per admitted model type. GPT-OSS gets a window that crosses node boundaries; the
# Qwen3.5 pair one period of its layer pattern (three GatedDeltaNet layers, one attention layer).
FAMILIES = {
    "gpt_oss": (TINY_MOE_FAMILIES["gpt_oss"], {"sliding_window": WINDOW, "num_hidden_layers": 4}),
    "qwen3": (TINY_DENSE_FAMILY, {"num_hidden_layers": 3}),
    "qwen3_moe": (TINY_MOE_FAMILIES["qwen3_moe"], {"num_hidden_layers": 3}),
    "qwen3_5_text": (TINY_QWEN35_DENSE_FAMILY, {"num_hidden_layers": 4}),
    "qwen3_5_moe_text": (TINY_MOE_FAMILIES["qwen3_5_moe_text"], {"num_hidden_layers": 4}),
}


def test_every_admitted_family_is_swept():
    """Families whose softmax layers read the plan at the flash seam; the key-selecting ones are swept
    by test_tree_sparse_attention.py."""
    assert set(FAMILIES) == {name for name, family in TREE_FAMILIES.items() if family.sparse is None}


def _paths() -> tuple[list[list[int]], list[list[int]]]:
    """A three-level tree with branches shorter and longer than the window, each ending in an answer. The
    two-level branch opens on token 0, so it sorts first under the trunk and the depth-by-depth order of the
    row's tokens is not their row order."""
    generator = torch.Generator().manual_seed(0)

    def tokens(n: int) -> list[int]:
        return torch.randint(0, VOCAB, (n,), generator=generator).tolist()

    trunk, middle = tokens(11), [0, *tokens(4)[1:]]
    paths = [trunk + tokens(5), trunk + middle + tokens(3), trunk + middle + tokens(9), trunk + tokens(12)]
    answers = (2, 2, 3, 4)
    labels = [[LABEL_IGNORE_INDEX] * (len(path) - n) + path[-n:] for path, n in zip(paths, answers, strict=True)]
    return paths, labels


def _stateful_plan(columns) -> dict:
    table = build_node_table(columns[POSITION_COLUMN], columns[DEPTH_COLUMN])
    return build_tree_attention_plan(table, sliding_window=None, full_form="gather", conv_kernel=4)


@pytest.fixture(scope="module")
def tree():
    paths, labels = _paths()
    columns = tree_columns(paths, [[label != LABEL_IGNORE_INDEX for label in path_labels] for path_labels in labels])
    assert int(columns[DEPTH_COLUMN].max()) == 2
    # A depth order that is its own inverse (the identity included) would hide a recurrence that skips or
    # misapplies the permutation back to row order.
    order = _stateful_plan(columns)["depth_order"]
    assert not torch.equal(order[order], torch.arange(len(order)))
    return paths, labels, columns


@pytest.fixture(scope="module", params=sorted(FAMILIES))
def model(request):
    torch.manual_seed(0)
    family, overrides = FAMILIES[request.param]
    model = tiny_family_model(family, overrides=overrides).eval()
    assert model.config.model_type == request.param
    randomize_sinks(model)
    return model


def _window(model) -> int | None:
    return getattr(model.config, "sliding_window", None)


def _oracle_masks(model, columns) -> dict[str, torch.Tensor] | torch.Tensor:
    """The dense tree masks in the shape the family's forward takes: one per layer type, or one."""
    full, sliding = tree_attention_masks(columns[POSITION_COLUMN], columns[DEPTH_COLUMN], _window(model))
    if sliding is None:
        return additive_mask(full)
    return {"full_attention": additive_mask(full), "sliding_attention": additive_mask(sliding)}


def _loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits[:-1], labels[1:], ignore_index=LABEL_IGNORE_INDEX, reduction="sum")


def _tree_forward(model, columns, **kwargs) -> torch.Tensor:
    return model(
        input_ids=columns["input_ids"][None], position_ids=columns[POSITION_COLUMN][None], use_cache=False, **kwargs
    ).logits[0]


def _plan(model, columns, form: str) -> dict:
    table = build_node_table(columns[POSITION_COLUMN], columns[DEPTH_COLUMN])
    return {
        TREE_ATTENTION_KEY: build_tree_attention_plan(table, full_form=form, **resolve_tree_plan_options(model.config))
    }


def _install_reference_flash_kernel(monkeypatch, calls: list[dict]) -> None:
    """transformers' FA4 path with the reference varlen kernel standing in for the CUDA one.

    The path's kwargs processor is built on importing ``flash_attn.cute``, which an image may ship
    unimportable (FA2's source tree does on Hopper)."""
    pytest.importorskip("flash_attn.cute")
    flash_utils.lazy_import_flash_attention("flash_attention_4")
    monkeypatch.setattr(flash_utils, "_flash_varlen_fn", reference_varlen(calls))
    monkeypatch.setattr(flash_utils, "_flash_fn", None)


def _install(model, monkeypatch, form: str) -> TreeFamily:
    """Tree attention on ``model``'s family for one test; every module attribute it replaces is restored."""
    family = TREE_FAMILIES[model.config.model_type]
    replaced = ["ALL_ATTENTION_FUNCTIONS"]
    if family.stateful is not None:
        replaced += [family.stateful.conv, family.stateful.recurrence]
    for name in replaced:
        monkeypatch.setattr(family.module, name, getattr(family.module, name))
    # The segment-aware kernels are CUDA builds; on CPU the toolkit's own routes stand in for them.
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", ())
    monkeypatch.setattr(model.config, "_attn_implementation", "flash_attention_4")
    install_tree_attention(model, form)
    assert isinstance(family.module.ALL_ATTENTION_FUNCTIONS, TreeAttentionRegistry)
    return family


@pytest.fixture(scope="module")
def alone(model, tree) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(row indices, logits) of every example of the tree run on its own by the stock forward."""
    _, _, columns = tree
    with torch.no_grad():
        return [
            (rows, model(input_ids=columns["input_ids"][rows][None], use_cache=False).logits[0])
            for rows in example_rows(build_node_table(columns[POSITION_COLUMN], columns[DEPTH_COLUMN]))
        ]


def _worst_gap(tree_logits: torch.Tensor, alone) -> float:
    return max((tree_logits[rows] - logits).abs().max().item() for rows, logits in alone)


def test_the_dense_tree_masks_reproduce_every_example_alone(model, tree, alone, monkeypatch):
    paths, labels, columns = tree
    backend = model.config._attn_implementation
    family = _install(model, monkeypatch, "gather")
    monkeypatch.setattr(model.config, "_attn_implementation", backend)
    # Attention takes the dense masks; only a stateful family's linear-attention layers read the plan.
    mixers = _plan(model, columns, "gather") if family.stateful else {}

    model.zero_grad()
    tree_logits = _tree_forward(model, columns, attention_mask=_oracle_masks(model, columns), **mixers)
    assert _worst_gap(tree_logits, alone) < 5e-5
    _loss(tree_logits, columns["labels"]).backward()
    tree_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    model.zero_grad()
    by_tokens = {tuple(path): path_labels for path, path_labels in zip(paths, labels, strict=True)}
    for rows, _ in alone:
        path = columns["input_ids"][rows].tolist()
        logits = model(input_ids=torch.tensor([path]), use_cache=False).logits[0]
        _loss(logits, torch.tensor(by_tokens[tuple(path)])).backward()
    scale = max(grad.abs().max().item() for grad in tree_grads.values())
    drift = max(
        (p.grad - tree_grads[name]).abs().max().item() for name, p in model.named_parameters() if p.grad is not None
    )
    assert drift < 1e-5 * scale

    # Without the masks the row is one causal sequence; with the masks alone a stateful family's
    # convolution and recurrence still run across it; with the window in row order a branch loses
    # its trunk. Each must move the logits, or the checks above compare nothing.
    assert (_tree_forward(model, columns, **mixers) - tree_logits).abs().max() > 1e-2
    if family.stateful:
        masked_only = _tree_forward(model, columns, attention_mask=_oracle_masks(model, columns))
        assert (masked_only - tree_logits).abs().max() > 1e-2
    if _window(model) is None:
        return
    full, _ = tree_attention_masks(columns[POSITION_COLUMN], columns[DEPTH_COLUMN], WINDOW)
    order = torch.arange(len(full))
    row_window = {
        "full_attention": additive_mask(full),
        "sliding_attention": additive_mask(full & (order[:, None] - order[None, :] < WINDOW)),
    }
    assert (_tree_forward(model, columns, attention_mask=row_window) - tree_logits).abs().max() > 1e-2


@pytest.mark.parametrize("form", ["split", "gather"])
def test_the_kernel_plans_reproduce_every_example_alone_through_the_flash_path(model, tree, alone, form, monkeypatch):
    _, _, columns = tree
    window = _window(model)
    has_sinks = any(hasattr(layer, "sinks") for layer in model.modules())

    calls: list[dict] = []
    _install_reference_flash_kernel(monkeypatch, calls)
    _install(model, monkeypatch, form)

    table = build_node_table(columns[POSITION_COLUMN], columns[DEPTH_COLUMN])
    logits = _tree_forward(model, columns, **_plan(model, columns, form))
    assert _worst_gap(logits, alone) < 5e-5

    tokens, nodes = len(columns["input_ids"]), len(table.starts)
    sliding = [call for call in calls if call["window"][0] is not None]
    assert bool(sliding) == (window is not None)
    # The window never reaches past its own length, so the sliding gather is bounded per node.
    assert all(call["keys"] <= tokens + (WINDOW - 1) * nodes for call in sliding)
    assert all(isinstance(length, int) for call in calls for length in call["max_seqlen"])
    full = [call for call in calls if call["window"][0] is None]
    if form == "split":
        # Per layer: the stock packed call (with the family's sink), then the sink-less rectangular
        # ancestor pass.
        assert [(call["causal"], call["sink"], call["queries"]) for call in full[:2]] == [
            (True, has_sinks, tokens),
            (False, False, tokens - int(table.lengths[table.parents == -1].sum())),
        ]
    else:
        assert all(call["causal"] and call["sink"] == has_sinks and call["keys"] > tokens for call in full)

    # A row without its plan runs as one causal sequence through the same path.
    assert _worst_gap(_tree_forward(model, columns), alone) > 1e-2


def test_a_gather_over_the_call_budget_gives_the_one_call_logits_and_gradients(model, tree, monkeypatch):
    """Under a budget smaller than the row's gathered keys every gather layer attends in several calls,
    none over the budget, with the logits of the single call and its gradients."""
    _, _, columns = tree
    calls: list[dict] = []
    _install_reference_flash_kernel(monkeypatch, calls)
    _install(model, monkeypatch, "gather")

    def run() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        calls.clear()
        model.zero_grad()
        logits = _tree_forward(model, columns, **_plan(model, columns, "gather"))
        _loss(logits, columns["labels"]).backward()
        return logits.detach(), {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    one_logits, one_grads = run()
    one_calls = len(calls)
    budget = max(call["keys"] for call in calls) // 3
    longest_node = int(_plan(model, columns, "gather")[TREE_ATTENTION_KEY]["full"]["max_seqlen_k"])
    assert longest_node <= budget, "the tree must leave room for whole nodes under the budget"
    for module in (tree_plan, tree_attention):
        monkeypatch.setattr(module, "GATHER_KEYS_PER_CALL", budget)
    chunked_logits, chunked_grads = run()

    assert len(calls) > one_calls and all(call["keys"] <= budget for call in calls)
    assert torch.equal(chunked_logits, one_logits)
    scale = max(grad.abs().max().item() for grad in one_grads.values())
    assert max((chunked_grads[name] - grad).abs().max().item() for name, grad in one_grads.items()) <= 1e-6 * scale


@pytest.mark.parametrize("use_reentrant", [True, False], ids=["reentrant", "non_reentrant"])
@pytest.mark.parametrize("form", ["split", "gather"])
@pytest.mark.parametrize("model_type", sorted(FAMILIES))
def test_a_tree_row_trains_identically_under_gradient_checkpointing(
    model_type, form, use_reentrant, tree, monkeypatch
):
    """A checkpointed decoder layer runs its forward again in the backward, so the flash view and a stateful family's
    convolution and scan are handed the plan a second time and must read it as the first pass did."""
    _, _, columns = tree
    tiny_family, overrides = FAMILIES[model_type]
    torch.manual_seed(0)
    model = tiny_family_model(tiny_family, overrides=overrides)
    randomize_sinks(model)
    _install_reference_flash_kernel(monkeypatch, [])
    _install(model, monkeypatch, form)
    plan = _plan(model, columns, form)

    def step() -> torch.Tensor:
        logits = _tree_forward(model, columns, **plan)
        _loss(logits, columns["labels"]).backward()
        return logits.detach()

    logit_gap, gradient_gap = checkpointing_gaps(model, step, use_reentrant=use_reentrant)
    assert logit_gap <= 1e-6 and gradient_gap <= 1e-6


def test_a_lora_wrapped_model_reads_the_tree_under_the_adapter(tree, monkeypatch):
    """A PeftModel calls the model it wraps through its own forward and nests the attention layers
    one wrapper down: the plan has to arrive there and the layers have to be found there."""
    _, _, columns = tree
    tiny_family, overrides = FAMILIES["qwen3_moe"]
    torch.manual_seed(0)
    adapter = LoraConfig(r=4, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
    model = get_peft_model(tiny_family_model(tiny_family, overrides=overrides), adapter).eval()
    table = build_node_table(columns[POSITION_COLUMN], columns[DEPTH_COLUMN])
    with torch.no_grad():
        alone = [
            (rows, model(input_ids=columns["input_ids"][rows][None], use_cache=False).logits[0])
            for rows in example_rows(table)
        ]
        _install_reference_flash_kernel(monkeypatch, [])
        _install(model, monkeypatch, "split")
        assert _worst_gap(_tree_forward(model, columns, **_plan(model, columns, "split")), alone) < 5e-5
        assert _worst_gap(_tree_forward(model, columns), alone) > 1e-2


def test_the_ancestor_merge_matches_finite_differences():
    generator = torch.Generator().manual_seed(1)
    out = torch.randn(9, 3, 4, dtype=torch.double, generator=generator, requires_grad=True)
    lse = torch.randn(3, 9, dtype=torch.double, generator=generator, requires_grad=True)
    out_anc = torch.randn(4, 3, 4, dtype=torch.double, generator=generator, requires_grad=True)
    lse_anc = torch.randn(3, 4, dtype=torch.double, generator=generator, requires_grad=True)
    index = torch.tensor([2, 3, 7, 8])
    assert torch.autograd.gradcheck(
        lambda *t: _MergeAncestors.apply(*t, index), (out, lse, out_anc, lse_anc), atol=1e-7
    )

    merged = _MergeAncestors.apply(out, lse, out_anc, lse_anc, index)
    weight = torch.softmax(torch.stack([lse[:, index], lse_anc]), dim=0).transpose(1, 2).unsqueeze(-1)
    assert torch.allclose(merged[index], weight[0] * out[index] + weight[1] * out_anc)
    untouched = torch.tensor([0, 1, 4, 5, 6])
    assert torch.equal(merged[untouched], out[untouched])


def test_the_query_key_normalization_is_the_scans_own_and_matches_finite_differences():
    module = TREE_FAMILIES["qwen3_5_text"].module
    states = torch.randn(2, 5, 3, 8, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: _L2Norm.apply(x, 1e-6), (states,), atol=1e-7)
    narrow = torch.randn(1, 7, 3, 8)
    assert torch.allclose(_L2Norm.apply(narrow, 1e-6), module.l2norm(narrow, eps=1e-6), atol=1e-7, rtol=0)
    # Under bf16 states the scale is computed in fp32, as the kernels compute it.
    half = narrow.to(torch.bfloat16)
    assert torch.equal(_L2Norm.apply(half, 1e-6), module.l2norm(half.float(), eps=1e-6).to(torch.bfloat16))


def test_a_tree_row_normalizes_query_and_key_once_not_once_per_depth(tree, monkeypatch):
    """The kernel keys a compile on the token count it normalizes; a depth holds any number of tokens."""
    _, _, columns = tree
    tiny_family, overrides = FAMILIES["qwen3_5_text"]
    model = tiny_family_model(tiny_family, overrides=overrides).eval()
    family = TREE_FAMILIES["qwen3_5_text"]
    stock, normalizes = getattr(family.module, family.stateful.recurrence), []

    def recording(*args, **kwargs):
        normalizes.append(kwargs["use_qk_l2norm_in_kernel"])
        return stock(*args, **kwargs)

    monkeypatch.setattr(family.module, family.stateful.recurrence, recording)
    backend = model.config._attn_implementation
    _install(model, monkeypatch, "gather")
    monkeypatch.setattr(model.config, "_attn_implementation", backend)
    with torch.no_grad():
        model(input_ids=columns["input_ids"][:5][None], use_cache=False)
        assert normalizes == [True] * len(normalizes) and normalizes, "the stock row asks the scan to normalize"
        normalizes.clear()
        _tree_forward(model, columns, attention_mask=_oracle_masks(model, columns), **_plan(model, columns, "gather"))
    depths = int(columns[DEPTH_COLUMN].max()) + 1
    assert normalizes == [False] * len(normalizes) and len(normalizes) % depths == 0 and normalizes


def test_every_depth_reads_a_view_of_one_permuted_copy_of_each_input(tree):
    """Each input of the scan is permuted into depth order once and every depth reads a view of that copy, so
    the backward is one inverse permutation per input. A gather per depth instead zero-fills, scatters and
    sums a full-size gradient per depth for every input."""
    _, _, columns = tree
    plan = _stateful_plan(columns)
    seen = []

    def scan(query, key, value, *, g, beta, initial_state, output_final_state, cu_seqlens, cu_seqlens_cpu):
        seen.append((query, key, value, g, beta))
        return value * 2, value.new_zeros(len(cu_seqlens_cpu) - 1, 2)

    tokens = len(columns["input_ids"])
    states = [torch.randn(1, tokens, 2, 4, requires_grad=True) for _ in range(3)]
    gates = [torch.randn(1, tokens, 2, requires_grad=True) for _ in range(2)]
    out, _ = _tree_recurrence(scan)(*states, g=gates[0], beta=gates[1], **{TREE_ATTENTION_KEY: plan})
    assert len(seen) == len(plan["levels"]) == 3
    for slot in range(5):
        permuted = seen[0][slot]._base
        assert permuted is not None and all(depth[slot]._base is permuted for depth in seen)
        assert type(permuted.grad_fn).__name__ == "_PermuteTokensBackward"
    assert type(out.grad_fn).__name__ == "_PermuteTokensBackward"


def _per_depth_gathers(scan, plan, query, key, value, g, beta) -> torch.Tensor:
    """The tree recurrence with every depth gathering its own tokens of every input, as a reference: the form
    the depth-order permutation replaces."""
    outputs, state, start = [], None, 0
    levels = plan["levels"]
    for depth, level in enumerate(levels):
        index = plan["depth_order"][start : start + level["cu_seqlens_host"][-1]]
        start += len(index)
        out, state = scan(
            *(states.index_select(1, index) for states in (query, key, value)),
            g=g.index_select(1, index),
            beta=beta.index_select(1, index),
            initial_state=None if state is None else state.index_select(0, level["parent"]),
            output_final_state=depth + 1 < len(levels),
            cu_seqlens=level["cu_seqlens"],
        )
        outputs.append(out)
    return torch.cat(outputs, dim=1).index_select(1, plan["restore"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
# GatedDeltaNet decays per head, KDA per channel.
@pytest.mark.parametrize(("model_type", "per_channel_decay"), [("qwen3_5_text", False), ("glm5_next_text", True)])
def test_the_depth_order_permutation_gives_the_per_depth_gathers_values(tree, model_type, per_channel_decay, dtype):
    """The same scan calls on the same values, through the family's own delta-rule scan (its CPU fallback here):
    the output equals the per-depth gathers' bit for bit, and every input's gradient exactly. Gradients compare
    as values: a gather's zero-filled scatter turns a -0.0 into +0.0, which the permutation keeps."""
    _, _, columns = tree
    family = TREE_FAMILIES[model_type]
    scan = getattr(family.module, family.stateful.recurrence)
    plan = _stateful_plan(columns)
    generator = torch.Generator().manual_seed(2)
    tokens, heads, dim = len(columns["input_ids"]), 2, 8
    gate_shape = (1, tokens, heads, dim) if per_channel_decay else (1, tokens, heads)
    query, key, value = (torch.randn(1, tokens, heads, dim, generator=generator) for _ in range(3))
    log_decay = -torch.randn(gate_shape, generator=generator).abs()
    beta = torch.randn(1, tokens, heads, generator=generator).sigmoid()
    inputs = [states.to(dtype) for states in (query, key, value, log_decay, beta)]
    upstream = torch.randn(1, tokens, heads, dim, generator=generator).to(dtype)

    def run(forward) -> list[torch.Tensor]:
        leaves = [states.clone().requires_grad_(True) for states in inputs]
        out = forward(*leaves)
        out.backward(upstream)
        return [out.detach(), *(leaf.grad for leaf in leaves)]

    permuted = run(lambda q, k, v, g, b: _tree_recurrence(scan)(q, k, v, g=g, beta=b, **{TREE_ATTENTION_KEY: plan})[0])
    gathered = run(lambda q, k, v, g, b: _per_depth_gathers(scan, plan, q, k, v, g, b))
    bits = {torch.float32: torch.int32, torch.bfloat16: torch.int16}[dtype]
    assert torch.equal(permuted[0].view(bits), gathered[0].view(bits))
    assert all(grad.abs().sum() > 0 for grad in gathered[1:]), "every input carries a gradient"
    assert all(torch.equal(mine, reference) for mine, reference in zip(permuted[1:], gathered[1:], strict=True))


def test_the_recurrence_hands_the_scan_each_depths_bounds_from_the_host(tree):
    """fla builds its chunk indices from ``cu_seqlens_cpu`` when given it instead of reading the device bounds
    back, a host sync per depth per layer. On a plan whose tensors sit on the meta device any device read
    raises, so the bounds the scan receives come from the plan's host copy."""
    _, _, columns = tree
    plan = _stateful_plan(columns)
    calls = []

    def scan(query, key, value, *, g, beta, initial_state, output_final_state, cu_seqlens, cu_seqlens_cpu):
        calls.append((cu_seqlens, cu_seqlens_cpu))
        return value, value.new_empty(len(cu_seqlens_cpu) - 1, 2)

    tokens = len(columns["input_ids"])
    states = [torch.empty(1, tokens, 2, 4, device="meta") for _ in range(3)]
    gates = [torch.empty(1, tokens, 2, device="meta") for _ in range(2)]
    meta_plan = _on_device(plan, torch.device("meta"))
    out, _ = _tree_recurrence(scan)(*states, g=gates[0], beta=gates[1], **{TREE_ATTENTION_KEY: meta_plan})
    assert out.shape == states[2].shape and len(calls) == len(plan["levels"]) == 3
    for (device_bounds, host_bounds), level in zip(calls, plan["levels"], strict=True):
        assert device_bounds.is_meta and host_bounds.device.type == "cpu"
        assert host_bounds.tolist() == level["cu_seqlens"].tolist()


def test_a_one_level_plan_hands_every_layer_the_same_bounds_objects():
    """fla caches the chunk indices of its last call, keyed on the identity of the bounds it was given. A plan of
    plain documents has one level, so every layer's scan must receive the same device and host bounds objects to
    reuse them; a fresh host tensor per call would rebuild the indices in every layer."""
    positions = torch.cat([torch.arange(length) for length in (7, 12, 5)])
    table = build_node_table(positions, torch.zeros_like(positions))
    plan = build_tree_attention_plan(table, sliding_window=None, full_form="gather", conv_kernel=4)
    calls = []

    def scan(query, key, value, *, g, beta, initial_state, output_final_state, cu_seqlens, cu_seqlens_cpu):
        calls.append((cu_seqlens, cu_seqlens_cpu))
        return value, None

    tokens = len(positions)
    states = [torch.randn(1, tokens, 2, 4) for _ in range(3)]
    gates = [torch.randn(1, tokens, 2) for _ in range(2)]
    for _layer in range(2):
        _tree_recurrence(scan)(*states, g=gates[0], beta=gates[1], **{TREE_ATTENTION_KEY: plan})
    assert len(plan["levels"]) == 1 and len(calls) == 2
    assert calls[0][0] is calls[1][0] and calls[0][1] is calls[1][1]
    assert calls[0][1].tolist() == plan["levels"][0]["cu_seqlens"].tolist() == [0, 7, 19, 24]


def test_the_key_gather_sums_a_shared_tokens_gradient_in_fp32():
    states = torch.randn(1, 2, 6, 4).to(torch.bfloat16).requires_grad_(True)
    index = torch.tensor([0, 1, 2, 0, 1, 3, 0, 1, 4, 5])
    gathered = _GatherTokens.apply(states, index, 2)
    assert torch.equal(gathered, states.index_select(2, index))
    upstream = torch.randn(gathered.shape).to(torch.bfloat16)
    gathered.backward(upstream)
    expected = torch.zeros(states.shape).index_add_(2, index, upstream.float()).to(torch.bfloat16)
    assert states.grad.dtype == torch.bfloat16 and torch.equal(states.grad, expected)


@pytest.mark.parametrize(
    ("tiny_family", "attn", "form", "match"),
    [
        pytest.param(
            TINY_MOE_FAMILIES["lfm2_moe"], "flash_attention_4", "split", "not available for model_type", id="family"
        ),
        pytest.param(TINY_MOE_FAMILIES["gpt_oss"], "eager", "split", "is verified on", id="backend"),
        pytest.param(TINY_MOE_FAMILIES["gpt_oss"], "flash_attention_3", "gather", "is verified on", id="fa3"),
        pytest.param(
            TINY_MOE_FAMILIES["gpt_oss"], "flash_attention_2", "split", "tree_full_attention: gather", id="lse"
        ),
    ],
)
def test_tree_attention_refuses_what_it_cannot_run(tiny_family, attn, form, match, monkeypatch):
    model = tiny_family_model(tiny_family)
    monkeypatch.setattr(model.config, "_attn_implementation", attn)
    registries = {family.module: family.module.ALL_ATTENTION_FUNCTIONS for family in TREE_FAMILIES.values()}
    with pytest.raises(ValueError, match=match):
        install_tree_attention(model, form)
    assert all(module.ALL_ATTENTION_FUNCTIONS is registry for module, registry in registries.items())


@pytest.mark.parametrize("family", ["gpt_oss", "qwen3_5_moe_text"])
def test_the_split_form_is_refused_on_fa4s_head_dim_256_kernel(family, monkeypatch):
    """That kernel's backward asserts on a log-sum-exp gradient, which only the split form produces."""
    model = tiny_family_model(TINY_MOE_FAMILIES[family], overrides={"head_dim": 256})
    monkeypatch.setattr(model.config, "_attn_implementation", "flash_attention_4")
    with pytest.raises(ValueError, match="head_dim-256 kernel"):
        install_tree_attention(model, "split")


@pytest.mark.parametrize("family", ["gpt_oss", "qwen3_5_moe_text"])
def test_the_gather_form_installs_on_fa4s_head_dim_256_kernel(family, monkeypatch):
    _install(tiny_family_model(TINY_MOE_FAMILIES[family], overrides={"head_dim": 256}), monkeypatch, "gather")


def test_attention_layers_from_another_module_are_refused(monkeypatch):
    """A remote-code class of an admitted model type resolves its interface in its own module: the
    view would sit unused on the stock one and every plan would be dropped without a word."""
    model = tiny_family_model(TINY_MOE_FAMILIES["gpt_oss"])
    monkeypatch.setattr(model.config, "_attn_implementation", "flash_attention_4")
    attention = model.model.layers[0].self_attn
    attention.__class__ = type("RemoteAttention", (type(attention),), {"__module__": "transformers_modules.remote"})
    module = TREE_FAMILIES["gpt_oss"].module
    registry = module.ALL_ATTENTION_FUNCTIONS
    with pytest.raises(ValueError, match="attention layers come from .*transformers_modules.remote"):
        install_tree_attention(model, "split")
    assert module.ALL_ATTENTION_FUNCTIONS is registry


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
