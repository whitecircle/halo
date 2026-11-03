#!/usr/bin/env python
"""Tree attention on a family whose softmax layers select their keys per query: GLM-5 Next mixes KDA linear attention
with DeepSeek sparse attention. ``tests/gpu/kernels/test_tree_sparse_attention.py`` pins the CUDA kernels."""

from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer
from torch.overrides import TorchFunctionMode

from halo.configs.script_arguments import ModelArguments
from halo.data.tree_layout import DEPTH_COLUMN, POSITION_COLUMN, build_trie, serialize_tree
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.model_loading.parallel.peft_setup import build_peft_config
from halo.models import segment_markers
from halo.models.attention import tree_sparse
from halo.models.attention.selection import uses_fa4_dedicated_head_dim
from halo.models.attention.tree import (
    TREE_FAMILIES,
    TreeAttentionRegistry,
    install_tree_attention,
    reject_unsupported_tree_attention,
    resolve_tree_plan_options,
)
from halo.models.attention.tree_plan import TREE_ATTENTION_KEY, build_node_table, build_tree_attention_plan
from halo.models.attention.tree_sparse import _SparseLatentAttention, tree_key_selection
from tests.common.packed_flash_varlen import HOST_READS
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model
from tests.common.trees import checkpointing_gaps, example_rows, randomize_indexer_pools, tree_columns

VOCAB = 1024
# Eight indexer heads: with fewer, a random model scores many pools exactly zero (every head's ReLU off), and the
# stock top-k breaks those ties in an unspecified order. Two pools per selection, so every path past twelve tokens
# selects.
SPARSE_FAMILIES = {
    "glm5_next_text": (
        TINY_MOE_FAMILIES["glm5_next"],
        {"index_n_heads": 8, "index_topk": 8, "index_kpool_always_select_tail": True},
    ),
}
# lora_target_modules spelling -> (refused before the load, PEFT wraps every sparse layer's kv_b_proj). A regex
# resolves only against the loaded module tree, which the first tree batch checks.
TARGET_SPELLINGS = {
    "listed": (["q_a_proj", "kv_b_proj"], True, True),
    "listed alone": (["kv_b_proj"], True, True),
    "layer-relative path": (["o_proj", "self_attn.kv_b_proj"], True, True),
    "all-linear": ("all-linear", True, True),
    "all-linear as a list": (["ALL-LINEAR"], True, True),
    "elsewhere": (["q_a_proj", "o_proj"], False, False),
    "regex": (r".*\.kv_b_proj", False, True),
}


def test_every_sparse_family_is_swept():
    assert set(SPARSE_FAMILIES) == {name for name, family in TREE_FAMILIES.items() if family.sparse}


def _paths() -> tuple[list[list[int]], list[list[int]]]:
    """Three levels under an 11-token trunk, so a pool straddles every branch point."""
    generator = torch.Generator().manual_seed(0)

    def tokens(n: int) -> list[int]:
        return torch.randint(0, VOCAB, (n,), generator=generator).tolist()

    trunk, middle = tokens(11), tokens(6)
    paths = [trunk + tokens(5), trunk + middle + tokens(3), trunk + middle + tokens(9), trunk + tokens(14)]
    answers = (2, 2, 3, 4)
    labels = [[LABEL_IGNORE_INDEX] * (len(path) - n) + path[-n:] for path, n in zip(paths, answers, strict=True)]
    return paths, labels


@pytest.fixture(scope="module")
def tree():
    paths, labels = _paths()
    columns = tree_columns(paths, [[label != LABEL_IGNORE_INDEX for label in path_labels] for path_labels in labels])
    assert int(columns[DEPTH_COLUMN].max()) == 2
    return paths, labels, columns


@pytest.fixture(scope="module", params=sorted(SPARSE_FAMILIES))
def model(request):
    torch.manual_seed(0)
    family, overrides = SPARSE_FAMILIES[request.param]
    model = tiny_family_model(family, overrides=overrides).eval()
    assert model.config.get_text_config().model_type == request.param
    randomize_indexer_pools(model)
    return model


def _install(model, monkeypatch) -> None:
    family = TREE_FAMILIES[model.config.get_text_config().model_type]
    for name in ("ALL_ATTENTION_FUNCTIONS", family.stateful.conv, family.stateful.recurrence):
        monkeypatch.setattr(family.module, name, getattr(family.module, name))
    monkeypatch.setattr(family.sparse, "forward", family.sparse.forward)
    # The segment-aware kernels are CUDA builds; on CPU the toolkit's own routes stand in for them.
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", ())
    install_tree_attention(model, "gather")


def _plan(model, columns) -> dict:
    table = build_node_table(columns[POSITION_COLUMN], columns[DEPTH_COLUMN])
    return build_tree_attention_plan(table, full_form="gather", **resolve_tree_plan_options(model.config))


def _forward(model, columns, plan: dict | None) -> torch.Tensor:
    kwargs = {} if plan is None else {TREE_ATTENTION_KEY: plan}
    return model(
        input_ids=columns["input_ids"][None], position_ids=columns[POSITION_COLUMN][None], use_cache=False, **kwargs
    ).logits[0]


def _loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits[:-1], labels[1:], ignore_index=LABEL_IGNORE_INDEX, reduction="sum")


def _alone(model, columns) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(row indices, logits) of every example of the tree run on its own by the stock forward."""
    with torch.no_grad():
        return [
            (rows, model(input_ids=columns["input_ids"][rows][None], use_cache=False).logits[0])
            for rows in example_rows(build_node_table(columns[POSITION_COLUMN], columns[DEPTH_COLUMN]))
        ]


@pytest.fixture(scope="module")
def alone(model, tree) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return _alone(model, tree[2])


def _worst_gap(tree_logits: torch.Tensor, alone) -> float:
    return max((tree_logits[rows] - logits).abs().max().item() for rows, logits in alone)


def _assert_tree_is_every_example_alone(model, paths, labels, columns, alone, plan) -> None:
    model.zero_grad()
    tree_logits = _forward(model, columns, plan)
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


def test_a_tree_row_reproduces_every_example_alone(model, tree, alone, monkeypatch):
    paths, labels, columns = tree
    _install(model, monkeypatch)
    plan = _plan(model, columns)
    assert "full" not in plan and "sparse" in plan
    _assert_tree_is_every_example_alone(model, paths, labels, columns, alone, plan)

    # Without the plan the row is one sequence everywhere; with the sparse links reading it as one path (KDA layers
    # still segmented) their selection and attention cross branches. Each must move the logits.
    assert _worst_gap(_forward(model, columns, None), alone) > 1e-2
    tokens = len(columns["input_ids"])
    one_path = {
        "predecessor": torch.arange(tokens) - 1,
        "subtree_end": torch.full((tokens,), tokens),
        "position": torch.arange(tokens),
        "tree_start": torch.zeros(tokens, dtype=torch.long),
    }
    with torch.no_grad():
        assert _worst_gap(_forward(model, columns, {**plan, "sparse": one_path}), alone) > 1e-2


@pytest.mark.parametrize("use_reentrant", [True, False], ids=["reentrant", "non_reentrant"])
@pytest.mark.parametrize("model_type", sorted(SPARSE_FAMILIES))
def test_a_tree_row_trains_identically_under_gradient_checkpointing(model_type, use_reentrant, tree, monkeypatch):
    """A checkpointed layer runs its forward again in the backward, so the sparse layers' forward and the KDA
    convolution and scan must read the plan as the first pass did."""
    _, _, columns = tree
    family, overrides = SPARSE_FAMILIES[model_type]
    torch.manual_seed(0)
    model = tiny_family_model(family, overrides=overrides)
    randomize_indexer_pools(model)
    _install(model, monkeypatch)
    plan = _plan(model, columns)

    def step() -> torch.Tensor:
        logits = _forward(model, columns, plan)
        _loss(logits, columns["labels"]).backward()
        return logits.detach()

    logit_gap, gradient_gap = checkpointing_gaps(model, step, use_reentrant=use_reentrant)
    assert logit_gap <= 1e-6 and gradient_gap <= 1e-6


def test_a_layer_reusing_the_previous_selection_reads_the_tree_selection(tree, monkeypatch):
    """GLM-5.3-Flash ships no ``shared`` DSA layer, but the stock forward supports them."""
    paths, labels, columns = tree
    family, overrides = SPARSE_FAMILIES["glm5_next_text"]
    torch.manual_seed(0)
    layout = {
        "layer_types": ["linear_attention", "indexed_attention", "indexed_attention", "linear_attention"],
        "indexer_types": ["full", "full", "shared", "full"],
    }
    model = tiny_family_model(family, overrides={**overrides, **layout}).eval()
    randomize_indexer_pools(model)
    attentions = [layer.self_attn for layer in model.model.language_model.layers]
    assert attentions[1].next_skip_topk and attentions[2].indexer is None
    alone = _alone(model, columns)
    _install(model, monkeypatch)
    _assert_tree_is_every_example_alone(model, paths, labels, columns, alone, _plan(model, columns))


def _random_paths(generator: torch.Generator, max_depth: int = 3) -> list[list[int]]:
    """Root-to-leaf paths of a tree of random node lengths, nodes shorter than a pool included."""

    def grow(prefix: list[int], depth: int) -> list[list[int]]:
        length = int(torch.randint(1, 15, (), generator=generator))
        tokens = prefix + torch.randint(0, VOCAB, (length,), generator=generator).tolist()
        if depth == max_depth:
            return [tokens]
        children = int(torch.randint(2, 4, (), generator=generator))
        return [path for _ in range(children) for path in grow(tokens, depth + 1)]

    return grow([], 0)


def test_the_selection_is_the_stock_indexers_on_every_path(model, monkeypatch):
    generator = torch.Generator().manual_seed(3)
    attention = next(
        layer.self_attn
        for layer in model.model.language_model.layers
        if getattr(layer.self_attn, "indexer", None) is not None
    )
    indexer = attention.indexer
    row_trees = []
    for _ in range(3):  # three trees side by side in one row, as a packed batch holds them
        paths = _random_paths(generator)
        trained = [[True] * len(path) for path in paths]
        row_trees.append(serialize_tree(build_trie(paths, [len(path) - 1 for path in paths], 1), paths, trained))
    position = torch.tensor([p for tree in row_trees for p in tree[POSITION_COLUMN]])
    depth = torch.tensor([d for tree in row_trees for d in tree[DEPTH_COLUMN]])
    table = build_node_table(position, depth)
    assert int(depth.max()) >= 2 and (table.lengths < indexer.index_kpool).any()
    links = build_tree_attention_plan(table, sliding_window=None, full_form="gather", sparse_attention=True)["sparse"]
    hidden = torch.randn(1, len(position), attention.hidden_size, generator=generator)

    def key_set(row: torch.Tensor) -> set[int]:
        return {int(key) for key in row if key >= 0}

    pools = int((position % indexer.index_kpool == indexer.index_kpool - 1).sum())
    with torch.no_grad():
        q_resid = attention.q_a_layernorm(attention.q_a_proj(hidden))
        one_call = tree_key_selection(indexer, hidden, q_resid, links)
        for chunk in (1, 5, 37):
            monkeypatch.setattr(tree_sparse, "INDEXER_SCORES_PER_CALL", indexer.n_heads * pools * chunk)
            starts = links["tree_start"][torch.arange(0, len(position), chunk)]
            ends = links["tree_start"][
                (torch.arange(0, len(position), chunk) + chunk - 1).clamp(max=len(position) - 1)
            ]
            assert chunk == 1 or (starts != ends).any(), "no chunk spans a tree boundary"
            chunked = tree_key_selection(indexer, hidden, q_resid, links)
            # A chunk scores its own window of pools, so the slots fall differently; the sets are the same.
            assert all(key_set(a) == key_set(b) for a, b in zip(chunked[0], one_call[0], strict=True))

        selecting = 0
        for rows in example_rows(table):
            solo = indexer(hidden[:, rows], q_resid[:, rows], torch.ones(1, len(rows), dtype=torch.bool), None)[0]
            for at, row in enumerate(rows):
                assert key_set(one_call[0, row]) == {int(rows[key]) for key in key_set(solo[at])}
            selecting += max(len(rows) // indexer.index_kpool - indexer.index_topk // indexer.index_kpool, 0)
        assert selecting, "no example is long enough to select among its pools"


class _HostReads(TorchFunctionMode):
    """Counts the calls that read a device tensor back to the host, each a sync on a GPU: the
    :data:`HOST_READS` methods and a boolean-mask index, whose output size is data-dependent."""

    def __init__(self):
        super().__init__()
        self.reads: list[str] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", None)
        if name in HOST_READS:
            self.reads.append(name)
        elif name == "__getitem__" and any(
            isinstance(index, torch.Tensor) and index.dtype == torch.bool
            for index in (args[1] if isinstance(args[1], tuple) else (args[1],))
        ):
            self.reads.append("boolean mask index")
        return func(*args, **(kwargs or {}))


def _selection_problem(model):
    """``(indexer, hidden, q_resid, links)`` of one DSA layer over a row of three random trees."""
    generator = torch.Generator().manual_seed(5)
    attention = next(
        layer.self_attn
        for layer in model.model.language_model.layers
        if getattr(layer.self_attn, "indexer", None) is not None
    )
    row_trees = []
    for _ in range(3):
        paths = _random_paths(generator)
        trained = [[True] * len(path) for path in paths]
        row_trees.append(serialize_tree(build_trie(paths, [len(path) - 1 for path in paths], 1), paths, trained))
    position = torch.tensor([p for tree in row_trees for p in tree[POSITION_COLUMN]])
    depth = torch.tensor([d for tree in row_trees for d in tree[DEPTH_COLUMN]])
    table = build_node_table(position, depth)
    links = build_tree_attention_plan(table, sliding_window=None, full_form="gather", sparse_attention=True)["sparse"]
    hidden = torch.randn(1, len(position), attention.hidden_size, generator=generator)
    with torch.no_grad():
        q_resid = attention.q_a_layernorm(attention.q_a_proj(hidden))
    return attention.indexer, hidden, q_resid, links


@pytest.mark.parametrize("chunk", [None, 3], ids=["one-call", "chunked"])
def test_a_plans_selection_reads_the_host_once_for_every_layer(model, monkeypatch, chunk):
    indexer, hidden, q_resid, links = _selection_problem(model)
    pools = int((links["position"] % indexer.index_kpool == indexer.index_kpool - 1).sum())
    if chunk is not None:
        monkeypatch.setattr(tree_sparse, "INDEXER_SCORES_PER_CALL", indexer.n_heads * pools * chunk)
    fresh = dict(links)
    with _HostReads() as first:
        selected = tree_key_selection(indexer, hidden, q_resid, links)
    with _HostReads() as later:
        for _ in range(3):  # the plan's other DSA layers, then the recompute of each
            assert torch.equal(tree_key_selection(indexer, hidden, q_resid, links), selected)
    assert sorted(first.reads) == ["nonzero", "tolist"], first.reads
    assert later.reads == [], later.reads
    assert torch.equal(tree_key_selection(indexer, hidden, q_resid, fresh), selected), "the kept plan drifted"


def _recorded_forward(query, latent, keys, scale) -> SimpleNamespace:
    """A function mode misses the autograd engine's backward, so the test calls it directly on this context."""
    ctx = SimpleNamespace()
    ctx.save_for_backward = lambda *tensors: setattr(ctx, "saved_tensors", tensors)
    with torch.no_grad():
        _SparseLatentAttention.forward(ctx, query, latent, keys, scale)
    return ctx


def test_the_latent_attention_backward_reads_nothing_back_to_the_host(monkeypatch):
    """The backward adds every slot, a pad's an exact zero, instead of compacting the selected ones by a count read
    back per chunk."""
    generator = torch.Generator().manual_seed(4)
    tokens, heads, rank, width = 12, 2, 6, 5
    query = torch.randn(tokens, heads, rank, generator=generator, requires_grad=True)
    latent = torch.randn(tokens, rank, generator=generator, requires_grad=True)
    keys = torch.randint(0, tokens, (tokens, width), generator=generator)
    keys[torch.rand(tokens, width, generator=generator) < 0.4] = -1
    keys[0] = -1
    monkeypatch.setattr(tree_sparse, "ATTENTION_KEYS_PER_CALL", 4 * width)
    out = _SparseLatentAttention.apply(query, latent, keys, 0.5)
    upstream = torch.randn(out.shape, generator=generator)
    expected = torch.autograd.grad(out, (query, latent), upstream)

    ctx = _recorded_forward(query.detach(), latent.detach(), keys, 0.5)
    with _HostReads() as reads:
        d_query, d_latent, _, _ = _SparseLatentAttention.backward(ctx, upstream)
    assert reads.reads == [], reads.reads
    assert torch.equal(d_query, expected[0]) and torch.equal(d_latent, expected[1])


def test_the_latent_attention_is_the_masked_softmax_with_exact_gradients():
    generator = torch.Generator().manual_seed(1)
    tokens, heads, rank, width = 9, 3, 5, 4
    query = torch.randn(tokens, heads, rank, dtype=torch.double, generator=generator, requires_grad=True)
    latent = torch.randn(tokens, rank, dtype=torch.double, generator=generator, requires_grad=True)
    keys = torch.full((tokens, width), -1, dtype=torch.long)
    for t in range(1, tokens):  # token 0 selects nothing: SDPA's safe softmax gives it a zero output
        chosen = torch.randperm(t + 1, generator=generator)[: min(width, t + 1)]
        keys[t, : len(chosen)] = chosen
    scale = 0.7

    keep = torch.zeros(tokens, tokens, dtype=torch.bool)
    for t in range(tokens):
        keep[t, keys[t][keys[t] >= 0]] = True
    dense = torch.einsum("thr,sr->ths", query, latent) * scale
    expected = torch.einsum("ths,sr->thr", dense.masked_fill(~keep[:, None], float("-inf")).softmax(-1), latent)
    expected[0] = 0
    out = _SparseLatentAttention.apply(query, latent, keys, scale)
    assert torch.allclose(out, expected, rtol=0, atol=1e-12) and torch.equal(out[0], torch.zeros_like(out[0]))
    assert torch.autograd.gradcheck(lambda q, c: _SparseLatentAttention.apply(q, c, keys, scale), (query, latent))


def _sparse_layers(model) -> list[torch.nn.Module]:
    family = TREE_FAMILIES["glm5_next_text"]
    return [module for module in model.modules() if isinstance(module, family.sparse)]


@pytest.mark.parametrize("model_type", sorted(SPARSE_FAMILIES))
def test_the_latent_fold_sees_one_dtype_under_autocast(model_type, tree, monkeypatch):
    """fp32 parameters under bf16 autocast, as fp32 masters train without FSDP. The query leaves its matmul in bf16 and
    the latent its norm in fp32; the latent attention takes both in bf16 and keeps its fp32 scores out of autocast."""
    _, _, columns = tree
    family, overrides = SPARSE_FAMILIES[model_type]
    torch.manual_seed(0)
    model = tiny_family_model(family, overrides=overrides)
    assert {param.dtype for param in model.parameters()} == {torch.float32}
    randomize_indexer_pools(model)
    _install(model, monkeypatch)
    seen = []
    forward = _SparseLatentAttention.forward

    def recording(ctx, query, latent, keys, scale):
        seen.append((query.dtype, latent.dtype, torch.is_autocast_enabled(query.device.type)))
        return forward(ctx, query, latent, keys, scale)

    monkeypatch.setattr(_SparseLatentAttention, "forward", staticmethod(recording))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = _forward(model, columns, _plan(model, columns))
    _loss(logits.float(), columns["labels"]).backward()
    assert seen and set(seen) == {(torch.bfloat16, torch.bfloat16, False)}
    grads = [layer.kv_b_proj.weight.grad for layer in _sparse_layers(model)]
    assert all(grad is not None and grad.isfinite().all() and grad.abs().sum() > 0 for grad in grads)


@pytest.mark.parametrize("spelling", sorted(TARGET_SPELLINGS))
def test_adapter_targets_on_kv_b_proj_are_refused_before_the_load(spelling, monkeypatch):
    """Refused wherever the run's PEFT config wraps kv_b_proj under a spelling readable without the model, and never
    where it wraps none."""
    targets, refused, wrapped = TARGET_SPELLINGS[spelling]
    family, overrides = SPARSE_FAMILIES["glm5_next_text"]
    model = tiny_family_model(family, overrides=overrides)
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", ())
    # The SFT script checks the targets as the parsed ModelArguments holds them.
    model_config = ModelArguments(use_peft=True, lora_r=4, lora_target_modules=targets)
    if refused:
        with pytest.raises(ValueError, match="Leave kv_b_proj out of lora_target_modules"):
            reject_unsupported_tree_attention(
                model.config, "sdpa", "gather", lora_target_modules=model_config.lora_target_modules
            )
    else:
        reject_unsupported_tree_attention(
            model.config, "sdpa", "gather", lora_target_modules=model_config.lora_target_modules
        )

    peft_config = build_peft_config(model, model_config)
    layers = _sparse_layers(get_peft_model(model, peft_config))
    assert layers and [isinstance(layer.kv_b_proj, BaseTunerLayer) for layer in layers] == [wrapped] * len(layers)


def test_attention_dropout_is_refused_before_the_load_and_only_for_the_folding_family(monkeypatch):
    family, overrides = SPARSE_FAMILIES["glm5_next_text"]
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", ())
    dropout = tiny_family_model(family, overrides={**overrides, "attention_dropout": 0.1}).config
    with pytest.raises(ValueError, match="takes no dropout, but attention_dropout=0.1"):
        reject_unsupported_tree_attention(dropout, "sdpa", "gather")
    reject_unsupported_tree_attention(tiny_family_model(family, overrides=overrides).config, "sdpa", "gather")
    # A family whose attention runs at the flash seam reads no kv_b_proj, so the targets and quantization don't apply.
    flash_family = tiny_family_model(TINY_MOE_FAMILIES["gpt_oss"]).config
    reject_unsupported_tree_attention(
        flash_family,
        "flash_attention_4",
        "split",
        lora_target_modules=["kv_b_proj"],
        quantization_config={"quant_method": "bitsandbytes"},
    )


def test_what_the_fold_would_skip_is_refused_at_the_first_tree_batch(tree, monkeypatch):
    """The SFT script installs tree attention on the bare model and the trainer wraps its adapters after, so the loaded
    layers are checked again: an adapter under a regex spelling, and a dropout set after the install."""
    _, _, columns = tree
    family, overrides = SPARSE_FAMILIES["glm5_next_text"]
    torch.manual_seed(0)
    model = tiny_family_model(family, overrides=overrides)
    _install(model, monkeypatch)
    plan = _plan(model, columns)
    adapted = get_peft_model(model, LoraConfig(r=4, target_modules=TARGET_SPELLINGS["regex"][0]))
    with torch.no_grad(), pytest.raises(ValueError, match="Leave kv_b_proj out of lora_target_modules"):
        _forward(adapted, columns, plan)

    torch.manual_seed(0)
    model = tiny_family_model(family, overrides=overrides).train()
    _install(model, monkeypatch)
    for layer in _sparse_layers(model):
        layer.attention_dropout = 0.1
    with torch.no_grad(), pytest.raises(ValueError, match="takes no dropout"):
        _forward(model, columns, plan)


def test_a_sparse_plan_is_refused_at_a_flash_interface(tree):
    """A sparse family's plan carries no flash parts, so a flash layer reading it would run the row as one sequence."""
    _, _, columns = tree
    table = build_node_table(columns[POSITION_COLUMN], columns[DEPTH_COLUMN])
    plan = build_tree_attention_plan(table, sliding_window=None, full_form="gather", sparse_attention=True)
    calls = []
    flash = TreeAttentionRegistry(None).wrap_flash_interface(lambda *args, **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="a sparse-attention tree plan reached the flash interface"):
        flash(torch.nn.Identity(), None, None, None, None, **{TREE_ATTENTION_KEY: plan})
    flash(torch.nn.Identity(), None, None, None, None)
    assert calls == [{}]


def test_the_backend_and_full_form_checks_do_not_apply_to_a_sparse_family(monkeypatch):
    """Its sparse layers attend in their own forward and its KDA layers on their own kernels; the segment-aware kernels
    still apply."""
    config = tiny_family_model(SPARSE_FAMILIES["glm5_next_text"][0]).config
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", ())
    reject_unsupported_tree_attention(config, "sdpa", "split")
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", (("fla>=0.2.2", lambda: False),))
    with pytest.raises(ValueError, match=r"fla>=0\.2\.2 unavailable"):
        reject_unsupported_tree_attention(config, "sdpa", "gather")


def test_the_split_forms_head_dim_256_refusal_does_not_apply_to_a_sparse_family(monkeypatch):
    """GLM-5 Next's DSA heads are 256 wide at release, but the sparse layers never reach FA4's head_dim-256 kernel."""
    config = tiny_family_model(SPARSE_FAMILIES["glm5_next_text"][0], overrides={"qk_nope_head_dim": 256}).config
    assert uses_fa4_dedicated_head_dim(config)
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", ())
    reject_unsupported_tree_attention(config, "flash_attention_4", "split")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
