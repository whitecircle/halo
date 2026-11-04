#!/usr/bin/env python
"""What GLM-5 Next's DSA layers run: the closed-form backward against SDPA, over a stand-in for the CUDA-only forward
kernel, the dispatch and fall-through, and the model-level path. ``tests/gpu/kernels/test_masked_matmul_attention.py``
holds the real kernel; ``scripts/profiling/benchmark_dsa_attention.py`` measures it."""

import contextlib

import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.autograd.function import FunctionCtx
from torch.distributed.tensor import DTensor, Replicate
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.glm4_moe_lite import modeling_glm4_moe_lite
from transformers.models.glm5_next import modeling_glm5_next

from halo.distributed.pipeline_parallel.families import glm5_next_stream_forward
from halo.kernels.fp32_matmul import GEMM_ALIGNMENT
from halo.model_loading.model_preparation import apply_family_patches
from halo.model_loading.parallel import whole_model
from halo.models.attention import masked_matmul as kernel
from halo.models.attention.masked_matmul import (
    _MaskedMatmulAttention,
    _plan_tiles,
    install_glm5_next_masked_matmul,
    masked_matmul,
    masked_matmul_attention,
    warm_masked_matmul_kernels,
)
from halo.models.attention.tree import TreeAttentionRegistry
from tests.common.distributed import fake_process_group_mesh
from tests.common.fake_cuda import CudaBmm, CudaLike, efficient_attention, masked_softmax_backward
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

HEADS, SEQ, DIM = 4, 96, 16
SCALE = DIM**-0.5
POOL = 4
# Whole score matrix, about a third of it, and a tile of a few queries.
BUDGETS = (HEADS * SEQ * SEQ, HEADS * SEQ * 32, HEADS * SEQ * 3)
OUT_TOLERANCE, GRAD_TOLERANCE = 1e-6, 1e-5
EXACT_TOLERANCE = 1e-12
# bf16 inputs, products and gradients against fp32 math on the same rounded inputs.
BF16_TOLERANCE = 2e-2
DSA_FIRST_LAYER_TYPES = ["indexed_attention", "indexed_attention", "linear_attention", "indexed_attention"]


@pytest.fixture(autouse=True)
def kernel_stand_ins(monkeypatch):
    monkeypatch.setattr(torch, "_scaled_dot_product_efficient_attention", efficient_attention)
    monkeypatch.setattr(kernel, "masked_softmax_backward", masked_softmax_backward)


def _dense_mask(batch, generator):
    return torch.ones(batch, 1, SEQ, SEQ, dtype=torch.bool)


def _causal_mask(batch, generator):
    return torch.ones(SEQ, SEQ, dtype=torch.bool).tril()[None, None].repeat(batch, 1, 1, 1)


def _pool_mask(batch, generator):
    """Each query's keys are whole pools ending at or before it, plus its own incomplete pool, as the indexer's are."""
    rows = []
    ends = torch.arange(POOL - 1, SEQ, POOL)
    positions = torch.arange(SEQ)
    for _ in range(batch):
        rows.append(positions[:, None] // POOL == positions[None] // POOL)
        rows[-1] = rows[-1] & (positions[None] <= positions[:, None])
        chosen = (torch.rand(SEQ, ends.numel(), generator=generator) < 0.25) & (ends <= positions[:, None])
        rows[-1] = rows[-1] | chosen.repeat_interleave(POOL, dim=1)
    return torch.stack(rows)[:, None]


def _padded_mask(batch, generator):
    """The later rows pad their tail: a padded query selects nothing, and nothing selects a padded key."""
    mask = _pool_mask(batch, generator)
    for row in range(1, batch):
        mask[row, 0, SEQ - 17 * row :] = False
        mask[row, 0, :, SEQ - 17 * row :] = False
    return mask


def _all_masked_query_mask(batch, generator):
    mask = _pool_mask(batch, generator)
    mask[0, 0, SEQ // 2] = False
    return mask


MASKS = {
    "dense": _dense_mask,
    "causal": _causal_mask,
    "pools": _pool_mask,
    "padded": _padded_mask,
    "all_masked_query": _all_masked_query_mask,
}


def _problem(batch=1, mask="pools", dtype=torch.float32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(batch, HEADS, SEQ, DIM, generator=generator, dtype=torch.float64).to(dtype) for _ in "qkv")
    upstream = torch.randn(batch, SEQ, HEADS, DIM, generator=generator, dtype=torch.float64).to(dtype)
    return q, k, v, MASKS[mask](batch, generator), upstream


def _sdpa(q, k, v, mask, upstream):
    q, k, v = (t.detach().clone().requires_grad_() for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=SCALE).transpose(1, 2)
    out.backward(upstream)
    return out.detach(), q.grad, k.grad, v.grad


def _kernel(q, k, v, mask, upstream):
    q, k, v = (t.detach().clone().requires_grad_() for t in (q, k, v))
    out = masked_matmul(q, k, v, mask, SCALE)
    out.backward(upstream)
    return out.detach(), q.grad, k.grad, v.grad


def _relative(got, truth):
    return ((got.double() - truth.double()).norm() / truth.double().norm()).item()


def _errors(got, truth):
    return dict(zip(("out", "q", "k", "v"), (_relative(a, b) for a, b in zip(got, truth, strict=True)), strict=True))


def _assert_matches(got, truth, *, out=OUT_TOLERANCE, grad=GRAD_TOLERANCE):
    errors = _errors(got, truth)
    assert errors["out"] < out, errors
    assert max(errors["q"], errors["k"], errors["v"]) < grad, errors


@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("mask", sorted(MASKS))
def test_output_and_gradients_match_sdpa_in_fp32(monkeypatch, mask, batch, budget):
    monkeypatch.setattr(kernel, "SCORES_PER_TILE", budget)
    problem = _problem(batch, mask)
    _assert_matches(_kernel(*problem), _sdpa(*problem))


@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("mask", sorted(MASKS))
def test_the_closed_form_is_exact_in_fp64(monkeypatch, mask, budget):
    """In fp32 a tolerance is rounding's; in fp64 anything left is the form itself."""
    monkeypatch.setattr(kernel, "SCORES_PER_TILE", budget)
    problem = _problem(2, mask, dtype=torch.float64)
    _assert_matches(_kernel(*problem), _sdpa(*problem), out=EXACT_TOLERANCE, grad=EXACT_TOLERANCE)


def test_an_all_masked_query_outputs_zero_and_takes_no_gradient():
    q, k, v, mask, upstream = _problem(1, "all_masked_query")
    out, d_q, d_k, d_v = _kernel(q, k, v, mask, upstream)
    query = SEQ // 2
    assert torch.equal(out[0, query], torch.zeros_like(out[0, query]))
    assert torch.equal(d_q[0, :, query], torch.zeros_like(d_q[0, :, query]))
    assert all(t.isfinite().all() for t in (out, d_q, d_k, d_v))


def test_a_tile_of_nothing_but_masked_queries_runs(monkeypatch):
    monkeypatch.setattr(kernel, "SCORES_PER_TILE", HEADS * SEQ * 4)
    q, k, v, mask, upstream = _problem(1)
    mask[0, 0, :8] = False
    assert _plan_tiles(mask, HEADS)[0].keys == slice(GEMM_ALIGNMENT)
    _assert_matches(_kernel(q, k, v, mask, upstream), _sdpa(q, k, v, mask, upstream))


def test_bf16_stays_within_rounding_of_fp32_math():
    q, k, v, mask, upstream = _problem(2, dtype=torch.bfloat16)
    wide = tuple(t.float() for t in (q, k, v)) + (mask, upstream.float())
    _assert_matches(_kernel(q, k, v, mask, upstream), _sdpa(*wide), out=BF16_TOLERANCE, grad=BF16_TOLERANCE)


def _without_delta(scores, allowed, lse, scale, upstream_scores, delta, writes_probs, dtype):
    return masked_softmax_backward(scores, allowed, lse, scale, upstream_scores, 0 * delta, writes_probs, dtype)


def test_the_gradient_check_bites_on_a_dropped_delta(monkeypatch):
    """Negative control: without the softmax's normalizer term the output is untouched and q and k are far off."""
    problem = _problem(2)
    monkeypatch.setattr(kernel, "masked_softmax_backward", _without_delta)
    errors = _errors(_kernel(*problem), _sdpa(*problem))
    assert errors["out"] < OUT_TOLERANCE, errors
    assert errors["v"] < GRAD_TOLERANCE, errors
    assert min(errors["q"], errors["k"]) > 100 * GRAD_TOLERANCE, errors


def test_the_check_bites_on_a_crop_that_loses_the_last_key(monkeypatch):
    """Negative control on the key set: the crop's bound is all that keeps a selected key in the tile."""
    count = kernel._tile_key_count
    monkeypatch.setattr(kernel, "_tile_key_count", lambda allowed: (count(allowed) - 1).clamp_min(1))
    problem = _problem(1, "causal")
    errors = _errors(_kernel(*problem), _sdpa(*problem))
    assert errors["out"] > 100 * OUT_TOLERANCE, errors


def test_tiles_keep_the_budget_and_crop_to_the_last_selected_key(monkeypatch):
    monkeypatch.setattr(kernel, "SCORES_PER_TILE", HEADS * SEQ * 10)
    _, _, _, mask, _ = _problem(2, "padded")
    tiles = _plan_tiles(mask, HEADS)
    for row in range(2):
        row_tiles = [tile for tile in tiles if tile.row == row]
        assert [q for tile in row_tiles for q in range(tile.queries.start, tile.queries.stop)] == list(range(SEQ))
        for tile in row_tiles:
            allowed = mask[row, 0, tile.queries]
            assert not allowed[:, tile.keys.stop :].any(), "the crop lost a selected key"
            assert HEADS * (tile.queries.stop - tile.queries.start) * tile.keys.stop <= kernel.SCORES_PER_TILE
            last = allowed.any(dim=0).nonzero().max().item() + 1 if allowed.any() else 1
            assert 0 <= tile.keys.stop - last < GEMM_ALIGNMENT, "at most the alignment's masked keys"
            assert tile.keys.stop % GEMM_ALIGNMENT == 0 or tile.keys.stop == SEQ
    assert max(tile.keys.stop for tile in tiles if tile.row == 1) <= SEQ - 16, "a padded row scores past its padding"


@pytest.mark.parametrize("seq", [96, 91])
def test_query_tiles_keep_the_gemm_alignment_but_for_the_row_end(monkeypatch, seq):
    monkeypatch.setattr(kernel, "SCORES_PER_TILE", HEADS * seq * 21)
    mask = torch.ones(1, 1, seq, seq, dtype=torch.bool).tril()
    tiles = _plan_tiles(mask, HEADS)
    assert all((tile.queries.stop - tile.queries.start) == 16 for tile in tiles[:-1])
    assert tiles[-1].queries.stop == seq
    assert all(tile.keys.stop % GEMM_ALIGNMENT == 0 for tile in tiles[:-1])
    assert tiles[-1].keys.stop == seq


def test_the_crop_reads_the_mask_so_a_non_causal_key_stays(monkeypatch):
    monkeypatch.setattr(kernel, "SCORES_PER_TILE", HEADS * SEQ * 16)
    _, _, _, mask, _ = _problem(1, "causal")
    assert _plan_tiles(mask, HEADS)[0].keys.stop == 16, "premise: a causal first tile crops to its own end"
    mask[0, 0, 0, SEQ - 1] = True
    assert _plan_tiles(mask, HEADS)[0].keys.stop == SEQ


GRAD_COUNTS = {"qkv": 5, "kv": 4, "qv": 4, "qk": 4, "q": 3, "k": 3, "v": 2}


@pytest.mark.parametrize("trainable", sorted(GRAD_COUNTS))
def test_a_frozen_input_costs_no_product(monkeypatch, trainable):
    """The scores, ``g @ v^T`` for q and k alike, and one each per trained input."""
    products = []
    bmm_fp32 = kernel.bmm_fp32
    monkeypatch.setattr(kernel, "bmm_fp32", lambda a, b, out=None: products.append(1) or bmm_fp32(a, b, out))
    q, k, v, mask, upstream = _problem(1)
    inputs = dict(zip("qkv", (q, k, v), strict=True))
    for name, tensor in inputs.items():
        tensor.requires_grad_(name in trainable)
    masked_matmul(q, k, v, mask, SCALE).backward(upstream)
    assert len(products) == GRAD_COUNTS[trainable]
    for name, tensor in inputs.items():
        assert (tensor.grad is not None) == (name in trainable)


def test_frozen_gradients_match_the_trained_ones():
    q, k, v, mask, upstream = _problem(1)
    _, d_q, d_k, d_v = _sdpa(q, k, v, mask, upstream)
    for name, truth, tensor in (("q", d_q, q), ("k", d_k, k), ("v", d_v, v)):
        tensor = tensor.clone().requires_grad_()
        inputs = {"q": q, "k": k, "v": v, name: tensor}
        masked_matmul(inputs["q"], inputs["k"], inputs["v"], mask, SCALE).backward(upstream)
        assert _relative(tensor.grad, truth) < GRAD_TOLERANCE


@pytest.mark.parametrize("under_no_grad", [True, False], ids=["checkpoint_first_pass", "nothing_requires_grad"])
def test_nothing_is_saved_without_a_gradient(monkeypatch, under_no_grad):
    """A checkpointed layer's first pass: the projections' outputs come out of ``no_grad`` without a gradient."""
    saved = []
    save = FunctionCtx.save_for_backward
    monkeypatch.setattr(FunctionCtx, "save_for_backward", lambda ctx, *t: saved.append(1) or save(ctx, *t))
    log_sumexp_requests = []

    def recording_kernel(*args, **kwargs):
        log_sumexp_requests.append(args[4])
        return efficient_attention(*args, **kwargs)

    monkeypatch.setattr(torch, "_scaled_dot_product_efficient_attention", recording_kernel)
    q, k, v, mask, _ = _problem(1)
    with torch.set_grad_enabled(not under_no_grad):
        trained = [t.clone().requires_grad_() if under_no_grad else t for t in (q, k, v)]
        out = masked_matmul(*(t * 1 for t in trained), mask, SCALE)
    assert not saved
    assert out.grad_fn is None
    assert log_sumexp_requests == [False]
    truth = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=SCALE).transpose(1, 2)
    assert _relative(out, truth) < OUT_TOLERANCE


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_every_product_keeps_half_cuda_operands(monkeypatch, dtype):
    """Upcast, the GEMMs would leave the tensor cores for the pinned ``highest`` fp32 path."""
    recorder = CudaBmm(torch.bmm)
    monkeypatch.setattr(torch, "bmm", recorder)
    monkeypatch.setattr(torch, "baddbmm", recorder.baddbmm)
    q, k, v, mask, upstream = _problem(1, dtype=dtype)
    q, k, v = (t.requires_grad_().as_subclass(CudaLike) for t in (q, k, v))
    out = masked_matmul(q, k, v, mask, SCALE)
    assert out.dtype == dtype
    out.backward(upstream.as_subclass(CudaLike))
    assert recorder.calls == [(dtype, dtype, torch.float32)] * 5


@pytest.fixture
def forced_kernel(monkeypatch):
    """The kernel's own device gate is what keeps CPU on SDPA; every other check stays live."""
    monkeypatch.setattr(kernel, "_KERNEL_DEVICES", ("cuda", "cpu"))


@pytest.fixture
def stock_registry(monkeypatch):
    """A process-wide write the test owns, as another test's load may have installed it."""
    monkeypatch.setattr(modeling_glm5_next, "ALL_ATTENTION_FUNCTIONS", ALL_ATTENTION_FUNCTIONS)


def _module(groups=1):
    module = torch.nn.Module()
    module.is_causal, module.num_key_value_groups = True, groups
    return module


def _spied_stock(monkeypatch):
    calls = []

    def stock(module, query, key, value, attention_mask, **kwargs):
        calls.append((query, key, value, attention_mask, kwargs))
        return torch.zeros(()), None

    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._global_mapping, "sdpa", stock)
    return calls


def _call(q=None, k=None, v=None, mask=None, **kwargs):
    base_q, base_k, base_v, base_mask, _ = _problem(2)
    q, k, v = (
        base if given is None else given for base, given in zip((base_q, base_k, base_v), (q, k, v), strict=True)
    )
    return q, k, v, base_mask if mask is None else mask, kwargs


def _dtensor(tensor):
    with fake_process_group_mesh(0, 1) as mesh:
        return DTensor.from_local(tensor, mesh, [Replicate()], run_check=False)


_REFUSED_CALLS = {
    "dropout": lambda: _call(dropout=0.1),
    "output_attentions": lambda: _call(output_attentions=True),
    "no_mask": lambda: _call()[:3] + (None, {}),
    "float_mask": lambda: _call(mask=torch.zeros(2, 1, SEQ, SEQ)),
    "mask_per_head": lambda: _call(mask=torch.ones(2, HEADS, SEQ, SEQ, dtype=torch.bool)),
    "kv_cache": lambda: _call(k=torch.randn(2, HEADS, SEQ + 4, DIM), v=torch.randn(2, HEADS, SEQ + 4, DIM)),
    "heads_differ": lambda: _call(k=torch.randn(2, HEADS // 2, SEQ, DIM), v=torch.randn(2, HEADS // 2, SEQ, DIM)),
    "value_width_differs": lambda: _call(v=torch.randn(2, HEADS, SEQ, DIM // 2)),
    "fp64": lambda: tuple(t.double() for t in _call()[:3]) + (_call()[3], {}),
    "head_width_off_the_kernels_alignment": lambda: _call(
        q=torch.randn(2, HEADS, SEQ, DIM + 2),
        k=torch.randn(2, HEADS, SEQ, DIM + 2),
        v=torch.randn(2, HEADS, SEQ, DIM + 2),
    ),
    "dtypes_differ": lambda: _call(k=torch.randn(2, HEADS, SEQ, DIM, dtype=torch.bfloat16)),
    "position_bias": lambda: _call(position_bias=torch.zeros(SEQ, SEQ)),
    "cache": lambda: _call(cache=object()),
    "dtensor": lambda: _call(q=_dtensor(torch.randn(2, HEADS, SEQ, DIM))),
}


@pytest.mark.parametrize("refused", sorted(_REFUSED_CALLS))
def test_a_call_the_kernel_does_not_serve_reaches_stock_unchanged(monkeypatch, forced_kernel, refused):
    calls = _spied_stock(monkeypatch)
    q, k, v, mask, kwargs = _REFUSED_CALLS[refused]()
    masked_matmul_attention(_module(), q, k, v, mask, scaling=SCALE, **kwargs)
    assert len(calls) == 1
    assert calls[0][0] is q and calls[0][1] is k and calls[0][2] is v and calls[0][3] is mask


def test_fp64_under_autocast_keeps_stock_as_autocast_leaves_it(monkeypatch, forced_kernel):
    calls = _spied_stock(monkeypatch)
    q, k, v, mask, _ = _problem(1, dtype=torch.float64)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        masked_matmul_attention(_module(), q, k, v, mask, scaling=SCALE)
    assert len(calls) == 1
    assert calls[0][0] is q


def test_cpu_stays_on_stock_without_the_forced_device(monkeypatch):
    calls = _spied_stock(monkeypatch)
    q, k, v, mask, kwargs = _call()
    masked_matmul_attention(_module(), q, k, v, mask, scaling=SCALE, **kwargs)
    assert len(calls) == 1


def test_grouped_key_value_heads_reach_stock(monkeypatch, forced_kernel):
    calls = _spied_stock(monkeypatch)
    q, k, v, mask, kwargs = _call()
    masked_matmul_attention(_module(groups=2), q, k, v, mask, scaling=SCALE, **kwargs)
    assert len(calls) == 1


def test_a_served_call_returns_the_stock_interface_shape(monkeypatch, forced_kernel):
    calls = _spied_stock(monkeypatch)
    q, k, v, mask, _ = _problem(2)
    out, weights = masked_matmul_attention(_module(), q, k, v, mask, scaling=SCALE, is_causal=True)
    assert not calls
    assert weights is None
    assert out.shape == (2, SEQ, HEADS, DIM)


def test_scaling_defaults_to_the_head_width(forced_kernel):
    q, k, v, mask, _ = _problem(1)
    out, _ = masked_matmul_attention(_module(), q, k, v, mask)
    truth = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2)
    assert _relative(out, truth) < OUT_TOLERANCE


@pytest.mark.parametrize("backward_under_autocast", [False, True])
def test_autocast_inputs_take_its_dtype_and_stay_there(forced_kernel, backward_under_autocast):
    """As autocast casts SDPA's: fp32 parameters under bf16 autocast still run the half path, and the gradients come
    back to each input's own dtype through the cast."""
    q, k, v, mask, upstream = _problem(1)
    q, k, v = (t.requires_grad_() for t in (q, k, v))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out, _ = masked_matmul_attention(_module(), q, k, v, mask, scaling=SCALE)
        assert out.dtype == torch.bfloat16
        if backward_under_autocast:
            out.backward(upstream.to(torch.bfloat16))
    if not backward_under_autocast:
        out.backward(upstream.to(torch.bfloat16))
    wide = tuple(t.detach().to(torch.bfloat16).float() for t in (q, k, v))
    truth = _sdpa(*wide, mask, upstream.to(torch.bfloat16).float())
    assert _relative(out.detach(), truth[0]) < BF16_TOLERANCE
    for tensor, reference in zip((q, k, v), truth[1:], strict=True):
        assert tensor.grad.dtype == torch.float32
        assert _relative(tensor.grad, reference) < BF16_TOLERANCE


def _tiny_glm5(**overrides):
    torch.manual_seed(0)
    overrides = {"layer_types": DSA_FIRST_LAYER_TYPES, "index_n_heads": 8, **overrides}
    return tiny_family_model(TINY_MOE_FAMILIES["glm5_next"], overrides=overrides)


def _batch(padded=False):
    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(1, 1024, (2, 48), generator=generator)
    attention = torch.ones_like(ids)
    if padded:
        attention[1, -9:] = 0
    return {"input_ids": ids, "attention_mask": attention, "labels": ids.masked_fill(attention == 0, -100)}


def _loss_and_grads(model, batch, *, kernel_on, monkeypatch):
    """One pass on stock SDPA or the kernel, counting which attention the DSA layers reached."""
    served = []
    apply = _MaskedMatmulAttention.apply
    monkeypatch.setattr(_MaskedMatmulAttention, "apply", lambda *a: served.append(1) or apply(*a))
    monkeypatch.setattr(kernel, "_KERNEL_DEVICES", ("cuda", "cpu") if kernel_on else ("cuda",))
    model.zero_grad()
    loss = model(**batch, use_cache=False).loss
    loss.backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    return loss.detach(), grads, len(served)


def _assert_model_paths_agree(model, batch, monkeypatch, *, min_passes):
    assert model.config.get_text_config()._attn_implementation == "sdpa"
    install_glm5_next_masked_matmul()
    stock_loss, stock_grads, stock_served = _loss_and_grads(model, batch, kernel_on=False, monkeypatch=monkeypatch)
    loss, grads, served = _loss_and_grads(model, batch, kernel_on=True, monkeypatch=monkeypatch)
    assert stock_served == 0
    assert served >= min_passes
    assert abs(loss - stock_loss) <= 1e-5 * abs(stock_loss)
    assert grads.keys() == stock_grads.keys()
    assert grads, "no parameter took a gradient"
    scale = max(g.abs().max().item() for g in stock_grads.values())
    worst = max((grads[name] - stock_grads[name]).abs().max().item() for name in grads)
    assert worst <= 1e-4 * scale, worst / scale
    return grads


@pytest.mark.parametrize("padded", [False, True], ids=["full", "padded"])
def test_model_kernel_path_matches_stock_on_every_parameter(monkeypatch, stock_registry, padded):
    model = _tiny_glm5().train()
    grads = _assert_model_paths_agree(model, _batch(padded), monkeypatch, min_passes=3)
    assert any("q_b_proj" in name for name in grads) and any("kv_b_proj" in name for name in grads)


def test_shared_indexer_layers_run_the_kernel(monkeypatch, stock_registry):
    model = _tiny_glm5(indexer_types=["full", "shared", "full", "full"]).train()
    _assert_model_paths_agree(model, _batch(), monkeypatch, min_passes=3)


def test_reentrant_checkpoint_recompute_matches_stock(monkeypatch, stock_registry):
    """A layer runs no_grad, then recomputes with grad and takes the backward; each pass is a kernel call."""
    model = _tiny_glm5().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.enable_input_require_grads()
    _assert_model_paths_agree(model, _batch(), monkeypatch, min_passes=6)


@pytest.mark.parametrize("targets", [["kv_b_proj"], ["q_b_proj"], ["kv_b_proj", "q_b_proj"]], ids="+".join)
def test_lora_on_the_attention_projections_matches_stock(monkeypatch, stock_registry, targets):
    torch.manual_seed(1)
    model = get_peft_model(_tiny_glm5(), LoraConfig(r=4, target_modules=targets, init_lora_weights="gaussian"))
    model.train()
    grads = _assert_model_paths_agree(model, _batch(), monkeypatch, min_passes=3)
    assert all("lora_" in name for name in grads)
    assert any(target in name for name in grads for target in targets)


def test_forward_only_matches_stock(monkeypatch, stock_registry):
    model = _tiny_glm5().eval()
    install_glm5_next_masked_matmul()
    batch = _batch()
    calls = []
    apply = _MaskedMatmulAttention.apply
    monkeypatch.setattr(_MaskedMatmulAttention, "apply", lambda *args: calls.append(1) or apply(*args))
    with torch.no_grad():
        stock = model(**batch, use_cache=False).logits
        assert not calls
        monkeypatch.setattr(kernel, "_KERNEL_DEVICES", ("cuda", "cpu"))
        served = model(**batch, use_cache=False).logits
    assert len(calls) == 3, "one per DSA layer"
    assert _relative(served, stock) < 1e-5


def test_the_pipeline_stream_forward_dispatches_to_the_kernel(monkeypatch, stock_registry, forced_kernel):
    model = _tiny_glm5().eval()
    install_glm5_next_masked_matmul()
    calls = []
    apply = _MaskedMatmulAttention.apply
    monkeypatch.setattr(_MaskedMatmulAttention, "apply", lambda *a: calls.append(1) or apply(*a))
    text = model.model.language_model
    embeds = text.embed_tokens(_batch()["input_ids"])
    stream = embeds[:, :, None].repeat(1, 1, model.config.get_text_config().hc_mult, 1)
    with torch.no_grad():
        glm5_next_stream_forward(text, inputs_embeds=stream, use_cache=False)
    assert len(calls) == 3, "one per DSA layer of the stage"


def test_the_vision_tower_stays_on_stock(monkeypatch, stock_registry, forced_kernel):
    """Its attention reads the same module name, with no mask."""
    calls = _spied_stock(monkeypatch)
    install_glm5_next_masked_matmul()
    interface = modeling_glm5_next.ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
    q, k, v, _, _ = _problem(1)
    interface(_module(), q, k, v, None, is_causal=False)
    assert len(calls) == 1
    assert calls[0][3] is None


def _glm5_config():
    return _tiny_glm5().config


def _recorded_warm_up(monkeypatch, model) -> list[tuple[bool, bool]]:
    launches = []

    def recording(scores, allowed, lse, scale, upstream_scores, delta, writes_probs, dtype):
        launches.append((writes_probs, upstream_scores is not None))
        return masked_softmax_backward(scores, allowed, lse, scale, upstream_scores, delta, writes_probs, dtype)

    monkeypatch.setattr(kernel, "masked_softmax_backward", recording)
    warm_masked_matmul_kernels(model, dtype=torch.bfloat16, device=torch.device("cpu"))
    return launches


def test_the_warm_up_launches_each_form_a_frozen_input_reaches(monkeypatch, stock_registry):
    model = _tiny_glm5()
    install_glm5_next_masked_matmul()
    assert sorted(_recorded_warm_up(monkeypatch, model)) == [(False, True), (True, False), (True, True)]


def test_the_warm_up_compiles_nothing_off_the_kernel(monkeypatch, stock_registry):
    model = _tiny_glm5()
    assert not _recorded_warm_up(monkeypatch, model), "the module is on the library's registry"
    install_glm5_next_masked_matmul()
    other = tiny_family_model(TINY_MOE_FAMILIES["glm4_moe_lite"])
    assert not _recorded_warm_up(monkeypatch, other), "no DSA layer on this model"


def test_every_load_warms_the_kernel_behind_the_fence(monkeypatch):
    warmed = []
    monkeypatch.setattr(whole_model, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(whole_model, "barrier_on_exit", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(whole_model, "warm_fa4_kernels", lambda *a, **k: None)
    monkeypatch.setattr(whole_model, "warm_flex_sliding_kernels", lambda *a, **k: None)
    monkeypatch.setattr(
        whole_model, "warm_masked_matmul_kernels", lambda model, *, dtype: warmed.append((model, dtype))
    )
    whole_model.warm_attention_kernels("model", dtype=torch.bfloat16)
    assert warmed == [("model", torch.bfloat16)]


def test_the_patch_applies_to_glm5_next_under_sdpa_only(monkeypatch, stock_registry):
    config = _glm5_config()
    for model_config in (config, config.get_text_config()):
        monkeypatch.setattr(modeling_glm5_next, "ALL_ATTENTION_FUNCTIONS", ALL_ATTENTION_FUNCTIONS)
        assert apply_family_patches(model_config, "sdpa") == "sdpa", "the config keeps naming sdpa"
        assert modeling_glm5_next.ALL_ATTENTION_FUNCTIONS["sdpa"] is masked_matmul_attention
    monkeypatch.setattr(modeling_glm5_next, "ALL_ATTENTION_FUNCTIONS", ALL_ATTENTION_FUNCTIONS)
    for attn_implementation in ("eager", None, "flash_attention_2"):
        apply_family_patches(config, attn_implementation)
        assert modeling_glm5_next.ALL_ATTENTION_FUNCTIONS is ALL_ATTENTION_FUNCTIONS, attn_implementation


def test_the_patch_leaves_other_families_alone(monkeypatch, stock_registry):
    apply_family_patches(tiny_family_model(TINY_MOE_FAMILIES["glm4_moe_lite"]).config, "sdpa")
    assert modeling_glm5_next.ALL_ATTENTION_FUNCTIONS is ALL_ATTENTION_FUNCTIONS


def test_the_local_mapping_does_not_leak_into_the_library_registry(monkeypatch, stock_registry):
    stock = ALL_ATTENTION_FUNCTIONS["sdpa"]
    install_glm5_next_masked_matmul()
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is stock
    assert modeling_glm4_moe_lite.ALL_ATTENTION_FUNCTIONS is ALL_ATTENTION_FUNCTIONS
    assert modeling_glm4_moe_lite.ALL_ATTENTION_FUNCTIONS["sdpa"] is stock
    assert masked_matmul_attention not in ALL_ATTENTION_FUNCTIONS._global_mapping.values()
    # A later library-wide registration still reaches the local instance.
    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._global_mapping, "spy_attention", stock)
    assert modeling_glm5_next.ALL_ATTENTION_FUNCTIONS["spy_attention"] is stock


@pytest.mark.parametrize("value", ["0", "false", "off"])
def test_the_knob_off_restores_stock(monkeypatch, stock_registry, value):
    monkeypatch.setenv("HALO_DSA_MATMUL", value)
    apply_family_patches(_glm5_config(), "sdpa")
    assert modeling_glm5_next.ALL_ATTENTION_FUNCTIONS is ALL_ATTENTION_FUNCTIONS


def test_the_knob_defaults_on(monkeypatch, stock_registry):
    monkeypatch.delenv("HALO_DSA_MATMUL", raising=False)
    apply_family_patches(_glm5_config(), "sdpa")
    assert modeling_glm5_next.ALL_ATTENTION_FUNCTIONS["sdpa"] is masked_matmul_attention


def test_the_install_is_idempotent_and_keeps_tree_attention_above_it(monkeypatch, stock_registry):
    install_glm5_next_masked_matmul()
    installed = modeling_glm5_next.ALL_ATTENTION_FUNCTIONS
    install_glm5_next_masked_matmul()
    assert modeling_glm5_next.ALL_ATTENTION_FUNCTIONS is installed

    view = TreeAttentionRegistry(installed)
    monkeypatch.setattr(modeling_glm5_next, "ALL_ATTENTION_FUNCTIONS", view)
    install_glm5_next_masked_matmul()
    assert modeling_glm5_next.ALL_ATTENTION_FUNCTIONS is view, "a later load must not drop the tree registry"
    assert view.get_interface("sdpa", None) is masked_matmul_attention


def test_a_tree_view_over_the_library_registry_stays_and_runs_stock(monkeypatch, stock_registry):
    view = TreeAttentionRegistry(ALL_ATTENTION_FUNCTIONS)
    monkeypatch.setattr(modeling_glm5_next, "ALL_ATTENTION_FUNCTIONS", view)
    install_glm5_next_masked_matmul()
    assert modeling_glm5_next.ALL_ATTENTION_FUNCTIONS is view
    assert view.get_interface("sdpa", None) is ALL_ATTENTION_FUNCTIONS["sdpa"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
