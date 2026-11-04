#!/usr/bin/env python
"""GLM-5 Next's DSA attention on the real mem-efficient kernel, against stock SDPA on the same inputs and an fp32
reference on neither: the tile plan, the kernel's bias and log-sum-exp contract, and the closed-form backward. The CPU
suite holds the backward's form over a stand-in for this kernel."""

import contextlib

import torch
from peft import LoraConfig, get_peft_model
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.glm5_next import modeling_glm5_next

from halo.kernels.masked_softmax import _softmax_backward_kernel, masked_softmax_backward
from halo.model_loading.model_preparation import apply_family_patches
from halo.models.attention import masked_matmul as kernel
from halo.models.attention.masked_matmul import (
    _MaskedMatmulAttention,
    masked_matmul,
    masked_matmul_attention,
    warm_masked_matmul_kernels,
)
from scripts.profiling.benchmark_dsa_attention import _fp32_reference, _pool_mask, _stock
from tests.common.fake_cuda import masked_softmax_backward as masked_softmax_backward_by_passes
from tests.common.harness import gpu_test_main
from tests.common.numerics import cos_sim, fro_rel_err
from tests.common.rank_log import log
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

HEADS, HEAD_DIM = 64, 256
SCALE = HEAD_DIM**-0.5
# Off every multiple the tiles keep, so the padded log-sum-exp, the bias's row stride and the last tile all show.
SEQ = 1531
# bf16 rounding sets both kernels' error; the closed form may not exceed stock's by more than this.
STOCK_ERROR_MARGIN = 1.25
# A tile grid changes only the order the kernel sums in.
SAME_KERNEL_REL_TOL = 1e-4
FP32_OUT_REL_TOL = 1e-5
FP32_GRAD_REL_TOL = 1e-4
BF16_ELEMENT_REL_TOL = 1e-3
FP32_LOSS_REL_TOL = 1e-5
GRAD_COSINE_FLOOR = 0.999
BF16_COSINE_FLOOR = 0.9
BF16_COSINE_MARGIN = 5e-3
DSA_PROJECTIONS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj")
TILE_BUDGETS = {"whole_row": 1 << 40, "default": kernel.SCORES_PER_TILE, "many_tiles": HEADS * SEQ * 40}
DSA_FIRST_LAYER_TYPES = ["indexed_attention", "indexed_attention", "linear_attention", "indexed_attention"]
# Every expert active, as top-k routing flips on bf16 noise, which would measure the router.
TINY_OVERRIDES = {"layer_types": DSA_FIRST_LAYER_TYPES, "index_n_heads": 8, "num_experts_per_tok": 8}


def _inputs(dtype: torch.dtype, batch: int = 1, seed: int = 0) -> tuple[torch.Tensor, ...]:
    """The layouts ``Glm5NextTextAttention`` hands over: strided q and v, contiguous k."""
    generator = torch.Generator("cuda").manual_seed(seed)

    def draw(*shape):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)

    q = draw(batch, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
    k = draw(batch, HEADS, SEQ, HEAD_DIM)
    v = draw(batch, SEQ, HEADS, 2 * HEAD_DIM).transpose(1, 2)[..., HEAD_DIM:]
    return q, k, v, draw(batch, SEQ, HEADS, HEAD_DIM)


def _run(attention, q, k, v, mask, upstream, trained: str = "qkv") -> tuple[torch.Tensor, ...]:
    inputs = [t.detach().requires_grad_(name in trained) for name, t in zip("qkv", (q, k, v), strict=True)]
    out = attention(*inputs, mask, SCALE)
    out.backward(upstream)
    return out.detach(), *(t.grad for t in inputs)


def _errors(measured, reference) -> list[float]:
    return [fro_rel_err(got.float(), truth) for got, truth in zip(measured, reference, strict=True)]


def _matches_stock_and_reference(q, k, v, mask, upstream, tag: str, checks: dict) -> None:
    reference = _fp32_reference(q, k, v, mask, SCALE, upstream)
    stock_run = _run(_stock, q, k, v, mask, upstream)
    stock = _errors(stock_run, reference)
    for budget_name, budget in TILE_BUDGETS.items():
        kernel.SCORES_PER_TILE = budget
        ran = _run(masked_matmul, q, k, v, mask, upstream)
        errors = _errors(ran, reference)
        same = fro_rel_err(ran[0].float(), stock_run[0].float())
        log(f"  [{tag}, {budget_name}] (out, d_q, d_k, d_v) error {errors}, stock {stock}, out vs stock {same:.2e}")
        checks[f"{tag}_{budget_name}_out_is_the_stock_kernels"] = same < SAME_KERNEL_REL_TOL
        checks[f"{tag}_{budget_name}_within_stock_error"] = all(
            error <= STOCK_ERROR_MARGIN * floor for error, floor in zip(errors, stock, strict=True)
        )
    kernel.SCORES_PER_TILE = TILE_BUDGETS["default"]


def _padded_batch_mask() -> torch.Tensor:
    """A second row with a padded tail, whose queries select nothing and whose keys nothing selects."""
    mask = _pool_mask(SEQ, torch.device("cuda")).repeat(2, 1, 1, 1)
    mask[1, 0, SEQ - 211 :] = False
    mask[1, 0, :, SEQ - 211 :] = False
    return mask


def _check_contract(checks: dict) -> None:
    mask = _pool_mask(SEQ, torch.device("cuda"))
    q, k, v, upstream = _inputs(torch.bfloat16)
    _matches_stock_and_reference(q, k, v, mask, upstream, "bf16", checks)

    q2, k2, v2, upstream2 = _inputs(torch.bfloat16, batch=2, seed=1)
    _matches_stock_and_reference(q2, k2, v2, _padded_batch_mask(), upstream2, "bf16_padded_batch", checks)

    q16, k16, v16, upstream16 = _inputs(torch.float16, seed=7)
    _matches_stock_and_reference(q16, k16, v16, mask, upstream16, "fp16", checks)

    q32, k32, v32, upstream32 = _inputs(torch.float32, seed=2)
    reference = _fp32_reference(q32, k32, v32, mask, SCALE, upstream32)
    errors = _errors(_run(masked_matmul, q32, k32, v32, mask, upstream32), reference)
    log(f"  [fp32] (out, d_q, d_k, d_v) error {errors}")
    checks["fp32_out_matches_reference"] = errors[0] < FP32_OUT_REL_TOL
    checks["fp32_gradients_match_reference"] = max(errors[1:]) < FP32_GRAD_REL_TOL

    # A negative control on the fp32 bound, where rounding is far below a lost key's share of the output.
    count = kernel._tile_key_count
    kernel._tile_key_count = lambda allowed: (count(allowed) - 1).clamp_min(1)
    lost = _errors(_run(masked_matmul, q32, k32, v32, mask, upstream32), reference)[0]
    kernel._tile_key_count = count
    log(f"  [negative control] fp32 out error with a lost key {lost:.2e}")
    checks["losing_a_key_breaks_the_output_check"] = lost > 2 * FP32_OUT_REL_TOL

    # The kernel, not the backend flags: stock under the same pin would run the math path.
    expected = masked_matmul(q, k, v, mask, SCALE)
    with sdpa_kernel([SDPBackend.MATH]):
        pinned = masked_matmul(q, k, v, mask, SCALE)
    checks["backend_flags_do_not_move_the_kernel"] = torch.equal(pinned, expected)


def _check_fused_softmax_backward(checks: dict) -> None:
    """The launch against the same arithmetic pass by pass, at unaligned tile shapes, past one block of keys and rows."""
    generator = torch.Generator("cuda").manual_seed(6)
    heads, queries, keys = 7, 37, 2291
    scores = torch.randn(heads, queries, keys, device="cuda", generator=generator)
    upstream_scores = torch.randn(heads, queries, keys, device="cuda", generator=generator)
    lse = torch.randn(heads, queries, device="cuda", generator=generator).contiguous()
    delta = torch.randn(heads, queries, device="cuda", generator=generator)
    stored = torch.rand(queries + 5, keys + 11, device="cuda", generator=generator) < 0.3
    stored[5 + 3] = False  # a query with no key
    allowed = stored[5:, 11:]
    inputs = (scores, allowed, lse, SCALE, upstream_scores, delta)
    for dtype in (torch.float32, torch.bfloat16):
        probs, d_scores = masked_softmax_backward(*inputs, True, dtype)
        truth_probs, truth_d_scores = masked_softmax_backward_by_passes(*inputs, True, dtype)
        probs_err, d_scores_err = fro_rel_err(probs, truth_probs), fro_rel_err(d_scores, truth_d_scores)
        log(f"  [fused softmax backward, {dtype}] probs rel {probs_err:.2e}, d_scores rel {d_scores_err:.2e}")
        tolerance = FP32_OUT_REL_TOL if dtype == torch.float32 else BF16_ELEMENT_REL_TOL
        checks[f"fused_probs_match_{dtype}"] = probs_err < tolerance
        checks[f"fused_d_scores_match_{dtype}"] = d_scores_err < tolerance
    masked = ~allowed
    checks["fused_masked_keys_are_exactly_zero"] = bool(
        (probs[:, masked] == 0).all() and (d_scores[:, masked] == 0).all()
    )
    only_probs = masked_softmax_backward(scores, allowed, lse, SCALE, None, None, True, torch.bfloat16)
    only_d_scores = masked_softmax_backward(*inputs, False, torch.bfloat16)
    checks["fused_probs_alone_are_the_same"] = only_probs[1] is None and torch.equal(only_probs[0], probs)
    checks["fused_cotangent_alone_is_the_same"] = only_d_scores[0] is None and torch.equal(only_d_scores[1], d_scores)


def _kernel_binaries() -> int:
    return len(_softmax_backward_kernel.device_caches[torch.cuda.current_device()][0])


def _check_warm_up_leaves_no_compile(checks: dict) -> None:
    """Every tile shape and view offset the padded batch below reaches, trained and frozen, on the warmed binaries."""
    _softmax_backward_kernel.device_caches.clear()
    model = tiny_family_model(TINY_MOE_FAMILIES["glm5_next"], overrides=TINY_OVERRIDES)
    apply_family_patches(model.config, "sdpa")
    warm_masked_matmul_kernels(model, dtype=torch.bfloat16)
    warmed = _kernel_binaries()
    q, k, v, upstream = _inputs(torch.bfloat16, batch=2, seed=8)
    kernel.SCORES_PER_TILE = HEADS * SEQ * 40
    for trained in ("qkv", "qk", "v"):
        _run(masked_matmul, q, k, v, _padded_batch_mask(), upstream, trained=trained)
    kernel.SCORES_PER_TILE = TILE_BUDGETS["default"]
    log(f"  [warm-up] {warmed} binaries, {_kernel_binaries()} after every tile")
    checks["warm_up_compiles_each_form"] = warmed == 3
    checks["no_tile_compiles_after_the_warm_up"] = _kernel_binaries() == warmed


def _check_all_masked_query(checks: dict) -> None:
    q, k, v, upstream = _inputs(torch.bfloat16, seed=3)
    mask = _pool_mask(SEQ, torch.device("cuda"))
    query = 100
    mask[0, 0, query] = False
    out, d_q, d_k, d_v = _run(masked_matmul, q, k, v, mask, upstream)
    checks["all_masked_query_outputs_zero"] = bool((out[0, query] == 0).all())
    checks["all_masked_query_takes_no_gradient"] = bool((d_q[0, :, query] == 0).all())
    checks["all_masked_query_leaves_everything_finite"] = all(bool(t.isfinite().all()) for t in (out, d_q, d_k, d_v))

    # A whole tile of them, whose softmax has no selected key at all.
    empty = _pool_mask(SEQ, torch.device("cuda"))
    empty[0, 0, :64] = False
    kernel.SCORES_PER_TILE = HEADS * SEQ * 64
    out, d_q, d_k, d_v = _run(masked_matmul, q, k, v, empty, upstream)
    kernel.SCORES_PER_TILE = TILE_BUDGETS["default"]
    checks["all_masked_tile_outputs_zero"] = bool((out[0, :64] == 0).all())
    checks["all_masked_tile_leaves_everything_finite"] = all(bool(t.isfinite().all()) for t in (out, d_q, d_k, d_v))


def _check_frozen_inputs(checks: dict) -> None:
    q, k, v, upstream = _inputs(torch.bfloat16, seed=4)
    mask = _pool_mask(SEQ, torch.device("cuda"))
    _, d_q, d_k, d_v = _run(masked_matmul, q, k, v, mask, upstream)
    for trained, full in (("q", d_q), ("k", d_k), ("v", d_v)):
        _, *grads = _run(masked_matmul, q, k, v, mask, upstream, trained=trained)
        alone = next(grad for grad in grads if grad is not None)
        checks[f"{trained}_alone_keeps_its_gradient"] = sum(grad is not None for grad in grads) == 1 and torch.equal(
            alone, full
        )


def _check_dispatch(checks: dict) -> None:
    q, k, v, _ = _inputs(torch.float32, seed=5)
    mask = _pool_mask(SEQ, torch.device("cuda"))
    module = torch.nn.Module()
    calls = []
    apply = _MaskedMatmulAttention.apply
    _MaskedMatmulAttention.apply = lambda *args: calls.append(1) or apply(*args)
    with torch.no_grad():
        out, _ = masked_matmul_attention(module, q, k, v, mask, scaling=SCALE)
        served = len(calls)
        calls.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cast, _ = masked_matmul_attention(module, q, k, v, mask, scaling=SCALE)
        cast_served = len(calls)
        calls.clear()
        masked_matmul_attention(module, q, k, v, mask, scaling=SCALE, output_attentions=True)
        refused = len(calls)
    _MaskedMatmulAttention.apply = apply
    checks["cuda_calls_reach_the_kernel"] = served == 1 and out.shape == (1, SEQ, HEADS, HEAD_DIM)
    checks["autocast_runs_the_kernel_in_its_dtype"] = cast_served == 1 and cast.dtype == torch.bfloat16
    checks["a_refused_call_stays_on_stock"] = refused == 0


@contextlib.contextmanager
def _stock_registry():
    installed = modeling_glm5_next.ALL_ATTENTION_FUNCTIONS
    modeling_glm5_next.ALL_ATTENTION_FUNCTIONS = ALL_ATTENTION_FUNCTIONS
    try:
        yield
    finally:
        modeling_glm5_next.ALL_ATTENTION_FUNCTIONS = installed


def _loss_and_grads(model, batch):
    model.zero_grad(set_to_none=True)
    loss = model(**batch, use_cache=False).loss
    loss.backward()
    return loss.detach().float(), {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}


def _counted_kernel_run(model, batch):
    calls = []
    apply = _MaskedMatmulAttention.apply
    _MaskedMatmulAttention.apply = lambda *args: calls.append(1) or apply(*args)
    loss, grads = _loss_and_grads(model, batch)
    _MaskedMatmulAttention.apply = apply
    return loss, grads, len(calls)


def _trained(grads: dict[str, torch.Tensor]) -> list[str]:
    """A zero-initialized LoRA B leaves its A's gradient exactly zero on both paths."""
    return [name for name, grad in grads.items() if grad.any()]


def _worst_cosine(grads, reference) -> float:
    return min(cos_sim(grads[name], reference[name], label=name) for name in _trained(reference))


def _model_matches_stock(model, batch, tag: str, checks: dict) -> None:
    served_loss, served, calls = _counted_kernel_run(model, batch)
    with _stock_registry():
        stock_loss, stock = _loss_and_grads(model, batch)
    cosine = _worst_cosine(served, stock)
    log(f"  [{tag}] loss {served_loss.item():.6f} vs {stock_loss.item():.6f}, worst grad cosine {cosine:.6f}")
    checks[f"{tag}_ran_the_kernel"] = calls >= 3
    checks[f"{tag}_loss_matches_stock"] = bool(abs(served_loss - stock_loss) <= FP32_LOSS_REL_TOL * abs(stock_loss))
    checks[f"{tag}_grads_the_same_parameters"] = served.keys() == stock.keys() and bool(_trained(stock))
    checks[f"{tag}_zero_grads_stay_zero"] = not any(served[name].any() for name in stock.keys() - set(_trained(stock)))
    checks[f"{tag}_grads_match_stock"] = cosine > GRAD_COSINE_FLOOR


def _bf16_model_is_as_close_to_fp32_as_stock(model, batch, checks: dict) -> None:
    """Held to the same weights' fp32 run, as two bf16 runs differ by their rounding. Over the DSA projections, since
    the gradient of a hyper-connection's ``base`` mostly cancels, leaving rounding alone on either path."""
    model.to(torch.bfloat16).to(torch.float32)
    _, reference, _ = _counted_kernel_run(model, batch)
    model.to(torch.bfloat16)
    _, served, calls = _counted_kernel_run(model, batch)
    with _stock_registry():
        _, stock = _loss_and_grads(model, batch)
    dsa = {name: grad for name, grad in reference.items() if any(part in name for part in DSA_PROJECTIONS)}
    served_cosine, stock_cosine = _worst_cosine(served, dsa), _worst_cosine(stock, dsa)
    log(f"  [bf16] worst DSA projection grad cosine to fp32 {served_cosine:.6f}, stock's {stock_cosine:.6f}")
    checks["bf16_ran_the_kernel"] = calls >= 3
    checks["bf16_stock_is_close_to_fp32"] = stock_cosine > BF16_COSINE_FLOOR
    checks["bf16_grads_as_close_to_fp32_as_stock"] = served_cosine >= stock_cosine - BF16_COSINE_MARGIN
    model.to(torch.float32)


def _check_model(checks: dict) -> None:
    torch.manual_seed(0)
    ids = torch.randint(1, 1024, (2, 96), device="cuda")
    attention = torch.ones_like(ids)
    attention[1, -9:] = 0
    batch = {"input_ids": ids, "attention_mask": attention, "labels": ids.masked_fill(attention == 0, -100)}

    model = tiny_family_model(TINY_MOE_FAMILIES["glm5_next"], overrides=TINY_OVERRIDES)
    apply_family_patches(model.config, "sdpa")
    checks["the_patch_installs_on_this_family"] = (
        modeling_glm5_next.ALL_ATTENTION_FUNCTIONS["sdpa"] is masked_matmul_attention
    )
    model = model.cuda().train()
    _model_matches_stock(model, batch, "fp32", checks)
    _bf16_model_is_as_close_to_fp32_as_stock(model, batch, checks)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.enable_input_require_grads()
    _model_matches_stock(model, batch, "fp32_reentrant", checks)
    adapted = get_peft_model(
        model, LoraConfig(r=4, target_modules=["kv_b_proj", "q_b_proj"], init_lora_weights="gaussian")
    )
    adapted.cuda()
    _model_matches_stock(adapted, batch, "fp32_reentrant_lora", checks)


@gpu_test_main(exact_world_size=1, prefix="masked_matmul_attention")
def run(ctx):
    checks: dict[str, bool] = {}
    _check_fused_softmax_backward(checks)
    _check_warm_up_leaves_no_compile(checks)
    _check_contract(checks)
    _check_all_masked_query(checks)
    _check_frozen_inputs(checks)
    _check_dispatch(checks)
    _check_model(checks)
    return {"checks": checks}


if __name__ == "__main__":
    run()
