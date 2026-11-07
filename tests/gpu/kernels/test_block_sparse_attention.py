#!/usr/bin/env python
"""Flash-Next's block-sparse QSA kernel against an fp64 gather reference and the tiled path, forward and backward, on
packed, padded and degenerate rows, through LoRA, under checkpointing, and for its compile count."""

import contextlib
import copy
import functools
import os
from unittest import mock

import pytest
import torch
import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen4_exp_modeling
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

from halo.kernels import block_sparse_attention as kernel
from halo.model_loading.finalization import finalize_loaded_model
from halo.models.families.qwen4_exp import sparse_attention
from halo.models.families.qwen4_exp.forwards import apply_qwen4_exp_patches
from halo.models.families.qwen4_exp.sparse_attention import (
    qsa_attention,
    qsa_attention_forward,
    select_qsa_keys,
    tiled_qsa_attention,
)
from halo.models.gradient_checkpointing import active_recompute_scope, scoped_checkpoint_func
from tests.common.harness import gpu_test_main, record_check
from tests.common.qwen4_exp import QSA_LAYER_IDX, load_tiny_cuda_model, training_batch
from tests.common.qwen4_exp_qsa import (
    ROWS,
    SHORT_DOCUMENT_ROWS,
    attention_operands,
    build_inputs,
    error_ratio,
    gather_reference,
    layout_config,
)

SEED = 11
# The kernel's error against fp64 over the tiled path's own, on the output and on every gradient.
KERNEL_TO_TILED_ERROR = 4.0
# Against the tiled path directly, as a share of the tensor's largest value.
DTYPE_TOLERANCE = {torch.bfloat16: 3e-2, torch.float16: 6e-3}
# Heads, K/V heads, head width: the tiny config's, a larger group at a wider head, a middle group, and the tile floor.
LAYOUTS = {"group-2": (4, 2, 64), "group-12": (12, 1, 128), "group-3": (6, 2, 32), "group-1": (2, 2, 16)}
REGIMES = {"sparse": {"indexer_compress_ratio": 4, "indexer_budget": 8}, "dense": {"indexer_budget": 4096}}
ALL_ROWS = {**ROWS, "short-documents": SHORT_DOCUMENT_ROWS}
# Triton specializes an integer argument on divisibility by 16, so 512 and 4096 are in beside the odd lengths.
PACK_LENGTHS = (1, 37, 512, 513, 2049, 4096, 4100)
NATIVE_LAYOUT = LAYOUTS["group-2"]
# A whole bf16 model on the kernel against the same model on the tiled path: loss, and the attention projections' gradients.
MODEL_LOSS_TOLERANCE = 2e-2
MODEL_GRAD_TOLERANCE = 1e-1


@contextlib.contextmanager
def qsa_kernel(mode: str):
    previous = os.environ.get("HALO_QSA_KERNEL")
    os.environ["HALO_QSA_KERNEL"] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HALO_QSA_KERNEL")
        else:
            os.environ["HALO_QSA_KERNEL"] = previous


def _case(layout, rows, dtype, regime=None):
    """The config, a selection computed on the GPU, and Q/K/V/upstream there in ``dtype``."""
    config = layout_config(*layout, **(regime or {}))
    torch.manual_seed(SEED)
    attention = qwen4_exp_modeling.Qwen4ExpTextAttention(config, 3).cuda()
    inputs = build_inputs(config, rows)
    cos, sin = inputs.position_embeddings
    selection = select_qsa_keys(
        attention.indexer,
        inputs.hidden.cuda(),
        (cos.cuda(), sin.cuda()),
        None if inputs.seq_idx is None else inputs.seq_idx.cuda(),
        None if inputs.padding_mask is None else inputs.padding_mask.cuda(),
    )
    batch, length = inputs.hidden.shape[:2]
    query, key, value, upstream = (t.cuda().to(dtype) for t in attention_operands(config, batch, length, SEED))
    return config, selection, (query, key, value), upstream


def _gradients(attend, operands, upstream):
    """The output as ``[batch, heads, length, dim]`` and the gradients to Q/K/V."""
    leaves = [t.detach().clone().requires_grad_() for t in operands]
    output = attend(*leaves)
    return output, torch.autograd.grad(output, leaves, upstream.transpose(1, 2).to(output.dtype))


def _kernel(selection, scale):
    return lambda q, k, v: kernel.block_sparse_attention(
        q, k, v, selection.blocks, selection.spans, selection.compress_ratio, scale
    )


def _tiled(selection, scale):
    return lambda q, k, v: tiled_qsa_attention(q, k, v, selection, scale).transpose(1, 2)


def _reference(blocks, spans, ratio, scale):
    return lambda q, k, v: gather_reference(q.double(), k.double(), v.double(), blocks, spans, ratio, scale)


def _worst_error(actual, expected):
    """The largest ``error_ratio`` over the output and the gradients."""
    return max(error_ratio(a, e) for a, e in zip((actual[0], *actual[1]), (expected[0], *expected[1]), strict=True))


def _assert_close(actual, expected, tolerance):
    worst = _worst_error(actual, expected)
    assert worst <= tolerance, f"worst error {worst:.3e} over {tolerance:.3e}"


def _compare_to_tiled(layout, rows, dtype, regime=None):
    config, selection, operands, upstream = _case(layout, rows, dtype, regime)
    scale = config.head_dim**-0.5
    got = _gradients(_kernel(selection, scale), operands, upstream)
    tiled = _gradients(_tiled(selection, scale), operands, upstream)
    _assert_close(got, tiled, DTYPE_TOLERANCE[dtype])
    return config, selection, operands, upstream, got


def test_matches_fp64_no_worse_than_the_tiled_path(layout, dtype):
    config, selection, operands, upstream = _case(layout, ROWS["packed"], dtype, REGIMES["sparse"])
    scale = config.head_dim**-0.5
    truth = _gradients(
        _reference(selection.blocks, selection.spans, selection.compress_ratio, scale),
        [t.double() for t in operands],
        upstream.double(),
    )
    got = _gradients(_kernel(selection, scale), operands, upstream)
    tiled = _gradients(_tiled(selection, scale), operands, upstream)
    for name, kernel_value, tiled_value, true_value in zip(
        ("output", "dq", "dk", "dv"), (got[0], *got[1]), (tiled[0], *tiled[1]), (truth[0], *truth[1]), strict=True
    ):
        kernel_error, tiled_error = error_ratio(kernel_value, true_value), error_ratio(tiled_value, true_value)
        assert kernel_error <= KERNEL_TO_TILED_ERROR * tiled_error, f"{name}: {kernel_error:.3e} vs {tiled_error:.3e}"


def test_matches_the_tiled_path_in_both_regimes(regime, rows, dtype):
    _compare_to_tiled(NATIVE_LAYOUT, rows, dtype, regime)


def test_left_padded_queries_read_zeros_and_carry_no_gradient(dtype):
    config, selection, operands, upstream, (output, (dq, _, _)) = _compare_to_tiled(
        NATIVE_LAYOUT, ROWS["left-padded"], dtype, REGIMES["sparse"]
    )
    assert (selection.spans[0, :80, 1] == 0).all()
    assert (output[0, :, :80] == 0).all()
    assert (dq[0, :, :80] == 0).all()
    tiled_output, (tiled_dq, _, _) = _gradients(_tiled(selection, config.head_dim**-0.5), operands, upstream)
    assert (tiled_output[0, :, :80] == 0).all()
    assert (tiled_dq[0, :, :80] == 0).all()


def test_right_padded_queries_match_the_tiled_path(dtype):
    """A right-padded query still sees its document, so it reads a non-zero output, as the tiled path gives it."""
    _, selection, operands, upstream, (output, _) = _compare_to_tiled(
        NATIVE_LAYOUT, ROWS["right-padded"], dtype, REGIMES["sparse"]
    )
    assert output[0, :, 250:].abs().max() > 0


def _handmade_selection(visible: list[int], block_topk: int, ratio: int):
    length = len(visible)
    blocks = torch.full((1, length, block_topk), -1, dtype=torch.int16, device="cuda")
    spans = torch.tensor([[[0, v] for v in visible]], dtype=torch.int32, device="cuda")
    return blocks, spans, ratio


def test_empty_selections_attend_the_tail_only_and_nothing_goes_nan(dtype):
    """Every block slot unused with fewer than ``ratio`` keys visible, and a query that sees none."""
    ratio = 4
    config = layout_config(*NATIVE_LAYOUT)
    blocks, spans, _ = _handmade_selection([0, 1, 2, 3, 3, 0], block_topk=2, ratio=ratio)
    query, key, value, upstream = (t.cuda().to(dtype) for t in attention_operands(config, 1, blocks.shape[1], SEED))
    scale = config.head_dim**-0.5
    got = _gradients(
        lambda q, k, v: kernel.block_sparse_attention(q, k, v, blocks, spans, ratio, scale),
        (query, key, value),
        upstream,
    )
    truth = _gradients(
        _reference(blocks, spans, ratio, scale), [t.double() for t in (query, key, value)], upstream.double()
    )
    assert all(torch.isfinite(t).all() for t in (got[0], *got[1]))
    assert (got[0][0, :, 0] == 0).all() and (got[0][0, :, 5] == 0).all()
    _assert_close(got, truth, DTYPE_TOLERANCE[dtype])


def test_negative_controls_fail_the_comparison():
    """One wrong block id, and a visible count off by one, each break the comparison the other tests rely on."""
    dtype = torch.bfloat16
    config, selection, operands, upstream = _case(NATIVE_LAYOUT, ROWS["packed"], dtype, REGIMES["sparse"])
    scale = config.head_dim**-0.5
    tiled = _gradients(_tiled(selection, scale), operands, upstream)
    wrong_blocks = selection.blocks.clone()
    query = int((selection.blocks[0, :, 0] >= 0).nonzero()[-1])
    wrong_blocks[0, query, 0] = (wrong_blocks[0, query, 0] + 1) % max(1, int(selection.spans[0, query, 1]) // 4)
    wrong_spans = selection.spans.clone()
    wrong_spans[..., 1] = (wrong_spans[..., 1] - 1).clamp(min=1)
    for blocks, spans in ((wrong_blocks, selection.spans), (selection.blocks, wrong_spans)):
        got = _gradients(
            lambda q, k, v: kernel.block_sparse_attention(q, k, v, blocks, spans, selection.compress_ratio, scale),
            operands,
            upstream,
        )
        with pytest.raises(AssertionError):
            _assert_close(got, tiled, DTYPE_TOLERANCE[dtype])


class _LoRALinear(nn.Module):
    """A frozen ``base`` plus a trained low-rank update, B nonzero so the adapter's own gradients are live."""

    def __init__(self, base: nn.Linear, rank: int = 4):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.randn(rank, base.in_features, dtype=base.weight.dtype, device="cuda") * 0.1)
        self.lora_b = nn.Parameter(torch.randn(base.out_features, rank, dtype=base.weight.dtype, device="cuda") * 0.1)

    def forward(self, x):
        return self.base(x) + (x @ self.lora_a.T) @ self.lora_b.T


def _lora_attention(dtype):
    config = layout_config(*NATIVE_LAYOUT, **REGIMES["sparse"])
    torch.manual_seed(SEED)
    attention = qwen4_exp_modeling.Qwen4ExpTextAttention(config, 3).cuda().to(dtype).requires_grad_(False)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(attention, name, _LoRALinear(getattr(attention, name)))
    return config, attention


def _lora_step(attention, inputs, dtype, mode, upstream, checkpointed=None):
    """Adapter gradients and the output of one attention forward under ``HALO_QSA_KERNEL=mode``."""
    attention.zero_grad(set_to_none=True)
    cos, sin = inputs.position_embeddings
    embeddings = (cos.cuda().to(dtype), sin.cuda().to(dtype))
    hidden = inputs.hidden.cuda().to(dtype).requires_grad_()
    seq_idx = inputs.seq_idx.cuda()

    def layer(h):
        return qsa_attention_forward(attention, h, embeddings, seq_idx=seq_idx)[0]

    with qsa_kernel(mode):
        output = layer(hidden) if checkpointed is None else scoped_checkpoint_func(checkpointed)(layer, hidden)
        output.backward(upstream)
    grads = {n: p.grad.clone() for n, p in attention.named_parameters() if p.requires_grad}
    return output.detach(), grads


def test_lora_gradients_flow_through_the_kernel(dtype):
    config, attention = _lora_attention(dtype)
    inputs = build_inputs(config, ROWS["packed"])
    torch.manual_seed(SEED)
    upstream = torch.randn(inputs.hidden.shape, device="cuda", dtype=dtype)
    on_output, on = _lora_step(attention, inputs, dtype, "on", upstream)
    off_output, off = _lora_step(attention, inputs, dtype, "off", upstream)
    assert on.keys() == off.keys() and len(on) == 8
    assert error_ratio(on_output, off_output) <= DTYPE_TOLERANCE[dtype]
    for name in on:
        assert on[name].abs().max() > 0, f"{name} received no gradient through the kernel"
        assert error_ratio(on[name], off[name]) <= 4 * DTYPE_TOLERANCE[dtype], name


def test_checkpointing_attends_the_forward_selection_and_matches_the_unchecked_run(reentrant, dtype=torch.bfloat16):
    config, attention = _lora_attention(dtype)
    inputs = build_inputs(config, ROWS["packed"])
    torch.manual_seed(SEED)
    upstream = torch.randn(inputs.hidden.shape, device="cuda", dtype=dtype)
    _, unchecked = _lora_step(attention, inputs, dtype, "on", upstream)
    blocks_seen = []
    attend = sparse_attention.block_sparse_attention

    def recording_attend(query, key, value, blocks, spans, ratio, scale):
        blocks_seen.append((active_recompute_scope().is_recompute, blocks))
        return attend(query, key, value, blocks, spans, ratio, scale)

    sparse_attention.block_sparse_attention = recording_attend
    try:
        _, checked = _lora_step(
            attention, inputs, dtype, "on", upstream, functools.partial(checkpoint, use_reentrant=reentrant)
        )
    finally:
        sparse_attention.block_sparse_attention = attend
    assert [recomputed for recomputed, _ in blocks_seen] == [False, True]
    assert blocks_seen[1][1] is blocks_seen[0][1], "the recompute selected its own blocks"
    for name in unchecked:
        assert error_ratio(checked[name], unchecked[name]) <= 4 * DTYPE_TOLERANCE[dtype], name


def _qsa_layers(model) -> int:
    return sum(isinstance(m, qwen4_exp_modeling.Qwen4ExpTextAttention) for m in model.modules())


def _model_step(model, batch: dict, mode: str, *, recomputed: bool = False):
    """Loss and full gradients of one step under ``HALO_QSA_KERNEL=mode``. On the kernel, every QSA layer must have
    run it (again in its checkpoint's recompute when ``recomputed``): a silent fallback to the tiled path would
    otherwise make the two modes agree."""
    model.zero_grad()
    with (
        qsa_kernel(mode),
        mock.patch.object(sparse_attention, "block_sparse_attention", wraps=kernel.block_sparse_attention) as launched,
    ):
        loss = model(use_cache=False, **batch).loss
        loss.backward()
    expected = _qsa_layers(model) * (2 if recomputed else 1) if mode == "on" else 0
    assert launched.call_count == expected, f"{launched.call_count} kernel launches where {expected} were due"
    grads = {n: p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad for n, p in model.named_parameters()}
    return loss.item(), {n: g.detach().clone() for n, g in grads.items() if g is not None}


def _assert_steps_agree(kernel_step, tiled_step, label: str) -> None:
    loss, grads = kernel_step
    tiled_loss, tiled = tiled_step
    assert abs(loss - tiled_loss) <= MODEL_LOSS_TOLERANCE * abs(tiled_loss), (label, loss, tiled_loss)
    assert grads.keys() == tiled.keys()
    attention = [n for n in tiled if any(f".{p}." in n for p in ("q_proj", "k_proj", "v_proj", "o_proj"))]
    assert attention, "no attention projection received a gradient"
    worst = max(error_ratio(grads[n], tiled[n]) for n in attention)
    assert worst <= MODEL_GRAD_TOLERANCE, f"{label}: attention-projection gradients differ by {worst:.2e}"


def test_model_on_the_kernel_matches_the_tiled_path(model, kind):
    batch = training_batch(model, kind)
    _assert_steps_agree(_model_step(model, batch, "on"), _model_step(model, batch, "off"), kind)


def test_model_checkpointing_attends_the_forward_selection_on_the_kernel(model, reentrant):
    batch = training_batch(model, "packed")
    plain = _model_step(model, batch, "on")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
    model.train()
    try:
        checkpointed = _model_step(model, batch, "on", recomputed=True)
    finally:
        model.gradient_checkpointing_disable()
    _assert_steps_agree(checkpointed, plain, f"reentrant={reentrant}")


def test_model_under_fsdp2_runs_the_kernel_on_the_sharded_projections(model):
    batch = training_batch(model, "packed")
    plain = _model_step(model, batch, "on")
    mesh = DeviceMesh("cuda", [0])
    sharded = copy.deepcopy(model)
    for layer in sharded.model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(sharded, mesh=mesh)
    assert isinstance(sharded.model.layers[QSA_LAYER_IDX].self_attn.q_proj.weight, DTensor)
    _assert_steps_agree(_model_step(sharded, batch, "on"), plain, "fsdp2")


def test_model_negative_control_a_wrong_selection_fails_the_comparison(model):
    """The model-level comparison must notice a kernel handed one wrong block id per query."""
    batch = training_batch(model, "packed")
    tiled = _model_step(model, batch, "off")
    real = kernel.block_sparse_attention

    def shifted(query, key, value, blocks, spans, ratio, scale):
        return real(query, key, value, blocks.roll(1, dims=-1), spans, ratio, scale)

    with mock.patch.object(kernel, "block_sparse_attention", shifted):
        wrong = _model_step(model, batch, "on")
    with pytest.raises(AssertionError):
        _assert_steps_agree(wrong, tiled, "shifted blocks")


def _recent_blocks(batch: int, length: int, ratio: int, block_topk: int):
    """One document per row: each query attends its most recent complete blocks."""
    visible = torch.arange(1, length + 1, device="cuda")
    selected = (visible // ratio).clamp(max=block_topk)
    slot = torch.arange(block_topk, device="cuda")
    blocks = torch.where(slot < selected[:, None], (visible // ratio)[:, None] - 1 - slot, -1)
    spans = torch.stack([torch.zeros_like(visible), visible], dim=-1)
    return (
        blocks.expand(batch, -1, -1).to(torch.int16).contiguous(),
        spans.expand(batch, -1, -1).to(torch.int32).contiguous(),
    )


def _compiled_binaries() -> int:
    device = torch.cuda.current_device()
    return sum(len(fn.device_caches[device][0]) for fn in (kernel._forward_kernel, kernel._backward_kernel))


def _launch(batch, length, config, dtype, layout="contiguous", index_dtype=torch.int16):
    ratio, block_topk = config.indexer_compress_ratio, config.indexer_budget // config.indexer_compress_ratio
    blocks, spans = _recent_blocks(batch, length, ratio, block_topk)
    blocks = blocks.to(index_dtype)
    heads, kv_heads, dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    q, k, v = (torch.randn(batch, h, length, dim, device="cuda", dtype=dtype) for h in (heads, kv_heads, kv_heads))
    if layout == "model":
        # The layer's own storage: [batch, length, heads, dim] transposed.
        q, k, v = (t.transpose(1, 2).contiguous().transpose(1, 2) for t in (q, k, v))
    elif layout == "misaligned":
        flat = torch.randn(q.numel() + 1, device="cuda", dtype=dtype)
        q = flat[1:].view_as(q)
    q, k, v = (t.requires_grad_() for t in (q, k, v))
    kernel.block_sparse_attention(q, k, v, blocks, spans, ratio, dim**-0.5).sum().backward()


def _clear_kernel_caches() -> None:
    """Forget every binary of the two kernels, so a count after this starts from nothing."""
    for fn in (kernel._forward_kernel, kernel._backward_kernel):
        fn.device_caches.clear()


def _step_lengths(config, dtype) -> None:
    for batch in (1, 2):
        for length in PACK_LENGTHS:
            _launch(batch, length, config, dtype)
    for layout in ("model", "misaligned"):
        for length in (37, 512, 513):
            _launch(1, length, config, dtype, layout)
    # Selection picks int32 block ids for a document past the int16 range, which no test row reaches.
    _launch(1, 513, config, dtype, index_dtype=torch.int32)


def test_no_compile_after_the_warm_up_for_any_pack_length(dtype=torch.bfloat16):
    config = layout_config(*NATIVE_LAYOUT, **REGIMES["sparse"])
    heads, kv_heads, dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    _clear_kernel_caches()
    assert _compiled_binaries() == 0
    kernel.warm_block_sparse_kernels(
        torch.device("cuda", torch.cuda.current_device()), dtype, heads, kv_heads, dim, config.indexer_compress_ratio
    )
    warm = _compiled_binaries()
    assert warm > 0, "the warm-up compiled nothing"
    _step_lengths(config, dtype)
    assert _compiled_binaries() == warm, f"{_compiled_binaries() - warm} binaries compiled after the warm-up"


def test_a_removed_warm_up_is_caught_by_the_count(dtype=torch.bfloat16):
    """Negative control: with no warm-up, the first steps compile, so the count above would fail."""
    config = layout_config(*NATIVE_LAYOUT, **REGIMES["sparse"])
    _clear_kernel_caches()
    _step_lengths(config, dtype)
    assert _compiled_binaries() > 0


def test_finalizing_a_loaded_model_leaves_nothing_to_compile_in_a_step(dtype=torch.bfloat16):
    """The load path itself warms: finalizing a model with QSA layers compiles the kernels, and a packed step then
    compiles none."""
    config = layout_config(*NATIVE_LAYOUT, **REGIMES["sparse"])
    attention = qwen4_exp_modeling.Qwen4ExpTextAttention(config, 3).cuda().to(dtype)
    apply_qwen4_exp_patches(config)
    _clear_kernel_caches()
    with qsa_kernel("on"):
        finalize_loaded_model(torch.nn.ModuleList([attention]))
        warm = _compiled_binaries()
        assert warm > 0, "finalizing warmed no kernel"
        inputs = build_inputs(config, ROWS["packed"])
        cos, sin = inputs.position_embeddings
        hidden = inputs.hidden.cuda().to(dtype).requires_grad_()
        output, _ = qsa_attention_forward(
            attention, hidden, (cos.cuda().to(dtype), sin.cuda().to(dtype)), seq_idx=inputs.seq_idx.cuda()
        )
        output.sum().backward()
    assert _compiled_binaries() == warm


def test_determinism_takes_the_tiled_path_or_refuses():
    config, selection, (query, key, value), _ = _case(NATIVE_LAYOUT, ROWS["packed"], torch.bfloat16, REGIMES["sparse"])
    scale = config.head_dim**-0.5
    previous = os.environ.get("FLASH_ATTENTION_DETERMINISTIC")
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    try:
        with qsa_kernel("auto"):
            assert torch.equal(
                qsa_attention(query, key, value, selection, scale),
                tiled_qsa_attention(query, key, value, selection, scale),
            )
        with qsa_kernel("on"), pytest.raises(ValueError, match="full_determinism"):
            qsa_attention(query, key, value, selection, scale)
    finally:
        if previous is None:
            os.environ.pop("FLASH_ATTENTION_DETERMINISTIC")
        else:
            os.environ["FLASH_ATTENTION_DETERMINISTIC"] = previous


def test_forward_and_dq_repeat_bitwise_and_dk_dv_agree_within_tolerance(dtype=torch.bfloat16):
    config, selection, operands, upstream = _case(NATIVE_LAYOUT, ROWS["packed-two-rows"], dtype, REGIMES["sparse"])
    scale = config.head_dim**-0.5
    first = _gradients(_kernel(selection, scale), operands, upstream)
    second = _gradients(_kernel(selection, scale), operands, upstream)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1][0], second[1][0])
    for a, b in zip(first[1][1:], second[1][1:], strict=True):
        assert error_ratio(a, b) <= 1e-2


@gpu_test_main(exact_world_size=1, prefix="block_sparse_attention", partial_state=False)
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    for dtype in DTYPE_TOLERANCE:
        for name, layout in LAYOUTS.items():
            record_check(
                checks,
                f"matches_fp64[{name}-{dtype}]",
                lambda: test_matches_fp64_no_worse_than_the_tiled_path(layout, dtype),
            )
        for regime_name, regime in REGIMES.items():
            for rows_name, rows in ALL_ROWS.items():
                record_check(
                    checks,
                    f"matches_tiled[{regime_name}-{rows_name}-{dtype}]",
                    lambda: test_matches_the_tiled_path_in_both_regimes(regime, rows, dtype),
                )
        record_check(
            checks, f"left_padding[{dtype}]", lambda: test_left_padded_queries_read_zeros_and_carry_no_gradient(dtype)
        )
        record_check(checks, f"right_padding[{dtype}]", lambda: test_right_padded_queries_match_the_tiled_path(dtype))
        record_check(
            checks,
            f"empty_selections[{dtype}]",
            lambda: test_empty_selections_attend_the_tail_only_and_nothing_goes_nan(dtype),
        )
        record_check(checks, f"lora[{dtype}]", lambda: test_lora_gradients_flow_through_the_kernel(dtype))
    record_check(checks, "negative_controls", test_negative_controls_fail_the_comparison)
    for reentrant in (True, False):
        record_check(
            checks,
            f"checkpointing[reentrant={reentrant}]",
            lambda: test_checkpointing_attends_the_forward_selection_and_matches_the_unchecked_run(reentrant),
        )
    record_check(checks, "no_compile_after_warm_up", test_no_compile_after_the_warm_up_for_any_pack_length)
    record_check(checks, "no_warm_up_compiles_in_a_step", test_a_removed_warm_up_is_caught_by_the_count)
    record_check(checks, "finalizing_warms", test_finalizing_a_loaded_model_leaves_nothing_to_compile_in_a_step)
    model = load_tiny_cuda_model(torch.bfloat16)
    for kind in ("packed", "padded"):
        record_check(checks, f"model[{kind}]", lambda: test_model_on_the_kernel_matches_the_tiled_path(model, kind))
    for reentrant in (True, False):
        record_check(
            checks,
            f"model_checkpointing[reentrant={reentrant}]",
            lambda: test_model_checkpointing_attends_the_forward_selection_on_the_kernel(model, reentrant),
        )
    record_check(
        checks, "model_fsdp2", lambda: test_model_under_fsdp2_runs_the_kernel_on_the_sharded_projections(model)
    )
    record_check(
        checks,
        "model_negative_control",
        lambda: test_model_negative_control_a_wrong_selection_fails_the_comparison(model),
    )
    record_check(checks, "determinism", test_determinism_takes_the_tiled_path_or_refuses)
    record_check(checks, "repeatability", test_forward_and_dq_repeat_bitwise_and_dk_dv_agree_within_tolerance)
    return {"checks": checks}


if __name__ == "__main__":
    run()
