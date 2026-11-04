#!/usr/bin/env python
"""Hyper-connection kernels match upstream's arithmetic under autograd: each kernel against its eager form, every
gradient included, then tiny GLM-5 Next and DeepSeek-V4 models with the kernels installed against upstream's forwards,
through the paths a decoder layer takes (reentrant recompute, frozen weights, LoRA, FSDP2, pipeline stages)."""

import contextlib
import itertools
from collections.abc import Callable, Iterator
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from halo.distributed.pipeline_parallel.stage import build_pipeline_stage
from halo.kernels import hyper_connection as hc
from halo.kernels.hyper_connection import collapse_streams, mix_streams, project_streams, sinkhorn_knopp
from halo.models.hyper_connections import _FAMILIES, apply_hyper_connection_patches
from halo.models.moe.family_specs import find_moe_family_spec
from tests.common.harness import gpu_test_main, record_check
from tests.common.numerics import fro_rel_err
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

EPS, NORM_EPS = 1e-6, 1e-5
# Against an fp64 reference, a kernel output or gradient holds under this or ``EAGER_ERROR_RATIO_MAX`` times what the
# eager form loses at the same dtype, whichever is larger: at one stream ``g - g * y`` cancels to the rounding of both.
FP32_TOL = 2e-6
BF16_TOL = 2e-3
EAGER_ERROR_RATIO_MAX = 2.0
# A near-cancelled model gradient, the hyper-connection scales above all, loses digits to rounding either way.
FP32_MODEL_GRAD_TOL = 5e-3
FP32_LOSS_TOL = 1e-5
RELEASE_HC = {"hc_mult": 4, "hc_sinkhorn_iters": 20}
# Selects every expert and every indexed key, as a flipped top-k moves whole gradients and swamps the rounding scored.
FLIP_FREE = {"num_experts_per_tok": 8, "index_topk": 4096}
# The kernels' summed gradient error over upstream's on a whole bf16 model, and its loss's relative error: one scalar,
# at the rounding noise of the head either way, so bounded outright.
BF16_MODEL_ERROR_RATIO_MAX = 1.1
BF16_MODEL_LOSS_TOL = 2e-4
# A bf16 result accumulated in fp32 is the exact result rounded once.
BF16_FLOOR_RATIO_MAX = 1.1
# Streams and Sinkhorn iterations of each.
SINKHORN_SHAPES = ((1, 1), (2, 4), (3, 2), (4, 20), (8, 5))
# Batch and tokens, on and off the edges of every program: 8 to 128 tokens for Sinkhorn, 64 for the projection.
TOKEN_SHAPES = ((1, 1), (1, 31), (1, 32), (1, 33), (3, 17), (1, 63), (1, 65), (2, 129))
# Streams and width of each.
STREAM_SHAPES = ((1, 64), (2, 40), (3, 33), (4, 16), (4, 18), (4, 1100), (8, 8))
DTYPE_TOLS = ((torch.float32, FP32_TOL), (torch.bfloat16, BF16_TOL))
# What a wrong backward must miss by, against the same reference.
WRONG_BACKWARD_ERROR_MIN = 1e-2
_MODEL_TOKENS = 48
_MODEL_BATCH = 2
_LORA_RANK = 4


def _generator(seed: int = 0) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _randn(*shape, dtype=torch.float32, seed: int = 0) -> torch.Tensor:
    return torch.randn(shape, generator=_generator(seed), device="cuda", dtype=dtype)


def _combs(batch: int, tokens: int, n: int, seed: int = 0) -> torch.Tensor:
    return torch.softmax(_randn(batch, tokens, n, n, seed=seed), dim=-1) + EPS


def _vjp(fn: Callable, inputs: tuple[torch.Tensor, ...], needed: tuple[bool, ...], dtype_of=lambda t: t.dtype):
    """``fn``'s output and the gradients of the inputs ``needed``, for one fixed random cotangent."""
    leaves = tuple(t.detach().to(dtype_of(t)).requires_grad_(need) for t, need in zip(inputs, needed, strict=True))
    out = fn(*leaves)
    cotangent = torch.randn(out.shape, generator=_generator(1), device="cuda", dtype=torch.float32).to(out.dtype)
    wanted = [leaf for leaf in leaves if leaf.requires_grad]
    return [out, *torch.autograd.grad(out, wanted, cotangent)]


def _errors(
    kernel: Callable, eager: Callable, inputs: tuple[torch.Tensor, ...], needed=None
) -> list[tuple[float, float]]:
    """``(kernel, eager)`` relative errors against the eager form at fp64, for the output and each gradient."""
    needed = needed or (True,) * len(inputs)
    kernel_results = _vjp(kernel, inputs, needed)
    eager_results = _vjp(eager, inputs, needed)
    for got, want in zip(kernel_results, eager_results, strict=True):
        assert (got.dtype, got.shape) == (want.dtype, want.shape), (got.dtype, got.shape, want.dtype, want.shape)
    exact = _vjp(eager, inputs, needed, lambda _: torch.float64)
    return [
        (fro_rel_err(got, want), fro_rel_err(rounded, want))
        for got, rounded, want in zip(kernel_results, eager_results, exact, strict=True)
    ]


def _assert_close(kernel: Callable, eager: Callable, inputs: tuple[torch.Tensor, ...], tol: float, needed=None):
    for kernel_error, eager_error in _errors(kernel, eager, inputs, needed):
        assert kernel_error <= max(tol, EAGER_ERROR_RATIO_MAX * eager_error), (kernel_error, eager_error, tol)


def _wide(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.promote_types(tensor.dtype, torch.float32))


def _eager_projection(stream, fn):
    flat = _wide(stream.view(*stream.shape[:-2], -1))
    return F.linear(flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + NORM_EPS), _wide(fn))


def _kernel_projection(stream, fn):
    return project_streams(stream, fn, NORM_EPS)


def _eager_collapse(stream, weight):
    return (weight.unsqueeze(-1) * stream).sum(dim=-2).to(stream.dtype)


def _eager_sinkhorn(comb, iters: int):
    comb = comb / (comb.sum(dim=-2, keepdim=True) + EPS)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + EPS)
    return comb


def _eager_mix(block, residual, post, comb):
    return post.unsqueeze(-1) * block.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), residual)


def _projection_inputs(batch, tokens, n, width, dtype=torch.float32, fn_dtype=None):
    lanes = (2 + n) * n
    return _randn(batch, tokens, n, width, dtype=dtype), _randn(
        lanes, n * width, dtype=fn_dtype or dtype, seed=2
    ) * 0.02


def _collapse_inputs(batch, tokens, n, width, dtype=torch.float32):
    return _randn(batch, tokens, n, width, dtype=dtype), torch.sigmoid(_randn(batch, tokens, n, seed=2))


def _mix_inputs(batch, tokens, n, width, dtype=torch.float32, block_dtype=None):
    return (
        _randn(batch, tokens, width, dtype=block_dtype or dtype),
        _randn(batch, tokens, n, width, dtype=dtype, seed=2),
        (2 * torch.sigmoid(_randn(batch, tokens, n, seed=3))).to(dtype),
        _combs(batch, tokens, n, seed=4).to(dtype),
    )


def test_sinkhorn_matches_autograd():
    for (n, iters), (batch, tokens) in itertools.product(SINKHORN_SHAPES, TOKEN_SHAPES):
        _assert_close(
            lambda comb: sinkhorn_knopp(comb, iters, EPS),
            lambda comb: _eager_sinkhorn(comb, iters),
            (_combs(batch, tokens, n),),
            FP32_TOL,
        )


def test_sinkhorn_reverses_near_zero_rows():
    """A row of ``eps`` floors divides by a scale ``eps`` itself dominates; the gradient grows by orders of magnitude."""
    comb = _combs(1, 40, 4)
    comb[:, ::3] = EPS
    comb[:, 1::3, 0] = EPS
    _assert_close(lambda c: sinkhorn_knopp(c, 20, EPS), lambda c: _eager_sinkhorn(c, 20), (comb,), FP32_TOL)


def test_sinkhorn_takes_a_zero_eps_at_a_padded_stream_count():
    """Three streams pad to four lanes, whose empty one has nothing to divide by."""
    comb = torch.softmax(_randn(1, 40, 3, 3), dim=-1)

    def eager(c):
        c = c / c.sum(dim=-2, keepdim=True)
        c = c / c.sum(dim=-1, keepdim=True)
        return c / c.sum(dim=-2, keepdim=True)

    _assert_close(lambda c: sinkhorn_knopp(c, 2, 0.0), eager, (comb,), FP32_TOL)


def test_sinkhorn_saves_only_the_scales_with_a_graph():
    """The eager loop keeps the matrix and its sums per division, 80 bytes a token a step; here one scale per step."""
    tokens, n, iters = 4096, 4, 20
    comb = _combs(1, tokens, n).requires_grad_(True)
    before = torch.cuda.memory_allocated()
    out = sinkhorn_knopp(comb, iters, EPS)
    held = torch.cuda.memory_allocated() - before
    expected = tokens * ((2 * iters - 1) * n + n * n) * comb.element_size()
    assert held <= expected * 1.01, (held, expected)
    del out

    with torch.no_grad():
        before = torch.cuda.memory_allocated()
        out = sinkhorn_knopp(comb, iters, EPS)
        held = torch.cuda.memory_allocated() - before
    assert held <= tokens * n * n * comb.element_size() * 1.01, held

    def recording_a_graph(*_args):
        raise AssertionError("a call with no graph to record reached the recording path")

    with patch.object(hc._SinkhornKnopp, "apply", recording_a_graph):
        assert torch.equal(sinkhorn_knopp(comb.detach(), iters, EPS), out)


def test_projection_matches_autograd():
    for (n, width), (batch, tokens), (dtype, tol) in itertools.product(STREAM_SHAPES, TOKEN_SHAPES, DTYPE_TOLS):
        _assert_close(_kernel_projection, _eager_projection, _projection_inputs(batch, tokens, n, width, dtype), tol)


def test_projection_reads_mixed_dtypes():
    """``fp32_non_ep_params`` keeps ``fn`` an fp32 master under bf16 streams."""
    for stream_dtype, fn_dtype in ((torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)):
        inputs = _projection_inputs(1, 70, 4, 32, stream_dtype, fn_dtype)
        _assert_close(_kernel_projection, _eager_projection, inputs, BF16_TOL)


def test_projection_takes_a_zero_stream():
    """A token at rest: its normalizer rests on ``norm_eps`` alone."""
    stream, fn = _projection_inputs(1, 70, 4, 32)
    stream[:, ::2] = 0.0
    _assert_close(_kernel_projection, _eager_projection, (stream, fn), FP32_TOL)


def test_collapse_matches_autograd():
    for (n, width), (batch, tokens), (dtype, tol) in itertools.product(STREAM_SHAPES, TOKEN_SHAPES, DTYPE_TOLS):
        _assert_close(collapse_streams, _eager_collapse, _collapse_inputs(batch, tokens, n, width, dtype), tol)


def test_mix_matches_autograd():
    for (n, width), (batch, tokens), (dtype, tol) in itertools.product(STREAM_SHAPES, TOKEN_SHAPES, DTYPE_TOLS):
        _assert_close(mix_streams, _eager_mix, _mix_inputs(batch, tokens, n, width, dtype), tol)


def test_mix_promotes_like_upstream():
    """Under the bf16 autocast of an fp32-master run a block's dtype can differ from the streams'; the output takes the
    wider, and each gradient its own input's."""
    for dtype, block_dtype in ((torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)):
        _assert_close(mix_streams, _eager_mix, _mix_inputs(1, 40, 4, 64, dtype, block_dtype), BF16_TOL)


def test_each_input_is_optional():
    """The first layer of a frozen embedding has a constant residual, and a frozen hyper-connection constant weights:
    each subset of inputs compiles a variant that writes only what was asked for."""
    cases = (
        (mix_streams, _eager_mix, _mix_inputs(1, 40, 4, 64)),
        (collapse_streams, _eager_collapse, _collapse_inputs(1, 40, 4, 64)),
        (_kernel_projection, _eager_projection, _projection_inputs(1, 70, 4, 32)),
    )
    for kernel, eager, inputs in cases:
        for needed in itertools.product((True, False), repeat=len(inputs)):
            if any(needed):
                _assert_close(kernel, eager, inputs, FP32_TOL, needed)


def test_bf16_rounds_once():
    """The kernels accumulate in fp32 and round at the store, so they land on the exact result's rounding floor, where
    upstream's product and sum round separately."""
    n, width = 4, 1100
    block, residual, post, comb = _mix_inputs(1, 65, n, width, torch.bfloat16)
    exact = _eager_mix(block.double(), residual.double(), post.double(), comb.double())
    floor = fro_rel_err(exact.to(torch.bfloat16), exact)
    error = fro_rel_err(mix_streams(block, residual, post, comb), exact)
    assert error <= floor * BF16_FLOOR_RATIO_MAX, (error, floor)
    stream, weight = _collapse_inputs(1, 65, n, width, torch.bfloat16)
    exact = _eager_collapse(stream.double(), weight.double())
    floor = fro_rel_err(exact.to(torch.bfloat16), exact)
    error = fro_rel_err(collapse_streams(stream, weight), exact)
    assert error <= floor * BF16_FLOOR_RATIO_MAX, (error, floor)


def test_a_second_backward_reads_the_same_state():
    """A retained graph replays its reverse from the saved state, which the first pass must leave intact."""
    comb = _combs(1, 40, 4).requires_grad_(True)
    out = sinkhorn_knopp(comb, 20, EPS)
    first = torch.autograd.grad(out.sum(), comb, retain_graph=True)[0]
    assert torch.equal(first, torch.autograd.grad(out.sum(), comb)[0])
    block, residual, post, comb = (t.requires_grad_(True) for t in _mix_inputs(1, 40, 4, 64))
    out = mix_streams(block, residual, post, comb)
    leaves = (block, residual, post, comb)
    first = torch.autograd.grad(out.sum(), leaves, retain_graph=True)
    for one, two in zip(first, torch.autograd.grad(out.sum(), leaves), strict=True):
        assert torch.equal(one, two)


def test_no_token_and_strided_inputs():
    """Upstream takes a batch with no token, each kernel's gradient included, and a strided tensor is copied, not read
    at the wrong offsets."""
    for kernel, inputs, shape in (
        (lambda comb: sinkhorn_knopp(comb, 4, EPS), (_combs(1, 0, 4),), (1, 0, 4, 4)),
        (collapse_streams, _collapse_inputs(1, 0, 4, 64), (1, 0, 64)),
        (_kernel_projection, _projection_inputs(1, 0, 4, 64), (1, 0, 24)),
        (mix_streams, _mix_inputs(1, 0, 4, 64), (1, 0, 4, 64)),
    ):
        out, *grads = _vjp(kernel, inputs, (True,) * len(inputs))
        assert out.shape == shape, out.shape
        assert [grad.shape for grad in grads] == [t.shape for t in inputs]
        assert all(grad.count_nonzero() == 0 for grad in grads)

    def every_other(fn):
        return lambda *inputs: fn(*(t[:, ::2] for t in inputs))

    _assert_close(every_other(mix_streams), every_other(_eager_mix), _mix_inputs(1, 40, 4, 64), FP32_TOL)
    _assert_close(
        every_other(collapse_streams), every_other(_eager_collapse), _collapse_inputs(1, 40, 4, 64), FP32_TOL
    )
    _assert_close(
        lambda comb: sinkhorn_knopp(comb[:, ::2], 4, EPS),
        lambda comb: _eager_sinkhorn(comb[:, ::2], 4),
        (_combs(1, 40, 4),),
        FP32_TOL,
    )
    _assert_close(
        lambda stream, fn: _kernel_projection(stream[:, ::2], fn),
        lambda stream, fn: _eager_projection(stream[:, ::2], fn),
        _projection_inputs(1, 40, 4, 32),
        FP32_TOL,
    )


def test_a_shape_the_streams_do_not_take_is_refused():
    """The kernels address every input off the streams' shape, where a mismatch would write out of bounds."""
    block, residual, post, comb = _mix_inputs(1, 8, 4, 64)
    stream, weight = _collapse_inputs(1, 8, 4, 64)
    stream, fn = _projection_inputs(1, 8, 4, 64)
    for call in (
        lambda: mix_streams(block[:, :-1], residual, post, comb),
        lambda: mix_streams(block, residual, post[:, :-1], comb),
        lambda: mix_streams(block, residual, post, comb[:, :-1]),
        lambda: collapse_streams(stream, weight[:, :-1]),
        lambda: project_streams(stream, fn[:, :-1], NORM_EPS),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("a mismatched shape reached a kernel")


def test_a_second_order_backward_is_refused():
    """With another differentiable term beside it, an undifferentiable backward would read as a zero second derivative."""
    comb = _combs(1, 8, 4).requires_grad_(True)
    loss = (sinkhorn_knopp(comb, 4, EPS).square() + comb.square()).sum()
    grad = torch.autograd.grad(loss, comb, create_graph=True)[0]
    try:
        grad.sum().backward()
    except RuntimeError as error:
        assert "marked with @once_differentiable" in str(error), error
        return
    raise AssertionError("a second-order backward through the kernels ran")


class _WrongOperand:
    """Launches ``kernel`` with one operand altered, as a backward with that bug would read it."""

    def __init__(self, kernel, index: int, alter: Callable[[torch.Tensor], torch.Tensor]):
        self.kernel, self.index, self.alter = kernel, index, alter

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            args = list(args)
            args[self.index] = self.alter(args[self.index])
            launch(*args, **kwargs)

        return run


def test_a_wrong_backward_fails():
    """Each comparison must see its bug class: reversed scales, a dropped norm term, a wrong per-token factor, a
    transposed matrix."""
    sinkhorn_inputs = (_combs(1, 40, 4),)
    wrong = (
        (
            "_sinkhorn_bwd_kernel",
            2,
            lambda scales: scales.flip(1),
            lambda comb: sinkhorn_knopp(comb, 20, EPS),
            lambda comb: _eager_sinkhorn(comb, 20),
            sinkhorn_inputs,
        ),
        (
            "_projection_stream_bwd_kernel",
            3,
            torch.zeros_like,
            _kernel_projection,
            _eager_projection,
            _projection_inputs(1, 70, 4, 32),
        ),
        (
            "_projection_fn_bwd_kernel",
            2,
            torch.ones_like,
            _kernel_projection,
            _eager_projection,
            _projection_inputs(1, 70, 4, 32),
        ),
        (
            "_collapse_bwd_kernel",
            2,
            lambda weight: weight.flip(-1),
            collapse_streams,
            _eager_collapse,
            _collapse_inputs(1, 40, 4, 64),
        ),
        (
            "_mix_bwd_kernel",
            4,
            lambda comb: comb.transpose(-1, -2).contiguous(),
            mix_streams,
            _eager_mix,
            _mix_inputs(1, 40, 4, 64),
        ),
    )
    for name, index, alter, kernel, eager, inputs in wrong:
        _assert_close(kernel, eager, inputs, FP32_TOL)
        with patch.object(hc, name, _WrongOperand(getattr(hc, name), index, alter)):
            worst = max(kernel_error for kernel_error, _ in _errors(kernel, eager, inputs))
        assert worst > WRONG_BACKWARD_ERROR_MIN, (name, worst)


def _compiled_binaries(kernel) -> int:
    return len(kernel.device_caches[torch.cuda.current_device()][0])


def test_one_compile_serves_every_token_count():
    """A token count must not select a binary of its own, as an unseen one would stall one rank mid-step."""
    kernels = (
        hc._sinkhorn_fwd_kernel,
        hc._sinkhorn_bwd_kernel,
        hc._projection_fwd_kernel,
        hc._projection_stream_bwd_kernel,
        hc._projection_fn_bwd_kernel,
        hc._collapse_fwd_kernel,
        hc._collapse_bwd_kernel,
        hc._mix_fwd_kernel,
        hc._mix_bwd_kernel,
    )

    def run(tokens: int) -> None:
        for fn, inputs in (
            (mix_streams, _mix_inputs(1, tokens, 4, 64)),
            (collapse_streams, _collapse_inputs(1, tokens, 4, 64)),
            (_kernel_projection, _projection_inputs(1, tokens, 4, 64)),
            (lambda comb: sinkhorn_knopp(comb, 4, EPS), (_combs(1, tokens, 4),)),
        ):
            leaves = tuple(t.requires_grad_(True) for t in inputs)
            torch.autograd.grad(fn(*leaves).sum(), leaves)

    for kernel in kernels:
        kernel.device_caches.clear()
    run(16)
    warm = [_compiled_binaries(kernel) for kernel in kernels]
    for tokens in (1, 17, 33, 48, 1025):
        run(tokens)
    assert [_compiled_binaries(kernel) for kernel in kernels] == warm, "a token count compiled a new binary"


def _recording_compilation(real: Callable, seen: list[bool]) -> Callable:
    def spy(*args, **kwargs):
        seen.append(torch.compiler.is_compiling())
        return real(*args, **kwargs)

    return spy


def test_a_compiled_caller_runs_the_kernels_eagerly():
    """Each public kernel stays out of the trace, where the spy its body calls would read a tracer."""
    block, residual, post, comb = _mix_inputs(1, 8, 4, 64)
    stream, weight = _collapse_inputs(1, 8, 4, 64)
    fn = _projection_inputs(1, 8, 4, 64)[1]
    combs = _combs(1, 8, 4)
    for function, owner, attribute in (
        (lambda: sinkhorn_knopp(combs, 4, EPS), hc, "_sinkhorn_forward"),
        (lambda: project_streams(stream, fn, NORM_EPS), hc._StreamProjection, "apply"),
        (lambda: collapse_streams(stream, weight), hc._StreamCollapse, "apply"),
        (lambda: mix_streams(block, residual, post, comb), hc._StreamMix, "apply"),
    ):
        seen: list[bool] = []
        with patch.object(owner, attribute, _recording_compilation(getattr(owner, attribute), seen)):
            torch.compile(function)()
        assert seen == [False], (attribute, seen)

    traced: list[bool] = []
    torch.compile(_recording_compilation(torch.neg, traced))(combs)
    assert traced == [True], "the spy cannot see a trace"


def _build(model_type: str, dtype: torch.dtype, **overrides) -> nn.Module:
    torch.manual_seed(7)
    return tiny_family_model(TINY_MOE_FAMILIES[model_type], overrides=overrides).to(device="cuda", dtype=dtype)


@contextlib.contextmanager
def _upstream_forwards(model_type: str) -> Iterator[None]:
    patched = [(owner, owner.forward) for owner, _ in _FAMILIES[find_moe_family_spec(model_type)]]
    for owner, forward in patched:
        owner.forward = forward.__wrapped__
    try:
        yield
    finally:
        for owner, forward in patched:
            owner.forward = forward


def _ids(model: nn.Module) -> torch.Tensor:
    vocab_size = model.config.get_text_config().vocab_size
    return torch.randint(0, vocab_size, (_MODEL_BATCH, _MODEL_TOKENS), generator=_generator(5), device="cuda")


def _plain_grad(parameter: nn.Parameter) -> torch.Tensor | None:
    grad = parameter.grad
    return grad.full_tensor() if isinstance(grad, DTensor) else grad


def _loss_and_grads(model: nn.Module) -> tuple[float, dict[str, torch.Tensor]]:
    model.zero_grad()
    ids = _ids(model)
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    grads = {name: _plain_grad(parameter) for name, parameter in model.named_parameters()}
    return loss.item(), {name: grad.detach().clone() for name, grad in grads.items() if grad is not None}


def _worst_grad_error(got: dict[str, torch.Tensor], want: dict[str, torch.Tensor]) -> tuple[float, str]:
    assert got.keys() == want.keys(), sorted(got.keys() ^ want.keys())
    scored = [(fro_rel_err(got[name], grad), name) for name, grad in want.items() if grad.norm() > 0]
    assert scored, "no parameter holds a gradient"
    return max(scored)


def _assert_kernels_match_upstream(model_type: str, model: nn.Module, tol: float = FP32_MODEL_GRAD_TOL) -> None:
    with _upstream_forwards(model_type):
        stock_loss, stock_grads = _loss_and_grads(model)
    loss, grads = _loss_and_grads(model)
    assert abs(loss - stock_loss) <= FP32_LOSS_TOL * abs(stock_loss), (loss, stock_loss)
    worst, name = _worst_grad_error(grads, stock_grads)
    assert worst < tol, f"{model_type} {name}: {worst:.2e}"


def _total_error(candidate: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> float:
    assert candidate.keys() == reference.keys(), sorted(candidate.keys() ^ reference.keys())
    return sum(float((candidate[name].double() - grad.double()).norm()) for name, grad in reference.items())


def test_model_matches_upstream_at_fp32(model_type):
    """On the tiny config, and on the stream count and Sinkhorn depth both families' configs default to."""
    _assert_kernels_match_upstream(model_type, _build(model_type, torch.float32))
    _assert_kernels_match_upstream(model_type, _build(model_type, torch.float32, **RELEASE_HC))


def test_model_bf16_error_is_no_worse_than_upstreams(model_type):
    """On the same weights and with no top-k to flip, each against an fp32 run of upstream's forwards."""
    fp32_model = _build(model_type, torch.float32, **FLIP_FREE)
    with _upstream_forwards(model_type):
        reference_loss, reference = _loss_and_grads(fp32_model)
    del fp32_model
    model = _build(model_type, torch.bfloat16, **FLIP_FREE)
    with _upstream_forwards(model_type):
        stock_loss, stock_grads = _loss_and_grads(model)
    loss, grads = _loss_and_grads(model)

    kernel_error, stock_error = _total_error(grads, reference), _total_error(stock_grads, reference)
    assert kernel_error <= BF16_MODEL_ERROR_RATIO_MAX * stock_error, (kernel_error, stock_error)
    assert abs(loss - reference_loss) <= BF16_MODEL_LOSS_TOL * reference_loss, (loss, stock_loss, reference_loss)


def test_recompute_matches_a_plain_backward(model_type):
    """Reentrant recompute records the graph on the second pass, the first having run with none to record;
    non-reentrant stores the kernels' saved state through its own hooks."""
    model = _build(model_type, torch.float32)
    _, plain = _loss_and_grads(model)
    for reentrant in (True, False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
        model.train()
        _, recomputed = _loss_and_grads(model)
        model.gradient_checkpointing_disable()
        worst, name = _worst_grad_error(recomputed, plain)
        assert worst < FP32_MODEL_GRAD_TOL, f"{model_type} reentrant={reentrant} {name}: {worst:.2e}"


def test_frozen_weights_match_upstream(model_type):
    """Frozen hyper-connections, whose weights still carry the stream's gradient, and frozen embeddings, whose first
    layer has a constant residual."""
    model = _build(model_type, torch.float32)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not ("_hc." in name or "embed_tokens" in name))
    _assert_kernels_match_upstream(model_type, model)
    hc_grads = [name for name, parameter in model.named_parameters() if "_hc." in name and parameter.grad is not None]
    assert not hc_grads, hc_grads


def test_lora_matches_upstream(model_type):
    model = _build(model_type, torch.float32)
    target = next(
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and "self_attn" in name and "vision" not in name
    )
    torch.manual_seed(3)
    peft_model = get_peft_model(model, LoraConfig(r=_LORA_RANK, target_modules=[target]))
    _assert_kernels_match_upstream(model_type, peft_model)


def test_fsdp2_matches_a_plain_run(model_type):
    """The layer's pre-forward hook hands the hyper-connection plain parameters, the kernels' ``fn`` gradient lands on
    the parameter FSDP2 syncs, and a reshard between the forward and the backward refills what the graph saved."""
    _, plain = _loss_and_grads(_build(model_type, torch.float32))
    mesh = DeviceMesh("cuda", [0])
    for reshard in (False, True):
        model = _build(model_type, torch.float32)
        layers = model.model.language_model.layers if model_type == "glm5_next" else model.model.layers
        for layer in layers:
            fully_shard(layer, mesh=mesh, reshard_after_forward=reshard)
        fully_shard(model, mesh=mesh, reshard_after_forward=reshard)
        assert isinstance(layers[0].attn_hc.fn, DTensor)
        _, sharded = _loss_and_grads(model)
        worst, name = _worst_grad_error(sharded, plain)
        assert worst < FP32_MODEL_GRAD_TOL, f"{model_type} reshard={reshard} {name}: {worst:.2e}"


def test_autocast_error_is_no_worse_than_upstreams(model_type):
    """Under the bf16 autocast of a LoRA or ``fp32_non_ep_params`` run upstream's projection rounds to bf16 and the
    streams can hold another dtype than a block's output. Each against the same model with no autocast."""
    for dtype in (torch.float32, torch.bfloat16):
        model = _build(model_type, dtype, **FLIP_FREE)
        with _upstream_forwards(model_type):
            _, reference = _loss_and_grads(model)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, stock_grads = _loss_and_grads(model)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, grads = _loss_and_grads(model)
        kernel_error = _total_error(grads, reference)
        stock_error = _total_error(stock_grads, reference)
        assert kernel_error <= BF16_MODEL_ERROR_RATIO_MAX * stock_error, (dtype, kernel_error, stock_error)


def test_pipeline_stages_reproduce_the_unsplit_model(model_type):
    """Non-first stages run the family's stream forward, which calls the same decoder layers."""
    split = [2, 2] if model_type == "glm5_next" else [2, 1]
    model = _build(model_type, torch.float32).eval()
    ids = _ids(model)
    with torch.no_grad():
        reference = model(input_ids=ids).logits
        activation = ids
        for rank in range(len(split)):
            stage = build_pipeline_stage(
                _build(model_type, torch.float32), rank, len(split), moe_balancing="none", pp_split=split
            )
            activation = stage.eval()(activation)
    assert torch.equal(activation, reference), fro_rel_err(activation, reference)


@gpu_test_main(exact_world_size=1, prefix="hc_kernels", partial_state=False)
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    record_check(checks, "sinkhorn_matches_autograd", test_sinkhorn_matches_autograd)
    record_check(checks, "sinkhorn_reverses_near_zero_rows", test_sinkhorn_reverses_near_zero_rows)
    record_check(
        checks, "sinkhorn_saves_only_the_scales_with_a_graph", test_sinkhorn_saves_only_the_scales_with_a_graph
    )
    record_check(checks, "projection_matches_autograd", test_projection_matches_autograd)
    record_check(checks, "projection_reads_mixed_dtypes", test_projection_reads_mixed_dtypes)
    record_check(checks, "projection_takes_a_zero_stream", test_projection_takes_a_zero_stream)
    record_check(checks, "collapse_matches_autograd", test_collapse_matches_autograd)
    record_check(checks, "mix_matches_autograd", test_mix_matches_autograd)
    record_check(checks, "mix_promotes_like_upstream", test_mix_promotes_like_upstream)
    record_check(checks, "each_input_is_optional", test_each_input_is_optional)
    record_check(checks, "bf16_rounds_once", test_bf16_rounds_once)
    record_check(checks, "a_second_backward_reads_the_same_state", test_a_second_backward_reads_the_same_state)
    record_check(checks, "no_token_and_strided_inputs", test_no_token_and_strided_inputs)
    record_check(checks, "a_shape_the_streams_do_not_take_is_refused", test_a_shape_the_streams_do_not_take_is_refused)
    record_check(checks, "a_second_order_backward_is_refused", test_a_second_order_backward_is_refused)
    record_check(checks, "a_wrong_backward_fails", test_a_wrong_backward_fails)
    record_check(checks, "one_compile_serves_every_token_count", test_one_compile_serves_every_token_count)
    record_check(checks, "a_compiled_caller_runs_the_kernels_eagerly", test_a_compiled_caller_runs_the_kernels_eagerly)
    for model_type in ("glm5_next", "deepseek_v4"):
        apply_hyper_connection_patches(_build(model_type, torch.float32).config)
        record_check(
            checks,
            f"model_matches_upstream_at_fp32[{model_type}]",
            lambda: test_model_matches_upstream_at_fp32(model_type),
        )
        record_check(
            checks,
            f"model_bf16_error_is_no_worse_than_upstreams[{model_type}]",
            lambda: test_model_bf16_error_is_no_worse_than_upstreams(model_type),
        )
        record_check(
            checks,
            f"recompute_matches_a_plain_backward[{model_type}]",
            lambda: test_recompute_matches_a_plain_backward(model_type),
        )
        record_check(
            checks,
            f"frozen_weights_match_upstream[{model_type}]",
            lambda: test_frozen_weights_match_upstream(model_type),
        )
        record_check(checks, f"lora_matches_upstream[{model_type}]", lambda: test_lora_matches_upstream(model_type))
        record_check(
            checks,
            f"fsdp2_matches_a_plain_run[{model_type}]",
            lambda: test_fsdp2_matches_a_plain_run(model_type),
        )
        record_check(
            checks,
            f"autocast_error_is_no_worse_than_upstreams[{model_type}]",
            lambda: test_autocast_error_is_no_worse_than_upstreams(model_type),
        )
        record_check(
            checks,
            f"pipeline_stages_reproduce_the_unsplit_model[{model_type}]",
            lambda: test_pipeline_stages_reproduce_the_unsplit_model(model_type),
        )
    return {"checks": checks, "metrics": {}}


if __name__ == "__main__":
    run()
