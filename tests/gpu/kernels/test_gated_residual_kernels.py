#!/usr/bin/env python
"""Flash-Next's gated-residual kernels against an fp64 copy of upstream's module and upstream's own forward, forward and
backward, then the tiny model with the kernels on and off through packed and padded rows, LoRA and both checkpointing
modes, and the kernels' compile count."""

import contextlib
import copy
import os
from unittest import mock

import torch
import torch.nn.functional as F
import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen4_exp_modeling
from peft import LoraConfig, get_peft_model
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from halo.kernels import gated_residual as kernels
from halo.kernels import hyper_connection
from halo.kernels.gated_residual import gate_mix, grouped_rms_norm
from halo.model_loading.finalization import finalize_loaded_model
from halo.models import kernel_warmups
from halo.models.families.qwen4_exp import gated_residual as family
from halo.models.families.qwen4_exp.gated_residual import (
    inject_block,
    patch_qwen4_exp_gated_residual,
    warm_gated_residual,
)
from tests.common.harness import gpu_test_main, record_check
from tests.common.numerics import fro_rel_err
from tests.common.qwen4_exp import (
    load_tiny_cuda_model,
    tiny_qwen4_exp_text_config,
    training_batch,
)

KNOB = "HALO_GATED_RESIDUAL_KERNELS"
LOWRANK = 16
# A kernel output or gradient holds under this or ``STOCK_ERROR_RATIO`` times what upstream's form loses at the same
# dtype against fp64, whichever is larger.
FLOORS = {torch.float32: 2e-6, torch.bfloat16: 2e-3, torch.float16: 5e-4}
STOCK_ERROR_RATIO = 1.5
# What a wrongly computed block must miss the same bound by.
CONTROL_MARGIN = 10.0
# Streams and hidden width: a multiple of the 512 tile, a width off it, one past it, and a stream count off a power of two.
SHAPES = ((4, 256), (4, 300), (2, 1100), (3, 70))
# Batch and tokens, down to one row.
ROW_SHAPES = ((1, 1), (2, 19), (1, 130))
MODEL_FP32_GRAD_TOL = 1e-4
MODEL_FP32_LOSS_TOL = 1e-5
_LORA_RANK = 4
_KERNEL_FUNCTIONS = (
    kernels._norm_fwd_kernel,
    kernels._norm_bwd_kernel,
    kernels._mix_fwd_kernel,
    kernels._mix_bwd_kernel,
    hyper_connection._inject_fwd_kernel,
    hyper_connection._inject_bwd_kernel,
)


@contextlib.contextmanager
def knob(value: str):
    previous = os.environ.get(KNOB)
    os.environ[KNOB] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(KNOB)
        else:
            os.environ[KNOB] = previous


def _residual(streams: int, hidden: int, dtype: torch.dtype, *, trained: bool = True) -> torch.nn.Module:
    config = tiny_qwen4_exp_text_config(hc_count=streams, hidden_size=hidden, hc_lowrank=LOWRANK)
    torch.manual_seed(streams * hidden)
    module = qwen4_exp_modeling.Qwen4ExpTextGatedResidual(config).cuda()
    with torch.no_grad():
        module.hc_norm.weight.normal_(std=0.3)
    return module.to(dtype).requires_grad_(trained)


def _block_output(module: torch.nn.Module, x: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
    mixed, stream, gates = module(x)
    return inject_block(stream, mixed + block, gates)


def _vjp(module: torch.nn.Module, x: torch.Tensor, block: torch.Tensor, mode: str):
    """Output, input gradient and parameter gradients of one gated residual and injection for a fixed cotangent."""
    module = copy.deepcopy(module)
    module.zero_grad()
    x, block = x.detach().clone().requires_grad_(), block.detach().clone()
    with knob(mode):
        out = _block_output(module, x, block)
    cotangent = torch.randn(out.shape, generator=torch.Generator(device="cuda").manual_seed(1), device="cuda")
    out.backward(cotangent.to(out.dtype))
    grads = {name: p.grad for name, p in module.named_parameters() if p.grad is not None}
    return out.detach(), x.grad, grads


def _inputs(streams: int, hidden: int, rows: tuple[int, int], dtype: torch.dtype):
    generator = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(*rows, streams * hidden, generator=generator, device="cuda")
    x[0, 0] = 0  # A row of zeros, as an all-padding position's stream.
    block = torch.randn(*rows, hidden, generator=generator, device="cuda")
    return x.to(dtype), block.to(dtype)


def _worst(candidate, reference) -> float:
    out, dx, grads = candidate
    ref_out, ref_dx, ref_grads = reference
    assert grads.keys() == ref_grads.keys()
    return max(
        fro_rel_err(tensor, ref)
        for tensor, ref in [(out, ref_out), (dx, ref_dx), *((grads[n], ref_grads[n]) for n in grads)]
        if ref.norm() > 0
    )


def _compare(streams, hidden, rows, dtype, trained=True):
    module = _residual(streams, hidden, dtype, trained=trained)
    x, block = _inputs(streams, hidden, rows, dtype)
    reference = _vjp(copy.deepcopy(module).double(), x.double(), block.double(), "0")
    stock = _vjp(module, x, block, "0")
    fused = _vjp(module, x, block, "1")
    return _worst(fused, reference), _worst(stock, reference), (module, x, block, reference)


def test_block_matches_fp64_no_worse_than_upstream(dtype):
    for streams, hidden in SHAPES:
        for rows in ROW_SHAPES:
            fused, stock, _ = _compare(streams, hidden, rows, dtype)
            bound = max(FLOORS[dtype], STOCK_ERROR_RATIO * stock)
            assert fused <= bound, f"S={streams} H={hidden} rows={rows} {dtype}: {fused:.2e} > {bound:.2e}"


def test_the_kernel_path_runs_for_trained_and_frozen_weights():
    for trained in (True, False):
        module = _residual(4, 256, torch.bfloat16, trained=trained)
        x, block = _inputs(4, 256, (2, 19), torch.bfloat16)
        with mock.patch.object(family, "grouped_rms_norm", wraps=grouped_rms_norm) as norm:
            _, _, grads = _vjp(module, x, block, "1")
        assert norm.call_count == 1, f"trained={trained}: the kernel path did not run"
        assert bool(grads) == trained
        assert all(g.abs().sum() > 0 for g in grads.values()), "a weight gradient is zero"


def test_modules_the_kernels_cannot_serve_run_upstream():
    module = _residual(4, 256, torch.bfloat16)
    x, _ = _inputs(4, 256, (2, 19), torch.bfloat16)
    hooked = copy.deepcopy(module)
    hooked.input_mix_weight_up.register_forward_hook(lambda *_: None)
    mismatched = copy.deepcopy(module)
    mismatched.input_mix_weight_up.float()
    for name, candidate in (("hook", hooked), ("dtype", mismatched)):
        with mock.patch.object(family, "grouped_rms_norm", wraps=grouped_rms_norm) as norm, knob("1"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                candidate(x)
        assert norm.call_count == 0, f"{name}: the kernels ran"
    with mock.patch.object(family, "grouped_rms_norm", wraps=grouped_rms_norm) as norm, knob("1"):
        module.cpu()(x.cpu())
        assert norm.call_count == 0, "a CPU input ran the kernels"


def test_negative_controls_fail_the_comparison():
    streams, hidden, rows, dtype = 4, 256, (2, 19), torch.bfloat16
    fused, stock, (module, x, block, reference) = _compare(streams, hidden, rows, dtype)
    bound = max(FLOORS[dtype], STOCK_ERROR_RATIO * stock)
    assert fused <= bound
    flipped = copy.deepcopy(module)
    with torch.no_grad():
        flipped.input_mix_weight_up.weight.neg_()
    wrong = _worst(_vjp(flipped, x, block, "1"), reference)
    assert wrong > CONTROL_MARGIN * bound, f"a flipped mix gate scored {wrong:.2e} against {bound:.2e}"

    def lost_gradients(ctx, grad_out):
        return (*(torch.zeros_like(saved) for saved in ctx.saved_tensors), None)

    with mock.patch.object(kernels._GateMix, "backward", lost_gradients):
        broken = _worst(_vjp(module, x, block, "1"), reference)
    assert broken > CONTROL_MARGIN * bound, f"a lost backward scored {broken:.2e} against {bound:.2e}"


def test_runs_repeat_bitwise():
    module = _residual(4, 300, torch.bfloat16)
    x, block = _inputs(4, 300, (2, 130), torch.bfloat16)
    first, second = _vjp(module, x, block, "1"), _vjp(module, x, block, "1")
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    assert all(torch.equal(first[2][name], second[2][name]) for name in first[2])


def _compiled_binaries() -> int:
    device = torch.cuda.current_device()
    return sum(len(fn.device_caches[device][0]) for fn in _KERNEL_FUNCTIONS)


def _clear_kernel_caches() -> None:
    for fn in _KERNEL_FUNCTIONS:
        fn.device_caches.clear()


# Triton specializes an integer argument on divisibility by 16, so 512 and 4096 are in beside the odd counts.
ROW_COUNTS = ((1, 1), (1, 37), (1, 512), (3, 513), (1, 2049), (1, 4096), (2, 4100))


def test_no_compile_after_the_warm_up_for_any_row_count():
    for dtype in (torch.bfloat16, torch.float32):
        module = _residual(4, 300, dtype)
        _clear_kernel_caches()
        assert _compiled_binaries() == 0
        with knob("1"):
            warm_gated_residual(torch.nn.ModuleList([module]))
        warm = _compiled_binaries()
        assert warm > 0, f"{dtype}: the warm-up compiled nothing"
        for rows in ROW_COUNTS:
            _vjp(module, *_inputs(4, 300, rows, dtype), "1")
        assert _compiled_binaries() == warm, f"{dtype}: {_compiled_binaries() - warm} binaries compiled after warm-up"
    for fn in _KERNEL_FUNCTIONS:
        sizes = [p for p, name in zip(fn.params, fn.arg_names, strict=True) if name in ("hidden", "width")]
        assert sizes and not any(p.is_constexpr for p in sizes), f"{fn.__name__} compiles per size"


def test_a_removed_warm_up_is_caught_by_the_count():
    """Negative control: with no warm-up the first steps compile, so the count above would fail."""
    module = _residual(4, 300, torch.bfloat16)
    _clear_kernel_caches()
    _vjp(module, *_inputs(4, 300, (1, 37), torch.bfloat16), "1")
    assert _compiled_binaries() > 0


def test_finalizing_a_loaded_model_leaves_nothing_to_compile_in_a_step(model):
    kernel_warmups.register_kernel_warmup(warm_gated_residual)
    _clear_kernel_caches()
    with knob("1"):
        finalize_loaded_model(model)
    warm = _compiled_binaries()
    assert warm > 0, "finalizing warmed no kernel"
    _loss_and_grads(model, training_batch(model, "packed"), "1")
    assert _compiled_binaries() == warm, f"{_compiled_binaries() - warm} binaries compiled after finalizing"


def test_inject_streams_is_mix_streams_at_the_identity_in_fp32():
    """The two kernel pairs differ only in the dtype rounding of the product, which fp32 does not have."""
    streams, hidden, rows = 4, 300, 19
    generator = torch.Generator(device="cuda").manual_seed(5)
    block, residual, gate, upstream = (
        torch.randn(shape, generator=generator, device="cuda")
        for shape in ((rows, hidden), (rows, streams, hidden), (rows, streams), (rows, streams, hidden))
    )
    identity = torch.eye(streams, device="cuda").expand(rows, streams, streams).contiguous()
    results = []
    for mix in (
        lambda b, r, g: hyper_connection.inject_streams(b, r, g),
        lambda b, r, g: hyper_connection.mix_streams(b, r, g, identity),
    ):
        leaves = [t.clone().requires_grad_() for t in (block, residual, gate)]
        out = mix(*leaves)
        results.append((out, *torch.autograd.grad(out, leaves, upstream)))
    for inject, mixed in zip(*results, strict=True):
        assert fro_rel_err(inject, mixed) < 1e-6


def test_functions_match_their_eager_forms():
    x, _ = _inputs(4, 300, (2, 19), torch.float32)
    rows = x.view(-1, 4 * 300)
    weight = torch.randn(4 * 300, device="cuda") * 0.1
    grouped = rows.view(-1, 4, 300)
    expected = grouped * torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + weight.view(4, 300))
    assert fro_rel_err(grouped_rms_norm(rows, weight, 1e-6, 4).view(-1, 4, 300), expected) < 1e-5
    down, w_up = torch.randn(rows.shape[0], LOWRANK, device="cuda"), torch.randn(4 * 300, LOWRANK, device="cuda")
    up = F.linear(F.silu(down / 4), w_up).view(-1, 4, 300)
    expected = (torch.sigmoid(up) * rows.view(-1, 4, 300)).mean(-2)
    assert fro_rel_err(gate_mix(rows, down, w_up, 4), expected) < 1e-5


def _residual_counts(model) -> tuple[int, int]:
    """Gated residuals of the model, and how many of them sit in decoder layers (those a checkpoint recomputes)."""
    names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, qwen4_exp_modeling.Qwen4ExpTextGatedResidual)
    ]
    return len(names), sum(".layers." in name for name in names)


def _loss_and_grads(model, batch: dict, mode: str, *, recomputed: bool = False):
    """Loss and gradients of one step. In mode ``1`` every gated residual must have run the kernels (a decoder layer's
    again in its checkpoint's recompute when ``recomputed``), so a silent fallback cannot make the modes agree."""
    model.zero_grad()
    with knob(mode), mock.patch.object(family, "grouped_rms_norm", wraps=grouped_rms_norm) as norm:
        loss = model(use_cache=False, **batch).loss
        loss.backward()
    if mode == "1":
        total, in_layers = _residual_counts(model)
        expected = total + (in_layers if recomputed else 0)
        assert norm.call_count == expected, f"{norm.call_count} of {expected} gated residual runs used the kernels"
    grads = {n: p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad for n, p in model.named_parameters()}
    return loss.item(), {n: g.detach().clone() for n, g in grads.items() if g is not None}


def _assert_models_agree(model, batch: dict, label: str) -> None:
    stock_loss, stock = _loss_and_grads(model, batch, "0")
    loss, grads = _loss_and_grads(model, batch, "1")
    assert abs(loss - stock_loss) <= MODEL_FP32_LOSS_TOL * abs(stock_loss), (label, loss, stock_loss)
    assert grads.keys() == stock.keys()
    worst = max(fro_rel_err(grads[n], g) for n, g in stock.items() if g.norm() > 0)
    assert worst < MODEL_FP32_GRAD_TOL, f"{label}: {worst:.2e}"


def test_model_matches_upstream_on_packed_and_paddedtraining_batch(model, kind):
    _assert_models_agree(model, training_batch(model, kind), kind)


def test_lora_gradients_flow_through_the_kernels(model):
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))
    ]
    torch.manual_seed(3)
    peft_model = get_peft_model(copy.deepcopy(model), LoraConfig(r=_LORA_RANK, target_modules=targets))
    batch = training_batch(peft_model, "packed")
    with knob("1"):
        _, grads = _loss_and_grads(peft_model, batch, "1")
    adapters = {n: g for n, g in grads.items() if "lora_" in n}
    assert adapters and any("lora_A" in n for n in adapters) and any(g.abs().sum() > 0 for g in adapters.values())
    _assert_models_agree(peft_model, batch, "lora")


def test_checkpointing_matches_the_plain_backward(model, reentrant):
    batch = training_batch(model, "packed")
    _, plain = _loss_and_grads(model, batch, "1")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
    model.train()
    try:
        _, checkpointed = _loss_and_grads(model, batch, "1", recomputed=True)
    finally:
        model.gradient_checkpointing_disable()
    assert checkpointed.keys() == plain.keys()
    worst = max(fro_rel_err(checkpointed[n], g) for n, g in plain.items() if g.norm() > 0)
    assert worst < MODEL_FP32_GRAD_TOL, f"reentrant={reentrant}: {worst:.2e}"


def test_fsdp2_matches_a_plain_run(model):
    """The layer's pre-forward hook hands the gated residual plain parameters, so the kernels see no DTensor, and the
    gradients land on the shards FSDP2 syncs; a reshard between forward and backward refills what the graph saved."""
    batch = training_batch(model, "packed")
    _, plain = _loss_and_grads(model, batch, "1")
    mesh = DeviceMesh("cuda", [0])
    for reshard in (False, True):
        sharded = copy.deepcopy(model)
        for layer in sharded.model.layers:
            fully_shard(layer, mesh=mesh, reshard_after_forward=reshard)
        fully_shard(sharded, mesh=mesh, reshard_after_forward=reshard)
        assert isinstance(sharded.model.layers[0].attn_hyper_connection.hc_norm.weight, DTensor)
        _, grads = _loss_and_grads(sharded, batch, "1")
        worst = max(fro_rel_err(grads[n], g) for n, g in plain.items() if g.norm() > 0)
        assert worst < MODEL_FP32_GRAD_TOL, f"reshard={reshard}: {worst:.2e}"


@gpu_test_main(exact_world_size=1, prefix="gated_residual_kernels", partial_state=False)
def run(ctx) -> dict:
    patch_qwen4_exp_gated_residual()
    checks: dict[str, bool] = {}
    for dtype in FLOORS:
        record_check(checks, f"matches_fp64[{dtype}]", lambda: test_block_matches_fp64_no_worse_than_upstream(dtype))
    record_check(checks, "trained_and_frozen", test_the_kernel_path_runs_for_trained_and_frozen_weights)
    record_check(checks, "unserved_modules", test_modules_the_kernels_cannot_serve_run_upstream)
    record_check(checks, "negative_controls", test_negative_controls_fail_the_comparison)
    record_check(checks, "repeatable", test_runs_repeat_bitwise)
    record_check(checks, "eager_forms", test_functions_match_their_eager_forms)
    record_check(checks, "no_compile_after_warm_up", test_no_compile_after_the_warm_up_for_any_row_count)
    record_check(checks, "no_warm_up_compiles_in_a_step", test_a_removed_warm_up_is_caught_by_the_count)
    record_check(checks, "inject_is_mix_at_identity", test_inject_streams_is_mix_streams_at_the_identity_in_fp32)
    model = load_tiny_cuda_model(torch.float32)
    for kind in ("packed", "padded"):
        record_check(
            checks,
            f"model[{kind}]",
            lambda: test_model_matches_upstream_on_packed_and_paddedtraining_batch(model, kind),
        )
    record_check(checks, "lora", lambda: test_lora_gradients_flow_through_the_kernels(model))
    for reentrant in (True, False):
        record_check(
            checks,
            f"checkpointing[reentrant={reentrant}]",
            lambda: test_checkpointing_matches_the_plain_backward(model, reentrant),
        )
    record_check(checks, "fsdp2", lambda: test_fsdp2_matches_a_plain_run(model))
    record_check(
        checks, "finalizing_warms", lambda: test_finalizing_a_loaded_model_leaves_nothing_to_compile_in_a_step(model)
    )
    return {"checks": checks}


if __name__ == "__main__":
    run()
