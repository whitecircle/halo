#!/usr/bin/env python
"""Muon optimizer (Newton-Schulz orthogonalization) on one GPU: the fused Triton step end to end.

Verifies that create_muon_optimizer:
1. Correctly splits 2D+ params (Muon) vs 1D params (AdamW scalar)
2. Achieves loss descent on a simple regression task
3. Handles models with only 2D params (no scalar optimizer)
4. Applies weight decay exactly, and only to the named decay params
5. Round-trips its state dict
6. Writes an orthogonalized update for every matrix the fused step batches, from the Nesterov
   momentum input on the step after

Each property is recorded as its own check, so one failure does not hide the others.

Run: torchrun --nproc_per_node=1 tests/gpu/optimizers/test_muon.py
"""

import copy
import math

import torch
import torch.nn as nn
from gram_newton_schulz.muon.muon_utils.muon_opt_utils import adjust_lr_rms_norm

from src.optimizers.muon import _GNS_CHUNK_SIZE, _muon_kernels_available, create_muon_optimizer
from tests.common.harness import gpu_test_main, record_check
from tests.common.utils import assert_optimizer_state_bit_exact, assert_orthogonalized, log, log_spectrum_matrix

# ─── Models ──────────────────────────────────────────────────────────────────

HIDDEN = 256
BATCH = 32
SEQ = 64
NUM_STEPS = 50
# Same-shape matrices one past the orthogonalization chunk, so the step stacks them in two chunks.
LAYERS_PAST_CHUNK = _GNS_CHUNK_SIZE + 1
# One bf16 write (stochastic or nearest rounding) lands within one ULP of the exact value, and a bf16
# ULP is at most eps of the value.
BF16_WRITE_RTOL = torch.finfo(torch.bfloat16).eps
# gram_newton_schulz runs its CUDA kernels from compute capability 9.0.
KERNEL_MIN_MAJOR = 9


class FFNModel(nn.Module):
    """Small FFN with both 2D (Linear.weight) and 1D (bias, LayerNorm) params."""

    def __init__(self, hidden=HIDDEN, layers=3):
        super().__init__()
        blocks = []
        for _ in range(layers):
            blocks.extend([nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU()])
        blocks.append(nn.Linear(hidden, hidden))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class NoBiasModel(nn.Module):
    """Model with only 2D params (no bias, no layernorm)."""

    def __init__(self, hidden=HIDDEN, layers=3):
        super().__init__()
        blocks = []
        for _ in range(layers):
            blocks.extend([nn.Linear(hidden, hidden, bias=False), nn.GELU()])
        blocks.append(nn.Linear(hidden, hidden, bias=False))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class ZeroMatrices(nn.Module):
    """Zero bf16 matrices in the shapes the fused step batches differently.

    ``wide`` has a smaller side above 256, the size from which Newton-Schulz dispatches to the CUDA
    kernels; the ``LAYERS_PAST_CHUNK`` same-shape ``layers`` span two orthogonalization chunks;
    ``experts`` is a 3-D stack behind 2-D params, the order the factory sorts a MoE model's matrices
    into, so it reaches Newton-Schulz whole as a batch of per-expert matrices. Its experts share one
    scale, so per-expert normalization is pinned on CPU, not here.
    """

    def __init__(self):
        super().__init__()
        self.wide = nn.Parameter(torch.zeros(320, 640, dtype=torch.bfloat16))
        self.layers = nn.ParameterList(
            nn.Parameter(torch.zeros(64, 96, dtype=torch.bfloat16)) for _ in range(LAYERS_PAST_CHUNK)
        )
        self.experts = nn.Parameter(torch.zeros(8, 96, 64, dtype=torch.bfloat16))


def run_training(model, optimizer, num_steps=NUM_STEPS, hidden=HIDDEN):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    x = torch.randn(BATCH, SEQ, hidden, device=device, dtype=dtype)
    y = torch.randn(BATCH, SEQ, hidden, device=device, dtype=dtype)
    losses = []

    for _ in range(num_steps):
        optimizer.zero_grad()
        out = model(x)
        loss = nn.functional.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def _fill_log_spectrum_grads(model: nn.Module, generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Give every matrix a fresh log-spectrum gradient; return them by parameter name, in fp64."""
    grads = {}
    for name, p in model.named_parameters():
        rows, cols = p.shape[-2:]
        matrices = [log_spectrum_matrix(rows, cols, generator) for _ in range(math.prod(p.shape[:-2]))]
        p.grad = torch.stack(matrices).view(p.shape).to(device=p.device, dtype=p.dtype)
        grads[name] = p.grad.double()
    return grads


def _assert_updates_from_zero(model: nn.Module, lr: float, sources: dict[str, torch.Tensor], step: int) -> int:
    """Each matrix a step wrote from zero weights is ``-adjusted_lr`` times the orthogonalized source."""
    checked = 0
    for name, p in model.named_parameters():
        rows, cols = p.shape[-2:]
        update = -p.detach().double() / adjust_lr_rms_norm(lr, p.shape)
        for index, (matrix, source) in enumerate(
            zip(update.view(-1, rows, cols), sources[name].view(-1, rows, cols), strict=True)
        ):
            assert_orthogonalized(matrix, source, f"step {step} {name}[{index}]")
            checked += 1
    return checked


# ─── Checks ──────────────────────────────────────────────────────────────────


def check_param_split():
    """Muon should put 2D+ params in Muon groups and 1D params in scalar AdamW."""
    model = FFNModel().cuda().to(torch.bfloat16)
    optimizer = create_muon_optimizer(model, lr=3e-4)

    muon_params = []
    for g in optimizer._muon_param_groups:
        muon_params.extend(g["params"])

    scalar_params = []
    assert optimizer.scalar_optimizer is not None, "FFN with bias/LN should have scalar optimizer"
    for g in optimizer.scalar_optimizer.param_groups:
        scalar_params.extend(g["params"])

    for name, p in model.named_parameters():
        if p.ndim >= 2:
            assert any(p is mp for mp in muon_params), f"{name} (ndim={p.ndim}) missing from Muon groups"
        else:
            assert any(p is sp for sp in scalar_params), f"{name} (ndim={p.ndim}) missing from scalar groups"

    muon_count = sum(p.numel() for p in muon_params)
    scalar_count = sum(p.numel() for p in scalar_params)
    log(f"  Muon params: {muon_count / 1e3:.1f}K, Scalar params: {scalar_count / 1e3:.1f}K")


def check_loss_descends():
    """Muon should decrease loss on a regression task."""
    torch.manual_seed(42)
    model = FFNModel().cuda().to(torch.bfloat16)
    optimizer = create_muon_optimizer(model, lr=2e-3)

    losses = run_training(model, optimizer, num_steps=100)

    assert not math.isnan(losses[-1]), "Loss is NaN"
    assert losses[-1] < losses[0], f"Loss should decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    reduction = (losses[0] - losses[-1]) / losses[0]
    log(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} ({reduction * 100:.1f}% reduction)")
    assert reduction > 0.1, f"Expected >10% loss reduction, got {reduction * 100:.1f}%"


def check_loss_descends_no_bias():
    """Muon should work on a model with only 2D params (no scalar optimizer)."""
    torch.manual_seed(42)
    model = NoBiasModel().cuda().to(torch.bfloat16)
    optimizer = create_muon_optimizer(model, lr=2e-3)

    assert optimizer.scalar_optimizer is None, "No 1D params, scalar optimizer should be None"

    losses = run_training(model, optimizer, num_steps=100)

    assert not math.isnan(losses[-1]), "Loss is NaN"
    assert losses[-1] < losses[0], f"Loss should decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    reduction = (losses[0] - losses[-1]) / losses[0]
    log(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} ({reduction * 100:.1f}% reduction)")


def check_decay_parameter_names():
    """The fused step decays exactly the named Muon matrices.

    A zero-gradient step isolates the decay term, since Newton-Schulz maps a zero matrix to zero: the
    named matrix shrinks by ``1 - lr * weight_decay`` (within one stochastic-rounding bf16 write) and
    the unnamed one is written back bit-identical.
    """
    torch.manual_seed(42)
    model = NoBiasModel(hidden=64, layers=1).cuda().to(torch.bfloat16)
    decayed, undecayed = model.net[0].weight, model.net[2].weight
    lr, weight_decay = 0.1, 0.5
    optimizer = create_muon_optimizer(model, lr=lr, weight_decay=weight_decay, decay_parameters={"net.0.weight"})

    groups = {id(p): g["weight_decay"] for g in optimizer._muon_param_groups for p in g["params"]}
    assert groups == {id(decayed): weight_decay, id(undecayed): 0.0}, f"Muon groups misrouted decay: {groups}"

    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()

    assert torch.equal(undecayed, before["net.2.weight"]), "an undecayed matrix moved on a zero-gradient step"
    expected = before["net.0.weight"].float() * (1 - lr * weight_decay)
    assert torch.allclose(decayed.float(), expected, rtol=BF16_WRITE_RTOL, atol=0), (
        f"decayed matrix off 1 - lr*wd = {1 - lr * weight_decay}: max relative error "
        f"{((decayed.float() - expected).abs() / expected.abs().clamp_min(1e-30)).max().item():.3e}"
    )


def check_state_dict_roundtrip():
    """Train k steps -> deepcopy state_dict -> fresh optimizer built with DIFFERENT hyperparams ->
    load: state tensors bit-exact (Muon momentum AND the scalar AdamW moments), restored group
    hyperparams follow the CHECKPOINT rather than the fresh constructor (``Muon.load_state_dict``
    copies them onto the live group dicts, which upstream's ``param_groups`` property shadows from
    the base restore), and a post-load step still moves weights."""
    torch.manual_seed(42)
    model = FFNModel().cuda().to(torch.bfloat16)
    optimizer = create_muon_optimizer(model, lr=2e-3, weight_decay=0.1, momentum=0.95)
    run_training(model, optimizer, num_steps=5)
    saved = copy.deepcopy(optimizer.state_dict())

    torch.manual_seed(7)
    model2 = FFNModel().cuda().to(torch.bfloat16)
    optimizer2 = create_muon_optimizer(model2, lr=5e-1, weight_decay=0.0, momentum=0.5)
    # Premise: the fresh constructor's hyperparams differ from the checkpoint's, otherwise the
    # follows-the-checkpoint assertion below could not tell the two apart.
    for live, saved_group in zip(optimizer2.param_groups, saved["param_groups"], strict=True):
        assert live["lr"] != saved_group["lr"] and live["weight_decay"] != saved_group["weight_decay"]

    optimizer2.load_state_dict(copy.deepcopy(saved))
    assert_optimizer_state_bit_exact(saved, optimizer2.state_dict())

    for i, (live, saved_group) in enumerate(zip(optimizer2.param_groups, saved["param_groups"], strict=True)):
        for key in ("lr", "momentum", "weight_decay"):
            if key in saved_group:
                assert live[key] == saved_group[key], (
                    f"group {i} '{key}': {live[key]} != saved {saved_group[key]} — restored "
                    f"hyperparams must follow the checkpoint, not the fresh constructor"
                )

    before = [p.detach().clone() for p in model2.parameters()]
    run_training(model2, optimizer2, num_steps=1)
    moved = any(not torch.equal(b, p.detach()) for b, p in zip(before, model2.parameters(), strict=False))
    assert moved, "post-load step did not move any weight"


def check_step_writes_orthogonalized_update():
    """Two steps from zero weights each write ``-adjusted_lr * NS(u)``, orthogonal per matrix.

    With zero weights and no decay the fused write leaves ``-p = adjusted_lr * NS(u)``, with
    ``adjusted_lr`` the rms-norm scale the step applies and ``u`` the Nesterov input
    ``momentum * m + g`` (Newton-Schulz ignores its scale). On the first step the buffer ``m`` is
    ``g1``; the second step, from zeroed weights on a fresh ``g2``, reads ``m = momentum * g1 + g2``.
    Each matrix of ``-p / adjusted_lr`` must sit in the Newton-Schulz band and point along its own
    ``u``'s polar factor, which a mis-ordered restack across chunks or a dropped learning-rate scale
    breaks on either step, and a dropped Nesterov term (cosine at most 0.94) or momentum (0.83) on the
    second. The buffer itself pins the momentum decay. On SM90+ the step must run ``wide`` through the
    CUDA kernels, so this covers the kernel backend rather than only the torch one.
    tests/cpu/optimizers/test_muon_orthogonalization.py covers the orthogonalizer alone on CPU.
    """
    generator = torch.Generator().manual_seed(0)
    model = ZeroMatrices().cuda()
    lr = 1e-2
    optimizer = create_muon_optimizer(model, lr=lr, weight_decay=0.0)
    momentum = optimizer.param_groups[0]["momentum"]

    first = _fill_log_spectrum_grads(model, generator)
    kernel_backend = optimizer.newton_schulz._kernel_backend
    if torch.cuda.get_device_capability()[0] < KERNEL_MIN_MAJOR:
        log("  Newton-Schulz: no CUDA kernels below SM90, torch backend only")
        optimizer.step()
    else:
        # Both images ship quack, so a failed probe here is a broken kernel stack, not a fallback.
        assert _muon_kernels_available(), "the CUDA Newton-Schulz kernels do not run on this SM90+ device"
        assert kernel_backend is not None, "the CUDA Newton-Schulz kernels run but the optimizer did not enable them"
        kernel_inputs = []
        kernel_sym_mm = kernel_backend.sym_mm

        def recording_sym_mm(a, b):
            kernel_inputs.append(tuple(a.shape))
            return kernel_sym_mm(a, b)

        kernel_backend.sym_mm = recording_sym_mm
        try:
            optimizer.step()
        finally:
            kernel_backend.sym_mm = kernel_sym_mm
        assert kernel_inputs, "no matrix reached the CUDA Newton-Schulz kernels during the step"
        log(f"  Newton-Schulz CUDA kernels ran on {sorted(set(kernel_inputs))}")
    checked = _assert_updates_from_zero(model, lr, first, step=1)

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    second = _fill_log_spectrum_grads(model, generator)
    optimizer.step()
    for name, p in model.named_parameters():
        buffer = optimizer.state[p]["momentum"].double()
        error = (buffer - (momentum * first[name] + second[name])).abs()
        bound = BF16_WRITE_RTOL * (momentum * first[name].abs() + second[name].abs())
        assert (error <= bound).all(), f"{name}: the momentum buffer is not momentum * g1 + g2"
    nesterov = {name: momentum * (momentum * first[name] + second[name]) + second[name] for name in first}
    checked += _assert_updates_from_zero(model, lr, nesterov, step=2)
    log(f"  {checked} matrix updates orthogonal over two steps")


# ─── Main ────────────────────────────────────────────────────────────────────


@gpu_test_main(exact_world_size=1, prefix="test_muon", partial_state=False)
def run(ctx) -> dict:
    log(f"GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    checks: dict[str, bool] = {}
    record_check(checks, "param_split", check_param_split)
    record_check(checks, "loss_descends", check_loss_descends)
    record_check(checks, "loss_descends_no_bias", check_loss_descends_no_bias)
    record_check(checks, "decay_parameter_names", check_decay_parameter_names)
    record_check(checks, "state_dict_roundtrip", check_state_dict_roundtrip)
    record_check(checks, "step_writes_orthogonalized_update", check_step_writes_orthogonalized_update)
    return {"checks": checks}


if __name__ == "__main__":
    run()
