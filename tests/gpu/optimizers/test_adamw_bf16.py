#!/usr/bin/env python
"""
Test: AdamWBF16 optimizer with stochastic rounding.

Verifies that AdamWBF16:
1. Keeps weights in bf16, states in bf16
2. Achieves loss comparable to fp32 master weights (not stale like pure bf16)
3. Handles mixed dtype models (bf16 + fp32 params)

Usage (single GPU, no torchrun needed):
    python tests/gpu/optimizers/test_adamw_bf16.py
"""

import math

import torch
import torch.nn as nn

from src.kernels.lowp.quantization import cached_fake_quant
from src.optimizers.adamw_bf16 import AdamWBF16
from src.optimizers.flash_adamw import create_flash_adamw_optimizer
from src.optimizers.muon import create_muon_optimizer

# ─── Model ───────────────────────────────────────────────────────────────────


class FFNBlock(nn.Module):
    def __init__(self, hidden_size=1024, intermediate_size=4096):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.norm(x)
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class SimpleModel(nn.Module):
    def __init__(self, num_layers=4, hidden_size=1024, intermediate_size=4096):
        super().__init__()
        self.layers = nn.ModuleList([FFNBlock(hidden_size, intermediate_size) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def create_model(dtype, device="cuda"):
    torch.manual_seed(42)
    model = SimpleModel(num_layers=4, hidden_size=1024, intermediate_size=4096)
    return model.to(device=device, dtype=dtype)


# ─── Metrics ─────────────────────────────────────────────────────────────────


def count_stale(model, old_params):
    stale = total = 0
    for p, old_p in zip(model.parameters(), old_params, strict=False):
        if p.requires_grad:
            stale += (p.data == old_p).sum().item()
            total += p.numel()
    return stale / total if total > 0 else 0


def snapshot_params(model):
    return [p.data.clone() for p in model.parameters()]


# ─── Training loop ───────────────────────────────────────────────────────────

HIDDEN_SIZE = 1024
BATCH_SIZE = 8
SEQ_LEN = 128
NUM_STEPS = 200


def run_training(model, optimizer, num_steps=NUM_STEPS, use_autocast=False):
    device = next(model.parameters()).device
    losses, stales = [], []

    for step in range(num_steps):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
        target = torch.randn_like(x)

        old = snapshot_params(model)
        optimizer.zero_grad()

        if use_autocast:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = nn.functional.mse_loss(out, target)
        else:
            out = model(x)
            loss = nn.functional.mse_loss(out.float(), target.float())

        loss.backward()
        optimizer.step()

        stales.append(count_stale(model, old))
        losses.append(loss.item())

    return losses, stales


# ─── Tests ───────────────────────────────────────────────────────────────────


def test_dtypes():
    """Verify that AdamWBF16 keeps weights in bf16 and states in bf16."""
    print("TEST 1: Dtype verification")
    model = create_model(torch.bfloat16)
    opt = AdamWBF16(model.parameters(), lr=1e-4)

    # Run one step to initialize states
    x = torch.randn(2, 16, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    loss = model(x).mean()
    loss.backward()
    opt.step()

    # Check param dtypes
    for name, p in model.named_parameters():
        assert p.dtype == torch.bfloat16, f"Param {name} is {p.dtype}, expected bf16"

    # Check state dtypes
    for p in model.parameters():
        state = opt.state[p]
        assert state["exp_avg"].dtype == torch.bfloat16, f"exp_avg is {state['exp_avg'].dtype}, expected bf16"
        assert state["exp_avg_sq"].dtype == torch.bfloat16, f"exp_avg_sq is {state['exp_avg_sq'].dtype}, expected bf16"

    print("  PASSED: All params bf16, all states bf16")


def test_vs_baselines():
    """Compare AdamWBF16 against pure bf16 (stale) and fp32 master (gold standard)."""
    print("\nTEST 2: Loss comparison vs baselines")

    # Pure bf16 baseline (stale weights)
    model_bf16 = create_model(torch.bfloat16)
    opt_bf16 = torch.optim.AdamW(model_bf16.parameters(), lr=1e-4, fused=True)
    losses_bf16, stales_bf16 = run_training(model_bf16, opt_bf16)

    # FP32 master + autocast (gold standard)
    model_fp32 = create_model(torch.float32)
    opt_fp32 = torch.optim.AdamW(model_fp32.parameters(), lr=1e-4, fused=True)
    losses_fp32, stales_fp32 = run_training(model_fp32, opt_fp32, use_autocast=True)

    # AdamWBF16 with SR
    model_sr = create_model(torch.bfloat16)
    opt_sr = AdamWBF16(model_sr.parameters(), lr=1e-4)
    losses_sr, stales_sr = run_training(model_sr, opt_sr)

    print(f"  Pure BF16:      loss={losses_bf16[-1]:.6f}  stale%={stales_bf16[-1]:.4f}")
    print(f"  FP32 master:    loss={losses_fp32[-1]:.6f}  stale%={stales_fp32[-1]:.4f}")
    print(f"  AdamWBF16 (SR): loss={losses_sr[-1]:.6f}  stale%={stales_sr[-1]:.4f}")

    # SR should be much closer to fp32 than to pure bf16
    gap_bf16 = abs(losses_bf16[-1] - losses_fp32[-1])
    gap_sr = abs(losses_sr[-1] - losses_fp32[-1])

    print(f"  Gap to fp32 master: bf16={gap_bf16:.6f}, SR={gap_sr:.6f}")

    assert gap_sr < gap_bf16, f"SR loss should be closer to fp32 than pure bf16. gap_sr={gap_sr}, gap_bf16={gap_bf16}"

    # SR should have lower stale% than pure bf16 at step 100
    assert stales_sr[99] < stales_bf16[99], "SR should have less stale params than pure bf16 at step 100"

    print("  PASSED: AdamWBF16 loss closer to fp32 master than pure bf16")


def test_mixed_dtypes():
    """Verify that mixed bf16+fp32 models work correctly."""
    print("\nTEST 3: Mixed dtype model (bf16 + fp32 params)")
    model = create_model(torch.bfloat16)

    # Upcast first layer to fp32 (simulates fp32_router/fp32_experts)
    model.layers[0] = model.layers[0].float()

    bf16_count = sum(p.numel() for p in model.parameters() if p.dtype == torch.bfloat16)
    fp32_count = sum(p.numel() for p in model.parameters() if p.dtype == torch.float32)
    print(f"  BF16 params: {bf16_count / 1e6:.1f}M, FP32 params: {fp32_count / 1e6:.1f}M")

    opt = AdamWBF16(model.parameters(), lr=1e-4)

    # Run training (autocast needed for mixed dtype, same as real training)
    losses, stales = run_training(model, opt, num_steps=50, use_autocast=True)

    # Verify dtypes are preserved
    for name, p in model.named_parameters():
        if "layers.0" in name:
            assert p.dtype == torch.float32, f"Layer 0 param {name} should stay fp32"
        else:
            assert p.dtype == torch.bfloat16, f"Param {name} should stay bf16"

    # Verify training didn't diverge
    assert not math.isnan(losses[-1]), "Loss is NaN"
    assert losses[-1] < losses[0], "Loss should decrease"

    print(f"  Final loss: {losses[-1]:.6f} (started at {losses[0]:.6f})")
    print("  PASSED: Mixed dtype model trains correctly, dtypes preserved")


def test_weight_decay_groups():
    """Verify weight decay is applied correctly (decay vs no-decay groups)."""
    print("\nTEST 4: Weight decay param groups")
    model = create_model(torch.bfloat16)

    # Standard HF-style param grouping
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if "norm" in name or "bias" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    opt = AdamWBF16(
        [
            {"params": decay_params, "weight_decay": 0.01},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=1e-4,
    )

    # Run a few steps
    for _ in range(10):
        x = torch.randn(2, 16, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
        loss = model(x).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

    assert not math.isnan(loss.item()), "Loss is NaN with param groups"
    print(f"  Final loss: {loss.item():.6f}")
    print("  PASSED: Param groups with different weight decay work correctly")


def test_sr_removes_second_moment_bias():
    """SR on exp_avg_sq removes the systematic bias of nearest rounding (Triton path).

    Drives a fixed tiny gradient so (1-beta2)*g^2 lands near the bf16 underflow floor.
    fp32 Adam is the ground truth; nearest-rounded bf16 storage is biased from it by a
    large, single-signed margin; AdamWBF16 (Triton SR kernel) is unbiased — it tracks
    fp32 and is strictly closer to it than to the nearest-rounded reference. This is the
    test that bites if SR is dropped from the second-moment store in the kernel. The
    assertion pins the bias MAGNITUDE (not its sign, which depends on the grad regime).
    """
    print("\nTEST 6: SR removes the second-moment bias (Triton)")
    beta2 = 0.999
    grad_val = 1e-4
    n_steps = 500
    size = 8192
    device = "cuda"

    # fp32 ground truth.
    g32 = torch.full((size,), grad_val, dtype=torch.float32, device=device)
    easq_fp32 = torch.zeros(size, dtype=torch.float32, device=device)
    for _ in range(n_steps):
        easq_fp32.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)
    fp32_mean = float(easq_fp32.mean())

    # Nearest-rounded bf16 reference (the buggy path).
    easq_nearest = torch.zeros(size, dtype=torch.bfloat16, device=device)
    for _ in range(n_steps):
        acc = easq_nearest.float()
        acc.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)
        easq_nearest = acc.to(torch.bfloat16)
    nearest_mean = float(easq_nearest.float().mean())

    # AdamWBF16 with the fused Triton SR kernel.
    p = nn.Parameter(torch.zeros(size, dtype=torch.bfloat16, device=device))
    opt = AdamWBF16([p], lr=1e-3, betas=(0.9, beta2), weight_decay=0.0, use_triton=True)
    for _ in range(n_steps):
        p.grad = torch.full((size,), grad_val, dtype=torch.bfloat16, device=device)
        opt.step()
    sr_mean = float(opt.state[p]["exp_avg_sq"].float().mean())

    print(f"  fp32={fp32_mean:.3e}  nearest={nearest_mean:.3e}  SR={sr_mean:.3e}")

    nearest_bias = abs(nearest_mean - fp32_mean) / fp32_mean
    assert nearest_bias > 0.15, f"premise broken: nearest bias only {nearest_bias:.1%}"
    sr_rel_err = abs(sr_mean - fp32_mean) / fp32_mean
    assert sr_rel_err < 0.05, f"SR easq should track fp32, rel-err {sr_rel_err:.1%}"
    assert abs(sr_mean - fp32_mean) < abs(sr_mean - nearest_mean), (
        "SR easq must be closer to fp32 than to the nearest-rounded reference"
    )
    print("  PASSED: Triton SR keeps the second moment unbiased")


def test_exp_avg_stays_nearest():
    """The first moment (exp_avg) MUST be stored with NEAREST rounding, never SR.

    exp_avg is a signed, ~zero-mean EMA: its truncation error is already unbiased, so SR
    would only inject noise. This test pins that invariant two ways, so it bites if SR is
    ever (wrongly) applied to exp_avg on either the Triton or eager path:

      1. DETERMINISM: run the identical step twice from identical inputs/state but with the
         global / SR RNG perturbed between runs. NEAREST rounding is a pure function of the
         fp32 value, so exp_avg must be BIT-IDENTICAL across the two runs. exp_avg_sq and the
         weight, which DO use SR, must DIFFER (the negative control — it proves the RNG
         perturbation actually changed the SR streams, so the exp_avg equality is meaningful
         rather than an RNG that simply never advanced).
      2. VALUE: exp_avg must equal the deterministic nearest-bf16 of the exact fp32 EMA
         (torch.equal), i.e. it carries no stochastic component.
    """
    print("\nTEST 7: exp_avg stays nearest-rounded (no SR on the first moment)")
    import src.optimizers.adamw_bf16 as adamw_mod

    beta1, beta2 = 0.9, 0.999
    grad_val = 3.0e-4  # signed, small enough that the bf16 store of the EMA is non-trivial
    n_steps = 16
    size = 8192
    device = "cuda"

    for use_triton in (True, False):
        path = "Triton" if use_triton else "eager"

        # fp32 ground-truth EMA of exp_avg over the same fixed gradient sequence.
        g32 = torch.full((size,), grad_val, dtype=torch.float32, device=device)
        ea_fp32 = torch.zeros(size, dtype=torch.float32, device=device)
        for _ in range(n_steps):
            ea_fp32.mul_(beta1).add_(g32, alpha=1.0 - beta1)
        ea_nearest_ref = ea_fp32.to(torch.bfloat16)  # deterministic nearest-bf16 of the exact EMA

        def run_once(rng_seed: int):
            # Reseed BOTH the dedicated SR RNG and torch's global RNG so the two SR noise
            # streams (weight + exp_avg_sq) differ between the two runs, while the inputs do not.
            adamw_mod._SR_RNG.seed(rng_seed)
            torch.manual_seed(rng_seed)
            p = nn.Parameter(torch.zeros(size, dtype=torch.bfloat16, device=device))
            opt = AdamWBF16([p], lr=1e-3, betas=(beta1, beta2), weight_decay=0.0, use_triton=use_triton)
            for _ in range(n_steps):
                p.grad = torch.full((size,), grad_val, dtype=torch.bfloat16, device=device)
                opt.step()
            st = opt.state[p]
            return p.data.clone(), st["exp_avg"].clone(), st["exp_avg_sq"].clone()

        w_a, ea_a, easq_a = run_once(0x1111)
        w_b, ea_b, easq_b = run_once(0x2222)

        # 1a. exp_avg is RNG-independent => bit-identical across the two RNG seeds (NEAREST).
        assert torch.equal(ea_a, ea_b), (
            f"[{path}] exp_avg differs across RNG seeds — it carries stochastic noise (SR applied?)"
        )
        # 1b. Negative control: weight and exp_avg_sq DO use SR, so they must differ across seeds.
        #     (This guards against a vacuous pass where the RNG simply never advanced.)
        assert not torch.equal(w_a, w_b), (
            f"[{path}] weight identical across RNG seeds — SR noise stream did not change; "
            "the exp_avg determinism check would be vacuous"
        )
        assert not torch.equal(easq_a, easq_b), (
            f"[{path}] exp_avg_sq identical across RNG seeds — SR not applied to the second moment"
        )

        # 2. exp_avg equals the deterministic nearest-bf16 reference (no stochastic component).
        assert torch.equal(ea_a, ea_nearest_ref), (
            f"[{path}] exp_avg != nearest-rounded fp32 EMA; first moment is not nearest-rounded"
        )
        print(f"  [{path}] exp_avg bit-identical across RNG seeds and == nearest ref; weight/easq vary (SR)")

    print("  PASSED: exp_avg is nearest-rounded on both paths; SR confined to weight + exp_avg_sq")


def test_sr_removes_second_moment_bias_eager():
    """Eager-path companion to test_sr_removes_second_moment_bias.

    Same fp32-ground-truth / nearest-control / SR-unbiased structure, but exercises
    _eager_adam_bf16_step (the no-Triton FSDP fallback), whose exp_avg_sq SR store is
    otherwise untested. Bites if SR is dropped from the second-moment store in the eager path.
    """
    print("\nTEST 8: SR removes the second-moment bias (eager fallback)")
    beta2 = 0.999
    grad_val = 1e-4
    n_steps = 500
    size = 8192
    device = "cuda"

    # fp32 ground truth.
    g32 = torch.full((size,), grad_val, dtype=torch.float32, device=device)
    easq_fp32 = torch.zeros(size, dtype=torch.float32, device=device)
    for _ in range(n_steps):
        easq_fp32.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)
    fp32_mean = float(easq_fp32.mean())

    # Nearest-rounded bf16 reference (the buggy path).
    easq_nearest = torch.zeros(size, dtype=torch.bfloat16, device=device)
    for _ in range(n_steps):
        acc = easq_nearest.float()
        acc.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)
        easq_nearest = acc.to(torch.bfloat16)
    nearest_mean = float(easq_nearest.float().mean())

    # AdamWBF16 with the EAGER fallback (use_triton=False).
    p = nn.Parameter(torch.zeros(size, dtype=torch.bfloat16, device=device))
    opt = AdamWBF16([p], lr=1e-3, betas=(0.9, beta2), weight_decay=0.0, use_triton=False)
    for _ in range(n_steps):
        p.grad = torch.full((size,), grad_val, dtype=torch.bfloat16, device=device)
        opt.step()
    sr_mean = float(opt.state[p]["exp_avg_sq"].float().mean())

    print(f"  fp32={fp32_mean:.3e}  nearest={nearest_mean:.3e}  SR={sr_mean:.3e}")

    nearest_bias = abs(nearest_mean - fp32_mean) / fp32_mean
    assert nearest_bias > 0.15, f"premise broken: nearest bias only {nearest_bias:.1%}"
    sr_rel_err = abs(sr_mean - fp32_mean) / fp32_mean
    assert sr_rel_err < 0.05, f"eager SR easq should track fp32, rel-err {sr_rel_err:.1%}"
    assert abs(sr_mean - fp32_mean) < abs(sr_mean - nearest_mean), (
        "eager SR easq must be closer to fp32 than to the nearest-rounded reference"
    )
    print("  PASSED: eager-path SR keeps the second moment unbiased")


def test_step_advances_the_version_counter():
    """A step must advance ``param._version`` on BOTH paths — it is what invalidates the
    low-precision weight cache.

    ``cached_fake_quant`` keys its per-step expert-weight round-trip on ``(weight._version, fmt,
    axis)``. The eager path advances the counter for free (ATen sees the in-place write), but the
    Triton kernel stores through a raw pointer and is invisible to the dispatcher, so it must bump
    the counter itself. Without that, every ``lowp_precision != 'bf16'`` run keeps serving the
    step-0 quantized weights while the master moves underneath them — a silently wrong QAT run, not
    a crash.

    The weight-changed assertion is the anti-vacuity control: a step that did nothing would leave
    the quantized copy equal too, and the cache assertion would pass for the wrong reason.
    """
    print("\nTEST 9: an optimizer step advances param._version (low-precision cache invalidation)")

    for use_triton in (True, False):
        path = "Triton" if use_triton else "eager"
        # 3D, mxfp8-block-divisible: the EP expert-weight shape cached_fake_quant is built for.
        p = torch.nn.Parameter(torch.randn(4, 128, 128, device="cuda", dtype=torch.bfloat16))
        opt = AdamWBF16([p], lr=1e-1, weight_decay=0.0, use_triton=use_triton)

        quant_before = cached_fake_quant(p, "mxfp8", 1).clone()
        weight_before = p.detach().clone()
        version_before = p._version

        p.grad = torch.randn_like(p)
        opt.step()

        assert not torch.equal(weight_before, p.detach()), f"{path}: the step did not move the weight"
        assert p._version > version_before, (
            f"{path}: param._version stayed at {p._version} across a step that changed the weight — "
            f"cached_fake_quant would serve the step-0 quantization for the whole run"
        )
        quant_after = cached_fake_quant(p, "mxfp8", 1)
        assert not torch.equal(quant_before, quant_after), (
            f"{path}: the cached low-precision weight did not follow the master across a step"
        )
        print(f"  {path}: _version {version_before} -> {p._version}, quantized weight tracked the master")

    print("  PASSED: both paths invalidate the low-precision weight cache")


def test_every_optimizer_advances_the_version_counter():
    """No optimizer this toolkit builds may leave ``param._version`` frozen across a step.

    ``cached_fake_quant`` keys the per-step expert-weight round-trip on that counter, so an
    optimizer whose update stores through raw pointers (every fused Triton one here) serves the
    step-0 quantization for the whole run while the master weights move — silently wrong under any
    ``lowp_precision`` other than bf16, and invisible at the bf16 default.

    Swept over all four rather than pinned to AdamWBF16: every fused Triton step is exposed, both
    FlashAdamW's (upstream code the toolkit only wraps) and Muon's, not just AdamWBF16's.
    The Muon leg must build through ``create_muon_optimizer`` — the bare upstream
    ``Muon`` writes via ATen ops that bump the counter themselves, and would mask the fused path.
    """
    print("\nTEST 10: every optimizer advances param._version (low-precision cache invalidation)")

    def _named(builder):
        def build(param):
            model = nn.Module()
            model.register_parameter("w", param)
            return builder(model)

        return build

    builders = {
        "AdamWBF16 (Triton)": lambda p: AdamWBF16([p], lr=1e-1, weight_decay=0.0),
        "AdamWBF16 (eager)": lambda p: AdamWBF16([p], lr=1e-1, weight_decay=0.0, use_triton=False),
        "Muon (fused step)": _named(lambda m: create_muon_optimizer(m, lr=1e-1, weight_decay=0.01)),
        "FlashAdamW": _named(lambda m: create_flash_adamw_optimizer(m, lr=1e-1, weight_decay=0.0)),
    }
    covered = set()
    for label, build in builders.items():
        param = torch.nn.Parameter(torch.randn(256, 256, device="cuda", dtype=torch.bfloat16))
        # No ImportError swallow: gram-newton-schulz is a core dependency and flashoptim is baked
        # into both images, so a failed build here is a regression, not an absent extra — catching
        # it would silently reduce this sweep to the two AdamWBF16 legs TEST 9 already covers.
        opt = build(param)
        covered.add(label)
        before_quant = cached_fake_quant(param, "mxfp8", -1).clone()
        before_weight = param.detach().clone()
        before_version = param._version
        param.grad = torch.randn_like(param)
        opt.step()

        assert not torch.equal(before_weight, param.detach()), f"{label}: the step did not move the weight"
        assert param._version > before_version, (
            f"{label}: param._version stayed at {param._version} across a step that changed the "
            f"weight — cached_fake_quant would serve the step-0 quantization for the whole run"
        )
        assert not torch.equal(before_quant, cached_fake_quant(param, "mxfp8", -1)), (
            f"{label}: the cached low-precision weight did not follow the master across a step"
        )
        print(f"  {label}: _version {before_version} -> {param._version}, cache tracked the master")

    assert covered == set(builders), f"legs never ran: {sorted(set(builders) - covered)}"
    print("  PASSED: no optimizer leaves the low-precision cache stale")


def test_state_dict_roundtrip():
    """Verify optimizer state can be saved and loaded (checkpoint compatibility)."""
    print("\nTEST 5: State dict save/load roundtrip")
    model = create_model(torch.bfloat16)
    opt = AdamWBF16(model.parameters(), lr=1e-4)

    # Run a few steps to populate state
    for _ in range(5):
        x = torch.randn(2, 16, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
        loss = model(x).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

    # Save state
    state_dict = opt.state_dict()

    # Create fresh optimizer and load state
    opt2 = AdamWBF16(model.parameters(), lr=1e-4)
    opt2.load_state_dict(state_dict)

    # Verify states match
    for p in model.parameters():
        s1 = opt.state[p]
        s2 = opt2.state[p]
        assert torch.equal(s1["exp_avg"], s2["exp_avg"]), "exp_avg mismatch after load"
        assert torch.equal(s1["exp_avg_sq"], s2["exp_avg_sq"]), "exp_avg_sq mismatch after load"
        assert s1["step"] == s2["step"], "step count mismatch after load"

    print("  PASSED: State dict roundtrip preserves all optimizer state")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, cannot run test")
        exit(1)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA device:     {torch.cuda.get_device_name(0)}")
    print(f"Model size:      {create_model(torch.float32, 'cpu').param_count() / 1e6:.1f}M params")
    print()

    test_dtypes()
    test_vs_baselines()
    test_mixed_dtypes()
    test_weight_decay_groups()
    test_sr_removes_second_moment_bias()
    test_sr_removes_second_moment_bias_eager()
    test_exp_avg_stays_nearest()
    test_step_advances_the_version_counter()
    test_every_optimizer_advances_the_version_counter()
    test_state_dict_roundtrip()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
