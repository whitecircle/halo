#!/usr/bin/env python
"""AdamWBF16 with stochastic rounding: bf16 weights and states, loss near fp32 masters, mixed-dtype models."""

import copy
import math

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import _init_optim_state

from halo.kernels.lowp.quantization import cached_fake_quant
from halo.optimizers.adamw_bf16 import AdamWBF16
from halo.optimizers.flash_adamw import build_flash_adamw_optimizer
from halo.optimizers.muon import build_muon_optimizer
from tests.common.harness import gpu_test_main, record_check
from tests.common.optimizer_state import assert_optimizer_state_bit_exact


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


def count_stale(model, old_params):
    stale = total = 0
    for p, old_p in zip(model.parameters(), old_params, strict=False):
        if p.requires_grad:
            stale += (p.data == old_p).sum().item()
            total += p.numel()
    return stale / total if total > 0 else 0


def snapshot_params(model):
    return [p.data.clone() for p in model.parameters()]


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


def test_dtypes():
    print("TEST 1: Dtype verification")
    model = create_model(torch.bfloat16)
    opt = AdamWBF16(model.named_parameters(), lr=1e-4)

    x = torch.randn(2, 16, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    loss = model(x).mean()
    loss.backward()
    opt.step()

    for name, p in model.named_parameters():
        assert p.dtype == torch.bfloat16, f"Param {name} is {p.dtype}, expected bf16"

    for p in model.parameters():
        state = opt.state[p]
        assert state["exp_avg"].dtype == torch.bfloat16, f"exp_avg is {state['exp_avg'].dtype}, expected bf16"
        assert state["exp_avg_sq"].dtype == torch.bfloat16, f"exp_avg_sq is {state['exp_avg_sq'].dtype}, expected bf16"

    print("  PASSED: All params bf16, all states bf16")


def test_vs_baselines():
    print("\nTEST 2: Loss comparison vs baselines")

    model_bf16 = create_model(torch.bfloat16)
    opt_bf16 = torch.optim.AdamW(model_bf16.parameters(), lr=1e-4, fused=True)
    losses_bf16, stales_bf16 = run_training(model_bf16, opt_bf16)

    model_fp32 = create_model(torch.float32)
    opt_fp32 = torch.optim.AdamW(model_fp32.parameters(), lr=1e-4, fused=True)
    losses_fp32, stales_fp32 = run_training(model_fp32, opt_fp32, use_autocast=True)

    model_sr = create_model(torch.bfloat16)
    opt_sr = AdamWBF16(model_sr.named_parameters(), lr=1e-4)
    losses_sr, stales_sr = run_training(model_sr, opt_sr)

    print(f"  Pure BF16:      loss={losses_bf16[-1]:.6f}  stale%={stales_bf16[-1]:.4f}")
    print(f"  FP32 master:    loss={losses_fp32[-1]:.6f}  stale%={stales_fp32[-1]:.4f}")
    print(f"  AdamWBF16 (SR): loss={losses_sr[-1]:.6f}  stale%={stales_sr[-1]:.4f}")

    gap_bf16 = abs(losses_bf16[-1] - losses_fp32[-1])
    gap_sr = abs(losses_sr[-1] - losses_fp32[-1])

    print(f"  Gap to fp32 master: bf16={gap_bf16:.6f}, SR={gap_sr:.6f}")

    assert gap_sr < gap_bf16, f"SR loss should be closer to fp32 than pure bf16. gap_sr={gap_sr}, gap_bf16={gap_bf16}"

    assert stales_sr[99] < stales_bf16[99], "SR should have less stale params than pure bf16 at step 100"

    print("  PASSED: AdamWBF16 loss closer to fp32 master than pure bf16")


def test_mixed_dtypes():
    print("\nTEST 3: Mixed dtype model (bf16 + fp32 params)")
    model = create_model(torch.bfloat16)

    # Simulates fp32_router/fp32_experts.
    model.layers[0] = model.layers[0].float()

    bf16_count = sum(p.numel() for p in model.parameters() if p.dtype == torch.bfloat16)
    fp32_count = sum(p.numel() for p in model.parameters() if p.dtype == torch.float32)
    print(f"  BF16 params: {bf16_count / 1e6:.1f}M, FP32 params: {fp32_count / 1e6:.1f}M")

    opt = AdamWBF16(model.named_parameters(), lr=1e-4)

    # Mixed dtypes need autocast, as in real training.
    losses, stales = run_training(model, opt, num_steps=50, use_autocast=True)

    for name, p in model.named_parameters():
        if "layers.0" in name:
            assert p.dtype == torch.float32, f"Layer 0 param {name} should stay fp32"
        else:
            assert p.dtype == torch.bfloat16, f"Param {name} should stay bf16"

    assert math.isfinite(losses[-1]), f"Loss is not finite: {losses[-1]}"
    assert losses[-1] < losses[0], "Loss should decrease"

    print(f"  Final loss: {losses[-1]:.6f} (started at {losses[0]:.6f})")
    print("  PASSED: Mixed dtype model trains correctly, dtypes preserved")


def test_weight_decay_groups():
    """Decay and no-decay param groups step with a finite loss (the decay itself is not checked)."""
    print("\nTEST 4: Weight decay param groups")
    model = create_model(torch.bfloat16)

    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if "norm" in name or "bias" in name:
            no_decay_params.append((name, p))
        else:
            decay_params.append((name, p))

    opt = AdamWBF16(
        [
            {"params": decay_params, "weight_decay": 0.01},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=1e-4,
    )

    for _ in range(10):
        x = torch.randn(2, 16, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
        loss = model(x).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

    assert math.isfinite(loss.item()), f"Loss is not finite with param groups: {loss.item()}"
    print(f"  Final loss: {loss.item():.6f}")
    print("  PASSED: Param groups with different weight decay step with a finite loss")


def test_sr_removes_second_moment_bias():
    """SR on exp_avg_sq removes nearest rounding's bias (Triton path). A fixed tiny gradient puts (1-beta2)*g^2 near
    the bf16 underflow floor. Pins the bias magnitude; its sign depends on the grad regime."""
    print("\nTEST 6: SR removes the second-moment bias (Triton)")
    beta2 = 0.999
    grad_val = 1e-4
    n_steps = 500
    size = 8192
    device = "cuda"

    g32 = torch.full((size,), grad_val, dtype=torch.float32, device=device)
    easq_fp32 = torch.zeros(size, dtype=torch.float32, device=device)
    for _ in range(n_steps):
        easq_fp32.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)
    fp32_mean = float(easq_fp32.mean())

    easq_nearest = torch.zeros(size, dtype=torch.bfloat16, device=device)
    for _ in range(n_steps):
        acc = easq_nearest.float()
        acc.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)
        easq_nearest = acc.to(torch.bfloat16)
    nearest_mean = float(easq_nearest.float().mean())

    p = nn.Parameter(torch.zeros(size, dtype=torch.bfloat16, device=device))
    opt = AdamWBF16([("p", p)], lr=1e-3, betas=(0.9, beta2), weight_decay=0.0, use_triton=True)
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
    """exp_avg keeps nearest rounding: a signed ~zero-mean EMA's truncation error is already unbiased, so SR adds
    noise. Renaming the param re-keys the noise: exp_avg stays bit-identical while exp_avg_sq and the weight differ."""
    print("\nTEST 7: exp_avg stays nearest-rounded (no SR on the first moment)")
    beta1, beta2 = 0.9, 0.999
    grad_val = 3.0e-4  # signed, small enough that the bf16 store of the EMA is non-trivial
    n_steps = 16
    size = 8192
    device = "cuda"

    for use_triton in (True, False):
        path = "Triton" if use_triton else "eager"

        g32 = torch.full((size,), grad_val, dtype=torch.float32, device=device)
        ea_fp32 = torch.zeros(size, dtype=torch.float32, device=device)
        for _ in range(n_steps):
            ea_fp32.mul_(beta1).add_(g32, alpha=1.0 - beta1)
        ea_nearest_ref = ea_fp32.to(torch.bfloat16)

        def run_once(name: str):
            # Another name keys other SR noise; the inputs stay identical.
            p = nn.Parameter(torch.zeros(size, dtype=torch.bfloat16, device=device))
            opt = AdamWBF16([(name, p)], lr=1e-3, betas=(beta1, beta2), weight_decay=0.0, use_triton=use_triton)
            for _ in range(n_steps):
                p.grad = torch.full((size,), grad_val, dtype=torch.bfloat16, device=device)
                opt.step()
            st = opt.state[p]
            return p.data.clone(), st["exp_avg"].clone(), st["exp_avg_sq"].clone()

        w_a, ea_a, easq_a = run_once("p")
        w_b, ea_b, easq_b = run_once("q")

        assert torch.equal(ea_a, ea_b), (
            f"[{path}] exp_avg differs across SR noise — it carries stochastic noise (SR applied?)"
        )
        # Negative control: the equality above is vacuous if the noise never changed.
        assert not torch.equal(w_a, w_b), (
            f"[{path}] weight identical across SR noise — the noise did not change; "
            "the exp_avg determinism check would be vacuous"
        )
        assert not torch.equal(easq_a, easq_b), (
            f"[{path}] exp_avg_sq identical across SR noise — SR not applied to the second moment"
        )

        assert torch.equal(ea_a, ea_nearest_ref), (
            f"[{path}] exp_avg != nearest-rounded fp32 EMA; first moment is not nearest-rounded"
        )
        print(f"  [{path}] exp_avg bit-identical across SR noise and == nearest ref; weight/easq vary (SR)")

    print("  PASSED: exp_avg is nearest-rounded on both paths; SR confined to weight + exp_avg_sq")


def test_sr_removes_second_moment_bias_eager():
    """Eager-path companion to test_sr_removes_second_moment_bias; nothing else tests this exp_avg_sq SR store."""
    print("\nTEST 8: SR removes the second-moment bias (eager fallback)")
    beta2 = 0.999
    grad_val = 1e-4
    n_steps = 500
    size = 8192
    device = "cuda"

    g32 = torch.full((size,), grad_val, dtype=torch.float32, device=device)
    easq_fp32 = torch.zeros(size, dtype=torch.float32, device=device)
    for _ in range(n_steps):
        easq_fp32.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)
    fp32_mean = float(easq_fp32.mean())

    easq_nearest = torch.zeros(size, dtype=torch.bfloat16, device=device)
    for _ in range(n_steps):
        acc = easq_nearest.float()
        acc.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)
        easq_nearest = acc.to(torch.bfloat16)
    nearest_mean = float(easq_nearest.float().mean())

    p = nn.Parameter(torch.zeros(size, dtype=torch.bfloat16, device=device))
    opt = AdamWBF16([("p", p)], lr=1e-3, betas=(0.9, beta2), weight_decay=0.0, use_triton=False)
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
    """A step must advance ``param._version``, which ``cached_fake_quant`` keys on. The Triton kernel stores through a
    raw pointer the dispatcher cannot see, so it must bump the counter itself, or ``lowp_precision != 'bf16'`` runs
    silently serve the step-0 quantized weights. The weight-changed assertion is the anti-vacuity control."""
    print("\nTEST 9: an optimizer step advances param._version (low-precision cache invalidation)")

    for use_triton in (True, False):
        path = "Triton" if use_triton else "eager"
        # 3D, mxfp8-block-divisible: the EP expert-weight shape cached_fake_quant is built for.
        p = torch.nn.Parameter(torch.randn(4, 128, 128, device="cuda", dtype=torch.bfloat16))
        opt = AdamWBF16([("p", p)], lr=1e-1, weight_decay=0.0, use_triton=use_triton)

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
    """No optimizer may leave ``param._version`` frozen, since ``cached_fake_quant`` keys on it. Muon builds through
    ``build_muon_optimizer``: the bare upstream ``Muon`` bumps the counter and would mask the fused path."""
    print("\nTEST 10: every optimizer advances param._version (low-precision cache invalidation)")

    def _named(builder):
        def build(param):
            model = nn.Module()
            model.register_parameter("w", param)
            return builder(model)

        return build

    builders = {
        "AdamWBF16 (Triton)": lambda p: AdamWBF16([("p", p)], lr=1e-1, weight_decay=0.0),
        "AdamWBF16 (eager)": lambda p: AdamWBF16([("p", p)], lr=1e-1, weight_decay=0.0, use_triton=False),
        "Muon (fused step)": _named(lambda m: build_muon_optimizer(m, lr=1e-1, weight_decay=0.01)),
        "FlashAdamW": _named(lambda m: build_flash_adamw_optimizer(m, lr=1e-1, weight_decay=0.0)),
    }
    for label, build in builders.items():
        param = torch.nn.Parameter(torch.randn(256, 256, device="cuda", dtype=torch.bfloat16))
        # No ImportError swallow: gram-newton-schulz is a core dependency and flashoptim is in both images, so a failed
        # build is a regression.
        opt = build(param)
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

    print("  PASSED: no optimizer leaves the low-precision cache stale")


def _deferred_vs_prescaled(scale: float, device: str, use_triton: bool):
    """Step two identical param sets, one prescaled in place by ``scale``, one through ``defer_grad_scale``. SR seeds
    depend only on the step and the parameter's name, so two fresh optimizers round alike."""
    torch.manual_seed(0)
    shapes = [((4099,), torch.bfloat16), ((33, 65), torch.bfloat16), ((17,), torch.float32)]
    base = [torch.randn(shape, device=device).to(dtype) for shape, dtype in shapes]
    grads = [torch.randn_like(t) for t in base]
    results = []
    for deferred in (False, True):
        params = [nn.Parameter(t.clone()) for t in base]
        optimizer = AdamWBF16(
            [(f"p{i}", p) for i, p in enumerate(params)], lr=1e-2, weight_decay=0.1, use_triton=use_triton
        )
        for _ in range(3):
            for p, g in zip(params, grads, strict=True):
                p.grad = g.clone() if deferred else g.clone().mul_(torch.tensor(scale, device=device))
            if deferred:
                optimizer.defer_grad_scale(torch.tensor(scale, device=device))
            optimizer.step()
        results.append([p.detach().clone() for p in params] + [optimizer.state[p]["exp_avg_sq"] for p in params])
    return results


def test_deferred_grad_scale_matches_prescaled_grads():
    """The deferred clip scale equals scaling gradients in place, bit for bit: an fp32 product in the step would pass
    the global norm's run-to-run noise on to the weights. 0.3 is inexact in bf16; 0.25 is exact and passes either
    way."""
    for scale in (0.3, 0.25):
        for device, use_triton in (("cuda", True), ("cpu", False)):
            prescaled, deferred = _deferred_vs_prescaled(scale, device, use_triton)
            for a, b in zip(prescaled, deferred, strict=True):
                assert torch.equal(a, b), (scale, device, use_triton)


def test_deferred_grad_scale_is_consumed_once_and_cleared_by_zero_grad():
    """The scale applies to one step only. A scale reaching a later step would move the moments (Adam's first update is
    nearly scale-free, its state is not)."""
    torch.manual_seed(0)
    base = torch.randn(4099, device="cuda").to(torch.bfloat16)
    grads = [torch.randn_like(base) for _ in range(3)]
    scale = torch.tensor(0.5, device="cuda")
    deferring, plain = nn.Parameter(base.clone()), nn.Parameter(base.clone())
    deferring_opt = AdamWBF16([("p", deferring)], lr=1e-2, weight_decay=0.0)
    plain_opt = AdamWBF16([("p", plain)], lr=1e-2, weight_decay=0.0)

    def agree() -> bool:
        a, b = deferring_opt.state[deferring], plain_opt.state[plain]
        return all(
            torch.equal(x, y)
            for x, y in ((deferring, plain), (a["exp_avg"], b["exp_avg"]), (a["exp_avg_sq"], b["exp_avg_sq"]))
        )

    deferring_opt.defer_grad_scale(scale)
    deferring.grad, plain.grad = grads[0].clone(), grads[0].clone().mul_(scale)
    deferring_opt.step()
    plain_opt.step()
    assert deferring_opt._grad_scale is None
    assert agree(), "the deferred step diverged from the prescaled one"

    deferring.grad, plain.grad = grads[1].clone(), grads[1].clone()
    deferring_opt.step()
    plain_opt.step()
    assert agree(), "the scale reached the step after the one it was deferred to"

    deferring_opt.defer_grad_scale(scale)
    deferring_opt.zero_grad()
    plain_opt.zero_grad()
    assert deferring_opt._grad_scale is None
    deferring.grad, plain.grad = grads[2].clone(), grads[2].clone()
    deferring_opt.step()
    plain_opt.step()
    assert agree(), "a scale deferred before zero_grad reached the next step"


def test_state_dict_roundtrip():
    """Save -> deepcopy -> load is bit-exact. ``load_state_dict`` keeps a same-dtype tensor, so the live dict would
    alias."""
    print("\nTEST 5: State dict save/load roundtrip")
    model = create_model(torch.bfloat16)
    opt = AdamWBF16(model.named_parameters(), lr=1e-4)

    for _ in range(5):
        x = torch.randn(2, 16, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
        loss = model(x).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

    saved = copy.deepcopy(opt.state_dict())
    opt2 = AdamWBF16(model.named_parameters(), lr=1e-4)
    opt2.load_state_dict(copy.deepcopy(saved))
    assert_optimizer_state_bit_exact(saved, opt2.state_dict())

    print("  PASSED: State dict roundtrip preserves all optimizer state")


def _train_on(optimizer, params: list[nn.Parameter], grads: list[list[torch.Tensor]]) -> None:
    for step_grads in grads:
        for p, g in zip(params, step_grads, strict=True):
            p.grad = g
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def test_restore_replays_the_rounding():
    """A restored optimizer rounds as the uninterrupted one on the Triton paths, restored through torch's zero-LR
    ``_init_optim_state`` as every sharded resume does. Fixed gradients keep the forward out of it."""
    print("\nTEST 11: A restored optimizer replays the uninterrupted run's rounding")
    builders = {
        "AdamWBF16 (Triton)": lambda m: AdamWBF16(m.named_parameters(), lr=1e-3),
        "Muon (fused step)": lambda m: build_muon_optimizer(m, lr=1e-2, weight_decay=0.01),
    }
    for name, build in builders.items():
        gen = torch.Generator(device="cuda").manual_seed(7)
        shapes = [(p.shape, p.dtype) for p in create_model(torch.bfloat16).parameters()]
        grads = [[torch.randn(shape, generator=gen, device="cuda", dtype=dtype) * 1e-3 for shape, dtype in shapes]]
        grads += [[g * (step + 2) for g in grads[0]] for step in range(5)]

        uninterrupted = create_model(torch.bfloat16)
        _train_on(build(uninterrupted), list(uninterrupted.parameters()), grads)

        first = create_model(torch.bfloat16)
        first_opt = build(first)
        _train_on(first_opt, list(first.parameters()), grads[:3])
        saved = copy.deepcopy(first_opt.state_dict())

        resumed = create_model(torch.bfloat16)
        with torch.no_grad():
            for p, w in zip(resumed.parameters(), first.parameters(), strict=True):
                p.copy_(w)
        resumed_opt = build(resumed)
        _init_optim_state(resumed_opt)
        resumed_opt.load_state_dict(copy.deepcopy(saved))
        _train_on(resumed_opt, list(resumed.parameters()), grads[3:])

        unequal = [
            n
            for (n, a), b in zip(uninterrupted.named_parameters(), resumed.parameters(), strict=True)
            if not torch.equal(a, b)
        ]
        assert not unequal, f"[{name}] {len(unequal)} params rounded differently after the restore: {unequal[:3]}"
        print(f"  [{name}] every param bit-identical to the uninterrupted run")
    print("  PASSED: restored optimizers round as the uninterrupted run did")


@gpu_test_main(exact_world_size=1, prefix="adamw_bf16", partial_state=False)
def run(ctx) -> dict:
    print(f"CUDA device:     {torch.cuda.get_device_name(ctx.device)}")
    print(f"Model size:      {create_model(torch.float32, 'cpu').param_count() / 1e6:.1f}M params")
    checks: dict[str, bool] = {}
    record_check(checks, "dtypes", test_dtypes)
    record_check(checks, "vs_baselines", test_vs_baselines)
    record_check(checks, "mixed_dtypes", test_mixed_dtypes)
    record_check(checks, "weight_decay_groups", test_weight_decay_groups)
    record_check(checks, "sr_removes_second_moment_bias", test_sr_removes_second_moment_bias)
    record_check(checks, "sr_removes_second_moment_bias_eager", test_sr_removes_second_moment_bias_eager)
    record_check(checks, "exp_avg_stays_nearest", test_exp_avg_stays_nearest)
    record_check(checks, "step_advances_the_version_counter", test_step_advances_the_version_counter)
    record_check(
        checks, "every_optimizer_advances_the_version_counter", test_every_optimizer_advances_the_version_counter
    )
    record_check(
        checks, "deferred_grad_scale_matches_prescaled_grads", test_deferred_grad_scale_matches_prescaled_grads
    )
    record_check(
        checks,
        "deferred_grad_scale_is_consumed_once_and_cleared_by_zero_grad",
        test_deferred_grad_scale_is_consumed_once_and_cleared_by_zero_grad,
    )
    record_check(checks, "state_dict_roundtrip", test_state_dict_roundtrip)
    record_check(checks, "restore_replays_the_rounding", test_restore_replays_the_rounding)
    return {"checks": checks}


if __name__ == "__main__":
    run()
