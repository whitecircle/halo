#!/usr/bin/env python
"""Every toolkit Liger family patch is numerically equivalent to the module it replaces.

The CPU coverage suite (``tests/cpu/kernels/test_liger_family_coverage.py``) proves each family's own
forward is the FUNCTION the declared kernel computes; this proves the kernel then computes it — the
Triton path, forward and backward, against the family's unpatched module.

Three layers, one per role a :class:`~src.kernels.liger.builder.LigerFamilySpec` can fill:

* the fused head, whole-model: the fused loss must reproduce the family's own head, including a
  declared ``logit_scale`` (Cohere folds it onto the hidden states, which is exact only because
  ``(s·h) @ Wᵀ == s·(h @ Wᵀ)``);
* every declared RMSNorm, against the family's own norm at the spec's ``offset``/``casting_mode``;
* every declared gated (GDN) norm, against the family's own module in fp64 — the one role whose
  kernel is deliberately MORE precise than the eager module, so it is pinned to the function rather
  than to the module's own bf16 rounding;
* every declared GLU MLP, against the family's own ``down(act(gate(x)) * up(x))``.

The head layer runs FIRST: the class swaps below are process-global and irreversible, so a reference
model built after them would no longer be the stock module.

    torchrun --nproc_per_node=1 tests/gpu/kernels/test_liger_family_kernels.py
"""

import importlib
import inspect
import time

import torch
from transformers import AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.patching import patch_moe_model_for_ep
from src.kernels.liger.families import LIGER_FAMILY_SPECS
from src.kernels.liger.orchestrator import resolve_liger_applier
from tests.common.ep_reference import compare_grad, ep_layers
from tests.common.harness import gpu_test_main, record_check
from tests.common.models import TINY_COHERE2_MOE_CONFIG, TINY_GLM4_MOE_LITE_CONFIG, TINY_QWEN35_MOE_CONFIG
from tests.common.utils import cos_sim, log

SEED = 42
BATCH, SEQ, HIDDEN, INTERMEDIATE, HEAD_DIM = 2, 64, 256, 512, 128
NORM_EPS = 1e-5

# The per-head plane a GDN block normalizes at a production sequence length (8192 tokens × 32 value
# heads of `HEAD_DIM`), plus the small 4-D and flattened 2-D forms — the kernel must reduce over the
# last dim in all three, not over whatever it flattens to.
GATED_NORM_SHAPES = ((1, 8192, 32, HEAD_DIM), (BATCH, SEQ, 4, HEAD_DIM), (BATCH * SEQ, HEAD_DIM))

# Liger's kernels reduce in fp32 and store in the input dtype, exactly as the eager modules do, so
# the residual is reduction-order noise: ~1 ULP of the storage dtype, plus the fused GLU's own
# budget (the same 5e-2 `tests/gpu/kernels/test_fused_glu.py` pins for bf16).
TOL_FP32 = 1e-4
TOL_BF16 = 3e-2
TOL_GLU_BF16 = 5e-2
# The gated norm is pinned against the FUNCTION, not against the eager module: fla keeps the whole
# reduction, the weight multiply and the gate in fp32, while the Qwen modules round the normalized
# activation to the storage dtype before the weight multiply — so the fused path is deliberately the
# more accurate of the two and cannot agree with eager to better than bf16 storage rounding. Worst
# measured across both weight dtypes and every shape below: fused-vs-oracle 3.3e-3 (dx),
# fused/eager error ratio 0.37-1.00. The cap is an order of magnitude above the first, and the ratio
# admits no regression — a wrong gate, a wrong reduction axis or a dropped fp32 upcast blows through
# both by orders of magnitude.
TOL_GATED_ORACLE = 2e-2
GATED_ORACLE_RATIO_MAX = 1.05
# Below this the eager module is already exact against the oracle and the ratio would divide by noise.
GATED_ORACLE_ERROR_FLOOR = 1e-5
# The op is memory-bound and eager materializes ~10 intermediates against the kernel's one; measured
# 2.3-4.6x at the production shape. The floor only has to catch a patch that silently stopped firing.
GATED_NORM_MIN_SPEEDUP = 1.2
LOSS_ATOL = 2e-2  # fused vs unfused CE: both bf16 matmul + fp32 reduction
GRAD_COS_MIN = 0.99

NATIVE_SPECS = [spec for spec in LIGER_FAMILY_SPECS if spec.modeling_module]

# The families whose whole-model head can be built from a tiny config in this repo. Cohere2 MoE is
# the one carrying a `logit_scale`; GLM-4.7-Flash is the family whose FLCE default production relies
# on for long-context runs.
FUSED_HEAD_FAMILIES = {
    "cohere2_moe": TINY_COHERE2_MOE_CONFIG,
    "glm4_moe_lite": TINY_GLM4_MOE_LITE_CONFIG,
}

# Delegating families whose shared expert survives the EP wrapper, keyed by the TEXT model_type: the
# composite config ignores flat decoder kwargs, so only the text one builds at tiny scale.
EP_SHARED_EXPERT_FAMILIES = {"qwen3_5_moe_text": TINY_QWEN35_MOE_CONFIG}

# The fused shared expert moves the loss by bf16 rounding on the block's output; measured 1.4e-5
# relative on this model, and a routing flip (the other thing it can cause) stayed under 1.2e-4.
EP_SHARED_EXPERT_LOSS_TOL = 5e-3
# Its whole-model gradient keeps the same norm to within bf16 accumulation over four layers.
EP_SHARED_EXPERT_GRAD_RATIO_TOL = 2e-2


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-9)).item()


def _flags(applier, **requested) -> dict:
    """``requested``, restricted to the roles this family's applier offers, everything else off."""
    parameters = set(inspect.signature(applier).parameters) - {"model"}
    return {name: requested.get(name, False) for name in parameters}


def _fwd_bwd(module, x: torch.Tensor):
    """Forward plus a scalar backward; returns the output and every gradient it produced."""
    x = x.clone().requires_grad_(True)
    out = module(x)
    out.float().pow(2).sum().backward()
    return out.detach(), x.grad, [p.grad for p in module.parameters()]


def _compare(name: str, reference, patched, tol: float) -> None:
    ref_out, ref_dx, ref_dw = reference
    pat_out, pat_dx, pat_dw = patched
    residuals = {"fwd": _rel(pat_out, ref_out), "dx": _rel(pat_dx, ref_dx)}
    for index, (ref_grad, pat_grad) in enumerate(zip(ref_dw, pat_dw, strict=True)):
        residuals[f"dw{index}"] = _rel(pat_grad, ref_grad)
    log(f"  {name}: " + " ".join(f"{key}={value:.1e}" for key, value in residuals.items()))
    worst = max(residuals, key=residuals.get)
    assert residuals[worst] < tol, f"{name} {worst} residual {residuals[worst]:.2e} exceeds {tol:.2e}"


def _snapshot_stock_classes(spec) -> dict[str, type]:
    """The family's stock patchable classes, captured before ANY applier runs.

    Both norm roles ride the same ``rms_norm`` flag, so one applier call swaps them together, and the
    EP check below applies ``swiglu`` before the per-module GLU check reaches it — a check that
    captured its own "original" afterwards would compare the patched class against itself and pass
    on nothing.
    """
    module = importlib.import_module(spec.modeling_module)
    return {name: getattr(module, name) for name in spec.rms_norm + spec.gated_rms_norm + spec.glu_mlp}


def _check_norms(spec, originals, device) -> None:
    module = importlib.import_module(spec.modeling_module)
    applier = resolve_liger_applier(spec.model_types[0])
    applier(**_flags(applier, rms_norm=True))

    for name in spec.rms_norm:
        original = originals[name]
        patched = getattr(module, name)
        assert patched is not original, f"{name} was not swapped"
        for dtype, tol in ((torch.float32, TOL_FP32), (torch.bfloat16, TOL_BF16)):
            torch.manual_seed(SEED)
            reference_norm = original(HIDDEN, NORM_EPS).to(device=device, dtype=dtype)
            reference_norm.weight.data.normal_()
            patched_norm = patched(HIDDEN, NORM_EPS).to(device=device, dtype=dtype)
            patched_norm.weight.data.copy_(reference_norm.weight.data)
            x = torch.randn(BATCH, SEQ, HIDDEN, device=device, dtype=dtype)
            _compare(f"{name}[{dtype}]", _fwd_bwd(reference_norm, x), _fwd_bwd(patched_norm, x), tol)


def _gated_norm_outputs(norm, x, gate, dy, dtype):
    """One gated norm's forward and every gradient, driven by a FIXED upstream gradient.

    Fixed rather than a loss over the output: the oracle's output differs from the bf16 one, so a
    loss computed from it would seed a different upstream gradient and the comparison would measure
    that instead of the kernel.
    """
    norm.zero_grad()
    xi = x.to(dtype).clone().requires_grad_(True)
    gi = gate.to(dtype).clone().requires_grad_(True)
    out = norm(xi, gi)
    out.backward(dy.to(out.dtype))
    return out.detach(), xi.grad, gi.grad, norm.weight.grad.clone()


def _compare_gated_norm(name, original, patched, shape, weight_dtype, device) -> None:
    """One (shape, weight dtype) case: as accurate as the eager module, and within storage rounding."""
    torch.manual_seed(SEED)
    weight = torch.empty(shape[-1]).normal_(mean=1.0, std=0.02)
    x = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    gate = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    dy = torch.randn(*shape, device=device, dtype=torch.bfloat16)

    def build(cls, dtype):
        # The weight is rounded to the STORAGE dtype first, so the oracle states the function of the
        # same weight the two bf16 modules hold — otherwise its own rounding is common-mode in both
        # errors and the accuracy ratio flatters the kernel.
        norm = cls(shape[-1], eps=NORM_EPS)
        norm.weight.data.copy_(weight.to(weight_dtype))
        return norm.to(device=device, dtype=dtype)

    reference_norm, patched_norm = build(original, weight_dtype), build(patched, weight_dtype)
    assert sorted(patched_norm.state_dict()) == sorted(reference_norm.state_dict()), (
        f"{name} gained or lost a checkpoint key"
    )

    # The oracle is the family's OWN module in fp64 — no reimplementation of the norm to drift from
    # it. Its forward upcasts internally, so it states the same function ~4 orders more precisely
    # than the bf16 storage under test.
    oracle = _gated_norm_outputs(build(original, torch.float64), x, gate, dy, torch.float64)
    eager = _gated_norm_outputs(reference_norm, x, gate, dy, torch.bfloat16)
    fused = _gated_norm_outputs(patched_norm, x, gate, dy, torch.bfloat16)

    assert fused[0].shape == torch.Size(shape) and fused[0].dtype == torch.bfloat16
    assert fused[3].dtype == weight_dtype, f"{name} returned a {fused[3].dtype} gradient for a {weight_dtype} weight"

    keys = ("fwd", "dx", "dgate", "dweight")
    errors = {key: _rel(fused[i], oracle[i]) for i, key in enumerate(keys)}
    ratios = {
        key: errors[key] / max(_rel(eager[i], oracle[i]), GATED_ORACLE_ERROR_FLOOR) for i, key in enumerate(keys)
    }
    agreement = max(_rel(fused[i], eager[i]) for i in range(len(keys)))
    log(
        f"  {name} {tuple(shape)} w={weight_dtype}: "
        + " ".join(f"{key}={errors[key]:.1e}/{ratios[key]:.2f}x" for key in keys)
        + f" (vs eager {agreement:.1e})"
    )

    worst = max(errors, key=errors.get)
    assert errors[worst] < TOL_GATED_ORACLE, (
        f"{name} {tuple(shape)} w={weight_dtype} {worst} error {errors[worst]:.2e} exceeds {TOL_GATED_ORACLE:.2e}"
    )
    worst_ratio = max(ratios, key=ratios.get)
    assert ratios[worst_ratio] <= GATED_ORACLE_RATIO_MAX, (
        f"{name} {tuple(shape)} w={weight_dtype} {worst_ratio} is {ratios[worst_ratio]:.2f}x less accurate than the "
        f"eager module it replaces"
    )


def _time_gated_norm(name, original, patched, device) -> None:
    """Forward+backward at the production shape; the fusion has to actually be the faster path."""
    shape = GATED_NORM_SHAPES[0]
    torch.manual_seed(SEED)
    x = torch.randn(*shape, device=device, dtype=torch.bfloat16).requires_grad_(True)
    gate = torch.randn(*shape, device=device, dtype=torch.bfloat16).requires_grad_(True)
    dy = torch.randn(*shape, device=device, dtype=torch.bfloat16)

    elapsed = []
    for cls in (original, patched):
        norm = cls(shape[-1], eps=NORM_EPS).to(device=device, dtype=torch.bfloat16)
        for _ in range(5):
            norm(x, gate).backward(dy)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            norm(x, gate).backward(dy)
        torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - start) / 20 * 1e3)

    speedup = elapsed[0] / elapsed[1]
    log(f"  {name} {tuple(shape)}: eager {elapsed[0]:.2f} ms, fused {elapsed[1]:.2f} ms fwd+bwd ({speedup:.2f}x)")
    assert speedup >= GATED_NORM_MIN_SPEEDUP, f"{name} fused gated norm is only {speedup:.2f}x the eager module"


def _check_gated_norms(spec, originals, device) -> None:
    """The GDN blocks' gated norm, against the family's own module driven in fp64.

    Liger has no kernel for `norm(x) * weight * act(gate)`; `fla`'s is the one these blocks would
    have used had their hub-kernel route been live. It has to agree on all three of the reduction
    axis, the activation, and the gradient wrt the GATE — a kernel gating on the wrong tensor still
    returns the right shape and a plausible loss. Both weight dtypes run: `fp32_non_ep_params` keeps
    these norms in fp32 on the recipes that train the families, and an fp32 master silently handed a
    bf16 gradient is an optimizer bug the shapes alone would not show.
    """
    module = importlib.import_module(spec.modeling_module)
    applier = resolve_liger_applier(spec.model_types[0])
    applier(**_flags(applier, rms_norm=True))

    for name in spec.gated_rms_norm:
        original = originals[name]
        patched = getattr(module, name)
        assert patched is not original, f"{name} was not swapped"
        assert patched.forward is not original.forward, f"{name} kept the eager forward"
        for weight_dtype in (torch.bfloat16, torch.float32):
            for shape in GATED_NORM_SHAPES:
                _compare_gated_norm(name, original, patched, shape, weight_dtype, device)
        _time_gated_norm(name, original, patched, device)


def _build_mlp(cls, config):
    """The family's MLP, whichever of the two constructor shapes on the roster it uses.

    The explicit width is this suite's own: a family whose MLP takes one is a MoE shared expert, and
    its text config spells its size per role (``moe_``/``shared_expert_intermediate_size``) rather
    than as the ``intermediate_size`` the dense families carry.
    """
    try:
        return cls(config)
    except TypeError:
        return cls(config, INTERMEDIATE)


def _check_glu_mlps(spec, originals, device) -> None:
    module = importlib.import_module(spec.modeling_module)
    # `get_text_config()`, not the config itself: a composite family's decoder sizes live on its text
    # sub-config, and the MLP reads `hidden_size` off whatever it is handed.
    config = CONFIG_MAPPING[spec.model_types[0]](
        hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, num_hidden_layers=2, vocab_size=128
    ).get_text_config()
    applier = resolve_liger_applier(spec.model_types[0])
    applier(**_flags(applier, swiglu=True))

    for name in spec.glu_mlp:
        original, patched = originals[name], getattr(module, name)
        assert patched is not original, f"{name} was not swapped"
        torch.manual_seed(SEED)
        reference_mlp = _build_mlp(original, config).to(device=device, dtype=torch.bfloat16)
        patched_mlp = _build_mlp(patched, config).to(device=device, dtype=torch.bfloat16)
        patched_mlp.load_state_dict(reference_mlp.state_dict())
        assert patched_mlp._halo_glu_mul is not None, f"{name}'s activation was not recognized as fusable"
        x = torch.randn(BATCH, SEQ, config.hidden_size, device=device, dtype=torch.bfloat16)
        _compare(name, _fwd_bwd(reference_mlp, x), _fwd_bwd(patched_mlp, x), TOL_GLU_BF16)


def _loss_and_grads(model, input_ids) -> tuple[float, dict[str, torch.Tensor]]:
    model.train()
    output = model(input_ids=input_ids, labels=input_ids.clone())
    output.loss.backward()
    return output.loss.item(), {n: p.grad.detach() for n, p in model.named_parameters() if p.grad is not None}


def _check_ep_shared_expert(model_type, tiny, stock, device) -> None:
    """The shared-expert MLP a delegating spec names must arrive fused THROUGH the EP wrapper.

    Upstream's ``swiglu`` sets only the routed-experts class, which ``patch_moe_model_for_ep``
    replaces wholesale (the wrapped layer has no ``experts`` submodule left, which is why that swap
    is inert rather than wrong). The shared expert is adopted unchanged, so it is fused only because
    the spec names its class — the reason the EP force-off does not apply to these families.

    Compared on the shared expert's own gradients, not the routed experts': a fused GLU moves the
    hidden states by a bf16 ULP, which flips a near-tied top-k pick and rewrites one expert's
    gradient wholesale. That is the router's behaviour, not the kernel's.
    """
    applier = resolve_liger_applier(model_type)
    spec = applier.spec
    originals = stock[spec.model_types[0]]
    # All-full-attention: the gated-delta-net layers pull in `fla`'s autotuned delta-rule kernels and
    # have no shared expert to check. The GDN norm is covered above, on its own.
    config = CONFIG_MAPPING[model_type](
        **{**tiny, "layer_types": ["full_attention"] * tiny["num_hidden_layers"], "attn_implementation": "eager"}
    )
    ep_config = EPConfig(ep_size=1, world_size=1, gpus_per_node=1, use_grouped_gemm=True)
    input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ), device=device)

    torch.manual_seed(SEED)
    reference = AutoModelForCausalLM.from_config(config).to(device=device, dtype=torch.bfloat16)
    weights = {name: tensor.clone() for name, tensor in reference.state_dict().items()}
    patch_moe_model_for_ep(reference, ep_config)
    reference_loss, reference_grads = _loss_and_grads(reference, input_ids)

    applier(**_flags(applier, swiglu=True))
    modeling = importlib.import_module(spec.modeling_module)
    for name in spec.glu_mlp:
        assert getattr(modeling, name) is not originals[name], f"{name} was not swapped"

    torch.manual_seed(SEED)
    fused = AutoModelForCausalLM.from_config(config).to(device=device, dtype=torch.bfloat16)
    fused.load_state_dict(weights)
    patch_moe_model_for_ep(fused, ep_config)

    wrapped = ep_layers(fused)
    assert wrapped, "no MoE layer was EP-wrapped — the check would prove nothing about the wrapper"
    assert not any(hasattr(layer, "experts") for layer in wrapped), (
        "the EP layer kept an `experts` submodule; upstream's LigerExperts swap would no longer be inert"
    )
    shared = [
        (f"{name}.{parameter}", module)
        for name, module in fused.named_modules()
        if type(module).__name__ in spec.glu_mlp
        for parameter, _ in module.named_parameters()
    ]
    assert shared, "the EP wrapper kept no MLP the spec names"
    assert all(getattr(module, "_halo_glu_mul", None) is not None for _, module in shared), (
        "the shared expert the EP wrapper adopted is not fused"
    )

    fused_loss, fused_grads = _loss_and_grads(fused, input_ids)
    loss_delta = abs(fused_loss - reference_loss) / max(abs(reference_loss), 1e-9)
    # Norm ratio and cosine, not a max-abs residual: these are WHOLE-MODEL gradients, so they carry
    # four layers of bf16 rounding on top of the kernel's own — the same reason the fused-head check
    # above compares cosines. The pair still separates the two failures that matter, a mis-scaled
    # activation (ratio) from a wrong one (cosine).
    compared = {name: compare_grad(fused_grads[name], reference_grads[name]) for name, _ in shared}
    worst_ratio = max(compared, key=lambda name: abs(compared[name][0] - 1.0))
    worst_cos = min(compared, key=lambda name: compared[name][1])
    log(
        f"  {model_type}: {len(wrapped)} EP layers, loss {reference_loss:.5f} vs {fused_loss:.5f} "
        f"(rel {loss_delta:.1e}), shared-expert grads: worst ratio {compared[worst_ratio][0]:.4f} "
        f"({worst_ratio}), worst cosine {compared[worst_cos][1]:.5f}"
    )
    assert loss_delta < EP_SHARED_EXPERT_LOSS_TOL, f"fused shared expert moved the loss by {loss_delta:.2e}"
    assert abs(compared[worst_ratio][0] - 1.0) < EP_SHARED_EXPERT_GRAD_RATIO_TOL, (
        f"{worst_ratio} gradient norm scaled by {compared[worst_ratio][0]:.4f}"
    )
    assert compared[worst_cos][1] > GRAD_COS_MIN, f"{worst_cos} gradient diverged (cos={compared[worst_cos][1]:.5f})"


def _check_rope(spec, device) -> None:
    """Liger's rotary must equal the family's own, on the `[batch, heads, seq, head_dim]` layout.

    The family applies it after transposing heads forward; a kernel expecting a different layout
    would rotate across the wrong axis and still return the right shape.
    """
    module = importlib.import_module(spec.modeling_module)
    original = module.apply_rotary_pos_emb
    applier = resolve_liger_applier(spec.model_types[0])
    applier(**_flags(applier, rope=True))
    patched = module.apply_rotary_pos_emb
    assert patched is not original, "the rotary was not swapped"

    head_dim, heads = 64, 4
    torch.manual_seed(SEED)
    base_q = torch.randn(BATCH, heads, SEQ, head_dim, device=device, dtype=torch.bfloat16)
    base_k = torch.randn(BATCH, heads, SEQ, head_dim, device=device, dtype=torch.bfloat16)
    angles = torch.rand(BATCH, SEQ, head_dim // 2, device=device, dtype=torch.float32) * 6.28
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).to(torch.bfloat16)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).to(torch.bfloat16)

    residuals = {}
    outputs = {}
    for name, rope in (("reference", original), ("patched", patched)):
        q, k = base_q.clone().requires_grad_(True), base_k.clone().requires_grad_(True)
        rotated_q, rotated_k = rope(q, k, cos, sin)
        (rotated_q.float().pow(2).sum() + rotated_k.float().pow(2).sum()).backward()
        outputs[name] = (rotated_q.detach(), rotated_k.detach(), q.grad, k.grad)
    for index, label in enumerate(("q", "k", "dq", "dk")):
        residuals[label] = _rel(outputs["patched"][index], outputs["reference"][index])
    log(f"  {spec.model_types[0]} rope: " + " ".join(f"{k}={v:.1e}" for k, v in residuals.items()))
    worst = max(residuals, key=residuals.get)
    assert residuals[worst] < TOL_BF16, f"rope {worst} residual {residuals[worst]:.2e} exceeds {TOL_BF16:.2e}"


def _check_fused_head(model_type, tiny_config, device) -> None:
    config = CONFIG_MAPPING[model_type](**{**tiny_config, "attn_implementation": "eager"})
    torch.manual_seed(SEED)
    reference = AutoModelForCausalLM.from_config(config).to(device=device, dtype=torch.bfloat16)
    reference.train()

    input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ), device=device)
    labels = input_ids.clone()
    reference_output = reference(input_ids=input_ids, labels=labels)
    reference_output.loss.backward()
    reference_grads = {name: p.grad.detach().clone() for name, p in reference.named_parameters() if p.grad is not None}

    applier = resolve_liger_applier(model_type)
    applier(**_flags(applier, fused_linear_cross_entropy=True))

    torch.manual_seed(SEED)
    fused = AutoModelForCausalLM.from_config(config).to(device=device, dtype=torch.bfloat16)
    fused.load_state_dict(reference.state_dict())
    fused.train()
    fused_output = fused(input_ids=input_ids, labels=labels)
    assert fused_output.logits is None, "the fused head must not materialize the logits plane"
    fused_output.loss.backward()

    delta = abs(fused_output.loss.item() - reference_output.loss.item())
    log(f"  {model_type}: loss {reference_output.loss.item():.6f} vs {fused_output.loss.item():.6f} (Δ{delta:.2e})")
    assert delta < LOSS_ATOL, f"{model_type} fused loss differs by {delta:.3e}"

    worst_name, worst_cos = None, 1.0
    for name, p in fused.named_parameters():
        if p.grad is None or name not in reference_grads:
            continue
        similarity = cos_sim(p.grad, reference_grads[name])
        if similarity < worst_cos:
            worst_name, worst_cos = name, similarity
    log(f"  {model_type}: worst gradient cosine {worst_cos:.5f} at {worst_name}")
    assert worst_cos > GRAD_COS_MIN, f"{model_type} gradient {worst_name} diverged (cos={worst_cos:.5f})"


@gpu_test_main(min_world_size=1, prefix="test_liger_family_kernels")
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    torch.cuda.set_device(ctx.device)

    # Every class swap below is process-global and irreversible, so the stock classes are captured
    # before the first applier runs rather than by each check for itself.
    stock = {spec.model_types[0]: _snapshot_stock_classes(spec) for spec in NATIVE_SPECS}

    # Before any class swap — a reference head built after one would not be the stock module.
    log("\n--- fused head (whole model) ---")
    for model_type, tiny_config in FUSED_HEAD_FAMILIES.items():
        record_check(
            checks,
            f"fused_head[{model_type}]",
            lambda model_type=model_type, tiny_config=tiny_config: _check_fused_head(
                model_type, tiny_config, ctx.device
            ),
        )

    log("\n--- RMSNorm ---")
    for spec in NATIVE_SPECS:
        if spec.rms_norm:
            record_check(
                checks,
                f"rms_norm[{spec.model_types[0]}]",
                lambda spec=spec: _check_norms(spec, stock[spec.model_types[0]], ctx.device),
            )

    log("\n--- gated RMSNorm (fla) ---")
    for spec in NATIVE_SPECS:
        if spec.gated_rms_norm:
            record_check(
                checks,
                f"gated_rms_norm[{spec.model_types[0]}]",
                lambda spec=spec: _check_gated_norms(spec, stock[spec.model_types[0]], ctx.device),
            )

    # Before the per-module GLU check: it needs an EP-wrapped model built from the STOCK classes.
    log("\n--- shared expert under the EP wrapper ---")
    for model_type, tiny_config in EP_SHARED_EXPERT_FAMILIES.items():
        record_check(
            checks,
            f"ep_shared_expert[{model_type}]",
            lambda model_type=model_type, tiny_config=tiny_config: _check_ep_shared_expert(
                model_type, tiny_config, stock, ctx.device
            ),
        )

    log("\n--- fused GLU MLP ---")
    for spec in NATIVE_SPECS:
        if spec.glu_mlp:
            record_check(
                checks,
                f"swiglu[{spec.model_types[0]}]",
                lambda spec=spec: _check_glu_mlps(spec, stock[spec.model_types[0]], ctx.device),
            )

    log("\n--- RoPE ---")
    for spec in NATIVE_SPECS:
        if spec.rope:
            record_check(checks, f"rope[{spec.model_types[0]}]", lambda spec=spec: _check_rope(spec, ctx.device))

    return {"checks": checks}


if __name__ == "__main__":
    run()
