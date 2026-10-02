#!/usr/bin/env python
"""Every MoE family, through the whole production kernel stack, computes the stock model's loss and
gradients.

The per-kernel suites pin each kernel against its own reference: ``test_fused_glu.py`` (the GLU
combines), ``test_moe_permute.py`` (the fused un-permute), ``test_liger_family_kernels.py`` and
``tests/cpu/kernels/test_native_rms_norm.py`` (each Liger role and norm casting mode). This file checks
that nothing breaks where they meet, per family: the family's Liger applier as the loader calls it and
the EP wrapper with grouped GEMM (ep_size 1, so every expert is local and the fused weighted un-permute
runs). A family whose wrapper latched the wrong combine, whose packed GLU read the wrong half, or whose
experts lost a gradient moves the loss or a gradient by far more than the tolerances below.

Each launch runs one family of :data:`tests.common.tiny_models.TINY_MOE_FAMILIES` (``--family``; the
manifest has a row per family), built by the roster, fixups included: an applier rebinds the family's HF
classes (and transformers' loss) for the rest of the process, so the stock references are built and run
before it is applied.

Two comparisons, both against the stock Hugging Face model built from the same weights:

* fp32 against fp32: the stack must match to fp32 round-off (reduction order, fused-kernel accumulation);
  torch's TF32 is off on both sides (``fla``'s Triton dots keep Triton's TF32 default on both). How far
  round-off moves each gradient is measured, not assumed: the stock model is rerun with every parameter
  nudged one ulp, and a parameter's bound widens to a multiple of the median move those nudges cause
  where that is looser than the fixed bound. The median, because a nudge that flips a discrete choice
  (GLM-5 Next's indexer top-k below) moves many gradients wholesale, which the stack at this seed does
  not. A gated-delta-rule layer with a steep decay needs the widening (Qwen3.5's first layer at this
  seed, log-decay down to -25 per token): an fp32 chunked backward gets its decay gradient to ~1e-2
  only (``fla``'s and transformers' torch path alike), ``fla`` re-draws that error under any round-off
  change of its inputs, and the parameters reached only through the decay (``A_log``, ``dt_bias``,
  ``in_proj_a``) move by ~1e-3 in cosine within the stock model itself.
* bf16 against the fp32 stock model: the stack's bf16 loss error, and its median per-parameter gradient
  error, must stay within a small multiple of the stock model's own bf16 error, since two correct bf16
  implementations already differ. A per-parameter bound would not hold: the hyper-connection scales of
  DeepSeek-V4 and GLM-5 Next are sums with heavy cancellation, where every bf16 run is noise.

GLM-5 Next's sparse-attention indexer picks KV blocks by a top-k (``index_topk``) that fp32
reduction-order noise flips on some inputs, which moves a few gradients wholesale in the stock model
and the stack alike; the fixed seed below has no such near-tie.

    torchrun --nproc_per_node=1 tests/gpu/kernels/test_family_kernel_stack_numerics.py --family qwen3_moe
"""

from __future__ import annotations

import argparse
import copy
import os
import statistics
import tempfile

import torch
from safetensors.torch import save_file

from src.distributed.expert_parallel.base_layer import find_ep_layers
from src.distributed.expert_parallel.config import get_num_experts
from src.distributed.expert_parallel.layers import roster  # noqa: F401  (registers the EP families)
from src.distributed.expert_parallel.patching import patch_moe_model_for_ep
from src.kernels.liger.orchestrator import apply_liger_kernel
from src.models.loading.config_levels import text_config
from tests.common.ep_merge_oracle import post_process_merged_weights
from tests.common.harness import gpu_test_main, log, record_check
from tests.common.parallelism import single_process_ep_config
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model
from tests.common.utils import cos_sim, fro_rel_err

SEED = 0
BATCH, SEQ = 2, 96

# fp32: fused kernels accumulate in another order than eager, and FLCE/Liger CE chunk the loss.
FP32_LOSS_RTOL = 1e-4
FP32_GRAD_COS_MIN = 0.9999
FP32_GRAD_NORM_RTOL = 2e-2
# The stock model's own round-off spread, the median over this many one-ulp nudges; a bound widens to this
# multiple of it, never past the floor and ceiling below. On the steep-decay GDN layer the stack's deviation
# measured 0.87x the spread, 1.2e-3 in 1 - cos; on GLM-5 Next two nudges in eight flip the indexer top-k.
ROUNDOFF_PROBES = 8
ROUNDOFF_SPREAD_MULTIPLE = 4.0
ROUNDOFF_GRAD_COS_FLOOR = 0.99
ROUNDOFF_GRAD_NORM_RTOL_CEILING = 5e-2
# bf16: the stack's error against the fp32 stock model, relative to the stock model's own bf16 error.
# Across seeds the stack's median gradient error stays within these ratios: a single seed is noisy, a
# wrong precision or a dropped fp32 accumulation is not.
BF16_LOSS_ERROR_RATIO_MAX = 3.0
BF16_MEDIAN_GRAD_ERROR_RATIO_MAX = 2.0
BF16_ERROR_FLOOR = 5e-3  # below this both errors are rounding noise and the ratio divides by it


def _norm_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return abs(a.double().norm().item() - b.double().norm().item()) / max(b.double().norm().item(), 1e-30)


def _loss_and_grads(model, input_ids) -> tuple[float, dict[str, torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    out = model(input_ids=input_ids, labels=input_ids)
    out.loss.backward()
    grads = {n: p.grad.detach().float().cpu() for n, p in model.named_parameters() if p.grad is not None}
    return out.loss.item(), grads


def _hf_named_grads(model_type: str, model_cls: type, config, grads: dict[str, torch.Tensor]):
    """EP-wrapper gradients under the stock model's parameter names.

    The family's own merge maps the wrapper's tensors to the checkpoint layout, and the model class's
    ``from_pretrained`` then converts that layout to the live one, exactly as a real save and load would
    (the class itself, so a remote-code family needs no modeling file beside the saved config).
    Returns the gradients plus the keys the load reported missing and unexpected.
    """
    merged = post_process_merged_weights(dict(grads), model_type, verbose=False)
    with tempfile.TemporaryDirectory() as directory:
        config.save_pretrained(directory)
        save_file({k: v.contiguous() for k, v in merged.items()}, os.path.join(directory, "model.safetensors"))
        loaded, info = model_cls.from_pretrained(directory, dtype=torch.float32, output_loading_info=True)
    named = {n: p.detach().cpu() for n, p in loaded.named_parameters()}
    return named, sorted(info["missing_keys"]), sorted(info["unexpected_keys"])


def _stock_model(model_cls: type, config, state: dict, dtype: torch.dtype):
    model = model_cls(copy.deepcopy(config)).cuda()
    model.load_state_dict(state)
    return model.to(dtype)


def _stack_model(model_cls: type, config, state: dict, dtype: torch.dtype):
    """The model as the production loader assembles it: the applier's class swaps (already applied),
    the reference weights, the EP wrapper."""
    model = _stock_model(model_cls, config, state, dtype)
    patch_moe_model_for_ep(model, single_process_ep_config(get_num_experts(config), use_grouped_gemm=True))
    return model


def _round_off_probe(model_cls: type, config, state: dict, probe: int):
    """The fp32 stock model with every parameter moved to an adjacent fp32 value, up or down per element."""
    model = _stock_model(model_cls, config, state, torch.float32)
    generator = torch.Generator().manual_seed(probe)
    with torch.no_grad():
        for param in model.parameters():
            up = torch.rand(param.shape, generator=generator) < 0.5
            param.copy_(torch.nextafter(param, torch.where(up, torch.inf, -torch.inf).to(param.device)))
    return model


def _round_off_spread(model_cls: type, config, state: dict, input_ids, ref_grads: dict) -> dict:
    """Per parameter, the median move of the stock gradient under :data:`ROUNDOFF_PROBES` one-ulp nudges:
    ``1 - cos`` and the relative change of its norm. Same kernels and autotuned tiles as the reference."""
    moves = {name: [] for name in ref_grads}
    for probe in range(ROUNDOFF_PROBES):
        _, grads = _loss_and_grads(_round_off_probe(model_cls, config, state, probe), input_ids)
        for name, want in ref_grads.items():
            cos = cos_sim(grads[name], want, label=f"round-off probe {probe} grad {name}")
            moves[name].append((1 - cos, _norm_rel(grads[name], want)))
    return {
        name: {
            "one_minus_cos": statistics.median(cos_move for cos_move, _ in pairs),
            "norm_rel": statistics.median(norm_move for _, norm_move in pairs),
        }
        for name, pairs in moves.items()
    }


def run_family(model_type: str, seed: int = SEED) -> dict:
    """The stock fp32 reference and its round-off spread, and the stock bf16 reference, first; then the
    applier, then the stack in fp32 and bf16."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    family = TINY_MOE_FAMILIES[model_type]
    torch.manual_seed(seed)
    reference = tiny_family_model(family).cuda().float()
    model_cls, config = type(reference), reference.config
    vocab = text_config(config).vocab_size
    input_ids = torch.randint(0, vocab, (BATCH, SEQ), generator=torch.Generator().manual_seed(seed)).cuda()
    state = {k: v.clone() for k, v in reference.state_dict().items()}
    ref_loss, ref_grads = _loss_and_grads(reference, input_ids)
    spread = _round_off_spread(model_cls, config, state, input_ids, ref_grads)
    ref16_loss, ref16_grads = _loss_and_grads(_stock_model(model_cls, config, state, torch.bfloat16), input_ids)
    del reference

    applied = apply_liger_kernel(copy.deepcopy(config), None, needs_ep_wrappers=True)
    result = {"model_type": model_type, "ref_loss": ref_loss, "round_off_spread": spread}
    for label, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        model = _stack_model(model_cls, config, state, dtype)
        layers = find_ep_layers(model)
        loss, grads = _loss_and_grads(model, input_ids)
        named, not_loaded, unexpected = _hf_named_grads(model_type, model_cls, config, grads)
        missing = sorted((set(ref_grads) - set(named)) | (set(not_loaded) & set(ref_grads)))
        per_param = {}
        for name in sorted((set(ref_grads) & set(named)) - set(missing)):
            got, want = named[name], ref_grads[name]
            if got.shape != want.shape:
                per_param[name] = {"shape": [list(got.shape), list(want.shape)]}
                continue
            entry = {
                "cos": cos_sim(got, want, label=f"{model_type} {label} grad {name}"),
                "norm_rel": _norm_rel(got, want),
                "rel": fro_rel_err(got, want),
            }
            if label == "bf16":
                entry["stock_rel"] = fro_rel_err(ref16_grads[name], want)
            per_param[name] = entry
        result[label] = {
            "loss": loss,
            "stock_bf16_loss": ref16_loss,
            "liger": {k: v for k, v in (applied or {}).items() if v},
            "ep_layers": len(layers),
            "combines": sorted({layer._glu_combine_name() for _, layer in layers}),
            "missing": missing,
            "extra": unexpected,
            "grads": per_param,
        }
        del model
        torch.cuda.empty_cache()
    return result


def _fp32_bounds(spread: dict) -> tuple[float, float]:
    """A parameter's fp32 ``(cos_min, norm_rtol)``: the fixed bounds, widened to its round-off spread."""
    widened_cos = max(ROUNDOFF_GRAD_COS_FLOOR, 1 - ROUNDOFF_SPREAD_MULTIPLE * spread["one_minus_cos"])
    widened_norm = min(ROUNDOFF_GRAD_NORM_RTOL_CEILING, ROUNDOFF_SPREAD_MULTIPLE * spread["norm_rel"])
    return min(FP32_GRAD_COS_MIN, widened_cos), max(FP32_GRAD_NORM_RTOL, widened_norm)


def check_family_stack_matches_the_stock_model(model_type: str, result: dict) -> None:
    fp32, bf16 = result["fp32"], result["bf16"]
    assert fp32["ep_layers"] > 0, f"{model_type}: nothing was EP-wrapped, the check would prove nothing"

    for label in ("fp32", "bf16"):
        run = result[label]
        assert not run["missing"], f"{model_type} {label}: no stack gradient for {run['missing']}"
        assert not run["extra"], f"{model_type} {label}: stack gradients with no stock parameter {run['extra']}"
        shapes = {n: e["shape"] for n, e in run["grads"].items() if "shape" in e}
        assert not shapes, f"{model_type} {label}: gradient shapes differ {shapes}"

    loss_rel = abs(fp32["loss"] - result["ref_loss"]) / abs(result["ref_loss"])
    assert loss_rel < FP32_LOSS_RTOL, f"{model_type}: fp32 loss {fp32['loss']} vs stock {result['ref_loss']}"
    for name, entry in fp32["grads"].items():
        cos_min, norm_rtol = _fp32_bounds(result["round_off_spread"][name])
        assert entry["cos"] > cos_min, (
            f"{model_type}: fp32 grad {name} cos {entry['cos']:.6f}, bound {cos_min:.6f} "
            f"(stock round-off spread {result['round_off_spread'][name]['one_minus_cos']:.1e})"
        )
        assert entry["norm_rel"] < norm_rtol, (
            f"{model_type}: fp32 grad {name} norm off {entry['norm_rel']:.2e}, bound {norm_rtol:.2e}"
        )

    stack_loss_err = abs(bf16["loss"] - result["ref_loss"])
    stock_loss_err = abs(bf16["stock_bf16_loss"] - result["ref_loss"])
    loss_floor = 1e-3 * abs(result["ref_loss"])
    assert stack_loss_err <= BF16_LOSS_ERROR_RATIO_MAX * stock_loss_err + loss_floor, (
        f"{model_type}: bf16 loss error {stack_loss_err:.2e} vs stock bf16 {stock_loss_err:.2e}"
    )
    stack_median = statistics.median(entry["rel"] for entry in bf16["grads"].values())
    stock_median = statistics.median(entry["stock_rel"] for entry in bf16["grads"].values())
    assert stack_median <= BF16_MEDIAN_GRAD_ERROR_RATIO_MAX * stock_median + BF16_ERROR_FLOOR, (
        f"{model_type}: bf16 median gradient error {stack_median:.3e} vs stock bf16 {stock_median:.3e}"
    )


@gpu_test_main(exact_world_size=1, prefix="family_kernel_stack")
def run(ctx) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(TINY_MOE_FAMILIES), required=True)
    model_type = parser.parse_args().family
    checks: dict[str, bool] = {}
    result = run_family(model_type)
    fixed = (FP32_GRAD_COS_MIN, FP32_GRAD_NORM_RTOL)
    widened = {n: _fp32_bounds(s) for n, s in result["round_off_spread"].items() if _fp32_bounds(s) != fixed}
    if widened:
        loosest = min(widened, key=lambda name: widened[name][0])
        log(
            f"{model_type}: {len(widened)} fp32 bounds widened by the stock round-off spread {sorted(widened)}; "
            f"loosest (cos_min, norm_rtol) {widened[loosest]} for {loosest}"
        )
    record_check(
        checks,
        f"stack_matches_the_stock_model[{model_type}]",
        lambda: check_family_stack_matches_the_stock_model(model_type, result),
    )
    return {"checks": checks}


if __name__ == "__main__":
    run()
