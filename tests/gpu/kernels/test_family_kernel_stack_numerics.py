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
  TF32 is off on both sides.
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
from tests.common.harness import gpu_test_main, record_check
from tests.common.parallelism import single_process_ep_config
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model
from tests.common.utils import cos_sim, fro_rel_err

SEED = 0
BATCH, SEQ = 2, 96

# fp32: fused kernels accumulate in another order than eager, and FLCE/Liger CE chunk the loss. Gradients
# agree to ~1e-6 in norm except through the linear-attention (fla) kernels of Qwen3.5 and GLM-5 Next. Their
# per-head scalar parameters (`A_log`, `dt_bias`) sum heavily cancelling terms, and fla picks its Triton tiles
# by timing, so their error moves with the tiles a run picks; they get a looser bound.
FP32_LOSS_RTOL = 1e-4
FP32_GRAD_COS_MIN = 0.9999
FP32_GRAD_NORM_RTOL = 2e-2
FLA_SCALAR_PARAMS = ("A_log", "dt_bias")
FP32_FLA_SCALAR_GRAD_NORM_RTOL = 5e-2
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


def run_family(model_type: str, seed: int = SEED) -> dict:
    """Stock fp32 and bf16 references first, then the applier, then the stack in fp32 and bf16."""
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
    ref16_loss, ref16_grads = _loss_and_grads(_stock_model(model_cls, config, state, torch.bfloat16), input_ids)
    del reference

    applied = apply_liger_kernel(copy.deepcopy(config), None, needs_ep_wrappers=True)
    result = {"model_type": model_type, "ref_loss": ref_loss}
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
        assert entry["cos"] > FP32_GRAD_COS_MIN, f"{model_type}: fp32 grad {name} cos {entry['cos']:.6f}"
        norm_rtol = FP32_FLA_SCALAR_GRAD_NORM_RTOL if name.endswith(FLA_SCALAR_PARAMS) else FP32_GRAD_NORM_RTOL
        assert entry["norm_rel"] < norm_rtol, f"{model_type}: fp32 grad {name} norm off {entry['norm_rel']:.2e}"

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
    record_check(
        checks,
        f"stack_matches_the_stock_model[{model_type}]",
        lambda: check_family_stack_matches_the_stock_model(model_type, result),
    )
    return {"checks": checks}


if __name__ == "__main__":
    run()
