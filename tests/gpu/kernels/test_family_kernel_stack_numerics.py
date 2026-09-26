#!/usr/bin/env python
"""Every MoE family, through the whole production kernel stack, computes the stock model's loss and
gradients.

The per-kernel suites pin each kernel against its own reference: ``test_fused_glu.py`` (the GLU
combines), ``test_moe_permute.py`` (the fused un-permute), ``test_liger_family_kernels.py`` (each Liger
role). This file checks that nothing breaks
where they meet, per family: the family's Liger applier as the loader calls it, the EP wrapper with
grouped GEMM (ep_size 1, so every expert is local and the fused weighted un-permute runs), and the
attention implementation an SDPA run resolves to. A family whose wrapper latched the wrong combine, whose
norm took the wrong casting mode, or whose packed GLU read the wrong half moves the loss or a gradient by far more than the tolerances below.

Two comparisons per family, both against the stock Hugging Face model built from the same weights:

* fp32 against fp32: the stack must match to fp32 round-off (reduction order, fused-kernel accumulation);
  TF32 is off on both sides.
* bf16 against the fp32 stock model: the stack's bf16 loss error, and its median per-parameter gradient
  error, must stay within a small multiple of the stock model's own bf16 error, since two correct bf16
  implementations already differ. A per-parameter bound would not hold: the hyper-connection scales of
  DeepSeek-V4 and GLM-5 Next are sums with heavy cancellation, where every bf16 run is noise.

GLM-5 Next's sparse-attention indexer picks KV blocks by a top-k (``index_topk``) that fp32
reduction-order noise flips on some inputs, which moves a few gradients wholesale in the stock model
and the stack alike; the fixed seed below has no such near-tie.

Each family runs in its own subprocess: an applier rebinds the family's HF classes (and transformers'
loss) for the rest of the process, which would turn every later family's "stock" reference into a
patched one.

    python tests/gpu/kernels/test_family_kernel_stack_numerics.py          # every family (pytest)
    python tests/gpu/kernels/test_family_kernel_stack_numerics.py --family qwen3_moe   # one, prints JSON
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from collections.abc import Callable
from typing import NamedTuple

import pytest
import torch

from tests.common import models as tiny_models

SEED = 0
BATCH, SEQ = 2, 96


class _Family(NamedTuple):
    """How to build one family's tiny model: its config (from the tiny configs), the Auto class that maps
    it, and a hook run on the freshly initialized reference (whose state the stack model then loads)."""

    config: Callable[[], object]
    auto: str = "AutoModelForCausalLM"
    prepare: Callable[[torch.nn.Module], None] | None = None


def _flat(model_type: str, tiny: str) -> Callable[[], object]:
    def build():
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        return CONFIG_MAPPING[model_type](**dict(getattr(tiny_models, tiny)))

    return build


def _composite(model_type: str, text: str, vision: str, **extra) -> Callable[[], object]:
    def build():
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        return CONFIG_MAPPING[model_type](
            text_config=dict(getattr(tiny_models, text)), vision_config=dict(getattr(tiny_models, vision)), **extra
        )

    return build


def _randomize_tid2eid(model: torch.nn.Module) -> None:
    """Random init leaves DeepSeek-V4's hash router table all-zero, which the EP wrapper refuses."""
    from tests.cpu.models.test_deepseek_v4_support import randomize_tid2eid

    randomize_tid2eid(model)


# The text model_type for Qwen3.5 MoE (its wrapper config ignores flat decoder kwargs); the composite
# VLM wrappers for GLM-5 Next and Step-3.7, which register no causal-LM class.
FAMILIES = {
    "gpt_oss": _Family(_flat("gpt_oss", "TINY_GPTOSS_CONFIG")),
    "qwen3_moe": _Family(_flat("qwen3_moe", "TINY_QWEN3_MOE_CONFIG")),
    "glm4_moe_lite": _Family(_flat("glm4_moe_lite", "TINY_GLM4_MOE_LITE_CONFIG")),
    "gemma4_text": _Family(_flat("gemma4_text", "TINY_GEMMA4_MOE_CONFIG")),
    "qwen3_5_moe_text": _Family(_flat("qwen3_5_moe_text", "TINY_QWEN35_MOE_CONFIG")),
    "cohere2_moe": _Family(_flat("cohere2_moe", "TINY_COHERE2_MOE_CONFIG")),
    "laguna": _Family(_flat("laguna", "TINY_LAGUNA_CONFIG")),
    "lfm2_moe": _Family(_flat("lfm2_moe", "TINY_LFM2_MOE_CONFIG")),
    "deepseek_v4": _Family(_flat("deepseek_v4", "TINY_DSV4_CONFIG"), prepare=_randomize_tid2eid),
    "step3p7": _Family(
        _composite("step3p7", "TINY_STEP3P7_CONFIG", "TINY_STEP3P7_VISION_CONFIG", image_token_id=2000),
        auto="AutoModelForImageTextToText",
    ),
    "glm5_next": _Family(
        _composite("glm5_next", "TINY_GLM5_CONFIG", "TINY_GLM5_VISION_CONFIG"), auto="AutoModelForImageTextToText"
    ),
    "zaya": _Family(_flat("zaya", "TINY_ZAYA_CONFIG")),
}

# fp32: fused kernels accumulate in another order than eager, and FLCE/Liger CE chunk the loss. Gradients
# agree to ~1e-6 in norm except through the linear-attention (fla) kernels of Qwen3.5 and GLM-5 Next. Their
# per-head scalar parameters (`A_log`, `dt_bias`) sum heavily cancelling terms, and fla picks its Triton tiles
# by timing, so the same seed measured 1.1e-2 to 2.1e-2 on Qwen3.5 depending on the tiles chosen.
FP32_LOSS_RTOL = 1e-4
FP32_GRAD_COS_MIN = 0.9999
FP32_GRAD_NORM_RTOL = 2e-2
FLA_SCALAR_PARAMS = ("A_log", "dt_bias")
FP32_FLA_SCALAR_GRAD_NORM_RTOL = 5e-2
# bf16: the stack's error against the fp32 stock model, relative to the stock model's own bf16 error.
# Over five seeds per family the stack's median gradient error measured 0.37-1.71x the stock model's:
# a single seed is noisy, a wrong precision or a dropped fp32 accumulation is not.
BF16_LOSS_ERROR_RATIO_MAX = 3.0
BF16_MEDIAN_GRAD_ERROR_RATIO_MAX = 2.0
BF16_ERROR_FLOOR = 5e-3  # below this both errors are rounding noise and the ratio divides by it


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    denom = a.norm() * b.norm()
    return 1.0 if denom == 0 else (a @ b / denom).item()


def _norm_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return abs(a.double().norm().item() - b.double().norm().item()) / max(b.double().norm().item(), 1e-30)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm().clamp(min=1e-30)).item()


# ---------------------------------------------------------------------------------------------------------
# Subprocess body: one family.
# ---------------------------------------------------------------------------------------------------------


def _init_single_process_group() -> None:
    import torch.distributed as dist

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29700 + os.getpid() % 200))
        dist.init_process_group("nccl", rank=0, world_size=1)


def _build_config(model_type: str, attn: str):
    config = FAMILIES[model_type].config()
    config._attn_implementation = attn
    return config


def _auto(model_type: str):
    import transformers

    return getattr(transformers, FAMILIES[model_type].auto)


def _stock_attn(model_type: str, config) -> str:
    """The reference attention: SDPA where the family supports it, eager otherwise."""
    model_cls = _auto(model_type)._model_mapping[type(config)]
    return "sdpa" if getattr(model_cls, "_supports_sdpa", False) else "eager"


def _loss_and_grads(model, input_ids) -> tuple[float, torch.Tensor, dict[str, torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    out = model(input_ids=input_ids, labels=input_ids)
    out.loss.backward()
    grads = {n: p.grad.detach().float().cpu() for n, p in model.named_parameters() if p.grad is not None}
    return out.loss.item(), grads


def _hf_named_grads(model_type: str, config, grads: dict[str, torch.Tensor]) -> tuple[dict, list, list]:
    """EP-wrapper gradients under the stock model's parameter names.

    The family's own merge maps the wrapper's tensors to the checkpoint layout, and the stock
    ``from_pretrained`` then converts that layout to the live one, exactly as a real save and load would.
    Returns the gradients plus the keys the load reported missing and unexpected.
    """
    import tempfile

    from safetensors.torch import save_file

    from tests.common.ep_merge_oracle import post_process_merged_weights

    merged = post_process_merged_weights(dict(grads), model_type, verbose=False)
    with tempfile.TemporaryDirectory() as directory:
        config.save_pretrained(directory)
        save_file({k: v.contiguous() for k, v in merged.items()}, os.path.join(directory, "model.safetensors"))
        loaded, info = _auto(model_type).from_pretrained(directory, dtype=torch.float32, output_loading_info=True)
    named = {n: p.detach().cpu() for n, p in loaded.named_parameters()}
    return named, sorted(info["missing_keys"]), sorted(info["unexpected_keys"])


def _stack_model(model_type: str, reference_state: dict, dtype: torch.dtype, attn: str):
    """The model as the production loader assembles it: applier first (class swaps), weights, EP wrapper."""
    from src.distributed.expert_parallel.config import EPConfig
    from src.distributed.expert_parallel.patching import patch_moe_model_for_ep
    from src.kernels.liger.orchestrator import apply_liger_kernel

    config = _build_config(model_type, attn)
    applied = apply_liger_kernel(config, None, needs_ep_wrappers=True)
    stack_attn = attn
    torch.manual_seed(SEED)
    model = _auto(model_type).from_config(config).cuda()
    model.load_state_dict(reference_state)
    model = model.to(dtype)
    ep_config = EPConfig(ep_size=1, world_size=1, gpus_per_node=1, use_grouped_gemm=True)
    ep_config.finalize_expert_assignment(_num_experts(config))
    patch_moe_model_for_ep(model, ep_config)
    return model, applied, stack_attn


def _num_experts(config) -> int:
    from src.models.loading.config_levels import text_config

    config = text_config(config)
    for name in ("num_experts", "num_local_experts", "n_routed_experts"):
        value = getattr(config, name, None)
        if value:
            return value
    raise AssertionError(f"{type(config).__name__} names no expert count")


def run_family(model_type: str, seed: int = SEED) -> dict:
    from accelerate import PartialState

    from src.distributed.expert_parallel.base_layer import find_ep_layers
    from src.distributed.expert_parallel.layers import roster  # noqa: F401  (registers the EP families)
    from src.models.loading.config_levels import text_config

    PartialState()
    _init_single_process_group()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    family = FAMILIES[model_type]
    probe = _build_config(model_type, "eager")
    attn = _stock_attn(model_type, probe)
    torch.manual_seed(seed)
    reference = _auto(model_type).from_config(_build_config(model_type, attn))
    if family.prepare is not None:
        family.prepare(reference)
    reference = reference.cuda().float()
    vocab = text_config(probe).vocab_size
    input_ids = torch.randint(0, vocab, (BATCH, SEQ), generator=torch.Generator().manual_seed(seed)).cuda()
    state = {k: v.clone() for k, v in reference.state_dict().items()}

    ref_loss, ref_grads = _loss_and_grads(reference, input_ids)
    reference_bf16 = _auto(model_type).from_config(_build_config(model_type, attn)).cuda()
    reference_bf16.load_state_dict(state)
    ref16_loss, ref16_grads = _loss_and_grads(reference_bf16.to(torch.bfloat16), input_ids)
    del reference, reference_bf16

    result = {"model_type": model_type, "reference_attn": attn, "ref_loss": ref_loss}
    for label, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        model, applied, stack_attn = _stack_model(model_type, state, dtype, attn)
        layers = find_ep_layers(model)
        loss, grads = _loss_and_grads(model, input_ids)
        named, not_loaded, unexpected = _hf_named_grads(model_type, _build_config(model_type, attn), grads)
        missing = sorted((set(ref_grads) - set(named)) | (set(not_loaded) & set(ref_grads)))
        extra = unexpected
        per_param = {}
        for name in sorted((set(ref_grads) & set(named)) - set(missing)):
            got, want = named[name], ref_grads[name]
            if got.shape != want.shape:
                per_param[name] = {"shape": [list(got.shape), list(want.shape)]}
                continue
            entry = {"cos": _cos(got, want), "norm_rel": _norm_rel(got, want), "rel": _rel(got, want)}
            if label == "bf16":
                entry["stock_rel"] = _rel(ref16_grads[name], want)
            per_param[name] = entry
        result[label] = {
            "loss": loss,
            "stock_bf16_loss": ref16_loss,
            "stack_attn": stack_attn,
            "liger": {k: v for k, v in (applied or {}).items() if v},
            "ep_layers": len(layers),
            "combines": sorted({layer._glu_combine_name() for _, layer in layers}),
            "missing": missing,
            "extra": extra,
            "grads": per_param,
        }
        del model
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------------------------------------
# pytest: one subprocess per family.
# ---------------------------------------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_RESULT_MARKER = "FAMILY_RESULT:"


def _family_result(model_type: str) -> dict:
    env = {**os.environ, "PYTHONPATH": os.getcwd() + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.run(
        [sys.executable, __file__, "--family", model_type], capture_output=True, text=True, env=env, timeout=900
    )
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER) :])
    raise AssertionError(
        f"{model_type}: no result (exit {proc.returncode})\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    )


@pytest.mark.parametrize("model_type", sorted(FAMILIES))
def test_family_stack_matches_the_stock_model(model_type):
    result = _family_result(model_type)
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


if __name__ == "__main__":
    if "--family" in sys.argv:
        family = sys.argv[sys.argv.index("--family") + 1]
        seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else SEED
        print(_RESULT_MARKER + json.dumps(run_family(family, seed)), flush=True)
    else:
        raise SystemExit(pytest.main([__file__, "-v"]))
