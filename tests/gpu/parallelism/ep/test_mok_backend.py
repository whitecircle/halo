#!/usr/bin/env python
"""``moe_backend: mok`` against the DeepEP path on one MoE block, at as many EP ranks as the launcher gives (1 or 4):
the same weights and tokens give the same output and the same token, router and expert-weight gradients within bf16
tolerance. ``--family qwen3`` stores separate GLU banks, ``qwen3_5`` a fused ``gate_up_proj`` beside a gated shared
expert, which stays on the layer's own path. ``--mxfp8`` runs MoK's MXFP8 forward and backward, which must land in fp8's
band of the bf16 reference instead. The token count is off MoK's 256 multiple, so the padded tail is checked.
``--skew`` routes every token of every rank to rank 0's experts: the most rows one rank can schedule, which a schedule
sized to the group's rows alone overflows (MoK then traps in its kernel), and more than its ring of routed rows holds.
"""

import argparse
import copy
import importlib
import sys

import torch
import torch.nn as nn
import triton
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from halo.distributed.expert_parallel import mok as mok_module
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.expert_parallel.mok import MoKExperts, require_mok
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.kernels.batched_transpose import transpose_last_two
from halo.kernels.lowp.spec import LowpSpec
from halo.optimizers.adamw_bf16 import AdamWBF16
from tests.common.harness import gpu_test_main
from tests.common.numerics import cos_sim, fro_rel_err
from tests.common.rank_log import log

SEED = 0
HIDDEN, INTERMEDIATE, EXPERTS, TOP_K = 512, 256, 32, 4
TOKENS = 1000  # padded to 1024
# Both paths accumulate in fp32 and round to bf16, in different orders and tilings.
REL_MAX, COS_MIN = 2e-2, 0.9995
# MXFP8 against bf16; the lower bound fails a run that silently stayed bf16. Weight gradients sum many fp8 products
# and land well inside it; the shared expert's stay bf16 on both paths.
MXFP8_BAND = (5e-3, 0.1)
MXFP8 = LowpSpec(precision="fp8", backend="native", backward_precision="fp8", apply_dense_mlp=False)
SIZES = {"hidden_size": HIDDEN, "moe_intermediate_size": INTERMEDIATE, "num_experts": EXPERTS}
FAMILIES = {
    "qwen3": lambda: Qwen3MoeSparseMoeBlock(Qwen3MoeConfig(**SIZES, num_experts_per_tok=TOP_K, norm_topk_prob=True)),
    "qwen3_5": lambda: Qwen3_5MoeSparseMoeBlock(
        Qwen3_5MoeTextConfig(**SIZES, num_experts_per_tok=TOP_K, shared_expert_intermediate_size=INTERMEDIATE)
    ),
}

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=tuple(FAMILIES), default="qwen3")
parser.add_argument("--skew", action="store_true")
parser.add_argument("--ep-size", type=int, help="The EP ranks this row launches at, checked against the launcher's")
parser.add_argument("--mxfp8", action="store_true", help="Run MoK's MXFP8 forward and backward")
parser.add_argument("--ranks", type=int, help="The ranks an --ep-size 1 row launches, each its own EP group")
ARGS = parser.parse_args()


def _ep_layer(block: nn.Module, backend: str, ep_size: int, lowp: LowpSpec | None = None) -> nn.Module:
    pc = ParallelismConfig(
        ep_size=ep_size,
        moe_backend=backend,
        ep_rows_per_device=1,
        ep_declared_max_length=TOKENS,
        fsdp_shard_ep1_experts=False,
        lowp=lowp or LowpSpec(),
    )
    holder = nn.Module()
    holder.mlp = copy.deepcopy(block)
    layer = apply_ep_layers(holder, pc.ep_config()).mlp
    if lowp is not None:
        layer.set_lowp_compute(lowp, weight_cacheable=True)
    return layer


def _route_to_rank_0(block: nn.Module, x: torch.Tensor, ep_size: int) -> torch.Tensor:
    """A router whose rank-0 experts score a shared direction and the rest its negative, and tokens carrying enough
    of that direction that their top-k never leaves rank 0."""
    direction = torch.randn(HIDDEN, device=x.device, generator=torch.Generator(x.device).manual_seed(SEED))
    direction = direction / direction.norm()
    local = EXPERTS // ep_size
    with torch.no_grad():
        router = block.gate.weight
        router[:local] += 0.5 * direction.to(router.dtype)
        router[local:] -= 0.5 * direction.to(router.dtype)
    return x + 8 * direction.to(x.dtype)


def _step(layer: nn.Module, x: torch.Tensor, grad: torch.Tensor) -> dict[str, torch.Tensor]:
    x = x.clone().requires_grad_(True)
    out = layer(x)
    out.backward(grad)
    return {
        "output": out.detach(),
        "grad_tokens": x.grad,
        **{f"grad_{name}": param.grad for name, param in layer.named_parameters() if param.grad is not None},
    }


routed = []  # the global expert ids of every layer forward
_record_expert_load = EPMoELayerBase._record_expert_load


def _recording_expert_load(self, experts):
    routed.append(experts.detach().clone())
    return _record_expert_load(self, experts)


EPMoELayerBase._record_expert_load = _recording_expert_load

warm_ups = []  # one entry per MoK warm-up that returned
compiled_past_the_warm_up = []  # Triton kernels the MoK step compiled after it
_warm = MoKExperts.warm


def _recording_warm(self, banks, top_k, mxfp8):
    _warm(self, banks, top_k, mxfp8)
    warm_ups.append(top_k)


MoKExperts.warm = _recording_warm


launched_in_the_warm_up, launched_past_it = set(), set()  # MoK's compiled entry points, by name


class _RecordedEntryPoints:
    """Stands in for ``mok.ops._C``. MoK compiles nothing at run time and CUDA loads each of its kernels at that
    kernel's first launch, so the step is warm when every entry point it calls ran in the warm-up."""

    def __init__(self, compiled):
        self._compiled = compiled

    def __getattr__(self, name):
        (launched_past_it if warm_ups else launched_in_the_warm_up).add(name)
        return getattr(self._compiled, name)


def _record_compile(*, fn, **_):
    if warm_ups:
        compiled_past_the_warm_up.append(fn.name)
    return False  # a truthy return would skip the compile


def _count_calls(module, name: str) -> list[int]:
    """Replace ``module.name`` for the process with a wrapper counting its calls."""
    real, calls = getattr(module, name), [0]

    def counted(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    setattr(module, name, counted)
    return calls


def _bits_equal(left: tuple[torch.Tensor, ...], right: tuple[torch.Tensor, ...]) -> bool:
    return all(
        torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
        for a, b in zip(left, right, strict=True)
    )


def _mxfp8_cast_checks(layer: nn.Module) -> dict[str, bool]:
    """The cast the layer's layout takes is MoK's own of the transposed bank, bit for bit, and the per-step cache
    serves a cast of the weights as they are after a fused AdamWBF16 step and after a ``load_state_dict``."""
    weights = tuple(param for _, param in layer.expert_named_params())
    quantize = mok_module._mxfp8_quantize()

    def cache_is_current() -> bool:
        served = mok_module._mxfp8_banks(weights).backward[0]
        return _bits_equal(served, mok_module._mxfp8_layer_cast(mok_module._layer_banks(weights)[0]))

    gate = mok_module._layer_banks(weights)[0]
    checks = {
        "the_layer_layout_cast_is_mok_s_cast_of_the_transposed_bank": _bits_equal(
            mok_module._mxfp8_layer_cast(gate), quantize(transpose_last_two(gate.detach()).contiguous(), True, True)
        ),
        "the_cache_served_the_step_s_cast": cache_is_current(),
    }
    served = [tuple(t.clone() for t in mok_module._mxfp8_banks(weights).backward[0])]
    AdamWBF16(layer.expert_named_params(), lr=1e-2).step()
    served.append(tuple(t.clone() for t in mok_module._mxfp8_banks(weights).backward[0]))
    checks["an_adamw_bf16_step_recasts_the_banks"] = cache_is_current() and not _bits_equal(*served)
    layer.load_state_dict({name: value * 0.75 for name, value in layer.state_dict().items()})
    recast = not _bits_equal(served[1], mok_module._mxfp8_banks(weights).backward[0])
    checks["a_load_state_dict_recasts_the_banks"] = cache_is_current() and recast
    return checks


_VARIANT = f"{'_mxfp8' if ARGS.mxfp8 else ''}{'_skew' if ARGS.skew else ''}"


@gpu_test_main(prefix=f"mok_backend_{ARGS.family}{_VARIANT}", partial_state=False)
def run(ctx) -> dict:
    torch.manual_seed(SEED)
    block = FAMILIES[ARGS.family]().to(ctx.device, torch.bfloat16)
    for param in block.parameters():
        nn.init.normal_(param, std=0.02)
    torch.manual_seed(SEED + 1 + ctx.rank)  # each rank routes its own tokens
    x = torch.randn(1, TOKENS, HIDDEN, device=ctx.device, dtype=torch.bfloat16)
    grad = torch.randn_like(x)
    if ARGS.skew:
        x = _route_to_rank_0(block, x, ctx.world_size)

    mok_ops = importlib.import_module("mok.ops")
    ep_size = 1 if ARGS.ranks else ctx.world_size
    reference = _step(_ep_layer(block, "deepep", ep_size), x, grad)
    previous, triton.knobs.runtime.jit_cache_hook = triton.knobs.runtime.jit_cache_hook, _record_compile
    mok_ops._C = _RecordedEntryPoints(mok_ops._C)
    try:
        mok_layer = _ep_layer(block, "mok", ep_size, MXFP8 if ARGS.mxfp8 else None)
        # An evaluation first: its cast, made under inference mode, serves the training step. The step runs under
        # saved-tensor hooks, as in a non-reentrant checkpoint, which hand its backward other objects for the weights.
        if ARGS.mxfp8:
            with torch.inference_mode():
                mok_layer(x)
        bank_casts = _count_calls(mok_module, "_mxfp8_layer_cast")
        with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
            mok = _step(mok_layer, x, grad)
        step_casts = bank_casts[0]
    finally:
        triton.knobs.runtime.jit_cache_hook = previous
        mok_ops._C = mok_ops._C._compiled

    # A bank no token reached has no gradient on the DeepEP path and a zero one from MoK's kernels.
    unreached = reference.keys() ^ mok.keys()
    checks = {
        "launched_at_the_rows_ep_size": (ARGS.ranks or ARGS.ep_size) in (None, ctx.world_size),
        "a_gradient_only_one_path_returns_is_zero": all(
            not (reference.get(name, mok.get(name))).any() for name in unreached
        ),
        "the_output_is_finite_and_nonzero": bool(reference["output"].isfinite().all() and reference["output"].any()),
        "the_warm_up_leaves_the_step_no_triton_kernel_to_compile": warm_ups == [TOP_K]
        and not compiled_past_the_warm_up,
        "the_step_calls_only_entry_points_the_warm_up_ran": bool(launched_past_it)
        and launched_past_it <= launched_in_the_warm_up,
    }
    if ARGS.skew:
        local = EXPERTS // ctx.world_size
        checks["every_token_routed_to_rank_0"] = len(routed) == 2 + ARGS.mxfp8 and all(
            bool((ids < local).all()) for ids in routed
        )
        checks["only_rank_0_takes_expert_gradients"] = (ctx.rank == 0) == ("grad_down_proj" in reference)
    if ARGS.mxfp8:
        checks["the_step_casts_no_bank_the_evaluation_cast"] = step_casts == 0
        checks |= _mxfp8_cast_checks(mok_layer)
    metrics = {}
    for name, expected in reference.items():
        if name in unreached:
            continue
        rel = fro_rel_err(mok[name].float(), expected.float())
        cos = cos_sim(mok[name].float(), expected.float(), label=name)
        metrics[f"{name}_rel"] = rel
        if ARGS.mxfp8 and "shared_expert" not in name:
            checks[f"{name}_in_mxfp8_band"] = MXFP8_BAND[0] < rel < MXFP8_BAND[1]
        else:
            checks[f"{name}_matches_deepep"] = rel < REL_MAX and cos > COS_MIN
        log(f"  {name:34s} rel {rel:.2e} cos {cos:.6f} norm {expected.float().norm():.3e}")
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    try:
        require_mok()
    except RuntimeError as exc:  # absent (Hopper image) or built for another GPU (B200)
        log(f"SKIP: {exc}")
        sys.exit(0)
    run()
