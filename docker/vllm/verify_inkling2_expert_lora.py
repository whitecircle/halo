"""GPU check of ``patch_vllm_inkling2``: the patched MXFP8 experts under vLLM's ``Inkling2MoEWithLoRA`` against a torch
reference over the dequantized weights, at Inkling-2-Small's expert shapes, one batch per routing path (fused, scatter)
and token tile, with two adapter slots of different ranks and tokens on each slot and on none.

Run on one Blackwell GPU in vLLM's Inkling 2 image, after ``patch_vllm_inkling2.py``:

    python3 verify_inkling2_expert_lora.py           # correctness
    python3 verify_inkling2_expert_lora.py --bench   # also the forward time with and without the adapters
"""

import os
import sys
import types

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.models.inkling2.lora import Inkling2MoEWithLoRA
from vllm.models.inkling2.moe import Inkling2Experts, Inkling2Gate
from vllm.models.inkling2.ops.moe import route
from vllm.models.inkling2.ops.mxfp8 import mxfp8_quantize

CONFIG = types.SimpleNamespace(
    hidden_size=4608,
    intermediate_size=2304,
    n_routed_experts=384,
    num_experts_per_tok=8,
    mlp_gate_softcap=4.0,
    mlp_up_softcap=25.0,
    route_scale=8.0,
    use_gate_bias=True,
    n_shared_experts=0,
)
# The slots' ranks: the first below the slot width, so its padding columns are read as zeros.
RANKS = (8, 16)
# Each token's slot: the first, the second, none.
TOKEN_SLOTS = (0, 1, -1)
# One per routing path and token tile: the fused route (tile 8), then the scatter route at tiles 32 and 64.
TOKENS = (8, 300, 700)
# Relative error of the expert output with LoRA against the reference, which models both GEMMs' MXFP8 inputs: the base
# output alone is 1.4% off it (the kernels' approximate sigmoid and tanh round some activations to other MXFP8 steps).
# The deltas are 10% to 40% of the output, so one dropped, or B's gate and up rows taken in the wrong order, fails.
TOLERANCE = 0.02


def _dequantize(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    exponents = torch.exp2(scales.float() - 127.0)
    return (values.float().unflatten(-1, (-1, 32)) * exponents[..., None]).flatten(-2)


def _mxfp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``weight`` [E, N, K] as MXFP8 values, row-major scales and their dequantization."""
    values, scales = mxfp8_quantize(weight.flatten(0, 1).contiguous())
    values, scales = values.view(*weight.shape), scales.view(*weight.shape[:2], -1)
    return values, scales, _dequantize(values, scales)


class _Adapter(types.SimpleNamespace):
    """One slot's adapters: ``a`` [E, r, in], ``b`` [E, out, r] per projection."""


def _softcap(x, cap):
    return cap * torch.tanh(x / cap)


def _reference(x, routing, plan, w13, w2, adapters, token_slots):
    """fp32 expert output from the routing ``route`` chose: each routed (token, expert) row through the dequantized
    weights, with both GEMMs' inputs rounded to MXFP8 as the kernels round them, and each token's slot's LoRA deltas
    over the bf16 ones."""
    src2dst, gammas, _rows, tiles = routing
    x_values, x_scales = mxfp8_quantize(x)
    x_q = _dequantize(x_values, x_scales)
    flat = (src2dst >= 0).nonzero().squeeze(1)
    rows = src2dst[flat].long()
    tokens = flat // CONFIG.num_experts_per_tok
    experts = tiles[rows // plan.tile].long()
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for expert in experts.unique().tolist():
        pick = experts == expert
        token = tokens[pick]
        slots = token_slots[token] if adapters else None
        gate_up = x_q[token] @ w13[expert].T
        for slot, lora in enumerate(adapters):
            on = slots == slot
            gate_up[on] += (x[token[on]].float() @ lora.a13[expert].float().T) @ lora.b13[expert].float().T
        gate, up = gate_up[:, 0::2], gate_up[:, 1::2]
        act = _softcap(gate, CONFIG.mlp_gate_softcap) * torch.sigmoid(gate) * _softcap(up, CONFIG.mlp_up_softcap)
        act = (act * gammas[rows[pick]][:, None]).bfloat16()
        down = _dequantize(*mxfp8_quantize(act)) @ w2[expert].T
        act = act.float()
        for slot, lora in enumerate(adapters):
            on = slots == slot
            down[on] += (act[on] @ lora.a2[expert].float().T) @ lora.b2[expert].float().T
        out.index_add_(0, token, down)
    return out


def _forward(x, gate, experts, lora):
    plan = experts.plan(x.shape[0])
    routing, quantized = route(
        x,
        gate.weight,
        gate.bias,
        gate.counters,
        experts.counts,
        gate.topk,
        gate.route_scale,
        experts.expert_bounds,
        plan,
        *experts.scales("w13"),
    )
    return experts(x, routing, quantized, plan, True, None, lora).float(), routing, plan


def _relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got - want).norm() / want.norm()).item()


def _wrapper(gate, experts, adapters, token_slots) -> Inkling2MoEWithLoRA:
    """vLLM's LoRA layer over ``experts`` with ``adapters`` in its slots and ``token_slots`` as the batch's mapping."""
    layer = Inkling2MoEWithLoRA(types.SimpleNamespace(gate=gate, experts=experts))
    layer.create_lora_weights(
        len(adapters), types.SimpleNamespace(max_lora_rank=max(RANKS), lora_dtype=torch.bfloat16)
    )
    for slot, lora in enumerate(adapters):
        layer.set_lora(slot, [lora.a13, lora.a2], [lora.b13, lora.b2])
    layer.set_mapping(types.SimpleNamespace(_token_lora_indices=token_slots))
    return layer


def _bench(gate, experts, layer, token_slots) -> None:
    """Median expert-forward time without and with the adapters, per batch size."""
    for tokens in (8, 256, 2048, 8192):
        x = torch.randn(tokens, CONFIG.hidden_size, device="cuda", dtype=torch.bfloat16)
        layer.set_mapping(types.SimpleNamespace(_token_lora_indices=token_slots[:tokens]))
        times = {}
        for name, lora in (("base", None), ("lora", layer)):
            for _ in range(3):
                _forward(x, gate, experts, lora)
            samples = []
            for _ in range(20):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                _forward(x, gate, experts, lora)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            times[name] = sorted(samples)[len(samples) // 2]
        overhead = times["lora"] / times["base"] - 1
        print(f"T={tokens}: base {times['base']:.3f} ms, with LoRA {times['lora']:.3f} ms (+{overhead:.0%})")  # noqa: T201


def main() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    init_distributed_environment(world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:29571", local_rank=0)
    initialize_model_parallel(1)
    torch.manual_seed(0)
    device = torch.device("cuda")
    hidden, inter, num_experts = CONFIG.hidden_size, CONFIG.intermediate_size, CONFIG.n_routed_experts
    with torch.device(device):
        gate = Inkling2Gate(CONFIG)
        experts = Inkling2Experts(CONFIG, "mxfp8")
    gate.weight.data.normal_(0, 0.02)
    gate.bias.data.zero_()
    dense = {}
    for name, (n, k) in zip(("w13", "w2"), experts.shapes, strict=False):
        values, scales, dense[name] = _mxfp8(
            torch.randn(num_experts, n, k, device=device, dtype=torch.bfloat16) / k**0.5
        )
        experts._load_weight(getattr(experts, f"{name}_weight"), values)
        experts._load_block_scales(getattr(experts, f"{name}_weight_scale"), scales)

    def adapter(n, k, scale):
        return (torch.randn(num_experts, n, k, device=device) * scale).bfloat16()

    def slot(rank, gate_up=True, down=True):
        b13, b2 = adapter(2 * inter, rank, 0.1 * (8 / rank) ** 0.5), adapter(hidden, rank, 0.1 * rank**-0.5)
        return _Adapter(
            a13=adapter(rank, hidden, hidden**-0.5),
            b13=b13 if gate_up else torch.zeros_like(b13),
            a2=adapter(rank, inter, inter**-0.5),
            b2=b2 if down else torch.zeros_like(b2),
        )

    cases = {
        "gate/up": [slot(rank, down=False) for rank in RANKS],
        "down": [slot(rank, gate_up=False) for rank in RANKS],
        "both": [slot(rank) for rank in RANKS],
    }
    token_slots = torch.tensor(TOKEN_SLOTS, device=device).repeat(max(TOKENS) * 12 // len(TOKEN_SLOTS) + 1)
    failures = []
    for tokens in TOKENS:
        x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
        base, routing, plan = _forward(x, gate, experts, None)
        base_ref = _reference(x, routing, plan, dense["w13"], dense["w2"], [], None)
        print(f"T={tokens} tile={plan.tile}: base vs reference {_relative_error(base, base_ref):.4f}")  # noqa: T201
        for name, adapters in cases.items():
            layer = _wrapper(gate, experts, adapters, token_slots[:tokens])
            got, routing, plan = _forward(x, gate, experts, layer)
            want = _reference(x, routing, plan, dense["w13"], dense["w2"], adapters, token_slots[:tokens])
            error = _relative_error(got, want)
            share = ((want - base_ref).norm() / base_ref.norm()).item()
            delta_error = _relative_error(got - base, want - base_ref)
            print(  # noqa: T201
                f"  {name}: delta {share:.3f} of the output; output error {error:.4f}, delta error {delta_error:.4f}"
            )
            if error > TOLERANCE or share < 0.05:
                failures.append(f"T={tokens} {name}: output error {error:.4f}, delta share {share:.4f}")
    if failures:
        raise SystemExit("FAIL: " + "; ".join(failures))
    print("RESULT: PASS")  # noqa: T201
    if "--bench" in sys.argv:
        _bench(gate, experts, _wrapper(gate, experts, cases["both"], token_slots), token_slots)


if __name__ == "__main__":
    with set_current_vllm_config(VllmConfig()):
        main()
