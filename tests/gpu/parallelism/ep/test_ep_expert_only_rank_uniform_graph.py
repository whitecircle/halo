#!/usr/bin/env python
"""Expert-only training at ep2 builds the same DeepEP autograd graph on every rank.

With only the experts trainable (native expert LoRA), nothing upstream of the first MoE layer requires
grad. A rank whose local experts receive no token in a layer computes a constant expert output there,
so unless the dispatch has a grad-requiring input, that rank's combine, and every later layer that
loses grad through it, carries no autograd node where its peers' do. Its backward then pairs mismatched
DeepEP collectives (barrier timeout), or, idle in every layer, it has no grad on its loss at all.

Routing is forced at the shared ``_dispatch_compute_combine`` seam, so which rank idles is exact and
family-independent. Before each scenario's backward, every rank's graph signature (loss requires grad,
DeepEP dispatch and combine node counts) is all-gathered: a non-uniform graph fails the scenario on
every rank together and skips its backward rather than hanging. A uniform graph backpropagates, and
each rank's expert-adapter gradients are compared with an ep1 reference that runs both ranks' batches
under the same routing and the same adapter values.

Scenarios:
  first — layer 0 sends every token to rank 0's experts; later layers spread tokens over every expert.
  all   — every layer sends every token to rank 0's experts.
  gc    — ``all`` under reentrant gradient checkpointing, for a family that allows it. transformers turns
          on input grads with checkpointing, so this dispatch never lacks a grad-requiring input: the
          scenario guards the replay path and passes with or without the fix.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep_expert_only_rank_uniform_graph.py --family zaya
"""

import contextlib
import copy
import sys

import torch
import torch.distributed as dist
from accelerate.state import GradientState
from transformers import GptOssConfig, GptOssForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.zaya.configuration_zaya import ZayaConfig
from transformers.models.zaya.modeling_zaya import ZayaForCausalLM

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.config import ExpertLoraSpec
from src.distributed.expert_parallel.patching import (
    create_ep_buffers,
    enable_ep_gradient_checkpointing,
    patch_moe_model_for_ep,
)
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.ep_reference import ep_layers, score_ep_grad_pairs
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_GPTOSS_CONFIG, TINY_QWEN3_MOE_CONFIG, TINY_ZAYA_CONFIG
from tests.common.peft_helpers import freeze_base_keep_expert_adapters
from tests.common.tolerances import TOL
from tests.common.utils import log, log_all

FAMILY = "qwen3_moe"
EP_SIZE = 2
BATCH, SEQ = 2, 48
SEED = 1234
LORA_R, LORA_ALPHA = 8, 16
# Random adapters (B is zero at init, which would leave every A gradient zero and uninformative).
ADAPTER_STD = 0.05
# The ep2 and ep1 forwards run the same bf16 math apart from DeepEP's combine; a wrong routing or a
# dropped layer moves the mean token loss by far more.
LOSS_ATOL = 2e-2

_FAMILIES = {
    "gpt_oss": (GptOssForCausalLM, GptOssConfig, TINY_GPTOSS_CONFIG),
    "qwen3_moe": (Qwen3MoeForCausalLM, Qwen3MoeConfig, TINY_QWEN3_MOE_CONFIG),
    "zaya": (ZayaForCausalLM, ZayaConfig, TINY_ZAYA_CONFIG),
}

_DEEPEP_NODES = ("DeepEPDispatchFunctionBackward", "DeepEPCombineFunctionBackward")
_SEAM = EPMoELayerBase._dispatch_compute_combine


@contextlib.contextmanager
def forced_routing(starved_layers: set[int]):
    """Replace every EP layer's selection at the dispatch seam.

    A starved layer sends token ``t``'s ``k``-th pick to expert ``(t + k) % (E / EP_SIZE)``, all owned by
    rank 0 at ep2; any other layer uses ``(t + k) % E``, which reaches every rank. The reference at ep1
    gets the same ids. Gate weights stay the router's own: selection is not differentiable.
    """

    def seam(self, flat, experts, weights, input_dtype):
        span = self.num_experts // EP_SIZE if self.test_layer_index in starved_layers else self.num_experts
        if span < experts.shape[1]:
            raise AssertionError(f"top_k={experts.shape[1]} needs {experts.shape[1]} distinct experts in {span}")
        rows = torch.arange(flat.shape[0], device=flat.device)[:, None]
        slots = torch.arange(experts.shape[1], device=flat.device)[None, :]
        return _SEAM(self, flat, (rows + slots) % span, weights, input_dtype)

    EPMoELayerBase._dispatch_compute_combine = seam
    try:
        yield
    finally:
        EPMoELayerBase._dispatch_compute_combine = _SEAM


def graph_signature(loss: torch.Tensor) -> list[int]:
    """``[loss requires grad, DeepEP dispatch nodes, DeepEP combine nodes]`` reachable from ``loss``."""
    counts = dict.fromkeys(_DEEPEP_NODES, 0)
    stack, seen = [loss.grad_fn], set()
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        name = type(node).__name__
        if name in counts:
            counts[name] += 1
        stack.extend(parent for parent, _ in node.next_functions)
    return [int(loss.requires_grad), *counts.values()]


def build_models(device):
    """The ep2 model and its ep1 reference: same base weights, same adapter values, adapters alone trainable."""
    model_class, config_class, config = _FAMILIES[FAMILY]
    torch.manual_seed(SEED)
    base = model_class(config_class(**config)).to(torch.bfloat16)
    reference = copy.deepcopy(base)
    spec = ExpertLoraSpec(r=LORA_R, alpha=LORA_ALPHA)
    ep_pc, ref_pc = ParallelismConfig(ep_size=EP_SIZE), ParallelismConfig(ep_size=1)
    ep_pc.expert_lora = ref_pc.expert_lora = spec
    model = patch_moe_model_for_ep(base.to(device), ep_pc.create_ep_config())
    reference = patch_moe_model_for_ep(reference.to(device), ref_pc.create_ep_config())
    create_ep_buffers(model)

    generator = torch.Generator().manual_seed(SEED)
    for index, (layer, ref_layer) in enumerate(zip(ep_layers(model), ep_layers(reference), strict=True)):
        layer.test_layer_index = ref_layer.test_layer_index = index
        for attr in sorted(ref_layer._expert_lora_attrs):
            for side in ("A", "B"):
                full = getattr(ref_layer, f"{attr}_lora_{side}")
                full.data.copy_(torch.randn(full.shape, generator=generator) * ADAPTER_STD)
                getattr(layer, f"{attr}_lora_{side}").data.copy_(full.data[layer.expert_start : layer.expert_end])
    freeze_base_keep_expert_adapters(model)
    freeze_base_keep_expert_adapters(reference)
    return model.train(), reference.train()


def score_adapter_grads(model, reference, name: str, checks: dict, metrics: dict) -> tuple[int, int]:
    """Each local adapter gradient against its reference slice; a bank the reference routes nothing to
    must hold no gradient at all. Returns the (active, idle) adapter-tensor counts."""
    pairs, idle_clean, idle = {}, [], 0
    for index, (layer, ref_layer) in enumerate(zip(ep_layers(model), ep_layers(reference), strict=True)):
        for attr in sorted(layer._expert_lora_attrs):
            for side in ("A", "B"):
                got = getattr(layer, f"{attr}_lora_{side}").grad
                expected = getattr(ref_layer, f"{attr}_lora_{side}").grad[layer.expert_start : layer.expert_end]
                if expected.any():
                    pairs[f"l{index}_{attr}_{side}"] = (got, expected)
                else:
                    idle += 1
                    idle_clean.append(got is None or not got.any())
    pair_checks, pair_metrics = {}, {}
    score_ep_grad_pairs(pairs, pair_checks, pair_metrics, cos_min=TOL.ep_grad_cosine_min)
    checks[f"{name}_active_bank_grads_match_reference"] = all(pair_checks.values())
    checks[f"{name}_idle_banks_hold_no_grad"] = all(idle_clean)
    cosines = [value for key, value in pair_metrics.items() if key.endswith("_cos")]
    metrics[f"{name}_min_cos_rank{dist.get_rank()}"] = min(cosines, default=float("nan"))
    log_all(f"  {name}: {len(pairs)} active adapter grads, {idle} idle")
    return len(pairs), idle


def run_scenario(ctx, model, reference, batches, name: str, starved: set[int], checks: dict, metrics: dict):
    model.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    with forced_routing(starved):
        loss = model(input_ids=batches[ctx.rank], labels=batches[ctx.rank]).loss
        signature = torch.tensor(graph_signature(loss), device=ctx.device)
        gathered = [torch.empty_like(signature) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, signature)
        checks[f"{name}_rank_uniform_graph"] = all(torch.equal(sig, signature) for sig in gathered)
        log(f"  {name}: graph signatures (loss requires grad, dispatch, combine) {[s.tolist() for s in gathered]}")
        if not checks[f"{name}_rank_uniform_graph"]:
            # Every rank sees the same gathered list, so all skip together: this backward would hang.
            return
        loss.backward()
        ref_losses = [reference(input_ids=batch, labels=batch).loss for batch in batches]
        (sum(ref_losses) / len(ref_losses)).backward()

    checks[f"{name}_loss_matches_reference"] = abs(loss.item() - ref_losses[ctx.rank].item()) < LOSS_ATOL
    active, idle = score_adapter_grads(model, reference, name, checks, metrics)
    # The scenario's premise: rank 0's experts receive tokens, and rank 1's idle wherever starved.
    checks[f"{name}_routing_premise"] = active > 0 if ctx.rank == 0 else idle > 0
    return signature.tolist()


def run(ctx):
    checks, metrics = {}, {}
    torch.cuda.set_device(ctx.device)
    GradientState()._set_sync_gradients(True)
    model, reference = build_models(ctx.device)
    layers = ep_layers(model)
    ep_config = layers[0].ep_config
    # The comparison assumes the in-backward hook regime: its /world divide makes the ep2 gradient the
    # gradient of the mean of both ranks' losses, which is what the reference differentiates.
    checks["hook_regime"] = not ep_config.defer_grad_sync and not ep_config.experts_fsdp_managed

    generator = torch.Generator().manual_seed(SEED)
    vocab = model.config.vocab_size
    batches = [
        torch.randint(0, vocab, (BATCH, SEQ), generator=generator).to(ctx.device) for _ in range(ctx.world_size)
    ]

    n_layers = len(layers)
    for name, starved in (("first", {0}), ("all", set(range(n_layers)))):
        signature = run_scenario(ctx, model, reference, batches, name, starved, checks, metrics)
        if signature is not None:
            # Every MoE layer's dispatch and combine join the graph, the first one's included.
            checks[f"{name}_every_layer_in_graph"] = signature == [1, n_layers, n_layers]

    if all(layer._supports_gradient_checkpointing for layer in layers):
        enable_ep_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": True})
        run_scenario(ctx, model, reference, batches, "gc", set(range(n_layers)), checks, metrics)
    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(exact_world_size=EP_SIZE, prefix="ep_expert_only_uniform_graph")(run)

if __name__ == "__main__":
    if "--family" in sys.argv:
        i = sys.argv.index("--family")
        FAMILY = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    if FAMILY not in _FAMILIES:
        raise SystemExit(f"--family must be one of {sorted(_FAMILIES)}, got {FAMILY!r}")
    main()
