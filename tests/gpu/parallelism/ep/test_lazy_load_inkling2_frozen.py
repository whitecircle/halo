#!/usr/bin/env python
"""Lazy EP loading of an Inkling 2 release-layout checkpoint (NVFP4 + MXFP8 experts) with frozen_quantized_experts:
each rank's banks hold its expert slice of the packed release tensors and dequantize to the reference weights, every
other tensor loads as from_pretrained loads it, the loss is the reference model's, and a backward step trains the
expert adapters while the banks stay untouched.

The reference is the same checkpoint with its experts dequantized to bf16, through from_pretrained."""

import torch
from transformers import AutoConfig
from transformers.models.inkling2.modeling_inkling2 import Inkling2ForConditionalGeneration

from halo.distributed.expert_parallel.families.inkling2 import EPInkling2MoELayer
from halo.distributed.expert_parallel.moe_layer.frozen_banks import FROZEN_SLOTS
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.ep import load_ep_model_lazy
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from tests.common.distributed import cleanup_dirs, shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.inkling2 import TINY_INKLING2_MXFP8_LAYERS, TINY_INKLING2_TEXT, write_tiny_inkling2_release
from tests.common.memory import cleanup_memory
from tests.common.rank_log import log
from tests.common.tolerances import TOL

SEED = 42
BATCH, SEQ = 2, 48
_EXPERTS_KEY = "model.language_model.layers.{layer}.mlp.experts.{weight}"


def _layer_index(name: str) -> int:
    return int(name.split(".layers.")[1].split(".")[0])


def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    device = ctx.device
    torch.cuda.set_device(device)
    release_dir = shared_scratch_dir("inkling2_frozen_release")
    reference_dir = shared_scratch_dir("inkling2_frozen_reference")
    if ctx.rank == 0:
        ctx.on_teardown(lambda: cleanup_dirs(release_dir, reference_dir))
        write_tiny_inkling2_release(release_dir, reference_dir, seed=SEED)
    ctx.barrier()

    torch.manual_seed(SEED)
    input_ids = torch.randint(0, TINY_INKLING2_TEXT["unpadded_vocab_size"], (BATCH, SEQ), device=device)

    reference = Inkling2ForConditionalGeneration.from_pretrained(reference_dir, dtype=torch.bfloat16).to(device)
    reference.train()
    ref_loss = reference(input_ids=input_ids, labels=input_ids).loss.item()
    reference_sd = {key: value.cpu() for key, value in reference.state_dict().items()}
    del reference
    cleanup_memory()
    metrics["ref_loss"] = ref_loss
    log(f"reference loss: {ref_loss:.6f}")

    pc = ParallelismConfig(
        ep_size=ctx.world_size, frozen_quantized_experts=True, expert_lora=ExpertLoraSpec(r=8, alpha=16)
    )
    model = load_ep_model_lazy(
        release_dir,
        pc.ep_config(),
        AutoConfig.from_pretrained(release_dir),
        dtype=torch.bfloat16,
        model_class=Inkling2ForConditionalGeneration,
        frozen_quantized_experts=True,
    )
    layers = {name: m for name, m in model.named_modules() if isinstance(m, EPInkling2MoELayer)}
    checks["every_moe_layer_patched"] = len(layers) == TINY_INKLING2_TEXT["num_hidden_layers"]
    checks["every_layer_holds_frozen_banks"] = all(m.has_frozen_banks for m in layers.values())

    formats_ok, banks_match = True, True
    for name, layer in layers.items():
        index = _layer_index(name)
        expected_format = "mxfp8" if index in TINY_INKLING2_MXFP8_LAYERS else "nvfp4"
        for slot in FROZEN_SLOTS:
            bank = layer.frozen_bank(slot.weight)
            formats_ok &= bank.quant_format == expected_format
            want = reference_sd[_EXPERTS_KEY.format(layer=index, weight=slot.weight)][
                layer.expert_start : layer.expert_end
            ]
            if not torch.equal(bank.dequantize_linear().cpu(), want):
                banks_match = False
                log(f"bank mismatch: layer {index} {slot.weight} experts {layer.expert_start}:{layer.expert_end}")
    checks["bank_formats_follow_the_release"] = formats_ok
    checks["banks_dequantize_to_the_reference_slice"] = banks_match

    expert_keys = {_EXPERTS_KEY.format(layer=i, weight=s.weight) for i in range(len(layers)) for s in FROZEN_SLOTS}
    loaded = model.state_dict()
    others_match = True
    for key, value in reference_sd.items():
        if key in expert_keys:
            continue
        live = loaded.get(key)
        if live is None or not torch.equal(live.cpu().to(value.dtype), value):
            others_match = False
            log(f"non-expert mismatch or missing: {key}")
    checks["non_expert_tensors_match_reference"] = others_match
    checks["no_bf16_expert_parameters"] = not any(
        name.endswith((".gate_up_proj", ".down_proj")) for name, _ in model.named_parameters()
    )

    model.train()
    loss = model(input_ids=input_ids, labels=input_ids).loss
    metrics["frozen_loss"] = loss.item()
    checks["loss_matches_reference"] = abs(loss.item() - ref_loss) < TOL.parallel_vs_baseline_loss_abs
    log(f"frozen lazy loss: {loss.item():.6f}  |Δref| = {abs(loss.item() - ref_loss):.2e}")

    loss.backward()
    adapter_grads = [
        getattr(layer, f"{slot.weight}_lora_B").grad for layer in layers.values() for slot in FROZEN_SLOTS
    ]
    checks["expert_adapters_receive_gradients"] = all(g is not None and g.abs().max() > 0 for g in adapter_grads)
    checks["banks_take_no_gradient"] = all(
        buffer.grad is None
        for layer in layers.values()
        for buffer in layer.buffers()
        if not buffer.is_floating_point()
    )
    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, prefix="lazy_load_inkling2_frozen")(run)

if __name__ == "__main__":
    main()
