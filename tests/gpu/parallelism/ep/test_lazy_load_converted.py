#!/usr/bin/env python
"""Lazy loading of hub-layout composites (GLM-5 Next, Step-3.7, Nemotron-H) and of a native MXFP4 gpt-oss matches a
from_pretrained reference, under EP=2 and pure ETP.

Their layout is bridged only inside ``from_pretrained``, including fan-in Concatenates and MXFP4's dequantize. A
half-read source hands each ETP rank the wrong half. The MXFP4 row also compares every EP parameter bitwise against the
dequantized reference under the same EP patching, and checks the loaded config no longer claims MXFP4.
"""

import copy
import sys

import torch
from transformers import (
    AutoConfig,
    GptOssConfig,
    GptOssForCausalLM,
    Mxfp4Config,
    NemotronH_Omni_Reasoning_V3,
    NemotronH_Omni_Reasoning_V3_Config,
)
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration
from transformers.models.step3p7.configuration_step3p7 import Step3p7Config
from transformers.models.step3p7.modeling_step3p7 import Step3p7ForConditionalGeneration

from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.ep import load_ep_model_lazy
from halo.model_loading.parallel.lazy_plan import is_lazy_loadable
from tests.common.distributed import cleanup_dirs, shared_scratch_dir
from tests.common.ep_reference import stores_expert_tp_split_experts
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import (
    TINY_GLM5_CONFIG,
    TINY_GLM5_VISION_CONFIG,
    TINY_GPTOSS_CONFIG,
    TINY_NEMOTRON_H_CONFIG,
    TINY_RADIO_VISION_CONFIG,
    TINY_STEP3P7_CONFIG,
    TINY_STEP3P7_VISION_CONFIG,
)
from tests.common.native_mxfp4 import write_native_mxfp4
from tests.common.numerics import tensors_equal_at_narrower_dtype
from tests.common.rank_log import log
from tests.common.tolerances import TOL

FAMILY = "glm5_next"
SEED = 42
BATCH, SEQ = 2, 64

_FAMILIES = {
    "glm5_next": (
        Glm5NextForConditionalGeneration,
        lambda: Glm5NextConfig(text_config=dict(TINY_GLM5_CONFIG), vision_config=dict(TINY_GLM5_VISION_CONFIG)),
        TINY_GLM5_CONFIG["mlp_layer_types"].count("sparse"),
        TINY_GLM5_CONFIG["vocab_size"],
    ),
    "step3p7": (
        Step3p7ForConditionalGeneration,
        lambda: Step3p7Config(text_config=dict(TINY_STEP3P7_CONFIG), vision_config=dict(TINY_STEP3P7_VISION_CONFIG)),
        TINY_STEP3P7_CONFIG["mlp_layer_types"].count("sparse"),
        TINY_STEP3P7_CONFIG["vocab_size"],
    ),
    "nemotron_h_omni": (
        NemotronH_Omni_Reasoning_V3,
        lambda: NemotronH_Omni_Reasoning_V3_Config(
            text_config=dict(TINY_NEMOTRON_H_CONFIG),
            vision_config=dict(TINY_RADIO_VISION_CONFIG),
            vision_hidden_size=TINY_RADIO_VISION_CONFIG["hidden_size"],
            projector_hidden_size=TINY_NEMOTRON_H_CONFIG["hidden_size"],
            image_token_id=18,
        ),
        TINY_NEMOTRON_H_CONFIG["layers_block_type"].count("moe"),
        TINY_NEMOTRON_H_CONFIG["vocab_size"],
    ),
    "gpt_oss_mxfp4": (
        GptOssForCausalLM,
        lambda: GptOssConfig(**TINY_GPTOSS_CONFIG),
        TINY_GPTOSS_CONFIG["num_hidden_layers"],
        TINY_GPTOSS_CONFIG["vocab_size"],
    ),
}
# Saved in the openai/gpt-oss-* layout, which from_pretrained reads only when asked to dequantize.
NATIVE_MXFP4_FAMILIES = frozenset({"gpt_oss_mxfp4"})


def _checkpoint_dir() -> str:
    return shared_scratch_dir(f"{FAMILY}_lazy_hub_ckpt")


def _build_checkpoint(model_class, make_config) -> None:
    torch.manual_seed(SEED)
    model = model_class(make_config()).to(torch.bfloat16)
    model.save_pretrained(_checkpoint_dir())
    if FAMILY in NATIVE_MXFP4_FAMILIES:
        write_native_mxfp4(_checkpoint_dir())
    del model
    cleanup_memory()


def _tensors_match(model, reference_sd) -> tuple[bool, str]:
    """EP-wrapped experts equal the local slice, in the wrapper's matmul convention (``[E_local, H, 2M]`` / ``[E_local,
    M, H]``)."""
    for key, live in model.state_dict().items():
        ref = reference_sd.get(key)
        if ref is None:
            continue  # EP-rewrapped expert params are compared through the wrapper below.
        if not tensors_equal_at_narrower_dtype(live.cpu(), ref):
            return False, key
    compared = 0
    for name, module in model.named_modules():
        if not isinstance(module, EPMoELayerBase):
            continue
        s, e = module.expert_start, module.expert_end
        for attr, _ in module._base_expert_named_params():
            want = reference_sd[f"{name}.experts.{attr}"][s:e].transpose(1, 2).contiguous()
            if not tensors_equal_at_narrower_dtype(getattr(module, attr).data.cpu(), want):
                return False, f"{name}.{attr} (expert slice {s}:{e})"
            compared += 1
    return compared > 0, "no EP layer compared"


def _matches_patched_reference(model, reference, ep_config) -> tuple[bool, str]:
    """Every tensor of the lazily loaded EP model equals the dequantized reference patched the same way, in whatever
    layout the family's EP layer stores: gpt-oss keeps ``[E, H, 2M]``, de-interleaved for grouped GEMM and ETP."""
    patched = apply_ep_layers(copy.deepcopy(reference), ep_config).state_dict()
    live = model.state_dict()
    if set(live) != set(patched):
        return False, f"key sets differ: {sorted(set(live) ^ set(patched))[:4]}"
    for key, tensor in live.items():
        if not tensors_equal_at_narrower_dtype(tensor.cpu(), patched[key].cpu()):
            return False, key
    return True, ""


def _native_mxfp4_checks(model, reference, ep_config, mode: str) -> dict[str, bool]:
    ok, where = _matches_patched_reference(model, reference, ep_config)
    if not ok:
        log(f"{mode} MISMATCH at {where}")
    return {
        f"{mode}_tensors_match_dequantized_reference": ok,
        f"{mode}_config_no_longer_claims_mxfp4": getattr(model.config, "quantization_config", None) is None,
    }


def run(ctx):
    model_class, make_config, num_moe_layers, vocab_size = _FAMILIES[FAMILY]
    native_mxfp4 = FAMILY in NATIVE_MXFP4_FAMILIES
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    device = ctx.device
    torch.cuda.set_device(device)
    ckpt = _checkpoint_dir()
    if ctx.rank == 0:
        ctx.on_teardown(lambda: cleanup_dirs(ckpt))
        _build_checkpoint(model_class, make_config)
    ctx.barrier()

    torch.manual_seed(SEED)
    input_ids = torch.randint(0, vocab_size, (BATCH, SEQ), device=device)
    labels = input_ids.clone()

    checks["lazy_gate_admits_checkpoint"] = is_lazy_loadable(ckpt)

    dequantize = {"quantization_config": Mxfp4Config(dequantize=True)} if native_mxfp4 else {}
    ref = model_class.from_pretrained(ckpt, dtype=torch.bfloat16, attn_implementation="eager", **dequantize).to(device)
    ref.train()
    ref_loss = ref(input_ids=input_ids, labels=labels).loss.item()
    reference_sd = {k: v.detach().cpu() for k, v in ref.state_dict().items()}
    log(f"reference loss: {ref_loss:.6f}")
    if not native_mxfp4:
        del ref  # The MXFP4 row keeps it to patch for the tensor comparisons.
    cleanup_memory()
    metrics["ref_loss"] = ref_loss

    pc = ParallelismConfig(ep_size=ctx.world_size)
    ep_config = pc.ep_config()
    model = load_ep_model_lazy(
        ckpt,
        ep_config,
        AutoConfig.from_pretrained(ckpt),
        dtype=torch.bfloat16,
        model_class=model_class,
        attn_implementation="eager",
    )
    ep_layers = [m for m in model.modules() if isinstance(m, EPMoELayerBase)]
    checks["ep_layers_patched"] = len(ep_layers) == num_moe_layers
    if native_mxfp4:
        checks |= _native_mxfp4_checks(model, ref, ep_config, "ep2")
    else:
        ok, where = _tensors_match(model, reference_sd)
        checks["ep2_tensors_match_reference"] = ok
        if not ok:
            log(f"MISMATCH at {where}")
    model.train()
    ep_loss = model(input_ids=input_ids, labels=labels).loss.item()
    metrics["ep2_loss"] = ep_loss
    checks["ep2_loss_matches_ref"] = abs(ep_loss - ref_loss) < TOL.parallel_vs_baseline_loss_abs
    log(f"EP2 lazy loss: {ep_loss:.6f}  |Δref| = {abs(ep_loss - ref_loss):.2e}")
    del model, ep_layers
    cleanup_memory()

    # Split on the concatenated halves.
    pc = ParallelismConfig(ep_size=1, expert_tp_size=2)
    ep_config = pc.ep_config()
    model = load_ep_model_lazy(
        ckpt,
        ep_config,
        AutoConfig.from_pretrained(ckpt),
        dtype=torch.bfloat16,
        model_class=model_class,
        attn_implementation="eager",
    )
    etp_layers = [m for m in model.modules() if isinstance(m, EPMoELayerBase)]
    checks["etp_split_glu_storage"] = bool(etp_layers) and all(stores_expert_tp_split_experts(ep) for ep in etp_layers)
    if native_mxfp4:
        checks |= _native_mxfp4_checks(model, ref, ep_config, "etp2")
    model.train()
    etp_loss = model(input_ids=input_ids, labels=labels).loss.item()
    metrics["etp_loss"] = etp_loss
    checks["etp_loss_matches_ref"] = abs(etp_loss - ref_loss) < TOL.parallel_vs_baseline_loss_abs
    log(f"ETP lazy loss: {etp_loss:.6f}  |Δref| = {abs(etp_loss - ref_loss):.2e}")

    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, exact_world_size=2, prefix="lazy_load_converted")(run)

if __name__ == "__main__":
    if "--family" in sys.argv:
        i = sys.argv.index("--family")
        FAMILY = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    if FAMILY not in _FAMILIES:
        raise SystemExit(f"--family must be one of {sorted(_FAMILIES)}, got {FAMILY!r}")
    main()
