#!/usr/bin/env python
"""Converter-backed lazy loading of a hub-layout composite checkpoint, vs a from_pretrained reference.

GLM-5 Next and Step-3.7 ship in a layout transformers bridges only inside ``from_pretrained``:
GLM-5 Next's vendor-namespace KDA / hyper-connection keys, per-expert projections and the
three-source ``q/k/v_conv1d → conv1d`` Concatenate; Step-3.7's prefix renames, ``moe.*`` → ``mlp.*``,
the two-source per-layer ``moe.gate_proj + moe.up_proj → gate_up_proj`` Concatenate and the Step-3.5
vision tower's chunked, RoPE-permuted ``in_proj``. The lazy loaders replay that mapping per key
(``_HUB_CONVERSION_KEYS`` → ``hub_conversion.py``), so this gate proves, against a
``from_pretrained`` reference on the same checkpoint (written by the family's own ``save_pretrained``):

  1. **EP=2** — every expert shard equals the reference's slice bitwise (a fan-in sliced through
     EVERY source), every other tensor equals it bitwise at the run's dtype, and the loss matches.
  2. **Pure ETP (ep1 × etp2)** — the fused halves split on the concatenated tensor (a half-read
     source would hand each rank the wrong half); the loss matches.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_lazy_load_converted.py --family glm5_next
"""

import os
import sys
import tempfile

import torch
from transformers import AutoConfig
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration
from transformers.models.step3p7.configuration_step3p7 import Step3p7Config
from transformers.models.step3p7.modeling_step3p7 import Step3p7ForConditionalGeneration

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.lazy_loader import lazy_loader_supports_checkpoint, load_ep_model_lazy
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.harness import gpu_test_main
from tests.common.models import (
    TINY_GLM5_CONFIG,
    TINY_GLM5_VISION_CONFIG,
    TINY_STEP3P7_CONFIG,
    TINY_STEP3P7_VISION_CONFIG,
)
from tests.common.utils import cleanup_memory, log, tensors_equal_at_narrower_dtype

FAMILY = "glm5_next"
SEED = 42
BATCH, SEQ = 2, 64
LOSS_TOL = 5e-2

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
}


def _checkpoint_dir() -> str:
    return os.path.join(tempfile.gettempdir(), f"{FAMILY}_lazy_hub_ckpt")


def _build_checkpoint(model_class, make_config) -> None:
    """Rank 0: save the tiny composite through its own ``save_pretrained`` — the hub layout."""
    torch.manual_seed(SEED)
    model = model_class(make_config()).to(torch.bfloat16)
    model.save_pretrained(_checkpoint_dir())
    del model
    cleanup_memory()


def _tensors_match(model, reference_sd) -> tuple[bool, str]:
    """Every loaded tensor equals the reference; EP-wrapped experts equal the local slice, in the
    wrapper's matmul convention (``[E_local, H, 2M]`` / ``[E_local, M, H]``)."""
    for key, live in model.state_dict().items():
        ref = reference_sd.get(key)
        if ref is None:
            continue  # EP-rewrapped expert params are compared through the wrapper below
        if not tensors_equal_at_narrower_dtype(live.cpu(), ref):
            return False, key
    compared = 0
    for name, module in model.named_modules():
        if not isinstance(module, EPMoELayerBase):
            continue
        s, e = module.expert_start, module.expert_end
        for attr in ("gate_up_proj", "down_proj"):
            want = reference_sd[f"{name}.experts.{attr}"][s:e].transpose(1, 2).contiguous()
            if not tensors_equal_at_narrower_dtype(getattr(module, attr).data.cpu(), want):
                return False, f"{name}.{attr} (expert slice {s}:{e})"
            compared += 1
    return compared > 0, "no EP layer compared"


def run(ctx):
    model_class, make_config, num_moe_layers, vocab_size = _FAMILIES[FAMILY]
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    device = ctx.device
    torch.cuda.set_device(device)
    ckpt = _checkpoint_dir()
    if ctx.rank == 0:
        _build_checkpoint(model_class, make_config)
    ctx.barrier()

    torch.manual_seed(SEED)
    input_ids = torch.randint(0, vocab_size, (BATCH, SEQ), device=device)
    labels = input_ids.clone()

    checks["lazy_gate_admits_checkpoint"] = lazy_loader_supports_checkpoint(ckpt)

    # ── Reference: from_pretrained on the same hub-layout checkpoint ──────────
    ref = model_class.from_pretrained(ckpt, dtype=torch.bfloat16, attn_implementation="eager").to(device)
    ref.train()
    ref_loss = ref(input_ids=input_ids, labels=labels).loss.item()
    reference_sd = {k: v.detach().cpu() for k, v in ref.state_dict().items()}
    log(f"reference loss: {ref_loss:.6f}")
    del ref
    cleanup_memory()
    metrics["ref_loss"] = ref_loss

    # ── Leg 1: EP=2 lazy load — bitwise tensors + loss ─────────────────────────
    pc = ParallelismConfig(ep_size=ctx.world_size)
    model = load_ep_model_lazy(
        ckpt,
        pc.create_ep_config(),
        AutoConfig.from_pretrained(ckpt),
        dtype=torch.bfloat16,
        model_class=model_class,
        attn_implementation="eager",
    )
    ep_layers = [m for m in model.modules() if isinstance(m, EPMoELayerBase)]
    checks["ep_layers_patched"] = len(ep_layers) == num_moe_layers
    ok, where = _tensors_match(model, reference_sd)
    checks["ep2_tensors_match_reference"] = ok
    if not ok:
        log(f"MISMATCH at {where}")
    model.train()
    ep_loss = model(input_ids=input_ids, labels=labels).loss.item()
    metrics["ep2_loss"] = ep_loss
    checks["ep2_loss_matches_ref"] = abs(ep_loss - ref_loss) < LOSS_TOL
    log(f"EP2 lazy loss: {ep_loss:.6f}  |Δref| = {abs(ep_loss - ref_loss):.2e}")
    del model, ep_layers
    cleanup_memory()

    # ── Leg 2: pure ETP (ep1 × etp2) lazy load — split on the concatenated halves ─
    pc = ParallelismConfig(ep_size=1, expert_tp_size=2)
    model = load_ep_model_lazy(
        ckpt,
        pc.create_ep_config(),
        AutoConfig.from_pretrained(ckpt),
        dtype=torch.bfloat16,
        model_class=model_class,
        attn_implementation="eager",
    )
    etp_layers = [m for m in model.modules() if isinstance(m, EPMoELayerBase)]
    checks["etp_split_glu_storage"] = bool(etp_layers) and all(
        hasattr(ep, "gate_proj") and not hasattr(ep, "gate_up_proj") for ep in etp_layers
    )
    model.train()
    etp_loss = model(input_ids=input_ids, labels=labels).loss.item()
    metrics["etp_loss"] = etp_loss
    checks["etp_loss_matches_ref"] = abs(etp_loss - ref_loss) < LOSS_TOL
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
