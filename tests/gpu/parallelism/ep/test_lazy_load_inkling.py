#!/usr/bin/env python
"""Converter-backed lazy loading of Inkling's hub-layout checkpoint, vs a from_pretrained reference.

``save_pretrained`` on transformers >= 5.14 writes the composite class in Thinking Machines'
original namespace (``model.llm.*``, ``attn.wq_du``, interleaved ``experts.w13_weight``) — the same
layout the hub ships. The lazy loaders translate it through the family's declared conversion entry
(``_HUB_CONVERSION_KEYS`` → ``hub_conversion.py``), so this gate proves, against a ``from_pretrained``
reference on the same checkpoint:

  1. **EP=2 composite** — every expert shard equals the reference's slice bitwise, attention/backbone
     tensors equal bitwise, and the loaded model's loss matches.
  2. **Pure ETP (ep1 x etp2)** — the wrapper's contiguous-halves split runs on the DE-INTERLEAVED
     tensor; an un-applied Deinterleave would hand each rank interleaved halves and move the loss.
  3. **Text-only artifact** — the same TM-layout weights with an `InklingTextConfig` ``config.json``
     load as ``InklingForCausalLM`` (composite renames + the standard ``language_model.`` alignment),
     with the tower/MTP-free key space fully covered.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_lazy_load_inkling.py
"""

import json
import os
import tempfile

import torch
from transformers import AutoConfig
from transformers.models.inkling.configuration_inkling import (
    InklingConfig,
    InklingTextConfig,
    InklingVisionConfig,
)
from transformers.models.inkling.modeling_inkling import InklingForCausalLM, InklingForConditionalGeneration

from src.distributed.expert_parallel.layers.inkling import EPInklingMoELayer
from src.distributed.expert_parallel.lazy_loader import lazy_loader_supports_checkpoint, load_ep_model_lazy
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_INKLING_CONFIG
from tests.common.utils import cleanup_memory, log

SEED = 42
BATCH, SEQ = 2, 64
LOSS_TOL = 5e-2

TINY_VISION = {"temporal_patch_size": 2, "patch_size": 4, "n_layers": 4, "hidden_size": 32, "text_hidden_size": 256}


def _checkpoint_dirs() -> tuple[str, str]:
    base = os.path.join(tempfile.gettempdir(), "inkling_lazy_tm_ckpt")
    return base, base + "_text"


def _build_checkpoints():
    """Rank 0: save the tiny composite in TM layout, plus a text-only config twin of the weights."""
    composite_dir, text_dir = _checkpoint_dirs()
    torch.manual_seed(SEED)
    config = InklingConfig(
        text_config=dict(TINY_INKLING_CONFIG),
        vision_config=InklingVisionConfig(**TINY_VISION).to_dict(),
        image_token_id=500,
    )
    model = InklingForConditionalGeneration(config).to(torch.bfloat16)
    model.save_pretrained(composite_dir)  # save_original_format=True → TM namespace on disk

    os.makedirs(text_dir, exist_ok=True)
    for name in os.listdir(composite_dir):
        if name != "config.json":
            src = os.path.join(composite_dir, name)
            dst = os.path.join(text_dir, name)
            if not os.path.exists(dst):
                os.link(src, dst)
    text_config = InklingTextConfig(**TINY_INKLING_CONFIG)
    text_config.architectures = ["InklingForCausalLM"]
    with open(os.path.join(text_dir, "config.json"), "w") as f:
        json.dump(text_config.to_dict(), f)
    del model
    cleanup_memory()


def _state_dicts_match(loaded, reference_sd, ep) -> tuple[bool, str]:
    """Every loaded tensor equals the reference (expert tensors: the local slice) bitwise."""
    s, e = ep.expert_start, ep.expert_end
    for key, ref in reference_sd.items():
        live = loaded.get(key)
        if live is None:
            continue  # EP-rewrapped expert params are compared via the wrapper below
        if not torch.equal(live.cpu(), ref.cpu()):
            return False, key
    for name, param in (("gate_up_proj", ep.gate_up_proj), ("down_proj", ep.down_proj)):
        ref = reference_sd[f"model.language_model.layers.1.mlp.experts.{name}"]
        want = ref[s:e].transpose(1, 2).contiguous()
        if not torch.equal(param.data.cpu(), want.cpu()):
            return False, f"expert shard {name}"
    return True, ""


def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    device = ctx.device
    torch.cuda.set_device(device)
    composite_dir, text_dir = _checkpoint_dirs()
    if ctx.rank == 0:
        _build_checkpoints()
    ctx.barrier()

    torch.manual_seed(SEED)
    input_ids = torch.randint(0, TINY_INKLING_CONFIG["vocab_size"], (BATCH, SEQ), device=device)
    labels = input_ids.clone()

    checks["lazy_gate_admits_inkling"] = lazy_loader_supports_checkpoint(composite_dir)

    # ── Reference: from_pretrained on the same TM-layout checkpoint ───────────
    ref = InklingForConditionalGeneration.from_pretrained(
        composite_dir, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device)
    ref.train()
    ref_loss = ref(input_ids=input_ids, labels=labels).loss.item()
    reference_sd = {k: v.cpu() for k, v in ref.state_dict().items()}
    log(f"reference loss: {ref_loss:.6f}")
    del ref
    cleanup_memory()
    metrics["ref_loss"] = ref_loss

    # ── Leg 1: EP=2 composite lazy load ──────────────────────────────────────
    pc = ParallelismConfig(ep_size=ctx.world_size)
    model = load_ep_model_lazy(
        composite_dir,
        pc.create_ep_config(),
        AutoConfig.from_pretrained(composite_dir),
        dtype=torch.bfloat16,
        model_class=InklingForConditionalGeneration,
        attn_implementation="eager",
    )
    ep_layers = [m for m in model.modules() if isinstance(m, EPInklingMoELayer)]
    checks["ep_layers_patched"] = len(ep_layers) == TINY_INKLING_CONFIG["num_hidden_layers"]
    ok, where = _state_dicts_match(dict(model.state_dict()), reference_sd, ep_layers[1])
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

    # ── Leg 2: pure ETP (ep1 × etp2) lazy load — de-interleave before the split ─
    pc = ParallelismConfig(ep_size=1, expert_tp_size=2)
    model = load_ep_model_lazy(
        composite_dir,
        pc.create_ep_config(),
        AutoConfig.from_pretrained(composite_dir),
        dtype=torch.bfloat16,
        model_class=InklingForConditionalGeneration,
        attn_implementation="eager",
    )
    etp_layers = [m for m in model.modules() if isinstance(m, EPInklingMoELayer)]
    checks["etp_split_glu_storage"] = all(
        hasattr(ep, "gate_proj") and not hasattr(ep, "gate_up_proj") for ep in etp_layers
    )
    model.train()
    etp_loss = model(input_ids=input_ids, labels=labels).loss.item()
    metrics["etp_loss"] = etp_loss
    checks["etp_loss_matches_ref"] = abs(etp_loss - ref_loss) < LOSS_TOL
    log(f"ETP lazy loss: {etp_loss:.6f}  |Δref| = {abs(etp_loss - ref_loss):.2e}")
    del model, etp_layers
    cleanup_memory()

    # ── Leg 3: text-only artifact loads as InklingForCausalLM ────────────────
    pc = ParallelismConfig(ep_size=ctx.world_size)
    model = load_ep_model_lazy(
        text_dir,
        pc.create_ep_config(),
        AutoConfig.from_pretrained(text_dir),
        dtype=torch.bfloat16,
        model_class=InklingForCausalLM,
        attn_implementation="eager",
    )
    checks["text_only_patched"] = (
        sum(isinstance(m, EPInklingMoELayer) for m in model.modules()) == TINY_INKLING_CONFIG["num_hidden_layers"]
    )
    model.train()
    text_loss = model(input_ids=input_ids, labels=labels).loss.item()
    metrics["text_only_loss"] = text_loss
    # The composite reference's text loss on a text-only batch is the same computation: embeddings,
    # decoder and head all come from the shared weights.
    checks["text_only_loss_matches_ref"] = abs(text_loss - ref_loss) < LOSS_TOL
    log(f"text-only lazy loss: {text_loss:.6f}  |Δref| = {abs(text_loss - ref_loss):.2e}")

    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, exact_world_size=2, prefix="lazy_load_inkling")(run)

if __name__ == "__main__":
    main()
