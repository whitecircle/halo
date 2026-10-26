#!/usr/bin/env python
"""Text-only Inkling under PP: stage-aware loading of the TM-layout checkpoint, bitwise vs full-load.

The hub checkpoint keeps Thinking Machines' namespace; the stage loader reads it through the
family's conversion entry (``conversion_selection.py``). This gate builds a 4-layer text-only artifact
(TM-layout weights + `InklingTextConfig` ``config.json``, ``layer_types`` period 2 so the pp2
boundary lands on a whole pattern period) and asserts, for both stages of a virtual ``pp_size=2``
pipeline walked by both ranks:

    load_ep_model_lazy(whole model) + build_pipeline_stage
        == load_pp_stage_model + build_pipeline_stage      (bitwise, per parameter)

with EP=2 inside the stage, so the converted + de-interleaved expert slice composes with the PP
layer filter and re-base. The full-model side of this equality is itself pinned bitwise against
``from_pretrained`` by ``tests/gpu/parallelism/ep/test_lazy_load_inkling.py``.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/pp/test_pp_inkling_text_only.py
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

from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.deepep.dispatcher import destroy_all_dispatchers
from src.distributed.pipeline_parallel.stage import build_pipeline_stage
from src.model_loading.parallel.ep import load_ep_model_lazy
from src.model_loading.parallel.pp import _load_stage_lazy
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_INKLING_CONFIG
from tests.common.pp_fakes import registered_pp_backbones
from tests.common.utils import cleanup_memory, log

PP_SIZE = 2
DTYPE = torch.bfloat16
# 4 layers, sliding pattern [s, f, s, f] (period 2): the pp2 boundary at layer 2 is a whole
# pattern period, which validate_partition requires for Inkling's enumerate-indexed masks.
TEXT_CONFIG = {**TINY_INKLING_CONFIG, "num_hidden_layers": 4, "local_layer_ids": [0, 2]}
# The default vision config builds ~600k-wide projection layers (the upstream scale planner
# fits real patch sizes) — the composite save needs this tiny tower, whose TM-namespace keys
# then double as the tower-drop coverage for the text-only load.
TINY_VISION = {"temporal_patch_size": 2, "patch_size": 4, "n_layers": 4, "hidden_size": 32, "text_hidden_size": 256}


def _write_checkpoint(ctx) -> str:
    """Rank 0 writes the TM-layout text-only artifact (composite save + text-only config.json)."""
    path = os.path.join(tempfile.gettempdir(), "pp_inkling_text_only_ckpt")
    if ctx.rank == 0:
        torch.manual_seed(1234)
        composite = InklingConfig(
            text_config=dict(TEXT_CONFIG),
            vision_config=InklingVisionConfig(**TINY_VISION).to_dict(),
            image_token_id=500,
        )
        InklingForConditionalGeneration(composite).to(DTYPE).save_pretrained(path)
        text_config = InklingTextConfig(**TEXT_CONFIG)
        text_config.architectures = ["InklingForCausalLM"]
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(text_config.to_dict(), f)
    ctx.barrier()
    return path


def _snapshot(stage) -> tuple[dict[str, torch.Tensor], list[str]]:
    tensors: dict[str, torch.Tensor] = {}
    stranded: list[str] = []
    for name, tensor in stage.named_parameters():
        if tensor.is_meta:
            stranded.append(name)
            continue
        tensors[stage.global_parameter_name(name)] = tensor.detach().clone()
    return tensors, stranded


def _release(*models) -> None:
    for model in models:
        del model
    destroy_all_dispatchers()
    cleanup_memory()


# Inkling is refused for PP until a schedule, end-to-end and save/resume row cover it; this is its loader row.
@registered_pp_backbones("InklingTextModel")
def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    device = torch.device("cuda", ctx.local_rank)
    torch.cuda.set_device(device)
    ckpt = _write_checkpoint(ctx)

    # One EP group spanning the world: the two ranks are one stage's ranks, and the virtual
    # pipeline is walked by both of them.
    ep_config = EPConfig(
        ep_size=2,
        stage_world_size=ctx.world_size,
        stage_base_rank=0,
        node_local=True,
        nvlink_domain_size=2,
        experts_fsdp_managed=False,
    )
    ctx.broadcast_seed(1234)

    total_compared = 0
    for pp_rank in range(PP_SIZE):
        reference_model = load_ep_model_lazy(
            ckpt,
            ep_config,
            AutoConfig.from_pretrained(ckpt),
            dtype=DTYPE,
            model_class=InklingForCausalLM,
            attn_implementation="eager",
        )
        reference = build_pipeline_stage(reference_model, pp_rank, PP_SIZE)
        ref_tensors, ref_stranded = _snapshot(reference)
        ref_range = reference.layer_range
        _release(reference, reference_model)

        stage_model = _load_stage_lazy(
            ckpt,
            pp_rank,
            PP_SIZE,
            config=AutoConfig.from_pretrained(ckpt),
            ep_config=ep_config,
            dtype=DTYPE,
            model_class=InklingForCausalLM,
            attn_implementation="eager",
        )
        stage = build_pipeline_stage(stage_model, pp_rank, PP_SIZE)
        new_tensors, new_stranded = _snapshot(stage)
        new_range = stage.layer_range
        _release(stage, stage_model)

        checks[f"stage{pp_rank}_same_layer_range"] = ref_range == new_range
        checks[f"stage{pp_rank}_no_meta_tensors"] = not ref_stranded and not new_stranded
        shared = sorted(set(ref_tensors) & set(new_tensors))
        mismatched = [name for name in shared if not torch.equal(ref_tensors[name], new_tensors[name])]
        total_compared += len(shared)
        # Vacuity guard: an empty comparison is the signature of a broken name map, not a pass.
        checks[f"stage{pp_rank}_bitwise_identical"] = bool(shared) and not mismatched
        checks[f"stage{pp_rank}_covers_its_layers"] = sum(1 for n in shared if ".layers." in n) >= (
            new_range[1] - new_range[0]
        )
        log(
            f"[stage {pp_rank}] layers={new_range} compared={len(shared)} "
            f"mismatched={len(mismatched)} first={mismatched[:3]}"
        )
        del ref_tensors, new_tensors
        cleanup_memory()

    metrics["total_compared"] = float(total_compared)
    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, exact_world_size=2, prefix="pp_inkling_text_only")(run)

if __name__ == "__main__":
    main()
