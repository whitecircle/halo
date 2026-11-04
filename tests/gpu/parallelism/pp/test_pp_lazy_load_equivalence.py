#!/usr/bin/env python
"""Stage-aware PP loading is bitwise identical to load-everything-then-drop.

An off-by-``lo`` key rebase loads layers ``0..k`` into every stage with no crash and a plausible loss, so only
``torch.equal`` catches it; a tolerance accepts a wrong-but-similar layer. ``--fp32_masters``: the stage reload skips
EP layers on resume, so the stage loader's read is the only one routers and experts get.
"""

import os
import shutil
import sys
import tempfile

import torch
import torch.distributed as dist
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    GptOssConfig,
    GptOssForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from halo.checkpoint.reading import resolve_checkpoint_weights
from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.deepep.dispatcher import destroy_all_dispatchers
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.pipeline_parallel.partition import resolve_layer_partition
from halo.distributed.pipeline_parallel.stage import PP_STAGE_PARTITION_ATTR, build_pipeline_stage
from halo.model_loading.parallel.ep import load_ep_model_lazy
from halo.model_loading.parallel.pp import _load_stage_lazy
from tests.common import pp_equivalence as ppeq
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import TINY_GPTOSS_CONFIG, TINY_QWEN3_CONFIG, TINY_QWEN3_MOE_CONFIG
from tests.common.rank_log import log

PP_SIZE = 2
# --pp_split lo,hi: a non-uniform partition, where equal stage sizes do not mask an off-by-lo rebase.
PP_SPLIT = None
# --ep 2: a tiny MoE sharded across the two ranks, so the PP layer filter must compose with the EP expert slice.
EP_SIZE = 1
# --head seq_cls: a head the checkpoint does not carry, which the loader must random-init.
HEAD = "causal_lm"
# --moe_family qwen3_moe: per-expert keys reach the loader through ExpertFuser; GptOss saves fused, so the default --ep
# node never exercises the fusing path under a stage re-base.
MOE_FAMILY = "gptoss"
FP32_MASTERS = False

DTYPE = torch.bfloat16


def _build_source_model():
    torch.manual_seed(1234)
    if EP_SIZE > 1 and MOE_FAMILY == "qwen3_moe":
        model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**TINY_QWEN3_MOE_CONFIG, pad_token_id=0, eos_token_id=1))
    elif EP_SIZE > 1:
        model = GptOssForCausalLM(GptOssConfig(**TINY_GPTOSS_CONFIG, pad_token_id=0, eos_token_id=1))
    else:
        model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG, attn_implementation="eager"))
    return model.to(torch.float32 if FP32_MASTERS else DTYPE)


def _model_class():
    return AutoModelForSequenceClassification if HEAD == "seq_cls" else AutoModelForCausalLM


def _write_checkpoint(ctx) -> str:
    suffix = "_fp32" if FP32_MASTERS else ""
    path = os.path.join(tempfile.gettempdir(), f"pp_lazy_load_equivalence_ep{EP_SIZE}_{MOE_FAMILY}{suffix}")
    if ctx.rank == 0:
        _build_source_model().save_pretrained(path, safe_serialization=True)
    ctx.barrier()
    return path


def _reference_stage(ckpt: str, pp_rank: int, ep_config, device):
    if ep_config is None:
        model = _model_class().from_pretrained(ckpt, dtype=DTYPE, attn_implementation="eager").to(device)
    else:
        model = load_ep_model_lazy(
            ckpt, ep_config, AutoConfig.from_pretrained(ckpt), dtype=DTYPE, attn_implementation="eager"
        )
    return build_pipeline_stage(model, pp_rank, PP_SIZE, pp_split=PP_SPLIT)


def _absent_head_is_usable(head: torch.Tensor | None, device) -> bool:
    """Real, finite and the same on every rank: FSDP2 never syncs parameters, so a per-rank random init trains two
    heads."""
    if head is None or head.is_meta or not bool(torch.isfinite(head).all()) or float(head.abs().sum()) == 0.0:
        return False
    rank0 = head.float().to(device)
    dist.broadcast(rank0, src=0)
    return bool(torch.equal(rank0, head.float().to(device)))


def _stage_aware_stage(ckpt: str, pp_rank: int, ep_config, device):
    """Also returns the task head captured before the split: materialized on every stage, dropped from all but the
    last."""
    del device
    model = _load_stage_lazy(
        ckpt,
        pp_rank,
        PP_SIZE,
        config=AutoConfig.from_pretrained(ckpt),
        pp_split=PP_SPLIT,
        ep_config=ep_config,
        dtype=DTYPE,
        model_class=_model_class(),
        attn_implementation="eager",
    )
    assert getattr(model, PP_STAGE_PARTITION_ATTR, None) is not None, "the loader must stamp its partition"
    score = getattr(model, "score", None)
    head = None if score is None else score.weight.detach().clone()
    return build_pipeline_stage(model, pp_rank, PP_SIZE, pp_split=PP_SPLIT), head


@gpu_test_main(min_world_size=2, exact_world_size=2, prefix="pp_lazy_load_equivalence")
def run(ctx):
    device = torch.device("cuda", ctx.local_rank)
    ckpt = _write_checkpoint(ctx)

    def _remove_checkpoint():
        if ctx.rank == 0:
            shutil.rmtree(ckpt, ignore_errors=True)

    ctx.on_teardown(_remove_checkpoint)

    # One EP group spanning the world: the two ranks are one stage's ranks, and both walk the virtual pipeline below.
    ep_config = (
        EPConfig(
            ep_size=EP_SIZE,
            stage_world_size=ctx.world_size,
            stage_base_rank=0,
            node_local=True,
            nvlink_domain_size=2,
            fp32_router=FP32_MASTERS,
            fp32_experts=FP32_MASTERS,
            experts_fsdp_managed=False,
        )
        if EP_SIZE > 1
        else None
    )

    # The keys both paths must carry, in the unsplit model's naming.
    checkpoint_keys = set(resolve_checkpoint_weights(ckpt).weight_map)

    checks, metrics = {}, {}
    worst_mismatch = None
    total_compared = 0

    # Only rank 0 wrote the checkpoint, so its RNG is one draw ahead. Production gets this identical seed, which the
    # absent-head init relies on, from configure_training_run's set_seed.
    ctx.broadcast_seed(1234)
    head_initialized = True

    for pp_rank in range(PP_SIZE):
        reference = _reference_stage(ckpt, pp_rank, ep_config, device)
        ref_tensors, ref_stranded = ppeq.snapshot_stage(reference, buffers=True)
        ref_range = reference.layer_range
        destroy_all_dispatchers()
        cleanup_memory()

        stage, head = _stage_aware_stage(ckpt, pp_rank, ep_config, device)
        if HEAD == "seq_cls":
            head_initialized &= _absent_head_is_usable(head, device)
        new_tensors, new_stranded = ppeq.snapshot_stage(stage, buffers=True)
        new_range = stage.layer_range
        # EP layers hold expert slices as their own attributes (``mlp.gate_up_proj``), outside the checkpoint's keys
        # (``mlp.experts.gate_up_proj``), so a comparison scoped to checkpoint keys alone would never see the experts.
        expert_banks = {
            stage.global_parameter_name(f"{path}.{name}")
            for path, layer in named_ep_layers(stage)
            for name, _ in layer._base_expert_named_params()
        }
        destroy_all_dispatchers()
        cleanup_memory()

        checks[f"stage{pp_rank}_same_layer_range"] = ref_range == new_range
        checks[f"stage{pp_rank}_no_meta_tensors"] = not ref_stranded and not new_stranded
        # The paths may differ on recomputed non-checkpoint buffers (the stage-aware path runs finalize_loaded_model's
        # rotary fixers). Any key the checkpoint carries must be on both sides.
        key_diff = set(ref_tensors) ^ set(new_tensors)
        checks[f"stage{pp_rank}_no_checkpoint_key_diff"] = not (key_diff & checkpoint_keys)
        if ref_stranded or new_stranded:
            log(f"[stage {pp_rank}] meta tensors: reference={ref_stranded[:5]} stage_aware={new_stranded[:5]}")
        if key_diff:
            log(
                f"[stage {pp_rank}] non-checkpoint key diff: "
                f"stage_aware_only={sorted(set(new_tensors) - set(ref_tensors))[:5]} "
                f"reference_only={sorted(set(ref_tensors) - set(new_tensors))[:5]}"
            )

        shared = set(ref_tensors) & set(new_tensors)
        compared = sorted(shared & (checkpoint_keys | expert_banks))
        if EP_SIZE > 1:
            checks[f"stage{pp_rank}_compares_expert_banks"] = bool(expert_banks) and expert_banks <= shared
        mismatched = [name for name in compared if not torch.equal(ref_tensors[name], new_tensors[name])]
        total_compared += len(compared)
        if mismatched and worst_mismatch is None:
            worst_mismatch = (pp_rank, mismatched[:5])
        # An empty comparison signals a broken name map.
        checks[f"stage{pp_rank}_bitwise_identical"] = bool(compared) and not mismatched
        checks[f"stage{pp_rank}_covers_its_layers"] = sum(1 for name in compared if ".layers." in name) >= (
            new_range[1] - new_range[0]
        )
        if FP32_MASTERS:
            # Bitwise agreement passes if both sides rounded alike; the masters must also carry precision bf16 cannot
            # hold.
            masters = [name for name in compared if new_tensors[name].dtype == torch.float32]
            checks[f"stage{pp_rank}_masters_fp32_on_both_sides"] = bool(masters) and all(
                ref_tensors[name].dtype == torch.float32 for name in masters
            )
            checks[f"stage{pp_rank}_router_and_experts_are_masters"] = any(
                name in expert_banks for name in masters
            ) and any(".router." in name or ".gate." in name for name in masters)
            checks[f"stage{pp_rank}_masters_exceed_bf16"] = any(
                bool((new_tensors[name] != new_tensors[name].to(torch.bfloat16).float()).any()) for name in masters
            )
            log(f"[stage {pp_rank}] fp32 masters: {len(masters)} e.g. {masters[:4]}")
        derived = sorted(shared - checkpoint_keys - expert_banks)
        checks[f"stage{pp_rank}_derived_state_finite"] = all(
            bool(torch.isfinite(new_tensors[name]).all()) for name in derived
        )
        log(
            f"[stage {pp_rank}] layers={new_range} compared={len(compared)} derived={len(derived)} "
            f"mismatched={len(mismatched)} first={mismatched[:3]}"
        )
        del ref_tensors, new_tensors
        cleanup_memory()

    # A partition both paths agree on being wrong the same way would otherwise pass every check above.
    torch.manual_seed(1234)
    full_partition = resolve_layer_partition(_build_source_model(), PP_SIZE, PP_SPLIT)
    checks["partition_covers_every_layer"] = full_partition[0][0] == 0 and all(
        full_partition[i][1] == full_partition[i + 1][0] for i in range(PP_SIZE - 1)
    )
    if PP_SPLIT is not None:
        checks["manual_split_honored"] = [hi - lo for lo, hi in full_partition] == PP_SPLIT

    if HEAD == "seq_cls":
        # A per-rank random init would diverge the FSDP replicas from step 1.
        checks["absent_head_initialized"] = head_initialized
    metrics["tensors_compared"] = total_compared
    metrics["ep_size"] = EP_SIZE
    metrics["fp32_masters"] = FP32_MASTERS
    if worst_mismatch is not None:
        log(f"first mismatching stage/keys: {worst_mismatch}")
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    if "--pp_split" in sys.argv:
        i = sys.argv.index("--pp_split")
        PP_SPLIT = [int(part) for part in sys.argv[i + 1].split(",")]
        del sys.argv[i : i + 2]
    if "--ep" in sys.argv:
        i = sys.argv.index("--ep")
        EP_SIZE = int(sys.argv[i + 1])
        del sys.argv[i : i + 2]
    if "--head" in sys.argv:
        i = sys.argv.index("--head")
        HEAD = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    if "--moe_family" in sys.argv:
        i = sys.argv.index("--moe_family")
        MOE_FAMILY = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    if "--fp32_masters" in sys.argv:
        FP32_MASTERS = True
        sys.argv.remove("--fp32_masters")
    sys.exit(run())
