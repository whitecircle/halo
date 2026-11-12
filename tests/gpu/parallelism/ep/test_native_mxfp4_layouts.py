#!/usr/bin/env python
"""A native MXFP4 gpt-oss checkpoint (the ``openai/gpt-oss-*`` layout) through the life of a run under the layouts the
EP=2 round trip (``test_native_mxfp4_roundtrip.py``) does not reach: pipeline stages, EP x ETP, and ep1 with its
experts sharded by FSDP2.

pp2 and ep2 x etp2 decode the experts in the lazy loaders. ep1 has no EP group, so it is a whole-model load:
``from_pretrained`` hands the release to transformers, which dequantizes it to bf16 in the Halo images (they ship no
``kernels`` package). Each layout saves through its own path: PP writes one shard per stage, ETP unfolds its column
halves and ep1 gathers DTensors. The run trains with a checkpoint, resumes
from it and saves. A save taken before the first step must hold every tensor of the dequantized ``from_pretrained``
reference bit for bit, the decoded experts included and the attention sinks apart, which the load resets by default.
The weights are bf16 from the load on, so no save may carry the release's ``quantization_config``, and a stock
``from_pretrained`` of the final save must load every weight, build no quantizer and reproduce the run's loss. The PP
checkpoint also goes through ``merge_pp_stage_parts``, as a save on per-node storage does before export.

    torchrun --nproc_per_node=4 tests/gpu/parallelism/ep/test_native_mxfp4_layouts.py --layout pp2
    torchrun --nproc_per_node=4 tests/gpu/parallelism/ep/test_native_mxfp4_layouts.py --layout ep2_etp2
    torchrun --nproc_per_node=2 tests/gpu/parallelism/ep/test_native_mxfp4_layouts.py --layout ep1
"""

import argparse
import json
import math
import os

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer, GptOssForCausalLM, Mxfp4Config, default_data_collator
from trl import SFTConfig

from halo.checkpoint.tools.pp_stage_merge import merge_pp_stage_parts
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.lazy_safetensors.conversion import MXFP4_PROJECTIONS
from halo.model_loading.parallel.lazy_plan import is_lazy_loadable
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.introspection.unwrapping import unwrap_toolkit_wrappers
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.checkpoint_io import fixed_batch_loss, loading_problems, safetensors_state_dict
from tests.common.distributed import ensure_model_downloaded, shared_output_dir, world_all
from tests.common.ep_reference import random_token_batch
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import QWEN3_0_6B
from tests.common.native_mxfp4 import write_native_mxfp4
from tests.common.numerics import tensors_equal_at_narrower_dtype
from tests.common.pp_trainers import build_pp_args, build_sft_rows, probe_step, reload_reproduces_probe_loss
from tests.common.rank_log import log
from tests.common.tiny_models import TINY_MOE_FAMILIES, build_tiny_family_checkpoint
from tests.common.tolerances import TOL
from tests.common.training_checks import step_losses

# World size and the layout's ParallelismConfig fields. ep1 keeps the default ``fsdp_shard_ep1_experts``, so its
# experts are FSDP2 DTensors.
LAYOUTS = {
    "pp2": (4, {"pp_size": 2, "gpus_per_node": 2, "nvlink_domain_size": 2}),
    "ep2_etp2": (4, {"ep_size": 2, "expert_tp_size": 2}),
    "ep1": (2, {"ep_size": 1}),
}
SEED = 42
SAVE_AT_STEP = 2
TOTAL_STEPS = 4
SEQ_LEN = 64
BATCH = 4

parser = argparse.ArgumentParser()
parser.add_argument("--layout", choices=sorted(LAYOUTS), required=True)
ARGS = parser.parse_args()
WORLD_SIZE, FIELDS = LAYOUTS[ARGS.layout]


def _trainer(model_path: str, pc, tokenizer, rows, output_dir: str, max_steps: int, *, resume: bool = False):
    model, _ = load_distributed_model(
        model_name_or_path=model_path,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        use_liger_kernel=False,
        trust_remote_code=True,
        preserve_checkpoint_precision=resume,
    )
    args = build_pp_args(
        SFTConfig,
        output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=BATCH,
        max_length=SEQ_LEN,
        save_every=None if resume else SAVE_AT_STEP,
        learning_rate=1e-5,
        bf16=True,
        gradient_checkpointing=False,
    )
    return DistributedSFTTrainer(
        model=model,
        args=args,
        train_dataset=rows,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        parallelism_config=pc,
    )


def _claims_mxfp4(directory: str) -> bool:
    with open(os.path.join(directory, "config.json")) as f:
        return "quantization_config" in json.load(f)


def _rank0_untrained_save_checks(source: str, initial_dir: str) -> dict[str, bool]:
    """Rank 0 alone. The save taken before the first step against what transformers loads from ``source`` when asked
    to dequantize."""
    reference = GptOssForCausalLM.from_pretrained(
        source, dtype=torch.bfloat16, attn_implementation="eager", quantization_config=Mxfp4Config(dequantize=True)
    ).state_dict()
    written = safetensors_state_dict(initial_dir)
    # ``reset_sinks`` defaults to true, so the sinks are the one tensor the load changes by design.
    differing = sorted(
        key
        for key in set(written) & set(reference)
        if not key.endswith(".sinks") and not tensors_equal_at_narrower_dtype(written[key], reference[key].cpu())
    )
    experts = [key for key in reference if key.rpartition(".")[2] in MXFP4_PROJECTIONS]
    log(
        f"untrained save: {len(written)} tensors, {len(experts)} decoded expert banks, "
        f"{len(differing)} differ from the dequantized reference {differing[:4]}"
    )
    return {
        "untrained_save_config_does_not_claim_mxfp4": not _claims_mxfp4(initial_dir),
        "untrained_save_has_the_reference_keys": set(written) == set(reference),
        "untrained_save_has_decoded_expert_banks": bool(experts) and all(key in written for key in experts),
        "untrained_save_equals_dequantized_reference": not differing,
    }


def _rank0_merge_pp(checkpoint: str, merged_dir: str) -> dict[str, bool]:
    """Rank 0 alone. The stage-sharded checkpoint merged as a per-node save is before export, then read back by a
    stock ``from_pretrained``."""
    merge_pp_stage_parts([checkpoint], merged_dir)
    merged, written = safetensors_state_dict(merged_dir), safetensors_state_dict(checkpoint)
    differing = sorted(k for k in set(merged) & set(written) if not torch.equal(merged[k], written[k]))
    if differing:
        log(f"merged and stage-sharded differ at {differing[:4]}")
    _, info = GptOssForCausalLM.from_pretrained(
        merged_dir, dtype=torch.bfloat16, attn_implementation="eager", output_loading_info=True
    )
    problems = loading_problems(info)
    log(f"merge_pp_stage_parts: {len(merged)} tensors, loading problems {problems or 'none'}")
    return {
        "pp_merge_config_does_not_claim_mxfp4": not _claims_mxfp4(merged_dir),
        "pp_merge_tensors_equal_the_stage_shards": set(merged) == set(written) and not differing,
        "pp_merge_loads_every_weight": not problems,
    }


def _rank0_reload(final_dir: str, device) -> tuple[GptOssForCausalLM, dict[str, bool]]:
    """Rank 0 alone. The final save through a stock ``from_pretrained``."""
    reloaded, info = GptOssForCausalLM.from_pretrained(
        final_dir, dtype=torch.bfloat16, attn_implementation="eager", output_loading_info=True
    )
    problems = loading_problems(info)
    log(f"stock reload of {final_dir}: loading problems {problems or 'none'}")
    return reloaded.to(device).eval(), {
        "stock_reload_loads_every_weight": not problems,
        "stock_reload_builds_no_quantizer": getattr(reloaded, "hf_quantizer", None) is None,
    }


@gpu_test_main(exact_world_size=WORLD_SIZE, prefix=f"native_mxfp4_layouts_{ARGS.layout}")
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    root = shared_output_dir(ctx)
    source = os.path.join(root, "gpt_oss_native_mxfp4")
    train_out = os.path.join(root, "train_out")
    ckpt_dir = os.path.join(train_out, f"checkpoint-{SAVE_AT_STEP}")
    initial_dir, final_dir, merged_dir = (os.path.join(root, kind) for kind in ("initial", "final", "merged"))

    ensure_model_downloaded(QWEN3_0_6B, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    if ctx.rank == 0:
        build_tiny_family_checkpoint(TINY_MOE_FAMILIES["gpt_oss"], source, tokenizer, SEED)
        write_native_mxfp4(source)
    ctx.barrier()
    checks["source_claims_mxfp4"] = _claims_mxfp4(source)
    checks["lazy_gate_admits_source"] = is_lazy_loadable(source)

    pp = "pp_size" in FIELDS
    pc = ParallelismConfig(world_size=ctx.world_size, **FIELDS) if pp else ParallelismConfig(**FIELDS)
    checks["layout_is_the_one_named"] = {
        "pp2": pc.pp_size == 2,
        "ep2_etp2": pc.ep_size == 2 and pc.expert_tp_size == 2,
        "ep1": pc.ep_size == 1 and pc.experts_fsdp_managed,
    }[ARGS.layout]
    vocab_size = AutoConfig.from_pretrained(source).vocab_size  # The tokenizer's, as the build sizes it.
    rows = build_sft_rows(vocab_size, seq_len=SEQ_LEN)

    log(f"--- {ARGS.layout}: train {SAVE_AT_STEP} steps from the MXFP4 source, checkpoint at step {SAVE_AT_STEP} ---")
    trainer = _trainer(source, pc, tokenizer, rows, train_out, SAVE_AT_STEP)
    ctx.on_teardown(trainer.cleanup_ep)
    live_config = unwrap_toolkit_wrappers(trainer.model).config
    checks["loaded_config_does_not_claim_mxfp4"] = getattr(live_config, "quantization_config", None) is None
    trainer.save_model(initial_dir)
    ctx.barrier()
    untrained = [_rank0_untrained_save_checks(source, initial_dir) if ctx.rank == 0 else None]
    dist.broadcast_object_list(untrained, src=0)
    checks |= untrained[0]
    trainer.train()
    first = step_losses(trainer)
    checks["first_phase_losses_finite"] = len(first) == SAVE_AT_STEP and all(math.isfinite(x) for x in first)
    del trainer
    cleanup_memory()
    ctx.barrier()
    checks["checkpoint_config_does_not_claim_mxfp4"] = not _claims_mxfp4(ckpt_dir)
    if pp:
        merged = [_rank0_merge_pp(ckpt_dir, merged_dir) if ctx.rank == 0 else None]
        dist.broadcast_object_list(merged, src=0)
        checks |= merged[0]

    log(f"--- {ARGS.layout}: resume from checkpoint-{SAVE_AT_STEP} to step {TOTAL_STEPS} ---")
    trainer = _trainer(ckpt_dir, pc, tokenizer, rows, train_out, TOTAL_STEPS, resume=True)
    ctx.on_teardown(trainer.cleanup_ep)
    trainer.train(resume_from_checkpoint=ckpt_dir)
    resumed = step_losses(trainer)
    checks["resumed_to_the_last_step"] = trainer.state.global_step == TOTAL_STEPS
    checks["resumed_losses_finite"] = bool(resumed) and all(math.isfinite(x) for x in resumed)

    log(f"--- {ARGS.layout}: final save, stock reload ---")
    trainer.save_model(final_dir)
    ctx.barrier()
    checks["final_config_does_not_claim_mxfp4"] = not _claims_mxfp4(final_dir)
    if pp:
        # One more pipeline step with no update scores the saved weights.
        probe_batch, probe_loss = probe_step(trainer, rows, BATCH)
        verdict = [None]
        if ctx.rank == 0:
            reloaded, reload_checks = _rank0_reload(final_dir, ctx.device)
            reload_checks["stock_reload_loss_matches_the_run"] = reload_reproduces_probe_loss(
                reloaded, probe_batch, rows, pc.data_parallel_size, probe_loss, ctx.device
            )
            verdict = [reload_checks]
    else:
        ids, labels = random_token_batch(vocab_size, batch=2, seq=SEQ_LEN, device=ctx.device, seed=SEED)
        run_loss = fixed_batch_loss(trainer.model, ids, labels)
        metrics["run_loss"] = run_loss
        checks["run_loss_finite"] = math.isfinite(run_loss)
        verdict = [None]
        if ctx.rank == 0:
            reloaded, reload_checks = _rank0_reload(final_dir, ctx.device)
            reload_loss = fixed_batch_loss(reloaded, ids, labels)
            log(f"run loss {run_loss:.6f} vs stock reload {reload_loss:.6f} (|Δ|={abs(reload_loss - run_loss):.2e})")
            reload_checks["stock_reload_loss_matches_the_run"] = (
                abs(reload_loss - run_loss) < TOL.parallel_vs_baseline_loss_abs
            )
            verdict = [reload_checks]
    dist.broadcast_object_list(verdict, src=0)
    checks |= verdict[0]
    checks = {name: world_all(ok, ctx.device) for name, ok in checks.items()}
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
