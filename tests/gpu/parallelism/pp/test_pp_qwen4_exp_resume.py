#!/usr/bin/env python
"""Qwen3.8-Flash-Next PP checkpoint save, export and resume through the real trainer, at pp2 x ep2 or pp2 x etp2.

Its n-gram tables are host tables: each stage's writer puts its own under the unsplit names beside its parts, and the
stage-aware loader attaches it from those files.
"""

import json
import os
import sys

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, default_data_collator
from trl import SFTConfig

from halo.checkpoint.format import pp_stage_part_prefix
from halo.checkpoint.reading import resolve_checkpoint_weights
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.pp import _load_stage_lazy
from halo.models.external_tables import named_external_tables
from halo.models.unwrapping import unwrap_toolkit_wrappers
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.checkpoint_io import loading_problems
from tests.common.distributed import shared_output_dir, world_all
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_QWEN4_EXP_CONFIG
from tests.common.pp_trainers import (
    build_pp_args,
    build_sft_rows,
    losses_identical_across_ranks,
    probe_step,
    reload_reproduces_probe_loss,
    restored_moments_match_shard,
    tiny_vocab_tokenizer,
)
from tests.common.qwen4_exp import EOS_TOKEN_ID, build_tiny_qwen4_exp_checkpoint, load_host_table_model, stamped_config
from tests.common.rank_log import log

PP_SIZE = 2
PP_SPLIT = [3, 5]
SEQ_LEN = 64
N_STEPS = 8
SAVE_STEP = 4
BATCH = 4
OVERRIDES = {
    "num_hidden_layers": 8,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 2,
    "ple_layer_ids": [2, 6],
}
# The global layer holding each stage's table.
TABLE_LAYERS = {0: 1, 1: 5}
MODES = {"ep2": {"ep_size": 2}, "etp2": {"expert_tp_size": 2}}
MODE = "ep2"


def _table_keys(weight_map: dict, layer: int) -> dict[str, str]:
    prefix = f"model.layers.{layer}.ple.ple_embedding.ngram_embedding.shard_"
    return {key: name for key, name in weight_map.items() if key.startswith(prefix)}


@gpu_test_main(exact_world_size=4, prefix="pp_qwen4_exp_resume")
def run(ctx):
    checks, metrics = {}, {}
    rank = dist.get_rank()
    tokenizer = tiny_vocab_tokenizer()
    config = ParallelismConfig(
        world_size=ctx.world_size,
        gpus_per_node=2,
        nvlink_domain_size=2,
        pp_size=PP_SIZE,
        pp_split=PP_SPLIT,
        **MODES[MODE],
    )
    root = shared_output_dir(ctx)
    base_dir = os.path.join(root, "base")
    if rank == 0:
        build_tiny_qwen4_exp_checkpoint(base_dir, overrides=OVERRIDES)
    dist.barrier()
    rows = build_sft_rows(TINY_QWEN4_EXP_CONFIG["vocab_size"])

    def build_trainer(subdir: str, model, max_steps: int, save_steps: int | None = None):
        args = build_pp_args(
            SFTConfig,
            os.path.join(root, subdir),
            max_steps=max_steps,
            per_device_train_batch_size=BATCH,
            max_length=SEQ_LEN,
            save_every=save_steps,
            bf16=True,
            gradient_checkpointing=False,  # Resume compares optimizer state, not the checkpoint path.
        )
        return DistributedSFTTrainer(
            model=model,
            args=args,
            train_dataset=rows,
            processing_class=tokenizer,
            data_collator=default_data_collator,
            parallelism_config=config,
            moe_balancing="none",
        )

    # Train 4 steps, checkpoint at step 4.
    base_model = apply_ep_layers(load_host_table_model(base_dir, dtype=torch.bfloat16), config.ep_config())
    trainer_a = build_trainer("runA", base_model, SAVE_STEP, save_steps=SAVE_STEP)
    ctx.on_teardown(trainer_a.cleanup_ep)
    trainer_a.train()
    ckpt = os.path.join(root, "runA", f"checkpoint-{SAVE_STEP}")
    dist.barrier()
    # TRL copies the full-size tokenizer's special-token ids into the tiny config, which the n-gram embedding would
    # restart its history on at the reload.
    if rank == 0:
        for name in ("config.json", "generation_config.json"):
            path = os.path.join(ckpt, name)
            if os.path.isfile(path):
                with open(path) as f:
                    saved = json.load(f)
                saved["pad_token_id"], saved["eos_token_id"] = 0, EOS_TOKEN_ID
                with open(path, "w") as f:
                    json.dump(saved, f)
    dist.barrier()

    weight_map = resolve_checkpoint_weights(ckpt).weight_map
    tables_ok = True
    for stage_rank, layer in TABLE_LAYERS.items():
        files = set(_table_keys(weight_map, layer).values())
        prefix = f"{pp_stage_part_prefix(stage_rank, PP_SIZE)}-table00-"
        on_disk = all(os.path.isfile(os.path.join(ckpt, name)) for name in files)
        tables_ok &= bool(files) and on_disk and all(name.startswith(prefix) for name in files)
    checks["each_stage_wrote_its_table_under_its_prefix"] = tables_ok

    probe_batch, probe_loss = probe_step(trainer_a, rows, BATCH)
    exported = False
    if rank == 0:
        source = load_host_table_model(base_dir)
        expected = {name: torch.cat(table.parts) for name, table in named_external_tables(source)}
        reloaded, info = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, output_loading_info=True)
        problems = loading_problems(info)
        log(f"[export] loading problems: {problems or 'none'}")
        state = reloaded.state_dict()
        tables_equal = all(torch.equal(state[f"{name}.weight"].cpu(), table) for name, table in expected.items())
        exported = (
            not problems
            and tables_equal
            and reload_reproduces_probe_loss(
                reloaded.to(ctx.device), probe_batch, rows, config.data_parallel_size, probe_loss, ctx.device
            )
        )
    flag = torch.tensor([float(exported)], device=ctx.device)
    dist.broadcast(flag, src=0)
    checks["export_reloads_unsplit_with_its_tables_and_reproduces_the_probe_loss"] = bool(flag.item())

    resumed = _load_stage_lazy(
        ckpt,
        config.pp_rank,
        config.pp_size,
        config=stamped_config(ckpt),
        pp_split=PP_SPLIT,
        ep_config=config.ep_config(),
        dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    trainer_b = build_trainer("runB", resumed, N_STEPS)
    ctx.on_teardown(trainer_b.cleanup_ep)
    trainer_b._load_from_checkpoint(ckpt)
    stage = unwrap_toolkit_wrappers(trainer_b.model)
    source_tables = {
        name: torch.cat(table.parts) for name, table in named_external_tables(load_host_table_model(base_dir))
    }
    attached = [(stage.global_parameter_name(name), table) for name, table in named_external_tables(stage)]
    checks["stage_attached_its_own_table_from_the_checkpoint"] = world_all(
        len(attached) == 1
        and attached[0][1].files.directory == ckpt
        and torch.equal(torch.cat(attached[0][1].parts), source_tables[attached[0][0]])
    )

    trainer_b.create_optimizer_and_scheduler(num_training_steps=N_STEPS)
    trainer_b._load_optimizer_and_scheduler(ckpt)
    expert_prefixes = tuple(f"{name}." for name, _ in named_ep_layers(stage))
    checks["stage_has_ep_layers"] = bool(expert_prefixes)
    expert_ok, n_expert_fqns = restored_moments_match_shard(stage, trainer_b.optimizer, ckpt, rank, expert_prefixes)
    checks["expert_optimizer_moments_bitwise_equal_shard"] = expert_ok
    metrics["expert_state_fqns"] = n_expert_fqns

    trainer_b.train(resume_from_checkpoint=ckpt)
    losses = [e["loss"] for e in trainer_b.state.log_history if "loss" in e]
    checks["resume_trained_to_completion"] = trainer_b.state.global_step == N_STEPS and len(losses) == N_STEPS
    checks["resumed_losses_finite"] = bool(torch.isfinite(torch.tensor(losses)).all())
    checks["losses_identical_across_ranks"] = losses_identical_across_ranks(losses)
    metrics["losses"] = ", ".join(f"{v:.5f}" for v in losses)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    if "--mode" in sys.argv:
        i = sys.argv.index("--mode")
        MODE = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    if MODE not in MODES:
        raise SystemExit(f"--mode must be one of {sorted(MODES)}, got {MODE!r}")
    sys.exit(run())
