#!/usr/bin/env python
"""PP save of a GptOss model whose sinks were dropped to None (the Hopper FA2 fine-tuning state).

Under FA2 the production loader sets every ``self_attn.sinks`` to None (``_reset_gpt_oss_sinks``),
which removes them from ``named_parameters()``/``state_dict()``. The PP save must re-emit them
neutralized (dtype min — zero softmax weight, the reset-sinks fine-tuning semantics) under global
names: the helper names sinks by position in the stage's sliced layer list, so a missing
``global_parameter_name`` remap would collide stage 1's re-emissions with stage 0's layers and the
merged checkpoint would drop half the sinks (a from_pretrained meta-tensor error) or overwrite the
wrong layers.

The pipeline runs FA4, so the test skips off Blackwell. GptOss FA2 runs only on Hopper: transformers
reroutes its flash_attention_2 to a hub FA3 kernel that has no Blackwell build. FA4's sink path
passes ``sinks`` straight through and tolerates None — the same no-sink softmax, so the save branch
under test is byte-identical to a Hopper FA2 run's. This test fails when:

  * any of the 8 ``model.layers.{i}.self_attn.sinks`` keys is missing from the merged checkpoint
    (re-emission skipped, or emitted under stage-local names)
  * a re-emitted sink is not the neutralized dtype-min fill
  * the checkpoint does not reload into the unsplit class with the exact reference key set
  * the reloaded model (eager + neutralized sinks — mathematically the same no-sink softmax) fails
    to reproduce the pipeline's probe loss on the final weights

Topology: 4 GPUs, pp2 × dp2 (dense FSDP2 stages — MoE without EP). Run:
    torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_gptoss_fa2_sinks_save.py
"""

import sys

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, GptOssForCausalLM, default_data_collator
from trl import SFTConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.models.attention.gpt_oss_sinks import _reset_gpt_oss_sinks
from src.trainers.sft import DistributedSFTTrainer
from tests.common.distributed import shared_output_dir
from tests.common.harness import gpu_test_main, skip_unless_blackwell
from tests.common.models import QWEN3_0_6B, TINY_GPTOSS_CONFIG
from tests.common.pp_equivalence import build_gptoss_model
from tests.common.pp_trainers import build_pp_args, build_sft_rows, probe_step, reload_reproduces_probe_loss
from tests.common.utils import log

PP_SIZE = 2
SEQ_LEN = 64
N_STEPS = 4
NUM_LAYERS = TINY_GPTOSS_CONFIG["num_hidden_layers"]
NUM_HEADS = TINY_GPTOSS_CONFIG["num_attention_heads"]


@gpu_test_main(exact_world_size=4, prefix="pp_gptoss_fa2_sinks")
def run(ctx):
    checks, metrics = {}, {}
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)

    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)

    # The Hopper FA2 production state (sinks=None), reproduced under FA4 — its sink pass-through
    # tolerates None, and the save path only keys on the None state, not the attention impl.
    model = build_gptoss_model()
    model.set_attn_implementation("flash_attention_4")
    _reset_gpt_oss_sinks(model, model.config, attn_implementation="flash_attention_2")
    checks["sinks_dropped_from_state_dict"] = not any("sinks" in k for k in model.state_dict())

    args = build_pp_args(
        SFTConfig,
        ctx.output_dir,
        max_steps=N_STEPS,
        per_device_train_batch_size=4,
        max_length=SEQ_LEN,
        bf16=True,
        gradient_checkpointing=False,
    )
    rows = build_sft_rows(TINY_GPTOSS_CONFIG["vocab_size"])
    trainer = DistributedSFTTrainer(
        model=model,
        args=args,
        train_dataset=rows,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        parallelism_config=config,
        moe_balancing="none",
    )
    trainer.train()

    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    checks["ran_all_steps"] = trainer.state.global_step == N_STEPS
    checks["losses_finite"] = len(losses) == N_STEPS and bool(torch.isfinite(torch.tensor(losses)).all())
    metrics["first_loss"], metrics["last_loss"] = losses[0], losses[-1]

    # Probe loss on the final weights: one more pipeline step with no optimizer update after it,
    # so the reloaded model must reproduce this replica's loss on the same batch.
    probe_batch, probe_loss = probe_step(trainer, rows, 4)
    metrics["probe_loss"] = float(probe_loss)

    save_dir = shared_output_dir(ctx)
    # The trainer synced config.pad/eos to the real Qwen tokenizer's ids, which exceed the tiny
    # vocab — restore so from_pretrained can build the embedding.
    trainer.model.config.pad_token_id, trainer.model.config.eos_token_id = 0, 1
    trainer.save_model(save_dir)
    dist.barrier()

    reload_ok = True
    sinks_neutralized_ok = True
    values_ok = True
    if ctx.rank == 0:
        try:
            reloaded = GptOssForCausalLM.from_pretrained(save_dir, dtype=torch.bfloat16)
            reference_keys = set(build_gptoss_model().state_dict())  # eager reference — sinks present
            reloaded_keys = set(reloaded.state_dict())
            if reloaded_keys != reference_keys:
                log(
                    f"reload key mismatch: missing={sorted(reference_keys - reloaded_keys)[:8]} "
                    f"extra={sorted(reloaded_keys - reference_keys)[:8]}"
                )
                reload_ok = False
            neutral = torch.full((NUM_HEADS,), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16)
            for i in range(NUM_LAYERS):
                sink = reloaded.state_dict().get(f"model.layers.{i}.self_attn.sinks")
                if sink is None or not torch.equal(sink, neutral):
                    log(f"[sinks] layer {i}: {'missing' if sink is None else 'not neutralized'}")
                    sinks_neutralized_ok = False
        except Exception as exc:
            log(f"[reload] FAILED: {exc!r}")
            reload_ok = False
            sinks_neutralized_ok = False
        try:
            # Eager forward with neutralized sinks = FA2's no-sink softmax (the sink column's
            # exp(dtype min) underflows to 0), so the reloaded probe loss must match the pipeline's.
            values_ok = reload_reproduces_probe_loss(
                reloaded.to(ctx.device), probe_batch, rows, config.data_parallel_size, probe_loss, ctx.device
            )
        except Exception as exc:
            log(f"[reload-value] FAILED: {exc!r}")
            values_ok = False

    flags = torch.tensor([float(reload_ok), float(sinks_neutralized_ok), float(values_ok)], device=ctx.device)
    dist.broadcast(flags, src=0)
    checks["save_reloads_with_reference_key_set"] = bool(flags[0].item())
    checks["all_sinks_reemitted_neutralized_under_global_names"] = bool(flags[1].item())
    checks["reloaded_model_reproduces_fa2_probe_loss"] = bool(flags[2].item())

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    skip_unless_blackwell("this test")
    sys.exit(run())
