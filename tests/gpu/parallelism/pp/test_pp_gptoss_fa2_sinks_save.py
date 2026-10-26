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
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, GptOssConfig, GptOssForCausalLM, default_data_collator
from trl import SFTConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.models.patches.gpt_oss_sinks import _reset_gpt_oss_sinks
from src.trainers.sft import DistributedSFTTrainer
from tests.common.harness import gpu_test_main, skip_unless_blackwell
from tests.common.models import QWEN3_0_6B, TINY_GPTOSS_CONFIG
from tests.common.utils import log

PP_SIZE = 2
SEQ_LEN = 64
N_STEPS = 4
NUM_LAYERS = TINY_GPTOSS_CONFIG["num_hidden_layers"]
NUM_HEADS = TINY_GPTOSS_CONFIG["num_attention_heads"]


def build_model(attn_implementation: str = "eager"):
    torch.manual_seed(1234)
    cfg = GptOssConfig(**TINY_GPTOSS_CONFIG, pad_token_id=0, eos_token_id=1)
    model = GptOssForCausalLM(cfg).to(torch.bfloat16)
    if attn_implementation != "eager":
        model.set_attn_implementation(attn_implementation)
    return model


def build_dataset() -> Dataset:
    g = torch.Generator().manual_seed(7)
    rows = torch.randint(4, TINY_GPTOSS_CONFIG["vocab_size"], (16, SEQ_LEN), generator=g)
    labels = rows.clone()
    for i in range(labels.size(0)):
        labels[i, : 4 + (i % 11)] = -100
    return Dataset.from_dict(
        {
            "input_ids": rows.tolist(),
            "labels": labels.tolist(),
            "attention_mask": torch.ones_like(rows).tolist(),
        }
    )


@gpu_test_main(exact_world_size=4, prefix="pp_gptoss_fa2_sinks")
def run(ctx):
    checks, metrics = {}, {}
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)

    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)

    # The Hopper FA2 production state (sinks=None), reproduced under FA4 — its sink pass-through
    # tolerates None, and the save path only keys on the None state, not the attention impl.
    model = build_model("flash_attention_4")
    _reset_gpt_oss_sinks(model, model.config, attn_implementation="flash_attention_2")
    checks["sinks_dropped_from_state_dict"] = not any("sinks" in k for k in model.state_dict())

    args = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=N_STEPS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,  # becomes the pipeline's microbatch count
        learning_rate=5e-3,
        lr_scheduler_type="constant",
        max_length=SEQ_LEN,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        logging_nan_inf_filter=False,
        seed=42,
        bf16=True,
        gradient_checkpointing=False,
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=args,
        train_dataset=build_dataset(),
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
    probe_dataset = build_dataset()
    dp_rank = config.get_data_parallel_rank()
    probe_rows = torch.tensor(probe_dataset["input_ids"][dp_rank * 4 : dp_rank * 4 + 4])
    probe_labels = torch.tensor(probe_dataset["labels"][dp_rank * 4 : dp_rank * 4 + 4])
    probe_batch = {
        "input_ids": probe_rows,
        "labels": probe_labels,
        "attention_mask": torch.ones_like(probe_rows),
    }
    trainer.state.global_step += 1
    probe_loss = trainer.training_step(trainer.model, dict(probe_batch))
    metrics["probe_loss"] = float(probe_loss)

    dirs: list = [None] * ctx.world_size
    dist.all_gather_object(dirs, ctx.output_dir)
    save_dir = dirs[0]
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
            reference_keys = set(build_model().state_dict())  # eager reference — sinks present
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
            with torch.no_grad():
                out = reloaded.to(ctx.device)(
                    input_ids=probe_batch["input_ids"].to(ctx.device),
                    attention_mask=probe_batch["attention_mask"].to(ctx.device),
                )
            shift_logits = out.logits[:, :-1].float()
            shift_labels = probe_batch["labels"][:, 1:].to(ctx.device)
            ce_sum = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            global_count = float((torch.tensor(probe_dataset["labels"][:8])[:, 1:] != -100).sum())
            expected_loss = float(ce_sum / (global_count / config.data_parallel_size))
            rel = abs(expected_loss - float(probe_loss)) / max(abs(expected_loss), 1e-9)
            log(f"[reload-value] probe={float(probe_loss):.6f} reloaded={expected_loss:.6f} rel={rel:.3e}")
            if rel > 5e-2:  # FA4 vs eager bf16 kernels differ; gate loose but real
                values_ok = False
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
