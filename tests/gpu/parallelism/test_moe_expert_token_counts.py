#!/usr/bin/env python
"""``moe_expert_token_counts.json`` through the real SFT trainer, one parallel layout per row.

The totals are raw counts, so what a ``moe/*`` share would cancel shows here: a token routed on several ranks (TP and
expert-TP partners), a pipeline stage missing from the file, a checkpoint recompute counted again. Each checkpoint's
file must hold, per layer, the tokens the run trained on times ``top_k``. Under EP each expert must hold exactly the
tokens dispatch delivered to it, counted at the expert compute, and a resume must continue the count from the
checkpoint's file. Gradient checkpointing is on, and steps 1, 2 and 4 are not logged.

The fsdp, tp2, cp2 and ep2aux rows balance with the aux loss, so ``MoEMetricsCallback`` counts at the routers' own
hooks, ep2aux's inside the EP layers. The others count in the EP layers' load recorder under a bias update
(``BalancingBiasCallback``).
"""

import argparse
import json
import os
import re
import sys

import torch
import torch.distributed as dist
from transformers import GptOssForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM, default_data_collator
from trl import SFTConfig

from halo.callbacks.balancing import configure_moe_balancing
from halo.callbacks.factory import build_run_callbacks
from halo.callbacks.moe_metrics import MoELoadMetricsBase, MoEMetricsCallback
from halo.checkpoint.format import MOE_EXPERT_TOKEN_COUNTS_FILE
from halo.configs.script_arguments import CommonScriptArguments
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.gradient_checkpointing import counts_toward_expert_load
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.checkpoint_io import save_schedule
from tests.common.distributed import shared_output_dir, world_all
from tests.common.harness import gpu_test_main
from tests.common.memory import finish_phase
from tests.common.models import TINY_GPTOSS_CONFIG, TINY_QWEN3_MOE_CONFIG
from tests.common.pp_equivalence import build_gptoss_model
from tests.common.pp_trainers import build_sft_rows, tiny_vocab_tokenizer
from tests.common.rank_log import log
from tests.common.resume_export import save_base_checkpoint

# name: (world size, layout, balancing). PP runs tiny GPT-OSS, whose bias exports, as PP requires; the rest tiny
# Qwen3-MoE. A cp row needs flash attention.
MODES = {
    "fsdp": (2, {"use_grouped_gemm": False}, "aux_loss"),
    "tp2": (2, {"tp_size": 2, "use_grouped_gemm": False}, "aux_loss"),
    "cp2": (2, {"cp_size": 2, "use_grouped_gemm": False}, "aux_loss"),
    "ep1": (2, {"ep_size": 1}, "bias_update_transient"),
    "ep2": (2, {"ep_size": 2}, "bias_update_transient"),
    "ep2aux": (2, {"ep_size": 2}, "aux_loss"),
    "ep2tp2": (2, {"ep_size": 2, "tp_size": 2}, "bias_update_transient"),
    "ep2cp2": (2, {"ep_size": 2, "cp_size": 2}, "bias_update_transient"),
    "ep2etp2": (4, {"ep_size": 2, "expert_tp_size": 2}, "bias_update_transient"),
    "pp2ep2": (4, {"pp_size": 2, "ep_size": 2, "gpus_per_node": 2, "nvlink_domain_size": 2}, "bias_update"),
}
SEQ_LEN = 64
BATCH = 2
ACCUMULATION = 2
TOTAL_STEPS = 4
SAVE_EVERY = 2
# Steps 1, 2 and 4 go unlogged, so a count kept only on logged steps misses them.
LOG_EVERY = 3
AUX_COEF = 0.01

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=sorted(MODES), default="ep2")
ARGS = parser.parse_args()
WORLD_SIZE, LAYOUT, BALANCING = MODES[ARGS.mode]


def _qwen3_moe_model() -> Qwen3MoeForCausalLM:
    torch.manual_seed(1234)
    config = Qwen3MoeConfig(
        **{**TINY_QWEN3_MOE_CONFIG, "router_aux_loss_coef": AUX_COEF}, pad_token_id=0, eos_token_id=1
    )
    return Qwen3MoeForCausalLM(config).to(torch.bfloat16)


def _sft_args(output_dir: str) -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        max_steps=TOTAL_STEPS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=ACCUMULATION,
        learning_rate=1e-4,
        lr_scheduler_type="constant",
        logging_steps=LOG_EVERY,
        **save_schedule(SAVE_EVERY),
        report_to=[],
        seed=42,
        bf16=True,
        gradient_checkpointing=True,
        max_length=SEQ_LEN,
        dataloader_num_workers=0,
    )


def _build_trainer(ctx, config: ParallelismConfig, source: str | None, output_dir: str, tokenizer, rows, *, resume):
    """The model, balancing and callbacks as the scripts wire them: PP from the module, the rest through the loader. A
    resume builds the model from the checkpoint, as the scripts do."""
    if config.is_pp_mode:
        model = (
            GptOssForCausalLM.from_pretrained(source, dtype=torch.bfloat16, attn_implementation="eager")
            if resume
            else build_gptoss_model()
        )
        apply_ep_layers(model, config.ep_config())
    else:
        model, _ = load_distributed_model(
            source,
            config,
            dtype=torch.bfloat16,
            trust_remote_code=False,
            attn_implementation="flash_attention_2" if config.is_cp_mode else "eager",
            preserve_checkpoint_precision=resume,
        )
    script_args = CommonScriptArguments(moe_balancing=BALANCING)
    args = _sft_args(output_dir)
    mode = configure_moe_balancing(script_args, model, config)
    trainer = DistributedSFTTrainer(
        model=model,
        args=args,
        train_dataset=rows,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        callbacks=build_run_callbacks(script_args, args, model, config, balancing_mode=mode),
        parallelism_config=config,
        moe_balancing=script_args.moe_balancing,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    return trainer


class _Tally:
    """What this rank trained on: tokens per micro-batch (all padding-free rows), and under EP the tokens dispatch
    delivered to each expert, counted at the expert compute on the pass that counts toward expert load."""

    def __init__(self, trainer, num_layers: int, num_experts: int):
        self.tokens = 0
        self.padding_free = True
        self.delivered = torch.zeros(num_layers, num_experts, dtype=torch.int64, device=torch.cuda.current_device())
        self.ep_layers = 0
        model = trainer._top_level_model()
        real_step = trainer.training_step

        def counting_step(model_arg, inputs, *args, **kwargs):
            self.tokens += inputs["input_ids"].numel()
            mask = inputs.get("attention_mask")
            self.padding_free &= mask is None or bool(mask.all())
            return real_step(model_arg, inputs, *args, **kwargs)

        trainer.training_step = counting_step
        for name, module in model.named_modules():
            if isinstance(module, EPMoELayerBase):
                global_name = model.global_parameter_name(name) if hasattr(model, "global_parameter_name") else name
                self._spy(module, int(re.search(r"layers\.(\d+)\.", global_name).group(1)))
                self.ep_layers += 1

    def _spy(self, layer: EPMoELayerBase, index: int) -> None:
        real = layer._compute_experts

        def compute(tokens, experts, weights, output_dtype, *, recv_counts):
            if layer.training and counts_toward_expert_load():
                local = experts[(experts >= 0) & (experts < layer.experts_per_rank)]
                self.delivered[index] += torch.bincount(local + layer.expert_start, minlength=layer.num_experts)
            return real(tokens, experts, weights, output_dtype, recv_counts=recv_counts)

        layer._compute_experts = compute


def _trained_tokens(tally: _Tally, config: ParallelismConfig) -> int:
    """COLLECTIVE. Tokens the run trained on: one count per data-parallel batch, from its lowest rank, since its TP, CP
    and expert-TP partners and its pipeline chain all train on it."""
    world = dist.get_world_size()
    dp_ranks = [torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(world)]
    dist.all_gather(dp_ranks, torch.tensor([config.get_data_parallel_rank()], device="cuda"))
    dp_ranks = [int(t.item()) for t in dp_ranks]
    leads = dp_ranks.index(config.get_data_parallel_rank()) == dist.get_rank()
    total = torch.tensor([tally.tokens if leads else 0], dtype=torch.int64, device="cuda")
    dist.all_reduce(total)
    return int(total.item())


def _delivered(tally: _Tally, config: ParallelismConfig) -> list[list[int]]:
    """COLLECTIVE. Tokens per layer and expert over the world, each counted once: TP and expert-TP partners both
    compute the tokens they share."""
    delivered = tally.delivered.clone()
    dist.all_reduce(delivered)
    return (delivered // config.router_token_replicas).tolist()


def _restore_tiny_token_ids(ctx, checkpoint: str) -> None:
    """COLLECTIVE. TRL copies the full-size tokenizer's special-token ids into the tiny config, which fails a reload's
    padding_idx assert; put back ids inside the tiny vocab."""
    if ctx.rank == 0:
        for name in ("config.json", "generation_config.json"):
            path = os.path.join(checkpoint, name)
            if os.path.isfile(path):
                with open(path) as handle:
                    saved = json.load(handle)
                saved["pad_token_id"], saved["eos_token_id"] = 0, 1
                with open(path, "w") as handle:
                    json.dump(saved, handle)
    dist.barrier()


def _read(checkpoint: str) -> dict | None:
    path = os.path.join(checkpoint, MOE_EXPERT_TOKEN_COUNTS_FILE)
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def _layer_totals(record: dict | None) -> list[int]:
    return [] if record is None else [sum(layer) for layer in record["counts"]]


@gpu_test_main(exact_world_size=WORLD_SIZE, prefix="moe_expert_token_counts")
def run(ctx):
    checks, metrics = {}, {}
    config = ParallelismConfig(world_size=ctx.world_size, **LAYOUT)
    family = TINY_GPTOSS_CONFIG if config.is_pp_mode else TINY_QWEN3_MOE_CONFIG
    num_layers = family["num_hidden_layers"]
    num_experts = family.get("num_local_experts", family.get("num_experts"))
    top_k = family["num_experts_per_tok"]
    tokenizer = tiny_vocab_tokenizer()
    rows = build_sft_rows(family["vocab_size"], n_rows=64, seq_len=SEQ_LEN)
    root = shared_output_dir(ctx)
    base = None if config.is_pp_mode else save_base_checkpoint(ctx, _qwen3_moe_model(), tokenizer, f"{root}/base")
    ep_mode = "ep_size" in LAYOUT

    # The unbroken run, saving at steps 2 and 4.
    trainer = _build_trainer(ctx, config, base, f"{root}/unbroken", tokenizer, rows, resume=False)
    callbacks = [cb for cb in trainer.callback_handler.callbacks if isinstance(cb, MoELoadMetricsBase)]
    checks["one_expert_load_callback"] = len(callbacks) == 1
    checks["aux_loss_counts_at_the_routers"] = all(
        isinstance(cb, MoEMetricsCallback) == (BALANCING == "aux_loss") for cb in callbacks
    )
    tally = _Tally(trainer, num_layers, num_experts)
    # PP turns accumulation into the schedule's microbatches, leaving one batch per step.
    accumulation = trainer.args.gradient_accumulation_steps
    checks["ep_layers_match_the_path"] = (tally.ep_layers > 0) == ep_mode
    trainer.train()
    tokens = _trained_tokens(tally, config)
    delivered = _delivered(tally, config)
    checks["padding_free_batches"] = world_all(tally.padding_free)
    finish_phase(trainer)

    at_2, at_4 = _read(f"{root}/unbroken/checkpoint-2"), _read(f"{root}/unbroken/checkpoint-4")
    checks["file_at_every_checkpoint"] = world_all(at_2 is not None and at_4 is not None)
    expected = tokens * top_k
    metrics["trained_tokens"], metrics["router_token_replicas"] = tokens, config.router_token_replicas
    metrics["layer_totals_at_4"] = _layer_totals(at_4)
    log(f"[rank {ctx.rank}] tokens={tokens} totals@2={_layer_totals(at_2)} totals@4={_layer_totals(at_4)}")
    checks["tokens_as_configured"] = tokens == TOTAL_STEPS * config.data_parallel_size * BATCH * accumulation * SEQ_LEN
    checks["every_layer_in_the_file"] = len(_layer_totals(at_4)) == num_layers
    checks["every_layer_holds_tokens_times_top_k"] = _layer_totals(at_4) == [expected] * num_layers
    checks["half_the_run_at_step_2"] = _layer_totals(at_2) == [expected // 2] * num_layers
    checks["steps_covered"] = (at_4 or {}).get("first_step") == 1 and (at_4 or {}).get("last_step") == TOTAL_STEPS
    checks["top_k_recorded"] = (at_4 or {}).get("top_k") == top_k
    if ep_mode:
        checks["every_expert_holds_what_dispatch_delivered"] = (at_4 or {}).get("counts") == delivered

    # Resumed at step 2 into a fresh output dir: the count continues from checkpoint-2's file.
    checkpoint = f"{root}/unbroken/checkpoint-2"
    _restore_tiny_token_ids(ctx, checkpoint)
    resumed = _build_trainer(ctx, config, checkpoint, f"{root}/resumed", tokenizer, rows, resume=True)
    resumed_tally = _Tally(resumed, num_layers, num_experts)
    resumed.train(resume_from_checkpoint=checkpoint)
    resumed_delivered = _delivered(resumed_tally, config)
    finish_phase(resumed)
    after = _read(f"{root}/resumed/checkpoint-4")
    log(f"[rank {ctx.rank}] resumed totals@4={_layer_totals(after)}")
    checks["resume_continues_the_count"] = _layer_totals(after) == [expected] * num_layers
    checks["resume_keeps_the_first_step"] = (after or {}).get("first_step") == 1
    if ep_mode:
        continued = [
            [old + new for old, new in zip(layer, more, strict=False)]
            for layer, more in zip((at_2 or {}).get("counts", []), resumed_delivered, strict=False)
        ]
        checks["resume_adds_what_dispatch_delivered_after_it"] = (after or {}).get("counts") == continued
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
