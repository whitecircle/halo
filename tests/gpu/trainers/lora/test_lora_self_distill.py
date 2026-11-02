#!/usr/bin/env python
"""The SDPG self-distillation trainer under adapters and full fine-tuning, on every parallel layout.

``DistributedSelfDistillationTrainer`` takes ``peft_config`` directly (wraps with get_peft_model internally, like
the GRPO trainers). Each mode trains through the production path and asserts that the OPD term fired and is
non-trivial (the privileged-hint teacher sees a different prompt from the student), that the loss is finite,
that what the mode trains moved (only the adapters, or the weights themselves on a full fine-tune), that the
layout is the one asked for, and that a checkpoint of the trained tensors round-trips.

Modes (``MODES``): lora / qlora (dense Qwen3-0.6B, FSDP2) · lora_ep / expert_lora (GptOss-20B, EP=``--ep``) ·
lora_etp (attention LoRA under pure expert TP) · lora_ep_etp (four ranks, EP2 x ETP2) · full_ep1 / ep1_unsharded
(GptOss-20B full fine-tune at ep1, the experts FSDP-sharded or whole and FSDP-ignored) · full_ep2_tp2 (GptOss-20B,
EP2 + TP2) · full_tp2 / zero3 / hsdp (dense full fine-tune under TP2, FULL_SHARD, HSDP over two simulated NVLink
domains). ``--world`` adds DP replicas (two EP groups, TP x DP) and ``--gpus-per-node`` simulated domains.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_self_distill.py --mode lora
"""

import argparse
import math
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM
from trl import SFTConfig

from halo.data.collators.self_distill import SelfDistillTextCollator
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.runtime.collectives import barrier
from halo.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
from tests.common.adapters import (
    DENSE_MODEL,
    MOE_MODEL,
    assert_adapter_checkpoint,
    assert_adapters_moved,
    assert_only_adapters_trainable,
    is_expert_lora_active,
    load_peft_model,
    snapshot_adapters,
    unwrap,
)
from tests.common.checkpoint_io import loading_problems
from tests.common.distributed import shared_output_dir, world_any
from tests.common.dp_layout import dp_layout_checks, hsdp_layout, probe_forward
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.parallel_shape import parallel_shape_checks
from tests.common.rank_log import log
from tests.common.weight_sync import local_parameters, moved_parameters

MAX_STEPS = 3
NUM_TRAIN_SAMPLES = 32
MAX_SEQ_LENGTH = 512
# The privileged hint the teacher reads and the student does not.
HINT_TEMPLATE = "\n[Hint] The correct answer is: {answer}.\n"
# The OPD term's floor once the teacher reads the hint: a teacher whose prompt equals the student's scores the
# student's own distribution and lands at zero up to bf16 noise.
OPD_FLOOR = 1e-3
ADAPTER_LEARNING_RATE = 1e-3
FULL_LEARNING_RATE = 1e-5
# Each mode's model, adapter (None: full fine-tune), ParallelismConfig fields (``ep`` the --ep width, HSDP sized
# by the world) and its default world size.
MODES = {
    "lora": ("dense", "lora", lambda ep, world: {}, 2),
    "qlora": ("dense", "qlora", lambda ep, world: {}, 2),
    "lora_ep": ("moe", "lora", lambda ep, world: {"ep_size": ep}, 2),
    "expert_lora": ("moe", "expert_lora", lambda ep, world: {"ep_size": ep}, 2),
    "lora_etp": ("moe", "lora", lambda ep, world: {"ep_size": 1, "expert_tp_size": 2}, 2),
    "lora_ep_etp": ("moe", "lora", lambda ep, world: {"ep_size": 2, "expert_tp_size": 2}, 4),
    "full_ep1": ("moe", None, lambda ep, world: {"ep_size": 1}, 2),
    "ep1_unsharded": ("moe", None, lambda ep, world: {"ep_size": 1, "fsdp_shard_ep1_experts": False}, 2),
    "full_ep2_tp2": ("moe", None, lambda ep, world: {"ep_size": 2, "tp_size": 2}, 2),
    "full_tp2": ("dense", None, lambda ep, world: {"tp_size": 2}, 2),
    "zero3": ("dense", None, lambda ep, world: {"fsdp_reshard_after_forward": True}, 2),
    "hsdp": ("dense", None, lambda ep, world: hsdp_layout(world), 4),
}

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="lora", choices=sorted(MODES))
parser.add_argument("--ep", type=int, default=2)
parser.add_argument("--world", type=int, default=None, help="ranks (default: the mode's own)")
parser.add_argument("--gpus-per-node", type=int, default=0, help="simulated NVLink domain width (0: the launch's)")
ARGS = parser.parse_args()
FAMILY, ADAPTER, FIELDS_FOR, DEFAULT_WORLD = MODES[ARGS.mode]
WORLD_SIZE = ARGS.world or DEFAULT_WORLD


def _assistant_marker(tokenizer) -> str:
    """Derive the model's assistant-turn header so completion-only masking works for any family.

    The hardcoded Qwen ``<|im_start|>assistant`` marker masks every token on gpt-oss (harmony format)
    → zero loss. Render a turn with a sentinel answer and return the text just before it; that
    sub-sequence is exactly what precedes the response in input_ids, so the collator finds it.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "ANSWERSENTINEL"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    head = rendered.split("ANSWERSENTINEL")[0]
    return head[head.rindex("Q") + 1 :]


def _self_distill_dataset(n: int) -> Dataset:
    rows = []
    for i in range(n):
        a, b = i % 9, (i * 2) % 7
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": f"What is {a} + {b}?"},
                    {"role": "assistant", "content": f"The answer is {a + b}."},
                ],
                "answer": str(a + b),
            }
        )
    return Dataset.from_list(rows)


def _load(fields: dict):
    """The policy through the production path: with its adapters (``(model, tokenizer, peft_config)``), or whole
    for a full fine-tune (``peft_config`` None)."""
    parallelism_config = ParallelismConfig(**fields)
    attn = "flex_attention" if parallelism_config.has_ep_group else "sdpa"
    if ADAPTER is not None:
        return load_peft_model(ADAPTER, parallelism_config, attn_implementation=attn), parallelism_config
    model, tokenizer = load_distributed_model(
        model_name_or_path=MOE_MODEL if FAMILY == "moe" else DENSE_MODEL,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return (model, tokenizer, None), parallelism_config


def _layer0_weights(model) -> dict[str, torch.Tensor]:
    """This rank's copy of every trainable tensor of the first decoder layer: its attention and norms, which
    every step's gradient reaches, and its experts, which only routed tokens reach."""
    return local_parameters(model, keep=lambda name, param: param.requires_grad and ".layers.0." in name)


def _full_finetune_moved(model, before: dict[str, torch.Tensor]) -> dict[str, bool]:
    """Every dense tensor of layer 0 moved on this rank, and some expert tensor moved somewhere in the world (a
    rank's experts move only where tokens were routed to them). COLLECTIVE (one flag reduction)."""
    after = _layer0_weights(model)
    expert_ids = {id(param) for layer in ep_layers(model) for _, param in layer.expert_named_params()}
    experts = {name for name, param in model.named_parameters() if id(param) in expert_ids and name in before}
    moved = set(moved_parameters(before, after))
    dense = set(before) - experts
    log(f"  layer 0 moved: {len(moved & dense)}/{len(dense)} dense, {len(moved & experts)}/{len(experts)} expert")
    checks = {"layer0_dense_weights_moved": bool(dense) and dense <= moved}
    if FAMILY == "moe":
        checks["some_expert_weight_moved"] = world_any(bool(moved & experts))
    return checks


def _full_checkpoint_round_trip(ctx, trainer) -> dict[str, bool]:
    """The trained model saved and read back complete through ``AutoModelForCausalLM``. COLLECTIVE (the save)."""
    export = os.path.join(shared_output_dir(ctx), "full_ckpt")
    trainer.save_model(export)
    barrier()
    if ctx.rank != 0:
        return {}
    _, info = AutoModelForCausalLM.from_pretrained(
        export, dtype=torch.bfloat16, trust_remote_code=True, output_loading_info=True
    )
    problems = loading_problems(info)
    log(f"  [checkpoint] {export}: {problems or 'complete'}")
    return {"full_checkpoint_loads_every_weight": not problems}


@gpu_test_main(exact_world_size=WORLD_SIZE, prefix="lora_self_distill")
def run(ctx) -> dict:
    fields = {
        **FIELDS_FOR(ARGS.ep, ctx.world_size),
        **({"gpus_per_node": ARGS.gpus_per_node} if ARGS.gpus_per_node else {}),
    }
    log(f"\n{'=' * 70}\n  Self-distill (SDPG) — mode={ARGS.mode}, world={ctx.world_size}, {fields}\n{'=' * 70}")
    (model, tokenizer, peft_config), parallelism_config = _load(fields)
    expert_lora = is_expert_lora_active(model)

    collator = SelfDistillTextCollator(
        tokenizer,
        max_length=MAX_SEQ_LENGTH,
        conversation_field="messages",
        hint_template=HINT_TEMPLATE,
        answer_field="answer",
        solution_field=None,
        confidence_power=1.0,
        response_prompt_template=_assistant_marker(tokenizer),
        train_on_completions_only=True,
    )
    config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=ADAPTER_LEARNING_RATE if ADAPTER else FULL_LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataset_num_proc=1,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    trainer = DistributedSelfDistillationTrainer(
        model,
        args=config,
        train_dataset=_self_distill_dataset(NUM_TRAIN_SAMPLES),
        processing_class=tokenizer,
        peft_config=peft_config,
        data_collator=collator,
        parallelism_config=parallelism_config,
        sdpg_loss="reverse_kl",
        sdpg_beta_base=1.0,
        reference_kl_coef=0.0,
        reference_kl_loss="unnormalized_kl",
        confidence_weight_opd=True,
        opd_exclude_eos=True,
    )
    ctx.on_teardown(trainer.cleanup_ep)

    wrapped = unwrap(trainer.model)
    checks: dict[str, bool] = parallel_shape_checks(wrapped, parallelism_config)
    if ADAPTER is not None:
        ok, detail = assert_only_adapters_trainable(wrapped)
        checks["only_adapters_trainable"] = ok
        log(f"  [trainable] {detail}")
        before = snapshot_adapters(wrapped, expert_lora=expert_lora)
    else:
        before = _layer0_weights(wrapped)

    barrier()
    result = trainer.train()
    checks["loss_finite"] = math.isfinite(result.training_loss)
    checks["steps_done"] = result.global_step == MAX_STEPS
    opd = [float(e["opd_loss"]) for e in trainer.state.log_history if "opd_loss" in e]
    checks["opd_fired"] = bool(opd)
    checks["opd_term_above_the_unhinted_floor"] = bool(opd) and max(opd) > OPD_FLOOR
    log(f"  training_loss={result.training_loss:.5f}, steps={result.global_step}, opd_loss per step={opd}")
    checks |= dp_layout_checks(
        wrapped,
        fields,
        world_size=ctx.world_size,
        ep_config=trainer._ep_config,
        forward=lambda: probe_forward(trainer.model, ctx.device),
    )

    if ADAPTER is not None:
        ok, detail = assert_adapters_moved(before, snapshot_adapters(unwrap(trainer.model), expert_lora=expert_lora))
        checks["adapters_moved"] = ok
        log(f"  [moved] {detail}")
        ok, detail = assert_adapter_checkpoint(trainer, ctx.output_dir, ctx.rank, expert_lora=expert_lora)
        checks["adapter_checkpoint"] = ok
        log(f"  [checkpoint] {detail}")
    else:
        checks |= _full_finetune_moved(unwrap(trainer.model), before)
        checks |= _full_checkpoint_round_trip(ctx, trainer)

    # Only rank 0 reads the checkpoint back; share its verdict so every rank exits alike.
    return {"checks": ctx.broadcast_checks(checks)}


if __name__ == "__main__":
    run()
