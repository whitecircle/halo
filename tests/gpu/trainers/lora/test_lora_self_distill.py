#!/usr/bin/env python
"""LoRA / QLoRA / native-expert-LoRA on the SDPG self-distillation trainer (+ adapter checkpoint).

``DistributedSelfDistillationTrainer`` takes ``peft_config`` directly (wraps with get_peft_model
internally, like the GRPO trainers). This drives the production adapter path through it and asserts
the adapter-only invariant, that the adapters move under the SDPG loss (L_sft + beta*L_OPD), and
that an adapter checkpoint round-trips.

Modes (``--mode``): lora / qlora (dense Qwen3-0.6B, FSDP2) · lora_ep / expert_lora (GptOss-20B, EP=2).

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_self_distill.py --mode lora
"""

import argparse
import math
import sys
import traceback

from accelerate import PartialState
from datasets import Dataset
from trl import SFTConfig

from src.data.collators.self_distill import SelfDistillTextCollator
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
from tests.common.distributed import cleanup_dirs, init_distributed, setup_cache_dirs, teardown_distributed
from tests.common.peft_helpers import (
    assert_adapter_checkpoint,
    assert_adapters_moved,
    assert_only_adapters_trainable,
    is_expert_lora_active,
    load_peft_model,
    snapshot_adapters,
    unwrap,
)
from tests.common.utils import cleanup_memory, log

MAX_STEPS = 3
NUM_TRAIN_SAMPLES = 32
MAX_SEQ_LENGTH = 512


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


def run(mode: str, ep: int) -> bool:
    rank, world_size, local_rank = init_distributed()
    PartialState()
    output_dir, cache_dir = setup_cache_dirs(f"lora_self_distill_{mode}", rank)
    log(f"\n{'=' * 70}\n  Self-distill (SDPG) adapter test — mode={mode}, world={world_size}, ep={ep}\n{'=' * 70}")
    checks: dict[str, bool] = {}
    try:
        parallelism_config = (
            ParallelismConfig(ep_size=ep) if mode in ("lora_ep", "expert_lora") else ParallelismConfig()
        )
        attn = "flex_attention" if parallelism_config.is_ep_mode else "sdpa"
        model, tokenizer, peft_config = load_peft_model(mode, parallelism_config, attn_implementation=attn)
        expert_lora = is_expert_lora_active(model)
        log(f"  expert_lora_active={expert_lora}, peft_config={'set' if peft_config else 'None'}")

        collator = SelfDistillTextCollator(
            tokenizer,
            max_length=MAX_SEQ_LENGTH,
            conversation_field="messages",
            hint_template="\n[Hint] The correct answer is: {answer}.\n",
            answer_field="answer",
            solution_field=None,
            response_prompt_template=_assistant_marker(tokenizer),
            train_on_completions_only=True,
        )
        config = SFTConfig(
            output_dir=output_dir,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            dataset_num_proc=1,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            fsdp="",
            ddp_find_unused_parameters=parallelism_config.is_ep_mode,
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
        )

        wrapped = unwrap(trainer.model)
        ok, detail = assert_only_adapters_trainable(wrapped)
        checks["only_adapters_trainable"] = ok
        log(f"  [trainable] {detail}")
        adapters_before = snapshot_adapters(wrapped, expert_lora=expert_lora)

        barrier()
        result = trainer.train()
        checks["loss_finite"] = math.isfinite(result.training_loss)
        checks["steps_done"] = result.global_step == MAX_STEPS
        opd = [e["opd_loss"] for e in trainer.state.log_history if "opd_loss" in e]
        checks["opd_fired"] = bool(opd)
        log(f"  training_loss={result.training_loss:.5f}, steps={result.global_step}, opd_logged={len(opd)}")

        adapters_after = snapshot_adapters(unwrap(trainer.model), expert_lora=expert_lora)
        ok, detail = assert_adapters_moved(adapters_before, adapters_after)
        checks["adapters_moved"] = ok
        log(f"  [moved] {detail}")

        ok, detail = assert_adapter_checkpoint(trainer, output_dir, rank, expert_lora=expert_lora)
        checks["adapter_checkpoint"] = ok
        log(f"  [checkpoint] {detail}")

        passed = all(checks.values())
        log(f"\n  SELF-DISTILL {mode.upper()} " + ("PASSED" if passed else "FAILED"))
        for k, v in checks.items():
            log(f"    - {'PASS' if v else 'FAIL'}: {k}")
        return passed
    except Exception as e:
        log(f"\n  SELF-DISTILL {mode} FAILED WITH ERROR: {e}")
        if rank == 0:
            traceback.print_exc()
        return False
    finally:
        if "trainer" in dir() and hasattr(trainer, "cleanup_ep"):
            trainer.cleanup_ep()
        barrier()
        cleanup_dirs(output_dir, cache_dir)
        cleanup_memory()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="lora", choices=["lora", "qlora", "lora_ep", "expert_lora"])
    parser.add_argument("--ep", type=int, default=2)
    args = parser.parse_args()
    success = run(args.mode, args.ep)
    teardown_distributed()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
