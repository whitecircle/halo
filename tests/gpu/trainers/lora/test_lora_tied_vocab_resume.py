#!/usr/bin/env python
"""A LoRA run that grew a tied model's vocabulary resumes its saved embedding under both tied names, and exports.

``added_special_tokens`` grows Qwen3-0.6B's tied embedding, so PEFT saves it with the adapters
(``save_embedding_layers="auto"``). A gathered save names the tied tensor once (``embed_tokens``) while the strict
restore's live selection names it under ``lm_head`` too; the restore has to take the saved value under every name of
the one tensor. Through the production path, on one rank, under FSDP2 (``--mode fsdp``) and under CP2:

  1. LoRA SFT to ``SAVE_STEP`` with a checkpoint there;
  2. a fresh model, its vocabulary grown again and the added rows zeroed (so only a real restore can bring the saved
     values back), resumed from that checkpoint to ``TOTAL_STEPS``: at the first resumed step the live embedding,
     read under ``embed_tokens`` and ``lm_head`` alike, equals the checkpoint's saved tensor, the two names are still
     one tensor, and the resumed steps are finite;
  3. the checkpoint merged into its base (``merge_peft_adapter``) loads as a complete causal LM at the grown vocab.

Run: torchrun --nproc_per_node=2 tests/gpu/trainers/lora/test_lora_tied_vocab_resume.py --mode fsdp
"""

import argparse
import math
import os

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, TrainerCallback
from trl import SFTConfig

from halo.checkpoint.format import ADAPTER_SAFETENSORS_FILE
from halo.checkpoint.parallel.adapters import saved_adapter_tensors
from halo.checkpoint.tools.adapter_merge import merge_peft_adapter
from halo.configs.script_arguments import CommonScriptArguments
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.tensor_parallel.hf_plan import is_input_embedding_tp_sharded
from halo.model_loading.tokenizer_setup import configure_run_tokenizer
from halo.runtime.dtensor import materialize_dtensor
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.adapters import load_peft_model, unwrap
from tests.common.checkpoint_io import loading_problems
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import shared_output_dir
from tests.common.harness import gpu_test_main
from tests.common.memory import finish_phase
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log
from tests.common.training_checks import step_losses

ADDED_TOKENS = ["<tied_vocab_x1>", "<tied_vocab_x2>"]
SAVE_STEP = 2
TOTAL_STEPS = 4
MAX_LENGTH = 512
EMBED_KEY = "base_model.model.model.embed_tokens.weight"
HEAD_KEY = "base_model.model.lm_head.weight"
MODES = {"single": (1, {}), "fsdp": (2, {}), "cp2": (2, {"cp_size": 2})}

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=sorted(MODES), required=True)
ARGS = parser.parse_args()
WORLD_SIZE, FIELDS = MODES[ARGS.mode]


class _LiveEmbedding(TrainerCallback):
    """At the first resumed step's start: the live tensor under each saved key, gathered whole, and whether
    ``embed_tokens`` and ``lm_head`` are still one tensor. COLLECTIVE (every rank gathers in key order)."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.captured: dict | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        model = unwrap(self.trainer.model)
        reshard_fsdp2_modules(model)
        live = saved_adapter_tensors(model)
        self.captured = {
            "tensors": {key: materialize_dtensor(live[key]).detach().float().cpu() for key in (EMBED_KEY, HEAD_KEY)},
            "tied": live[EMBED_KEY] is live[HEAD_KEY],
        }


def _build(output_dir: str, *, max_steps: int, zero_added_rows: bool):
    """The LoRA policy with its vocabulary grown through the script's tokenizer seam, and its trainer."""
    pc = ParallelismConfig(**FIELDS)
    model, tokenizer, peft_config = load_peft_model("lora", pc, model_name=QWEN3_0_6B)
    vocab = model.get_input_embeddings().num_embeddings
    tokenizer = configure_run_tokenizer(
        CommonScriptArguments(added_special_tokens=ADDED_TOKENS),
        model,
        tokenizer,
        MAX_LENGTH,
        embeddings_sharded=is_input_embedding_tp_sharded,
    )
    if zero_added_rows:
        with torch.no_grad():
            model.get_input_embeddings().weight[vocab:].zero_()
    trainer = DistributedSFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=output_dir,
            max_steps=max_steps,
            per_device_train_batch_size=2,
            learning_rate=1e-4,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=1,
            save_strategy="steps",
            save_steps=SAVE_STEP,
            report_to="none",
            logging_nan_inf_filter=False,
            max_length=MAX_LENGTH,
            dataloader_num_workers=0,
        ),
        train_dataset=create_sft_dataset(32, tokenizer, seed=42),
        processing_class=tokenizer,
        parallelism_config=pc,
        peft_config=peft_config,
    )
    return trainer, vocab


@gpu_test_main(exact_world_size=WORLD_SIZE, prefix=f"lora_tied_vocab_resume_{ARGS.mode}")
def run(ctx):
    root = shared_output_dir(ctx)
    run_dir = os.path.join(root, "run")
    checkpoint = os.path.join(run_dir, f"checkpoint-{SAVE_STEP}")
    checks: dict[str, bool] = {}

    trainer, vocab = _build(run_dir, max_steps=SAVE_STEP, zero_added_rows=False)
    grown = unwrap(trainer.model).get_input_embeddings().num_embeddings
    checks["vocabulary_grew_by_the_added_tokens"] = grown >= vocab + len(ADDED_TOKENS)
    checks["base_model_is_tied"] = unwrap(trainer.model).config.get_text_config().tie_word_embeddings
    trainer.train()
    finish_phase(trainer)
    del trainer
    ctx.barrier()
    saved = load_file(os.path.join(checkpoint, ADAPTER_SAFETENSORS_FILE)) if ctx.rank == 0 else {}
    checks["checkpoint_saves_the_grown_embedding"] = ctx.rank != 0 or (
        EMBED_KEY in saved and saved[EMBED_KEY].shape[0] == grown
    )

    resumed, _ = _build(os.path.join(root, "resumed"), max_steps=TOTAL_STEPS, zero_added_rows=True)
    probe = _LiveEmbedding(resumed)
    resumed.add_callback(probe)
    resumed.train(resume_from_checkpoint=checkpoint)
    losses = step_losses(resumed)
    checks["resumed_to_the_last_step"] = resumed.state.global_step == TOTAL_STEPS
    checks["resumed_steps_finite"] = len(losses) >= TOTAL_STEPS - SAVE_STEP and all(
        math.isfinite(loss) for loss in losses[-(TOTAL_STEPS - SAVE_STEP) :]
    )
    captured = probe.captured or {"tensors": {}, "tied": False}
    checks["embed_tokens_and_lm_head_still_one_tensor"] = captured["tied"]
    if ctx.rank == 0 and EMBED_KEY in saved:
        want = saved[EMBED_KEY].float()
        for key, name in ((EMBED_KEY, "embed_tokens"), (HEAD_KEY, "lm_head")):
            got = captured["tensors"].get(key)
            checks[f"{name}_restored_to_the_saved_embedding"] = got is not None and torch.equal(got, want)
        checks["added_rows_restored_off_the_zeroed_fresh_rows"] = bool(want[vocab:].abs().sum() > 0)
    finish_phase(resumed)
    ctx.barrier()

    if ctx.rank == 0:
        merged = os.path.join(root, "merged")
        merge_peft_adapter(checkpoint, merged, verbose=False)
        export, info = AutoModelForCausalLM.from_pretrained(merged, dtype=torch.bfloat16, output_loading_info=True)
        problems = loading_problems(info)
        checks["merged_export_loads_every_weight"] = not problems
        checks["merged_export_holds_the_grown_vocabulary"] = export.get_input_embeddings().num_embeddings == grown
        log(
            f"  merged export {merged}: {problems or 'complete'}, vocab {export.get_input_embeddings().num_embeddings}"
        )
    return {"checks": ctx.broadcast_checks(checks)}


if __name__ == "__main__":
    run()
