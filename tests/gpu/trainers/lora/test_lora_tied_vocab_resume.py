#!/usr/bin/env python
"""A LoRA run that grew its vocabulary saves the added rows with its adapters, resumes them and exports them.

Those rows exist in no base checkpoint, and the harness runs ``HF_HUB_OFFLINE=1``, so every adapter save carries them.
A gathered save names a tied tensor once while the strict restore names it under ``lm_head`` too. The resume model's
added rows are zeroed, so only a real restore can bring them back.
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
from halo.models.unwrapping import plain_adapter_key
from halo.runtime.dtensor import materialize_dtensor
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.adapters import load_peft_model, unwrap
from tests.common.checkpoint_io import loading_problems
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import shared_output_dir
from tests.common.harness import gpu_test_main
from tests.common.memory import finish_phase
from tests.common.models import QWEN3_0_6B, QWEN3_30B_A3B
from tests.common.rank_log import log
from tests.common.training_checks import step_losses

# 151,669 tokenizer ids and 151,936 padded checkpoint rows: 300 tokens grow them to 151,969.
ADDED_TOKENS = [f"<grown_vocab_{i}>" for i in range(300)]
SAVE_STEP = 2
TOTAL_STEPS = 4
MAX_LENGTH = 512
EMBED_KEY = "base_model.model.model.embed_tokens.weight"
HEAD_KEY = "base_model.model.lm_head.weight"
# mode: (ranks, ParallelismConfig fields, model, adapter mode)
MODES = {
    "single": (1, {}, QWEN3_0_6B, "lora"),
    "fsdp": (2, {}, QWEN3_0_6B, "lora"),
    "cp2": (2, {"cp_size": 2}, QWEN3_0_6B, "lora"),
    "qlora": (2, {}, QWEN3_0_6B, "qlora"),
    "ep2": (2, {"ep_size": 2}, QWEN3_30B_A3B, "lora"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=sorted(MODES), required=True)
ARGS = parser.parse_args()
WORLD_SIZE, FIELDS, MODEL, ADAPTER = MODES[ARGS.mode]


class _LiveEmbedding(TrainerCallback):
    """COLLECTIVE (every rank gathers the same keys in order). At the first resumed step: the live tensor under each
    embedding key, spelled as the saver writes it, and whether ``embed_tokens`` and ``lm_head`` are one tensor."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.captured: dict | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        model = unwrap(self.trainer.model)
        reshard_fsdp2_modules(model)
        live = {plain_adapter_key(key): tensor for key, tensor in saved_adapter_tensors(model).items()}
        present = [key for key in (EMBED_KEY, HEAD_KEY) if key in live]
        self.captured = {
            "tensors": {key: materialize_dtensor(live[key]).detach().float().cpu() for key in present},
            "tied": len(present) == 2 and live[EMBED_KEY] is live[HEAD_KEY],
        }


def _build(output_dir: str, *, max_steps: int, zero_added_rows: bool):
    pc = ParallelismConfig(**FIELDS)
    model, tokenizer, peft_config = load_peft_model(ADAPTER, pc, model_name=MODEL)
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
            for module in (model.get_input_embeddings(), model.get_output_embeddings()):
                module.weight[vocab:].zero_()
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
    return trainer, vocab, len(tokenizer)


@gpu_test_main(exact_world_size=WORLD_SIZE, prefix=f"lora_tied_vocab_resume_{ARGS.mode}")
def run(ctx):
    root = shared_output_dir(ctx)
    run_dir = os.path.join(root, "run")
    checkpoint = os.path.join(run_dir, f"checkpoint-{SAVE_STEP}")
    checks: dict[str, bool] = {}

    trainer, vocab, tokens = _build(run_dir, max_steps=SAVE_STEP, zero_added_rows=False)
    grown = unwrap(trainer.model).get_input_embeddings().num_embeddings
    tied = bool(unwrap(trainer.model).config.get_text_config().tie_word_embeddings)
    saved_keys = (EMBED_KEY,) if tied else (EMBED_KEY, HEAD_KEY)
    checks["vocabulary_grew_past_the_checkpoint_rows"] = grown == tokens > vocab
    checks["base_model_tying_is_the_modes"] = tied == (MODEL == QWEN3_0_6B)
    trainer.train()
    finish_phase(trainer)
    del trainer
    ctx.barrier()
    saved = load_file(os.path.join(checkpoint, ADAPTER_SAFETENSORS_FILE)) if ctx.rank == 0 else {}
    checks["checkpoint_saves_the_grown_embedding"] = ctx.rank != 0 or all(
        key in saved and saved[key].shape[0] == grown for key in saved_keys
    )

    resumed, _, _ = _build(os.path.join(root, "resumed"), max_steps=TOTAL_STEPS, zero_added_rows=True)
    probe = _LiveEmbedding(resumed)
    resumed.add_callback(probe)
    resumed.train(resume_from_checkpoint=checkpoint)
    losses = step_losses(resumed)
    checks["resumed_to_the_last_step"] = resumed.state.global_step == TOTAL_STEPS
    checks["resumed_steps_finite"] = len(losses) >= TOTAL_STEPS - SAVE_STEP and all(
        math.isfinite(loss) for loss in losses[-(TOTAL_STEPS - SAVE_STEP) :]
    )
    captured = probe.captured or {"tensors": {}, "tied": False}
    if tied:
        checks["embed_tokens_and_lm_head_still_one_tensor"] = captured["tied"]
    if ctx.rank == 0 and all(key in saved for key in saved_keys):
        for key, name in ((EMBED_KEY, "embed_tokens"), (HEAD_KEY, "lm_head")):
            want = saved[EMBED_KEY if tied else key].float()
            got = captured["tensors"].get(key)
            checks[f"{name}_restored_to_the_saved_tensor"] = got is not None and torch.equal(got, want)
        checks["added_rows_restored_off_the_zeroed_fresh_rows"] = all(
            bool(saved[key][vocab:].float().abs().sum() > 0) for key in saved_keys
        )
    finish_phase(resumed)
    ctx.barrier()

    if ctx.rank == 0:
        # Rank 1 waits in broadcast_checks meanwhile: the CPU merge must finish inside the NCCL watchdog.
        merged = os.path.join(root, "merged")
        merge_peft_adapter(checkpoint, merged, dtype=torch.bfloat16, verbose=False)
        export, info = AutoModelForCausalLM.from_pretrained(merged, dtype=torch.bfloat16, output_loading_info=True)
        problems = loading_problems(info)
        checks["merged_export_loads_every_weight"] = not problems
        checks["merged_export_holds_the_grown_vocabulary"] = export.get_input_embeddings().num_embeddings == grown
        if all(key in saved for key in saved_keys):
            for module, key, name in (
                (export.get_input_embeddings(), EMBED_KEY, "embed_tokens"),
                (export.get_output_embeddings(), EMBED_KEY if tied else HEAD_KEY, "lm_head"),
            ):
                exported = module.weight[vocab:].detach()
                checks[f"merged_{name}_holds_the_saved_added_rows"] = torch.equal(
                    exported, saved[key][vocab:].to(exported.dtype)
                )
        log(
            f"  merged export {merged}: {problems or 'complete'}, vocab {export.get_input_embeddings().num_embeddings}"
        )
    return {"checks": ctx.broadcast_checks(checks)}


if __name__ == "__main__":
    run()
