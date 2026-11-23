#!/usr/bin/env python
"""``init_from_scratch`` through the real loaders and SFT trainer, one parallel layout per row (issue #146).

The gate: a row trains two steps at learning rate 0 and saves through the gathered save, and the checkpoint must hold,
bit for bit, the weights :func:`build_keyed_model` draws on CPU for the same config and seed. Every row is held to the
same CPU draw, so every layout starts from the same weights, whatever the GPU. Then the run trains for real: four
steps with checkpoints at 2 and 4, a resume from 2 that ends where the unbroken run ends, and a stock
``from_pretrained`` reload of the final checkpoint whose head and backbone moved off the init. Every run loads as the
SFT script loads it, so the resume reads its checkpoint where the script does (EP, CP and TP) and never draws again.
ep1 runs with its experts FSDP2-sharded and as plain tensors, and the draw has to land in both.
"""

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM, default_data_collator
from trl import SFTConfig

from halo.callbacks.balancing import configure_moe_balancing
from halo.callbacks.factory import build_run_callbacks
from halo.checkpoint.reading import load_full_state_dict
from halo.checkpoint.writing import normalize_gathered_state_dict
from halo.configs.script_arguments import CommonScriptArguments, ModelArguments
from halo.distributed.parallelism_config import ParallelismConfig
from halo.entrypoints.model_setup import load_model_and_tokenizer
from halo.entrypoints.run_setup import prepare_distributed_resume
from halo.entrypoints.script import ScriptContext
from halo.model_loading.parallel.keyed_init import build_keyed_model
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.checkpoint_io import save_schedule
from tests.common.distributed import pin_deterministic_ep_dispatch, shared_output_dir
from tests.common.harness import gpu_test_main
from tests.common.memory import finish_phase
from tests.common.models import TINY_QWEN3_MOE_CONFIG
from tests.common.pp_trainers import build_sft_rows, tiny_vocab_tokenizer
from tests.common.rank_log import log
from tests.common.resume_export import export_checks, final_weight_checks

# name: layout. Every route that holds the whole model on a rank, a cp row on flash attention.
MODES = {
    "fsdp": {"use_grouped_gemm": False},
    "ep1": {"ep_size": 1},
    "ep1_unsharded": {"ep_size": 1, "fsdp_shard_ep1_experts": False},
    "ep2": {"ep_size": 2},
    "tp2": {"tp_size": 2, "use_grouped_gemm": False},
    "cp2": {"cp_size": 2, "use_grouped_gemm": False},
    "ep2cp2": {"ep_size": 2, "cp_size": 2},
    "ep2tp2": {"ep_size": 2, "tp_size": 2},
}
WORLD_SIZE = 2
INIT_SEED = 1234
SEQ_LEN = 64
BATCH = 2
TOTAL_STEPS = 4
SAVE_EVERY = 2

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=sorted(MODES), default="ep2")
ARGS = parser.parse_args()


def _tiny_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(**TINY_QWEN3_MOE_CONFIG, pad_token_id=0, eos_token_id=1, dtype=torch.bfloat16)


def _sft_args(output_dir: str, *, learning_rate: float, steps: int, resume_from: str | None) -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        resume_from_checkpoint=resume_from,
        max_steps=steps,
        per_device_train_batch_size=BATCH,
        learning_rate=learning_rate,
        lr_scheduler_type="constant",
        **save_schedule(SAVE_EVERY),
        report_to=[],
        seed=INIT_SEED,
        bf16=True,
        max_length=SEQ_LEN,
        dataloader_num_workers=0,
    )


def _train(ctx, config, source: str, output_dir: str, tokenizer, rows, *, learning_rate, steps, resume_from=None):
    """COLLECTIVE. A from-scratch run, fresh or resumed from ``resume_from``, loaded as the SFT script loads it: the
    resume resolves its weights' source as the script does, then the trainer restores its state."""
    script_args = CommonScriptArguments(moe_balancing="aux_loss")
    args = _sft_args(output_dir, learning_rate=learning_rate, steps=steps, resume_from=resume_from)
    model_config = ModelArguments(model_name_or_path=source)
    checkpoint, weights_source = prepare_distributed_resume(args, model_config, config, overwrite_output_dir=True)
    context = ScriptContext(config, ctx.local_rank, checkpoint, weights_source, DistributedSFTTrainer)
    model, _ = load_model_and_tokenizer(
        context,
        args,
        model_config,
        script_args,
        attn_default="flash_attention_2" if config.is_cp_mode else "eager",
        init_from_scratch=True,
    )
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
    trainer.train(resume_from_checkpoint=checkpoint)
    finish_phase(trainer)


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


def _init_checks(checkpoint: str) -> dict[str, bool]:
    """Rank 0: the checkpoint against the keyed CPU draw, tensor by tensor, in the checkpoint's own layout."""
    saved = load_full_state_dict(checkpoint) or {}
    drawn = build_keyed_model(
        Qwen3MoeForCausalLM, _tiny_config(), torch.bfloat16, seed=INIT_SEED, trust_remote_code=False
    )
    # In the layout the save writes (per-expert keys), so the expert banks are compared too.
    reference = normalize_gathered_state_dict(drawn, drawn.state_dict(), keep_live_dtype=True)
    missing = sorted(set(reference) - set(saved))
    extra = sorted(set(saved) - set(reference))
    differ = sorted(
        k for k in set(reference) & set(saved) if not torch.equal(saved[k], reference[k].to(saved[k].dtype))
    )
    log(
        f"  init vs the keyed CPU draw: {len(reference)} tensors, missing {missing[:4]}, extra {extra[:4]}, "
        f"differ {differ[:4]}"
    )
    return {"init_keys_match": not missing and not extra, "init_equals_the_keyed_draw": not differ and bool(saved)}


@gpu_test_main(exact_world_size=WORLD_SIZE, prefix="init_from_scratch_layouts")
def run(ctx):
    checks, metrics = {}, {}
    config = ParallelismConfig(world_size=ctx.world_size, **MODES[ARGS.mode])
    if config.has_ep_group:
        pin_deterministic_ep_dispatch()
    root = shared_output_dir(ctx)
    tokenizer = tiny_vocab_tokenizer()
    rows = build_sft_rows(TINY_QWEN3_MOE_CONFIG["vocab_size"], n_rows=64, seq_len=SEQ_LEN)
    source = f"{root}/source"
    if ctx.rank == 0:
        _tiny_config().save_pretrained(source)
        tokenizer.save_pretrained(source)
    ctx.barrier()

    # The gate: two steps at learning rate 0 leave the init as drawn, and the gathered save writes it.
    _train(ctx, config, source, f"{root}/init", tokenizer, rows, learning_rate=0.0, steps=SAVE_EVERY)
    init_checkpoint = f"{root}/init/checkpoint-{SAVE_EVERY}"
    _restore_tiny_token_ids(ctx, init_checkpoint)
    if ctx.rank == 0:
        checks |= _init_checks(init_checkpoint)

    # It trains, resumes and exports like any run.
    _train(ctx, config, source, f"{root}/unbroken", tokenizer, rows, learning_rate=1e-4, steps=TOTAL_STEPS)
    for step in (SAVE_EVERY, TOTAL_STEPS):
        _restore_tiny_token_ids(ctx, f"{root}/unbroken/checkpoint-{step}")
    _train(
        ctx,
        config,
        source,
        f"{root}/resumed",
        tokenizer,
        rows,
        learning_rate=1e-4,
        steps=TOTAL_STEPS,
        resume_from=f"{root}/unbroken/checkpoint-{SAVE_EVERY}",
    )
    if ctx.rank == 0:
        final_checks, final_metrics = final_weight_checks(
            f"{root}/unbroken/checkpoint-{TOTAL_STEPS}", f"{root}/resumed/checkpoint-{TOTAL_STEPS}"
        )
        checks |= final_checks
        metrics |= final_metrics
        checks |= export_checks(
            AutoModelForCausalLM, f"{root}/unbroken/checkpoint-{TOTAL_STEPS}", init_checkpoint, torch.bfloat16
        )
    # Rank 0 read the checkpoints from disk, and every rank reports its verdict.
    shared = [checks, metrics]
    dist.broadcast_object_list(shared, src=0)
    return {"checks": shared[0], "metrics": shared[1]}


if __name__ == "__main__":
    sys.exit(run())
