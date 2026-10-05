"""Native dense SFT TP adapter lifecycle shared by the TP=2 and TP=4 GPU entries."""

import os
import subprocess
import sys
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file as safetensors_load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.env import env_str
from src.models.patches.buffer_fixes import finalize_loaded_model
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.models import QWEN3_0_6B
from tests.common.peft_helpers import assert_adapters_moved, snapshot_adapters
from tests.common.tolerances import TOL
from tests.common.tp_lora_native import assert_native_factors, assert_replicas_equal, require_native_runtime
from tests.common.utils import cleanup_memory, log, training_run_checks

QWEN3_MODEL = env_str("HALO_TEST_LORA_SAVE_LOAD_MODEL", QWEN3_0_6B)
MAX_STEPS = 5
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 2e-4
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
SEED = 42
LOGIT_RTOL = 3e-2
LOGIT_ATOL = TOL.logprob_atol
MERGE_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "after_training" / "merge_peft_adapters.py"


def _adapter_maps_match(actual: dict, expected: dict, *, replay: bool = False) -> bool:
    rtol = TOL.replayed_resume_weight_rtol if replay else 0.0
    return actual.keys() == expected.keys() and all(
        torch.allclose(actual[key].float(), expected[key].float(), rtol=rtol, atol=TOL.weight_atol) for key in expected
    )


def _probe_logits(model, tokenizer, local_rank):
    training = model.training
    model.eval()
    tokens = tokenizer("The river flows through the city.", return_tensors="pt").to(f"cuda:{local_rank}")
    try:
        with torch.no_grad():
            return model(**tokens, use_cache=False).logits.float().cpu()
    finally:
        model.train(training)


def _native_tp_trainer(tp_size, output_dir, tokenizer, train_dataset, eval_dataset=None, *, best=False):
    require_native_runtime()
    torch.manual_seed(SEED)
    pc = ParallelismConfig(tp_size=tp_size)
    model, _ = load_distributed_model(
        model_name_or_path=QWEN3_MODEL,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_liger_kernel=False,
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "o_proj"],
        init_lora_weights=True,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    assert_native_factors(model)
    assert_replicas_equal(model, tp_size)
    config = SFTConfig(
        output_dir=output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        use_liger_kernel=False,
        report_to="none",
        logging_steps=1,
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=MAX_STEPS,
        eval_strategy="steps" if best else "no",
        eval_steps=1 if best else None,
        load_best_model_at_end=best,
        metric_for_best_model="best_probe" if best else None,
        greater_is_better=False if best else None,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        seed=SEED,
        fsdp="",
    )
    return DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=pc,
    )


class _BestAdapterSnapshots(TrainerCallback):
    def __init__(self):
        self.snapshots = {}

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Force an earlier checkpoint to win while retaining the real evaluation and reload paths.
        metrics["eval_best_probe"] = float(state.global_step)

    def on_save(self, args, state, control, model, **kwargs):
        self.snapshots[state.global_step] = snapshot_adapters(model, expert_lora=False)


def _stock_reload_and_merge(adapter_dir, expected_weights, expected_logits, tokenizer, local_rank, merge_dir):
    base = AutoModelForCausalLM.from_pretrained(
        QWEN3_MODEL,
        dtype=torch.bfloat16,
        device_map={"": local_rank},
        attn_implementation="flash_attention_2",
    )
    finalize_loaded_model(base)
    stock = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    assert _adapter_maps_match(snapshot_adapters(stock, expert_lora=False), expected_weights), (
        "stock PEFT missed trained factors"
    )
    torch.testing.assert_close(
        _probe_logits(stock, tokenizer, local_rank), expected_logits, rtol=LOGIT_RTOL, atol=LOGIT_ATOL
    )
    targets = {
        name
        for name, _ in base.named_parameters()
        if name.endswith((".q_proj.base_layer.weight", ".o_proj.base_layer.weight"))
    }
    before_merge = {
        name.replace(".base_layer", ""): param.detach().cpu().clone()
        for name, param in base.named_parameters()
        if name in targets
    }
    # Use the CLI's CPU backend for the strict BF16 merge oracle, after checking CUDA inference.
    stock.cpu()
    merged_oracle = stock.merge_and_unload()
    expected_merged = {
        name: param.detach().cpu().clone() for name, param in merged_oracle.named_parameters() if name in before_merge
    }
    assert expected_merged and any(not torch.equal(value, before_merge[key]) for key, value in expected_merged.items())
    del stock, base, merged_oracle
    cleanup_memory()

    merge_env = dict(os.environ)
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        merge_env.pop(key, None)
    merge_env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(
        [
            sys.executable,
            str(MERGE_SCRIPT),
            "--adapter_dir",
            adapter_dir,
            "--output_dir",
            merge_dir,
            "--device_map",
            "cpu",
            "--dtype",
            "bfloat16",
            "--attn_implementation",
            "eager",
            "--quiet",
        ],
        env=merge_env,
        check=True,
        timeout=600,
    )
    merged = AutoModelForCausalLM.from_pretrained(
        merge_dir, dtype=torch.bfloat16, device_map={"": local_rank}, attn_implementation="flash_attention_2"
    )
    finalize_loaded_model(merged)
    actual_merged = {
        name: param.detach().cpu() for name, param in merged.named_parameters() if name in expected_merged
    }
    assert _adapter_maps_match(actual_merged, expected_merged), "the merge tool did not fold every trained target"
    assert not any("lora_" in name for name, _ in merged.named_parameters())
    torch.testing.assert_close(
        _probe_logits(merged, tokenizer, local_rank), expected_logits, rtol=LOGIT_RTOL, atol=LOGIT_ATOL
    )
    del merged
    cleanup_memory()


def run_tp_lora_lifecycle(ctx, *, tp_size: int) -> bool:
    """Native SFT at one TP group: train, save, replay, best reload, stock load and CLI merge."""
    assert ctx.world_size == tp_size, "native TP lifecycle tests require DP=1"
    rank, local_rank, base_output_dir = ctx.rank, ctx.local_rank, ctx.output_dir
    require_native_runtime()
    ensure_model_downloaded(QWEN3_MODEL, rank)
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)
    output_dir = os.path.join(base_output_dir, "lora_tp_qwen3")
    trainer = _native_tp_trainer(tp_size, output_dir, tokenizer, train_dataset)
    before = snapshot_adapters(trainer.model, expert_lora=False)
    result = trainer.train()
    checks = training_run_checks(result, trainer, MAX_STEPS)
    checks["tp_active"] = trainer.is_tp_mode
    trained = snapshot_adapters(trainer.model, expert_lora=False)
    updated, detail = assert_adapters_moved(before, trained)
    checks["adapters_updated"] = (
        updated
        and before.keys() == trained.keys()
        and all(not torch.equal(before[name], trained[name]) for name in before)
    )
    log(f"  Native TP adapters: {detail}")
    assert_replicas_equal(trainer.model, tp_size)
    expected_logits = _probe_logits(trainer.model, tokenizer, local_rank)
    adapter_dir = os.path.join(output_dir, "final_adapter")
    trainer.save_model(adapter_dir)
    dist.barrier()
    if rank == 0:
        saved = safetensors_load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
        normalized = {name.replace(".default.", "."): value for name, value in trained.items()}
        checks["saved_full_factors"] = os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")) and (
            _adapter_maps_match(saved, normalized)
        )
    resume_dir = os.path.join(output_dir, f"checkpoint-{MAX_STEPS - 1}")
    saved_resume = safetensors_load_file(os.path.join(resume_dir, "adapter_model.safetensors"))
    normalized_before = {name.replace(".default.", "."): value for name, value in before.items()}
    checks["resume_contains_trained_B"] = any(
        "lora_B" in name and not torch.equal(value, normalized_before[name]) for name, value in saved_resume.items()
    )
    del trainer
    cleanup_memory()
    dist.barrier()

    resumed = _native_tp_trainer(tp_size, os.path.join(base_output_dir, "lora_tp_resume"), tokenizer, train_dataset)
    resumed_result = resumed.train(resume_from_checkpoint=resume_dir)
    checks["resumed_steps"] = resumed_result.global_step == MAX_STEPS
    checks["resume_next_step_matches"] = _adapter_maps_match(
        snapshot_adapters(resumed.model, expert_lora=False), trained, replay=True
    )
    assert_replicas_equal(resumed.model, tp_size)
    del resumed
    cleanup_memory()
    dist.barrier()

    if rank == 0:
        try:
            _stock_reload_and_merge(
                adapter_dir,
                trained,
                expected_logits,
                tokenizer,
                local_rank,
                os.path.join(base_output_dir, "lora_tp_merged"),
            )
            checks["stock_load_and_cli_merge"] = True
        except Exception:
            traceback.print_exc()
            checks["stock_load_and_cli_merge"] = False
            cleanup_memory()
    stock_result = torch.tensor(int(checks.get("stock_load_and_cli_merge", True)), device=f"cuda:{local_rank}")
    dist.all_reduce(stock_result, op=dist.ReduceOp.MIN)
    if not stock_result.item():
        return False

    best = _native_tp_trainer(
        tp_size, os.path.join(base_output_dir, "lora_tp_best"), tokenizer, train_dataset, eval_dataset, best=True
    )
    snapshots = _BestAdapterSnapshots()
    best.add_callback(snapshots)
    best.train()
    checks["earlier_best_selected"] = best.state.best_model_checkpoint.endswith("checkpoint-1")
    checks["best_reload_not_vacuous"] = not _adapter_maps_match(snapshots.snapshots[1], snapshots.snapshots[MAX_STEPS])
    checks["best_adapters_restored"] = _adapter_maps_match(
        snapshot_adapters(best.model, expert_lora=False), snapshots.snapshots[1]
    )
    assert_replicas_equal(best.model, tp_size)
    del best
    cleanup_memory()
    result_t = torch.tensor(int(all(checks.values())), device=f"cuda:{local_rank}")
    dist.all_reduce(result_t, op=dist.ReduceOp.MIN)
    return bool(result_t.item())
