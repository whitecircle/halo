"""The shared body of the fp32-pinned-parameter GPU rows: a pinned family trains in its run dtype.

transformers pins some DeepSeek-V4, GLM-5 Next and Inkling parameters in fp32
(``_keep_in_fp32_modules_strict``), and every training loader casts them to the run dtype. A row loads a
tiny checkpoint of one family through the production path, checks no floating base parameter is off the
run dtype, trains a few steps, and on an adapter row checks the adapters moved and stayed identical on
every rank (an adapter FSDP2 leaves out of its shard groups gets no gradient all-reduce, so its ranks
drift apart while each loss stays finite).
"""

import os
import shutil

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded, shared_scratch_dir, world_any
from tests.common.models import QWEN3_0_6B
from tests.common.peft_helpers import assert_adapters_moved, load_peft_model, snapshot_adapters, unwrap
from tests.common.tiny_models import build_tiny_pinned_checkpoint
from tests.common.utils import log, log_all

RUN_DTYPE = torch.bfloat16
SEED = 42
NUM_STEPS = 2
MAX_LENGTH = 128
# High enough that one step moves a zero-init lora_B past bf16 resolution.
LEARNING_RATE = 2e-3

# The modes a row trains: full fine-tuning, or the load_peft_model adapter modes.
MODES = ("full", "expert_lora", "mixed")


def _off_run_dtype(model: torch.nn.Module) -> list[str]:
    """Floating base parameters not in the run dtype (adapters are the trainer's to cast)."""
    return [
        name
        for name, param in model.named_parameters()
        if param.is_floating_point() and param.dtype != RUN_DTYPE and "lora_" not in name
    ]


def _differing_from_rank0(adapters: dict[str, torch.Tensor], device: torch.device) -> list[str]:
    """Adapters whose value on this rank is not bitwise rank 0's. Collective (one broadcast each)."""
    differing = []
    for name in sorted(adapters):
        local = adapters[name].to(device)
        reference = local.clone()
        dist.broadcast(reference, src=0)
        if not torch.equal(local, reference):
            differing.append(name)
    return differing


def run_pinned_family_row(ctx, family: str, mode: str, pc: ParallelismConfig) -> dict:
    """Load ``family`` under ``pc``, train ``NUM_STEPS`` in ``mode``, and return the harness result."""
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    torch.cuda.set_device(ctx.device)

    ensure_model_downloaded(QWEN3_0_6B, ctx.rank)  # tokenizer only
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    scratch = shared_scratch_dir(f"pinned_{family}")
    base_dir = os.path.join(scratch, "base")
    if ctx.rank == 0:
        shutil.rmtree(scratch, ignore_errors=True)
        ctx.on_teardown(lambda: shutil.rmtree(scratch, ignore_errors=True))
        build_tiny_pinned_checkpoint(family, base_dir, tokenizer=tokenizer, seed=SEED)
    ctx.barrier()

    if mode == "full":
        model, tokenizer = load_distributed_model(
            model_name_or_path=base_dir,
            parallelism_config=pc,
            dtype=RUN_DTYPE,
            attn_implementation="eager",
            use_liger_kernel=False,
        )
        peft_config = None
    else:
        model, tokenizer, peft_config = load_peft_model(
            mode, pc, model_name=base_dir, attn_implementation="eager", use_liger_kernel=False
        )
    off = _off_run_dtype(model)
    log(f"{family} {mode} ep{pc.ep_size}: {len(off)} base params off {RUN_DTYPE} after load {off[:6]}")
    checks["loaded_in_run_dtype"] = not off

    args = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=NUM_STEPS,
        per_device_train_batch_size=2,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        max_length=MAX_LENGTH,
        bf16=True,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        logging_nan_inf_filter=False,
        dataloader_drop_last=True,
        seed=SEED,
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=args,
        train_dataset=create_sft_dataset(16, tokenizer, seed=SEED),
        processing_class=tokenizer,
        parallelism_config=pc,
        peft_config=peft_config,
    )
    ctx.on_teardown(trainer.cleanup_ep)

    adapters_before = snapshot_adapters(unwrap(trainer.model), expert_lora=False) if mode != "full" else {}
    result = trainer.train()
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    log(f"{family} {mode} ep{pc.ep_size} losses: {losses}")
    metrics["final_train_loss"] = result.training_loss
    checks["trained_all_steps"] = result.global_step == NUM_STEPS
    checks["losses_finite"] = bool(losses) and bool(torch.isfinite(torch.tensor(losses)).all())

    if mode != "full":
        adapters_after = snapshot_adapters(unwrap(trainer.model), expert_lora=False)
        moved, detail = assert_adapters_moved(adapters_before, adapters_after)
        log(f"adapters: {detail}")
        checks["adapters_moved"] = moved
        differing = _differing_from_rank0(adapters_after, ctx.device)
        if differing:
            log_all(f"{len(differing)} adapters differ from rank 0: {differing[:6]}")
        metrics["adapters_differing_from_rank0"] = len(differing)
        checks["adapters_identical_across_ranks"] = not world_any(bool(differing))
    return {"checks": checks, "metrics": metrics}
