"""Shared GPU row bodies for the precision and gradient-sync suites over tiny MoE checkpoints.

transformers pins some DeepSeek-V4, GLM-5 Next and Inkling parameters in fp32
(``_keep_in_fp32_modules_strict``), and every training loader casts them to the run dtype, or keeps the
stored fp32 values when the run holds fp32 masters. :func:`run_pinned_family_row` loads a tiny checkpoint
of one family through the production path, checks the resulting parameter dtypes, and trains a few SFT
steps. The helpers below it (:func:`tiny_family_checkpoint`, :func:`load_row_model`, :func:`train_row`)
are shared with the FSDP2-exclusion suite, whose rows set up their own trainable set. Every row that
trains a strict subset of the model checks the trainable parameters moved and ended bitwise identical on
every rank: one FSDP2 leaves out of its shard groups with no other sync trains on its own rank's batch and
drifts while each loss stays finite.
"""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.models.structure import fp32_pinned_param_names
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import world_any
from tests.common.models import QWEN3_0_6B
from tests.common.peft_helpers import assert_adapters_moved, load_peft_model, unwrap
from tests.common.tiny_models import TINY_MOE_FAMILIES, shared_tiny_family_checkpoint
from tests.common.utils import log, log_all, params_off_dtype

RUN_DTYPE = torch.bfloat16
SEED = 42
NUM_STEPS = 2
MAX_LENGTH = 128
# High enough that one step moves a zero-init lora_B past bf16 resolution.
LEARNING_RATE = 2e-3

# The modes a row trains: full fine-tuning, or the load_peft_model adapter modes.
MODES = ("full", "expert_lora", "mixed")


def tiny_family_checkpoint(ctx, family: str, *, fp32_pins: bool = False) -> str:
    """``family``'s tiny checkpoint at the Qwen3 tokenizer's vocab, built by rank 0 for every rank. Collective."""
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    return shared_tiny_family_checkpoint(
        ctx, TINY_MOE_FAMILIES[family], f"tiny_{family}", tokenizer, SEED, fp32_pins=fp32_pins
    )


def load_row_model(base_dir: str, mode: str, pc: ParallelismConfig):
    """``(model, tokenizer, peft_config)`` through the production loader for ``mode``."""
    if mode == "full":
        model, tokenizer = load_distributed_model(
            model_name_or_path=base_dir,
            parallelism_config=pc,
            dtype=RUN_DTYPE,
            attn_implementation="eager",
            use_liger_kernel=False,
        )
        return model, tokenizer, None
    return load_peft_model(mode, pc, model_name=base_dir, attn_implementation="eager", use_liger_kernel=False)


def base_params_off_run_dtype(model: torch.nn.Module) -> list[str]:
    """Floating base parameters not in the run dtype (adapters are the trainer's to cast)."""
    return [name for name in params_off_dtype(model, RUN_DTYPE) if "lora_" not in name]


def _snapshot_trainable(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Full CPU copies of every trainable parameter. Collective when any is a DTensor."""
    snapshot = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            full = param.full_tensor() if isinstance(param, DTensor) else param
            snapshot[name] = full.detach().float().cpu()
    return snapshot


def _differing_from_rank0(tensors: dict[str, torch.Tensor], device: torch.device) -> list[str]:
    """Names whose value on this rank is not bitwise rank 0's. Collective (one broadcast each)."""
    differing = []
    for name in sorted(tensors):
        local = tensors[name].to(device)
        reference = local.clone()
        dist.broadcast(reference, src=0)
        if not torch.equal(local, reference):
            differing.append(name)
    return differing


def train_row(ctx, model, tokenizer, pc: ParallelismConfig, peft_config, *, check_sync: bool) -> dict:
    """Train ``NUM_STEPS`` SFT steps and return ``{"checks", "metrics"}``.

    ``check_sync`` adds the moved and cross-rank-identical checks over every trainable parameter.
    """
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
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

    before = _snapshot_trainable(unwrap(trainer.model)) if check_sync else {}
    result = trainer.train()
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    log(f"losses: {losses}")
    metrics["final_train_loss"] = result.training_loss
    checks["trained_all_steps"] = result.global_step == NUM_STEPS
    checks["losses_finite"] = bool(losses) and bool(torch.isfinite(torch.tensor(losses)).all())
    if not check_sync:
        return {"checks": checks, "metrics": metrics}

    after = _snapshot_trainable(unwrap(trainer.model))
    moved = [name for name in before if not torch.equal(before[name], after[name])]
    log(f"{len(moved)}/{len(before)} trainable params moved")
    checks["trainable_params_moved"] = bool(moved) and all(torch.isfinite(t).all() for t in after.values())
    adapters = [name for name in before if "lora_" in name]
    if adapters:
        adapters_moved, detail = assert_adapters_moved(
            {name: before[name] for name in adapters}, {name: after[name] for name in adapters}
        )
        log(f"adapters: {detail}")
        checks["adapters_moved"] = adapters_moved
    differing = _differing_from_rank0(after, ctx.device)
    if differing:
        log_all(f"{len(differing)} trainable params differ from rank 0: {differing[:6]}")
    metrics["trainable_params_differing_from_rank0"] = len(differing)
    checks["trainable_params_identical_across_ranks"] = not world_any(bool(differing))
    return {"checks": checks, "metrics": metrics}


def _pins_hold_stored_fp32(model: torch.nn.Module, family: str, base_dir: str) -> tuple[bool, str]:
    """Every pinned parameter is fp32 and bitwise the checkpoint's stored value, which a bf16 round trip
    would change (the premise, checked too)."""
    load_class = TINY_MOE_FAMILIES[family].load_class
    reference = dict(load_class.from_pretrained(base_dir, dtype=torch.float32).named_parameters())
    pinned = sorted(fp32_pinned_param_names(model))
    lossy = [name for name in pinned if not torch.equal(reference[name], reference[name].bfloat16().float())]
    if not lossy:
        return False, "no pinned value is lost to a bf16 round trip; the check would be vacuous"
    wrong = [
        name
        for name, param in model.named_parameters()
        if name in pinned and (param.dtype != torch.float32 or not torch.equal(param.detach().cpu(), reference[name]))
    ]
    return not wrong, f"{len(pinned)} pins, {len(lossy)} bf16-lossy, off the stored fp32 value: {wrong[:6]}"


def run_pinned_family_row(ctx, family: str, mode: str, pc: ParallelismConfig) -> dict:
    """Load ``family`` under ``pc``, check its parameter dtypes, and train ``NUM_STEPS`` in ``mode``."""
    torch.cuda.set_device(ctx.device)
    base_dir = tiny_family_checkpoint(ctx, family, fp32_pins=pc.fp32_non_ep_params)
    model, tokenizer, peft_config = load_row_model(base_dir, mode, pc)

    checks: dict[str, bool] = {}
    if pc.fp32_non_ep_params:
        # Other fp32 parameters are the run's own (fp32_non_ep_params forces an fp32 router).
        checks["pins_hold_stored_fp32"], detail = _pins_hold_stored_fp32(model, family, base_dir)
        log(f"{family} {mode} ep{pc.ep_size} fp32 masters: {detail}")
    else:
        off = base_params_off_run_dtype(model)
        log(f"{family} {mode} ep{pc.ep_size}: {len(off)} base params off {RUN_DTYPE} after load {off[:6]}")
        checks["loaded_in_run_dtype"] = not off

    result = train_row(ctx, model, tokenizer, pc, peft_config, check_sync=mode != "full")
    result["checks"] = {**checks, **result["checks"]}
    return result
