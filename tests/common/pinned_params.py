"""Shared GPU row bodies for the precision and gradient-sync suites over tiny MoE checkpoints, and the
stored-fp32 pin oracle they share with the CPU loader tests.

transformers pins some DeepSeek-V4, GLM-5 Next and Inkling parameters in fp32
(``_keep_in_fp32_modules_strict``), and every training loader casts them to the run dtype, or keeps the
stored fp32 values when the run holds fp32 masters (:func:`stored_fp32_pins`, :func:`pins_off_stored`).
:func:`run_pinned_family_row` loads a tiny checkpoint of one family through the production path, checks
the resulting parameter dtypes, and trains a few SFT steps. The helpers below it
(:func:`tiny_family_checkpoint`, :func:`load_row_model`, :func:`train_row`) are shared with the
FSDP2-exclusion suite, whose rows set up their own trainable set. Every row that trains a strict subset of
the model checks the trainable parameters moved and ended bitwise identical on every rank, bar the expert
shards EP distributes: one FSDP2 leaves out of its shard groups with no other sync trains on its own
rank's batch and drifts while each loss stays finite.
"""

from collections.abc import Mapping

import torch
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.expert_parallel.base_layer import find_ep_layers
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import group_max_abs_diff
from tests.common.models import QWEN3_0_6B
from tests.common.peft_helpers import assert_adapters_moved, load_peft_model, unwrap
from tests.common.tiny_models import TINY_MOE_FAMILIES, shared_tiny_family_checkpoint
from tests.common.utils import (
    log,
    params_off_dtype,
    snapshot_trainable,
    step_losses,
    training_run_checks,
)

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


def stored_fp32_pins(family: str, path: str) -> tuple[frozenset[str], dict[str, torch.Tensor]]:
    """The parameters a run-dtype ``from_pretrained`` of ``family``'s checkpoint at ``path`` leaves fp32 (the
    pins as transformers applies them), and every parameter's stored value, read at fp32.

    Raises unless a run-dtype round trip changes some pin's stored value (no pin at all included): a
    comparison against values the run dtype holds exactly passes a loader that round-trips them.
    """
    load_class = TINY_MOE_FAMILIES[family].load_class
    pinned = frozenset(params_off_dtype(load_class.from_pretrained(path, dtype=RUN_DTYPE), RUN_DTYPE))
    stored = dict(load_class.from_pretrained(path, dtype=torch.float32).named_parameters())
    if all(torch.equal(stored[name], stored[name].to(RUN_DTYPE).float()) for name in pinned):
        raise AssertionError(
            f"premise: none of {family}'s {len(pinned)} pins at {path} loses its stored value to a {RUN_DTYPE} "
            f"round trip, so a check against them passes a loader that round-trips them"
        )
    return pinned, stored


def pins_off_stored(model: torch.nn.Module, stored: Mapping[str, torch.Tensor], pins: Mapping[str, str]) -> list[str]:
    """The ``pins`` (``model`` name -> checkpoint name) ``model`` does not hold at fp32, bitwise ``stored``."""
    params = dict(model.named_parameters())
    return sorted(
        name
        for name, key in pins.items()
        if name not in params
        or params[name].dtype != torch.float32
        or not torch.equal(params[name].detach().cpu(), stored[key])
    )


def _rank_local_param_names(model: torch.nn.Module) -> frozenset[str]:
    """The parameters each rank holds its own slice of, which legitimately differ across ranks: the expert
    weights and expert adapters of every EP layer whose experts are distributed (``ep_group_size > 1``)."""
    local = {
        id(param)
        for _, layer in find_ep_layers(model)
        if layer.ep_config.ep_group_size > 1
        for _, param in layer.expert_named_params()
    }
    return frozenset(name for name, param in model.named_parameters() if id(param) in local)


def train_row(ctx, model, tokenizer, pc: ParallelismConfig, peft_config, *, check_sync: bool) -> dict:
    """Train ``NUM_STEPS`` SFT steps and return ``{"checks", "metrics"}``.

    ``check_sync`` adds the moved check over every trainable parameter, and the cross-rank-identical one
    over every trainable parameter but the rank-local expert shards, where any remain.
    """
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

    before = snapshot_trainable(trainer.model) if check_sync else {}
    result = trainer.train()
    checks = training_run_checks(result, trainer, NUM_STEPS)
    checks["logged_every_step"] = len(step_losses(trainer)) == NUM_STEPS
    metrics = {"final_train_loss": result.training_loss}
    if not check_sync:
        return {"checks": checks, "metrics": metrics}

    after = snapshot_trainable(trainer.model)
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
    rank_local = _rank_local_param_names(unwrap(trainer.model))
    replicated = {name: tensor for name, tensor in after.items() if name not in rank_local}
    log(f"{len(replicated)} trainable params compared across ranks, {len(after) - len(replicated)} rank-local")
    if replicated:
        # A world-wide verdict per tensor, so every rank holds the same list.
        differing = [name for name in sorted(replicated) if group_max_abs_diff(replicated[name].to(ctx.device)) != 0.0]
        if differing:
            log(f"{len(differing)} trainable params differ from rank 0: {differing[:6]}")
        metrics["trainable_params_differing_from_rank0"] = len(differing)
        checks["trainable_params_identical_across_ranks"] = not differing
    return {"checks": checks, "metrics": metrics}


def run_pinned_family_row(ctx, family: str, mode: str, pc: ParallelismConfig) -> dict:
    """Load ``family`` under ``pc``, check its parameter dtypes, and train ``NUM_STEPS`` in ``mode``."""
    torch.cuda.set_device(ctx.device)
    base_dir = tiny_family_checkpoint(ctx, family, fp32_pins=pc.fp32_non_ep_params)
    model, tokenizer, peft_config = load_row_model(base_dir, mode, pc)

    checks: dict[str, bool] = {}
    if pc.fp32_non_ep_params:
        # Other fp32 parameters are the run's own (fp32_non_ep_params forces an fp32 router).
        pinned, stored = stored_fp32_pins(family, base_dir)
        off = pins_off_stored(model, stored, {name: name for name in pinned})
        log(f"{family} {mode} ep{pc.ep_size} fp32 masters: {len(pinned)} pins, off the stored fp32 value: {off[:6]}")
        checks["pins_hold_stored_fp32"] = not off
    else:
        off = base_params_off_run_dtype(model)
        log(f"{family} {mode} ep{pc.ep_size}: {len(off)} base params off {RUN_DTYPE} after load {off[:6]}")
        checks["loaded_in_run_dtype"] = not off

    result = train_row(ctx, model, tokenizer, pc, peft_config, check_sync=mode != "full")
    result["checks"] = {**checks, **result["checks"]}
    return result
