#!/usr/bin/env python
"""A non-finite optimizer step is skipped on every rank; weights already non-finite stop the run.

Each row trains ``DistributedSFTTrainer`` on one clip and sharding path:

  * ``--mode ep1``: a tiny Qwen3-MoE at ep1 under FSDP2 (DTensor experts), whose EP clip hands its
    coefficient to AdamWBF16 (``defer_grad_scale``) instead of scaling the gradients.
  * ``--mode ep2``: the same model at ep2, whose experts are FSDP-ignored plain tensors.
  * ``--mode ep2_dp2``: ep2 on four ranks, two EP groups: the deferred cross-replica expert sweep.
  * ``--mode ep2_tp2``: ep2 with attention TP on four ranks, two EP groups: the EP clip's deferred sweep and its
    TP-replicated sync together.
  * ``--mode ep2_cp2``: ep2 and Ulysses CP over the same two ranks (padded rows).
  * ``--mode etp2``: pure expert-tensor parallelism, each expert FFN split over both ranks.
  * ``--mode dense``: dense Qwen3-0.6B under FSDP2; also launchable through ``accelerate launch``, whose
    optimizer wrapper the guard steps through.
  * ``--mode tp``: dense Qwen3-0.6B at ``tp_size=2``, whose TP clip reduces DTensor and replicated norms.
  * ``--mode cp``: dense Qwen3-0.6B at ``cp_size=2`` (Ulysses, padded rows).
  * ``--mode qlora``: QLoRA on the dense model, plain DP with the QLoRA gradient sweep.
  * ``--mode pp``: a tiny dense Qwen3 at ``pp_size=2`` on four ranks, two per stage: stage-local parameters,
    the pipeline clip.
  * ``--mode pp_ep``: a tiny GptOss at ``pp_size=2`` x ``ep_size=2`` on four ranks: stage-local EP experts, the
    deferred sweep under the pipeline clip.

The last rank poisons a step by multiplying its final norm's output by NaN, which leaves every forward
collective (DeepEP dispatch included) finite and makes the backward NaN; under PP that rank is the last
stage. AdamWBF16, the bf16 rows' optimizer, skips on the device and the guard reads the verdict a step late:
at the step's own log when it logs, else at the next step's norm. QLoRA's torch AdamW skips on the host. Four
phases per row:

  1. **Skip.** The last rank poisons step ``POISON_STEP``, every step logging. The gradient reductions carry
     the NaN to every rank, so every rank reads a NaN global norm and skips: every parameter shard and
     optimizer moment after the step equals its value before it, and the optimizer's step counts leave it out.
     The later steps train, the windows after the skipped one log finite losses, and the logs carry
     ``nonfinite_steps_skipped``. Where ranks hold copies of a parameter (expert shards across EP groups, plain
     replicated parameters, TP replicas), the copies end the run bitwise equal: each step's sync ran.
  2. **Late verdict.** The same run logging once, over all ``TOTAL_STEPS``: the poisoned step logs nothing, so
     its verdict is read at the next step's norm. The step is skipped as in phase 1, and the window's loss and
     ``train_loss`` are the mean of phase 1's losses over the applied steps.
  3. **Streak.** At ``max_consecutive_nonfinite_steps=1``, two poisoned steps in a row inside one logging
     window raise on every rank, the first skipped and the second refused: on the device path one step late,
     at the next step's norm, and either way before any later update.
  4. **Corrupted weights.** Rank 0 writes NaN into one trainable ``q_proj`` shard before a step. The step's
     norm is NaN, and every rank raises naming that parameter instead of skipping.

``--max-grad-norm 0`` turns clipping off, which leaves transformers' norm-only pass (the installed clip at an
infinite threshold) as the only call that runs the gradient sweeps before the optimizer's own step pre-hooks.
Such a row also trains one step clipping at a threshold no norm reaches, and must match it at step 1: the
gradients the optimizer took and the logged norm. A sweep that runs at neither call, or twice, moves the
gradients; one that runs only after the norm was read moves the norm.

Run:
    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_nonfinite_step_guard.py --mode ep1
    torchrun --nproc_per_node=4 tests/gpu/trainers/sft/test_sft_nonfinite_step_guard.py --mode ep2_dp2
    torchrun --nproc_per_node=4 tests/gpu/trainers/sft/test_sft_nonfinite_step_guard.py --mode ep2_dp2 \\
        --max-grad-norm 0
    torchrun --nproc_per_node=4 tests/gpu/trainers/sft/test_sft_nonfinite_step_guard.py --mode pp
    accelerate launch --config_file launcher-configs/accelerate/fsdp2_gradop_config.yaml --num_processes=2 \\
        tests/gpu/trainers/sft/test_sft_nonfinite_step_guard.py --mode dense
"""

import argparse
import math
import os
from typing import NamedTuple

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer, TrainerCallback, default_data_collator
from trl import SFTConfig

from halo.data.collators.factory import select_data_collator
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.distributed.mesh import MeshDim, mesh_dim_names, tp_group
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.tensor_parallel.plain_params import hand_sliced_suffixes
from halo.model_loading.parallel.loader import load_distributed_model
from halo.optimizers.step_skip import DeviceStepSkipMixin
from halo.runtime.collectives import broadcast_from_rank0
from halo.runtime.consensus import rank_consensus
from halo.runtime.dtensor import to_local
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.adapters import load_peft_model
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import group_max_abs_diff, shared_output_dir
from tests.common.harness import gpu_test_main
from tests.common.memory import finish_phase
from tests.common.models import QWEN3_0_6B, TINY_GPTOSS_CONFIG, TINY_QWEN3_CONFIG
from tests.common.numerics import worst_relative_l2
from tests.common.pinned_params import SEED, load_row_model, tiny_family_checkpoint
from tests.common.pp_equivalence import build_dense_model, build_patched_moe_model
from tests.common.pp_trainers import build_sft_rows
from tests.common.rank_log import log

FAMILY = "qwen3_moe"
TOTAL_STEPS = 5
POISON_STEP = 3
# Two runs of one row agree on a step's logged loss up to reduction order; a divisor that counts the skipped step
# misses it by a fifth.
LOSS_RTOL = 2e-2
BATCH_SIZE = 2
MAX_LENGTH = 128
PP_MAX_LENGTH = 64
LEARNING_RATE = 2e-3
# A clipping threshold no step's norm reaches: the clip path's sweeps, then a coefficient of exactly 1.
NONBINDING_MAX_GRAD_NORM = 1e9
# Two runs of one step differ only where an expert's gradient sums its tokens in DeepEP's arrival order (bf16
# rounding, under one ULP); a skipped sweep, or a non-idempotent leg (the expert SUM, the per-head norm SUM) run
# twice, moves a gradient by its own size.
RERUN_GRAD_REL_TOL = 2e-2
RERUN_GRAD_NORM_REL_TOL = 1e-3


class _Mode(NamedTuple):
    world_size: int
    # "moe": the tiny family checkpoint; "dense" / "qlora": Qwen3-0.6B; "pp": the tiny dense Qwen3; "pp_moe": the
    # tiny GptOss, a pipeline-registered MoE family.
    model: str
    axes: dict


MODES = {
    "ep1": _Mode(2, "moe", {"ep_size": 1}),
    "ep2": _Mode(2, "moe", {"ep_size": 2}),
    "ep2_dp2": _Mode(4, "moe", {"ep_size": 2}),
    "ep2_tp2": _Mode(4, "moe", {"ep_size": 2, "tp_size": 2}),
    # Node-local EP+CP needs ep_group_size == the NVLink domain, so on two ranks EP and CP share them.
    "ep2_cp2": _Mode(2, "moe", {"ep_size": 2, "cp_size": 2}),
    "etp2": _Mode(2, "moe", {"ep_size": 1, "expert_tp_size": 2}),
    "dense": _Mode(2, "dense", {}),
    "tp": _Mode(2, "dense", {"tp_size": 2}),
    "cp": _Mode(2, "dense", {"cp_size": 2}),
    "qlora": _Mode(2, "qlora", {}),
    # A stage needs two ranks and must end on an NVLink-domain boundary: one two-GPU domain per stage.
    "pp": _Mode(4, "pp", {"pp_size": 2, "gpus_per_node": 2, "nvlink_domain_size": 2}),
    "pp_ep": _Mode(4, "pp_moe", {"pp_size": 2, "ep_size": 2, "gpus_per_node": 2, "nvlink_domain_size": 2}),
}

_PARSER = argparse.ArgumentParser(description=__doc__)
_PARSER.add_argument("--mode", choices=MODES, required=True)
_PARSER.add_argument(
    "--max-grad-norm",
    type=float,
    default=1.0,
    help="the run's max_grad_norm (transformers' default 1.0); 0 leaves only the norm-only pass",
)
ARGS = _PARSER.parse_args()
MODE = MODES[ARGS.mode]
NORM_ONLY = ARGS.max_grad_norm <= 0


def _final_norm(model) -> torch.nn.Module:
    """The decoder's final norm, the one module named ``norm`` (the per-layer and per-head norms are not)."""
    (norm,) = [module for name, module in model.named_modules() if name.split(".")[-1] == "norm"]
    return norm


class _PoisonSteps(TrainerCallback):
    """The last rank multiplies its final norm's output by NaN on the optimizer steps in ``steps``."""

    def __init__(self, trainer, steps):
        self.steps = frozenset(steps)
        self.armed = False
        if self.steps and dist.get_rank() == dist.get_world_size() - 1:
            _final_norm(trainer.model).register_forward_hook(self._poison)

    def _poison(self, module, inputs, output):
        return output * float("nan") if self.armed else None

    def on_step_begin(self, args, state, control, **kwargs):
        self.armed = state.global_step + 1 in self.steps

    def on_step_end(self, args, state, control, **kwargs):
        self.armed = False


def _stepped_optimizer(trainer):
    """The optimizer that steps, inside accelerate's wrapper under ``accelerate launch``."""
    return getattr(trainer.optimizer, "optimizer", trainer.optimizer)


def _steps_counted(trainer) -> set[int]:
    return {int(state["step"]) for state in _stepped_optimizer(trainer).state.values() if "step" in state}


def _local_state(trainer) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """This rank's parameter shards and optimizer moments, copied."""
    model = trainer.model
    reshard_fsdp2_modules(model)
    shards = {name: to_local(param.detach()).clone() for name, param in model.named_parameters()}
    moments = [
        to_local(value).clone()
        for state in trainer.optimizer.state.values()
        for key, value in sorted(state.items())
        if key in ("exp_avg", "exp_avg_sq")
    ]
    return shards, moments


class _AroundStep(TrainerCallback):
    """This rank's parameter shards and optimizer moments right before and right after optimizer step ``step``."""

    def __init__(self, trainer, step: int):
        self.trainer, self.step = trainer, step
        self.before = self.after = None

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step + 1 == self.step:
            self.before = _local_state(self.trainer)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            self.after = _local_state(self.trainer)


class _StepGrads(TrainerCallback):
    """This rank's gradients as optimizer step ``step`` took them: after the clip's sweeps and the optimizer's step
    pre-hooks, before ``zero_grad``."""

    def __init__(self, trainer, step: int):
        self.trainer, self.step = trainer, step
        self.grads = None

    def on_optimizer_step(self, args, state, control, **kwargs):
        if state.global_step + 1 == self.step:
            self.grads = {
                name: to_local(param.grad.detach()).clone()
                for name, param in self.trainer.model.named_parameters()
                if param.grad is not None
            }


class _CorruptShard(TrainerCallback):
    """Rank 0 writes NaN into the first element of ``name``'s shard before optimizer step ``step``."""

    def __init__(self, trainer, name: str, step: int):
        self.trainer, self.name, self.step = trainer, name, step

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step + 1 != self.step or dist.get_rank() != 0:
            return
        model = self.trainer.model
        reshard_fsdp2_modules(model)
        with torch.no_grad():
            to_local(dict(model.named_parameters())[self.name]).view(-1)[0] = float("nan")


def _sft_config(
    output_dir: str, *, logging_steps: int, max_grad_norm: float, max_steps: int = TOTAL_STEPS
) -> SFTConfig:
    pipelined = MODE.model in ("pp", "pp_moe")
    return SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=4 if pipelined else BATCH_SIZE,
        # Under PP the accumulation steps become the pipeline's microbatches.
        gradient_accumulation_steps=2 if pipelined else 1,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        max_grad_norm=max_grad_norm,
        max_length=PP_MAX_LENGTH if pipelined else MAX_LENGTH,
        packing=False,
        bf16=True,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        logging_steps=logging_steps,
        save_strategy="no",
        report_to=[],
        logging_nan_inf_filter=False,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        seed=SEED,
    )


def _trainer_inputs(pc: ParallelismConfig, moe_checkpoint: str | None) -> dict:
    """The row's model, tokenizer, dataset and the trainer kwargs that go with them."""
    if MODE.model in ("pp", "pp_moe"):
        moe = MODE.model == "pp_moe"
        return {
            "model": build_patched_moe_model(pc) if moe else build_dense_model(),
            "processing_class": AutoTokenizer.from_pretrained(QWEN3_0_6B),
            "train_dataset": build_sft_rows(
                (TINY_GPTOSS_CONFIG if moe else TINY_QWEN3_CONFIG)["vocab_size"], n_rows=128, seq_len=PP_MAX_LENGTH
            ),
            "data_collator": default_data_collator,
        }
    peft_config = None
    # Ulysses runs FlashAttention whatever the label, so a CP row loads on it.
    attn_implementation = "flash_attention_2" if pc.is_cp_mode else "eager"
    if MODE.model == "moe":
        model, tokenizer, _ = load_row_model(moe_checkpoint, "full", pc, attn_implementation=attn_implementation)
    elif MODE.model == "qlora":
        model, tokenizer, peft_config = load_peft_model(
            "qlora", pc, model_name=QWEN3_0_6B, attn_implementation="eager", use_liger_kernel=False
        )
    else:
        model, tokenizer = load_distributed_model(
            QWEN3_0_6B,
            pc,
            trust_remote_code=False,
            attn_implementation=attn_implementation,
            use_liger_kernel=False,
        )
    inputs = {
        "model": model,
        "processing_class": tokenizer,
        "train_dataset": create_sft_dataset(64, tokenizer, seed=SEED),
        "peft_config": peft_config,
    }
    if pc.is_cp_mode:
        # Ulysses has no per-document boundaries: padded rows whose length divides cp_size.
        inputs["data_collator"] = select_data_collator(
            tokenizer, train_on_completions_only=False, use_context_parallel=True, pad_to_multiple_of=pc.cp_size
        )
    return inputs


def _same(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def _raised(trainer) -> str | None:
    """The RuntimeError ``trainer.train()`` raised on this rank, or None."""
    try:
        trainer.train()
    except RuntimeError as error:
        return str(error)
    return None


def _first_grad_norm(trainer) -> float | None:
    """The gradient norm ``trainer`` logged at its first step, or None."""
    return next((e["grad_norm"] for e in trainer.state.log_history if "grad_norm" in e), None)


def _tp_replicated(param: DTensor) -> bool:
    """Whether every rank of the TP group holds the same local values of ``param``: replicated on its mesh's ``tp``
    dim, on a bare (unnamed) TP mesh, or on a mesh without a ``tp`` dim (FSDP's dp shard of a TP-replicated
    parameter)."""
    names = mesh_dim_names(param.device_mesh)
    if MeshDim.TP in names:
        return param.placements[names.index(MeshDim.TP)].is_replicate()
    if not names:
        return all(placement.is_replicate() for placement in param.placements)
    return True


def _replica_sets(trainer) -> dict[str, tuple[dist.ProcessGroup | None, list[tuple[str, torch.Tensor]]]]:
    """The trainable parameters this rank holds copies of on other ranks, by the group whose ranks hold them alike:
    the distributed expert shards across the EP groups (``experts``), the plain non-expert parameters every rank
    holds whole (``replicated``, the world), and the TP-replicated DTensors across the TP group (``tp``). Read off
    each parameter's layout, not off the sweeps' own classification. A kind whose group does not exist has none."""
    model = trainer._top_level_model()
    layers = [layer for _, layer in named_ep_layers(model)]
    experts = {
        id(param) for layer in layers if layer.ep_config.ep_group_size > 1 for _, param in layer.expert_named_params()
    }
    hand_sliced = hand_sliced_suffixes(model)
    replica_group = layers[0].ep_config.expert_replica_group if layers else None
    group_tp = tp_group(trainer._device_mesh)
    sets = {"experts": (replica_group, []), "replicated": (dist.group.WORLD, []), "tp": (group_tp, [])}
    for name, param in model.named_parameters():
        if not param.requires_grad or (hand_sliced and name.endswith(hand_sliced)):
            continue
        if id(param) in experts:
            kind = "experts"
        elif not isinstance(param, DTensor):
            kind = "replicated"
        elif _tp_replicated(param):
            kind = "tp"
        else:
            continue
        if sets[kind][0] is not None:
            sets[kind][1].append((name, param))
    return sets


def _replica_checks(trainer, pc: ParallelismConfig, metrics: dict) -> dict[str, bool]:
    """COLLECTIVE. Every copy of a parameter is bitwise its peers' after the run, and the copies the layout makes
    were found: the expert shards wherever several EP groups split the world, the TP replicas under TP."""
    reshard_fsdp2_modules(trainer.model)
    differing, counts = [], {}
    for kind, (group, params) in _replica_sets(trainer).items():
        counts[kind] = len(params)
        differing += [
            f"{kind}:{name}" for name, param in params if group_max_abs_diff(to_local(param.detach()), group) != 0.0
        ]
    metrics |= {f"replica_params_{kind}": count for kind, count in counts.items()}
    if differing:
        log(f"  {len(differing)} parameter copies differ from their group's first rank: {differing[:6]}")
    checks = {"replicas_identical_after_training": not differing}
    if pc.ep_size > 1 and pc.num_ep_groups > 1:
        checks["expert_replicas_compared"] = counts["experts"] > 0
    if pc.tp_size > 1:
        checks["tp_replicas_compared"] = counts["replicated"] + counts["tp"] > 0
    return checks


@gpu_test_main(exact_world_size=MODE.world_size, prefix=f"sft_nonfinite_step_guard_{ARGS.mode}")
def run(ctx):
    torch.cuda.set_device(ctx.device)
    moe_checkpoint = tiny_family_checkpoint(ctx, FAMILY) if MODE.model == "moe" else None
    out = os.path.join(shared_output_dir(ctx), "train_out")

    def make_trainer(
        *, poison_steps=(), logging_steps=1, max_grad_norm=ARGS.max_grad_norm, max_steps=TOTAL_STEPS, **guard_kwargs
    ):
        # Seeds every random draw of a row's build (QLoRA's adapters), so two runs of one row take the same steps.
        torch.manual_seed(SEED)
        pc = ParallelismConfig(**MODE.axes)
        trainer = DistributedSFTTrainer(
            args=_sft_config(out, logging_steps=logging_steps, max_grad_norm=max_grad_norm, max_steps=max_steps),
            parallelism_config=pc,
            **_trainer_inputs(pc, moe_checkpoint),
            **guard_kwargs,
        )
        trainer.add_callback(_PoisonSteps(trainer, poison_steps))
        ctx.on_teardown(trainer.cleanup_ep)
        return trainer

    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}

    log(
        f"\n[1/4] the last rank poisons step {POISON_STEP} of {TOTAL_STEPS} at max_grad_norm={ARGS.max_grad_norm}, "
        "every step logging"
    )
    trainer = make_trainer(poison_steps=(POISON_STEP,))
    around = _AroundStep(trainer, POISON_STEP)
    trainer.add_callback(around)
    step_one = _StepGrads(trainer, 1) if NORM_ONLY else None
    if step_one is not None:
        trainer.add_callback(step_one)
    result = trainer.train()
    metrics["final_train_loss"] = result.training_loss
    before, after = around.before, around.after
    final, _ = _local_state(trainer)
    skipped = before is not None and after is not None
    checks["every_rank_skipped_the_poisoned_step"] = rank_consensus(
        skipped and _same(list(before[0].values()), list(after[0].values())) and _same(before[1], after[1])
    ).all_ok
    checks["moments_existed_to_compare"] = rank_consensus(skipped and bool(before[1])).all_ok
    checks["later_steps_trained"] = rank_consensus(
        skipped
        and not _same(list(after[0].values()), list(final.values()))
        and all(torch.isfinite(t).all() for t in final.values())
    ).all_ok
    logs = {e["step"]: e for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e}
    log(f"  logged losses: {[round(logs[s]['loss'], 4) for s in sorted(logs)]}")
    checks["poisoned_step_logged_nan"] = math.isnan(logs.get(POISON_STEP, {}).get("loss", 0.0))
    checks["later_windows_logged_finite_losses"] = sorted(logs) == list(range(1, TOTAL_STEPS + 1)) and all(
        math.isfinite(logs[s]["loss"]) for s in logs if s != POISON_STEP
    )
    checks["skip_counted_in_the_logs"] = [logs[s].get("nonfinite_steps_skipped", 0) for s in sorted(logs)] == [
        0 if s < POISON_STEP else 1 for s in range(1, TOTAL_STEPS + 1)
    ]
    checks["run_mean_loss_finite"] = math.isfinite(result.training_loss)
    device_path = isinstance(_stepped_optimizer(trainer), DeviceStepSkipMixin)
    log(f"  verdict path: {'device' if device_path else 'host'}")
    checks["step_counts_leave_out_the_skipped_step"] = rank_consensus(
        _steps_counted(trainer) == {TOTAL_STEPS - 1}
    ).all_ok
    applied = [logs.get(s, {}).get("loss", math.nan) for s in range(1, TOTAL_STEPS + 1) if s != POISON_STEP]
    applied_mean = sum(applied) / len(applied)
    pc = trainer.parallelism_config
    if pc.pp_size == 1 and ((pc.ep_size > 1 and pc.num_ep_groups > 1) or pc.tp_size > 1):
        checks |= _replica_checks(trainer, pc, metrics)
    finish_phase(trainer)
    ctx.barrier()

    if NORM_ONLY:
        log(f"\n[sync] step 1 clipping at max_grad_norm={NONBINDING_MAX_GRAD_NORM:g}, against the norm-only pass's")
        reference = make_trainer(max_grad_norm=NONBINDING_MAX_GRAD_NORM, max_steps=1)
        reference_step_one = _StepGrads(reference, 1)
        reference.add_callback(reference_step_one)
        reference.train()
        worst, name = worst_relative_l2(step_one.grads or {}, reference_step_one.grads or {})
        log(f"  step-1 gradients: worst relative difference {worst:.3e} ({name})")
        metrics["norm_only_grad_max_rel_diff"] = worst
        checks["norm_only_pass_grads_match_the_clipping_run"] = rank_consensus(worst <= RERUN_GRAD_REL_TOL).all_ok
        norm_only, clipping = _first_grad_norm(trainer), _first_grad_norm(reference)
        norm_rel = abs(norm_only - clipping) / clipping if norm_only is not None and clipping else math.inf
        log(f"  step-1 grad_norm: norm-only {norm_only}, clipping {clipping}")
        metrics["norm_only_grad_norm_rel_diff"] = norm_rel
        checks["norm_only_pass_norm_matches_the_clipping_run"] = rank_consensus(
            norm_rel <= RERUN_GRAD_NORM_REL_TOL
        ).all_ok
        finish_phase(reference)
        ctx.barrier()

    log(f"\n[2/4] the same poisoned step inside one {TOTAL_STEPS}-step logging window")
    trainer = make_trainer(poison_steps=(POISON_STEP,), logging_steps=TOTAL_STEPS)
    around = _AroundStep(trainer, POISON_STEP)
    trainer.add_callback(around)
    late = trainer.train()
    before, after = around.before, around.after
    skipped = before is not None and after is not None
    checks["late_verdict_skipped_the_poisoned_step"] = rank_consensus(
        skipped and _same(list(before[0].values()), list(after[0].values())) and _same(before[1], after[1])
    ).all_ok
    checks["late_verdict_step_counts_leave_out_the_skipped_step"] = rank_consensus(
        _steps_counted(trainer) == {TOTAL_STEPS - 1}
    ).all_ok
    windows = {e["step"]: e for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e}
    window = windows.get(TOTAL_STEPS, {})
    log(f"  window loss {window.get('loss')}, train_loss {late.training_loss}; phase 1 applied mean {applied_mean}")
    checks["late_window_averages_the_applied_steps"] = sorted(windows) == [TOTAL_STEPS] and math.isclose(
        window["loss"], applied_mean, rel_tol=LOSS_RTOL
    )
    checks["late_window_counts_the_skip"] = window.get("nonfinite_steps_skipped") == 1
    checks["late_train_loss_averages_the_applied_steps"] = math.isclose(
        late.training_loss, applied_mean, rel_tol=LOSS_RTOL
    )
    finish_phase(trainer)
    ctx.barrier()

    log("\n[3/4] two poisoned steps in a row at max_consecutive_nonfinite_steps=1, inside one logging window")
    trainer = make_trainer(poison_steps=(2, 3), logging_steps=TOTAL_STEPS, max_consecutive_nonfinite_steps=1)
    first = _AroundStep(trainer, 1)
    trainer.add_callback(first)
    raised = _raised(trainer)
    log(f"  raised at global step {trainer.state.global_step}: {raised}")
    checks["streak_raises_on_every_rank"] = rank_consensus(
        raised is not None and "Optimizer steps 2-3: 2 consecutive steps" in raised
    ).all_ok
    # A device skip's verdict is read at the next step's norm: step 3's raises in step 4, before its update.
    checks["streak_raises_at_its_verdict_read"] = trainer.state.global_step == (3 if device_path else 2)
    shards, moments = _local_state(trainer)
    checks["no_update_after_the_last_applied_step"] = rank_consensus(
        first.after is not None
        and _same(list(first.after[0].values()), list(shards.values()))
        and _same(first.after[1], moments)
    ).all_ok
    finish_phase(trainer)
    ctx.barrier()

    log("\n[4/4] rank 0 corrupts one weight shard before step 2")
    trainer = make_trainer()
    # Rank 0's name: under PP the other stage holds other layers.
    name = broadcast_from_rank0(
        next(n for n, p in trainer.model.named_parameters() if p.requires_grad and "q_proj" in n)
    )
    trainer.add_callback(_CorruptShard(trainer, name, step=2))
    raised = _raised(trainer)
    log(f"  raised: {raised}")
    checks["corrupted_weights_raise_on_every_rank_naming_the_parameter"] = rank_consensus(
        raised is not None and "already hold NaN/inf before the update" in raised and name in raised
    ).all_ok
    finish_phase(trainer)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
