"""``gradient_checkpointing_plan: auto``: measure one fully checkpointed step, agree a lighter checkpointing policy
across the world, apply it from the next step on, and keep its record in ``output_dir``, whose unfinished plan
floors the next run's there (the policy math: :mod:`halo.models.checkpointing_plan`)."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

import torch
import torch.utils.checkpoint
import transformers
from transformers import TrainerCallback

from halo.checkpoint.atomic import staged_file
from halo.checkpoint.format import RUN_PLAN_FILE
from halo.distributed.parallelism_config import ParallelismConfig
from halo.hardware import detect_gpu_model
from halo.models.checkpointing_plan import (
    PLAN_OFF_REMEDY,
    RUN_PLAN_VERSION,
    PlanContext,
    RunPlan,
    StepMemory,
    apply_checkpointing_policy,
    candidate_policies,
    fingerprint_mismatches,
    plan_budget_bytes,
    plan_layers,
    recompute_floor_index,
    reject_unplannable_run,
    resolve_policy,
)
from halo.models.gradient_checkpointing import in_backward_pass
from halo.runtime.consensus import (
    DeferredRankFailure,
    barrier_on_exit,
    rank_max,
    rank_min,
    reject_across_ranks,
    reject_divergent_settings,
)
from halo.runtime.filesystem import is_fs_save_rank
from halo.runtime.ranks import get_global_world_size, is_global_main_process

logger = logging.getLogger(__name__)

# The step of this process measured: its first allocates the optimizer moments lazily and runs every JIT
# warm-up, so the first at the steady-state footprint is the second (a resume counts from its own start).
MEASURE_STEP = 2
# Fixed-point scale of a rank's reserved-to-allocated ratio in the int64 world reduction.
RESERVED_RATIO_SCALE = 10**6


@dataclass(frozen=True)
class MemoryReader:
    """The probe's readings: host-side allocator stats on CUDA (no device sync), and once per plan the device
    bytes outside the allocator."""

    allocated: Callable[[], int]
    peak: Callable[[], int]
    reserved_peak: Callable[[], int]
    reset_peak: Callable[[], None]
    non_torch: Callable[[], int]


def _cuda_non_torch_bytes() -> int:
    free, total = torch.cuda.mem_get_info()
    return (total - free) - torch.cuda.memory_reserved()


CUDA_MEMORY = MemoryReader(
    torch.cuda.memory_allocated,
    torch.cuda.max_memory_allocated,
    torch.cuda.max_memory_reserved,
    torch.cuda.reset_peak_memory_stats,
    _cuda_non_torch_bytes,
)


class StepMemoryProbe:
    """One step's :class:`~halo.models.checkpointing_plan.StepMemory`, from hooks on every
    :func:`~halo.models.checkpointing_plan.plan_layers` layer: the peak at each micro-step's first backward
    recompute, every layer recompute's allocation delta and a checkpoint's input size, plus the peak before the
    optimizer step (:meth:`mark_backward_end`). Early stop is off for the step, so a non-reentrant recompute runs to
    its layer's end (same numerics, one step's extra work)."""

    def __init__(self, model, reader: MemoryReader):
        self._reader = reader
        self._head_phase_peak = 0
        self._backward_peak: int | None = None
        self._recompute: list[int] = []
        self._layer_input = 0
        self._start = 0
        self._in_backward_recompute = False
        self._handles = [
            handle
            for layer in plan_layers(model)
            for handle in (layer.register_forward_pre_hook(self._pre), layer.register_forward_hook(self._post))
        ]
        self._early_stop = torch.utils.checkpoint.set_checkpoint_early_stop(False)
        self._early_stop.__enter__()
        reader.reset_peak()

    def mark_backward_end(self) -> None:
        """Take the peak of the step's forwards and backwards, before the optimizer step allocates its own."""
        self._backward_peak = self._reader.peak()

    def close(self) -> StepMemory | None:
        """Remove the hooks; the step's reading, or None when no layer recomputed (nothing checkpointed)."""
        for handle in self._handles:
            handle.remove()
        self._early_stop.__exit__(None, None, None)
        if not self._recompute:
            return None
        peak = self._reader.peak()
        return StepMemory(
            peak=peak,
            head_phase_peak=self._head_phase_peak,
            backward_peak=peak if self._backward_peak is None else self._backward_peak,
            layer_recompute_bytes=max(self._recompute),
            layer_input_bytes=self._layer_input,
            reserved_peak=self._reader.reserved_peak(),
            non_torch_bytes=self._reader.non_torch(),
        )

    def _pre(self, _module, args):
        if not in_backward_pass():
            self._in_backward_recompute = False
            if args and isinstance(args[0], torch.Tensor):
                self._layer_input = args[0].untyped_storage().nbytes()
            return
        if not self._in_backward_recompute:
            self._head_phase_peak = max(self._head_phase_peak, self._reader.peak())
            self._in_backward_recompute = True
        self._start = self._reader.allocated()

    def _post(self, _module, _args, _out):
        if in_backward_pass():
            self._recompute.append(self._reader.allocated() - self._start)


def step_memory_across_ranks(local: StepMemory) -> StepMemory:
    """COLLECTIVE: the conservative world view of every rank's reading, in one all-reduce. The prediction grows
    with the peaks and the recompute bytes and shrinks with the checkpoint input, and the budget shrinks with the
    non-torch bytes and the reserved-to-allocated ratio, so the input takes the world MIN and the rest the world
    MAX. The ratio is each rank's own (every device holds its own reserve), and the world's reserved peak is the
    world peak at the largest one."""
    ratio = -(-local.reserved_peak * RESERVED_RATIO_SCALE // max(local.peak, 1))
    peak, head, backward, recompute, neg_input, world_ratio, non_torch = rank_max(
        [
            local.peak,
            local.head_phase_peak,
            local.backward_peak,
            local.layer_recompute_bytes,
            -local.layer_input_bytes,
            ratio,
            local.non_torch_bytes,
        ]
    )
    reserved = -(-peak * world_ratio // RESERVED_RATIO_SCALE)
    return StepMemory(peak, head, backward, recompute, -neg_input, reserved, non_torch)


def agree_plan(plan: RunPlan) -> None:
    """COLLECTIVE: raise on every rank when any rank resolved a different plan, naming the fields. The inputs are
    world-reduced before the choice, so a divergence is a resolver bug; under FSDP it would hang, since
    checkpointed and plain layers gather at different times."""
    reject_divergent_settings(
        asdict(plan), "The resolved gradient-checkpointing plan", f"{PLAN_OFF_REMEDY} to bypass the plan."
    )


def save_run_plan(directory: str, plan: RunPlan) -> None:
    """COLLECTIVE: publish ``plan`` as ``directory/RUN_PLAN_FILE`` from the FS-aware save rank(s); a write failure
    raises on every rank."""
    with barrier_on_exit(f"Writing {RUN_PLAN_FILE} to {directory}"):
        if is_fs_save_rank():
            os.makedirs(directory, exist_ok=True)
            with staged_file(directory, RUN_PLAN_FILE) as staged:
                staged.write_text(json.dumps(asdict(plan), indent=1, sort_keys=True))


def _read_run_plan(path: str) -> RunPlan | None:
    """The record at ``path``, or None when there is none or another record version wrote it (no floor)."""
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        raw = json.load(handle)
    if raw.get("version") != RUN_PLAN_VERSION:
        if is_global_main_process():
            logger.warning(
                f"{path} has record version {raw.get('version')!r}, not {RUN_PLAN_VERSION}: it sets no floor here."
            )
        return None
    return RunPlan(**raw)


def restore_recompute_floor(output_dir: str, fingerprint: dict, num_layers: int) -> int:
    """COLLECTIVE: the floor index the record a previous run left in ``output_dir`` sets on this run's plan
    (:func:`~halo.models.checkpointing_plan.recompute_floor_index`). Each rank reads its own node's copy and the
    world takes the highest floor any copy sets; an unreadable copy raises on every rank."""
    path = os.path.join(output_dir, RUN_PLAN_FILE)
    guard = DeferredRankFailure(f"Reading {path} (delete it on every node to plan without its floor)", ValueError)
    record = guard.run(lambda: _read_run_plan(path))
    floor = guard.run(lambda: recompute_floor_index(record, fingerprint, num_layers))
    guard.reject()
    world_floor = rank_max([floor])[0]
    if is_global_main_process():
        mismatches = fingerprint_mismatches(record.fingerprint, fingerprint) if record is not None else []
        if world_floor:
            logger.warning(
                f"A run that left {path} ended before it finished training on its checkpointing plan (an OOM, a "
                f"crash or a preemption), so this run checkpoints at least one step more: every_n_layers="
                f"{candidate_policies(num_layers)[world_floor]} or tighter (0 = off, 1 = every layer). Delete the "
                f"file on every node to plan without that floor."
            )
        elif mismatches:
            logger.warning(f"{path} was made for other inputs ({'; '.join(mismatches)}): it sets no floor here.")
    return world_floor


def build_plan_context(
    training_config,
    model,
    parallelism_config: ParallelismConfig,
    trainer_cls: type,
    *,
    packed_rows: bool,
    multimodal_rows: bool,
) -> PlanContext:
    """The refusal facts of this run, ``packed_rows`` and ``multimodal_rows`` as its collator builds them; its
    fingerprint is :func:`build_plan_fingerprint`'s."""
    gc_kwargs = training_config.gradient_checkpointing_kwargs or {}
    return PlanContext(
        num_layers=len(plan_layers(model)),
        gradient_checkpointing=bool(training_config.gradient_checkpointing),
        every_n_layers_configured="every_n_layers" in gc_kwargs,
        offload=bool(gc_kwargs.get("offload", False)),
        pp_size=parallelism_config.pp_size,
        torch_compile=bool(training_config.torch_compile),
        trains_on_generated_rollouts=trainer_cls.SYNCS_TO_EXTERNAL_GENERATOR,
        auto_find_batch_size=bool(training_config.auto_find_batch_size),
        packed_rows=packed_rows,
        multimodal_rows=multimodal_rows,
        ep_group_size=parallelism_config.ep_group_size,
    )


def build_plan_fingerprint(
    training_config, model, parallelism_config: ParallelismConfig, *, num_layers: int, total_device_bytes: int
) -> dict:
    """The run inputs an unfinished plan's record must match to floor this run's: model shape, row shape, parallel
    layout, device, attention backend, and the torch and transformers versions."""
    text_config = model.config.get_text_config()
    return {
        "model_type": text_config.model_type,
        "num_layers": num_layers,
        "hidden_size": text_config.hidden_size,
        "vocab_size": text_config.vocab_size,
        "attn_implementation": model.config._attn_implementation,
        "max_length": getattr(training_config, "max_length", None),
        "per_device_train_batch_size": training_config.per_device_train_batch_size,
        "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
        "world_size": get_global_world_size(),
        "parallel_mode": parallelism_config.mode_string or "standard",
        "ep_size": parallelism_config.ep_size,
        "cp_size": parallelism_config.cp_size,
        "tp_size": parallelism_config.tp_size,
        "expert_tp_size": parallelism_config.expert_tp_size,
        "gpu_model": detect_gpu_model(),
        "total_device_bytes": total_device_bytes,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


class CheckpointingPlanCallback(TrainerCallback):
    """Measure this process's step :data:`MEASURE_STEP` fully checkpointed and checkpoint only what the budget needs
    from the next step on, never below ``floor_index`` (:func:`restore_recompute_floor`).

    The applied plan's record in ``output_dir`` is written unfinished, and train end marks it finished, so the next
    run there knows whether this one survived its plan. Every rank runs the same hooks at the same step, so the
    collectives are ordered.
    """

    def __init__(
        self,
        context: PlanContext,
        *,
        total_device_bytes: int,
        floor_index: int = 0,
        reader: MemoryReader = CUDA_MEMORY,
    ):
        self.context = context
        self.total_device_bytes = total_device_bytes
        self.floor_index = floor_index
        self.reader = reader
        self.plan: RunPlan | None = None
        self._steps_begun = 0
        self._probe: StepMemoryProbe | None = None

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        self._steps_begun += 1
        if self.plan is None and self._steps_begun == MEASURE_STEP:
            self._probe = StepMemoryProbe(model, self.reader)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self._probe is not None:
            self._probe.mark_backward_end()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self._probe is None:
            return
        reading, self._probe = self._probe.close(), None
        # Joined before the reduce: a rank raising alone would strand its peers in it.
        reject_across_ranks(
            None if reading is not None else "no checkpointed layer recomputed in the measured step",
            "The checkpointing plan's measurement",
        )
        world = step_memory_across_ranks(reading)
        budget = plan_budget_bytes(world, self.total_device_bytes, self.context.ep_group_size)
        every_n, predicted = resolve_policy(world, self.context.num_layers, budget, self.floor_index)
        self.plan = RunPlan(every_n_layers=every_n, fingerprint=self.context.fingerprint)
        agree_plan(self.plan)
        apply_checkpointing_policy(model, every_n)
        save_run_plan(args.output_dir, self.plan)
        if is_global_main_process():
            gib = 2**30
            floor = (
                f", floored at every_n_layers={candidate_policies(self.context.num_layers)[self.floor_index]}"
                if self.floor_index
                else ""
            )
            logger.info(
                f"Checkpointing plan at step {state.global_step}: every_n_layers={every_n} (0 = off{floor}), "
                f"predicted peak {predicted / gib:.2f} GiB allocated against a {budget / gib:.2f} GiB budget (fully "
                f"checkpointed: peak {world.peak / gib:.2f} GiB, reserved {world.reserved_peak / gib:.2f} GiB, one "
                f"layer's activations {world.layer_recompute_bytes / gib:.3f} GiB)."
            )

    def on_train_end(self, args, state, control, **kwargs):
        if self.plan is not None:
            save_run_plan(args.output_dir, replace(self.plan, finished=True))


def build_checkpointing_plan_callback(
    training_config,
    model,
    parallelism_config: ParallelismConfig,
    trainer_cls: type,
    *,
    packed_rows: bool,
    multimodal_rows: bool,
) -> CheckpointingPlanCallback:
    """COLLECTIVE: the callback for ``gradient_checkpointing_plan: auto``, floored by the record in ``output_dir``
    (:func:`restore_recompute_floor`) and budgeted on the world's smallest device. Raises ``ValueError`` with every
    reason when the run cannot take a plan, and on every rank when the ranks' fingerprints differ (one plan serves
    every rank)."""
    if not torch.cuda.is_available():
        raise ValueError("gradient_checkpointing_plan: auto measures CUDA allocator readings and needs a GPU.")
    context = build_plan_context(
        training_config,
        model,
        parallelism_config,
        trainer_cls,
        packed_rows=packed_rows,
        multimodal_rows=multimodal_rows,
    )
    reject_unplannable_run(context)
    # The world MIN: one GPU model can report different totals (ECC mode, driver reservations).
    total_device_bytes = rank_min([torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory])[0]
    fingerprint = build_plan_fingerprint(
        training_config,
        model,
        parallelism_config,
        num_layers=context.num_layers,
        total_device_bytes=total_device_bytes,
    )
    context = replace(context, fingerprint=fingerprint)
    reject_divergent_settings(
        fingerprint,
        "The gradient-checkpointing plan's inputs",
        "gradient_checkpointing_plan: auto plans one policy for every rank; set it to off on a mixed cluster.",
    )
    floor_index = restore_recompute_floor(training_config.output_dir, fingerprint, context.num_layers)
    return CheckpointingPlanCallback(context, total_device_bytes=total_device_bytes, floor_index=floor_index)
