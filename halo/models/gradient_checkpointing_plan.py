"""The gradient-checkpointing plan: which text decoder layers a run checkpoints, chosen from one fully checkpointed
step. Policies replay the same forward, so losses and gradients match full checkpointing on deterministic kernels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers.modeling_layers import GradientCheckpointingLayer

from halo.models.module_tree import backbone_with_layers, decoder_layers
from halo.models.unwrapping import base_transformers_model

RUN_PLAN_VERSION = 1
CHECKPOINTING_OFF = 0
# Device share a prediction may fill. Dropless EP routing moves the hot rank's receive count from step to
# step, so an EP group plans with more headroom than its measured step showed.
DENSE_BUDGET_FRACTION = 0.85
EP_BUDGET_FRACTION = 0.75
PLAN_OFF_REMEDY = (
    "Set gradient_checkpointing_plan: off (with gradient_checkpointing_kwargs.every_n_layers for a fixed policy)"
)


@dataclass(frozen=True)
class StepMemory:
    """Bytes. ``head_phase_peak`` is the peak before the first backward recompute, ``backward_peak`` before the
    optimizer step; ``layer_recompute_bytes`` is what one layer's recompute leaves allocated (its uncheckpointed saved
    set) and ``non_torch_bytes`` the memory outside the allocator (CUDA context, DeepEP and NCCL buffers)."""

    peak: int
    head_phase_peak: int
    backward_peak: int
    layer_recompute_bytes: int
    layer_input_bytes: int
    reserved_peak: int
    non_torch_bytes: int


@dataclass(frozen=True)
class PlanContext:
    """The rank-uniform facts the refusal rules read, and the fingerprint a recorded plan is valid for."""

    num_layers: int
    gradient_checkpointing: bool
    every_n_layers_configured: bool
    offload: bool
    pp_size: int
    torch_compile: bool
    trains_on_generated_rollouts: bool
    auto_find_batch_size: bool
    packed_rows: bool
    multimodal_rows: bool
    ep_group_size: int
    fingerprint: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GradientCheckpointingPlan:
    every_n_layers: int
    fingerprint: dict[str, Any]
    finished: bool = False
    version: int = RUN_PLAN_VERSION


def refusal_reasons(ctx: PlanContext) -> list[str]:
    reasons = []
    if not ctx.gradient_checkpointing:
        reasons.append("gradient_checkpointing is off: there is no recompute to relax")
    if ctx.num_layers == 0:
        reasons.append(
            "the text decoder layers are not GradientCheckpointingLayer (the model checkpoints at the model level), "
            "so no per-layer policy applies"
        )
    if ctx.every_n_layers_configured:
        reasons.append("gradient_checkpointing_kwargs sets every_n_layers: an explicit policy is never overridden")
    if ctx.offload:
        reasons.append("activation offload holds the checkpoints on the host, so device readings do not price them")
    if ctx.pp_size > 1:
        reasons.append(
            "pipeline parallelism: a stage holds one activation set per in-flight microbatch, so one step's "
            "reading prices no schedule position, and the stage builder owns its checkpointing"
        )
    if ctx.torch_compile:
        reasons.append("torch_compile: flipping a layer's checkpointing retraces its graph on every rank")
    if ctx.trains_on_generated_rollouts:
        reasons.append(
            "the trainer re-enables checkpointing around each generation (TRL), which resets every layer's flag"
        )
    if ctx.auto_find_batch_size:
        reasons.append(
            "auto_find_batch_size answers an OOM by halving the batch mid-run, under a plan measured for the full one"
        )
    if not ctx.packed_rows:
        reasons.append(
            "the collator does not pack the rows, so micro-batches vary in token count and the measured step need "
            "not be the largest (SFT packs them under packing: true or from a corpus pre-processed packed)"
        )
    if ctx.multimodal_rows:
        reasons.append("multimodal rows: vision activations do not scale with the measured text shape")
    return reasons


def reject_unplannable_run(ctx: PlanContext) -> None:
    reasons = refusal_reasons(ctx)
    if reasons:
        raise ValueError(
            f"gradient_checkpointing_plan: auto is refused for this run: {'; '.join(reasons)}. {PLAN_OFF_REMEDY}, "
            "or change what a reason names."
        )


def checkpointed_layer_count(num_layers: int, every_n_layers: int) -> int:
    """Under transformers' ``index % k == 0`` rule; 0 for off."""
    if every_n_layers == CHECKPOINTING_OFF:
        return 0
    return -(-num_layers // every_n_layers)


def predict_peak(step: StepMemory, num_layers: int, every_n_layers: int) -> int:
    """A layer that stops being checkpointed holds ``layer_recompute_bytes - layer_input_bytes`` more through the
    forward and the backward, and with no layer checkpointed the backward loses its recompute; the optimizer phase,
    after the activations are freed, is the same under every policy."""
    checkpointed = checkpointed_layer_count(num_layers, every_n_layers)
    extra = (num_layers - checkpointed) * (step.layer_recompute_bytes - step.layer_input_bytes)
    head_phase = step.head_phase_peak + extra
    backward = step.backward_peak + extra - (step.layer_recompute_bytes if checkpointed == 0 else 0)
    return max(head_phase, backward, step.peak)


def candidate_policies(num_layers: int) -> list[int]:
    """From least to most recompute. Every step checkpoints more layers than the last, so a restart whose floor rises
    one step (:func:`recompute_floor_index`) always moves activations into recompute."""
    policies, checkpointed = [CHECKPOINTING_OFF], 0
    for k in (8, 4, 3, 2):
        count = checkpointed_layer_count(num_layers, k)
        if k <= num_layers and count > checkpointed:
            policies.append(k)
            checkpointed = count
    return [*policies, 1]


def plan_budget_bytes(step: StepMemory, total_device_bytes: int, ep_group_size: int) -> int:
    """The budget fraction net of the memory outside the allocator, scaled by the measured allocated-to-reserved ratio,
    since the device holds what the allocator reserves, fragmentation included."""
    fraction = EP_BUDGET_FRACTION if ep_group_size > 1 else DENSE_BUDGET_FRACTION
    device_share = int(total_device_bytes * fraction) - step.non_torch_bytes
    return device_share * step.peak // max(step.reserved_peak, step.peak)


def resolve_policy(step: StepMemory, num_layers: int, budget_bytes: int, floor_index: int = 0) -> tuple[int, int]:
    """The least-recompute policy from ``floor_index`` on whose predicted peak fits, with that prediction."""
    for k in candidate_policies(num_layers)[floor_index:]:
        predicted = predict_peak(step, num_layers, k)
        if predicted <= budget_bytes or k == 1:
            return k, predicted
    raise AssertionError("unreachable: full checkpointing is always a candidate")


def fingerprint_mismatches(saved: dict[str, Any], live: dict[str, Any]) -> list[str]:
    return [
        f"{key}: saved={saved.get(key, '<absent>')!r} live={live.get(key, '<absent>')!r}"
        for key in sorted(set(saved) | set(live))
        if saved.get(key) != live.get(key)
    ]


def recompute_floor_index(
    record: GradientCheckpointingPlan | None, fingerprint: dict[str, Any], num_layers: int
) -> int:
    """One step more recompute than a plan the previous run ended before finishing on (an OOM, a crash or a
    preemption), so a plan never replays into the same OOM. 0 without a record for these inputs or a finished run."""
    if record is None or record.finished or fingerprint_mismatches(record.fingerprint, fingerprint):
        return 0
    policies = candidate_policies(num_layers)
    if record.every_n_layers not in policies:
        raise ValueError(
            f"The checkpointing plan record names every_n_layers={record.every_n_layers}, not one of the policies "
            f"{policies} the plan chooses from for {num_layers} layers."
        )
    return min(policies.index(record.every_n_layers) + 1, len(policies) - 1)


def plan_layers(model: torch.nn.Module) -> list[GradientCheckpointingLayer]:
    """The text decoder list, or none when its layers are not all ``GradientCheckpointingLayer``. A multimodal
    checkpoint's vision blocks keep their flags, as a text run never calls them."""
    backbone = backbone_with_layers(base_transformers_model(model))
    layers = list(decoder_layers(backbone)) if backbone is not None else []
    return layers if all(isinstance(layer, GradientCheckpointingLayer) for layer in layers) else []


def apply_checkpointing_policy(model: torch.nn.Module, every_n_layers: int) -> int:
    """Only the flags move, as ``gradient_checkpointing_disable``/``enable`` would replace the installed scoped
    checkpoint function. Returns the count checkpointed."""
    layers = plan_layers(model)
    if not layers:
        raise RuntimeError(
            f"{type(model).__name__} has no text decoder GradientCheckpointingLayer to set a policy on."
        )
    for index, layer in enumerate(layers):
        layer.gradient_checkpointing = every_n_layers != CHECKPOINTING_OFF and index % every_n_layers == 0
    return sum(layer.gradient_checkpointing for layer in layers)
