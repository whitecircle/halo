"""The gradient-checkpointing plan: which text decoder layers a run checkpoints, chosen from one fully checkpointed
step.

The policies are transformers' ``every_n_layers=k`` rule (every k-th decoder layer checkpointed) and off; they replay
the same forward, so losses and gradients match full checkpointing bitwise where the kernels are deterministic. Pure
over rank-uniform inputs; the measurement, the collectives and the record a run leaves in ``output_dir`` are
:mod:`halo.callbacks.checkpointing_plan`'s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers.modeling_layers import GradientCheckpointingLayer

from halo.models.module_tree import backbone_with_layers, decoder_layers
from halo.models.unwrapping import base_transformers_model

RUN_PLAN_VERSION = 1
# The every_n_layers spelling of "no layer checkpointed".
CHECKPOINTING_OFF = 0
# Device share a prediction may fill. Dropless EP routing moves the hot rank's receive count from step to
# step, so an EP group plans with more headroom than its measured step showed.
DENSE_BUDGET_FRACTION = 0.85
EP_BUDGET_FRACTION = 0.75


@dataclass(frozen=True)
class StepMemory:
    """One fully checkpointed step's allocator readings, in bytes.

    ``head_phase_peak`` is the peak before the step's first backward recompute (forward plus head backward),
    ``backward_peak`` the peak before the optimizer step, ``peak`` the whole step's; ``layer_recompute_bytes``
    is what one layer's recompute leaves allocated (its uncheckpointed saved set) and ``layer_input_bytes`` a
    checkpoint's held input. ``non_torch_bytes`` is the device memory outside the allocator (the CUDA context,
    DeepEP and NCCL buffers).
    """

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
class RunPlan:
    """The policy a run applied, the inputs it was measured for, and whether that run finished training on it."""

    every_n_layers: int
    fingerprint: dict[str, Any]
    finished: bool = False
    version: int = RUN_PLAN_VERSION


def refusal_reasons(ctx: PlanContext) -> list[str]:
    """Why this run cannot relax its checkpointing; empty when it can."""
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
    """Raise ``ValueError`` with every :func:`refusal_reasons` entry when the run cannot take a plan."""
    reasons = refusal_reasons(ctx)
    if reasons:
        raise ValueError(f"gradient_checkpointing_plan: auto is refused for this run: {'; '.join(reasons)}.")


def checkpointed_layer_count(num_layers: int, every_n_layers: int) -> int:
    """Layers transformers checkpoints under ``every_n_layers`` (``index % k == 0``); 0 for off."""
    if every_n_layers == CHECKPOINTING_OFF:
        return 0
    return -(-num_layers // every_n_layers)


def predict_peak(step: StepMemory, num_layers: int, every_n_layers: int) -> int:
    """Predicted step peak under ``every_n_layers`` from a fully checkpointed ``step`` at the same shape.

    A layer that stops being checkpointed holds ``layer_recompute_bytes - layer_input_bytes`` more through the
    forward and the backward, and with no layer checkpointed the backward loses its recompute; the optimizer
    phase, after the activations are freed, is the same under every policy.
    """
    checkpointed = checkpointed_layer_count(num_layers, every_n_layers)
    extra = (num_layers - checkpointed) * (step.layer_recompute_bytes - step.layer_input_bytes)
    head_phase = step.head_phase_peak + extra
    backward = step.backward_peak + extra - (step.layer_recompute_bytes if checkpointed == 0 else 0)
    return max(head_phase, backward, step.peak)


def candidate_policies(num_layers: int) -> list[int]:
    """Policies from least to most recompute: off, every_n_layers 8, 4, 3 and 2 where each checkpoints more layers
    than the one before, then full. The first past off checkpoints at least an eighth of the layers, and every step
    more than the last, so a restart whose floor rises one step (:func:`recompute_floor_index`) always moves
    activations into recompute."""
    policies, checkpointed = [CHECKPOINTING_OFF], 0
    for k in (8, 4, 3, 2):
        count = checkpointed_layer_count(num_layers, k)
        if k <= num_layers and count > checkpointed:
            policies.append(k)
            checkpointed = count
    return [*policies, 1]


def plan_budget_bytes(step: StepMemory, total_device_bytes: int, ep_group_size: int) -> int:
    """The allocated bytes a policy's prediction may reach on a device of ``total_device_bytes``: the budget
    fraction net of the memory outside the allocator, scaled by the measured step's allocated-to-reserved
    ratio, since the device holds what the allocator reserves (fragmentation included), not what it hands out."""
    fraction = EP_BUDGET_FRACTION if ep_group_size > 1 else DENSE_BUDGET_FRACTION
    device_share = int(total_device_bytes * fraction) - step.non_torch_bytes
    return device_share * step.peak // max(step.reserved_peak, step.peak)


def resolve_policy(step: StepMemory, num_layers: int, budget_bytes: int, floor_index: int = 0) -> tuple[int, int]:
    """The least-recompute policy from :func:`candidate_policies` index ``floor_index`` on whose predicted peak fits
    ``budget_bytes``, with that prediction; full checkpointing (1) when nothing lighter fits."""
    for k in candidate_policies(num_layers)[floor_index:]:
        predicted = predict_peak(step, num_layers, k)
        if predicted <= budget_bytes or k == 1:
            return k, predicted
    raise AssertionError("unreachable: full checkpointing is always a candidate")


def fingerprint_mismatches(saved: dict[str, Any], live: dict[str, Any]) -> list[str]:
    """Fingerprint keys whose saved and live values differ, a key either side lacks included."""
    return [
        f"{key}: saved={saved.get(key, '<absent>')!r} live={live.get(key, '<absent>')!r}"
        for key in sorted(set(saved) | set(live))
        if saved.get(key) != live.get(key)
    ]


def recompute_floor_index(record: RunPlan | None, fingerprint: dict[str, Any], num_layers: int) -> int:
    """The :func:`candidate_policies` index a run's measured plan may not go below, from the record a previous run in
    its ``output_dir`` left: one step more recompute than a plan that run ended before it finished training on (an
    OOM, a crash or a preemption: a plan must not replay into the same OOM on every restart), full checkpointing
    staying full; 0 (no floor) without a record made for these inputs, or when its run finished."""
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
    """The layers a plan counts, measures and sets, in order: the text decoder list under every wrapper, or none
    when its layers are not all ``GradientCheckpointingLayer``. A multimodal checkpoint's vision blocks checkpoint
    too and come first in ``modules()``, but a text run never calls them, so they keep their flags."""
    backbone = backbone_with_layers(base_transformers_model(model))
    layers = list(decoder_layers(backbone)) if backbone is not None else []
    return layers if all(isinstance(layer, GradientCheckpointingLayer) for layer in layers) else []


def apply_checkpointing_policy(model: torch.nn.Module, every_n_layers: int) -> int:
    """Set each :func:`plan_layers` flag for ``every_n_layers``; returns the count checkpointed.

    Only the flags move: the installed checkpoint function (Halo's scoped one, reentrant under EP/CP/MoE)
    stays, which ``gradient_checkpointing_disable``/``enable`` would replace. The ``index % k == 0`` rule is
    transformers' own (``PreTrainedModel._set_gradient_checkpointing``), indexed over the text decoder list.
    """
    layers = plan_layers(model)
    if not layers:
        raise RuntimeError(
            f"{type(model).__name__} has no text decoder GradientCheckpointingLayer to set a policy on."
        )
    for index, layer in enumerate(layers):
        layer.gradient_checkpointing = every_n_layers != CHECKPOINTING_OFF and index % every_n_layers == 0
    return sum(layer.gradient_checkpointing for layer in layers)
