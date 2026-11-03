"""The routed half of a first-step comparison: what a run's EP layers computed, vs a stock reference routed on the
run's own picks. On a tiny MoE the routed output reaches the loss too weakly to show a combine or expert-TP fault, so
it is compared where it is computed. Pinning the picks stops a bf16 near-tie moving tokens between experts."""

import collections
import contextlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import TrainerCallback
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssTopKRouter
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter

from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from tests.common.distributed import world_all
from tests.common.ep_reference import (
    armed_routing_capture,
    duplicate_pick_tokens,
    full_grad,
    score_ep_grad_pairs,
    top_k_shortfall,
    zero_filled_expert_banks,
)
from tests.common.first_step import ROUTING_DEPENDENT_PARAM
from tests.common.numerics import max_or_nan, min_or_nan, worst_relative_l2
from tests.common.rank_log import log, log_all
from tests.common.tolerances import TOL

# ``(router, logits, picks) -> weights``.
GateWeights = Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class RoutedStep:
    """What a run's EP layers did over the forwards recorded, on the host, by layer path in module order."""

    layer_classes: dict[str, type] = field(default_factory=dict)
    # Per forward: the ``[tokens, top_k]`` global expert ids dispatched on, and the output.
    picks: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    outputs: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    # Under their stock names, and each layer's expert banks by stock container key.
    router_grads: dict[str, torch.Tensor] = field(default_factory=dict)
    expert_grads: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)

    def duplicate_pick_tokens(self) -> int:
        """Tokens whose picks name one expert twice: a reference pinned on them routes alike, so only this count shows
        them."""
        return sum(duplicate_pick_tokens(picks) for chunks in self.picks.values() for picks in chunks)


@dataclass
class PinnedReference:
    outputs: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    # Names the reference's expert gradients.
    expert_paths: dict[str, str] = field(default_factory=dict)
    # Per router call: the largest shortfall of a pinned pick below the stock router's k-th logit.
    regrets: list[float] = field(default_factory=list)
    # Left unconsumed per layer: a reference that ran fewer forwards.
    unconsumed: dict[str, int] = field(default_factory=dict)


def _host(output) -> torch.Tensor:
    tensor = output[0] if isinstance(output, tuple) else output
    return tensor.detach().cpu()


@contextlib.contextmanager
def capturing_routed_forwards(model: torch.nn.Module, routed: RoutedStep):
    layers = named_ep_layers(model)
    if not layers:
        raise AssertionError(f"{type(model).__name__} holds no EP layer to record")
    picks = [routed.picks.setdefault(path, []) for path, _ in layers]
    with armed_routing_capture(layers, picks, torch.Tensor.cpu):
        handles = []
        for path, layer in layers:
            routed.layer_classes[path] = type(layer)
            sink = routed.outputs.setdefault(path, [])
            handles.append(
                layer.register_forward_hook(lambda module, args, output, sink=sink: sink.append(_host(output)))
            )
        try:
            yield routed
        finally:
            for handle in handles:
                handle.remove()


def gather_routed_grads(model: torch.nn.Module, routed: RoutedStep) -> None:
    """COLLECTIVE (expert gathers, a sharded router's ``full_tensor``): each EP layer's router and expert gradients
    into ``routed``, whole on the host."""
    for path, layer in named_ep_layers(model):
        for name, param in layer._live_router_module().named_parameters():
            routed.router_grads[f"{path}.{layer._ROUTER_ATTR}.{name}"] = full_grad(param).cpu()
        routed.expert_grads[path] = {key: bank.float().cpu() for key, bank in zero_filled_expert_banks(layer).items()}


class _RoutedGradRecorder(TrainerCallback):
    """Gathers the routed gradients at the first optimizer step, after the clip seam's EP sync."""

    def __init__(self, model: torch.nn.Module, routed: RoutedStep):
        self._model = model
        self._routed = routed

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if state.global_step == 0:
            gather_routed_grads(self._model, self._routed)


def record_routed_first_step(trainer) -> RoutedStep:
    """Arm ``trainer``'s EP layers for its first optimizer step and return the :class:`RoutedStep` it fills. Call
    before ``trainer.train()``."""
    routed = RoutedStep()
    microbatches = trainer.args.gradient_accumulation_steps
    compute_loss = trainer.compute_loss
    recorded = 0

    def recording_compute_loss(model, inputs, *args, **kwargs):
        nonlocal recorded
        if not (model.training and recorded < microbatches):
            return compute_loss(model, inputs, *args, **kwargs)
        recorded += 1
        with capturing_routed_forwards(trainer.model, routed):
            return compute_loss(model, inputs, *args, **kwargs)

    trainer.compute_loss = recording_compute_loss
    trainer.add_callback(_RoutedGradRecorder(trainer.model, routed))
    return routed


def _qwen3_moe_gate_weights(router: torch.nn.Module, logits: torch.Tensor, picks: torch.Tensor) -> torch.Tensor:
    """``Qwen3MoeTopKRouter``'s gate weights at ``picks``: fp32 softmax over every expert, renormalized under
    ``norm_topk_prob``."""
    weights = F.softmax(logits, dtype=torch.float, dim=-1).gather(-1, picks)
    if router.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.to(logits.dtype)


def _gpt_oss_gate_weights(router: torch.nn.Module, logits: torch.Tensor, picks: torch.Tensor) -> torch.Tensor:
    """``GptOssTopKRouter``'s gate weights at ``picks``: the softmax over the selected logits."""
    return F.softmax(logits.gather(-1, picks), dim=-1, dtype=logits.dtype)


PINNED_GATE_WEIGHTS: dict[type, GateWeights] = {
    Qwen3MoeTopKRouter: _qwen3_moe_gate_weights,
    GptOssTopKRouter: _gpt_oss_gate_weights,
}


def _gate_weights_for(path: str, router: torch.nn.Module | None) -> GateWeights:
    """Matched by ``isinstance`` so an FSDP2-wrapped router keeps its gate."""
    for router_cls, gate_weights in PINNED_GATE_WEIGHTS.items():
        if isinstance(router, router_cls):
            return gate_weights
    raise NotImplementedError(f"{path}: no pinned gate for the stock router {type(router).__name__}")


def _pin(path: str, queue: collections.deque, gate_weights: GateWeights, regrets: list[float]):
    def hook(module, args, output):
        if not queue:
            raise RuntimeError(f"{path}: the reference ran more forwards than the run recorded")
        logits = output[0]
        picks = queue.popleft().to(device=logits.device, dtype=torch.long)
        if picks.shape[0] != logits.shape[0]:
            raise RuntimeError(f"{path}: {picks.shape[0]} recorded picks for {logits.shape[0]} reference tokens")
        with torch.no_grad():
            regrets.append(top_k_shortfall(logits.detach().float(), picks).max().item())
        return logits, gate_weights(module, logits, picks), picks

    return hook


@contextlib.contextmanager
def pinned_reference_routing(model: torch.nn.Module, routed: RoutedStep):
    """Inside, stock MoE blocks route each forward on the next recorded picks; yields the :class:`PinnedReference` they
    fill."""
    pinned = PinnedReference()
    queues, handles = {}, []
    try:
        for path, layer_cls in routed.layer_classes.items():
            block = model.get_submodule(path)
            router = layer_cls._declared_router(block)
            gate_weights = _gate_weights_for(path, router)
            experts = layer_cls._find_experts_container(block)
            pinned.expert_paths[path] = next(f"{path}.{name}" for name, mod in block.named_modules() if mod is experts)
            queues[path] = collections.deque(routed.picks[path])
            sink = pinned.outputs.setdefault(path, [])
            handles.append(router.register_forward_hook(_pin(path, queues[path], gate_weights, pinned.regrets)))
            handles.append(
                block.register_forward_hook(lambda module, args, output, sink=sink: sink.append(_host(output)))
            )
        yield pinned
    finally:
        for handle in handles:
            handle.remove()
        pinned.unconsumed = {path: len(queue) for path, queue in queues.items()}


def _layer_outputs(outputs: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {path: torch.cat([o.reshape(-1, o.shape[-1]) for o in chunks]) for path, chunks in outputs.items()}


def _grad_pairs(routed: RoutedStep, pinned: PinnedReference, reference_grads: dict[str, torch.Tensor]):
    """``name -> (run, reference)`` gradients for every router and expert, the covered stock names, and experts zero on
    both sides or one."""
    pairs, covered, zero_both, zero_one = {}, set(), [], []
    for name, grad in routed.router_grads.items():
        pairs[name] = (grad, reference_grads.get(name))
        covered.add(name)
    for path, banks in routed.expert_grads.items():
        for key, bank in banks.items():
            stock = f"{pinned.expert_paths[path]}.{key}"
            covered.add(stock)
            want = reference_grads.get(stock)
            for expert in range(bank.shape[0]):
                label = f"{stock}[{expert}]"
                if want is None or want.shape != bank.shape:
                    pairs[label] = (bank[expert], None)
                    continue
                got_zero, want_zero = not bank[expert].any(), not want[expert].any()
                if got_zero and want_zero:
                    zero_both.append(label)
                elif got_zero or want_zero:
                    zero_one.append(label)
                else:
                    pairs[label] = (bank[expert], want[expert])
    return pairs, covered, zero_both, zero_one


def _score_pairs(pairs, *, cos_min: float, ratio_band: tuple[float, float]):
    """``(every pair matches, its metrics)``; a pair whose reference is missing or misshapen fails with no score."""
    checks, metrics = {}, {}
    scorable = {name: (got, want) for name, (got, want) in pairs.items() if want is not None}
    score_ep_grad_pairs(scorable, checks, metrics, cos_min=cos_min, ratio_band=ratio_band)
    return len(scorable) == len(pairs) and all(checks.values()), metrics


def routed_first_step_checks(
    routed: RoutedStep,
    pinned: PinnedReference,
    reference_grads: dict[str, torch.Tensor],
    *,
    device,
    output_rtol: float = TOL.routed_output_rel_l2,
    grad_cos_min: float = TOL.routed_grad_cosine_min,
    grad_ratio_band: tuple[float, float] = TOL.routed_grad_norm_ratio_band,
    regret_max: float = TOL.router_near_tie_logit_abs,
) -> tuple[dict[str, bool], dict[str, float]]:
    """Routed checks on a recorded step against its pinned reference. Forward, output and pick verdicts are world-wide
    (COLLECTIVE). An expert zero on one side only fails, and no token may name one expert twice. Every
    :data:`ROUTING_DEPENDENT_PARAM` must be compared."""
    forwards = {path: len(chunks) for path, chunks in routed.outputs.items()}
    pinned_forwards = {path: len(chunks) for path, chunks in pinned.outputs.items()}
    all_pinned = (
        bool(forwards)
        and len(set(forwards.values())) == 1
        and min(forwards.values()) > 0
        and forwards == pinned_forwards
        and all(len(routed.picks[path]) == count for path, count in forwards.items())
        and not any(pinned.unconsumed.values())
        and len(pinned.regrets) == sum(forwards.values())
    )
    regret = max_or_nan(pinned.regrets, default=math.nan)
    duplicated = routed.duplicate_pick_tokens()
    output_error, worst_layer = math.inf, None
    if all_pinned:
        output_error, worst_layer = worst_relative_l2(_layer_outputs(routed.outputs), _layer_outputs(pinned.outputs))

    pairs, covered, zero_both, zero_one = _grad_pairs(routed, pinned, reference_grads)
    routing_params = {name for name in reference_grads if ROUTING_DEPENDENT_PARAM.search(name)}
    grads_match, grad_metrics = _score_pairs(pairs, cos_min=grad_cos_min, ratio_band=grad_ratio_band)
    cosines = [value for key, value in grad_metrics.items() if key.endswith("_cos")]
    ratios = [value for key, value in grad_metrics.items() if key.endswith("_norm_ratio")]

    checks = {
        "routed_forwards_all_pinned": world_all(all_pinned, device),
        "routed_picks_are_reference_near_ties": world_all(regret <= regret_max, device),
        "routed_picks_are_distinct_experts": world_all(duplicated == 0, device),
        "routed_outputs_match_reference": world_all(output_error <= output_rtol, device),
        "routed_grads_cover_every_routing_param": bool(routing_params) and covered == routing_params,
        "routed_grads_match_reference": grads_match and not zero_one,
    }
    metrics = {
        "routed_output_max_rel_l2": output_error,
        "routed_pick_regret_max": regret,
        "routed_duplicate_pick_tokens": duplicated,
        "routed_grad_min_cos": min_or_nan(cosines, default=math.nan),
        "routed_grad_min_norm_ratio": min_or_nan(ratios, default=math.nan),
        "routed_grad_max_norm_ratio": max_or_nan(ratios, default=math.nan),
        "routed_grad_pairs": len(pairs),
    }
    log_all(
        f"  routed outputs over {len(forwards)} layers x {min(forwards.values(), default=0)} forwards: worst rel L2 "
        f"{output_error:.3e} at {worst_layer} (tol {output_rtol}); worst pick shortfall "
        f"{regret:.3e} (tol {regret_max}); {duplicated} tokens naming one expert twice"
    )
    log(
        f"  routed grads over {len(pairs)} router/expert slices: cos >= {metrics['routed_grad_min_cos']:.6f} "
        f"(min {grad_cos_min}), norm ratio {metrics['routed_grad_min_norm_ratio']:.5f}.."
        f"{metrics['routed_grad_max_norm_ratio']:.5f} (band {grad_ratio_band}); {len(zero_both)} experts zero on "
        f"both sides, {len(zero_one)} on one {zero_one[:4]}; uncovered routing params "
        f"{sorted(routing_params - covered)[:4]}"
    )
    return checks, metrics
