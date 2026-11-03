"""The undistributed reference EP, ETP and TP tests compare against, its routing pins, and the wrong-expert negative
control. Reference and parallel model share one ``attn_implementation``, since a sink-dropping backend moves a GptOss
loss by nats."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase, named_ep_layers
from halo.kernels.histogram import sync_free_bincount
from halo.kernels.moe_permute import MoEGatherPermute, build_inv_map
from halo.models.moe.expert_layout import ExpertLayout, ExpertLayoutKind, to_hub_expert_tensors
from halo.runtime.dtensor import materialize_dtensor
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.datasets import CHAT_TEMPLATE_NOW
from tests.common.memory import cleanup_memory
from tests.common.numerics import cos_sim
from tests.common.rank_log import log_all
from tests.common.route_sort_reference import receive_counts
from tests.common.tolerances import TOL

# The EP layer adopts the HF layer's own module, so these paths match in the sharded and dense models.
_ROUTER_SUFFIXES = ("router.weight", "gate.weight")

GROUPED_SORT_OUTPUTS = ("sorted_tokens", "offs", "sorted_token_idx", "sorted_weights", "sorted_expert_ids", "inv_map")

# Norm ratio and cosine for an EP gradient with nothing to compare: no agreement bound accepts it, so it cannot be a
# negative control's upper bound either.
MISSING_GRAD_SCORE = -1.0

_LAYER_INDEX = re.compile(r"\.layers\.(\d+)\.")
_ANY_LAYER = ".layers.*."
# Recorded by score_ep_grad_pairs under a negligible floor: no parameter kind is negligible on every layer.
KIND_SCORED_CHECK = "negligible_grads_leave_every_kind_scored"
# Recorded by score_against_bf16_noise callers: no parameter is outside the bars at its best resolved step.
NOISE_SCORED_CHECK = "noise_excuses_leave_every_parameter_scored"
# Per step, each parameter's ``name -> (gradient, bf16 reference, fp32 reference)``.
GradSteps = list[dict[str, tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]]]

# Deterministic (user, assistant) turns. Index 0 is the short default; the rest are longer and mutually distinct.
FIXED_CONVERSATIONS = (
    ("What is 42 + 58?", "The sum of 42 and 58 is 100."),
    (
        "What is 42 + 58? Please explain step by step.",
        "To calculate 42 + 58, I add the ones place first: 2 + 8 = 10, "
        "carry the 1. Then tens place: 4 + 5 + 1 = 10. So the answer is 100.",
    ),
    (
        "Explain the concept of prime numbers with examples.",
        "A prime number is only divisible by 1 and itself. Examples: 2, 3, 5, 7, 11. "
        "The number 4 is not prime because 4 = 2 x 2.",
    ),
    (
        "How do neural networks learn from data?",
        "Neural networks learn by adjusting weights through backpropagation. "
        "The loss function measures error, and gradients flow backwards to update parameters.",
    ),
    (
        "What is the difference between TCP and UDP protocols?",
        "TCP is connection-oriented and reliable with ordered delivery. "
        "UDP is connectionless and faster but without delivery guarantees.",
    ),
    (
        "Describe the water cycle in detail.",
        "Water evaporates from surfaces, rises as vapor, condenses into clouds, "
        "and falls as precipitation. It then flows through rivers back to oceans.",
    ),
)


def ep_layers(model) -> list:
    return [layer for _name, layer in named_ep_layers(model)]


def live_routers(layers: list) -> list[torch.nn.Module | None]:
    """Each EP layer's own router, resolved now so a PEFT ``modules_to_save`` copy wins; None where it sits outside the
    layer (Gemma 4)."""
    return [layer._live_router_module() for layer in layers]


def router_parameters(layers: list) -> list[torch.nn.Parameter]:
    return [param for router in live_routers(layers) if router is not None for param in router.parameters()]


def routed_parameters(layers: list) -> list[torch.nn.Parameter]:
    """Every parameter of the EP ``layers``' routed path: expert weights (native adapters included) and their own
    routers."""
    experts = [param for layer in layers for _, param in layer.expert_named_params()]
    return experts + router_parameters(layers)


def router_parameter_names(model) -> set[str]:
    routers = {id(param) for param in router_parameters(ep_layers(model)) if param.requires_grad}
    return {name for name, param in model.named_parameters() if id(param) in routers}


def sweep_ep_grads(model: torch.nn.Module) -> None:
    """COLLECTIVE where its legs reduce: the trainer's post-backward EP sweep, once per optimizer step, for tests with
    no trainer."""
    layers = ep_layers(model)
    expert_ids = {id(p) for layer in layers for _, p in layer.expert_named_params()}
    ep_ids = {id(p) for layer in layers for p in layer.parameters()}
    host = SimpleNamespace(
        _ep_config=layers[0].ep_config,
        state=SimpleNamespace(global_step=0),
        _get_sharded_expert_param_ids=lambda: expert_ids,
        _get_ep_param_ids=lambda: ep_ids,
        _top_level_model=lambda: model,
    )
    host._classify_deferred_params = MethodType(DistributedTrainerMixin._classify_deferred_params, host)
    DistributedTrainerMixin._sync_deferred_expert_grads(host)


@contextmanager
def armed_routing_capture(
    named_layers: list[tuple[str, EPMoELayerBase]],
    sinks: list[list[torch.Tensor]],
    convert: Callable[[torch.Tensor], torch.Tensor],
) -> Iterator[None]:
    """Arm the EP layers' routing capture over the body; on exit append each layer's ``[T, top_k]`` picks per forward
    to ``sinks``. Refuses a layer another reader has armed, since the two would split its picks."""
    for path, layer in named_layers:
        if layer._capture_routing or layer._captured_routing_chunks:
            raise RuntimeError(f"{path}: routing capture is already armed by another reader")
    for _, layer in named_layers:
        layer._capture_routing = True
    try:
        yield
    finally:
        for (_, layer), sink in zip(named_layers, sinks, strict=True):
            layer._capture_routing = False
            sink.extend(convert(chunk) for chunk in layer._captured_routing_chunks)
            layer._captured_routing_chunks = []


@contextmanager
def captured_picks(model) -> Iterator[list[list[torch.Tensor]]]:
    """Each EP layer's top-k picks over the body, in :func:`ep_layers` order."""
    named_layers = named_ep_layers(model)
    captured: list[list[torch.Tensor]] = [[] for _ in named_layers]
    with armed_routing_capture(named_layers, captured, torch.Tensor.long):
        yield captured


@dataclass(frozen=True)
class PinnedRoute:
    """One EP layer's forwards routed on given picks, scored against its router's logits. A shortfall is how far a
    token's weakest pick sits below the router's k-th logit: 0 for its own picks, rounding for a flipped near-tie, a
    top-k gap for a wrong pick."""

    regret: float
    # 0 where the router selects the top-k of its logits.
    own_regret: float
    # The worst shortfall of each token's (k+1)-th best expert: what the nearest wrong pick scores.
    next_regret: float
    duplicated: int
    flipped: int
    tokens: int


def top_k_shortfall(logits: torch.Tensor, picks: torch.Tensor) -> torch.Tensor:
    kth = logits.topk(picks.shape[-1], dim=-1).values[:, -1]
    return kth - logits.gather(-1, picks).min(dim=-1).values


def duplicate_pick_tokens(picks: torch.Tensor) -> int:
    """Tokens whose ``[T, top_k]`` picks name one expert twice, which :func:`top_k_shortfall` cannot see."""
    return int((picks.sort(dim=-1).values.diff(dim=-1) == 0).any(dim=-1).sum())


def _pinned_route(layer, mask: torch.Tensor, logits: list[torch.Tensor], own: list[torch.Tensor]) -> PinnedRoute:
    if not logits:
        raise AssertionError(f"{type(layer).__name__}: no forward ran while its routing was pinned")
    logits, own = torch.cat(logits), torch.cat(own)
    if logits.shape[0] != mask.shape[0] or layer._forced_consumed_total != mask.shape[0]:
        raise AssertionError(
            f"{type(layer).__name__}: its pinned forwards ran {logits.shape[0]} tokens and consumed "
            f"{layer._forced_consumed_total} picks of a {mask.shape[0]}-token mask: the picks do not tile its forwards"
        )
    ranked = logits.topk(mask.shape[-1] + 1, dim=-1).values
    return PinnedRoute(
        regret=top_k_shortfall(logits, mask).max().item(),
        own_regret=top_k_shortfall(logits, own).max().item(),
        next_regret=(ranked[:, -2] - ranked[:, -1]).max().item(),
        duplicated=duplicate_pick_tokens(mask),
        flipped=int((mask.sort(dim=-1).values != own.sort(dim=-1).values).any(dim=-1).sum()),
        tokens=mask.shape[0],
    )


def routing_can_be_pinned(model, input_ids: torch.Tensor) -> bool:
    """Whether every EP layer hands routing replay the logits of a router returning ``(logits, scores, indices)``; runs
    one no-grad forward."""
    layers = ep_layers(model)
    routers = live_routers(layers)
    if not layers or any(router is None for router in routers):
        return False
    outputs: dict[int, object] = {}
    replayed: dict[int, torch.Tensor] = {}

    def record_output(index: int):
        def hook(module, args, output) -> None:
            outputs[index] = output

        return hook

    def record_replay(index: int, replay):
        def replay_recorded(router_logits, indices, weights):
            replayed[index] = router_logits
            return replay(router_logits, indices, weights)

        return replay_recorded

    handles = [router.register_forward_hook(record_output(index)) for index, router in enumerate(routers)]
    try:
        for index, layer in enumerate(layers):
            layer._replay_router_choice = record_replay(index, layer._replay_router_choice)
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        for handle in handles:
            handle.remove()
        for layer in layers:
            vars(layer).pop("_replay_router_choice", None)
    return all(
        isinstance(outputs.get(index), tuple) and len(outputs[index]) == 3 and replayed.get(index) is outputs[index][0]
        for index in range(len(layers))
    )


@contextmanager
def routing_pinned(model, picks: list[torch.Tensor]) -> Iterator[list[PinnedRoute]]:
    """Route ``model``'s EP layers on ``picks`` over the body (per layer ``[T, top_k]`` global ids) and yield each
    :class:`PinnedRoute` on exit. For routers returning HF's ``(logits, scores, indices)``, and forwards no backward
    recomputes outside a checkpoint scope, which would redraw the picks."""
    layers = ep_layers(model)
    if len(picks) != len(layers):
        raise AssertionError(f"{len(picks)} pick masks for {len(layers)} EP layers")
    logits: list[list[torch.Tensor]] = [[] for _ in layers]
    own: list[list[torch.Tensor]] = [[] for _ in layers]

    def record(index: int):
        def hook(module, args, output) -> None:
            if not (isinstance(output, tuple) and len(output) == 3):
                raise AssertionError(
                    f"{type(module).__name__} returned {type(output).__name__}, not HF's (logits, scores, indices): "
                    f"no logits to score pinned picks against"
                )
            logits[index].append(output[0].detach().float())
            own[index].append(output[2].detach().long())

        return hook

    handles, routes = [], []
    try:
        for index, (layer, router, mask) in enumerate(zip(layers, live_routers(layers), picks, strict=True)):
            if router is None:
                raise AssertionError(f"{type(layer).__name__} routes through no router module of its own")
            handles.append(router.register_forward_hook(record(index)))
            layer._forced_topk_indices = mask.long()
            layer._forced_cursor = layer._forced_consumed_total = 0
        yield routes
        routes.extend(
            _pinned_route(layer, mask, layer_logits, layer_own)
            for layer, mask, layer_logits, layer_own in zip(layers, picks, logits, own, strict=True)
        )
    finally:
        for handle in handles:
            handle.remove()
        for layer in layers:
            layer._forced_topk_indices = layer._replay_flip_counts = None
            layer._forced_cursor = layer._forced_consumed_total = 0


def named_router_weights(model) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if name.endswith(_ROUTER_SUFFIXES)]


def find_router_weight(model) -> tuple[str, torch.nn.Parameter]:
    routers = named_router_weights(model)
    if not routers:
        raise AssertionError(f"no router/gate weight found in {type(model).__name__} — nothing to compare")
    return routers[0]


def full_grad(param: torch.nn.Parameter) -> torch.Tensor:
    grad = param.grad
    if grad is None:
        raise AssertionError("parameter has no gradient — backward did not reach it")
    if hasattr(grad, "full_tensor"):
        grad = grad.full_tensor()
    return grad.detach().float().clone()


def gathered_expert_grads(layer) -> dict[str, torch.Tensor]:
    """COLLECTIVE over the layer's expert-TP and EP groups and an FSDP2 bank's mesh: expert gradients as the stock
    container's full banks. A rank whose experts no token reached contributes zeros."""
    local = {
        name: materialize_dtensor(torch.zeros_like(param) if param.grad is None else param.grad)
        for name, param in layer.expert_named_params()
    }
    interleaved = layer.FAMILY.expert_layout.kind is ExpertLayoutKind.INTERLEAVED_GLU
    layout = ExpertLayout(ExpertLayoutKind.INTERLEAVED_GLU if interleaved else ExpertLayoutKind.FUSED, key_prefix="")
    device = next(iter(local.values())).device
    return to_hub_expert_tensors(layout, layer.gather_expert_tensors(local), device=device)


def compare_grad(got: torch.Tensor, reference: torch.Tensor, label: str) -> tuple[float, float]:
    """``label``'s gradient against its single-GPU reference as (norm ratio, cosine similarity). Split since a
    mis-scaled reduction moves only the ratio and a routing corruption can move only the cosine. The cosine goes first:
    it raises on a zero-norm or non-finite operand before the ratio could divide by zero."""
    got, reference = got.float(), reference.float()
    cosine = cos_sim(got, reference, label=label)
    return got.norm().item() / reference.norm().item(), cosine


def compare_ep_grad(got: torch.Tensor | None, reference: torch.Tensor, label: str) -> tuple[float, float]:
    """:func:`compare_grad` for an EP gradient; a missing or misshapen one scores ``MISSING_GRAD_SCORE`` so the pair
    fails by name."""
    if got is None or got.shape != reference.shape:
        return MISSING_GRAD_SCORE, MISSING_GRAD_SCORE
    return compare_grad(got, reference, label)


def decoder_layer_index(name: str) -> int | None:
    match = _LAYER_INDEX.search(name)
    return None if match is None else int(match.group(1))


def parameter_kind(name: str) -> str:
    return _LAYER_INDEX.sub(_ANY_LAYER, name)


def groups_never_scored(entries: Iterable[tuple[str, bool]], group: Callable[[str], str]) -> list[str]:
    """The ``group(name)`` keys none of whose ``(name, scored on the direction and norm bars)`` entries is scored."""
    scored: dict[str, bool] = defaultdict(bool)
    for name, on_the_bars in entries:
        scored[group(name)] |= on_the_bars
    return sorted(key for key, any_scored in scored.items() if not any_scored)


def per_layer_grad_floors(
    pairs: dict[str, tuple[torch.Tensor | None, torch.Tensor]], fraction: float
) -> dict[str, float]:
    """Each pair's negligible floor: ``fraction`` of the largest reference norm in its decoder layer, so every layer
    keeps its largest gradients scored."""
    largest: dict[int | None, float] = defaultdict(float)
    for name, (_, reference) in pairs.items():
        layer = decoder_layer_index(name)
        largest[layer] = max(largest[layer], reference.float().norm().item())
    return {name: fraction * largest[decoder_layer_index(name)] for name in pairs}


def _negligible_error(got: torch.Tensor | None, reference: torch.Tensor, floor: float) -> float | None:
    """``got``'s absolute error against a ``reference`` under ``floor``; None for a pair the direction and norm scores
    take."""
    if got is None or got.shape != reference.shape or reference.float().norm() >= floor:
        return None
    return (got.float() - reference.float()).norm().item()


def score_ep_grad_pairs(
    pairs: dict[str, tuple[torch.Tensor | None, torch.Tensor]],
    checks: dict[str, bool],
    metrics: dict[str, float],
    *,
    cos_min: float,
    ratio_band: tuple[float, float] = TOL.ep_grad_norm_ratio_band,
    negligible_floors: dict[str, float] | None = None,
) -> None:
    """Record ``{name}_matches`` and its scores for each ``name -> (EP gradient, reference)`` pair; logs from every
    rank. With ``negligible_floors``, a pair under its floor is held to it as absolute error, and
    :data:`KIND_SCORED_CHECK` requires every parameter kind to keep a scored pair."""
    low, high = ratio_band
    for name, (got, reference) in pairs.items():
        floor = None if negligible_floors is None else negligible_floors[name]
        error = None if floor is None else _negligible_error(got, reference, floor)
        if error is not None:
            metrics[f"{name}_negligible_abs_error"] = error
            checks[f"{name}_matches"] = error < floor
            norm = reference.float().norm().item()
            if checks[f"{name}_matches"]:
                log_all(
                    f"  negligible {name}: reference norm {norm:.3e} under its floor {floor:.3e}, error {error:.3e}"
                )
            else:
                log_all(
                    f"  GRAD MISMATCH {name}: negligible reference (norm {norm:.3e}), error {error:.3e} past {floor:.3e}"
                )
            continue
        ratio, cosine = compare_ep_grad(got, reference, name)
        metrics[f"{name}_cos"] = cosine
        metrics[f"{name}_norm_ratio"] = ratio
        checks[f"{name}_matches"] = cosine > cos_min and low < ratio < high
        if not checks[f"{name}_matches"]:
            shape = None if got is None else tuple(got.shape)
            log_all(f"  GRAD MISMATCH {name}: cos={cosine:.5f} norm_ratio={ratio:.4f} shape={shape}")
    if negligible_floors is not None:
        # A negligible pair is small on every step, so only its kind's other layers score it.
        unscored = groups_never_scored(
            ((name, f"{name}_negligible_abs_error" not in metrics) for name in pairs), parameter_kind
        )
        checks[KIND_SCORED_CHECK] = not unscored
        if unscored:
            log_all(f"  GRAD MISMATCH: every pair of {unscored} is negligible, so no layer scores the kind")


def lacks_gradient(grad: torch.Tensor | None) -> bool:
    """Whether ``grad`` is None (a rank whose experts no token reached) or exact zeros (what an unsharded reference
    sums there)."""
    return grad is None or not bool(grad.any())


@dataclass(frozen=True)
class StepGradPairs:
    pairs: dict[str, tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]]
    missing: list[str]
    # Without a gradient (:func:`lacks_gradient`) on each side, which must be the same.
    gradless: list[str]
    reference_gradless: list[str]


def pair_step_grads(
    grads: dict[str, torch.Tensor | None],
    ref_grads: dict[str, torch.Tensor | None],
    fp32_grads: dict[str, torch.Tensor | None],
    slices: dict[str, slice],
) -> StepGradPairs:
    """Pair ``grads`` with the bf16 and fp32 references, cutting an expert bank to the rank's ``slices``; a missing
    gradient pairs as None. A parameter only one side leaves gradient-less is one whose contribution the other dropped
    or invented."""
    missing = {name for name in grads if name not in ref_grads}
    cuts = {name: slices.get(name, slice(None)) for name in grads if name not in missing}
    references = {name: None if ref_grads[name] is None else ref_grads[name][cut] for name, cut in cuts.items()}
    gradless = {name for name in cuts if lacks_gradient(grads[name])}
    reference_gradless = {name for name, reference in references.items() if lacks_gradient(reference)}
    pairs = {
        name: (None if name in gradless else grads[name], reference, fp32_grads[name][cuts[name]])
        for name, reference in references.items()
        if name not in reference_gradless
    }
    return StepGradPairs(pairs, sorted(missing), sorted(gradless), sorted(reference_gradless))


def scores_any(steps: GradSteps, names: Iterable[str]) -> bool:
    """Whether some step pairs one of ``names`` with a reference gradient; else a rank's expert sync is compared on
    nothing."""
    names = set(names)
    return any(name in names for pairs in steps for name in pairs)


@dataclass(frozen=True)
class NoisyPairScore:
    """Direction and norm scores against a bf16 reference, and the excuse its measured bf16 noise gives outside the
    bars."""

    ratio: float
    cosine: float
    norm_in_band: bool
    direction_in_bars: bool
    # How far inside the bars, as a factor: the least of the norm ratio to each band end and ``1 - cos_min`` over ``1 -
    # cosine``. Under 1 outside them.
    bar_margin: float
    # The fp32 reference's norm, and the bf16 reference's over the largest one in its decoder layer.
    fp32_norm: float
    layer_fraction: float
    # Distance from the fp32 reference (inf for a missing or misshapen gradient), and the parameter's noise.
    fp32_error: float
    noise: float
    excused: bool

    @property
    def in_bars(self) -> bool:
        return self.norm_in_band and self.direction_in_bars

    @property
    def matches(self) -> bool:
        return self.in_bars or self.excused

    @property
    def noise_multiple(self) -> float:
        return self.fp32_error / self.noise if self.noise > 0 else math.inf


def score_against_bf16_noise(
    steps: GradSteps,
    *,
    cos_min: float,
    routed: frozenset[str] = frozenset(),
    ratio_band: tuple[float, float] = TOL.ep_grad_norm_ratio_band,
    noise_factor: float = TOL.bf16_noise_excuse_factor,
    excusable_fraction: float = TOL.bf16_noise_excusable_layer_fraction,
) -> list[dict[str, NoisyPairScore]]:
    """Score each step on the bars against the bf16 reference, excusing a near-cancelled pair within ``noise_factor``
    of its noise from fp32. Routed parameters are never excused: an fp32 reference that cannot route on the policy's
    picks has noise near a fifth of the norm, where the factor would excuse a halved gradient. Noise is the largest gap
    over steps, since one step's draw can land near zero."""
    noise: dict[str, float] = defaultdict(float)
    for pairs in steps:
        for name, (_, bf16, fp32) in pairs.items():
            noise[name] = max(noise[name], (bf16.float() - fp32.float()).norm().item())
    low, high = ratio_band
    scored = []
    for step, pairs in enumerate(steps):
        largest = per_layer_grad_floors({name: (got, bf16) for name, (got, bf16, _) in pairs.items()}, 1.0)
        scores = {}
        for name, (got, bf16, fp32) in pairs.items():
            ratio, cosine = compare_ep_grad(got, bf16, f"step{step}/{name}")
            norm_in_band, direction_in_bars = low < ratio < high, cosine > cos_min
            direction_margin = (1 - cos_min) / (1 - cosine) if cosine < 1 else math.inf
            layer_fraction = bf16.float().norm().item() / largest[name]
            shaped = got is not None and got.shape == fp32.shape
            error = (got.float() - fp32.float()).norm().item() if shaped else math.inf
            excused = (
                not (norm_in_band and direction_in_bars)
                and name not in routed
                and layer_fraction < excusable_fraction
                and error <= noise_factor * noise[name]
            )
            scores[name] = NoisyPairScore(
                ratio=ratio,
                cosine=cosine,
                norm_in_band=norm_in_band,
                direction_in_bars=direction_in_bars,
                bar_margin=min(ratio / low, high / ratio, direction_margin),
                fp32_norm=fp32.float().norm().item(),
                layer_fraction=layer_fraction,
                fp32_error=error,
                noise=noise[name],
                excused=excused,
            )
        scored.append(scores)
    return scored


def best_resolved_scores(scored: list[dict[str, NoisyPairScore]]) -> dict[str, NoisyPairScore]:
    """Each parameter's score at the step where its fp32 reference is largest; a cancellation's rounding can inflate
    bf16 past that step."""
    resolved: dict[str, NoisyPairScore] = {}
    for scores in scored:
        for name, score in scores.items():
            if name not in resolved or score.fp32_norm > resolved[name].fp32_norm:
                resolved[name] = score
    return resolved


def parameters_outside_the_bars(scored: list[dict[str, NoisyPairScore]]) -> list[str]:
    """Parameters outside the bars at their best resolved step, where a doubled or dropped sync moves the pair by its
    factor. Per parameter since a fault on one layer persists while its kind's other layers score inside the bars."""
    return sorted(name for name, score in best_resolved_scores(scored).items() if not score.in_bars)


def fault_passes_bf16_noise_scoring(
    steps: GradSteps,
    kind: str,
    fault: Callable[[int, str, torch.Tensor], torch.Tensor],
    *,
    cos_min: float,
    ratio_band: tuple[float, float] = TOL.ep_grad_norm_ratio_band,
    noise_factor: float = TOL.bf16_noise_excuse_factor,
    excusable_fraction: float = TOL.bf16_noise_excusable_layer_fraction,
) -> bool:
    """Negative control, which must return False: whether ``kind``'s pairs with ``fault`` applied still pass the bf16
    noise scoring. The other pairs stay in, as their layers' largest gradients set the kind's excusable fractions."""
    faulty = [
        {
            name: (fault(step, name, got) if got is not None and parameter_kind(name) == kind else got, bf16, fp32)
            for name, (got, bf16, fp32) in pairs.items()
        }
        for step, pairs in enumerate(steps)
    ]
    scored = [
        {name: score for name, score in scores.items() if parameter_kind(name) == kind}
        for scores in score_against_bf16_noise(
            faulty,
            cos_min=cos_min,
            ratio_band=ratio_band,
            noise_factor=noise_factor,
            excusable_fraction=excusable_fraction,
        )
    ]
    return all(score.matches for scores in scored for score in scores.values()) and not parameters_outside_the_bars(
        scored
    )


def fixed_chat_batch(
    tokenizer,
    seq_len: int,
    device: str,
    seed: int = 42,
    *,
    conversation: int = 0,
    broadcast: bool = False,
):
    """Deterministic right-padded chat batch; padding is labelled ``-100``. The date is pinned to
    :data:`CHAT_TEMPLATE_NOW`: a render kwarg named ``strftime_now`` shadows the jinja global transformers injects."""
    torch.manual_seed(seed)
    user, assistant = FIXED_CONVERSATIONS[conversation % len(FIXED_CONVERSATIONS)]
    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        tokenize=False,
        add_generation_prompt=False,
        strftime_now=CHAT_TEMPLATE_NOW.strftime,
    )
    tokens = tokenizer(text, return_tensors="pt", padding="max_length", max_length=seq_len, truncation=True)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    if broadcast:
        for tensor in (input_ids, attention_mask, labels):
            dist.broadcast(tensor, src=0)
    return input_ids, attention_mask, labels


class GroupedSortLayer(SimpleNamespace):
    """A stand-in with only what ``EPMoELayerBase._sort_tokens_for_grouped_mm`` reads off ``self``."""

    _require_recv_counts = EPMoELayerBase._require_recv_counts


def received_expert_ids(num_tokens: int, top_k: int, experts_per_rank: int, ep_size: int, seed: int) -> torch.Tensor:
    """A DeepEP-shaped receive on rank 0: top-k global experts kept as local ids where rank 0 owns them, ``-1``
    elsewhere."""
    g = torch.Generator().manual_seed(seed)
    picks = torch.rand(num_tokens, experts_per_rank * ep_size, generator=g).argsort(dim=1)[:, :top_k]
    local = torch.where(picks < experts_per_rank, picks, -1)
    return local[(local >= 0).any(dim=1)]


def dispatched_receive_counts(layer, experts: torch.Tensor) -> tuple[int, ...] | None:
    """The ``recv_counts`` a dispatch hands ``layer``'s expert compute; None at ``ep_size`` 1, where nothing is
    dispatched."""
    return receive_counts(experts, layer.experts_per_rank) if layer.ep_size > 1 else None


def nonzero_grouped_sort(layer, tokens: torch.Tensor, experts: torch.Tensor, weights: torch.Tensor):
    """Grouped sort compacting DeepEP's padding with a device-sized ``nonzero``: the reference the counted compaction
    must match bitwise."""
    N, width = experts.shape
    flat_indices = experts.reshape(-1)
    flat_weights = weights.reshape(-1)
    flat_token_idx = torch.arange(N, device=tokens.device).unsqueeze(1).expand(-1, width).reshape(-1)
    if layer.ep_size > 1:
        valid_idx = ((flat_indices >= 0) & (flat_indices < layer.experts_per_rank)).nonzero(as_tuple=True)[0]
        flat_indices = flat_indices[valid_idx]
        flat_weights = flat_weights[valid_idx]
        flat_token_idx = flat_token_idx[valid_idx]
    order = torch.argsort(flat_indices, stable=True)
    sorted_expert_ids = flat_indices[order]
    sorted_token_idx = flat_token_idx[order]
    sorted_weights = flat_weights[order]
    counts = sync_free_bincount(sorted_expert_ids, layer.experts_per_rank, dtype=torch.long)
    offs = torch.cumsum(counts, dim=0).to(torch.int32)
    inv_map = build_inv_map(sorted_token_idx, N, width)
    sorted_tokens = MoEGatherPermute.apply(tokens, sorted_token_idx, inv_map)
    return sorted_tokens, offs, sorted_token_idx, sorted_weights, sorted_expert_ids, inv_map


def where_expert_loop(layer, tokens: torch.Tensor, experts: torch.Tensor, weights: torch.Tensor, output_dtype):
    """The per-expert loop selecting slots with a device-sized ``torch.where``: the reference the counted loop must
    match bitwise."""
    ep_size = layer.ep_size
    layer.ep_size = 1
    try:
        return layer._compute_experts_weighted(
            tokens, experts, weights, output_dtype, layer._expert_forward, recv_counts=None
        )
    finally:
        layer.ep_size = ep_size


def grouped_sort_mismatches(got, want) -> list[str]:
    return [
        name
        for name, g, w in zip(GROUPED_SORT_OUTPUTS, got, want, strict=True)
        if g.dtype != w.dtype or g.shape != w.shape or not torch.equal(g, w)
    ]


def random_token_batch(vocab_size: int, batch: int, seq: int, device, seed: int = 123):
    """Deterministic ``(input_ids, labels)``; seeds the global RNG. Every fourth label is masked so each CP chunk keeps
    active labels."""
    torch.manual_seed(seed)
    ids = torch.randint(0, vocab_size, (batch, seq), device=device)
    labels = ids.clone()
    labels[:, ::4] = -100
    return ids, labels


def dense_reference(
    model_name: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    device: str,
    *,
    attn_implementation: str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[float, torch.Tensor]:
    """Plain undistributed forward+backward; returns (loss, router grad). A mis-scaled reduction shows only in the
    gradient."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, trust_remote_code=True, attn_implementation=attn_implementation
    ).to(device)
    model.train()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
    outputs.loss.backward()
    loss = outputs.loss.item()
    grad = full_grad(find_router_weight(model)[1])
    del model, outputs
    cleanup_memory()
    return loss, grad


def broadcast_reference(loss: float, grad: torch.Tensor | None, device: str, rank: int) -> tuple[float, torch.Tensor]:
    loss_t = torch.tensor([loss], device=device, dtype=torch.float32)
    dist.broadcast(loss_t, src=0)
    shape = torch.zeros(2, dtype=torch.long, device=device)
    if rank == 0:
        shape[0], shape[1] = grad.shape
    dist.broadcast(shape, src=0)
    if grad is None:
        grad = torch.zeros(int(shape[0]), int(shape[1]), dtype=torch.float32, device=device)
    dist.broadcast(grad, src=0)
    return float(loss_t.item()), grad


def roll_expert_banks(model, shift: int = 1) -> int:
    """Negative control: rotate every EP layer's local expert bank by ``shift``, which is what a wrong-expert routing
    bug does. Same rotation on every rank, so the reduced loss stays well defined. Returns the tensors rotated; 0 must
    fail the test."""
    rotated = 0
    with torch.no_grad():
        for layer in ep_layers(model):
            for _, param in layer.expert_named_params():
                if param.shape[0] < 2:
                    continue
                param.data.copy_(torch.roll(param.data, shifts=shift, dims=0))
                rotated += 1
    return rotated
