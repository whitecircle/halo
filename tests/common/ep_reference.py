"""The undistributed reference EP, ETP and TP tests compare against, the routing that pins it to a parallel run's
top-k picks, and the negative control that proves the comparison is sensitive to expert identity: a parallel MoE
forward that sends tokens to the wrong expert still yields a finite loss and finite gradients. Reference and parallel
model share one ``attn_implementation``, since a sink-dropping backend moves a GptOss loss by nats.
"""

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

# Router parameter suffixes across the MoE roster: GptOss/DeepSeek-V4 name it ``router``,
# Qwen3/GLM4/Mistral4 name it ``gate``. The EP layer adopts the HF layer's own module
# (``EPMoELayerBase._find_gate_or_router``), so the parameter path is identical in the sharded and
# the dense model, which makes the two directly comparable.
_ROUTER_SUFFIXES = ("router.weight", "gate.weight")

# The six tensors ``EPMoELayerBase._sort_tokens_for_grouped_mm`` returns, in order.
GROUPED_SORT_OUTPUTS = ("sorted_tokens", "offs", "sorted_token_idx", "sorted_weights", "sorted_expert_ids", "inv_map")

# Norm ratio and cosine reported for an EP gradient with nothing to compare: no agreement bound accepts
# it, which also makes it unfit for a negative control's upper bound.
MISSING_GRAD_SCORE = -1.0

# A parameter name's decoder layer index, which also wildcards a name into its kind across layers.
_LAYER_INDEX = re.compile(r"\.layers\.(\d+)\.")
_ANY_LAYER = ".layers.*."
# The check :func:`score_ep_grad_pairs` records under a negligible floor: no parameter kind is negligible on every layer.
KIND_SCORED_CHECK = "negligible_grads_leave_every_kind_scored"
# The check a :func:`score_against_bf16_noise` caller records: no parameter is outside the bars at its best resolved
# step (:func:`parameters_outside_the_bars`).
NOISE_SCORED_CHECK = "noise_excuses_leave_every_parameter_scored"
# Per step, each parameter's ``name -> (gradient, bf16 reference, fp32 reference)``: what that scoring reads.
GradSteps = list[dict[str, tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]]]

# Deterministic conversations for the fixed batches, as (user, assistant) turns. Index 0 is the
# default short exchange; the rest are longer and mutually distinct, for callers that fill more of
# the sequence window or need several independent inputs.
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
    """Every EP/ETP-wrapped MoE layer in ``model``, by the walk the toolkit's own EP consumers use."""
    return [layer for _name, layer in named_ep_layers(model)]


def live_routers(layers: list) -> list[torch.nn.Module | None]:
    """Each of the EP ``layers``' own router module, resolved now so a PEFT ``modules_to_save`` copy is the one
    returned; None for a family whose router sits outside its EP layers (Gemma 4's sibling router)."""
    return [layer._live_router_module() for layer in layers]


def router_parameters(layers: list) -> list[torch.nn.Parameter]:
    """Every parameter of the EP ``layers``' own routers (:func:`live_routers`)."""
    return [param for router in live_routers(layers) if router is not None for param in router.parameters()]


def routed_parameters(layers: list) -> list[torch.nn.Parameter]:
    """Every parameter of the EP ``layers``' routed path: their expert weights (native expert adapters included) and
    their own routers (:func:`router_parameters`)."""
    experts = [param for layer in layers for _, param in layer.expert_named_params()]
    return experts + router_parameters(layers)


def router_parameter_names(model) -> set[str]:
    """The trainable parameters of every EP layer's own router (:func:`router_parameters`), by name."""
    routers = {id(param) for param in router_parameters(ep_layers(model)) if param.requires_grad}
    return {name for name, param in model.named_parameters() if id(param) in routers}


def sweep_ep_grads(model: torch.nn.Module) -> None:
    """COLLECTIVE where its legs reduce: the trainer's post-backward EP sweep over ``model``'s EP layers, as a
    test driving the layers without a trainer runs it once per optimizer step after the backward."""
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
    """The EP layers' routing capture armed over the body; on exit each layer's picks are appended to its entry of
    ``sinks`` through ``convert``: one ``[T, top_k]`` tensor of global expert ids per forward, in call order, so a
    backward's recomputes follow every forward. Refuses a layer another reader has armed, whose picks the two would
    split."""
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
    """Each of ``model``'s EP layers' top-k picks over the body (:func:`armed_routing_capture`), in :func:`ep_layers`
    order."""
    named_layers = named_ep_layers(model)
    captured: list[list[torch.Tensor]] = [[] for _ in named_layers]
    with armed_routing_capture(named_layers, captured, torch.Tensor.long):
        yield captured


@dataclass(frozen=True)
class PinnedRoute:
    """One EP layer's forwards routed on given top-k picks, scored against its router's own logits. A shortfall is how
    far a token's weakest pick sits below the router's own k-th logit: 0 for the router's own picks, a near-tie's
    rounding where reduction order flipped one, a top-k gap for a wrong pick. A token picking one expert twice falls
    short by nothing, so it is counted on its own."""

    # The worst shortfall of the pinned picks.
    regret: float
    # The worst shortfall of the router's own picks: 0 where it selects the top-k of its logits.
    own_regret: float
    # The worst shortfall of each token's (k+1)-th best expert: what the nearest wrong pick scores here.
    next_regret: float
    # Tokens whose pinned picks name one expert more than once.
    duplicated: int
    # Tokens whose pinned experts differ from the router's own.
    flipped: int
    tokens: int


def top_k_shortfall(logits: torch.Tensor, picks: torch.Tensor) -> torch.Tensor:
    """Per token, how far the weakest of ``picks`` sits below the ``picks.shape[-1]``-th largest of ``logits``."""
    kth = logits.topk(picks.shape[-1], dim=-1).values[:, -1]
    return kth - logits.gather(-1, picks).min(dim=-1).values


def duplicate_pick_tokens(picks: torch.Tensor) -> int:
    """The tokens whose ``[T, top_k]`` ``picks`` name one expert more than once, which :func:`top_k_shortfall` cannot
    see: a repeated top pick falls short by nothing."""
    return int((picks.sort(dim=-1).values.diff(dim=-1) == 0).any(dim=-1).sum())


def _pinned_route(layer, mask: torch.Tensor, logits: list[torch.Tensor], own: list[torch.Tensor]) -> PinnedRoute:
    """``layer``'s :class:`PinnedRoute` over the router outputs its pinned forwards recorded."""
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
    """Whether :func:`routing_pinned` can route ``model`` on given picks and score them: every EP layer hands routing
    replay (``_replay_router_choice``) the logits of a router module returning HF's ``(logits, scores, indices)``. Read
    off one no-grad forward over ``input_ids``."""
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
    """``model``'s EP layers routing on ``picks`` over the body (each layer's ``[T, top_k]`` global expert ids, in
    :func:`ep_layers` order, ``T`` every token its forwards run there, in order), through routing replay, which gates
    them with the router's own logits; on exit, each layer's :class:`PinnedRoute`. For routers whose module returns
    HF's ``(logits, scores, indices)``, and for forwards no backward recomputes outside a checkpoint scope, which would
    draw the picks again."""
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
    """Every MoE router weight in module order, by parameter name."""
    return [(name, param) for name, param in model.named_parameters() if name.endswith(_ROUTER_SUFFIXES)]


def find_router_weight(model) -> tuple[str, torch.nn.Parameter]:
    """First MoE router weight in module order."""
    routers = named_router_weights(model)
    if not routers:
        raise AssertionError(f"no router/gate weight found in {type(model).__name__} — nothing to compare")
    return routers[0]


def full_grad(param: torch.nn.Parameter) -> torch.Tensor:
    """A parameter's gradient as a full fp32 tensor, whether it is plain or a TP DTensor."""
    grad = param.grad
    if grad is None:
        raise AssertionError("parameter has no gradient — backward did not reach it")
    if hasattr(grad, "full_tensor"):
        grad = grad.full_tensor()
    return grad.detach().float().clone()


def gathered_expert_grads(layer) -> dict[str, torch.Tensor]:
    """COLLECTIVE over the layer's expert-TP and EP groups, and an FSDP2-sharded bank's mesh: its expert gradients
    as the stock container's full banks (``[E, 2M, H]`` gate-up and ``[E, H, M]`` down, or GPT-OSS's interleaved
    banks and biases), a rank whose experts no token reached contributing zeros."""
    local = {
        name: materialize_dtensor(torch.zeros_like(param) if param.grad is None else param.grad)
        for name, param in layer.expert_named_params()
    }
    interleaved = layer.FAMILY.expert_layout.kind is ExpertLayoutKind.INTERLEAVED_GLU
    layout = ExpertLayout(ExpertLayoutKind.INTERLEAVED_GLU if interleaved else ExpertLayoutKind.FUSED, key_prefix="")
    device = next(iter(local.values())).device
    return to_hub_expert_tensors(layout, layer.gather_expert_tensors(local), device=device)


def compare_grad(got: torch.Tensor, reference: torch.Tensor, label: str) -> tuple[float, float]:
    """Compare ``label``'s gradient to its single-GPU reference as (norm ratio, cosine similarity).

    Split rather than collapsed into one relative error because the two bug classes are independent:
    a mis-scaled cross-rank reduction moves the ratio and leaves the cosine at 1, while a
    routing/permutation corruption moves the cosine and can leave the ratio at 1. A single bound would
    have to be loosened past the weaker signal to absorb the other's noise. The cosine goes first: it
    raises on a zero-norm or non-finite operand before the ratio could divide by a zero norm.
    """
    got, reference = got.float(), reference.float()
    cosine = cos_sim(got, reference, label=label)
    return got.norm().item() / reference.norm().item(), cosine


def compare_ep_grad(got: torch.Tensor | None, reference: torch.Tensor, label: str) -> tuple[float, float]:
    """:func:`compare_grad` for an EP-side gradient that may be missing or misshapen.

    A severed backward leaves ``got`` None and layout drift changes its shape; either returns
    ``MISSING_GRAD_SCORE`` for both scores, so the pair fails under its own name instead of raising.
    """
    if got is None or got.shape != reference.shape:
        return MISSING_GRAD_SCORE, MISSING_GRAD_SCORE
    return compare_grad(got, reference, label)


def decoder_layer_index(name: str) -> int | None:
    """The index of the decoder layer parameter ``name`` sits in, None outside the layer stack."""
    match = _LAYER_INDEX.search(name)
    return None if match is None else int(match.group(1))


def parameter_kind(name: str) -> str:
    """Parameter ``name`` across decoder layers: its layer index wildcarded."""
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
    """Each pair's negligible floor: ``fraction`` of the largest reference gradient norm in its decoder layer, the
    parameters outside the layer stack taken as one more layer. Per layer, so every layer keeps its largest gradients
    on the direction and norm scores however small its gradients run."""
    largest: dict[int | None, float] = defaultdict(float)
    for name, (_, reference) in pairs.items():
        layer = decoder_layer_index(name)
        largest[layer] = max(largest[layer], reference.float().norm().item())
    return {name: fraction * largest[decoder_layer_index(name)] for name in pairs}


def _negligible_error(got: torch.Tensor | None, reference: torch.Tensor, floor: float) -> float | None:
    """``got``'s absolute error against a ``reference`` whose norm is under ``floor``; None for a pair the direction and
    norm scores take, a missing or misshapen ``got`` included."""
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
    """Record ``{name}_matches`` and its scores for each ``name -> (EP gradient, reference)`` pair.

    Both scores gate: a routing or permutation corruption moves the cosine, a missing or doubled
    cross-rank divide moves the norm ratio by the EP size. Mismatches are logged from every rank,
    since an expert bank's gradient is rank-local.

    With ``negligible_floors`` (:func:`per_layer_grad_floors`), a pair whose reference norm is under its floor is
    logged and held to that floor as absolute error instead, and :data:`KIND_SCORED_CHECK` requires every parameter
    kind (its name across layers) to keep a pair on the scores: a per-kind sync or shard fault reaches the kind's other
    layers.
    """
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
        # Per kind across layers: a negligible pair is small on every step, so only its kind's other layers score it.
        unscored = groups_never_scored(
            ((name, f"{name}_negligible_abs_error" not in metrics) for name in pairs), parameter_kind
        )
        checks[KIND_SCORED_CHECK] = not unscored
        if unscored:
            log_all(f"  GRAD MISMATCH: every pair of {unscored} is negligible, so no layer scores the kind")


def lacks_gradient(grad: torch.Tensor | None) -> bool:
    """Whether ``grad`` carries no gradient: None, as a rank keeps for experts no token reached, or exact zeros, as an
    unsharded reference sums for them."""
    return grad is None or not bool(grad.any())


@dataclass(frozen=True)
class StepGradPairs:
    """One step's gradients matched with their references (:func:`pair_step_grads`)."""

    # ``name -> (gradient, bf16 reference, fp32 reference)``, what :func:`score_against_bf16_noise` reads.
    pairs: dict[str, tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]]
    # Parameters the reference holds no entry for.
    missing: list[str]
    # Parameters without a gradient (:func:`lacks_gradient`) on each side, which must be the same.
    gradless: list[str]
    reference_gradless: list[str]


def pair_step_grads(
    grads: dict[str, torch.Tensor | None],
    ref_grads: dict[str, torch.Tensor | None],
    fp32_grads: dict[str, torch.Tensor | None],
    slices: dict[str, slice],
) -> StepGradPairs:
    """Each of ``grads`` with the bf16 and fp32 references' (an EP expert bank's cut to the experts ``slices`` gives the
    rank), and the parameters each side leaves without a gradient. A parameter only one side leaves without one is one
    whose contribution the other dropped or invented. Each parameter the reference holds a gradient for is paired, with
    None where ``grads`` holds none, which scores as missing."""
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
    """Whether some step of ``steps`` pairs one of ``names`` with a reference gradient to score. Parameters without a
    gradient on both sides pass the gradient-less match on nothing: a rank whose experts no token reaches at any step
    would have its expert sync compared on nothing."""
    names = set(names)
    return any(name in names for pairs in steps for name in pairs)


@dataclass(frozen=True)
class NoisyPairScore:
    """A gradient's direction and norm scores against a bf16 reference, and the excuse that reference's own measured
    bf16 noise gives it outside the bars."""

    ratio: float
    cosine: float
    norm_in_band: bool
    direction_in_bars: bool
    # How far inside the bars the pair sits, as a factor: the least of the norm ratio's to each end of the band and
    # ``1 - cos_min`` over ``1 - cosine``; under 1 outside them.
    bar_margin: float
    # The fp32 reference's norm, and the bf16 reference's over the largest one in its decoder layer
    # (:func:`per_layer_grad_floors`).
    fp32_norm: float
    layer_fraction: float
    # Distance from the fp32 reference's gradient (inf for a missing or misshapen gradient), and the parameter's noise.
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
        """The distance from the fp32 reference in multiples of the noise, what the excuse holds to its factor."""
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
    """Score each step's ``name -> (gradient, bf16 reference, fp32 reference)`` on the direction and norm bars against
    the bf16 reference, excusing a near-cancelled pair outside them that sits within ``noise_factor`` of the
    parameter's bf16 noise from the fp32 reference.

    A ``routed`` parameter (a router's or an expert bank's) is never excused: an fp32 reference that cannot route on the
    policy's picks routes some tokens differently, which puts their measured noise at a fifth or more of their norm,
    where the factor would excuse a halved gradient. Only a pair whose bf16 reference is under ``excusable_fraction`` of
    the largest bf16 reference gradient in its decoder layer is near-cancelled. A parameter's noise is the bf16
    reference's largest distance from the fp32 one over the steps: a near-cancelled vector's rounding follows the size of
    the terms it sums rather than their sum, and one step's draw of a few-element error can land near zero. A missing or
    misshapen gradient is never excused.
    """
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
    """Each parameter's score at its best resolved step, where its fp32 reference gradient is largest: a cancellation
    shrinks the sum it draws, and a bf16 reference's own rounding can inflate it at a cancelled step past the step that
    resolves it."""
    resolved: dict[str, NoisyPairScore] = {}
    for scores in scored:
        for name, score in scores.items():
            if name not in resolved or score.fp32_norm > resolved[name].fp32_norm:
                resolved[name] = score
    return resolved


def parameters_outside_the_bars(scored: list[dict[str, NoisyPairScore]]) -> list[str]:
    """The parameters outside the bars at their best resolved step (:func:`best_resolved_scores`). Per parameter, not
    per kind across layers, since a fault on one layer's parameter persists through every step while its kind's other
    layers score inside the bars; and at that step, since at a cancelled one a doubled gradient can land inside the bars
    while the noise excuses it at the others. A doubled or dropped sync moves the pair at its best resolved step by its
    factor, which leaves the bars unless the pair's own rounding already sits that close to a bar: such a parameter is
    held through its kind's other members, which a sync fault on the kind reaches as well."""
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
    """Whether parameter ``kind``'s pairs in ``steps``, each gradient replaced by ``fault(step, name, gradient)``, still
    match under :func:`score_against_bf16_noise` with none of them outside the bars at its best resolved step
    (:func:`parameters_outside_the_bars`): a negative control, which must not. The other pairs stay in the scoring, as
    their layers' largest gradients set the kind's excusable fractions."""
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
    """Deterministic right-padded chat batch. Padding is labelled ``-100`` so it never scores.

    ``conversation`` indexes :data:`FIXED_CONVERSATIONS` (modulo its length), ``broadcast`` shares
    rank 0's tensors with every rank, and the seed pins the tokenizer-independent parts so a rerun on
    one rank reproduces the same batch. The rendered date is pinned to :data:`CHAT_TEMPLATE_NOW`:
    ``strftime_now`` is a jinja global transformers injects into every chat template, and a same-named
    render kwarg shadows it, which is a no-op for templates that never call it.
    """
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
    """A stand-in carrying only what ``EPMoELayerBase._sort_tokens_for_grouped_mm`` reads off ``self``:
    ``ep_size``, ``experts_per_rank`` and the receive-count check."""

    _require_recv_counts = EPMoELayerBase._require_recv_counts


def received_expert_ids(num_tokens: int, top_k: int, experts_per_rank: int, ep_size: int, seed: int) -> torch.Tensor:
    """A DeepEP-shaped receive on rank 0: each token's ``top_k`` distinct global experts, kept as local ids where
    rank 0 owns them and ``-1`` elsewhere, for the tokens with at least one local expert."""
    g = torch.Generator().manual_seed(seed)
    picks = torch.rand(num_tokens, experts_per_rank * ep_size, generator=g).argsort(dim=1)[:, :top_k]
    local = torch.where(picks < experts_per_rank, picks, -1)
    return local[(local >= 0).any(dim=1)]


def dispatched_receive_counts(layer, experts: torch.Tensor) -> tuple[int, ...] | None:
    """The ``recv_counts`` a dispatch hands ``layer``'s expert compute for the received ids ``experts``: their
    :func:`receive_counts` at ``ep_size > 1``, None at ``ep_size`` 1, where nothing is dispatched."""
    return receive_counts(experts, layer.experts_per_rank) if layer.ep_size > 1 else None


def nonzero_grouped_sort(layer, tokens: torch.Tensor, experts: torch.Tensor, weights: torch.Tensor):
    """The grouped sort compacting DeepEP's padding with a device-sized ``nonzero``: the reference the counted
    compaction must equal bit for bit."""
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
    """``layer``'s per-expert loop selecting each expert's slots with a device-sized ``torch.where`` (its
    ``ep_size == 1`` branch): the reference the counted loop must equal bit for bit."""
    ep_size = layer.ep_size
    layer.ep_size = 1
    try:
        return layer._compute_experts_weighted(
            tokens, experts, weights, output_dtype, layer._expert_forward, recv_counts=None
        )
    finally:
        layer.ep_size = ep_size


def grouped_sort_mismatches(got, want) -> list[str]:
    """The names of the grouped-sort outputs that differ in dtype, shape or any bit."""
    return [
        name
        for name, g, w in zip(GROUPED_SORT_OUTPUTS, got, want, strict=True)
        if g.dtype != w.dtype or g.shape != w.shape or not torch.equal(g, w)
    ]


def random_token_batch(vocab_size: int, batch: int, seq: int, device, seed: int = 123):
    """Deterministic random ``(input_ids, labels)`` with every fourth label masked to ``-100``.

    Seeds the global RNG. The mask is interleaved rather than a contiguous span so every CP rank's
    sequence chunk keeps active labels, a non-zero loss and a gradient through its attention slice.
    """
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
    """Plain undistributed forward+backward on the same checkpoint. Returns (loss, router grad).

    The gradient half is what catches a mis-scaled cross-rank reduction: that bug leaves the
    forward exact and only rescales the router gradient.
    """
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
    """Share a rank-0-only reference with every rank (they all compare their own result against it)."""
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
    """Negative control: rotate every EP layer's local expert bank by ``shift`` experts.

    Router, dispatch and combine are untouched; only which FFN a dispatched token lands in changes,
    which is what a wrong-expert routing or ownership bug does. Every rank applies the same rotation
    to its own bank, so the perturbed model stays rank-consistent and its reduced loss is well
    defined. Returns the number of tensors rotated; 0 means the control never engaged, which must
    fail the test.
    """
    rotated = 0
    with torch.no_grad():
        for layer in ep_layers(model):
            for _, param in layer.expert_named_params():
                # Leading dim is the expert axis for every family's expert weights and biases.
                if param.shape[0] < 2:
                    continue
                param.data.copy_(torch.roll(param.data, shifts=shift, dims=0))
                rotated += 1
    return rotated
