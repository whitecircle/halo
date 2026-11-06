"""Routing-replay stand-ins shared by the CPU suites: a bare EP layer and the two engine wire formats."""

import base64
import io

import numpy as np
import torch
from torch import nn

from halo.distributed.expert_parallel.moe_layer.shared_experts_layer import EPGroupLimitedMoELayerBase
from halo.models.moe.family_specs import MoEFamilySpec
from halo.trainers.grpo.environmental.rollout_routing import RolloutRouting
from halo.trainers.grpo.environmental.routing_replay import RoutingReplayInjector


class BareEPLayer(EPGroupLimitedMoELayerBase):
    """Minimal EP layer for the selection helpers: no dispatcher, rooted at the class owning group-limited routing, so
    helpers are what families inherit."""

    FAMILY = MoEFamilySpec(model_types=())

    def __init__(self, *, top_k: int, num_experts: int, **attrs):
        nn.Module.__init__(self)
        self.top_k = top_k
        self.num_experts = num_experts
        self.n_group = 1
        self.topk_group = 1
        self.norm_topk_prob = True
        self.routed_scaling_factor = 1.0
        self._forced_topk_indices = None
        self._forced_cursor = 0
        self._forced_consumed_total = 0
        self._capture_routing = False
        self._captured_routing_chunks = []
        self._replay_flip_counts = None
        for key, value in attrs.items():
            setattr(self, key, value)

    def forward(self, hidden_states, **kwargs):  # pragma: no cover — never called
        raise NotImplementedError


# A selection whose k-th and (k+1)-th router probabilities sit within this gap is a near-tie: bf16 hidden states that
# differ between packed and per-row forwards can tip it. About twice the widest margin of any flip seen on the tiny
# EP2 MoE (2.4e-3); the median selection has 1.3e-2.
NEAR_TIE_MARGIN_MAX = 5e-3
# Near-ties are a few tenths of the selections, so 2% still refuses a wholesale reroute of them; seen flip rates are 3e-3.
FLIP_RATE_MAX = 2e-2


def router_margins(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Softmax probability of the k-th selected expert minus the (k+1)-th, per token, for ``[..., experts]`` logits."""
    top = torch.topk(torch.softmax(logits.float(), dim=-1), top_k + 1, dim=-1).values
    return top[..., top_k - 1] - top[..., top_k]


def check_flips_are_near_ties(
    captured: torch.Tensor,
    reference: torch.Tensor,
    margins: torch.Tensor,
    *,
    near_tie_max: float = NEAR_TIE_MARGIN_MAX,
    flip_rate_max: float = FLIP_RATE_MAX,
) -> dict[str, float]:
    """Compare two ``[rows, seq, layers, top_k]`` routing captures (expert id sets per selection, ``-1`` where the
    position is not real) against the reference run's ``[rows, seq, layers]`` selection margins. Raises AssertionError
    when a selection that differs has a reference margin above ``near_tie_max`` or the flip rate exceeds
    ``flip_rate_max``; returns the flip rate, the flips' widest margin and the near-tie share of all selections."""
    real = reference[..., 0] >= 0
    assert torch.equal(captured[..., 0] >= 0, real), "the two captures cover different positions"
    flipped = (captured.sort(-1).values != reference.sort(-1).values).any(-1) & real
    total = int(real.sum())
    rate = int(flipped.sum()) / total
    widest = float(margins[flipped].max()) if flipped.any() else 0.0
    stats = {
        "flip_rate": rate,
        "widest_flip_margin": widest,
        "near_tie_share": float((margins[real] <= near_tie_max).float().mean()),
    }
    wide = int((flipped & (margins > near_tie_max)).sum())
    assert wide == 0, f"{wide} wide-margin selections differ (widest flipped margin {widest:.2e} > {near_tie_max:.1e})"
    assert rate <= flip_rate_max, f"{rate:.2e} of the selections differ (cap {flip_rate_max:.1e})"
    return stats


def bare_replay_injector(*, layers: int, top_k: int, num_experts: int = 8) -> RoutingReplayInjector:
    """An injector over ``layers`` :class:`BareEPLayer` engine layers, each held by the injector in order."""
    return RoutingReplayInjector(
        [BareEPLayer(top_k=top_k, num_experts=num_experts) for _ in range(layers)],
        engine_layers=layers,
        layer_indices=list(range(layers)),
    )


def npy_routing_payload(array) -> str:
    """vLLM's wire form: a base64 ``.npy`` (the header carries the shape)."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(array))
    return base64.b64encode(buf.getvalue()).decode()


def raw_routing_payload(array) -> str:
    """SGLang's wire form: base64 raw little-endian int32 rows, the shape implied."""
    return base64.b64encode(np.asarray(array, dtype=np.int32).tobytes()).decode()


def distinct_expert_picks(
    prefix: tuple[int, ...], layers: int, top_k: int, num_experts: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """An engine's ``[*prefix, layers, top_k]`` int16 picks: random experts, distinct within a token as a router's
    top-k is (a repeated id faults DeepEP dispatch)."""
    scores = torch.rand(*prefix, layers, num_experts, generator=generator)
    return scores.topk(top_k, dim=-1).indices.to(torch.int16)


def ragged_routing(masks: torch.Tensor, prompt_width: int) -> RolloutRouting:
    """The round's routing of a dense ``[rows, prompt + completion, layers, top_k]`` mask: each row whole, so padding
    rides as ``-1`` and ``pad_rows`` returns the dense rows back."""
    rows, width, layers, top_k = masks.shape
    return RolloutRouting(
        list(masks.cpu().unbind()), [prompt_width] * rows, prompt_width, width - prompt_width, layers, top_k
    )
