#!/usr/bin/env python
"""Routing replay over the environmental trainer's gradient forward replays only the positions it attends. A dropped
row's completion rides that forward as attention padding, and replaying it would count noise as flips."""

from collections import defaultdict
from typing import NamedTuple

import pytest
import torch

from halo.trainers.grpo.environmental.routing_replay import RoutingReplayInjector
from halo.trainers.grpo.environmental.step_batch import ROUTING_ROWS_KEY
from halo.trainers.grpo.environmental.trajectory_tokenize import TurnRow
from halo.trainers.grpo.logprob_scoring import dense_row_spans
from tests.common.env_grpo_batch import batch_host, build, episode
from tests.common.length_deal import single_process_accelerator
from tests.common.routing import BareEPLayer

_LAYERS, _TOP_K, _EXPERTS = 2, 2, 8
# The router's pick at a position it attends ranges over the first _CONTEXT_EXPERTS; on attention padding it is
# _PADDING_PICK, which no attended position shares.
_CONTEXT_EXPERTS = 6
_PADDING_PICK = (6, 7)
# Two GRPO groups of two: the first spreads in reward and trains, the second ties and is dropped.
_REWARDS = (1.0, 0.0, 0.5, 0.5)
_PROMPTS = ([11, 12, 13], [14], [15, 16], [17, 18, 19, 20])
_COMPLETIONS = ([21, 22, 23], [24, 25], [26, 27, 28, 29], [30, 31, 32])
_DROPPED_ROWS = (2, 3)

_REPLAY_MODES = pytest.mark.parametrize("replay", ["rollout", "recompute"])
_LOSS_PATHS = pytest.mark.parametrize("dense", [False, True], ids=["padded", "dense"])


class _LossForward(NamedTuple):
    """One gradient forward: flip rate, forced (token, layer) selections, trained batch, routed selections, and the
    ``[rows, seq, layers, top_k]`` routing the batch records (R2's captured mask, R3's rows padded from the host)."""

    rate: float | None
    forced: int
    batch: dict
    routed: list[torch.Tensor]
    masks: torch.Tensor


def _attended_pick(tokens: torch.Tensor, positions: torch.Tensor, layer: int) -> torch.Tensor:
    """The selection a router reading context makes at each token, ``[T, top_k]``: what the engine recorded."""
    first = (tokens + positions + layer) % _CONTEXT_EXPERTS
    return torch.stack([first, (first + 1) % _CONTEXT_EXPERTS], dim=-1)


def _engine_routing(prompt: list[int], completion: list[int]) -> torch.Tensor:
    """The engine's ``[P + C - 1, layers, top_k]`` routing: every forwarded token, the last sampled one never."""
    tokens = torch.tensor(prompt + completion[:-1])
    positions = torch.arange(len(tokens))
    return torch.stack([_attended_pick(tokens, positions, i) for i in range(_LAYERS)], dim=1).to(torch.int16)


def _rows(corrupt: tuple[int, int] | None = None) -> list[TurnRow]:
    """``corrupt`` = ``(row, position)`` moves that one engine pick off what the trainer's router computes there."""
    rows = []
    for r, (prompt, completion) in enumerate(zip(_PROMPTS, _COMPLETIONS, strict=True)):
        routing = _engine_routing(prompt, completion)
        if corrupt is not None and corrupt[0] == r:
            routing[corrupt[1]] = (routing[corrupt[1]] + 2) % _CONTEXT_EXPERTS
        rows.append(
            TurnRow(
                torch.tensor(prompt),
                torch.tensor(completion),
                torch.ones(len(completion), dtype=torch.long),
                None,
                (routing, len(prompt)),
            )
        )
    return rows


def _forwarding_scorer(injector: RoutingReplayInjector, dense: bool, routed: list[torch.Tensor]):
    """The policy forward as the EP layers see it. The padded path forwards rows whole, the dense path each span."""

    def picks(ids: torch.Tensor, attended: torch.Tensor, layer: int) -> torch.Tensor:
        positions = attended.long().cumsum(-1) - 1
        padding = torch.tensor(_PADDING_PICK).expand(*ids.shape, _TOP_K)
        return torch.where(attended[..., None], _attended_pick(ids, positions, layer), padding)

    def route(layer: BareEPLayer, selection: torch.Tensor) -> None:
        selection = layer._apply_routing_replay(selection)
        if layer._capture_routing:
            layer._captured_routing_chunks.append(selection.to(torch.int16))
        routed.append(selection)

    def score(_model, input_ids, attention_mask, logits_to_keep, compute_entropy=False, **_kwargs):
        attended = attention_mask.bool()
        for i, layer in enumerate(injector._layers):
            if dense:
                for r, (lo, hi) in enumerate(dense_row_spans(attention_mask)[0]):
                    route(layer, picks(input_ids[r, lo:hi], attended[r, lo:hi], i))
            else:
                route(layer, picks(input_ids, attended, i).reshape(-1, _TOP_K))
        logps = torch.zeros(input_ids.size(0), logits_to_keep, requires_grad=torch.is_grad_enabled())
        return logps, torch.zeros_like(logps.detach()), None

    return score


def _host(rows: list[TurnRow], *, replay: str, dense: bool, routed: list[torch.Tensor]):
    """The trainer's real batch build and loss forward over ``rows``, its EP layers bare."""
    width = max(len(c) for c in _COMPLETIONS)
    host = batch_host(rows, torch.zeros(len(rows), width), training=True)
    injector = RoutingReplayInjector(
        [BareEPLayer(top_k=_TOP_K, num_experts=_EXPERTS) for _ in range(_LAYERS)],
        engine_layers=_LAYERS,
        layer_indices=list(range(_LAYERS)),
    )
    host._routing_injector, host._replay_row_spans = injector, dense
    host._rollout_routing_replay = replay == "rollout"
    if replay == "recompute":
        # A config that runs the no-grad recompute pass R2 captures in.
        host.num_iterations = 2
    # TRL's loss state, a single-process DAPO step.
    host.accelerator = single_process_accelerator()
    host.use_liger_kernel = False
    host.aux_loss_enabled = False
    host.top_entropy_quantile = 1.0
    host.off_policy_mask_threshold = None
    host.importance_sampling_level = "token"
    host.loss_type = "dapo"
    host.epsilon_low, host.epsilon_high = 0.2, 0.28
    host.use_vllm = host.vllm_importance_sampling_correction = False
    host.max_completion_length = width
    host.current_gradient_accumulation_steps = 1
    host._entropy_bonus_enabled = False
    host.args.delta = None
    host._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    host._get_per_token_logps_and_entropies = _forwarding_scorer(injector, dense, routed)
    return host, injector


def _loss_forward(*, replay: str, dense: bool, corrupt: tuple[int, int] | None = None) -> _LossForward:
    routed: list[torch.Tensor] = []
    host, injector = _host(_rows(corrupt), replay=replay, dense=dense, routed=routed)
    batch = build(host, [episode(reward) for reward in _REWARDS])
    # The recompute pass's selections; the gradient forward's follow.
    routed.clear()
    host.compute_loss(host.model, batch)
    injector.disarm()
    forced = sum(int(layer._replay_flip_counts[1]) for layer in injector._layers)
    masks = host._rollout_routing.pad_rows(batch[ROUTING_ROWS_KEY].tolist(), torch.device("cpu"))
    return _LossForward(injector.drain_flip_rate(), forced, batch, routed, masks)


def _recorded_positions(forward: _LossForward) -> tuple[int, int]:
    """``(every position the batch's routing mask records, those on attention padding)``, per layer."""
    batch = forward.batch
    recorded = forward.masks[:, :, 0, 0] >= 0
    attended = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1).bool()
    return int(recorded.sum()), int((recorded & ~attended).sum())


@_REPLAY_MODES
@_LOSS_PATHS
def test_the_step_drops_its_tied_group_into_attention_padding(replay, dense):
    forward = _loss_forward(replay=replay, dense=dense)
    batch = forward.batch
    dropped = list(_DROPPED_ROWS)
    assert not batch["completion_mask"][dropped].any(), "the tied group's completions should be attention padding"
    assert batch["completion_mask"][[0, 1]].any(dim=1).all(), "the spread group should keep its completions"
    _, on_padding = _recorded_positions(forward)
    assert on_padding > 0, "the dropped completions were routed when recorded, so the batch carries that routing"


def test_the_padded_recompute_capture_records_ordinary_padding():
    """The padded no-grad pass also records padding no forward attends; arming it whole would force those positions."""
    forward = _loss_forward(replay="recompute", dense=False)
    assert not forward.batch["prompt_mask"].all(), "the prompts should differ in length, so the batch left-pads some"
    assert (forward.masks >= 0).all()


@_LOSS_PATHS
@_REPLAY_MODES
def test_a_round_carries_row_ids_in_place_of_a_dense_mask(replay, dense):
    """The batch holds one int64 id per row, and no tensor spans the round's width in the routing's layers."""
    batch = _loss_forward(replay=replay, dense=dense).batch
    ids = batch[ROUTING_ROWS_KEY]
    assert ids.dtype == torch.long and ids.tolist() == list(range(len(_REWARDS)))
    assert not any(isinstance(v, torch.Tensor) and v.dtype == torch.int16 for v in batch.values())


@_REPLAY_MODES
@_LOSS_PATHS
def test_a_dropped_rows_completion_is_not_replayed_or_counted(replay, dense):
    """A router that agrees with the recorded routing wherever it reads context flips nothing."""
    forward = _loss_forward(replay=replay, dense=dense)
    recorded, on_padding = _recorded_positions(forward)
    assert forward.rate == 0.0
    assert forward.forced == _LAYERS * (recorded - on_padding)


@_LOSS_PATHS
def test_a_genuine_flip_on_an_attended_position_is_still_counted(dense):
    """Negative control: one engine pick off the router's on a kept row's completion token counts once per layer."""
    corrupt_row, corrupt_position = 0, len(_PROMPTS[0]) + 1
    forward = _loss_forward(replay="rollout", dense=dense, corrupt=(corrupt_row, corrupt_position))
    recorded, on_padding = _recorded_positions(forward)
    assert forward.rate == pytest.approx(1 / (recorded - on_padding))


@_REPLAY_MODES
def test_the_padded_forward_routes_a_dropped_completion_by_its_own_pick(replay):
    forward = _loss_forward(replay=replay, dense=False)
    prompt_width = forward.batch["prompt_ids"].size(1)
    first_layer = forward.routed[0].view(len(_REWARDS), -1, _TOP_K)[:, prompt_width:]
    recorded = forward.masks[:, prompt_width:, 0]
    for r in _DROPPED_ROWS:
        replayed = recorded[r, :, 0] >= 0
        assert replayed.any()
        assert (first_layer[r][replayed] == torch.tensor(_PADDING_PICK)).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
