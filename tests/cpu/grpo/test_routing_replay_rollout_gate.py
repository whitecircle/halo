#!/usr/bin/env python
"""The ``routing_replay='rollout'`` (R3) batch gate. Training without the engine's selection is silently wrong, yet a
step that trains nothing must pass, and an engine-cut turn that may train carries its routing."""

import types
from collections import defaultdict

import numpy as np
import pytest
import torch

from halo.environments.episode import RolloutResult
from halo.environments.messages import Message, Trajectory
from halo.trainers.grpo.environmental.routing_replay import RoutingReplayInjector
from halo.trainers.grpo.environmental.step_batch import BatchBuildFence
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer as Trainer
from tests.common.routing import BareEPLayer, raw_routing_payload
from tests.common.world_metrics import attach_world_metrics, flushed_metrics

# The engine's rows are per decoder layer; the EP layers are two of the four.
ENGINE_LAYERS, EP_LAYER_INDICES, TOP_K, EXPERTS = 4, [1, 3], 2, 8
ENGINE_PROMPT = [11, 22, 33]
SAMPLED = [44, 55]


def _routing_payload(tokens: int) -> str:
    """SGLang's wire form, one row per decoder layer."""
    return raw_routing_payload(np.tile(np.arange(TOP_K, dtype=np.int32), (tokens, ENGINE_LAYERS, 1)))


def _rollout(
    *,
    truncated: bool,
    routing: bool,
    captured: bool = True,
    routing_tokens: int = len(ENGINE_PROMPT) + len(SAMPLED),
) -> RolloutResult:
    """One episode with a single assistant turn; ``routing_tokens`` defaults to the prompt + completion convention."""
    traj = Trajectory()
    traj.add_message(Message.user("solve it"))
    traj.add_message(
        Message.assistant(
            "answer",
            token_ids=list(SAMPLED) if captured else None,
            prompt_token_ids=list(ENGINE_PROMPT),
            routing_mask=_routing_payload(routing_tokens) if routing else None,
            routing_prompt_tokens=len(ENGINE_PROMPT) if routing else None,
            truncated=truncated,
        )
    )
    return RolloutResult(prompt="solve it", trajectory=traj)


def _host():
    """The real per-turn tokenizer, gate and raise. Single-process, so the fence's ``reject`` raises directly."""
    host = types.SimpleNamespace(
        _rollout_routing_replay=True,
        _routing_injector=RoutingReplayInjector(
            [BareEPLayer(top_k=TOP_K, num_experts=EXPERTS) for _ in EP_LAYER_INDICES],
            engine_layers=ENGINE_LAYERS,
            layer_indices=EP_LAYER_INDICES,
        ),
        _batch_errors=BatchBuildFence(),
        async_config=types.SimpleNamespace(rollout_chat_template_kwargs={}, max_train_row_tokens=None),
        _carry_reasoning=False,
        _rows_over_cap=0,
        _warned_once=set(),
        _is_correction=False,
        _tokenizer=types.SimpleNamespace(model_max_length=4096),
        pad_token_id=0,
        eos_token_id=1,
        _metrics=defaultdict(lambda: defaultdict(list)),
    )
    attach_world_metrics(host)
    for name in (
        "_tokenize_trajectory_turns",
        "_masked_trajectory_tensors",
        "_context_limit",
        "_template_kwargs",
        "_admit_row",
        "_sampled_turn_row",
        "_stack_rows",
        "_assemble_rollout_routing",
        "_rollout_masks_with_coverage",
    ):
        setattr(host, name, types.MethodType(getattr(Trainer, name), host))
    return host


def _gate(host, rows):
    """Run the gate over ``rows`` exactly as ``_build_training_tensors`` assembles its inputs."""
    rows = [r._replace(loss_mask=r.loss_mask.to(torch.bool)) for r in rows]
    step = host._stack_rows(rows, torch.device("cpu"))
    return host._assemble_rollout_routing(rows, step, torch.device("cpu"), "train")


def test_an_engine_cut_turn_keeps_its_routing_on_a_negative_only_row():
    host = _host()
    rows = host._tokenize_trajectory_turns(_rollout(truncated=True, routing=True))
    assert len(rows) == 1 and rows[0].negative_only
    assert rows[0].loss_mask.tolist() == [1] * len(SAMPLED)
    assert rows[0].turn_routing is not None
    assert host._batch_errors.reason is None


def test_gate_assembles_the_engine_mask_for_a_negative_only_row():
    """A tagged row trains under a negative advantage, so it replays the engine's selection like any row."""
    host = _host()
    masks = _gate(host, host._tokenize_trajectory_turns(_rollout(truncated=True, routing=True)))
    assert masks is not None and int(masks.min()) == 0
    assert flushed_metrics(host)["routing/rollout_full_frac"] == [1.0]


def test_gate_raises_when_a_negative_only_row_carries_no_routing():
    host = _host()
    rows = host._tokenize_trajectory_turns(_rollout(truncated=True, routing=False))
    with pytest.raises(ValueError, match="no rollout in this batch returned routed_experts"):
        _gate(host, rows)


def test_gate_skips_a_batch_that_trains_nothing():
    """Cut turns the engine returned no ids for yield no row at all, so this is an empty step, no capture failure."""
    host = _host()
    rows = [
        row
        for _ in range(2)
        for row in host._tokenize_trajectory_turns(_rollout(truncated=True, routing=False, captured=False))
    ]
    assert not any(r.loss_mask.any() for r in rows)
    assert _gate(host, rows) is None


def test_gate_raises_when_a_trainable_row_carries_no_routing():
    host = _host()
    rows = host._tokenize_trajectory_turns(_rollout(truncated=False, routing=False))
    assert rows[0].loss_mask.tolist() == [1] * len(SAMPLED)
    with pytest.raises(ValueError, match="no rollout in this batch returned routed_experts"):
        _gate(host, rows)


def test_gate_raises_on_a_mixed_batch_that_still_trains_something():
    """A capture-less trainable row would train with no selection to replay, the state the gate exists to refuse."""
    host = _host()
    rows = host._tokenize_trajectory_turns(_rollout(truncated=True, routing=False, captured=False))
    rows += host._tokenize_trajectory_turns(_rollout(truncated=False, routing=False))
    with pytest.raises(ValueError, match="routed_experts"):
        _gate(host, rows)


def test_gate_assembles_the_engine_mask_for_a_trainable_row():
    host = _host()
    rows = host._tokenize_trajectory_turns(_rollout(truncated=False, routing=True))
    masks = _gate(host, rows)
    assert masks is not None
    assert tuple(masks.shape) == (1, len(ENGINE_PROMPT) + len(SAMPLED), len(EP_LAYER_INDICES), TOP_K)
    # Every position of the covered row carries an engine id, not the -1 natural-routing sentinel.
    assert int(masks.min()) == 0
    assert flushed_metrics(host)["routing/rollout_full_frac"] == [1.0]


# Matches none of the conventions (prompt + completion, one short of it, completion-only).
_UNRESOLVED_TOKENS = len(ENGINE_PROMPT) + len(SAMPLED) + 3


def test_gate_raises_when_every_routed_row_is_unresolved():
    """Routing matching no engine convention leaves every position on natural routing, so R3 would train inert with
    only a coverage metric to show it."""
    host = _host()
    rows = host._tokenize_trajectory_turns(_rollout(truncated=False, routing=True, routing_tokens=_UNRESOLVED_TOKENS))
    with pytest.raises(ValueError, match="no routed row on this rank"):
        _gate(host, rows)


def test_gate_keeps_a_partly_unresolved_batch():
    """One unresolved row among replayed ones is coverage drift to watch, never a failure."""
    host = _host()
    rows = host._tokenize_trajectory_turns(_rollout(truncated=False, routing=True))
    rows += host._tokenize_trajectory_turns(_rollout(truncated=False, routing=True, routing_tokens=_UNRESOLVED_TOKENS))
    assert _gate(host, rows) is not None
    assert flushed_metrics(host)["routing/rollout_unresolved_frac"] == [0.5]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
