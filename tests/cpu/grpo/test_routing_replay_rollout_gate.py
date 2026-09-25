#!/usr/bin/env python
"""CPU pin for the ``routing_replay='rollout'`` (R3) batch gate in
``DistributedAsyncEnvironmentalGRPOTrainer._assemble_rollout_routing``.

The gate exists because training under R3 without the engine's selection is silently wrong, which
includes a batch whose every routed row matches no engine coverage convention. What it must NOT do is
fail a step that trains nothing: every assistant turn the engine cut off at its token
cap is excluded from the training rows (``_tokenize_trajectory_turns``), so a policy emitting nothing
but runaway completions produces a batch of fully masked rows carrying no routing — the zero-gradient
step every other mode takes as a no-op. Both halves are driven end to end here: the real per-turn
tokenizer builds the rows, the real gate judges them.

    python tests/cpu/grpo/test_routing_replay_rollout_gate.py
"""

import types
from collections import defaultdict

import numpy as np
import pytest
import torch
from accelerate import PartialState
from trl.trainer.utils import pad

from src.environments.base import Message, Trajectory
from src.environments.episode import RolloutResult
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer as Trainer
from src.trainers.grpo.rollout.routing_replay import RoutingReplayInjector
from tests.common.grpo_metrics import attach_world_metrics, flushed_metrics
from tests.common.routing import BareEPLayer, raw_routing_payload

PartialState()  # the gate logs through accelerate, which refuses to log without it

# The engine's rows are per decoder layer; the EP layers are two of the four.
ENGINE_LAYERS, EP_LAYER_INDICES, TOP_K, EXPERTS = 4, [1, 3], 2, 8
ENGINE_PROMPT = [11, 22, 33]
SAMPLED = [44, 55]


def _routing_payload(tokens: int) -> str:
    """SGLang's wire form, one row per decoder layer."""
    return raw_routing_payload(np.zeros((tokens, ENGINE_LAYERS, TOP_K), dtype=np.int32))


def _rollout(
    *, truncated: bool, routing: bool, routing_tokens: int = len(ENGINE_PROMPT) + len(SAMPLED)
) -> RolloutResult:
    """One episode with a single assistant turn carrying the engine's ids (and optionally its routing,
    ``routing_tokens`` rows long: the default is the full prompt + completion convention)."""
    traj = Trajectory()
    traj.add_message(Message.user("solve it"))
    traj.add_message(
        Message.assistant(
            "answer",
            token_ids=list(SAMPLED),
            prompt_token_ids=list(ENGINE_PROMPT),
            routing_mask=_routing_payload(routing_tokens) if routing else None,
            routing_prompt_tokens=len(ENGINE_PROMPT) if routing else None,
            truncated=truncated,
        )
    )
    return RolloutResult(prompt="solve it", trajectory=traj)


def _host():
    """A trainer stand-in exposing the REAL per-turn tokenizer, gate and uniform raise.

    Single-process, so ``_raise_batch_error_uniformly`` skips its all-reduce and raises the recorded
    error directly — the gate's verdict is the assertion, not a mocked one.
    """
    host = types.SimpleNamespace(
        _rollout_routing_replay=True,
        _routing_injector=RoutingReplayInjector(
            [BareEPLayer(top_k=TOP_K, num_experts=EXPERTS) for _ in EP_LAYER_INDICES],
            engine_layers=ENGINE_LAYERS,
            layer_indices=EP_LAYER_INDICES,
        ),
        _batch_build_error=None,
        _rollout_template_kwargs={},
        _carry_reasoning=False,
        _max_train_row_tokens=None,
        _rows_over_cap=0,
        _warned_capture_missing=False,
        _tokenizer=types.SimpleNamespace(model_max_length=4096),
        pad_token_id=0,
        eos_token_id=1,
        _metrics=defaultdict(lambda: defaultdict(list)),
    )
    attach_world_metrics(host)
    for name in ("_tokenize_trajectory_turns", "_masked_trajectory_tensors", "_context_limit"):
        setattr(host, name, types.MethodType(getattr(Trainer, name), host))
    host._assemble_rollout_routing = types.MethodType(Trainer._assemble_rollout_routing, host)
    host._raise_batch_error_uniformly = types.MethodType(Trainer._raise_batch_error_uniformly, host)
    return host


def _gate(host, rows):
    """Run the gate over ``rows`` exactly as ``_build_training_tensors`` assembles its inputs."""
    tool_masks = [r.completion_mask.to(torch.bool) for r in rows]
    return host._assemble_rollout_routing(
        [r.turn_routing for r in rows],
        [torch.ones(len(r.prompt_ids), dtype=torch.bool) for r in rows],
        [r.completion_ids for r in rows],
        pad(tool_masks, padding_value=0, padding_side="right"),
        pad([r.prompt_ids for r in rows], padding_value=0, padding_side="left"),
        pad([r.completion_ids for r in rows], padding_value=0, padding_side="right"),
        torch.device("cpu"),
        "train",
    )


def test_engine_cut_turns_leave_a_masked_row_with_no_routing():
    """The mechanism: an excluded turn takes its routing with it, however well the engine captured it."""
    host = _host()
    rows = host._tokenize_trajectory_turns(_rollout(truncated=True, routing=True))
    assert len(rows) == 1
    assert rows[0].completion_mask.tolist() == [0], "an all-excluded trajectory must yield a masked row"
    assert rows[0].turn_routing is None
    assert host._batch_build_error is None


def test_gate_skips_a_batch_that_trains_nothing():
    """No routing plus no trainable token is an empty step, not a capture failure."""
    host = _host()
    rows = [row for _ in range(2) for row in host._tokenize_trajectory_turns(_rollout(truncated=True, routing=True))]
    assert not any(r.completion_mask.any() for r in rows)
    assert _gate(host, rows) is None


def test_gate_raises_when_a_trainable_row_carries_no_routing():
    """Anti-vacuity: the capture failure the gate exists for still fails, naming the serve flags."""
    host = _host()
    rows = host._tokenize_trajectory_turns(_rollout(truncated=False, routing=False))
    assert rows[0].completion_mask.tolist() == [1] * len(SAMPLED)
    with pytest.raises(ValueError, match="no rollout in this batch returned routed_experts"):
        _gate(host, rows)


def test_gate_raises_on_a_mixed_batch_that_still_trains_something():
    """Masked rows do not excuse a capture-less trainable one: that row would train with no selection
    to replay, which is the state the gate exists to refuse."""
    host = _host()
    rows = host._tokenize_trajectory_turns(_rollout(truncated=True, routing=True))
    rows += host._tokenize_trajectory_turns(_rollout(truncated=False, routing=False))
    with pytest.raises(ValueError, match="routed_experts"):
        _gate(host, rows)


def test_gate_assembles_the_engine_mask_for_a_trainable_row():
    """The pass-through case: a captured turn replays, and its coverage is recorded as fully aligned."""
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
    """Routing that lines up with no engine convention leaves every position on natural routing, so
    R3 would train inert with only a coverage metric to show it."""
    host = _host()
    rows = host._tokenize_trajectory_turns(_rollout(truncated=False, routing=True, routing_tokens=_UNRESOLVED_TOKENS))
    with pytest.raises(ValueError, match="no routed row on this rank"):
        _gate(host, rows)


def test_gate_keeps_a_partly_unresolved_batch():
    """Anti-vacuity: one unresolved row among replayed ones is coverage drift to watch, not a failure."""
    host = _host()
    rows = host._tokenize_trajectory_turns(_rollout(truncated=False, routing=True))
    rows += host._tokenize_trajectory_turns(_rollout(truncated=False, routing=True, routing_tokens=_UNRESOLVED_TOKENS))
    assert _gate(host, rows) is not None
    assert flushed_metrics(host)["routing/rollout_unresolved_frac"] == [0.5]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
