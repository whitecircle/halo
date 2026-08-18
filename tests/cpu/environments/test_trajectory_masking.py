#!/usr/bin/env python
"""Tests for Environmental-GRPO multi-turn trajectory tokenization + masking.

`_tokenize_trajectory` must (1) train on assistant tokens across *every* turn — not
collapse to turn 1 at the first per-turn ``<|im_end|>`` EOS — and (2) mask out
environment-injected tool/user tokens so they never receive policy gradient. These are
pure tokenizer mechanics, so they run on CPU against a real chat tokenizer (Qwen3, whose
EOS ``<|im_end|>`` doubles as the turn delimiter — the exact case a first-EOS
heuristic mis-handles). Run: ``pytest tests/cpu/environments/test_trajectory_masking.py``.
"""

import os
from types import MethodType, SimpleNamespace

import pytest
from transformers import AutoTokenizer

from src.environments.base import BaseEnvironment, Message, Trajectory
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer

MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def tokenizer():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return AutoTokenizer.from_pretrained(MODEL)


def _stub(tokenizer):
    """Minimal object exposing the attributes _tokenize_trajectory reads. The real
    ``_render_messages_to_ids`` is bound so the render goes through the genuine chat-template path
    (see the sibling stub in tests/cpu/grpo/test_token_id_training.py)."""
    stub = SimpleNamespace(
        processing_class=tokenizer,
        max_completion_length=2048,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        _tools_schema=None,  # resolved None → cached_property short-circuits, no env built
        _batch_build_error=None,
    )
    stub._tokenizer = tokenizer  # mirror __init__: processing_class may be a Processor
    stub._render_messages_to_ids = MethodType(DistributedAsyncEnvironmentalGRPOTrainer._render_messages_to_ids, stub)
    stub._context_limit = MethodType(DistributedAsyncEnvironmentalGRPOTrainer._context_limit, stub)
    stub._masked_trajectory_tensors = MethodType(
        DistributedAsyncEnvironmentalGRPOTrainer._masked_trajectory_tensors, stub
    )
    return stub


def _result(messages, prompt="task"):
    traj = Trajectory()
    for m in messages:
        traj.add_message(m)
    return SimpleNamespace(trajectory=traj, prompt=prompt)


def _tokenize(stub, result):
    return DistributedAsyncEnvironmentalGRPOTrainer._tokenize_trajectory(stub, result)


def test_multi_turn_assistant_tokens_all_trainable(tokenizer):
    # Both assistant turns stay trainable even though the first ends in <|im_end|> (EOS == delimiter).
    stub = _stub(tokenizer)
    messages = [
        Message.user("What is 2+2? Use the calculator."),
        Message.assistant("Let me compute that."),
        Message.tool("4", tool_call_id="c1", name="calc"),
        Message.assistant("The answer is 4."),
    ]
    _, completion_ids, completion_mask = _tokenize(stub, _result(messages))

    assert completion_ids.shape == completion_mask.shape
    assert completion_mask.sum() > 0
    # A mask collapsed at the first EOS would be entirely zero over the second assistant turn.
    assert completion_mask[len(completion_mask) // 2 :].sum() > 0

    trained_ids = completion_ids[completion_mask.bool()]
    trained_text = tokenizer.decode(trained_ids)
    assert "answer is 4" in trained_text


def test_tool_tokens_are_masked(tokenizer):
    stub = _stub(tokenizer)
    sentinel = "ZZQTOOLSENTINEL"
    messages = [
        Message.user("Look it up."),
        Message.assistant("Searching now."),
        Message.tool(f"result: {sentinel}", tool_call_id="c1", name="search"),
        Message.assistant("Found it."),
    ]
    _, completion_ids, completion_mask = _tokenize(stub, _result(messages))

    trained_text = tokenizer.decode(completion_ids[completion_mask.bool()])
    masked_text = tokenizer.decode(completion_ids[~completion_mask.bool()])
    assert sentinel not in trained_text
    assert sentinel in masked_text


def test_single_turn_trains_assistant(tokenizer):
    stub = _stub(tokenizer)
    messages = [Message.user("Hi"), Message.assistant("Hello there!")]
    _, completion_ids, completion_mask = _tokenize(stub, _result(messages))
    assert completion_mask.sum() > 0
    assert "Hello there" in tokenizer.decode(completion_ids[completion_mask.bool()])


def test_empty_trajectory_fallback(tokenizer):
    stub = _stub(tokenizer)
    prompt_ids, completion_ids, completion_mask = _tokenize(stub, _result([]))
    assert completion_ids.numel() == 1
    assert completion_mask.numel() == 1


# Reasoning (CoT) capture/gating contract: the final turn's CoT must be trainable, yet never reach a
# vLLM generation request (unknown field → 400, and prior-turn CoT must stay out of context).


def test_to_dict_includes_reasoning_only_when_requested():
    """Training tokenization opts in; a None CoT never emits the key (the harmony template keys on
    ``"thinking" in message`` and would concat None)."""
    assert Message.assistant("answer", thinking="COT").to_dict(include_thinking=True)["thinking"] == "COT"
    assert "thinking" not in Message.assistant("answer").to_dict(include_thinking=True)


def test_add_action_message_carries_reasoning_from_context():
    """The rollout loop passes the captured reasoning via context['reasoning']; it must land on the
    assistant Message so the final turn's CoT is trainable."""
    traj = Trajectory()
    # Instance method: it reads self.max_tool_calls_per_turn, whose None default means no truncation.
    BaseEnvironment._add_action_message(
        SimpleNamespace(max_tool_calls_per_turn=None),
        traj,
        "the answer",
        {"reasoning": "the chain of thought", "tool_calls": [{"id": "c1"}]},
    )
    msg = traj.messages[-1]
    assert msg.thinking == "the chain of thought"
    assert msg.tool_calls == [{"id": "c1"}]
    assert "thinking" not in msg.to_dict()


def test_add_action_message_caps_tool_calls_to_execution_limit():
    """The stored assistant tool_calls must be capped to max_tool_calls_per_turn — else the message
    advertises more calls than there are tool-result messages, a structural mismatch."""
    traj = Trajectory()
    ctx = {"tool_calls": [{"id": f"c{i}"} for i in range(8)]}
    BaseEnvironment._add_action_message(SimpleNamespace(max_tool_calls_per_turn=5), traj, "act", ctx)
    assert len(traj.messages[-1].tool_calls) == 5


class _EffortRecordingTokenizer:
    """Records the reasoning_effort kwarg every apply_chat_template render receives.

    Renders a minimal per-message template and tokenizes one id per character, so span location
    SUCCEEDS: a stand-in whose render ignored its messages would return the same ids for every probe,
    and the recorded efforts would then only prove the effort reached a failed location.
    """

    eos_token_id = 1
    pad_token_id = 0
    model_max_length = 100_000
    name_or_path = "effort-recorder"  # every PreTrainedTokenizerBase carries it; error paths name it

    def __init__(self):
        self.efforts = []

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False, **kwargs):
        self.efforts.append(kwargs.get("reasoning_effort", "<<unset>>"))
        text = "".join(f"<{m['role']}>{m.get('content', '')}</{m['role']}>" for m in messages)
        return f"{text}<assistant>" if add_generation_prompt else text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def _tokenize_with(tok, reasoning_effort):
    stub = SimpleNamespace(
        processing_class=tok,
        max_completion_length=2048,
        eos_token_id=1,
        pad_token_id=0,
        _tools_schema=None,
        _batch_build_error=None,
    )
    stub._tokenizer = tok
    stub._render_messages_to_ids = MethodType(DistributedAsyncEnvironmentalGRPOTrainer._render_messages_to_ids, stub)
    stub._context_limit = MethodType(DistributedAsyncEnvironmentalGRPOTrainer._context_limit, stub)
    traj = Trajectory(reasoning_effort=reasoning_effort)
    traj.add_message(Message.user("solve it"))
    traj.add_message(Message.assistant("done", thinking="cot"))
    row = DistributedAsyncEnvironmentalGRPOTrainer._tokenize_trajectory(
        stub, SimpleNamespace(trajectory=traj, prompt="p")
    )
    # The efforts below only mean something if the renders they came from located a real span.
    assert stub._batch_build_error is None, stub._batch_build_error
    assert row[2].sum() > 0, "the trained span is empty, so the recorded renders proved nothing"
    return row


def test_render_threads_resolved_reasoning_effort():
    # Every render must carry the resolved effort, or the trained prompt's level differs from generation's.
    tok = _EffortRecordingTokenizer()
    prompt_ids, completion_ids, mask = _tokenize_with(tok, "high")
    assert tok.efforts, "no renders recorded"
    assert all(e == "high" for e in tok.efforts), tok.efforts
    # The located span is the assistant turn's own text, not the template's scaffolding.
    trained = "".join(chr(t) for t, m in zip(completion_ids.tolist(), mask.tolist(), strict=True) if m)
    assert trained == "done</assistant>"
    assert "".join(chr(t) for t in prompt_ids.tolist()) == "<user>solve it</user><assistant>"


def test_render_omits_effort_when_unset():
    # reasoning_effort None → no kwarg passed (template falls back to its own default).
    tok = _EffortRecordingTokenizer()
    _tokenize_with(tok, None)
    assert tok.efforts and all(e == "<<unset>>" for e in tok.efforts), tok.efforts


def test_summarize_episode_generation_tokens_mean_max_p90():
    # Guards the aggregation the trainer logs as episode/generation_tokens{,_max,_p90}.
    from src.trainers.grpo.rollout.rollout_metrics import _summarize_episode_generation_tokens

    vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # each = one episode's total (summed over turns)
    s = _summarize_episode_generation_tokens(vals)
    assert s["episode/generation_tokens"] == 55.0  # mean, NOT the max/last
    assert s["episode/generation_tokens_max"] == 100.0
    # nearest-rank p90 of 10 sorted values -> ceil(0.9 * 10) = 9th = 90
    assert s["episode/generation_tokens_p90"] == 90.0
    # empty batch is inert (no rollouts this step), not a crash
    empty = _summarize_episode_generation_tokens([])
    assert empty == {
        "episode/generation_tokens": 0.0,
        "episode/generation_tokens_max": 0.0,
        "episode/generation_tokens_p90": 0.0,
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
