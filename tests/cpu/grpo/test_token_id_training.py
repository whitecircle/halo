"""Golden tests for env-GRPO ``train_on_sampled_tokens`` (training on the actual sampled generation
token ids captured from vLLM logprobs, instead of re-tokenizing a chat-template re-render).

Two units are covered without a GPU / a live model:
  1. ``_extract_token_ids`` — parsing the ``token_id:N`` logprobs the server emits under
     ``--return-tokens-as-token-ids`` (and refusing malformed/absent logprobs, so the caller falls back
     to re-tokenization rather than training on a corrupted sequence).
  2. The per-turn assembly invariant — each assistant turn's row completion is EXACTLY that turn's
     captured sampled ids (verbatim, no re-tokenization or trimming), and the row prompt is the
     serving-template render of the prior history. This is the whole point of the feature: the loss
     reinforces the sampled tokens verbatim, model-agnostically.

    python tests/cpu/grpo/test_token_id_training.py
"""

import sys
import types

import pytest

from src.environments.engine_wire import _extract_token_ids, _extract_token_logprobs


def _choice(tokens):
    return {"logprobs": {"content": [{"token": t} for t in tokens]}}


def _choice_lp(pairs):
    return {"logprobs": {"content": [{"token": t, "logprob": lp} for t, lp in pairs]}}


def test_extract_token_ids_parses_token_id_form():
    ids = _extract_token_ids(_choice(["token_id:200005", "token_id:17196", "token_id:200008"]))
    assert ids == [200005, 17196, 200008]


def test_extract_token_ids_rejects_missing_or_malformed():
    # No logprobs at all → None (caller falls back to re-tokenization).
    assert _extract_token_ids({}) is None
    assert _extract_token_ids({"logprobs": None}) is None
    assert _extract_token_ids({"logprobs": {"content": []}}) is None
    # A plain-text token (server not run with --return-tokens-as-token-ids) → None, not a guess.
    assert _extract_token_ids(_choice(["token_id:1", "hello"])) is None
    assert _extract_token_ids(_choice(["token_id:notanint"])) is None


def test_extract_token_logprobs_parses_values():
    lps = _extract_token_logprobs(_choice_lp([("token_id:1", -0.5), ("token_id:2", -1.25), ("token_id:3", 0.0)]))
    assert lps is not None and lps == [-0.5, -1.25, 0.0]
    # Feeds the logps/sampling_mean diagnostic: token-weighted mean over the turn.
    assert sum(lps) / len(lps) == pytest.approx(-0.5833, abs=1e-3)


def test_extract_token_logprobs_rejects_missing_or_malformed():
    assert _extract_token_logprobs({}) is None
    assert _extract_token_logprobs({"logprobs": None}) is None
    assert _extract_token_logprobs({"logprobs": {"content": []}}) is None
    # An entry with no numeric logprob → None (don't fabricate a confidence value).
    assert _extract_token_logprobs({"logprobs": {"content": [{"token": "token_id:1"}]}}) is None
    assert _extract_token_logprobs(_choice_lp([("token_id:1", "nan")])) is None


# --- per-turn assembly invariant (needs the gpt-oss tokenizer; skips cleanly offline) -------------


def _make_trainer_stub(tok):
    """A minimal object exposing exactly what _tokenize_trajectory_turns touches, with the real methods
    bound so the prompt render is the genuine template path."""
    from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer as Trainer

    stub = types.SimpleNamespace(
        processing_class=tok, _tokenizer=tok, _tools_schema=None, _rollout_routing_replay=False
    )
    # gpt-oss ships model_max_length as the unset sentinel, so _context_limit falls back to the config.
    stub.model = types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=131072))
    stub._render_messages_to_ids = types.MethodType(Trainer._render_messages_to_ids, stub)
    stub._context_limit = types.MethodType(Trainer._context_limit, stub)
    stub._tokenize_trajectory_turns = types.MethodType(Trainer._tokenize_trajectory_turns, stub)
    return stub


def test_per_turn_prompt_uses_engine_ids_verbatim():
    """When a turn carries the ENGINE's rendered prompt ids (vLLM return_token_ids), the row's prompt
    must be those ids verbatim — no template re-render (which drifts from the server on effort
    steering, tool-schema formatting, and channel placement). Turns without them fall back to the
    render. Model-agnostic: the ids are opaque to the assembly."""
    try:
        from src.environments.base import Message, Trajectory
        from src.environments.episode import RolloutResult
        from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer as Trainer
    except Exception as e:
        pytest.skip(f"trainer import unavailable: {e}")

    engine_prompt = [11, 22, 33, 44, 55]
    comp = [7, 8, 9]
    traj = Trajectory()
    traj.add_message(Message.user("solve it"))
    traj.add_message(Message.assistant("done", token_ids=comp, prompt_token_ids=engine_prompt))
    result = RolloutResult(prompt="solve it", trajectory=traj)

    fake_tok = types.SimpleNamespace(model_max_length=int(1e30))
    stub = types.SimpleNamespace(
        processing_class=fake_tok, _tokenizer=fake_tok, _tools_schema=None, _rollout_routing_replay=False
    )
    stub.model = types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=131072))

    def _no_render(*a, **k):
        raise AssertionError("engine prompt ids present — the re-render path must not run")

    stub._render_messages_to_ids = _no_render
    stub._context_limit = types.MethodType(Trainer._context_limit, stub)
    stub._tokenize_trajectory_turns = types.MethodType(Trainer._tokenize_trajectory_turns, stub)

    rows = stub._tokenize_trajectory_turns(result)
    assert len(rows) == 1
    assert rows[0][0].tolist() == engine_prompt, "row prompt must be the engine's rendered ids verbatim"
    assert rows[0][1].tolist() == comp


def test_per_turn_assembly_uses_sampled_ids_verbatim():
    try:
        from transformers import AutoTokenizer

        from src.environments.base import Message, Trajectory
        from src.environments.episode import RolloutResult
        from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer  # noqa: F401

        tok = AutoTokenizer.from_pretrained("unsloth/gpt-oss-20b-BF16")
    except Exception as e:
        pytest.skip(f"gpt-oss tokenizer / trainer import unavailable: {e}")

    # Distinct sampled-id sequences per turn; assembly must reproduce each without re-tokenizing.
    turn1 = tok(
        '<|channel|>commentary to=functions.python_repl <|constrain|>json<|message|>{"code":"print(1)"}<|call|>',
        add_special_tokens=False,
    )["input_ids"]
    turn2 = tok("<|channel|>final<|message|>done", add_special_tokens=False)["input_ids"]

    traj = Trajectory()
    traj.add_message(Message.user("solve it"))
    traj.add_message(
        Message.assistant(
            "",
            tool_calls=[{"type": "function", "function": {"name": "python_repl", "arguments": '{"code":"print(1)"}'}}],
            token_ids=turn1,
        )
    )
    traj.add_message(Message.tool("1", tool_call_id="c1", name="python_repl"))
    traj.add_message(Message.assistant("done", token_ids=turn2))
    result = RolloutResult(prompt="solve it", trajectory=traj)

    rows = _make_trainer_stub(tok)._tokenize_trajectory_turns(result)

    assert len(rows) == 2, "one training row per assistant turn"
    # The mask is all-ones because every token of an assistant turn was generated.
    assert rows[0][1].tolist() == turn1
    assert rows[1][1].tolist() == turn2
    assert rows[0][2].tolist() == [1] * len(turn1)
    assert rows[1][2].tolist() == [1] * len(turn2)
    # Turn 2's prompt must be the context the model conditioned on: prior tool call, then a gen prompt.
    prompt2 = tok.decode(rows[1][0].tolist())
    assert "to=functions.python_repl" in prompt2 and prompt2.rstrip().endswith("<|start|>assistant")


def test_prompt_render_includes_tool_schema():
    """The recompute prompt must condition on the SAME tool-definition block vLLM sampled under: the
    rollout sends env.get_tools_schema() as chat-template `tools=`, so the trainer render must too.
    Omitting it drops ~2/3 of the prompt (the harmony tool block) and mis-conditions every completion
    token — the mechanism behind is_ratio≈0.47. This fails if the render stops passing tools."""
    try:
        from transformers import AutoTokenizer

        from src.environments.base import Message, Trajectory
        from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer as Trainer

        tok = AutoTokenizer.from_pretrained("unsloth/gpt-oss-20b-BF16")
        # Production pins this harmony template on both the trainer and the vLLM server; mirror it.
        with open("jinja-templates/gpt-oss/gpt-oss-harmony.jinja") as f:
            tok.chat_template = f.read()
    except Exception as e:
        pytest.skip(f"gpt-oss tokenizer unavailable: {e}")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "python_repl",
                "description": "Execute Python code and return stdout.",
                "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
            },
        }
    ]
    stub = types.SimpleNamespace(
        processing_class=tok, _tokenizer=tok, _tools_schema=tools, _rollout_routing_replay=False
    )
    stub._render_messages_to_ids = types.MethodType(Trainer._render_messages_to_ids, stub)

    traj = Trajectory()
    traj.add_message(Message.user("add two numbers"))
    msgs = traj.messages

    ids_with = stub._render_messages_to_ids(msgs, True, {})
    stub._tools_schema = None
    ids_without = stub._render_messages_to_ids(msgs, True, {})

    text_with = tok.decode(ids_with)
    assert "python_repl" in text_with, "tool schema must appear in the rendered prompt"
    assert len(ids_with) > len(ids_without), "rendering tools must add the tool-definition block"


def _turns_stub():
    """A stub exposing only _tokenize_trajectory_turns + its render dep, no tokenizer needed for the
    old-logps rows (fallback path is not exercised)."""
    import types

    from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer as Trainer

    stub = types.SimpleNamespace(_warned_capture_missing=False, _rollout_routing_replay=False)
    stub._render_messages_to_ids = lambda *a, **k: [1, 2, 3]
    stub._context_limit = lambda: 10**9  # no overflow in this test's tiny rows
    stub._tokenize_trajectory = lambda result: (__import__("torch").tensor([0]),) * 3
    stub._tokenize_trajectory_turns = types.MethodType(Trainer._tokenize_trajectory_turns, stub)
    return stub


def test_old_logps_aligned_1to1_with_sampled_ids():
    """The behavior-policy reference (4th tuple element) is the turn's sampling logprobs, exactly aligned
    with completion_ids — so the PPO ratio exp(cur - old) is well-defined per sampled token."""
    import torch

    from src.environments.base import Message, Trajectory
    from src.environments.episode import RolloutResult

    traj = Trajectory()
    traj.add_message(Message.user("q"))
    traj.add_message(Message.assistant("a1", token_ids=[10, 11, 12], token_logprobs=[-0.1, -0.2, -0.3]))
    traj.add_message(Message.tool("obs", tool_call_id="c1", name="t"))
    traj.add_message(Message.assistant("a2", token_ids=[20, 21], token_logprobs=[-0.4, -0.5]))

    rows = _turns_stub()._tokenize_trajectory_turns(RolloutResult(prompt="q", trajectory=traj))

    assert len(rows) == 2
    for (_, comp, mask, old, _routing), ids, lps in zip(
        rows, [[10, 11, 12], [20, 21]], [[-0.1, -0.2, -0.3], [-0.4, -0.5]], strict=True
    ):
        assert comp.tolist() == ids
        assert old is not None and old.dtype == torch.float32
        assert old.shape == comp.shape, "old_logps must be 1:1 with completion ids"
        assert torch.allclose(old, torch.tensor(lps))


def test_old_logps_none_when_a_turn_lacks_logprobs():
    """A turn with ids but no logprobs → old is None for that row, so the batch-level gate disables the
    IS trust region (never a misaligned/partial reference)."""
    from src.environments.base import Message, Trajectory
    from src.environments.episode import RolloutResult

    traj = Trajectory()
    traj.add_message(Message.user("q"))
    traj.add_message(Message.assistant("a1", token_ids=[10, 11], token_logprobs=[-0.1, -0.2]))
    traj.add_message(Message.assistant("a2", token_ids=[20, 21], token_logprobs=None))

    rows = _turns_stub()._tokenize_trajectory_turns(RolloutResult(prompt="q", trajectory=traj))
    assert rows[0][3] is not None and rows[1][3] is None


def test_turns_path_records_context_overflow():
    """The sampled-tokens (default) path must record an oversize error when a per-turn row exceeds the
    context window — never silently truncate. The single-sequence path already does; this locks the
    turns path, which every shipped env config uses."""
    from src.environments.base import Message, Trajectory
    from src.environments.episode import RolloutResult

    stub = _turns_stub()
    stub._context_limit = lambda: 4  # render=3 prompt tokens + 5 completion ids = 8 > 4
    stub._batch_build_error = None
    traj = Trajectory()
    traj.add_message(Message.user("q"))
    traj.add_message(Message.assistant("a", token_ids=[1, 2, 3, 4, 5], token_logprobs=[-0.1] * 5))

    stub._tokenize_trajectory_turns(RolloutResult(prompt="q", trajectory=traj))
    assert stub._batch_build_error is not None
    assert "context window" in stub._batch_build_error


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
