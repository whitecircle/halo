"""Env-GRPO ``train_on_sampled_tokens``: parsing ``token_id:N`` logprobs and the per-turn row assembly. A row's
completion is that turn's captured ids verbatim, its prompt the serving-template render of the prior history."""

import types

import pytest
import torch

from halo.environments.engine_wire import _extract_token_ids, _extract_token_logprobs
from halo.environments.episode import RolloutResult
from halo.environments.messages import Message, Trajectory
from halo.trainers.grpo.environmental.step_batch import BatchBuildFence
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer as Trainer
from tests.common.models import GPT_OSS_20B
from tests.common.tokenizers import load_cached_tokenizer


def _choice(tokens):
    return {"logprobs": {"content": [{"token": t} for t in tokens]}}


def _choice_lp(pairs):
    return {"logprobs": {"content": [{"token": t, "logprob": lp} for t, lp in pairs]}}


def test_extract_token_ids_parses_token_id_form():
    ids = _extract_token_ids(_choice(["token_id:200005", "token_id:17196", "token_id:200008"]))
    assert ids == [200005, 17196, 200008]


def test_extract_token_ids_rejects_missing_or_malformed():
    # No logprobs at all: the caller falls back to re-tokenization.
    assert _extract_token_ids({}) is None
    assert _extract_token_ids({"logprobs": None}) is None
    # An empty content list is a zero-token completion the capture did return.
    assert _extract_token_ids({"logprobs": {"content": []}}) == []
    # A plain-text token means the server ran without --return-tokens-as-token-ids.
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
    assert _extract_token_logprobs({"logprobs": {"content": []}}) == []
    assert _extract_token_logprobs({"logprobs": {"content": [{"token": "token_id:1"}]}}) is None
    assert _extract_token_logprobs(_choice_lp([("token_id:1", "nan")])) is None


def _make_trainer_stub(tok):
    """Real methods bound, so the prompt render takes the real template path."""
    stub = types.SimpleNamespace(
        processing_class=tok,
        _tokenizer=tok,
        _tools_schema=None,
        _rollout_routing_replay=False,
        async_config=types.SimpleNamespace(rollout_chat_template_kwargs={}, max_train_row_tokens=None),
        _carry_reasoning=False,
        _rows_over_cap=0,
    )
    # gpt-oss ships model_max_length as the unset sentinel, so _context_limit falls back to the config.
    stub.model = types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=131072))
    stub._render_messages_to_ids = types.MethodType(Trainer._render_messages_to_ids, stub)
    stub._context_limit = types.MethodType(Trainer._context_limit, stub)
    stub._tokenize_trajectory_turns = types.MethodType(Trainer._tokenize_trajectory_turns, stub)
    for name in ("_template_kwargs", "_admit_row", "_sampled_turn_row"):
        setattr(stub, name, types.MethodType(getattr(Trainer, name), stub))
    return stub


def test_per_turn_prompt_uses_engine_ids_verbatim():
    """A turn carrying the engine's prompt ids uses them verbatim, since a template re-render drifts from the server on
    effort steering, tool-schema formatting and channel placement. Turns without them fall back to the render."""
    engine_prompt = [11, 22, 33, 44, 55]
    comp = [7, 8, 9]
    traj = Trajectory()
    traj.add_message(Message.user("solve it"))
    traj.add_message(Message.assistant("done", token_ids=comp, prompt_token_ids=engine_prompt))
    result = RolloutResult(prompt="solve it", trajectory=traj)

    fake_tok = types.SimpleNamespace(model_max_length=int(1e30))
    stub = types.SimpleNamespace(
        processing_class=fake_tok,
        _tokenizer=fake_tok,
        _tools_schema=None,
        _rollout_routing_replay=False,
        async_config=types.SimpleNamespace(rollout_chat_template_kwargs={}, max_train_row_tokens=None),
        _carry_reasoning=False,
        _rows_over_cap=0,
    )
    stub.model = types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=131072))

    def _no_render(*a, **k):
        raise AssertionError("engine prompt ids present — the re-render path must not run")

    stub._render_messages_to_ids = _no_render
    stub._context_limit = types.MethodType(Trainer._context_limit, stub)
    stub._tokenize_trajectory_turns = types.MethodType(Trainer._tokenize_trajectory_turns, stub)
    for name in ("_template_kwargs", "_admit_row", "_sampled_turn_row"):
        setattr(stub, name, types.MethodType(getattr(Trainer, name), stub))

    rows = stub._tokenize_trajectory_turns(result)
    assert len(rows) == 1
    assert rows[0][0].tolist() == engine_prompt, "row prompt must be the engine's rendered ids verbatim"
    assert rows[0][1].tolist() == comp


class _EffortRecorder:
    """Records the ``reasoning_effort`` every render receives; one id per character."""

    model_max_length = 100_000

    def __init__(self):
        self.efforts = []

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False, **kwargs):
        self.efforts.append(kwargs.get("reasoning_effort", "<<unset>>"))
        return "".join(f"<{m['role']}>{m.get('content', '')}" for m in messages)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_a_turn_without_engine_prompt_ids_renders_its_prompt_at_the_episodes_level():
    """The fallback render stands in for the engine's: it must state the level the episode was sampled at, or the
    row trains under a prompt the model never saw."""
    traj = Trajectory(reasoning_effort="high")
    traj.add_message(Message.user("solve it"))
    traj.add_message(Message.assistant("done", token_ids=[7, 8, 9]))
    recorder = _EffortRecorder()
    rows = _make_trainer_stub(recorder)._tokenize_trajectory_turns(RolloutResult(prompt="solve it", trajectory=traj))
    assert rows[0][1].tolist() == [7, 8, 9]
    assert recorder.efforts and set(recorder.efforts) == {"high"}, recorder.efforts


def test_per_turn_assembly_uses_sampled_ids_verbatim():
    tok = load_cached_tokenizer(GPT_OSS_20B)

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
    """The rollout sends ``env.get_tools_schema()`` as chat-template ``tools=``. Omitting it from the render drops ~2/3
    of the prompt (the harmony tool block) and mis-conditions every completion token."""
    tok = load_cached_tokenizer(GPT_OSS_20B)
    # Production pins this harmony template on both the trainer and the vLLM server; mirror it.
    with open("jinja-templates/gpt-oss/gpt-oss-harmony.jinja") as f:
        tok.chat_template = f.read()

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


def test_fallback_context_render_is_the_engine_view():
    """With carried reasoning only the previous assistant turn keeps its thought; without it none does."""

    def _episode():
        traj = Trajectory()
        traj.add_message(Message.user("q"))
        traj.add_message(Message.assistant("first", thinking="one", token_ids=[5], prompt_token_ids=[1]))
        traj.add_message(Message.tool("out", "c0", "echo"))
        traj.add_message(Message.assistant("second", thinking="two", token_ids=[6], prompt_token_ids=[1, 2]))
        traj.add_message(Message.assistant("third", thinking="three", token_ids=[7]))  # no engine prompt ids
        return RolloutResult(prompt="q", trajectory=traj)

    for carry, expected in ((True, [None, "two"]), (False, [None, None])):
        seen = []

        def render(msgs, add_generation_prompt, _kwargs, include_thinking=True):
            seen.append([m.thinking for m in msgs if m.role == "assistant"])
            return [1, 2, 3]

        stub = _render_stub(render)
        stub._carry_reasoning = carry
        stub._tokenize_trajectory_turns(_episode())
        assert seen == [expected], carry


def test_a_per_turn_row_over_the_train_row_cap_leaves_the_batch():
    """The cap is the rank's memory bound. The episode's other turns stay and the trainer's counter records it."""
    traj = Trajectory()
    traj.add_message(Message.user("q"))
    traj.add_message(Message.assistant("short", token_ids=[5, 6], prompt_token_ids=[1, 2]))
    traj.add_message(Message.assistant("long", token_ids=[7, 8, 9, 10], prompt_token_ids=[1, 2, 3, 4]))
    stub = _turns_stub()
    stub.async_config.max_train_row_tokens = 6
    rows = stub._tokenize_trajectory_turns(RolloutResult(prompt="q", trajectory=traj))
    assert [row.completion_ids.tolist() for row in rows] == [[5, 6]]
    assert stub._rows_over_cap == 1
    assert len(_turns_stub()._tokenize_trajectory_turns(RolloutResult(prompt="q", trajectory=traj))) == 2


def test_every_turn_over_the_cap_trains_as_one_masked_row():
    traj = Trajectory()
    traj.add_message(Message.user("q"))
    traj.add_message(Message.assistant("long", token_ids=[7, 8, 9, 10], prompt_token_ids=[1, 2, 3, 4]))
    stub = _turns_stub()
    stub.async_config.max_train_row_tokens = 6
    stub.eos_token_id = 2
    stub.pad_token_id = 0
    stub._masked_trajectory_tensors = types.MethodType(Trainer._masked_trajectory_tensors, stub)
    rows = stub._tokenize_trajectory_turns(RolloutResult(prompt="q", trajectory=traj))
    assert len(rows) == 1 and rows[0].loss_mask.tolist() == [0]
    assert stub._rows_over_cap == 1


def _turns_stub():
    stub = types.SimpleNamespace(
        _warned_once=set(),
        _rollout_routing_replay=False,
        async_config=types.SimpleNamespace(rollout_chat_template_kwargs={}, max_train_row_tokens=None),
        _carry_reasoning=False,
        _rows_over_cap=0,
    )
    stub._render_messages_to_ids = lambda *a, **k: [1, 2, 3]
    stub._context_limit = lambda: 10**9
    stub._tokenize_trajectory = lambda result: (torch.tensor([0]),) * 3
    stub._tokenize_trajectory_turns = types.MethodType(Trainer._tokenize_trajectory_turns, stub)
    for name in ("_template_kwargs", "_admit_row", "_sampled_turn_row"):
        setattr(stub, name, types.MethodType(getattr(Trainer, name), stub))
    return stub


def test_old_logps_aligned_1to1_with_sampled_ids():
    """The 4th tuple element is the turn's sampling logprobs, so the PPO ratio exp(cur - old) holds per token."""
    traj = Trajectory()
    traj.add_message(Message.user("q"))
    traj.add_message(Message.assistant("a1", token_ids=[10, 11, 12], token_logprobs=[-0.1, -0.2, -0.3]))
    traj.add_message(Message.tool("obs", tool_call_id="c1", name="t"))
    traj.add_message(Message.assistant("a2", token_ids=[20, 21], token_logprobs=[-0.4, -0.5]))

    rows = _turns_stub()._tokenize_trajectory_turns(RolloutResult(prompt="q", trajectory=traj))

    assert len(rows) == 2
    for (_, comp, mask, old, _routing, _negative_only), ids, lps in zip(
        rows, [[10, 11, 12], [20, 21]], [[-0.1, -0.2, -0.3], [-0.4, -0.5]], strict=True
    ):
        assert comp.tolist() == ids
        assert old is not None and old.dtype == torch.float32
        assert old.shape == comp.shape, "old_logps must be 1:1 with completion ids"
        assert torch.allclose(old, torch.tensor(lps))


def test_old_logps_none_when_a_turn_lacks_logprobs():
    """The batch-level gate then disables the IS trust region, so a misaligned or partial reference never trains."""
    traj = Trajectory()
    traj.add_message(Message.user("q"))
    traj.add_message(Message.assistant("a1", token_ids=[10, 11], token_logprobs=[-0.1, -0.2]))
    traj.add_message(Message.assistant("a2", token_ids=[20, 21], token_logprobs=None))

    rows = _turns_stub()._tokenize_trajectory_turns(RolloutResult(prompt="q", trajectory=traj))
    assert rows[0][3] is not None and rows[1][3] is None


def test_turns_path_records_context_overflow():
    """The turns path, which every shipped env config uses, must record an oversize error and never truncate."""
    stub = _turns_stub()
    stub._context_limit = lambda: 4  # render=3 prompt tokens + 5 completion ids = 8 > 4
    stub._batch_errors = BatchBuildFence()
    stub.async_config = types.SimpleNamespace(rollout_chat_template_kwargs={}, max_train_row_tokens=None)
    stub._carry_reasoning = False
    stub._rows_over_cap = 0
    traj = Trajectory()
    traj.add_message(Message.user("q"))
    traj.add_message(Message.assistant("a", token_ids=[1, 2, 3, 4, 5], token_logprobs=[-0.1] * 5))

    stub._tokenize_trajectory_turns(RolloutResult(prompt="q", trajectory=traj))
    assert stub._batch_errors.reason is not None
    assert "context window" in stub._batch_errors.reason


def _render_stub(render):
    stub = types.SimpleNamespace(
        _warned_once=set(),
        _rollout_routing_replay=False,
        _rollout_backend="vllm",
        _batch_errors=BatchBuildFence(),
        async_config=types.SimpleNamespace(rollout_chat_template_kwargs={}, max_train_row_tokens=None),
        _carry_reasoning=False,
        _rows_over_cap=0,
        eos_token_id=2,
        pad_token_id=0,
        _tokenizer=types.SimpleNamespace(name_or_path="stub-model"),
    )
    stub._render_messages_to_ids = render
    stub._context_limit = lambda: 10**9
    stub._masked_trajectory_tensors = types.MethodType(Trainer._masked_trajectory_tensors, stub)
    stub._invalidate_unrenderable_episode = types.MethodType(Trainer._invalidate_unrenderable_episode, stub)
    stub._tokenize_trajectory = types.MethodType(Trainer._tokenize_trajectory, stub)
    stub._tokenize_trajectory_turns = types.MethodType(Trainer._tokenize_trajectory_turns, stub)
    for name in ("_template_kwargs", "_admit_row", "_sampled_turn_row"):
        setattr(stub, name, types.MethodType(getattr(Trainer, name), stub))
    return stub


def test_unrenderable_trajectory_invalidates_the_episode_instead_of_the_run():
    """When the chat template rejects a trajectory (a malformed tool-call payload) the episode drops as invalid: a
    masked row, no group-baseline weight, and no fatal batch error to take every rank down."""

    def render(*_a, **_k):
        raise TypeError("Can only get item pairs from a mapping.")

    stub = _render_stub(render)
    traj = Trajectory()
    traj.add_message(Message.user("q"))
    traj.add_message(Message.assistant("a1", token_ids=None))  # engine returned no ids for this turn
    result = RolloutResult(prompt="q", trajectory=traj)

    rows = stub._tokenize_trajectory_turns(result)

    assert stub._batch_errors.reason is None, "a data-local template failure must not fail the run"
    assert traj.episode_invalid
    assert len(rows) == 1 and int(rows[0][2].sum()) == 0, "the fallback row must carry zero loss weight"


def test_per_turn_prefix_render_failure_drops_the_episode_not_the_run():
    calls = []

    def render(msgs, *_a, **_k):
        calls.append(len(msgs))
        raise ValueError("template rejects this prefix")

    stub = _render_stub(render)
    traj = Trajectory()
    traj.add_message(Message.user("q"))
    # Sampled ids present, engine prompt ids absent -> the per-turn path must re-render the prefix.
    traj.add_message(Message.assistant("a1", token_ids=[10, 11], token_logprobs=[-0.1, -0.2]))
    assert not traj.messages[-1].prompt_token_ids
    result = RolloutResult(prompt="q", trajectory=traj)

    rows = stub._tokenize_trajectory_turns(result)

    assert calls, "the prefix re-render must have been attempted"
    assert stub._batch_errors.reason is None
    assert traj.episode_invalid
    assert len(rows) == 1 and int(rows[0][2].sum()) == 0


def test_an_untrainable_turns_prefix_render_failure_drops_only_that_turn():
    """An untrainable turn's row only ever adds a penalty, so a prefix the template rejects costs that row alone."""

    def render(*_a, **_k):
        raise ValueError("template rejects this prefix")

    stub = _render_stub(render)
    traj = Trajectory()
    traj.add_message(Message.user("q"))
    traj.add_message(Message.assistant("cut", token_ids=[10, 11], truncated=True))
    traj.add_message(Message.user("continue"))
    traj.add_message(Message.assistant("a2", token_ids=[20, 21], prompt_token_ids=[1, 2, 3]))
    rows = stub._tokenize_trajectory_turns(RolloutResult(prompt="q", trajectory=traj))

    assert not traj.episode_invalid and stub._batch_errors.reason is None
    assert [(r.completion_ids.tolist(), r.negative_only) for r in rows] == [([20, 21], False)]


def test_prefix_render_failure_masks_the_episode_even_when_the_whole_render_would_succeed():
    """That render can succeed where the prefix render failed, handing an episode already out of the group baseline a
    weighted row (and, under rollout routing replay, one with no captured routing)."""

    def render(*_a, **_k):
        raise ValueError("template rejects this prefix")

    stub = _render_stub(render)
    stub._tokenize_trajectory = lambda _result: (
        torch.tensor([1, 2]),
        torch.tensor([10, 11]),
        torch.tensor([1, 1]),  # a whole-trajectory render that would train the turn
    )
    traj = Trajectory()
    traj.add_message(Message.user("q"))
    traj.add_message(Message.assistant("a1", token_ids=[10, 11], token_logprobs=[-0.1, -0.2]))
    result = RolloutResult(prompt="q", trajectory=traj)

    rows = stub._tokenize_trajectory_turns(result)

    assert traj.episode_invalid
    assert len(rows) == 1 and int(rows[0][2].sum()) == 0


def test_prefix_render_failure_on_a_later_turn_masks_the_earlier_rows_too():
    def render(*_a, **_k):
        raise ValueError("template rejects this prefix")

    stub = _render_stub(render)
    traj = Trajectory()
    traj.add_message(Message.user("q"))
    traj.add_message(Message.assistant("a1", token_ids=[10, 11], prompt_token_ids=[1, 2, 3]))
    traj.add_message(Message.user("more"))
    traj.add_message(Message.assistant("a2", token_ids=[12, 13]))  # engine prompt ids absent
    result = RolloutResult(prompt="q", trajectory=traj)

    rows = stub._tokenize_trajectory_turns(result)

    assert traj.episode_invalid
    assert len(rows) == 1 and int(rows[0][2].sum()) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
