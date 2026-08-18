#!/usr/bin/env python
"""Pinning tests: the env-GRPO trajectory row IS one serving-template render, and the loss mask marks
exactly the spans of it the policy generated.

Chat templates are not prefix-monotone — the render of the first N messages is not generally a prefix
of the render of the first N+1 — so a row assembled from independently rendered per-turn prefixes
contains tokens no serving render produces: harmony marks the LAST assistant turn with ``<|return|>``
and every earlier one with ``<|end|>``, Qwen3 moves reasoning in and out of a turn depending on what
follows it. :func:`_prefix_delta_row` builds such a row as an independent contrast.

Placement is checked against oracles that do NOT read the implementation's own output:

* :data:`EXPECTED_SPAN_BOUNDARIES` — the first and last token of every trained span, per model and
  conversation. A span shifted by even one token changes one of them.
* :func:`_content_bounds` — where a message's content sits in the render, recovered by re-rendering
  the conversation with that content perturbed and diffing. Each span must cover its own content and
  reach no neighbour's.
* the final turn's span must end at the end of the render, and injected markers must never be trained.

Real tokenizers on purpose: a stand-in with a monotone template is exactly what these assertions must
not be satisfiable by. ``google/gemma-3-4b-it`` is the monotone control, so the machinery is shown not
to disturb the easy case.

    python tests/cpu/grpo/test_trajectory_render_fidelity.py
"""

import os
import types

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from src.environments.base import Message, Trajectory
from src.environments.episode import RolloutResult
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer as Trainer
from tests.common.tokenizers import try_cached_tokenizer

# Distinctive, non-overlapping markers so a decode-based assertion names exactly one message.
TOOL_CALL = [{"id": "c1", "type": "function", "function": {"name": "calc", "arguments": '{"a":"ARGSAAA"}'}}]


def _tool_convo():
    """user -> assistant(tool call + CoT) -> tool -> assistant(answer + CoT): every tool-using env's
    shape. Strictly alternating templates (Gemma) reject it."""
    return [
        Message.user("Add 21 and 21 with the calculator. USERAAA"),
        Message.assistant("", thinking="THINKAAA", tool_calls=TOOL_CALL),
        Message.tool("TOOLAAA", tool_call_id="c1", name="calc"),
        Message.assistant("The answer is ANSWERAAA.", thinking="THINKBBB"),
    ]


def _chat_convo():
    """user -> assistant -> user -> assistant: multi-turn dialogue, no tools (every template renders
    it). The turn-1 terminator is the one harmony rewrites."""
    return [
        Message.user("Who are you? USERAAA"),
        Message.assistant("I am REPLYAAA.", thinking="THINKAAA"),
        Message.user("Now add 21 and 21. USERBBB"),
        Message.assistant("It is REPLYBBB.", thinking="THINKBBB"),
    ]


def _whitespace_convo():
    """Assistant content opening with newlines, so the template's own trailing newline and the policy's
    first characters compete for one BPE token — the case where a span boundary is easiest to misplace."""
    return [
        Message.user("Print blank lines then a word. USERAAA"),
        Message.assistant("\n\nREPLYAAA", thinking="THINKAAA"),
        Message.user("Again please. USERBBB"),
        Message.assistant("\n\nREPLYBBB", thinking="THINKBBB"),
    ]


def _truncated_convo():
    """Ends on an env observation the policy never answers (the ``max_turns`` cap), so the render's last
    block is not an assistant turn."""
    return [
        Message.system("You are a calculator agent. SYSAAA"),
        Message.user("Add twice. USERAAA"),
        Message.assistant("", thinking="THINKAAA", tool_calls=TOOL_CALL),
        Message.tool("TOOLAAA", tool_call_id="c1", name="calc"),
        Message.assistant("", thinking="THINKBBB", tool_calls=TOOL_CALL),
        Message.tool("TOOLBBB", tool_call_id="c1", name="calc"),
    ]


CONVOS = {
    "tool": _tool_convo,
    "chat": _chat_convo,
    "whitespace": _whitespace_convo,
    "truncated": _truncated_convo,
}

# (first, last) token of every trained span, hand-checked against the full render; absent pairs are
# shapes the template rejects, kept honest by test_every_renderable_case_has_expected_boundaries.
EXPECTED_SPAN_BOUNDARIES = {
    ("Qwen/Qwen3-0.6B", "tool"): (("<think>", "\n"), ("<think>", "\n")),
    ("Qwen/Qwen3-0.6B", "chat"): (("I", "\n"), ("<think>", "\n")),
    ("Qwen/Qwen3-0.6B", "whitespace"): (("\n\n\n", "\n"), ("<think>", "\n")),
    ("Qwen/Qwen3-0.6B", "truncated"): (("<think>", "\n"), ("<think>", "\n")),
    # harmony: the mid-episode terminator is <|call|> / <|end|>, and <|return|> only ends the episode.
    ("openai/gpt-oss-20b", "tool"): ((" to", "<|call|>"), ("<|channel|>", "<|return|>")),
    ("openai/gpt-oss-20b", "chat"): (("<|channel|>", "<|end|>"), ("<|channel|>", "<|return|>")),
    ("openai/gpt-oss-20b", "whitespace"): (("<|channel|>", "<|end|>"), ("<|channel|>", "<|return|>")),
    ("openai/gpt-oss-20b", "truncated"): (("<|channel|>", "<|call|>"), ("<|channel|>", "<|call|>")),
    # Monotone control: its header absorbs the leading newlines, the opposite of Qwen3 on this convo.
    ("google/gemma-3-4b-it", "chat"): (("I", "\n"), ("It", "\n")),
    ("google/gemma-3-4b-it", "whitespace"): (("RE", "\n"), ("RE", "\n")),
}
CASES = sorted(EXPECTED_SPAN_BOUNDARIES)
MODELS = sorted({model for model, _ in CASES})

# Markers owned by env-injected messages: training on them is gradient on text the environment wrote.
INJECTED_MARKERS = ("USERAAA", "USERBBB", "TOOLAAA", "TOOLBBB", "SYSAAA")


def _stub(tok):
    """Minimal object exposing what ``_tokenize_trajectory`` reads, with the real render/limit methods
    bound so tokenization goes through the genuine chat-template path."""
    stub = types.SimpleNamespace(
        processing_class=tok,
        _tokenizer=tok,
        _tools_schema=None,
        _batch_build_error=None,
        _warned_capture_missing=False,
        # Named in the fall-back warning, whose remedy differs per engine.
        _rollout_backend="vllm",
        _rollout_routing_replay=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        # gpt-oss ships model_max_length as the unset sentinel; _context_limit then reads the config.
        model=types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=131072)),
    )
    stub._masked_trajectory_tensors = types.MethodType(Trainer._masked_trajectory_tensors, stub)
    stub._render_messages_to_ids = types.MethodType(Trainer._render_messages_to_ids, stub)
    stub._context_limit = types.MethodType(Trainer._context_limit, stub)
    stub._tokenize_trajectory = types.MethodType(Trainer._tokenize_trajectory, stub)
    stub._tokenize_trajectory_turns = types.MethodType(Trainer._tokenize_trajectory_turns, stub)
    return stub


def _load(model):
    """Tokenizer stub, or ``None`` when the tokenizer is not in the local cache."""
    tokenizer = try_cached_tokenizer(model)
    return None if tokenizer is None else _stub(tokenizer)


def _case(model, convo_name):
    """Tokenizer stub + messages + authoritative render, skipping only when the tokenizer is absent."""
    stub = _load(model)
    if stub is None:
        pytest.skip(f"{model} tokenizer unavailable")
    messages = CONVOS[convo_name]()
    return stub, messages, _render(stub, messages)


def _render(stub, messages, add_generation_prompt=False):
    return stub._render_messages_to_ids(messages, add_generation_prompt, {}, include_thinking=True)


def _renders(stub, messages):
    """The render, or ``None`` when the template rejects the conversation shape."""
    try:
        return _render(stub, messages)
    except Exception:
        return None


def _row(stub, messages):
    result = RolloutResult(prompt="task", trajectory=_trajectory(messages))
    prompt_ids, completion_ids, loss_mask = stub._tokenize_trajectory(result)
    return prompt_ids.tolist(), completion_ids.tolist(), loss_mask.tolist()


def _trajectory(messages):
    traj = Trajectory()
    for m in messages:
        traj.add_message(m)
    return traj


def _prefix_delta_row(stub, messages):
    """Row assembled by rendering each cumulative message prefix and concatenating the deltas.

    The construction a non-monotone template invalidates, kept as an independent contrast: where it
    disagrees with the full render, the trained row must follow the render and not this.
    """
    first = next(i for i, m in enumerate(messages) if m.role == "assistant")
    prompt = stub._render_messages_to_ids(messages[:first], True, {})
    completion: list[int] = []
    previous = prompt
    for idx in range(first, len(messages)):
        next_is_assistant = idx + 1 < len(messages) and messages[idx + 1].role == "assistant"
        current = stub._render_messages_to_ids(messages[: idx + 1], next_is_assistant, {})
        completion.extend(current[len(previous) :])
        previous = current
    return prompt, completion


def _mask_spans(mask):
    """Contiguous ``[start, end)`` runs of ones — the trained turns."""
    spans = []
    start = None
    for i, value in enumerate([*mask, 0]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            spans.append((start, i))
            start = None
    return spans


def _content_bounds(stub, messages, index):
    """``[start, end)`` of message ``index``'s content inside the authoritative render.

    Independent of the code under test: render the conversation twice, once with that message's content
    replaced, and take the common prefix and common suffix. What the two renders share is template
    scaffolding or another message; what differs is this message's content. ``None`` when the message
    has no content to locate.
    """
    if not messages[index].content:
        return None
    full = _render(stub, messages)
    perturbed = list(messages)
    perturbed[index] = Message(
        role=messages[index].role,
        content="ZQXPERTURBEDZQX content of a deliberately different length",
        name=messages[index].name,
        tool_calls=messages[index].tool_calls,
        tool_call_id=messages[index].tool_call_id,
        thinking=messages[index].thinking,
    )
    other = _render(stub, perturbed)
    head = 0
    while head < min(len(full), len(other)) and full[head] == other[head]:
        head += 1
    tail = 0
    while tail < min(len(full), len(other)) - head and full[-1 - tail] == other[-1 - tail]:
        tail += 1
    return head, len(full) - tail


@pytest.mark.parametrize(("model", "convo_name"), CASES)
def test_trained_row_is_exactly_the_authoritative_render(model, convo_name):
    """prompt + completion is the single full-conversation render, token for token.

    The row-level half of the contract: every trained token comes from a render a serving engine
    produces, so nothing is spliced in and no context can vanish. It says nothing about WHERE the mask
    falls — that is what the boundary and content oracles below are for.
    """
    stub, messages, full = _case(model, convo_name)
    prompt, completion, mask = _row(stub, messages)

    assert prompt + completion == full, "the trained row is not the serving-template render"
    assert len(mask) == len(completion)
    assert stub._batch_build_error is None, stub._batch_build_error


@pytest.mark.parametrize(("model", "convo_name"), CASES)
def test_trained_span_boundaries_match_the_expected_tokens(model, convo_name):
    """Each trained span starts and ends on the expected token.

    The placement oracle: the values in :data:`EXPECTED_SPAN_BOUNDARIES` are hand-checked against the
    render, so a span shifted by one token in either direction fails here — which an assertion that
    slices the same array the mask indexes cannot detect.
    """
    stub, messages, _ = _case(model, convo_name)
    _, completion, mask = _row(stub, messages)
    spans = _mask_spans(mask)
    expected = EXPECTED_SPAN_BOUNDARIES[model, convo_name]

    assert len(spans) == len(expected), f"expected {len(expected)} trained spans, got {len(spans)}"
    for turn, ((start, end), (head, tail)) in enumerate(zip(spans, expected, strict=True)):
        assert stub._tokenizer.decode([completion[start]]) == head, f"turn {turn} starts on the wrong token"
        assert stub._tokenizer.decode([completion[end - 1]]) == tail, f"turn {turn} ends on the wrong token"


@pytest.mark.parametrize(("model", "convo_name"), CASES)
def test_each_span_covers_its_own_content_and_no_neighbour_content(model, convo_name):
    """Every assistant turn's span covers that turn's content and stops short of its neighbours'.

    Bounds come from :func:`_content_bounds`, which locates content by perturbing it and diffing two
    renders — no dependence on the spans under test. A span shifted off its own content, or one that
    reaches into the environment's text, fails here.
    """
    stub, messages, full = _case(model, convo_name)
    prompt, completion, mask = _row(stub, messages)
    offset = len(prompt)
    assistant_indices = [i for i, m in enumerate(messages) if m.role == "assistant"]
    spans = dict(zip(assistant_indices, _mask_spans(mask), strict=True))
    bounds = {idx: _content_bounds(stub, messages, idx) for idx in range(len(messages))}

    assert len(full) == offset + len(completion)
    for idx, (start, end) in spans.items():
        own = bounds[idx]
        if own is not None:
            assert start + offset <= own[0] and own[1] <= end + offset, (
                f"turn {idx}'s span [{start + offset}, {end + offset}) does not cover its content {own}"
            )
        for other, message in enumerate(messages):
            if other == idx or message.role == "assistant" or bounds[other] is None:
                continue
            other_start, other_end = bounds[other]
            assert end + offset <= other_start or other_end <= start + offset, (
                f"turn {idx}'s span overlaps message {other}'s content [{other_start}, {other_end})"
            )


@pytest.mark.parametrize(("model", "convo_name"), CASES)
def test_environment_injected_text_is_never_trained(model, convo_name):
    """Env/tool/user text stays visible to attention and carries no loss; the assistant's own text
    carries loss."""
    stub, messages, _ = _case(model, convo_name)
    _, completion, mask = _row(stub, messages)
    tok = stub._tokenizer

    trained = tok.decode([t for t, m in zip(completion, mask, strict=True) if m])
    row = tok.decode(completion)
    # Anchor: both loops below only assert inside a guard, and on a trajectory whose assistant turns
    # carry no content — the ``truncated`` case, CoT plus tool calls only — neither fires. Training
    # nothing at all would otherwise satisfy "no injected marker is trained".
    assert trained.strip(), "nothing in this trajectory is trained"
    for marker in INJECTED_MARKERS:
        if marker in row:
            assert marker not in trained, f"injected marker {marker} is trained"
    for message in messages:
        if message.role == "assistant" and message.content.strip():
            assert message.content.strip() in trained, f"assistant text not trained: {message.content!r}"


@pytest.mark.parametrize(("model", "convo_name"), CASES)
def test_final_turn_span_ends_where_the_render_ends(model, convo_name):
    """A trajectory ending on an assistant turn trains through the last token of the render — that is
    where the episode's stop token lives. One ending on an observation must not."""
    stub, messages, full = _case(model, convo_name)
    prompt, _completion, mask = _row(stub, messages)
    last_end = _mask_spans(mask)[-1][1] + len(prompt)

    if messages[-1].role == "assistant":
        assert last_end == len(full), "the final turn's span stops before the end of the render"
    else:
        assert last_end < len(full), "a trailing observation must not be trained"


def test_mid_episode_turns_never_train_the_end_of_episode_terminator():
    """Where a template rewrites an earlier turn's terminator, the end-of-episode form appears nowhere
    in that turn's trained span.

    Rendering a turn on its own yields the end-of-episode form (harmony's ``<|return|>``); training it
    mid-episode teaches the policy to stop after its first tool call. Collecting the cases that
    actually rewrite keeps this from going vacuous if the fixtures stop covering such a template.
    """
    rewritten = []
    loaded = False
    for model, convo_name in CASES:
        stub = _load(model)
        if stub is None:
            continue
        loaded = True
        messages = CONVOS[convo_name]()
        prompt, completion, mask = _row(stub, messages)
        full = _render(stub, messages)
        assistant_indices = [i for i, m in enumerate(messages) if m.role == "assistant"]
        for idx, (start, end) in list(zip(assistant_indices, _mask_spans(mask), strict=True))[:-1]:
            solo = _renders(stub, messages[: idx + 1])
            if solo is None or solo[-1] == full[len(prompt) + end - 1]:
                continue
            rewritten.append((model, convo_name, idx))
            assert solo[-1] not in completion[start:end], (
                f"{model}/{convo_name} turn {idx} trains {stub._tokenizer.decode([solo[-1]])!r}, the "
                f"terminator its template uses only at the end of an episode"
            )
    if not loaded:
        pytest.skip("no tokenizer in the matrix is cached")
    assert rewritten, "no fixture exercises a template that rewrites an earlier turn's terminator"


def test_every_renderable_case_has_expected_boundaries():
    """Every (model, conversation) pair whose template renders the shape is pinned in
    :data:`EXPECTED_SPAN_BOUNDARIES`, so a case cannot quietly drop out of the boundary oracle."""
    missing = []
    loaded = False
    for model in MODELS:
        stub = _load(model)
        if stub is None:
            continue
        loaded = True
        for convo_name in CONVOS:
            renderable = _renders(stub, CONVOS[convo_name]()) is not None
            if renderable and (model, convo_name) not in EXPECTED_SPAN_BOUNDARIES:
                missing.append((model, convo_name))
    if not loaded:
        pytest.skip("no tokenizer in the matrix is cached")
    assert not missing, f"renderable cases without expected span boundaries: {missing}"


def test_fixtures_exercise_non_monotone_and_monotone_templates():
    """Anti-vacuity control for the matrix.

    The assertions above only bite when a fixture's template rewrites earlier turns. Assert the matrix
    holds both kinds: cases where the prefix-diffing row differs from the authoritative render, and at
    least one where it does not.
    """
    non_monotone, monotone = [], []
    for model, convo_name in CASES:
        stub = _load(model)
        if stub is None:
            continue
        messages = CONVOS[convo_name]()
        full = _renders(stub, messages)
        if full is None:
            continue
        prompt, completion = _prefix_delta_row(stub, messages)
        (monotone if prompt + completion == full else non_monotone).append((model, convo_name))

    if not non_monotone and not monotone:
        pytest.skip("no tokenizer in the matrix is cached")
    assert len(non_monotone) >= 2, f"fixtures no longer exercise non-monotone templates: {monotone}"
    assert monotone, f"fixtures lost the monotone control: {non_monotone}"


@pytest.mark.parametrize(("model", "convo_name"), CASES)
def test_prefix_diffing_row_is_rejected_where_it_diverges(model, convo_name):
    """Where the prefix-diffing row differs from the authoritative render the trained row follows the
    render; where the template is monotone the two agree, so nothing regresses."""
    stub, messages, full = _case(model, convo_name)
    naive_prompt, naive_completion = _prefix_delta_row(stub, messages)
    prompt, completion, _ = _row(stub, messages)

    assert prompt + completion == full
    if naive_prompt + naive_completion != full:
        assert prompt + completion != naive_prompt + naive_completion


def test_sampled_token_path_bypasses_re_rendering():
    """``train_on_sampled_tokens`` with the engine's ids present trains those ids verbatim: the
    re-render path stays out of the token-id path."""
    stub, messages, _ = _case("Qwen/Qwen3-0.6B", "tool")
    sampled = {1: [101, 102, 103], 3: [201, 202]}
    for idx, ids in sampled.items():
        messages[idx].token_ids = ids
        messages[idx].prompt_token_ids = [900 + idx, 901 + idx]

    rows = stub._tokenize_trajectory_turns(RolloutResult(prompt="task", trajectory=_trajectory(messages)))

    assert [row.completion_ids.tolist() for row in rows] == [sampled[1], sampled[3]]
    assert [row.prompt_ids.tolist() for row in rows] == [[901, 902], [903, 904]]
    assert stub._batch_build_error is None


def test_sampled_token_path_falls_back_to_the_identical_render_row():
    """With no captured ids the turns path falls back to the single re-rendered row, and that row is
    identical to the one ``_tokenize_trajectory`` builds — the two paths cannot disagree about what the
    trajectory is."""
    stub, messages, full = _case("openai/gpt-oss-20b", "tool")
    result = RolloutResult(prompt="task", trajectory=_trajectory(messages))

    rows = stub._tokenize_trajectory_turns(result)
    prompt, completion, mask = _row(stub, messages)

    assert len(rows) == 1, "no sampled ids → one re-tokenized row, not one row per turn"
    assert rows[0].prompt_ids.tolist() == prompt
    assert rows[0].completion_ids.tolist() == completion
    assert rows[0].completion_mask.tolist() == mask
    assert prompt + completion == full


def test_unlocatable_turn_span_is_reported_not_guessed():
    """A render that cannot be decomposed surfaces a named error (which
    ``_raise_batch_error_uniformly`` re-raises on every rank) and yields a fully masked row."""
    stub, messages, _ = _case("Qwen/Qwen3-0.6B", "tool")
    real_render = stub._render_messages_to_ids

    def _unanchorable(msgs, add_generation_prompt, template_kwargs, include_thinking=True):
        ids = real_render(msgs, add_generation_prompt, template_kwargs, include_thinking)
        # Full render untouched; every probe render gets a token prepended, so nothing anchors.
        return ids if len(msgs) == len(messages) and not add_generation_prompt else [0, *ids]

    stub._render_messages_to_ids = _unanchorable
    _, completion, mask = _row(stub, messages)

    assert stub._batch_build_error is not None
    assert "Qwen/Qwen3-0.6B" in stub._batch_build_error
    assert sum(mask) == 0, "an unlocatable trajectory must not contribute gradient"
    assert len(completion) == 1


def test_consecutive_assistant_turns_are_reported_not_guessed():
    """Two assistant messages in a row leave the second turn's context unanchorable on a template that
    rewrites an assistant-final prefix. The boundary between the generation prompt and the policy's own
    tokens is then unknowable — a header's token length is not turn-invariant, since BPE merges its
    last token with the turn's first content token differently per turn — so it must be reported rather
    than carried over from another turn.
    """
    stub = _load("openai/gpt-oss-20b")
    if stub is None:
        pytest.skip("gpt-oss tokenizer unavailable")
    messages = [
        Message.user("Say it in two messages. USERAAA"),
        Message.assistant("First REPLYAAA.", thinking="THINKAAA"),
        Message.assistant("Second REPLYBBB.", thinking="THINKBBB"),
    ]

    _, completion, mask = _row(stub, messages)

    assert stub._batch_build_error is not None, "an unknowable turn boundary must be reported"
    assert "unknowable" in stub._batch_build_error
    assert sum(mask) == 0 and len(completion) == 1


def test_template_render_failure_is_reported_not_raised():
    """A template that rejects the trajectory itself is recorded, not raised: a raw exception on one
    rank alone leaves its peers in the batch collectives until the NCCL watchdog fires."""
    stub, messages, _ = _case("Qwen/Qwen3-0.6B", "chat")

    def _explode(*_args, **_kwargs):
        raise TypeError("Can only get item pairs from a mapping")

    stub._render_messages_to_ids = _explode
    _, completion, mask = _row(stub, messages)

    assert stub._batch_build_error is not None
    assert "cannot render this trajectory" in stub._batch_build_error
    assert sum(mask) == 0 and len(completion) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
