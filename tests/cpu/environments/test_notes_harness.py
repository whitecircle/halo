#!/usr/bin/env python
"""The code-contests notes harness: a turn's message and notes come before its actions, the notes and every message
survive tool turns, a turn of text alone goes on, the size cap binds but is never stated, and every turn's
prompt is rebuilt from the history alone, so the trainer's per-turn render reads what the engine was sent."""

import json

import pytest

from halo.environments.episodes.episode_keys import COMPLETED_KEY, LAST_TURN_KEY
from halo.environments.sandbox.local import LocalSubprocessSandbox
from halo.environments.tasks.coding.code_contests import NOTES_KEY, CodeContestsEnvironment
from halo.environments.tasks.coding.code_contests_prompts import (
    AUDIT_CHECKS,
    LATER_RUN_REFUSAL_NOTE,
    NOTES_CHECK_DESCRIPTIONS,
    NOTES_CHECKS,
)
from halo.environments.tools.notes import NotesConfig, notes_after_call
from halo.trainers.grpo.environmental.environmental_rules import reject_whole_trajectory_render_of_rewritten_view

_ECHO = "print(int(input()) + 1)"


def _env(notes=None, **kwargs):
    return CodeContestsEnvironment(
        language="python", sandbox=LocalSubprocessSandbox(), max_test_calls=6, notes=notes, **kwargs
    )


def _reset(env):
    answer = {"tests": [{"input": "1\n", "output": "2\n"}]}
    ids, steps = env.reset(["Add one to n."], [{"answer": json.dumps(answer)}])
    return ids, steps[0]


def _turn(env, ids, text, *calls):
    base = len(env.get_trajectories(ids)[0].messages)
    tool_calls = [
        {"id": f"t{base}c{i}", "function": {"name": name, "arguments": json.dumps(arguments)}}
        for i, (name, arguments) in enumerate(calls)
    ]
    return env.step(ids, [text], [{"finish_reason": "stop", "tool_calls": tool_calls}])[0]


def _notes_block(observation):
    content = observation[-1]["content"]
    return content[content.index("<notes") :]


def test_one_call_writes_several_sections_and_leaves_the_others():
    notes = notes_after_call({}, "## idea\ngreedy by deadline\n## tests\nnone yet", None, 200)
    notes = notes_after_call(notes, "## tests\nsample ok", "replace", 200)
    notes = notes_after_call(notes, "## idea\nexchange argument holds", "append", 200)
    assert notes == {"idea": "greedy by deadline\nexchange argument holds", "tests": "sample ok"}
    assert notes_after_call(notes, "## tests\n", None, 200) == {"idea": notes["idea"]}
    assert notes_after_call({}, "plain text without headers", None, 200) == {"main": "plain text without headers"}


def test_a_long_section_name_is_a_header_not_body_text():
    name = "observations about the constraints and the bound on k"
    assert notes_after_call({}, f"## {name}\nk <= n", None, 200) == {name: "k <= n"}


def test_a_turn_can_analyse_note_and_act_and_the_next_prompt_carries_the_notes():
    env = _env(notes={})
    ids, first = _reset(env)
    assert first.observation[-1] == {"role": "user", "content": "<notes>\n(empty)\n</notes>"}
    step = _turn(
        env,
        ids,
        "n fits in an int; answer is n + 1.",
        ("notes", {"text": "## idea\nanswer = n + 1\n## next\nrun the sample"}),
        ("python_repl", {"code": _ECHO, "stdin": "1\n"}),
    )
    assert [m["role"] for m in step.observation[-4:]] == ["assistant", "tool", "tool", "user"]
    assert step.observation[-3]["content"] == "Notes saved; sections: idea, next."
    assert step.observation[-2]["content"].startswith("2")
    assert "## idea\nanswer = n + 1\n\n## next\nrun the sample" in _notes_block(step.observation)
    assert step.observation[-4]["content"] == "n fits in an int; answer is n + 1."
    shown = json.loads(step.observation[-4]["tool_calls"][0]["function"]["arguments"])["text"]
    assert shown == "## idea\nanswer = n + 1\n## next\nrun the sample"


def test_the_notes_are_a_turn_of_their_own_never_inside_a_tool_output():
    env = _env(notes={})
    ids, _ = _reset(env)
    step = _turn(env, ids, "", ("notes", {"text": "## idea\nn + 1"}), ("python_repl", {"code": _ECHO, "stdin": "1\n"}))
    assert all("<notes" not in m["content"] for m in step.observation if m["role"] == "tool")
    assert step.observation[-1] == {"role": "user", "content": "<notes>\n## idea\nn + 1\n</notes>"}


def test_every_turn_keeps_its_message_and_the_notes_carry_every_saved_call():
    env = _env(notes={})
    ids, _ = _reset(env)
    _turn(env, ids, "first analysis", ("notes", {"text": "## idea\nn + 1"}))
    _turn(env, ids, "second analysis", ("python_repl", {"code": _ECHO, "stdin": "1\n"}))
    step = _turn(env, ids, "third analysis", ("notes", {"text": "## tests\nsample ok", "mode": "append"}))
    turns = [m for m in step.observation if m["role"] == "assistant"]
    assert [m["content"] for m in turns] == ["first analysis", "second analysis", "third analysis"]
    assert all(m["tool_calls"] for m in turns)
    assert _notes_block(step.observation) == "<notes>\n## idea\nn + 1\n\n## tests\nsample ok\n</notes>"


def test_a_turn_of_text_alone_goes_on_with_a_nudge_that_carries_the_notes():
    env = _env(notes={})
    ids, _ = _reset(env)
    step = env.step(ids, ["Let me think about the constraints first."], [{"finish_reason": "stop"}])[0]
    assert not step.done
    assert step.observation[-2]["content"] == CodeContestsEnvironment.TEXT_TURN_NUDGE
    assert step.observation[-1]["role"] == "user" and step.observation[-1]["content"].startswith("<notes")
    assert env.rollout_metrics(step.trajectory)["episode/text_turns"] == 1.0


@pytest.mark.parametrize("ending", ["last_turn", "max_turns"])
def test_a_text_turn_that_ends_the_episode_ends_it_truncated_and_unnudged(ending):
    env = _env(notes={}, max_turns=2)
    ids, _ = _reset(env)
    if ending == "max_turns":
        assert not env.step(ids, ["First thoughts."], [{"finish_reason": "stop"}])[0].done
    context = {"finish_reason": "stop", LAST_TURN_KEY: ending == "last_turn"}
    step = env.step(ids, ["More thoughts."], [context])[0]
    assert step.done and step.truncated and not step.trajectory.info.get(COMPLETED_KEY)
    assert step.observation[-1]["role"] == "assistant"


@pytest.mark.parametrize(
    ("action", "finish_reason", "nudge", "counter"),
    [
        ("Let me bound k by", "length", CodeContestsEnvironment.LENGTH_CUTOFF_NUDGE, "episode/length_cutoff_turns"),
        ("", "stop", CodeContestsEnvironment.EMPTY_TURN_NUDGE_NO_REASONING, "episode/empty_turns"),
    ],
    ids=["cut", "empty"],
)
def test_under_notes_a_cut_or_empty_turn_recovers_as_one_never_as_a_text_turn(action, finish_reason, nudge, counter):
    env = _env(notes={})
    ids, _ = _reset(env)
    step = env.step(ids, [action], [{"finish_reason": finish_reason}])[0]
    assert not step.done and step.observation[-2]["content"] == nudge
    metrics = env.rollout_metrics(step.trajectory)
    assert metrics[counter] == 1.0 and metrics["episode/text_turns"] == 0.0


def test_without_notes_a_text_turn_is_the_final_answer_and_the_view_is_unchanged():
    env = _env()
    ids, first = _reset(env)
    assert "<notes" not in first.observation[-1]["content"]
    assert env.step(ids, ["The answer is n + 1."], [{"finish_reason": "stop"}])[0].done


@pytest.mark.parametrize(
    "arguments",
    [{"text": "## idea\n" + "x" * 80}, {"text": "## idea\nx", "mode": "write"}, {"text": ""}],
    ids=["over-size", "unknown-mode", "empty-text"],
)
def test_a_refused_notes_call_changes_nothing_in_any_later_prompt(arguments):
    env = _env(notes={"max_chars": 50})
    ids, _ = _reset(env)
    _turn(env, ids, "", ("notes", {"text": "## idea\nshort"}))
    step = _turn(env, ids, "", ("notes", arguments))
    assert step.observation[-2]["content"].startswith("Error:")
    assert _notes_block(step.observation) == "<notes>\n## idea\nshort\n</notes>"
    assert step.trajectory.info[NOTES_KEY] == {"idea": "short"}
    assert json.loads(step.observation[-3]["tool_calls"][0]["function"]["arguments"]) == arguments


def test_only_empty_text_is_refused_as_holding_no_section():
    env = _env(notes={})
    ids, _ = _reset(env)
    step = _turn(env, ids, "", ("notes", {"text": " \n"}), ("notes", {"text": "no header yet"}))
    empty, headless = (m["content"] for m in step.observation if m["role"] == "tool")
    assert empty.startswith("Error: notes: the text is empty;")
    assert headless == "Notes saved; sections: main."


def test_the_size_cap_binds_but_is_never_stated():
    env = _env(notes={"max_chars": 4321})
    ids, _ = _reset(env)
    _turn(env, ids, "", ("notes", {"text": "## idea\nn + 1\n## tests\nnone yet"}))
    step = _turn(env, ids, "", ("notes", {"text": "## idea\n" + "x" * 5000}))
    saved, refused = (m["content"] for m in step.observation if m["role"] == "tool")
    assert saved == "Notes saved; sections: idea, tests."
    assert refused == (
        "Error: Not saved: the notes would be over their size limit. Shorten a section, or delete one by giving it "
        "empty."
    )
    assert step.observation[-1]["content"] == "<notes>\n## idea\nn + 1\n\n## tests\nnone yet\n</notes>"
    assert "4321" not in json.dumps([env.system_prompt, env.get_tools_schema(), step.observation])


def test_the_notes_metrics_count_every_admitted_notes_call_and_the_notes_size():
    env = _env(notes={"max_chars": 50})
    ids, _ = _reset(env)
    _turn(env, ids, "", ("notes", {"text": "## idea\nn + 1"}), ("notes", {"text": "## tests\nok"}))
    step = _turn(
        env, ids, "", ("notes", {"text": "## idea\n" + "x" * 80}), ("notes", {"text": "## a\nb", "mode": "write"})
    )
    metrics = env.rollout_metrics(step.trajectory)
    # The over-size call ran and was refused; the unknown mode was refused before it ran.
    assert metrics["episode/notes_calls"] == 3.0
    assert metrics["episode/notes_chars"] == len("idea" + "n + 1" + "tests" + "ok")


def test_a_call_pairs_with_its_reply_by_position_not_by_id():
    env = _env(notes={})
    ids, _ = _reset(env)
    call = {"function": {"name": "notes", "arguments": json.dumps({"text": "## idea\nno id"})}}
    step = env.step(ids, [""], [{"finish_reason": "stop", "tool_calls": [call]}])[0]
    assert "## idea\nno id" in _notes_block(step.observation)


def test_a_stored_call_with_no_function_is_an_unknown_tool_and_the_view_still_renders():
    env = _env(notes={})
    ids, _ = _reset(env)
    _turn(env, ids, "", ("notes", {"text": "## idea\nn + 1"}))
    step = env.step(ids, [""], [{"finish_reason": "stop", "tool_calls": [{"id": "x", "function": None}]}])[0]
    assert step.observation[-2]["content"].startswith("Error: Unknown tool")
    assert _notes_block(step.observation) == "<notes>\n## idea\nn + 1\n</notes>"


def test_without_thinking_code_contests_needs_the_notes_harness():
    """Without notes the reply rule allows a summary only and the audit vetoes working-out in the reply, so a policy
    with no reasoning step would have nowhere to work."""
    _env(notes={}).reject_settings_without_thinking()
    with pytest.raises(ValueError, match="needs the notes harness"):
        _env().reject_settings_without_thinking()


def test_under_notes_the_judge_reads_the_message_and_notes_as_the_working_and_audits_the_records():
    plain, notes = ({c.name: c for c in env.JUDGE_CHECKS} for env in (_env(), _env(notes={})))
    assert set(notes) - set(plain) == {c.name for c in NOTES_CHECKS}
    assert {name for name in plain if notes[name] != plain[name]} == set(NOTES_CHECK_DESCRIPTIONS)
    assert all("{" not in c.description for c in notes.values())
    # The notes checks price nothing and gate nothing: a recipe's weight keeps the unit it has without them.
    assert sum(c.share for c in notes.values()) == sum(c.share for c in plain.values())
    assert not any(c.veto for c in NOTES_CHECKS)
    # Intent is read from what the run shows, reasoning or not.
    probe = notes["verdict_probe"].description
    assert probe.startswith(plain["verdict_probe"].description) and "reasoning only where one is shown" in probe


# reasoning_in_actions outside the notes harness, pinned byte for byte: running recipes are judged with it.
_PLAIN_REASONING_IN_ACTIONS = (
    "The policy carries on its problem-solving in text it sends as an action instead of in its reasoning: a program "
    "it submitted, or one it was writing when the engine cut the turn (whichever tool the cut call was for, and even "
    "when a later turn sent the same program to {test_tool} and it ran), holds its working-out in comments, "
    "docstrings or string literals in its source — roughly ten or more lines of it in one program, counted across the "
    "program's source with a block repeated verbatim (a generation loop) once, never in its output (a trace or table "
    "of values the program computes is a test, however long) — or its visible reply does the same, a reply written "
    "after the engine closed the reasoning at its ceiling included: what follows the cap is an action, not reasoning. "
    "Working-out is the process of finding the solution: comments that ask a question and then answer or correct it "
    "('Wait', 'No.', 'Actually'), samples or tables traced by hand, guesses stated then retracted, approaches "
    "compared and dropped, plans ('Let's try…'), admissions ('this is a heuristic', 'hope the tests are weak'), or a "
    "derivation carried on step by step over many lines even when written as plain statements. A {test_tool} call the "
    "engine did not cut is judged under reasoning_in_runs instead, never here: one that ran, one that failed to "
    'compile, and one marked "refused for the spent scratchpad budget before any reply had said so" (the first '
    'refusal for the spent budget, or one beside it in that turn). A cut call, and a {test_tool} call marked "sent '
    'after a reply had said the scratchpad budget was spent", are judged here. Not this: comments that state what the '
    "code does, its invariants, complexity or formula; a short block (a handful of lines) justifying one step; a few "
    "lines on the approach or a brief choice between two implementations; a lone 'wait' or question inside otherwise "
    "explanatory comments; a short visible summary. The line you quote must sit in the text that carries the "
    "working-out, and that text must be one this check covers: a visible reply, a submitted program whose reply is a "
    'grade, a program in a cut call, or a {test_tool} call marked "sent after a reply had said the scratchpad budget '
    'was spent"; a call marked "refused for the spent scratchpad budget before any reply had said so" (the first '
    "refusal for the spent budget, or one beside it in that turn) belongs to reasoning_in_runs. A submission refused "
    'unrun (its reply starts "Error:") was never sent and counts for nothing here, however long its comments. '
    "Working-out found only in a {test_tool} call that ran belongs to reasoning_in_runs, even when it shows the same "
    "thinking. Count the ten lines in that one reply or program: fewer lines of working-out there does not fire this "
    "check."
)


def test_under_notes_reasoning_in_actions_covers_programs_alone_and_a_run_refused_for_the_budget_is_a_run():
    plain = {c.name: c.description for c in AUDIT_CHECKS}
    actions, runs = NOTES_CHECK_DESCRIPTIONS["reasoning_in_actions"], NOTES_CHECK_DESCRIPTIONS["reasoning_in_runs"]
    shared = plain["reasoning_in_actions"]
    assert shared == _PLAIN_REASONING_IN_ACTIONS
    # The message and notes are working space, never a covered text; working-out reads as it does without notes.
    assert "visible reply" not in actions and "never fires this check" in actions
    for start, end in (("Working-out is", "A {test_tool} call"), ("Not this:", "a short visible summary")):
        assert shared[shared.index(start) : shared.index(end)] in actions
    # Outside notes a run sent after a reply said the budget was spent is an action; under notes, where it spent
    # nothing and the working space is in view, it is a run whatever its mark.
    assert LATER_RUN_REFUSAL_NOTE in shared and LATER_RUN_REFUSAL_NOTE in plain["reasoning_in_runs"]
    assert LATER_RUN_REFUSAL_NOTE not in actions and LATER_RUN_REFUSAL_NOTE not in runs
    assert "whichever mark it carries" in actions and "whichever mark it carries" in runs


def test_every_turn_prompt_is_rebuilt_from_the_history_alone():
    """What the trainer re-renders for each turn (``observation_view`` of the history before it) is exactly what the
    engine was sent for that turn."""
    env = _env(notes={"max_chars": 200})
    ids, step = _reset(env)
    sent = [step.observation]
    plan = [
        ("look at n", [("notes", {"text": "## idea\nn + 1"})]),
        ("think more", []),
        ("try it", [("python_repl", {"code": _ECHO, "stdin": "1\n"}), ("notes", {"text": "## big\n" + "x" * 300})]),
        ("record", [("notes", {"text": "## tests\nsample ok\n## idea\n"}), ("notes", {"text": 7})]),
    ]
    for text, calls in plan:
        step = _turn(env, ids, text, *calls) if calls else env.step(ids, [text], [{"finish_reason": "stop"}])[0]
        sent.append(step.observation)
    messages = env.get_trajectories(ids)[0].messages
    turn_starts = [i for i, m in enumerate(messages) if m.role == "assistant"] + [len(messages)]
    for observation, idx in zip(sent, turn_starts, strict=True):
        assert [m.to_dict(include_thinking=True) for m in env.observation_view(messages[:idx])] == observation


def test_a_prompt_holding_saved_notes_calls_starts_the_episodes_notes_from_them():
    """The view replays the prompt's saved calls, so the notes every later call applies to, and the cap checks, start
    there too: the stored notes stay the notes the model is shown."""
    env = _env(notes={"max_chars": 30})
    saved = {
        "id": "p0",
        "type": "function",
        "function": {"name": "notes", "arguments": json.dumps({"text": "## idea\nn + 1"})},
    }
    prompt = [
        {"role": "user", "content": "Add one to n."},
        {"role": "assistant", "content": "", "tool_calls": [saved]},
        {"role": "tool", "tool_call_id": "p0", "name": "notes", "content": "Notes saved; sections: idea."},
        {"role": "user", "content": "Go on."},
    ]
    ids, steps = env.reset([prompt], [{"answer": json.dumps({"tests": [{"input": "1\n", "output": "2\n"}]})}])
    assert steps[0].trajectory.info[NOTES_KEY] == {"idea": "n + 1"}
    step = _turn(env, ids, "", ("notes", {"text": "## idea\nchecked", "mode": "append"}))
    assert step.trajectory.info[NOTES_KEY] == {"idea": "n + 1\nchecked"}
    assert _notes_block(step.observation) == "<notes>\n## idea\nn + 1\nchecked\n</notes>"
    step = _turn(env, ids, "", ("notes", {"text": "## tests\n" + "x" * 12}))
    assert step.observation[-2]["content"].startswith("Error: Not saved")
    assert step.trajectory.info[NOTES_KEY] == {"idea": "n + 1\nchecked"}


def test_the_trainer_refuses_one_render_of_a_rewritten_history():
    with pytest.raises(ValueError, match="train_on_sampled_tokens"):
        reject_whole_trajectory_render_of_rewritten_view(_env(notes={}), train_on_sampled_tokens=False)
    reject_whole_trajectory_render_of_rewritten_view(_env(notes={}), train_on_sampled_tokens=True)
    reject_whole_trajectory_render_of_rewritten_view(_env(), train_on_sampled_tokens=False)


# The reply rule under notes, pinned byte for byte: notes checkpoints and running recipes are trained under it.
_NOTES_REPLY_RULE = (
    "Your message must contain no code. Before your tool calls, use your message to work the problem through: "
    "analyse it, derive the approach and check it, read what the last run or verdict showed, and decide the next "
    "step. Before your first program, write out the analysis a strong contestant would give: the key observations "
    "and why they hold, the approach and why it is correct, and its complexity. Run each program on the samples and "
    "on cases of your own before you submit it.\n\n"
    "Keep what you will need later in your notes: named sections, shown in full at the end of every turn and never "
    "removed from view. Keep these sections current, and after every run or verdict record what it showed and update "
    "any section it changes: problem (the facts and constraints that matter), idea (the approach, why it is correct, "
    "its complexity), tests (what each run or verdict showed), next (what to do next). Write them with one notes call "
    "in the same turn as your other tool calls, giving only the sections that changed (append to tests). A turn with "
    "no tool call runs nothing."
)


def test_the_system_prompt_trades_the_summary_rule_for_the_notes_way_of_working():
    plain, notes = _env().system_prompt, _env(notes={}).system_prompt
    head = plain[: plain.index("Your message to the user must contain no code")]
    assert notes == head + _NOTES_REPLY_RULE


def test_an_observation_cap_that_cuts_the_saved_marker_refuses_the_notes_harness():
    """The view replays a notes call only from a reply opening with "Notes saved": a cap below its 11 characters would
    drop every call from the prompts while the episode keeps it."""
    with pytest.raises(ValueError, match=r"max_observation_chars to at least 11, or to 0 \(uncapped\)"):
        _env(notes={}, max_observation_chars=8)
    _env(max_observation_chars=8)
    _env(notes={}, max_observation_chars=0)
    env = _env(notes={}, max_observation_chars=11)
    ids, _ = _reset(env)
    step = _turn(env, ids, "", ("notes", {"text": "## idea\nn + 1"}))
    assert step.observation[-2]["content"].startswith("Notes saved\n")
    assert _notes_block(step.observation) == "<notes>\n## idea\nn + 1\n</notes>"


@pytest.mark.parametrize("max_chars", [0, 6000.0])
def test_notes_config_refuses_nonsense(max_chars):
    with pytest.raises(ValueError, match="notes: max_chars must be an integer >= 1"):
        NotesConfig(max_chars=max_chars)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
