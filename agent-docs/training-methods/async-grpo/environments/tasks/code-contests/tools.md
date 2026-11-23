# Tools

The model gets two tools (three under [notes](#notes)): a scratchpad that runs a program on an input it supplies, and `submit_solution`, the
only graded channel. Their per-episode budgets, `max_test_calls` and `max_submissions`, are set per
[effort level](README.md#reasoning-effort). Both tools:

- State no budget, in their description or in any reply. A refusal names what to do, never a number.
- Neither refuse nor price a program's comments or what a run showed; the [audit judge](reward.md#audit-judge)
  handles both.
- Cut a scratchpad run's output that would push its reply past `max_observation_chars` (`…[truncated N chars]`),
  so the notices after it survive; a submission's reply is cut at the cap from its end.

## Scratchpad

The scratchpad is `python_repl` when the run fixes `python`, else `run_code`.

- It runs a program in the grading sandbox on the `stdin` the call supplies (empty by default), never on the
  graded tests.
- Each call is one-shot: nothing a run writes survives into the next.
- A run gets the [per-test time limit](grading.md#time-limits) its language is graded at.
- The reply leads with any error (the compiler's first diagnostics, or a crash's signal and stderr tail), then
  stdout ([Using it from Python](../../sandbox.md#using-it-from-python)).
- A [sandbox fault](../../sandbox.md#sandbox-faults) ends the episode.
- An empty or whitespace-only `stdin` adds a note that none was passed, unless the build failed. The run still
  spends one of `max_test_calls`; it is never refused, since a self-test may embed its input.

## submit_solution

`submit_solution` grades a complete stdin/stdout program against the hidden tests ([Grading](grading.md)). There
is no fenced-code-block fallback.

- A submission that passes every hidden test ends the episode, as reaching `max_submissions` does. A later
  submission in the same turn is not graded.
- A program identical to one this episode already graded is not regraded: its reply repeats that grade under
  `Identical to submission k, already graded: …` (`episode/identical_resubmissions`). It is an ordinary submission
  otherwise: it spends one of `max_submissions`, pays the resubmission price, and its turn trains. Comments and
  trailing whitespace are no change. The offline re-grader gives it its slot.

## Refused calls

A call is refused when:

- it is past its budget: the scratchpad replies
  `Error: Not run: this task's scratchpad budget is spent. Submit your solution with submit_solution.`, and
  `submit_solution` past `max_submissions` in the same turn
  `Error: Not graded: this task's submission budget is spent.`;
- its arguments do not bind: an undeclared argument such as `language` on a single-language run
  (`Error: <tool>: unknown argument 'language'; …`), a list or object as `code` or `stdin`, a `language`
  outside the run's list, or a value ending in its own closing tag (a `stdin` ending `</stdin>`, see
  [Argument binding](../../protocols/native-tool-use.md#argument-binding));
- its program is [evidently in another language](#evident-language) than its `language` names.

A refused call pays `tool_error_penalty`, never `tool_success_reward`. It spends its turn but none of
`max_test_calls` or `max_submissions`, and grades nothing. A turn whose every call was refused or unknown is
[untrainable](../../../objective.md#untrainable-turns).

Not refusals:

- A call to a tool whose budget is 0 (`ToolDisabled`: the [`leaderboard`](evaluation.md#evaluation-protocols)
  scratchpad, `max_test_calls: 0`) gets the same reply and price, but its turn is not flagged.
- A submission after an accept in the same turn that the budget still admits is returned to its budget, neither
  paid nor charged. It counts in `episode/tool_calls`, but not as a successful call.

## Languages

`language` takes `python`, `cpp`, `c`, or a list the model picks from
([sandbox languages](../../sandbox.md#languages)).

- With a list, both tools take a required `language` argument, and each program is graded in the language its
  call names. The episode's last language is its `language` [metrics](../../../monitoring.md#logged-metrics)
  slice.
- With one language, the tools declare no `language` argument and refuse a call naming one.
- The tool descriptions name the toolchain (compile flags, Python version) on `local` and `bubblewrap`; `remote`
  states none.

### Evident language

With a language list, a program evidently written in another of its languages is not run:

```text
Error: Not run: this looks like cpp code sent with language "python"; send it again with language "cpp". No scratchpad run was spent.
```

`submit_solution` says `Not graded` instead. The call is [refused](#refused-calls): it pays `tool_error_penalty`,
spends no budget, grades nothing and leaves the `language` slice unchanged, and a turn of such calls alone is
untrainable, so no solve reinforces the wasted turn. The check (`find_evident_language` in `program_source.py`) acts
only when a Python parse and the program's textual marks point the same way; a program it misses runs as labeled.
A submission after an accept takes the post-accept reply whatever its label.

## Notes

`notes: {}` adds a `notes` tool and turns the episode into a notes harness, for a policy that works without a
reasoning step: each turn's message holds its working, and the notes hold what later turns need. `max_chars` caps the
notes' size ([default](README.md#configuration)).
`NotesConfig`, the tool and the view live in `halo/environments/tools/notes.py`.

- The notes are named sections, written in the format they are shown in: one call's `text` holds any number of
  sections, each a line `## <name>` and its text (text before the first header is section `main`). `mode: replace`
  (the default) sets each given section and deletes one given empty; `append` extends each. Sections left out stay.
  The reply names the sections the notes hold (`Notes saved; sections: idea, tests.`); a call past `max_chars`, or
  with no section, is [refused](#refused-calls) and changes nothing. The cap is enforced but never stated: no tool
  description, prompt, reply, refusal or notes block gives a size.
- Every turn's prompt ends with the whole notes as a user turn of their own (`<notes>…</notes>`), after the tool
  replies, never inside a tool's output; notes calls stay in the history as written. The replay pairs each call with
  its reply by position, replays only a call whose reply opens `Notes saved`, and reads it through the parsing and
  binding it ran under; a `max_observation_chars` that would cut that prefix (1 to 10) is refused at construction.
- The system prompt's brief-summary rule gives way to working in the message and keeping `problem`, `idea`, `tests` and
  `next` sections current.
- A turn of text alone does not end the episode: it is nudged to act. A text turn at `max_turns`, or when the
  [episode output budget](../../../episodes.md#episode-output-budget) affords no further turn, ends the episode
  truncated, which pays the overflow. Counters: [Behavior counters](reward.md#behavior-counters).
- The [audit judge](reward.md#audit-judge) reads the messages and the notes as the policy's working, beside its
  reasoning where one is shown (`NOTES_CHECK_DESCRIPTIONS`): `reasoning_in_actions` covers programs alone, a notes call
  is never a `notepad_run`, and `verdict_probe` reads intent from the messages and notes too. Two logged checks audit
  the notes, `false_record` and `unexplained_program`. The no-thinking recipes give the judge `notes-no-thinking.txt`
  as its [context](reward.md#audit-judge).
- The prompt is a function of the history alone (`observation_view`). Its notes turn moves to the end of every prompt,
  so no prompt extends the one before it and training needs `train_on_sampled_tokens: true`
  ([Single-row fallback](../../../objective.md#single-row-fallback)).

## Recovery nudges

Outside notes, a final text answer ends the episode ungraded, so the nudge after a cut or empty turn
([Unproductive turns](../../../episodes.md#unproductive-turns)) asks for a tool call and names `submit_solution`. A
turn cut while writing its call is asked to call again with its working kept out of the program's comments.
