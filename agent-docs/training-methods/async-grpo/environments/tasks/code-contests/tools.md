# Tools

The model gets two tools: a scratchpad that runs a program on an input it supplies, and `submit_solution`, the
only graded channel. Their per-episode budgets, `max_test_calls` and `max_submissions`, are set per
[effort level](README.md#reasoning-effort). Both tools:

- State no budget, in their description or in any reply. A refusal names what to do, never a number.
- Neither refuse nor price a program's comments or what a run showed; the [audit judge](reward.md#audit-judge)
  handles both.
- Cut output that would push a reply past `max_observation_chars` (`…[truncated N chars]`), so the notes after it
  survive.

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
- A program identical to one this episode already graded is refused unrun and returned to its budget
  (`episode/identical_resubmissions`); comments, trailing whitespace and blank lines are no change. The offline
  re-grader does not count it either.

## Refused calls

A call is refused when:

- it is past its budget: the scratchpad replies
  `Error: Not run: this task's scratchpad budget is spent. Submit your solution with submit_solution.`, and
  `submit_solution` past `max_submissions` in the same turn
  `Error: Not graded: this task's submission budget is spent.`;
- its arguments do not bind: an undeclared argument such as `language` on a single-language run
  (`Error: <tool>: unknown argument 'language'; …`), a list or object as `code` or `stdin`, or a `language`
  outside the run's list;
- it resubmits a graded program: `Error: Not graded: this program is identical to one already graded …`.

A refused call pays `tool_error_penalty`, never `tool_success_reward`. It spends its turn but none of
`max_test_calls` or `max_submissions`, and grades nothing. A turn whose every call was refused or unknown is
[untrainable](../../../objective.md#untrainable-turns), and the next turn runs on the
[retry reserve](../../../reasoning.md#retry-reserve).

Not refusals:

- A call to a tool whose budget is 0 (`ToolDisabled`: the [`leaderboard`](evaluation.md#evaluation-protocols)
  scratchpad, `max_test_calls: 0`) gets the same reply and price, but its turn is not flagged and the next gets no
  retry reserve.
- A call returned to its budget (a program sent with the [wrong language](#evident-language), a submission after
  an accept that the budget still admits) is neither paid nor charged. It counts in `episode/tool_calls`, but not
  as a successful call.

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
Not run: this looks like cpp code sent with language "python"; send it again with language "cpp". No scratchpad run was spent.
```

`submit_solution` says `Not graded` instead. The call is returned to the budget: no charge, no grade, and the
`language` slice unchanged. The check (`evident_language` in `code_contests.py`) acts only when a Python parse and
the program's textual marks point the same way; a program it misses runs as labeled.

## Recovery nudges

A final text answer ends the episode ungraded, so the nudge after a cut or empty turn
([Unproductive turns](../../../episodes.md#unproductive-turns)) asks for a tool call and names `submit_solution`. A
turn cut while writing its call is asked to call again with its reasoning kept out of the program's comments.
