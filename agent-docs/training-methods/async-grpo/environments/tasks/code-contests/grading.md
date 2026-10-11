# Grading

`submit_solution` grades through `grade_solution` (`src/environments/envs/tasks/coding/grading.py`). The offline
re-grader shares it, so the same rules grade online and offline; the re-grader alone stops at the first failing test
and runs without `max_grading_seconds` ([Re-grading](../../../evaluation.md#re-grading-recorded-trajectories)). How the grade is paid:
[Reward ladder](reward.md#reward-ladder).

## Grading rules

- One sandbox session serves the whole grade: on `local` and `bubblewrap` a compiled submission builds once, and
  the session resets after every test.
- The verdict lists non-passing tests only, one entry per distinct verdict with the tests that failed the same way
  folded in, at most five.

### Build failures

A compile failure is graded once against the whole pool, as is Python source that does not compile on `local`
and `bubblewrap`.

| Backend | Compile error, `outcome` | Compile error, `full` | Python syntax error |
|---|---|---|---|
| `bubblewrap` | Compiler's first error | Compiler's first error | First error in both modes, no run |
| `local` | Verdict class alone | Compiler's first error | First error in both modes, no run |
| `remote` | Verdict class alone | Compiler's first error | Runs; graded as the service reports it |

- `bubblewrap` shows the error: its one build runs before any test, on an empty stdin.
- `local` hides it: a program can force a rebuild that reads a test's input.
- `remote` hides it: its build shares each test's request. It may run another Python version, so Python source
  runs there.
- A Python syntax error quotes the source alone (`SyntaxError: invalid syntax (line 3)`).

### Comparison

- `exact` comparison spuriously fails correct Codeforces solutions, hence the `codeforces` preset.
- Token comparison accepts a real-valued token within `1e-6 × max(1, |expected|)`, only when the expected token
  looks like a float, so integer answers stay exact.

### Verdict detail

`outcome` shows each failed test's verdict class alone: `FAIL`, `RUNTIME ERROR`, `TIME LIMIT EXCEEDED`,
`OUTPUT LIMIT EXCEEDED`, `COMPILATION ERROR` ([build failures](#build-failures)), or `ERROR` for a test lost to
infra (its text goes to the log). Stderr, exit status and output size are withheld because each can carry the
hidden input. `stop_on_first_failure` narrows what reaches the policy to the first failure, the Codeforces contract.

`full` adds a runtime error's signal or exit status and stderr tail, an output overrun's size against its cap, an
infra error's text, and a wrong answer's expected and produced output. A second submission then becomes a free test
oracle, and probing out-earns scratchpad testing.

Scratchpad runs on the model's own inputs show their output in both modes.

### Time limits

- The per-test cap is the payload's `time_limit`, else `timeout_per_test`. An interpreted language is floored at
  `timeout_per_test`, so a C++-tuned limit cannot fail a correct CPython solution. A compiled one is scaled by
  `compiled_time_limit_scale`. Both are clamped to `max_time_limit`.
- A list mixing interpreted and compiled languages states the floor, scale and clamp in the system prompt.
- A payload without `time_limit` gets each language's limit appended to the task message
  (`Each test of this problem runs under 5 s in python, 10 s in cpp.`).
- On `local` / `bubblewrap` a compiled program's stack is raised to the memory limit, as on Codeforces, up to the
  process's hard stack limit ([Limits](../../sandbox.md#limits)).

### Grading budget

Tests run sequentially, so a problem with hundreds of tests stalls the round. `max_grading_seconds` bounds one
grade, checked between tests. Size it for an honest solution: an ungraded test never counts as passed.

- Stopped after a failed test: a wrong answer.
- Stopped before any failure: grade 0, and the episode is marked invalid, out of the group baseline
  (`episode/grade_inconclusive`).
- `episode/tests_graded_frac` shows a partial grade.

### Special judges

A per-problem `checker` (Python) in the payload overrides comparison:
`python checker.py input.txt correct_output.txt solution_output.txt`. The submission is accepted only when the
checker exits cleanly and its last stdout token is `1`. The checker runs at the 15 s infra default, never the
solution's limit.

### Infra errors

- Tests lost to the backend with none passed and none failed mark the episode invalid, out of the group baseline
  (`episode/grading_infra_outage`).
- Lost tests beside passes and no failure leave the grade inconclusive, as a budget stop does.
- Verdicts, not infra: a build past the compile limit (compile error), a program that replaces its working
  directory (runtime error), one that floods its output (output-limit or runtime error). A program that removes
  its working directory runs the next test in a fresh one.

Other routes into an infra error: [Sandbox faults](../../sandbox.md#sandbox-faults).
