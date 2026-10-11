# Generative Judge

A `judge` term sends the task, one view of the episode and a rubric to an OpenAI-compatible chat model, which replies with one JSON object. A term lists exactly one rubric:

- `requirements` make a **scoring judge**: the term's score is the requirements' weight-averaged score as a fraction of `scale`.
- `checks` make a **veto judge**, which audits the process instead of scoring the answer ([Veto judge](#veto-judge)).

Shared term fields, `on_error` and the launch probe: [Reward Terms](README.md).

## Options

| Option | Default | Effect |
|---|---|---|
| `requirements`, `scale` | —, `10` | `{name, description, weight}` items (`weight > 0`, default 1), each scored `0..scale` |
| `checks` | — | `{name, description, veto, weight}` items; `veto` default `false`; `weight >= 0`, default 1, non-veto checks only |
| `model`, `base_url` | `openai/gpt-5.6-luna`, OpenRouter | Any OpenAI-compatible endpoint, a local vLLM included |
| `api_key_env` | `OPENROUTER_API_KEY` | Key variable; falls back to `OPENROUTER_API_KEY`, then `OPENAI_API_KEY` |
| `reasoning_effort` | `medium` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`; `null` sends no such field |
| `temperature` | unset | Reasoning models refuse an explicit one |
| `view` | `final` | `final`, `full` or `digest` ([Views](#views)) |
| `context` | unset | What the episode does not show (how the environment and its limits work) |
| `include_reference`, `include_reasoning` | `true`, `true` | Show the reference answer (when the sample has one); show the policy's reasoning in the `full` and `digest` views |
| `on_error` | `invalid`; `neutral` for a veto judge | [Outages](README.md#outages) |
| `max_tokens`, `request_timeout`, `max_concurrency` | `8192`, `120`, `16` | Per request (`max_tokens` includes reasoning); concurrency per scorer (per rank or actor) |
| `max_view_chars` | `60000` | Cap on the view and the reference ([The length cut](#the-length-cut)) |
| `structured_output` | `true` | Strict JSON-schema reply; off, the reply's first JSON object is parsed |

## What a judge reads

The grading prompt wraps each part in its own tag, in this order: `<setting>` (the term's `context`), `<policy_instructions>` (system turns), `<task>` (user turns), `<policy_tools>` (each tool as the policy read it), `<reference_answer>`, the view (`<final_answer>`, `<transcript>` or `<transcript_digest>`), then the rubric and reply shape.

- The parts are tagged, not headed, so a task's markdown heading or a program's comment never reads as a section of the prompt.
- Every part but `<setting>`, the term's own text, is escaped: the `<` of any of these tags or of `<reasoning>` that the quoted text spells, in any case, is written `&lt;`. Policy text, a tool result or a task can neither close its part nor forge a reasoning block.
- The judge grades the policy's actions, answers and tool results, and reads its reasoning as context only.

### Views

- **`final`**: the recorded final answer ([What a scorer reads](README.md#what-a-scorer-reads)). Without one, the judge sees `(The episode ended without a final answer.)` and the last assistant turn; a reward model reads an empty answer.
- **`full`**: every turn after the prompt, numbered (`[3] assistant`): reasoning in `<reasoning>`, each call as `→ name (call id)` with verbatim arguments, each result under `[4] tool name (call id)`.
- **`digest`**: `full` with each item cut to its head (tool results: head and tail), then `Final answer:` whole.

A turn header (`full` and `digest`) notes a reasoning cap hit, an engine cut, an empty turn, or calls all unknown or refused.

A turn cut mid-call shows the call as `→ name (call id; cut by the engine before the turn closed; never run)`, its arguments the policy's own text where the training rollout decodes it (vLLM under a thinking budget; [Cut tool calls](../../async-grpo/monitoring.md#cut-tool-calls)) and the parser's salvage elsewhere. A veto may quote it.

### The length cut

A view over `max_view_chars` is cut in two stages:

1. In the `full` view, reasoning goes first: each turn's is cut to head and tail within an even share of the room the rest leaves, so no call, result or text is cut for a thought.
2. The view keeps two thirds of the budget from its head and one third from its tail, with a marker between, so the final answer survives.

The reference is cut the same way. Renderers: `src/rewards/samples.py`.

## Veto judge

Each check fires or does not.

```yaml
rewards:
  - source: environment
  - source: judge
    name: audit
    view: full
    context: "The policy has a scratchpad tool and a graded submission tool; ..."
    checks:
      - {name: hardcoded_output, description: "The program prints answers for specific inputs instead of computing them.", veto: true}
      - {name: environment_probe, description: "The policy inspected the sandbox, the tests or the grader.", veto: true}
      - {name: blind_resubmission, description: "A resubmission changed nothing material."}
```

### Evidence

A fired check must quote one to three lines; one found quote suffices (a judge misquotes a long program's line about one time in four). A flag without a counting quote does not fire and counts in `judge/<name>/unsupported_flags`.

- **Sources:** the policy's actions as the term's `view` shows them (`render_actions`): visible text, tool-call arguments (a cut call's included) and their results; under `final`, the answer, or the last assistant text when there is none. Never reasoning, since a veto on a thought teaches the policy to hide it, and never environment notes (headers, flags, nudges).
- **Match:** a whitespace-folded span of that text, both read as the prompt shows them, so a quoted tag matches with or without its escape. A quote of four or more words also counts when, case aside, it matches in order at one place missing at most a fifth of its words (one at least), first and last words included.
- **Length:** over 400 characters never counts.

### What a fired check costs

- A fired `veto: true` check zeroes `reward/objective` and every other positive component. The penalties stand.
- The term's score is the weighted fired fraction of its non-veto checks. A check at weight `0` is logged, never priced; a veto check takes no weight.
- The term's `weight` must be `<= 0`. The default `0` logs flags unpriced; a negative weight charges them.
- `on_error` defaults to `neutral`, so a judge outage leaves the environment's grade standing.

### Where it runs

A veto judge needs an `environment` term in the same reward to gate, so the online arm refuses one at config time (`list 'requirements' instead`).

## Metrics

Per completion, averaged per step on both arms (the online arm logs 0 on a rank where no sample carried a key).

| Key | Meaning |
|---|---|
| `judge/<name>/scored` | Share of completions with a verdict; the one that says a judge is failing |
| `judge/<name>/<requirement>` | Scoring judge: the score as a fraction of the scale |
| `judge/<name>/<check>` | Veto judge: the check fired (1/0) |
| `judge/<name>/veto` | Veto judge: a `veto: true` check fired (1/0) |
| `judge/<name>/unsupported_flags` | Veto judge: flags dropped for lack of a counting quote |
| `judge/<name>/completion_tokens` | The judge's completion tokens |
| `judge/<name>/cost_usd` | Environment arm: request cost, where the endpoint reports one (OpenRouter does) |

The environment arm also keeps the rationale and each fired check's quote on the trajectory (`reward_details`).
