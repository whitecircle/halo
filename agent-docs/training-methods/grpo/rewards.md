# Reward Terms

A GRPO reward is a sum of **terms**. Each term names a source of a score in `[0, 1]` and prices it as `weight × score ^ exponent`. Terms are listed under `rewards:` in the config and parsed into typed dataclasses at config time (`src/rewards/terms.py`), so a bad term fails before any server is touched. Both GRPO arms read the same list: the [online arm](online-grpo.md#rewards) turns every term into one TRL reward function, the [environment arm](async-grpo/README.md) composes them into the episode reward.

```yaml
rewards:
  - source: accuracy                 # online arm: the last \boxed{} equals the answer
  - source: judge
    name: quality
    weight: 0.3
    requirements:
      - {name: correctness, description: "The final answer is correct and fully justified."}
      - {name: clarity, description: "The solution is easy to follow.", weight: 0.5}
  - source: reward_model
    name: preference
    weight: 0.5
    url: http://localhost:8100
    model: Skywork/Skywork-Reward-V2-Llama-3.1-8B
```

Every term takes `name` (its metric key, `reward/<name>` on the environment arm, `rewards/<name>/mean` on the online arm; required for the scored sources, defaulted by the graders), `weight` (signed; negative makes a penalty) and `exponent` (`> 0`; above 1 is convex credit, so half a score earns under half the weight). Names must be unique. The `environment` term is the exception — it is always named `objective`; any other `name` raises. A constant offset is not a knob: it cancels in the group baseline.

| `source` | Score | Arm |
|---|---|---|
| `environment` | the environment's own grade (a solve on every hidden test, an answer match) | environments |
| `judge` | a generative judge: a score over `requirements`, or a veto over `checks` | both; a veto judge environments only |
| `reward_model` | a served Bradley-Terry or sequence-classification model | both |
| `accuracy`, `format` | the RLVR graders | online |

A `judge` or `reward_model` term is scored by an external backend once the completion is in. Two of the options common to both: `view`, what the scorer reads of the episode — `final` (the final answer alone), `full` (every turn) or `digest` (a compact account of every turn; judge only) — and `on_error`, what a verdict it never reached means: `invalid` (the default) voids the completion's verdict — an episode leaves the group baseline, a TRL row loses the term — while `neutral` prices the term at 0 and keeps the completion valid ([Outages](#outages)).

## Generative judge

An OpenAI-compatible chat model reads the task, one view of the episode and a rubric, and replies with one JSON object. A **scoring judge** lists `requirements`: each is scored from 0 to `scale`, and the term's score is their weight-averaged fraction of the scale. A **veto judge** lists `checks` instead ([Veto judge](#veto-judge)); a term lists exactly one of the two. Defaults: `openai/gpt-5.6-luna` on OpenRouter at `reasoning_effort: medium`, strict JSON-schema output, key from `OPENROUTER_API_KEY`.

| Option | Default | Effect |
|---|---|---|
| `requirements` | — | list of `{name, description, weight}`; `weight > 0`, default 1; makes a scoring judge |
| `checks` | — | list of `{name, description, veto}`; `veto` default `false`; makes a veto judge |
| `scale` | `10` | each requirement is scored `0..scale` |
| `model`, `base_url` | `openai/gpt-5.6-luna`, OpenRouter | any OpenAI-compatible endpoint, a local vLLM included |
| `api_key_env` | `OPENROUTER_API_KEY` | the variable holding the key; then `OPENROUTER_API_KEY`, `OPENAI_API_KEY` |
| `reasoning_effort` | `medium` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`; `null` sends no such field |
| `temperature` | unset | reasoning models refuse an explicit one |
| `view` | `final` | `final`, `full` or `digest` ([What a judge reads](#what-a-judge-reads)) |
| `include_reference` | `true` | shows the row's reference answer when the sample carries one |
| `include_reasoning` | `true` | shows the policy's reasoning in the `full` and `digest` views (the environment arm's turns carry it) |
| `on_error` | `invalid`; `neutral` for a veto judge | what a verdict the judge never reached means ([Outages](#outages)) |
| `max_tokens`, `request_timeout`, `max_concurrency` | `8192`, `120`, `16` | per request (reasoning tokens count against `max_tokens`); concurrency per scorer instance (per rank or actor) |
| `max_view_chars` | `60000` | cut applied to the view **and** to the reference answer, head and tail kept |
| `structured_output` | `true` | off, the reply is parsed as the first JSON object it contains |

### What a judge reads

The grading prompt is the task (the prompt's user turns), the tools the policy could call (one line each: name, parameters, the head of the description), the reference answer, the view of the episode, the rubric and the reply shape. The system prompt tells the judge to grade the policy's actions, answers and tool results, and to read its reasoning as context, never as the thing graded.

- `final` — the final answer as the protocol recorded it ([What a scorer reads](#what-a-scorer-reads)). An episode that ended without one shows `(The episode ended without a final answer.)` and the last assistant turn after it, so a cut-off fragment or a tool-call turn reads as what it is, never as the answer; a reward model reads an empty answer instead.
- `full` — every turn after the prompt, numbered (`[3] assistant`), the reasoning set apart in a `<reasoning>` block, each tool call as `→ name (call id)` with every argument verbatim on its own lines (a program as the policy wrote it), and each result under its own `[4] tool name (call id)` header. A turn the engine cut at its length limit, one that ended with neither text nor a tool call, and one whose every call named a tool that does not exist, was refused unrun or ran and showed nothing carry that note in their header. A turn cut while writing a call shows the partial call the engine's parser salvaged from it as `→ name (call id; cut by the engine before the turn closed; never run)` with its arguments, as context: the policy's later turns never see it, and a [veto](#veto-judge) never quotes it.
- `digest` — the same blocks with each item cut to its head (reasoning, assistant text, each tool-call argument, a cut call's included) or head and tail (tool results), then `Final answer:` whole, the artifact an audit reads.

A view past `max_view_chars` keeps two thirds of the budget from its head and the rest from its tail, marking the cut between them, so the answer or submission at the end of a long transcript is never what the cut removes. The reference is cut the same way. The renderers live in `src/rewards/samples.py`.

### Veto judge

A veto judge audits the process instead of scoring the answer. Each check either fires or does not, and a fired check must quote the span that shows it verbatim from the policy's actions the judge read — its visible text, tool calls and results, never its reasoning or the partial call of a cut turn, so neither a thought nor a call that never ran is punished: a quote not found there (whitespace folded, at most 400 characters) does not fire, and `judge/<name>/unsupported_flags` counts the dropped ones. A fired `veto: true` check strips the episode of every credit: `reward/objective` and every other positive component go to 0, the penalties stand. The term's own score is the fired fraction of its non-veto checks, so its `weight` must be `<= 0` — `0` by default, which logs the flags without pricing them; a negative weight charges them. A config listing `checks` defaults `on_error` to `neutral`, so a judge outage leaves the environment's exact grade standing. A veto judge needs an `environment` term in the same reward (there is nothing else to gate), and the online arm refuses one at config time (`list 'requirements' instead`): its independent TRL reward functions have no objective to gate.

```yaml
rewards:
  - source: environment
  - source: judge
    name: audit
    view: digest
    checks:
      - {name: hardcoded_output, description: "The submitted program prints answers for specific inputs instead of computing them.", veto: true}
      - {name: environment_probe, description: "The policy inspected the sandbox, the tests or the grader instead of solving the problem.", veto: true}
      - {name: blind_resubmission, description: "A resubmission changed nothing material in the previous program."}
```

### Metrics

Per completion, averaged per step on both arms (the online arm logs 0 on a rank where no sample carried a key): `judge/<name>/<requirement>` (the score as a fraction of the scale), `judge/<name>/completion_tokens`, and `judge/<name>/scored` — the share of completions the judge reached a verdict on, the metric that says a judge is failing. A veto judge logs `judge/<name>/<check>` (fired, 1/0), `judge/<name>/veto` and `judge/<name>/unsupported_flags` instead of requirement scores. The environment arm also logs `judge/<name>/cost_usd` where the endpoint reports a cost (OpenRouter does) and keeps the rationale, with each fired check's quote, on the trajectory (`reward_details`).

### Probe and retries

Both arms probe every external term at launch with a one-line sample in the run's exact request shape (rank 0, verdict broadcast), so a bad URL, key, model or parameter fails the launch instead of every episode. `request_timeout` bounds each attempt, not the term's whole call: the client retries a request up to four times, and a reply that carries no choices with an upstream 408, 429, 500, 502, 503 or 504 (how OpenRouter reports a rate-limited model) is retried four more times at 2, 4, 8 and 16 s (`chat_completion`, `src/inference/openai_client.py`), so one stuck judge call can hold a step for several minutes; lower the timeout before the retries.

## Served reward model

A `*ForSequenceClassification` or `*ForRewardModel` checkpoint served by the rollout engines' pooling routes. The scorer renders the conversation with the model's own chat template (`tokenizer`, default `model`) and tokenizes it without added special tokens — the conversation itself, which the online script keeps beside the rendered prompt TRL generates from — then maps the head's logit through `sigmoid((logit − logit_shift) / logit_scale)`.

| Backend | Serve | Route |
|---|---|---|
| `vllm` | `vllm serve <model> --runner pooling` (`--convert classify` for a decoder checkpoint vLLM does not map to a classification head) | `POST /classify` with the rendered text, `add_special_tokens: false`, `use_activation: false`; reads `probs[label_index]` |
| `sglang` | `python -m sglang.launch_server --model-path <model> --is-embedding` | `POST /classify` with the token ids; reads `embedding[label_index]` |

`url` is the server root, without `/v1`. `backend` is `vllm` (or `sglang`), `label_index` `0`, `logit_shift` `0.0`, `logit_scale` `1.0`, `on_error` `invalid`, and `view` `final` — the prompt plus one assistant message holding the final answer; `full` renders every turn, and `digest` is refused (a chat template renders turns, not a summary). `batch_size` (8) rows per request, `max_concurrency` (4) requests in flight per scorer, `request_timeout` 60 s. The raw logit logs as `reward_model/<name>/logit`, the verdict rate as `reward_model/<name>/scored`. The online arm also keeps TRL's in-process path, reachable through the trainer API and not the YAML (a config lists `rewards:` terms): a model id in `reward_funcs` loads a sequence-classification head on every rank, which costs trainer memory; the served route does not.

## Outages

A scorer never raises for a sample: a failed request, an unparseable reply or a prompt it could not build comes back as a verdict-less result, and the term's `on_error` decides what that means. `<source>/<name>/scored` is the per-completion 1/0 either way.

| `on_error` | Online arm (TRL row) | Environment arm (episode) |
|---|---|---|
| `invalid` (default) | the term returns `None`, which TRL sums as 0 beside the row's other terms; the row leaves the baseline only when every term returned `None` | the term prices 0 and the episode is marked `episode_invalid`, out of the group baseline, the scorer's error as its `episode_invalid_reason`; the eval runner reports it as an error row |
| `neutral` (default for a veto judge) | the term returns 0 | the term prices 0 and the episode stays valid, training on its other terms |

## Environment arm

An environment grades, the terms price. `_grade_episode(trajectory, context)` returns an `EpisodeGrade`: `objective`, the environment's score in `[0, 1]` (a solve on every hidden test, an answer match), and `shaping`, its own episode-level terms by bare name. The episode reward is the sum of its `reward/*` components:

- `reward/turn_shaping` — the per-turn deltas accrued during the episode (tool credit and penalties, ReAct thought credit).
- `reward/tool_shaping` — native-protocol environments only: their episode-level knobs (`no_tool_use_penalty`, `turn_overflow_penalty`, `length_cutoff_penalty`), from `_episode_shaping`.
- `reward/<name>` — each of the environment's shaping terms (code contests: `submission`, `resubmission`).
- `reward/objective` — `weight × grade ^ exponent` from the `environment` term; 0 once a veto check fired, as is every other positive component.
- `reward/<name>` — each `judge` / `reward_model` term.

The trainer charges its length terms on top, outside these components: `reward/reasoning_floor` and, with the price on, `reward/reasoning_price` ([Reasoning length reward](async-grpo/rollouts.md#reasoning-length-reward)). Each episode's settled components and every scored term's rationale are also saved with its completion ([Saving trajectories](async-grpo/rollouts.md#saving-trajectories)).

`rewards` defaults to `[{source: environment}]` and reaches the environment constructor as `reward_terms`. A class declares its shaping names in `SHAPING_COMPONENTS` (the union over the class hierarchy); `turn_shaping` and every declared shaping name are reserved for shaping, and a reward term may not take one. There is no failure offset: a constant cancels in the group baseline. An environment whose grade carries no signal (a null `answer` cell, a grader outage, a code grade that stopped before any test failed) grades 0 and marks the episode `episode_invalid`, out of the baseline.

### What a scorer reads

One sample serves every external term, each reading its own view of it: the prompt turns (everything before the first assistant turn), the policy's turns after them with their reasoning and the flags of a turn the engine cut, an empty one, or one whose calls were all rejected, the calls a cut turn never ran (`Message.cut_tool_calls`; the generative judge's `full` and `digest` views render them, the chat-message form a reward model reads drops them), the final answer, the reference and the environment's tool schemas. Two hooks on `BaseEnvironment` fill the episode-specific parts:

- `_final_answer(trajectory)` — what the episode delivered, or `None` when it did not complete. The native protocol records its `final_response`, ReAct its `Final Answer`, code contests the submitted program as a fenced code block; the base returns `None`.
- `_scoring_reference(trajectory)` — the reference as the grader compares it: `context["answer"]` by default; `exam_qa` hands the choice letter, not the index; code contests hands `None`, so the hidden tests never reach a judge.

### Settlement

The external terms are scored after the episode ends. At the terminal step the environment prices its own side and marks the reward pending; the episode dispatcher — the Ray actor and the eval runner both drive through it — awaits `settle_async` for every episode it closed, and a sync caller uses `env.settle(ids)`. Reading `rollout_metrics` on an unsettled episode raises. Settlement prices each scored term, zeroes `reward/objective` and every other positive component when a veto check fired, and books a verdict-less term by its `on_error` ([Outages](#outages)); `episode/reward_scored` is 1 only when every external term reached a verdict, whichever way a miss was settled.

An episode the **eval** driver lost (a generation that failed past its retries) is graded on what it earned and never sent to a scorer; unless its own request caused the failure, the eval then discards that grade and reports the sample as a [generation error](environments/evaluation.md#running-an-evaluation). The training actor instead drops the partial trajectory and returns an error row at reward 0. An episode a [sandbox fault](environments/sandbox.md#sandbox-faults) ended is never sent to a scorer either, and logs no `episode/reward_scored`. The launch probe covers every external term through the environment's `verify_backend`. One scorer per environment instance, so judge concurrency is `max_concurrency` per Ray actor.

Writing one: [Custom Environments](environments/custom-environments.md).

## Failure and cost

Under tensor parallelism every TP rank scores its copy of the group and the leader's tensors are broadcast, so verdict noise cannot diverge the update, but judge calls scale with `tp_size`. Judge cost is per graded completion and per eval sample; the completion-token and cost metrics above are the budget to watch.
