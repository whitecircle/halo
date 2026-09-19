# Reward Terms

A GRPO reward is a sum of **terms**. Each term names a source of a score in `[0, 1]` and prices it as `weight × score ^ exponent`. Terms are listed under `rewards:` in the config and parsed into typed dataclasses at config time (`src/rewards/spec.py`), so a bad term fails before any server is touched. Both GRPO arms read the same list: the [online arm](online-grpo.md#rewards) turns every term into one TRL reward function, the [environment arm](async-grpo/README.md) composes them into the episode reward.

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

Every term takes `name` (its metric key, `reward/<name>` on the environment arm, `rewards/<name>/mean` on the online arm), `weight` (signed; negative makes a penalty) and `exponent` (`> 0`; above 1 is convex credit, so half a score earns under half the weight). Names must be unique. The `environment` term is the exception — it is always named `objective`; any other `name` raises. A constant offset is not a knob: it cancels in the group baseline.

| `source` | Score | Arm |
|---|---|---|
| `environment` | the environment's own grade (pass fraction, answer match, adherence) | environments |
| `judge` | a generative judge over the requirements below | both |
| `reward_model` | a served Bradley-Terry or sequence-classification model | both |
| `accuracy`, `format` | the RLVR graders | online |

## Generative judge

An OpenAI-compatible chat model reads the task, the response and the rubric, and replies with one JSON object of integer scores, one per requirement, from 0 to `scale`. The term's score is the weight-averaged fraction of the scale. Defaults: `openai/gpt-5.6-luna` on OpenRouter at `reasoning_effort: medium`, strict JSON-schema output, key from `OPENROUTER_API_KEY`.

| Option | Default | Effect |
|---|---|---|
| `requirements` | required | list of `{name, description, weight}`; `weight > 0`, default 1 |
| `instructions` | none | grading guidance appended to the rubric |
| `scale` | `10` | each requirement is scored `0..scale` |
| `model`, `base_url` | `openai/gpt-5.6-luna`, OpenRouter | any OpenAI-compatible endpoint, a local vLLM included |
| `api_key_env` | `OPENROUTER_API_KEY` | the variable holding the key; then `OPENROUTER_API_KEY`, `OPENAI_API_KEY` |
| `reasoning_effort` | `medium` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`; `null` sends no such field |
| `temperature` | unset | reasoning models refuse an explicit one |
| `transcript` | `final` | `final` grades the last assistant message; `full` shows every turn with tool calls and results |
| `include_reference` | `true` | shows the row's reference answer when it has one |
| `max_tokens`, `request_timeout`, `max_concurrency` | `8192`, `120`, `16` | per request (reasoning tokens count against `max_tokens`); concurrency per scorer instance (per rank or actor) |
| `max_transcript_chars` | `60000` | cut applied to the graded transcript **and** to the reference answer |
| `structured_output` | `true` | off, the reply is parsed as the first JSON object it contains |

Both arms log `<name>/scored_frac`, the share of calls that returned a usable verdict — the metric that says a judge is failing. Per-requirement scores log as `judge/<name>/<requirement>` with `judge/<name>/completion_tokens`; the environment arm also logs `judge/<name>/cost_usd` where the endpoint reports a cost (OpenRouter does) and keeps the judge's rationale on the trajectory (`reward_details`). A failed or unparseable verdict scores `None`: TRL drops that term for the row; the environment arm marks the episode invalid.

Both arms probe every judge and reward-model term at launch with a one-line sample in the run's exact request shape (rank 0, verdict broadcast), so a bad URL, key, model or parameter fails the launch instead of every episode. `request_timeout` bounds each attempt, not the term's whole call: a request retries up to four times, and a reply that carries no choices with an upstream 429 or 5xx (how OpenRouter reports a rate-limited model) is retried four more times with exponential backoff, so one stuck judge call can hold a step for several minutes; lower the timeout before the retries.

## Served reward model

A `*ForSequenceClassification` or `*ForRewardModel` checkpoint served by the rollout engines' pooling routes. The scorer renders the conversation with the model's own chat template (`tokenizer`, default `model`) and tokenizes it without added special tokens, the exact input TRL's in-process reward-model path scores, then maps the head's logit through `sigmoid((logit − logit_shift) / logit_scale)`.

| Backend | Serve | Route |
|---|---|---|
| `vllm` | `vllm serve <model> --runner pooling` (`--convert classify` for a decoder checkpoint vLLM does not map to a classification head) | `POST /classify` with the rendered text, `add_special_tokens: false`, `use_activation: false`; reads `probs[label_index]` |
| `sglang` | `python -m sglang.launch_server --model-path <model> --is-embedding` | `POST /classify` with the token ids; reads `embedding[label_index]` |

`url` is the server root, without `/v1`. `backend` is `vllm` (or `sglang`), `label_index` `0`, `logit_shift` `0.0`, `logit_scale` `1.0`, and `transcript` `final` — the policy's last assistant message alone, `full` for every turn. `batch_size` (8) rows per request, `max_concurrency` (4) requests in flight per scorer, `request_timeout` 60 s. The raw logit logs as `reward_model/<name>/logit`. The online arm also keeps TRL's in-process path, reachable through the trainer API and not the YAML (a config lists `rewards:` terms): a model id in `reward_funcs` loads a sequence-classification head on every rank, which costs trainer memory; the served route does not.

## Environment arm

An environment grades, the terms price. `_grade_episode(trajectory, context)` returns an `EpisodeGrade`: `objective`, the environment's score in `[0, 1]` (pass fraction, answer match, adherence), and `shaping`, its own episode-level terms by bare name. The episode reward is the sum of its `reward/*` components:

- `reward/turn_shaping` — the per-turn deltas accrued during the episode (tool credit and penalties, ReAct thought credit).
- `reward/tool_shaping` — native-protocol environments only: their episode-level knobs (`no_tool_use_penalty`, `multi_turn_reward`, `turn_overflow_penalty`, `length_cutoff_penalty`), from `_episode_shaping`.
- `reward/<name>` — each of the environment's shaping terms (code contests: `submission`, `execution`, `tested_submission`, `resubmission`).
- `reward/objective` — `weight × grade ^ exponent` from the `environment` term.
- `reward/<name>` — each `judge` / `reward_model` term.

`rewards` defaults to `[{source: environment}]` and reaches the environment constructor as `reward_terms`. A class declares its shaping names in `SHAPING_COMPONENTS` (the union over the class hierarchy); `turn_shaping` and every declared shaping name are reserved for shaping, and a reward term may not take one. There is no failure offset: a constant cancels in the group baseline. An environment whose grade carries no signal (a null `answer` cell, a grader outage) grades 0 and marks the episode `episode_invalid`, out of the baseline.

The external terms are scored after the episode ends. At the terminal step the environment prices its own side and marks the reward pending; the episode dispatcher — the Ray actor and the eval runner both drive through it — awaits `settle_async` for every episode it closed, and a sync caller uses `env.settle(ids)`. Reading `rollout_metrics` on an unsettled episode raises. An episode the **eval** driver lost (a generation that raised) is graded on what it earned and never sent to a scorer; the training actor instead drops the partial trajectory and returns an error row at reward 0. A scorer reads the prompt turns before the first assistant turn, the policy's turns after them, and the row's `answer` as the reference; code contests hands it the submitted program as a fenced code block. A failed verdict contributes 0 for that term and marks the episode `episode_invalid`, which the eval runner reports as an error row; `episode/reward_scored` is the per-episode 1/0. The launch probe covers every external term through the environment's `verify_backend`. One scorer per environment instance, so judge concurrency is `max_concurrency` per Ray actor.

Writing one: [Custom Environments](environments/custom-environments.md).

## Failure and cost

A scorer never raises for a sample. Under tensor parallelism every TP rank scores its copy of the group and the leader's tensors are broadcast, so verdict noise cannot diverge the update, but judge calls scale with `tp_size`. Judge cost is per graded completion and per eval sample; the completion-token and cost metrics above are the budget to watch.
