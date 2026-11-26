# GRPO Rewards

A GRPO reward in Halo is a list of terms under `rewards:` in the config. Online GRPO and Async GRPO with
Environments score every completion they generate with it, and `halo run score-offline-grpo` uses the same
list to write the rewards of an [offline GRPO](offline-grpo.md#data) dataset.

## How terms add up

Each term names a `source`, scores a completion between 0 and 1 and adds `weight × score ^ exponent` to the
reward. `weight` defaults to 1 and may be negative, which turns the term into a penalty. `exponent` defaults to
1 and must be above 0; above 1, half a score earns less than half the weight.

A term's `name` is its metric key, `rewards/<name>/mean` in online GRPO and `reward/<name>` in Async GRPO. A
judge or a reward model must be given one; the other sources default to their own name, and the environment
term is always `objective`. Names must be unique, so a second term of the same source needs its own.

## The sources

### Accuracy and format

These two rules work in online GRPO and in offline scoring, and they send nothing anywhere. `accuracy`, the
online GRPO default, scores 1 when the completion's last `\boxed{...}` equals the row's `answer`, after commas,
dollar signs and a GSM8K `####` prefix are stripped. `format` scores 1 when the completion matches the regular
expression in `pattern`. Its default asks for `<think>…</think>` and then `<answer>…</answer>`, so set
`pattern` to the shape your prompt asks for. Use a rule whenever one can check the answer: it makes no
request and never times out.

### The environment's grade

In Async GRPO, `source: environment` turns the environment's own grade into reward: a solved problem, a
matched answer, a passing test. It is the default reward there, and a reward holds at most one. Keep it as the
main term wherever the environment can grade the task; a veto judge needs it. In `native_*` and `mcp`, a row
without an `answer` grades any finished episode 1, so the term then rewards finishing, not being right.

### An LLM judge

A `judge` term sends the task and the completion to a chat model behind any OpenAI-compatible API and turns
its JSON reply into a score. Pick it when no rule can check the answer but you can say in words what a good
one does. The default is `openai/gpt-5.6-luna` on OpenRouter with the key in `OPENROUTER_API_KEY`; `model`
and `base_url` point it elsewhere. A `localhost` server needs no key (a keyed one reads `VLLM_API_KEY`),
`api.openai.com` reads `OPENAI_API_KEY`, and any other host needs `api_key_env`, the variable holding its key.

A scoring judge lists `requirements`. The judge scores each from 0 to `scale` (10 by default), and the term
takes their weighted average. A veto judge lists `checks` instead and audits how the episode went. It works
only in Async GRPO next to an `environment` term: when a check marked `veto: true` fires, every positive part
of the episode's reward goes to 0 and the penalties stay. The term's own `weight` is 0 or negative.

`view` picks what the judge reads: `final`, the default, is the final answer; `full` is every turn with its
reasoning and tool results; `digest` shortens each. `context`, or a `context_file` read when the config is
parsed, tells the judge what the episode cannot show, such as the task's rules. For long episodes, raise
`max_view_chars`, which caps how much of the episode the judge reads, and `max_tokens`, which the judge's own
reasoning counts against.

### A reward model

A `reward_model` term scores with a Bradley-Terry or sequence-classification model you serve yourself, on
its own vLLM server (`--runner pooling`) or SGLang server (`--is-embedding`), apart from the rollout server.
Pick it when you already have a trained preference model and a GPU for it;
[Reward modeling](reward-and-classification.md#reward-modeling) trains one. Set `url` to the server root,
`model` to the served model and, for SGLang, `backend: sglang` (the default is `vllm`). The term renders the
conversation with the model's chat template and maps the logit to a score with a sigmoid, which
`logit_shift` and `logit_scale` move and stretch.

### White Circle policies

A `whitecircle` term sends the completion, in its conversation, to the White Circle API, which checks it
against the policies of your White Circle environment. Pick it when the reward is "breaks none of these
rules" and you can write the rules in plain language, such as security or safety rules. It needs
`environment_id` and `WHITECIRCLE_API_KEY`. With no list, the term scores 1 when nothing flags the reply and
0 otherwise. With `policies:` it scores the share of its own policies the reply did not violate, so several
terms can share one environment at different weights. A reply with nothing to check scores 0, and under the
default `view: final` so does an Async GRPO episode that ended without a final answer (`full` reads every
turn).

`halo run create-whitecircle-policies` creates the policies from a JSON file and prints the `rewards:` block.
Its runs share one White Circle condition, so read the
[cookbook](../cookbooks/halo-whitecircle-cookbook.md#2-create-the-policies) before a second run.

## An example

```yaml
rewards:
  - source: accuracy                 # 1 when the last \boxed{} matches the answer column
  - source: judge
    name: clarity
    weight: 0.3
    requirements: [{name: steps, description: "Every step follows from the one before it."}]
  - source: whitecircle
    name: safety
    weight: 0.5
    environment_id: <your environment id>
```

In online GRPO, a completion with the right answer, a clarity score of 8 out of 10 and no flagged policy
earns `1 + 0.3 × 0.8 + 0.5 × 1 = 1.74`. In Async GRPO, put `source: environment` in place of `accuracy`.

## What leaves your machine

- `accuracy` and `format` are computed in the trainer, and the `environment` grade wherever the environment
  runs it: a `remote`, `modal` or `baseten` sandbox receives the programs it runs.
- A judge receives, for every scored completion and eval sample, the term's `context`, the policy's system
  prompt, task and tool descriptions, the row's reference answer (unless `include_reference: false`) and the
  chosen view, with the reasoning under `full` and `digest` (unless `include_reasoning: false`). In online
  GRPO the completion is the generated text, so reasoning the model writes into it is sent in every view. It
  goes to OpenRouter by default; a `localhost` `base_url` keeps it on your machine.
- A reward model receives the rendered conversation at your `url`, with no authentication header.
- White Circle receives the prompt turns, system prompt included, and the final answer or, under
  `view: full`, every turn with its tool calls and outputs, at `us.whitecircle.com` (`eu.whitecircle.com` with
  `region: eu`). Tool schemas and the reference answer stay local, and so does the reasoning in Async GRPO.

## When a scorer fails

A failed request, a timeout or a reply Halo cannot read leaves that completion without a score, and the step
goes on. `on_error` decides what the missing score means:

- `invalid`, the default: in Async GRPO the episode leaves its group and the loss, and two steps in a row
  with no valid episode anywhere stop the run. In online GRPO the term counts 0, the completion leaves its
  group only when every term failed, and the run never stops.
- `neutral` counts the term as 0 and trains the completion on its other terms. It is the default for a veto
  judge, so a judge outage leaves the environment's grade standing and those episodes train unaudited.

Choose `invalid` when the term carries the main signal: in Async GRPO an outage then trains nothing instead of
a made-up 0. Choose `neutral` for a side check you can lose for a while. In online GRPO the two differ only
where every term failed, so next to a rule or a second working scorer they train the same. Offline scoring
leaves out a completion that any term could not score, unless that term is `neutral`.

A judge call can take up to 25 attempts, each bounded by `request_timeout`, and each source retries on its own
schedule ([Probe and retries](../../agent-docs/training-methods/grpo/rewards/README.md#probe-and-retries) ↗).
In Async GRPO the scoring runs as the episode ends, inside its `episode_timeout`, so a scorer still retrying at
that deadline loses the episode as an error, which `on_error` does not cover.

Three mistakes stop the run instead: a `whitecircle` term without `WHITECIRCLE_API_KEY` on any node that
scores it, a reward model whose tokenizer does not load at the launch probe, and `remove_unused_columns: true`
in online GRPO, which hides the scorers' columns.

## The launch probe

Before the first step, rank 0 sends each judge, reward model and White Circle term one tiny sample in the
run's exact request shape. A wrong URL, key, model name or request parameter stops the launch there, and so
does a White Circle environment that checks no policy on the reply, or a listed policy that no condition
selects. Only rank 0 is probed, so every other node, and in Async GRPO every
[Ray actor host](async-grpo-environments.md#the-environments), needs what the probe checked. A `localhost`
URL reaches a different server on each node.

## What to watch

- `<source>/<name>/scored` is the share of completions that got a score. Below 1, that scorer is failing.
- Async GRPO also logs `episode/reward_scored`, 1 when every external term scored the episode, and
  `sampling/invalid_episode_frac`.
- A judge logs a key per requirement or check, and its cost in `judge/<name>/completion_tokens` and, in Async
  GRPO where the endpoint reports it, `judge/<name>/cost_usd`, both over the replies it could read.

Every external term scores every completion, eval samples included, with at most `max_concurrency` requests
in flight per trainer rank in online GRPO and per Ray actor in Async GRPO. In online GRPO every rank of a
tensor-parallel group sends the same requests, so calls and cost scale with `tp_size` or `expert_tp_size`.

Every option, outage and metric of each source:
[Reward Terms](../../agent-docs/training-methods/grpo/rewards/README.md) ↗ ·
[Generative Judge](../../agent-docs/training-methods/grpo/rewards/judge.md) ↗ ·
[Served Reward Model](../../agent-docs/training-methods/grpo/rewards/reward-model.md) ↗ ·
[White Circle](../../agent-docs/training-methods/grpo/rewards/whitecircle.md) ↗.
