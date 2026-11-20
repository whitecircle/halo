# White Circle

A `whitecircle` reward term scores each completion with the White Circle API (`POST /api/events`): Halo sends the
completion and its conversation, the API returns a verdict for each policy of a White Circle environment (flagged or
not flagged), and Halo converts the verdicts into a score in `[0, 1]`. The term is available in online GRPO,
environmental GRPO and offline reward scoring. Fields shared by all reward terms, `on_error` and the launch probe: [Reward Terms](README.md).

```yaml
rewards:
  - source: accuracy
  - source: whitecircle
    name: policy
    weight: 0.5
    environment_id: your_environment_id
    policies: [policy_a, policy_b]    # optional: score only these policy ids
```

```bash
WHITECIRCLE_API_KEY=wc-... halo launch grpo/online examples/grpo/online/whitecircle/qwen3-4b-gsm8k-whitecircle.yaml
```

This recipe trains Qwen3-4B on GSM8K with a vLLM server started as in [Online GRPO](../online.md); set its
`environment_id` first. `halo run create-whitecircle-policies` creates the policies of a JSON file in an environment
and prints the matching `rewards:` block; `human-docs/cookbooks/halo-whitecircle-cookbook.md` walks through a full run.

## Score

Only the verdicts on the completion's own events count: its assistant text and the arguments of its tool calls. A
flagged verdict on the prompt or on a tool output does not lower the score.

| Term lists | Score | Flagged (`whitecircle/<name>/flagged`) |
|---|---|---|
| Neither `policies` nor `checklist` | 1 if no verdict on the completion is flagged, else 0 | Score is 0 |
| `policies` | Fraction of the listed policies with no flagged verdict on the completion | Score is below 1 |
| `checklist` | 1 minus the confidence; confidence = fraction of the checklist's items with a flagged verdict on the completion | Confidence is 0.5 or more |

- With `policies` or `checklist`, verdicts of policies outside the list do not affect the score, so several terms
  can share one environment, each with its own policies.
- A term lists `policies` or `checklist`, not both.
- A completion with no text and no tool call scores 0, and Halo sends no request: it violates nothing, and full credit for it would reward an empty reply.

### Checklist

A checklist is one assessment split into items. Each item is a White Circle policy; the term's score and metrics
use one confidence, not one verdict per item. In the policy file of `halo run create-whitecircle-policies`, a checklist is an
object with `checklist` (its name), `allowed_content` (added to every item) and `items` (each with `name`,
`flagged_content` and `allowed_content`). Each item's policy is named `<checklist> · <item name>`. The policy name is
part of the input the White Circle API judges with, so the item name states what the item checks. Write each item
for a separate fault, so that one fault matches one item.

## Request

- Halo sends one request per completion under a new `external_session_id`: the prompt turns and then the completion
  turns, as events in order. The API evaluates each event with the events before it as context, and never with the
  conversation of another completion.
- `view: final` sends the prompt and one assistant message with the final answer. `view: full` sends every turn: an
  assistant tool call becomes a `tool` event with its `arguments`, a tool output a `tool` event with its `output`.
  `view: digest` raises an error.
- A conversation of more than 50 events (the API limit per request) is sent in consecutive requests of at most 50
  events under the same session.
- Headers: `Authorization: Bearer $WHITECIRCLE_API_KEY` and `whitecircle-version: 2026-06-01`. Halo reads the key
  from the environment variable when it creates the HTTP client and never logs it. If the variable is unset,
  creating the term's scorer raises an error.

## Options

| Option | Default | Meaning |
|---|---|---|
| `environment_id` | — | The White Circle environment whose policies apply; required |
| `policies` | `[]` | Policy ids to score ([Score](#score)) |
| `checklist` | `[]` | Policy ids of a checklist's items ([Checklist](#checklist)) |
| `region` | `us` | `us` (`https://us.whitecircle.com`) or `eu` (`https://eu.whitecircle.com`) |
| `view` | `final` | `final` or `full` ([Request](#request)) |
| `request_timeout` | `30` s | Timeout of one request attempt |
| `max_retries`, `retry_wait` | `4`, `1.0` s | Number of retries of one request, and the wait before the first retry ([Failures](#failures)) |
| `max_concurrency` | `8` | Maximum completions with a request in progress, per scorer (one scorer per rank or Ray actor) |
| `on_error` | `invalid` | Reward of a completion without a score ([Outages](README.md#outages)) |

## Failures

Halo retries a timeout, a transport error, and an HTTP 408, 429 or 5xx up to `max_retries` times. The wait before
each retry is the `Retry-After` value of the response (at most 60 s), or else `retry_wait × 2^attempt`. Any other
error status, such as 400 for a malformed request or 401 or 403 for the key, fails immediately.

A completion gets no score when:

- the retries are used up, or the API returns an error status;
- an event of the completion has no verdict (a status other than `completed`, or no `flagged` field);
- the API returns no policy verdict for any event of the completion. For an event that no condition assigns a
  policy to, the API returns `completed` and not flagged, with a `note`; Halo includes the note in the error;
- the API returns no verdict on the completion for a listed policy or checklist item.

Each failure logs a warning and `whitecircle/<name>/scored` 0, and `on_error` sets the reward. The launch probe sends
a one-line conversation, so the launch fails on a wrong key, an unknown environment, an environment that assigns no
policy to an assistant message, or a listed policy id that no condition assigns.

## Metrics

| Key | Meaning |
|---|---|
| `whitecircle/<name>/scored` | Fraction of completions with a score |
| `whitecircle/<name>/flagged` | 1 when the completion counts as flagged ([Score](#score)), else 0 |
| `whitecircle/<name>/<policy id>` | With `policies`: 1 when this policy's verdict on the completion is flagged, else 0 |
| `whitecircle/<name>/confidence` | With `checklist`: the confidence ([Score](#score)) |

In environmental GRPO, Halo saves the names of the policies with a flagged verdict in the trajectory (`reward_details`).

## Offline GRPO

`halo run score-offline-grpo` writes the `rewards` of an [offline GRPO](../offline.md) dataset from the `rewards`
list of a training YAML, such as the recipe above:

```bash
WHITECIRCLE_API_KEY=wc-... halo run score-offline-grpo \
    examples/grpo/online/whitecircle/qwen3-4b-gsm8k-whitecircle.yaml \
    --input rows.jsonl --output scored.jsonl --answer_field answer
```

- Each row has `prompt` and `completions`. `--answer_field` names the ground-truth column for an `accuracy` term or
  a judge term.
- The reward of a completion is the sum of `weight × score ^ exponent` over the terms, computed by the same reward
  functions as in online GRPO.
- Halo probes every external term first. A completion that any term cannot score is dropped and counted, not
  written with a substitute reward; with `on_error: neutral` that term contributes 0 and the completion is kept. A
  row with no completions left is dropped.
