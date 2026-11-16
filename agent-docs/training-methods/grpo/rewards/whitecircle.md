# White Circle

A `whitecircle` term checks each completion, in its conversation, against the policies of a White Circle environment through the public API (`POST /api/events`), on both arms. Shared term fields, `on_error` and the launch probe: [Reward Terms](README.md).

```yaml
rewards:
  - source: accuracy
  - source: whitecircle
    name: policy
    weight: 0.5
    environment_id: your_environment_id
    policies: [policy_a, policy_b]    # optional: read only these policy ids
```

```bash
WHITECIRCLE_API_KEY=wc-... halo launch grpo/online examples/grpo/online/whitecircle/qwen3-4b-gsm8k-whitecircle.yaml
```

The recipe trains Qwen3-4B on GSM8K against a vLLM server started as in [Online GRPO](../online.md); set its `environment_id` first.

## Score

The score reads only the policy's own events: the completion's assistant text and its calls' arguments. A flagged prompt or tool result costs nothing.

- No `policies`: 1 when no event of the completion is flagged, else 0.
- `policies` listed: the fraction of them that no event of the completion violated. Other policies are ignored.
- A completion with no text and no call has nothing to check and scores 1, with no request.

## Request

- Each completion is one request under a fresh `external_session_id`: the prompt turns, then the completion's, as events in order, so each event is checked against the ones before it and never against another completion's conversation.
- The `final` view sends the prompt and one assistant message holding the final answer. `full` sends every turn: an assistant call is a `tool` event with its `arguments`, a tool result a `tool` event with its `output`. `digest` is refused.
- A conversation over 50 events (the API's cap) goes in consecutive requests of up to 50 under the same session.
- Headers: `Authorization: Bearer $WHITECIRCLE_API_KEY` and `whitecircle-version: 2026-06-01`. The key is read from the environment when the client is built and is never logged; with it unset, building the scorer raises.

## Options

| Option | Default | Effect |
|---|---|---|
| `environment_id` | — | The White Circle environment whose policies apply; required |
| `policies` | `[]` | Policy ids the score reads ([Score](#score)) |
| `region` | `us` | `us` (`https://us.whitecircle.com`) or `eu` (`https://eu.whitecircle.com`) |
| `view` | `final` | `final` or `full` ([Request](#request)) |
| `request_timeout` | `30` s | Per attempt |
| `max_retries`, `retry_wait` | `4`, `1.0` s | Retries of one request and its first backoff ([Failures](#failures)) |
| `max_concurrency` | `8` | Completions in flight per scorer (per rank or actor) |
| `on_error` | `invalid` | [Outages](README.md#outages) |

## Failures

A timeout, a transport error, a 408, a 429 or a 5xx is retried up to `max_retries` times, after the `Retry-After` seconds the server sent (capped at 60) or `retry_wait × 2^attempt`. Any other error status (a 400 for a malformed request, a 401 or 403 for the key) fails at once.

A check fails, and the completion gets no verdict, when:

- the retries run out or the API answers an error status;
- an event of the completion comes back without a verdict (a status other than `completed`, or no `flagged`); the error carries the API's `note`, such as no condition selecting a policy;
- a listed policy was checked on none of the completion's events.

Each failure logs a warning and `whitecircle/<name>/scored` 0, and `on_error` prices it. The launch probe checks a one-line conversation, so a bad key, an unknown environment, an environment that selects no policy for an assistant message, or a listed policy id no condition selects fails the launch.

## Metrics

| Key | Meaning |
|---|---|
| `whitecircle/<name>/scored` | Share of completions with a verdict |
| `whitecircle/<name>/flagged` | An event of the completion was flagged (1/0) |
| `whitecircle/<name>/<policy id>` | A listed policy was violated (1/0) |

The environment arm saves the names of the violated policies with the trajectory (`reward_details`).

## Offline GRPO

`score_offline_grpo.py` writes an [offline-GRPO](../offline.md) dataset's `rewards` with the `rewards` list of any training YAML, the recipe above included:

```bash
WHITECIRCLE_API_KEY=wc-... halo run score-offline-grpo \
    examples/grpo/online/whitecircle/qwen3-4b-gsm8k-whitecircle.yaml \
    --input rows.jsonl --output scored.jsonl --answer_field answer
```

- Each row carries `prompt` and `completions`; `--answer_field` names the ground truth an `accuracy` term or a judge reads.
- A completion's reward is the sum of `weight × score ^ exponent` over the terms, through the same reward functions as the online arm.
- Every external term is probed first. A completion any term left without a verdict is dropped and counted, not written with a forced reward (under `on_error: neutral` the term prices 0 and the completion stays); a row left with none is dropped.
