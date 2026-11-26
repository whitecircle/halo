# Reward Terms

A GRPO reward is a sum of **terms**. Each term names a source of a score in `[0, 1]` and prices it as `weight × score ^ exponent`. Terms are listed under `rewards:` and parsed at config time (`halo/rewards/terms.py`), so a bad term fails before any server is contacted.

The [online arm](../online.md#rewards) turns each term into one TRL reward function; the [environment arm](#environment-arm) composes them into the episode reward.

```yaml
rewards:
  - source: accuracy                 # online arm only
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

| Field | Default | Effect |
|---|---|---|
| `source` | — | Where the score comes from (table below) |
| `name` | Required for `judge` and `reward_model`; defaulted otherwise (`whitecircle` to `whitecircle`) | The metric key: `reward/<name>` on the environment arm, `rewards/<name>/mean` on the online arm. Unique. The `environment` term is always `objective`; any other `name` raises |
| `weight` | `1.0` (`0` for a veto judge) | Signed; negative makes a penalty |
| `exponent` | `1.0` | `> 0`. Above 1 is convex credit (half a score earns under half the weight), below 1 concave |

There is no constant-offset knob, failure offset included: a constant cancels in the group baseline (under `use_maxrl` it still changes each group's divisor).

| `source` | Score | Arm |
|---|---|---|
| `environment` | the environment's own grade | environments |
| `judge` | a [generative judge](judge.md): a score over `requirements`, or a veto over `checks` | both; a veto judge environments only |
| `reward_model` | a [served](reward-model.md) Bradley-Terry or sequence-classification model | both |
| `whitecircle` | a [White Circle](whitecircle.md) policy check of the completion | both |
| `accuracy`, `format` | the [RLVR graders](../online.md#rewards) | online |

## Common options

The external sources (`judge`, `reward_model`, `whitecircle`) score the finished completion and take:

| Option | Default | Effect |
|---|---|---|
| `view` | `final` | What the scorer reads: `final` (the final answer), `full` (every turn) or `digest` (a compact account of every turn; judge only) |
| `on_error` | `invalid`; `neutral` for a veto judge | What a missing verdict means: `invalid` takes the completion out of the group baseline (on the online arm only when every term missed), `neutral` prices the term 0 and keeps it ([Outages](#outages)) |

## Outages

A scorer never raises for a sample. A failed request, an unparseable reply or an unbuildable prompt returns no verdict, and `<source>/<name>/scored` logs 0 for that completion.

| `on_error` | Online arm (TRL row) | Environment arm (episode) |
|---|---|---|
| `invalid` (default) | The term returns `None`, summed as 0. The row leaves the baseline only when every term returned `None` | The term prices 0 and the episode is `episode_invalid`, out of the group baseline, with the scorer's error as `episode_invalid_reason`. The eval runner reports an error row |
| `neutral` (veto judge default) | The term returns 0 | The term prices 0; the episode stays valid |

On the environment arm, a second step in a row with no valid episode on any rank stops the run ([Batch construction](../../async-grpo/objective.md#batch-construction)). The online arm has no such halt: during an outage the term adds 0 beside the other terms, or, when every term missed, the row trains at advantage 0, and the run goes on. Watch `<source>/<name>/scored`.

Three configuration faults raise instead of booking an outage:

- a `whitecircle` term with `WHITECIRCLE_API_KEY` unset, when its scorer is built: at launch on every rank of the online arm; on the environment arm in rank 0's launch probe and as each Ray actor starts, which fails the run's start on every rank;
- a reward model whose tokenizer does not load, at its first scoring: the launch probe, then each rank or Ray actor. A Ray actor whose node cannot load it raises in every settlement, each losing its episode as an error row (`episode/error_rate`);
- on the online arm, a term whose dataset column TRL did not forward ([Rewards](../online.md#rewards)).

## Probe and retries

At launch, rank 0 probes every external term with a one-line sample in the run's exact request shape, so a bad URL, key, model or parameter fails the launch, not every episode. The environment arm probes through the environment's `verify_backend`.

The probe runs in global rank 0's process alone, while every scorer sends its own requests: each trainer rank on the online arm, each Ray actor on the environment arm. On several nodes, or on a shared Ray cluster (`ray_address`, where an actor takes its node's `ray start` environment; [Ray Cluster](../../../infrastructure/ray.md#multi-node)), every node needs the term's key variable and a route to its endpoint, and a loopback URL reaches a different server, or none, on each node. A node that cannot reach a term's endpoint, or lacks a judge's key, books outages there, each settled by the term's `on_error` ([Outages](#outages)); `<source>/<name>/scored` below 1 shows it.

Judge requests retry at two levels (`halo/inference/openai_client.py`):

- the SDK retries a timeout, a connection error or a 408, 409, 429 or 5xx status up to four times, with `request_timeout` bounding each wait within an attempt (connecting, sending, each gap in the reply), not the attempt itself, after about 0.5, 1, 2 and 4 s (or the server's `Retry-After`, up to 60 s);
- a reply with no choices that carries a 408, 429, 500, 502, 503 or 504 error code (how OpenRouter reports a rate-limited model) sends the whole request again, up to four times, after 2, 4, 8 and 16 s.

The levels nest, so one judge call makes at most 25 attempts, and nothing bounds the call as a whole: each attempt can wait `request_timeout` for its reply to start, 50 minutes plus the backoff over 25 at the default 120 s, and a reply that arrives in pieces stretches its attempt further. Lower the timeout first. White Circle requests retry on their own schedule ([Failures](whitecircle.md#failures)), and reward-model requests never retry ([Requests](reward-model.md#requests)).

## Environment arm

The environment grades; the terms price. `_grade_episode(trajectory, context)` returns an `EpisodeGrade`: `objective` (the score in `[0, 1]`) and `shaping` (the environment's own episode-level terms, by bare name). Writing one: [Custom Environments](../../async-grpo/environments/custom.md).

The episode reward is the sum of its `reward/*` components:

- `reward/turn_shaping`: per-turn deltas (tool credit and penalties, ReAct thought credit).
- `reward/tool_shaping`: [native-protocol](../../async-grpo/environments/protocols/native-tool-use.md) episode knobs (`no_tool_use_penalty`, `turn_overflow_penalty`, `length_cutoff_penalty`).
- `reward/<name>`: each environment shaping term (code contests: `submission`, `resubmission`) and each external term.
- `reward/objective`: `weight × grade ^ exponent` from the `environment` term.

The trainer adds its length charges, `reward/reasoning_price` and `reward/reasoning_floor` (each logged while its term is on), outside these components ([Reasoning length reward](../../async-grpo/reasoning.md#reasoning-length-reward)). Components and rationales are saved with each completion ([Saved trajectories](../../async-grpo/monitoring.md#saved-trajectories)).

- `rewards` defaults to `[{source: environment}]` and reaches the environment constructor as `reward_terms`.
- `turn_shaping` and the class's `SHAPING_COMPONENTS` names (unioned over the hierarchy) are reserved; no reward term may take one.
- A grade with no signal (a null `answer`, a grader outage, a code grade that stopped before any test failed) is 0 and marks the episode `episode_invalid`.

### What a scorer reads

One sample serves every external term, each through its own view. It carries:

- the prompt turns (everything before the first assistant turn) and the tool schemas;
- the policy's turns, with reasoning and the capped, cut, empty or rejected flags;
- the calls a cut turn never ran (`Message.cut_tool_calls`), which only the judge's `full` and `digest` views show;
- the environment's notes on single calls (`Message.call_notes`), shown beside each call in those views;
- the final answer and the reference.

A reward model and White Circle read the chat turns alone (role, content, tool calls and their results); reasoning, the turn flags, call notes and cut calls reach only the judge's `full` and `digest` views.

Three `EnvironmentBase` hooks fill the episode-specific parts:

- `_final_answer(trajectory)`: what the episode delivered (the native protocol's `final_response`, ReAct's `Final Answer`, code contests' submitted program as a fenced code block), or `None` (the base) when it delivered none.
- `_scoring_reference(trajectory)`: the reference, `context["answer"]` by default. `exam_qa` hands the choice letter; code contests hands `None`, so the hidden tests never reach a judge.
- `_episode_facts(trajectory)`: the episode facts the environment decides itself (declared in `EPISODE_FACTS`), which a judge check names in `unless` to stand exempt under; none in the base. Code contests establishes `failed_with_no_runs_left`.

### Settlement

External terms are scored after the episode ends. At the terminal step the environment prices its own side and marks the reward pending. The episode dispatcher (Ray actor and eval runner alike) then awaits `settle_async`; a synchronous caller uses `env.settle(ids)`. Settlement prices each term, applies a fired veto and books a missing verdict by its `on_error`.

- Settlement runs on the actor that played the episode, inside its `episode_timeout`. A judge call can spend 25 attempts before it books an outage ([Probe and retries](#probe-and-retries)), so an episode whose scoring outlasts the deadline is cancelled as an error row (`episode/error_rate`), which `on_error` never sees. Keep each term's timeout and retries well inside what generation leaves.
- Reading `rollout_metrics` on an unsettled episode raises.
- `episode/reward_scored` is 1 only when every external term reached a verdict; a miss settled `neutral` still logs 0.
- Judge concurrency is `max_concurrency` per Ray actor (one scorer per environment instance).

Some episodes never reach a scorer:

- **A generation that failed past its retries.** The training actor returns an error row at reward 0. The eval driver grades what the episode earned, but reports a [generation error](../../async-grpo/evaluation.md#failed-generations) instead unless its own request caused the failure.
- **A [sandbox fault](../../async-grpo/environments/sandbox.md#sandbox-faults).** The episode logs no `episode/reward_scored`.
- **An episode that outgrew the served window.** Every external term prices 0 for it without marking it invalid, and it logs no `episode/reward_scored`. Its environment grade stands, an invalid one included; when its row trains is on [Context window](../../async-grpo/episodes.md#context-window).

## Failure and cost

- On the online arm under TP or ETP, every rank of a group scores its copy and the leader's result is broadcast, so judge calls scale with `tp_size` or `expert_tp_size`. The environment arm scores once per group, since only the leader's actors roll out.
- Judge cost is per graded completion and per eval sample; watch the judge's [completion-token and cost metrics](judge.md#metrics).

## Pages

- [Generative Judge](judge.md)
- [Served Reward Model](reward-model.md)
- [White Circle](whitecircle.md)
