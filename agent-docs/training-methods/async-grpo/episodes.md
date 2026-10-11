# Episodes

An episode is one multi-turn conversation: the engine samples an assistant turn, the environment answers it, and the
loop runs until a final answer, `max_turns`, or a spent budget. Knobs are top-level
[`AsyncTrainingConfig`](../../reference/configuration-reference.md#asynctrainingconfig) fields except
`carry_reasoning`, `max_length_cutoff_recoveries` and `length_cutoff_penalty` (`environment_kwargs`), `max_turns`
(`EnvironmentConfig`), and `max_prompt_length`, `chat_template` and `force_chat_template` (script arguments).
Reasoning budgets have [their own page](reasoning.md).

## Sampling

| Knob | Default | Effect |
|---|---|---|
| `rollout_temperature` | `0.7` | sampling temperature |
| `rollout_top_p` | `0.95` | in `(0, 1]` |
| `rollout_top_k` | `-1` (off) | `-1` or ≥ 1; `0` is refused, since SGLang rejects it |
| `rollout_min_p` | `0` (off) | in `[0, 1]`; vLLM rejects every request with `min_p > 0` on a server running speculative decoding (MTP) |
| `rollout_repetition_penalty` | `1` (off) | in `(0, 2]`; another value is refused at startup under a sequence-level ratio (`isr_geo_band_*`, `isr_opsm_delta`; [Trust region masks](objective.md#trust-region-masks)), since the trainer's recomputed log-probs omit it |

Rollout and eval requests send all five, so the model's `generation_config.json` defaults never apply. TRL's own
sampling fields reach no sampler and are refused when set ([Rollout backend](setup.md#rollout-backend)).

## Trajectory length

| Knob | Default | Effect |
|---|---|---|
| `rollout_max_tokens` | `32768` | caps **one turn**, reasoning and answer together |
| `max_turns` | `null`: the environment class's own (15 on `code_contests` / `codeforces`, 20 on `swe`, 8 on `exam_qa`, 10 elsewhere) | an episode that burns them ends truncated |
| `rollout_max_episode_tokens` | `null`: `max_turns × rollout_max_tokens` | caps what an **episode's** assistant turns sample together, recovery turns included; at least `rollout_max_tokens` |
| `max_prompt_length` | `null` | drops dataset rows whose rendered prompt exceeds it |
| `max_train_row_tokens` | `null`: no cap beyond the context check | memory bound on a training row; must exceed `rollout_max_tokens` |

### Episode output budget

`rollout_max_episode_tokens` binds per turn and is never stated to the model.

- A turn's `max_tokens` is the smaller of its own cap and what the episode has left. Its reasoning cap shrinks by as
  much, so the turn keeps its [answer room](reasoning.md#answer-room).
- An episode left with less than that room ends truncated, priced like a `max_turns` overflow. Watch
  `episode/output_budget_exhausted`.
- It travels as `max_tokens`, so it holds on both engines.

### Context window

The trajectory is **never truncated**: a row over the model's context window raises on every rank together. The
startup check warns when the worst-case episode overflows the served window
([Rollout backend](setup.md#rollout-backend)); size `rollout_max_episode_tokens` so it fits.

`max_completion_length` is no knob here: the script pins it to `rollout_max_tokens` (TRL reads it only as the
`dr_grpo` normalizer) and raises on another explicit value.

### Training-row cap

A per-turn row over `max_train_row_tokens` (prompt plus completion) leaves the batch while the episode's other turns
train; a whole-trajectory row trains at zero weight. Setting the cap logs `sampling/rows_over_cap_frac`, the share of
rows left out.

The episode budget leaves tool output between turns unbounded, so it does not replace this cap.

## Tool calls

Native-tool environments send their `tools` schema, so the server needs the matching
`--tool-call-parser` (per-family values: [Rollout Servers](../../infrastructure/rollout-servers.md#vllm)).

- A mismatched parser fails silently: calls come back as text with no `tool_calls`, so a native-protocol episode ends on that turn as a plain-text final answer.
- A missing parser is a 400 on vLLM; SGLang returns the call as text.
- [ReAct](environments/protocols/react.md) environments send no `tools` and are served without a parser.

`rollout_stop_tokens` (default empty) lists special-token strings sent as `stop_token_ids`. Use it for a tool-call
terminator the model's `eos_token_id` omits: gpt-oss with harmony disabled ends a call with `<|call|>` and otherwise
hallucinates the result (the gpt-oss recipes set `["<|call|>"]`). An unknown name raises at trainer construction.

## Unproductive turns

A turn is unproductive when it is:

- **cut**: the engine stopped it at its token cap;
- **empty**: the model ended it with neither a tool call nor visible content;
- **rejected**: every tool call named a nonexistent tool. Under the native tool-call protocol a call refused unrun
  also counts ([Untrainable turns](environments/protocols/native-tool-use.md#untrainable-turns)); ReAct flags unknown
  tools only.

What follows:

- A cut or empty turn is nudged and retried within `max_turns` and `max_length_cutoff_recoveries` (`null` = no cap
  beyond `max_turns`), a cap the two kinds share. A rejected turn is answered with the refusals and spends a turn, not
  the cap.
- Every cut turn counts under `episode/length_cutoff_turns` and every empty one under `episode/empty_turns`, the one
  that ends the episode included. Under the native protocol, a turn cut at its token cap with a tool call the
  server's parser salvaged also counts under `episode/length_cutoff_in_call_turns`.
- The native protocol charges `length_cutoff_penalty` (default `0`) per recovered cut or empty turn
  ([Native tool-use](environments/protocols/native-tool-use.md#reward)).
- A turn past the cap, on the episode's last turn, or on the last turn the output budget affords ends the episode
  truncated, priced like a `max_turns` overflow.
- The next turn runs under a reduced reasoning cap ([Retry reserve](reasoning.md#retry-reserve)).
- Their rows train only under a negative advantage ([Untrainable turns](objective.md#untrainable-turns)).

Under [carried reasoning](#carried-reasoning) a retry thinks on from where the cut stopped, so a per-turn budget binds
only once cuts are priced.

### Engine aborts

An engine abort never reaches the environment: the actor re-issues the turn up to `max_retries` times (default `3`),
then the episode errors into a masked row.

## Carried reasoning

`carry_reasoning: true` (default off) sends the latest assistant turn's reasoning back, so the next turn continues
that thought. Earlier turns stay visible text, so a request grows by at most one reasoning budget.

- The trainer refuses it under `rollout_backend: sglang` (unverified there).
- A step whose turns carry no reasoning while carried reasoning or a reasoning term is on warns once: the server most
  likely runs without a reasoning parser.
- The template decides where a carried thought renders. The stock Qwen3.6 template renders it only on turns after
  the last user message, so a recovery nudge hides it, unless `rollout_chat_template_kwargs: {preserve_thinking: true}`
  is set; the stock Qwen3.5 template has no such variable. The shipped Qwen3.6 effort template always renders it.

## Chat template

Two safe setups:

- **Built-in** (default): no `chat_template:` on the trainer and no `--chat-template` on the server, so both load the
  checkpoint's.
- **Custom**: `chat_template: <file>.jinja` plus `force_chat_template: true` on the trainer and
  `--chat-template <same file>` on the server, the file visible inside its container.

`chat_template:` without `force_chat_template: true` is a no-op wherever the checkpoint ships a template. With it, and
no `--chat-template` on the server, the two sides diverge silently. The re-tokenization paths
([Training rows](objective.md#training-rows)) then score log-probs against a prompt the policy never generated under;
per-turn rows take the engine's ids and cannot drift.

`rollout_chat_template_kwargs` applies template variables to every request **and** the trainer's own renders. It
refuses `reasoning_effort`, `reasoning_budget` and `reasoning_budget_soft`, which are per episode
([Effort templates](reasoning.md#effort-templates)).
