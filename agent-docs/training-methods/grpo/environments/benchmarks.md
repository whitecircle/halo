# QA Benchmark Environments

Two registry names cover question answering with rule-based rewards and no neural reward model:
`qa_search` for factual QA with web search (SimpleQA, GAIA, TriviaQA, PopQA) and `exam_qa` for
multiple-choice or open-ended exams (MMLU-Pro, GPQA, MMLU, ARC). Both live in
`src/environments/envs/tasks/qa.py` and speak native tool calls, so a tool-carrying variant needs a
tool-call parser on the server ([Rollout Configuration](../async-grpo/rollouts.md#tool-calls)).

`qa_search` is a factory preset over `NativeToolUseEnvironment` — search tools, a research-assistant
prompt, `require_tool_use=True` — not its own class. `ExamQAEnvironment` is closed-book by default.
`examples/grpo/environmental/qwen3_5/vllm/qwen3.6-35b-a3b-exam-qa-full-ep4.yaml` is a shipped recipe.

## Configuration

```yaml
environment_type: qa_search   # or exam_qa
max_turns: 10                 # qa_search default 10, exam_qa 8
environment_kwargs:
  search_backend: duckduckgo
  include_python_tools: false
```

| Knob | Default | Effect |
|---|---|---|
| `search_backend` | auto | `serper`, `brave`, `tavily`, `duckduckgo`; auto-selects by which API key is set, keyless DuckDuckGo last. Validated at construction |
| `include_python_tools` | `false` | `qa_search` only: adds the sandboxed `python` tool for numeric QA |
| `open_book` | `false` | `exam_qa` only: registers the search tool. Setting `search_backend` closed-book raises |
| `system_prompt` | class prompt | Replaces the built-in instructions |
| `require_tool_use` | `true` on `qa_search` | Flags an episode that called no tool; the charge is `no_tool_use_penalty` |

A fifth backend, `mock`, returns fabricated snippets and is refused unless
`HALO_ALLOW_MOCK_SEARCH=1`: its results pay `tool_success_reward` like a real search, so a training
run reaching it would teach the policy that invented evidence works.

## Tools

- `web_search` — query plus optional `max_results` (5); returns titles, snippets and URLs. On `qa_search` always, on `exam_qa` only under `open_book`.
- `python` — sandboxed REPL, `qa_search` under `include_python_tools`.

## Reward

Grading is all-or-nothing, through `src/rewards/matching.py`: `exact_match` (case-insensitive
after normalization), then `numeric_match` — the response's first number against the whole expected
value, at `rtol=0.01` / `atol=1e-6`, percentages divided by 100. A match grades 1, anything else 0,
and the reward's `environment` term prices the grade ([Reward Terms](../rewards.md#environment-arm)).
There is no fuzzy or substring matcher — "7" must not match "17".

Online GRPO's `accuracy` term uses a different grader — a strict boxed exact match that splits
`####` and strips `,`/`$` ([Online GRPO → Rewards](../online-grpo.md#rewards)). The two score the
same row differently; a recipe picks one.

Normalization reads `\boxed{42}`, `**42**` and leading "The answer is" / "Therefore" phrasings. It
takes the `\boxed{...}` whose opening brace is rightmost and matches braces by depth, so
`\boxed{\frac{1}{2}}` survives. It does not split a GSM8K-style `#### N` suffix: reduce such an
`answer` column to the final value before training.

A row with `choices` switches `exam_qa` to letter grading: the response's choice letter (A–J) is
extracted from "A", "(A)", "A.", "The answer is A" and compared to the expected letter, and the
choices are appended to the prompt.

`answer` may be that letter or a 0-based index into `choices` (MMLU and ARC ship an int). Anything
else, an out-of-range index included, raises at episode start rather than grading every completion
0 at zero group variance.

## Dataset

```json
{"prompt": "What year was the Eiffel Tower completed?", "answer": "1889"}
{"prompt": "Largest planet?", "answer": "B", "choices": ["A: Mars", "B: Jupiter", "C: Saturn"]}
```

`choices` reaches the environment as a context column: name it in `context_fields` (training) or
`--context_fields choices` (eval). `answer` is required (`requires_answer`) for both `exam_qa` and
`qa_search`: a dataset without it is refused at trainer construction.

## Evaluation

```bash
python scripts/environments/inference/run_env.py --env_type exam_qa \
    --dataset <hub-id-or-dir> --split test --prompt_field question --answer_field answer \
    --context_fields choices --group_by subject \
    --base_url http://localhost:8000/v1 --model <served-name> --num_examples 100
```

Flags, output files and the `--training_config` contract: [Evaluating on an Environment](evaluation.md).

## Related pages

- [Native Tool-Use](native-tool-use.md) — the protocol both environments run on
- [Code Contests](code-contests.md) — hidden-test competitive programming
- [Environments](README.md) — registry and `environment_kwargs` plumbing
