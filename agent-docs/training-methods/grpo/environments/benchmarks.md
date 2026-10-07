# QA Benchmark Environments

Two registry names cover question answering with rule-based rewards and no neural reward model:
`qa_search` for factual QA with web search (SimpleQA, GAIA, TriviaQA, PopQA) and `exam_qa` for
multiple-choice or open-ended exams (MMLU-Pro, GPQA, MMLU, ARC). Both live in
`src/environments/envs/tasks/qa.py` and speak native tool calls, so a tool-carrying variant needs a
tool-call parser on the server ([Rollout Configuration](../async-grpo/rollouts.md#tool-calls)).

`qa_search` is a factory preset over `NativeToolUseEnvironment` — search tools, a research-assistant
prompt, a required `answer` column — not its own class. `ExamQAEnvironment` is closed-book by default.
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
| `include_python_tools` | `false` | `qa_search` only: adds the in-process restricted `python` REPL for numeric QA |
| `open_book` | `false` | `exam_qa` only: registers the search tool. Setting `search_backend` closed-book raises |
| `system_prompt` | class prompt | Replaces the built-in instructions |

A fifth backend, `mock`, returns fabricated snippets and is refused unless
`HALO_ALLOW_MOCK_SEARCH=1`: its results pay `tool_success_reward` like a real search, so a training
run reaching it would teach the policy that invented evidence works.

## Tools

- `web_search` — query plus optional `max_results` (5); returns titles, snippets and URLs. On `qa_search` always, on `exam_qa` only under `open_book`.
- `python` — the in-process restricted REPL, `qa_search` under `include_python_tools`.

## Reward

Grading is all-or-nothing, through `src/rewards/graders/matching.py`: `exact_match` (case-insensitive
after normalization), then `numeric_match` — every number in the response against the one value the
expected answer states (read the same way, so `$18` and `18 dollars` expect 18), at `rtol=0.01` /
`atol=1e-6`, percentages divided by 100 and `3,500` or `10\,000` read as one number. The response must
commit to one value: a hedge (`7 or 8`) grades 0, and so does a correct answer that restates working
with other numbers (`7, since 3 + 4 = 7`). A number that is an operand of arithmetic between numbers
(`1/2`, `2024-01-01`), of a power (`10^3`, `10²`), root (`\sqrt{2}`), constant (`2\pi`) or function
(`log 2`) grades 0 too, as does a stated bound (`x < 3`, `x \le 3`). A power after a letter is a unit
exponent (`9.8 m/s^2`, `5 m²`), and a number glued to a letter is part of a token (`H2O`). A number joined
to a one-letter variable by an operator is an operand too (`1/x`, `n+1`), but one glued to a letter
reads as a number plus a unit, so `2x` is not caught, nor are word forms (`square root of 2`, `at least 3`). A match grades 1, anything else 0, and the reward's `environment`
term prices the grade ([Reward Terms](../rewards.md#environment-arm)). There is no fuzzy or substring
matcher — "7" must not match "17".

Online GRPO's `accuracy` term uses a different grader — a strict boxed exact match that splits
`####` and strips `,`/`$` ([Online GRPO → Rewards](../online-grpo.md#rewards)). The two score the
same row differently; a recipe picks one.

Normalization reads `\boxed{42}`, a lone `**42**` and leading "The answer is" / "Therefore"
phrasings; several bold spans (`**7** or **8**`) stay in the text as the hedge they are. It takes the
`\boxed{...}` whose opening brace is rightmost and matches braces by depth, so `\boxed{\frac{1}{2}}`
survives. It does not split a GSM8K-style `#### N` suffix: reduce such an
`answer` column to the final value before training.

A row with `choices` switches `exam_qa` to letter grading: the response's choice letter (A–J) is
extracted from "A", "(A)", "A.", "The answer is A" and compared to the expected letter, and the
choices are appended to the prompt.

`answer` may be that letter or a 0-based int index into `choices` (MMLU ships the index). A digit
string raises: ARC's `answerKey` is sometimes a 1-based label (`"1"`–`"5"`), so convert it to a
letter when preparing the data. Anything else, an out-of-range index included, raises at episode
start rather than grading every completion 0 at zero group variance.

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
