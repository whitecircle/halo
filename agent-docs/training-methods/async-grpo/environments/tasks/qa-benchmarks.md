# QA Benchmark Environments

Two registry names cover question answering with rule-based rewards (`src/environments/envs/tasks/qa.py`). Both speak native tool calls, so a tool-carrying variant needs a tool-call parser on the server ([Tool calls](../../episodes.md#tool-calls)).

| Name | Task | Shape |
|---|---|---|
| `qa_search` | Factual QA with web search (SimpleQA, GAIA, TriviaQA, PopQA) | A factory preset over `NativeToolUseEnvironment`: search tools, a research-assistant prompt, a required `answer` column |
| `exam_qa` | Multiple-choice or open-ended exams (MMLU-Pro, GPQA, MMLU, ARC) | `ExamQAEnvironment`, closed-book by default |

Shipped recipe: `examples/grpo/environmental/qwen3_5/vllm/qwen3.6-35b-a3b-exam-qa-full-ep4.yaml`.

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
| `search_backend` | auto | `serper`, `brave`, `tavily`, `duckduckgo`; auto picks by which API key is set, keyless DuckDuckGo last. Validated at construction |
| `include_python_tools` | `false` | `qa_search` only: adds the in-process restricted `python` REPL |
| `open_book` | `false` | `exam_qa` only: registers the search tool. Setting `search_backend` closed-book raises |
| `system_prompt` | class prompt | Replaces the built-in instructions |

A fifth backend, `mock`, returns fabricated snippets and is refused unless `HALO_ALLOW_MOCK_SEARCH=1`. Its results pay `tool_success_reward`, so training on it teaches the policy that invented evidence works.

## Tools

- `web_search` — query plus optional `max_results` (5); returns titles, snippets and URLs. Always on `qa_search`; on `exam_qa` under `open_book`.
- `python` — on `qa_search` under `include_python_tools`.

## Reward

Grading is all-or-nothing, 1 or 0 (`src/rewards/graders/matching.py`), priced by the reward's `environment` term ([Reward Terms](../../../grpo/rewards/README.md#environment-arm)).

1. `exact_match` — case-insensitive, after normalization.
2. `numeric_match` — every number in the response against the one value the expected answer states (`$18` and `18 dollars` expect 18). Tolerance `rtol=0.01` / `atol=1e-6`; percentages are divided by 100, and `3,500` or `10\,000` read as one number.

There is no fuzzy or substring matcher: "7" must not match "17".

Online GRPO's `accuracy` term grades the same row differently: a strict boxed exact match that splits `####` and strips `,`/`$` ([Online GRPO → Rewards](../../../grpo/online.md#rewards)).

### One committed value

The response must commit to one value. These grade 0:

- a hedge (`7 or 8`);
- a correct answer restated with other numbers (`7, since 3 + 4 = 7`);
- a number inside arithmetic, a power, root, constant or function (`1/2`, `10^3`, `\sqrt{2}`, `2\pi`, `log 2`), or joined to a variable (`1/x`, `n+1`);
- a stated bound (`x < 3`).

Not operands: a power after a letter is a unit exponent (`9.8 m/s^2`), and a digit after a letter is part of a token
(`H2O`), not a number. A number before a letter reads as a number plus a unit, so `2x` is not caught, nor are word
forms (`square root of 2`).

### Normalization

- Reads `\boxed{42}`, a lone `**42**`, and leading "The answer is" / "Therefore". Several bold spans (`**7** or **8**`) stay a hedge.
- Takes the rightmost `\boxed{...}` and matches braces by depth, so `\boxed{\frac{1}{2}}` survives.
- Does not split a GSM8K-style `#### N` suffix: reduce such an `answer` column to the final value before training.

### Multiple choice

A row with `choices` switches `exam_qa` to letter grading and appends the choices to the prompt, each led by its letter. A choice already labelled with its own letter (`"B: Jupiter"`) is shown as written; more than ten choices are refused.

- The response's letter (A–J) is extracted ("A", "(A)", "The answer is A") and compared to the expected one. In prose only an uppercase letter standing alone counts ("the answer is a tie" and "the answer is definitely C" name no choice); a bracketed or bare letter may be lowercase.
- `answer` may be that letter or a 0-based int index into `choices` (MMLU ships the index).
- A digit string raises: convert ARC's numeric `answerKey` labels (`"1"`–`"5"`, 1-based) to letters when preparing the data.
- Anything else, an out-of-range index included, raises at episode start.

## Dataset

```json
{"prompt": "What year was the Eiffel Tower completed?", "answer": "1889"}
{"prompt": "Largest planet?", "answer": "B", "choices": ["A: Mars", "B: Jupiter", "C: Saturn"]}
```

- Name `choices` in `context_fields` (training) or `--context_fields choices` (eval), or it never reaches the environment.
- `answer` is required for both environments (`requires_answer`).

## Evaluation

```bash
python scripts/environments/inference/run_env.py --env_type exam_qa \
    --dataset <hub-id-or-dir> --split test --prompt_field question --answer_field answer \
    --context_fields choices --group_by subject \
    --base_url http://localhost:8000/v1 --model <served-name> --num_examples 100
```

Flags, output files and the `--training_config` contract: [Evaluation](../../evaluation.md).

## Related pages

- [Native Tool-Use](../protocols/native-tool-use.md) — the protocol both environments run on.
- [Code Contests](code-contests/README.md) — hidden-test competitive programming.
- [Environments](../README.md) — registry and `environment_kwargs` plumbing.
