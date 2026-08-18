# Benchmark Environments

Environments for evaluating RL-trained models against common benchmarks. All use rule-based rewards, no neural reward models.

## SearchQA (`qa_search`)

Factual question answering with web search (SimpleQA, GAIA, TriviaQA). `create_qa_search_environment` (`src/environments/envs/tasks/qa.py`) is a factory preset over `NativeToolUseEnvironment` — search tools, a research-assistant prompt, and `require_tool_use=True` — not a dedicated class.

```python
from src.environments.envs.tasks.qa import create_qa_search_environment

env = create_qa_search_environment(
    max_turns=10,
    search_backend="duckduckgo",  # serper | brave | tavily | duckduckgo; None (default) auto-selects by API key
    include_python_tools=True,    # default False; adds the Python REPL for numeric QA
)
```

A fifth backend, `mock`, returns fabricated snippets and is refused unless `HALO_ALLOW_MOCK_SEARCH=1` is set: its results pay `tool_success_reward` like a real search, so a run that reached it would teach the policy that invented evidence works. It exists for UI and test runs.

**Tools:** `web_search` + optional `python`. **Reward:** validates the final response against `context["answer"]` (exact match or numeric tolerance; fuzzy partial credit off by default). `require_tool_use` flags a response that never called a tool; the charge for it is `no_tool_use_penalty` (default 0).

**Dataset format:** `{"prompt": "What year was the Eiffel Tower completed?", "answer": "1889"}`

```yaml
environment_type: qa_search
max_turns: 10
environment_kwargs:
  search_backend: duckduckgo
  include_python_tools: false
```

## ExamQA (`exam_qa`)

Multiple-choice and open-ended exam questions (MMLU-Pro, GPQA, MMLU, ARC). `max_turns` defaults to 8.

```python
from src.environments.envs.tasks.qa import ExamQAEnvironment

env = ExamQAEnvironment(success_reward=1.0)                           # closed-book
env = ExamQAEnvironment(open_book=True, search_backend="duckduckgo")  # open-book (web search)
```

**Tools:** none (closed-book) or `web_search` (open-book). **Reward:** multiple-choice extracts the choice letter (A–J, up to 10 choices) and compares it to the expected answer; open-ended uses the standard validation chain. A `choices` field in the row triggers multiple-choice grading, and its entries are appended to the prompt:

```json
{
  "prompt": "Which is the largest planet in our solar system?",
  "answer": "B",
  "choices": ["A: Mars", "B: Jupiter", "C: Saturn", "D: Neptune"]
}
```

Open-ended omits `choices`: `{"prompt": "What is the capital of France?", "answer": "Paris"}`

`answer` may be the letter or a **0-based index into `choices`** (MMLU and ARC ship an int): `ExamQAEnvironment._expected_choice_letter` normalizes it at reset, since `multiple_choice_match` (its neighbour in `qa.py`) scores only letters. Any other shape — an out-of-range index, a bool, free text — raises at episode start rather than scoring every completion `failure_reward` and handing the GRPO group zero variance.

## CodeContests & Codeforces

Competitive programming with hidden-test grading. See [Code Contests](code-contests.md).

## Evaluating a model on an environment

Two eval scripts share one rollout/reporting core (`src/environments/eval_runner.py`) and one CLI surface (`scripts/environments/_common.py`). Both run the same `reset`/`step` loop as training against an OpenAI-compatible endpoint — vLLM or OpenRouter — via `--base_url` + `--api_key` + `--model`. Sampling is one `RolloutConfig`, the object the training rollout hands its actors, so an eval draws from the policy the way training did; `--top_p` defaults to the training rollout's `0.95` rather than the server's own default, and the recorded trajectory meta states it alongside `temperature` and `max_tokens`. Tool-using envs need tool calling enabled (e.g. vLLM `--tool-call-parser ... --enable-auto-tool-choice`).

A sample is a *success* when its reward ≥ `--success_threshold` (default `1.0`); `--num_samples k` adds `success@k`. Reasoning models need a large `--max_tokens` — too low truncates the chain of thought and scores 0. The client (`create_openai_client`) defaults to `max_retries=4`. `--temperature` defaults to `0.7` on `run_env.py` and `0.2` on `run_code_contests.py`.

The report logs per-run trajectory telemetry (mean turns, `used_tools` %, mean tool calls, mean completion tokens, `length_capped` %), so token starvation is distinguishable from wrong answers.

**`run_code_contests.py`** — competitive programming; applies a dataset adapter (`codeforces`, `deepcoder`, `livecodebench`, `icpc`, `hlce`), prompts in `--language`, buckets by the adapter's group field. See [Code Contests](code-contests.md).

**`run_env.py`** — generic runner for the other envs (QA, exam, SWE, MCP); reads `--prompt_field` / `--answer_field` columns, extra columns via `--context_fields`, optional `--group_by`. `--dataset` takes a Hub id or a `save_to_disk` directory (a bare `Dataset` or a `DatasetDict`, in which case `--split` selects).

```bash
python scripts/environments/inference/run_env.py \
    --env_type qa_search --dataset basicv8vc/SimpleQA --split test \
    --prompt_field problem --answer_field answer \
    --base_url https://openrouter.ai/api/v1 --api_key "$OPENROUTER_API_KEY" \
    --model qwen/qwen3-235b-a22b --num_examples 50
```

Both take `--env_kwargs` (a JSON dict merged into the env config) for the per-env settings their own flags do not cover.

### Recording trajectories

Pass `--save_trajectories <path.jsonl>` to either script to persist the full run. Line 1 is a `{"type": "meta", ...}` record (model, env/dataset/split, sampling params, `system_prompt`, `tools` schemas, plus whatever the task runner adds — adapter, language, grading knobs); each following line is a `{"type": "episode", ...}` record addressable by `index` and `id`, with `group`, `sample_index` (the sample's position under `--num_samples`), `reward`, `success`, `stats`, and the serialized trajectory (`messages`, `total_reward`, `done`, `truncated`, the episode's `reasoning_effort` / `reasoning_budget`, and `info`).

Two contracts govern that record. `serialize_trajectory` never **leaks the answer key**: `info` drops every `_`-prefixed field (hidden tests, checker source, per-problem time limit), the raw `tool_calls` log, and `context`, keeping the grading verdict. The one underscore key that survives is `_eval_stats`, re-keyed to `eval_stats` — per-episode telemetry, not answer-key data.

No chain-of-thought is persisted either. The eval stamps the engine's reasoning channel onto the assistant message exactly as the training rollout does, so an env reading it behaves identically online and offline, but the file is written through the same reasoning-free `to_dict` the generation request uses.

For a sweep, pass `--trajectory_dir <folder>` instead — each run is auto-named `<model>__<adapter>__<split>__<language>.jsonl` for the coding script, `<model>__<env_type>__<split>.jsonl` for the generic one.

### Offline re-grading (contention-free)

Grading runs untrusted solutions against a wall-clock limit, so at extreme aggregate concurrency that wall-clock inflates under host load and a correct, fast solution can be spuriously TIME-LIMIT-EXCEEDED. Decouple generation from grading: record trajectories during the parallel sweep, then re-grade offline in one concurrency-bounded process where each run gets a dedicated core (see the [concurrency gate](sandbox.md#concurrency-gate)).

```bash
python scripts/environments/inference/regrade_trajectories.py \
    /mnt/eval/trajectories/*.jsonl --workers 64 --output /mnt/eval/regraded.jsonl
```

`regrade_trajectories.py` reads each file's meta record, rebuilds every problem's hidden tests from its dataset by `index` (the dump drops them), and re-runs each episode's recorded `submit_solution` calls, up to the submission budget that episode ran under (its `episode_max_submissions` stamp, else the env's `max_submissions`), through the same `grade_solution` the environment uses, on the `GradingSpec` the run itself was scored under — stamped whole into the meta line's `env_grading` block (`GradingSpec.to_meta()`, derived from the dataclass so a knob added to it cannot be silently defaulted offline) and applied over the rebuilt environment's own spec, which supplies the sandbox. A block written under a retired field spelling is refused rather than ignored. Two knobs are deliberately dropped offline: grading short-circuits at the first failing test (`s@k` needs the all-pass verdict, not the pass fraction) and the per-submission `max_grading_seconds` budget does not apply (it exists to stop one episode pinning a rollout worker). Reports `s@1` / `s@2` per file. Keep `--workers` at or below the host core count; datasets are loaded once and cached across a model × language matrix.

Re-grading needs the code-contest meta: `env_type`, `adapter`, `dataset`, `model` and `language` must all be present, and the script raises before grading a file that lacks one. Only `run_code_contests.py` stamps the full set — `run_env.py` writes the generic eval meta (no `adapter`/`language`), so its dumps cannot be re-graded.

## Shared reward validation

All benchmark environments validate through `src/environments/rewards.py`: `exact_match` (case-insensitive after normalization), `numeric_match` (extract numbers, compare with `rtol=0.01` / `atol=1e-6`, percentages handled), `validate_answer` (full chain), and `compute_answer_reward`. Grading is all-or-nothing: the default chain is exact + numeric, and a match pays `success_reward`, anything else `failure_reward`. Letter grading (`multiple_choice_match`) lives beside its only caller in `src/environments/envs/tasks/qa.py`.

Normalization handles `\boxed{42}`, `**42**`, "The answer is…", "Therefore,…", and surrounding whitespace. It takes the `\boxed{...}` whose **opening** is rightmost — the last box among siblings, the **inner** one when boxes nest — matching braces by depth so nested LaTeX survives (`\boxed{\frac{1}{2}}`). It does not split a GSM8K-style `#### N` rationale suffix: reduce such an `answer` column to the final value before it reaches the environment.

Online GRPO's RLVR script deliberately does **not** use this chain — its `accuracy_reward` is a strict boxed exact-match that splits `####` and strips `,`/`$` instead ([RLVR reward functions](../online-grpo.md#reward-functions)). The two graders score the same row differently; a recipe picks one.
