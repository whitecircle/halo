# Datasets

A code-contests row carries the problem statement as `prompt` and its test set as `answer`.

## Payload

- `answer` is the grading payload, a JSON string or dict, and is required (`requires_answer`). A bare test list,
  `{"test_cases": [...]}` and the full form below are accepted.
- A payload that holds no tests or does not parse fails the episode at reset as a rollout error, which the trainer
  drops from the group baseline.

```json
{"prompt": "<problem statement>",
 "answer": "{\"tests\": [{\"input\": \"2 3\\n\", \"output\": \"5\\n\"}], \"checker\": null, \"time_limit\": 2.0}"}
```

`checker` and `time_limit` are per-problem overrides ([Special judges](grading.md#special-judges),
[Time limits](grading.md#time-limits)).

## Adapters

Adapters live in `halo/environments/tasks/coding/contest_datasets.py`:

| Adapter | Source | Role | Notes |
|---|---|---|---|
| `codeforces` | `open-r1/codeforces` | RL pool | `verifiable` config; `generated_checker` judges; interactive rows dropped; generated tests join via `--tests_table`; marks [examples-only problems](#examples-only-problems) |
| `hardtests` | `sigcp/hardtests_problems` + `_tests` | RL pool | Codeforces ratings kept; Luogu's 7 levels (also used by AtCoder, SPOJ and UVa) map to 800-2900, so every non-Codeforces problem of the 2600-3500 band sits at 2900; needs `--tests_table`; judging function becomes the checker |
| `deepcoder` | `agentica-org/DeepCoder-Preview-Dataset` | RL pool | stdin/stdout tests; functional specs skipped; no report bucket |
| `livecodebench` | `livecodebench/code_generation_lite` | benchmark | Functional rows skipped; [contest-date window and platform filter](evaluation.md#livecodebench-window) |
| `icpc` | `RUC-AIBOX/ICPC-Eval` | benchmark | Streamed; `traditional` graded, `spj` skipped |
| `hlce` | `HumanLastCodeExam/icpc-world-finals` | benchmark | Streamed; stdin/stdout `test_cases` |

## Building a pool

`scripts/environments/preparation/prepare_code_dataset.py` builds a training pool from the three RL adapters. It
composes the statement, packs the payload, and drops rows this environment cannot grade.

```bash
python scripts/environments/preparation/prepare_code_dataset.py \
    --adapter hardtests --dataset sigcp/hardtests_problems --min_rating 800 \
    --tests_table "$HALO_DATA_ROOT/hardtests-tests-compact" \
    --holdout_per_band 100 --push_to_hub your-org/code-contests-hardtests-rl --push_bands
```

| Flag | Effect |
|---|---|
| `--min_rating` / `--max_rating` | Bound difficulty; unrated rows are dropped with them |
| `--exclude_keys` | Removes listed ids before the hold-out is carved (held-out and benchmark ids) |
| `--drop_train_keys` | Removes listed ids from the train split after the hold-out is carved, so the test split stays the same: `scripts/environments/preparation/hardtests_train_exclusions.txt` lists the known defects of the HardTests extra-hard band |
| `--holdout_per_band` | Carves a deterministic `test` split |
| `--push_bands` | Publishes `full` plus one config per rating band (`medium` 1500-1999, `hard` 2000-2599, `extra-hard` 2600-3500) over a shared test split. A curriculum stage selects its pool this way (`your-org/code-contests-hardtests-rl:hard`) |
| `--verify_checkers` | On by default. Drops a problem whose checker rejects a test's own reference output, or accepts garbage. Needs a sandbox backend on the preparation host |
| `--reject_constant_answers` | On by default. Drops a problem whose suite accepts one constant answer (`0`, `1`, `-1`, `YES`, `NO`, ...) on every test: a lax checker or a suite of identical answers pays its reward to a program that never reads the input |
| `--tests_table` | Joins a compacted test corpus by problem id. A table may cover some splits only; one matching no row of any split exits before any split is filtered |
| `--include_examples_only` | Keeps the [examples-only problems](#examples-only-problems), dropped and counted per split by default |

A bulky test corpus goes through `scripts/environments/preparation/compact_code_tests.py` first. It reduces
open-r1's generated tests or HardTests' encoded suites to one row per problem: at most 40 tests of up to 256 KB
each, two of which may reach 4 MB so a maximum-size input survives.

### Examples-only problems

An examples-only problem is graded only on its statement's example inputs, which the prompt shows with their
answers, so a program printing those answers passes. The `codeforces` adapter marks a row whose official tests are
the statement's samples and that no joined suite covers: in `open-r1/codeforces` `verifiable`, 219 of the 422 `test`
problems and 1,303 of the 8,299 gradable `train` ones. The preparation script drops them and the
[evaluation](evaluation.md#scoring) leaves them out, each unless `--include_examples_only`.
