#!/usr/bin/env python
"""Evaluate a model on competitive-programming problems (`code_contests` / `codeforces` env) against an
OpenAI-compatible endpoint.

Reports `success@1` / `success@k` bucketed by the adapter's report field. A solution counts as solved when it passes
every test in the pool, whatever the shaped reward. The server needs tool calling enabled (vLLM `--tool-call-parser
qwen3_xml --enable-auto-tool-choice`), and reasoning models need a large `--max_tokens`: too low truncates the chain of
thought before any solution and scores 0. A problem whose only tests are its statement's examples is left out and
counted unless `--include_examples_only`.

    python scripts/environments/inference/eval_code_contests.py \\
        --dataset open-r1/codeforces --config verifiable --split test --adapter codeforces \\
        --base_url http://localhost:8000/v1 --model Qwen/Qwen3.6-35B-A3B \\
        --num_examples 50 --max_tokens 24576
"""

import argparse

from halo.entrypoints.code_contests_eval import (
    CODING_ENV_TYPES,
    DEFAULT_ENV_TYPE,
    DEFAULT_TEMPERATURE,
    NO_REASONING_EFFORT,
    SOLUTION_HEADROOM_TOKENS,
    run_code_contests_eval,
)
from halo.entrypoints.eval_contract import DEFAULT_SPLIT
from halo.environments.rollout_config import RolloutConfig
from halo.environments.tasks.coding.code_contests import (
    DEFAULT_REASONING_EFFORT,
    EVAL_PROTOCOLS,
    CodeContestsEnvironment,
)
from halo.environments.tasks.coding.contest_datasets import CODE_DATASET_ADAPTERS
from halo.log import configure_cli_logging
from scripts.environments.eval_arguments import add_eval_args


def _add_selection_args(p: argparse.ArgumentParser) -> None:
    # None lets resolve_split tell an explicit split, which a single-split source refuses, from an omitted one.
    p.add_argument(
        "--split",
        default=None,
        help=f"Dataset split (default: {DEFAULT_SPLIT}). A source that ships a single split of its own reads "
        "that split and refuses another.",
    )
    p.add_argument(
        "--adapter",
        default="codeforces",
        choices=sorted(name for name, a in CODE_DATASET_ADAPTERS.items() if a.scores_raw_rows),
        help="Adapter that composes raw contest rows into eval examples. This script always reads a raw "
        "dataset; one already prepared by scripts/environments/preparation/prepare_code_dataset.py is not its input.",
    )
    dated = ", ".join(sorted(name for name, a in CODE_DATASET_ADAPTERS.items() if a.contest_date))
    p.add_argument(
        "--start_date",
        default=None,
        help=f"Score only problems whose contest is dated on or after this day (YYYY-MM-DD, inclusive). "
        f"Adapters stamping a contest date: {dated}.",
    )
    p.add_argument(
        "--end_date",
        default=None,
        help="Score only problems whose contest is dated on or before this day (YYYY-MM-DD, inclusive).",
    )
    p.add_argument(
        "--platform",
        default=None,
        help="Comma-separated platforms to score, spelled as the dataset spells them ("
        + "; ".join(
            f"{name}: {', '.join(a.platforms)}" for name, a in sorted(CODE_DATASET_ADAPTERS.items()) if a.platforms
        )
        + "). Default: every platform.",
    )
    examples_only = ", ".join(sorted(name for name, a in CODE_DATASET_ADAPTERS.items() if a.examples_only))
    p.add_argument(
        "--include_examples_only",
        action="store_true",
        help="Also score problems whose only tests are the statement's examples, which a program printing a "
        f"constant passes for many (adapters that mark them: {examples_only}). Default: left out and counted.",
    )
    p.add_argument(
        "--eval_protocol",
        default=None,
        choices=sorted(EVAL_PROTOCOLS),
        help="Evaluation protocol: harness runs the configured budgets (the agentic loop); leaderboard pins "
        "one graded submission and no scratchpad runs at every effort level. Default: the training "
        "config's under --training_config, else harness.",
    )
    p.add_argument(
        "--num_examples", type=int, default=50, help="Cap on problems, taken in the adapter's order (0 = all)."
    )
    p.add_argument(
        "--num_samples", type=int, default=1, help="Samples per problem (success@k; success@1 counts the first)."
    )


def _add_generation_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--language",
        default=None,
        help="Solution language to prompt for and grade (python/cpp/c). A comma-separated list lets the "
        "model choose per program and grades each in the language it named. Default: the training config's "
        "under --training_config, else the environment's own.",
    )
    p.add_argument(
        "--max_turns",
        type=int,
        default=None,
        help="Max env turns per episode (default: the training config's under --training_config, else the "
        "environment's own). Agentic models iterate test→fix→submit, so a tight cap (e.g. 6) truncates "
        "them mid-loop.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=f"Sampling temperature (default: the training config's under --training_config, else "
        f"{DEFAULT_TEMPERATURE}).",
    )
    p.add_argument(
        "--reasoning_effort",
        default=None,
        choices=[*sorted(CodeContestsEnvironment.REASONING_EFFORT_PROFILES), NO_REASONING_EFFORT],
        help=f"Solver reasoning effort, passed to the model's chat template (low/medium/high), or "
        f"{NO_REASONING_EFFORT}: no level and no thinking budget, for a non-thinking model. "
        f"{NO_REASONING_EFFORT} sends no reasoning_effort, so a thinking template keeps its default: "
        f"Qwen3/3.5/3.6 still think unless enable_thinking=false reaches the template (vLLM "
        f"--default-chat-template-kwargs). Default: the "
        f"training config's under --training_config (a null there is {NO_REASONING_EFFORT}), else "
        f"{DEFAULT_REASONING_EFFORT}. Sets the default --max_tokens unless --max_tokens or "
        f"--training_config is given.",
    )
    p.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Max tokens per generation. Default: the training config's rollout_max_tokens under "
        "--training_config, else the effort profile's thinking budget ("
        + ", ".join(
            f"{level}={p['thinking_tokens']}" for level, p in CodeContestsEnvironment.REASONING_EFFORT_PROFILES.items()
        )
        + ", or a reasoning_effort_profiles override in --env_kwargs)"
        + f" plus {SOLUTION_HEADROOM_TOKENS} solution headroom; under --reasoning_effort {NO_REASONING_EFFORT}, "
        f"the training rollout's default {RolloutConfig.max_tokens}, and under a soft budget the larger of the two.",
    )
    p.add_argument("--max_workers", type=int, default=16, help="Concurrent episodes.")
    p.add_argument(
        "--env_kwargs",
        default="{}",
        help="JSON dict merged into the env config, for the grading knobs this script does not expose "
        "(e.g. stop_on_first_failure, timeout_per_test, max_submissions, sandbox_backend).",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a model on competitive programming (code_contests/codeforces).")
    p.add_argument(
        "--env_type",
        default=None,
        choices=CODING_ENV_TYPES,
        help=f"Coding env (default: the training config's environment_type under --training_config, else "
        f"{DEFAULT_ENV_TYPE}).",
    )
    add_eval_args(p)
    _add_selection_args(p)
    _add_generation_args(p)
    return p.parse_args()


def main() -> None:
    configure_cli_logging()
    run_code_contests_eval(parse_args())


if __name__ == "__main__":
    main()
