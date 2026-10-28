#!/usr/bin/env python
"""
Evaluate a model on a registered environment (QA, exam, SWE, MCP, …) against an OpenAI-compatible
endpoint, reading prompts/answers from dataset columns.

This is the **generic** eval runner. It reads `--prompt_field` / `--answer_field` columns and runs the
env's `reset`/`step` rollout against `--base_url` + `--api_key` (vLLM *or* OpenRouter), reporting
mean reward and success@k. The rollout loop and reporting are shared via
:mod:`halo.environments.eval_runner`.

For competitive programming (`code_contests` / `codeforces`) use
`scripts/environments/inference/eval_code_contests.py` instead; it carries the dataset adapters,
solution language and per-benchmark report buckets, keeping that logic out of this generic runner.

Per-env settings go through `--env_kwargs` (a JSON dict merged into the env config), e.g.
`--env_kwargs '{"search_backend": "duckduckgo"}'` or `'{"open_book": true}'`. Tool-using envs need a
server with tool calling enabled. `--training_config <yaml>` evaluates a trained policy under its own
run's contract — the YAML's environment config and rollout settings (chat-template variables, stop
tokens, thinking budget, sampling) — with any flag passed explicitly laid over them.

Examples:
    # Factual QA over SimpleQA against a local vLLM server
    python scripts/environments/inference/eval_env.py \\
        --env_type qa_search --dataset basicv8vc/SimpleQA --split test \\
        --prompt_field problem --answer_field answer \\
        --base_url http://localhost:8000/v1 --model Qwen/Qwen3.6-35B-A3B --num_examples 100

    # Multiple-choice exam, bucketed by category, via OpenRouter
    python scripts/environments/inference/eval_env.py \\
        --env_type exam_qa --dataset <multiple-choice-dataset> --split test \\
        --prompt_field question --answer_field answer --context_fields choices --group_by subject \\
        --base_url https://openrouter.ai/api/v1 --api_key "$OPENROUTER_API_KEY" --model qwen/qwen3-235b-a22b

`exam_qa` grades multiple choice by letter: the row needs a `choices` column (the option strings,
passed through with `--context_fields choices`) and an `answer` that is a letter A-J or a 0-based int
index into `choices` (a raw `cais/mmlu` ships the index); any other answer, a digit string included,
raises at episode start.
Grading notes: `agent-docs/training-methods/async-grpo/environments/tasks/qa-benchmarks.md`.
"""

import argparse

from halo.configs.script_arguments import DEFAULT_ANSWER_FIELD
from halo.entrypoints.eval_contract import DEFAULT_SPLIT
from halo.entrypoints.generic_env_eval import DEFAULT_ID_FIELD, run_generic_env_eval
from halo.environments.rollout_config import RolloutConfig
from halo.log import configure_cli_logging
from scripts.environments.eval_arguments import add_eval_args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a model on a registered environment via an OpenAI-compatible endpoint."
    )
    p.add_argument(
        "--env_type",
        default=None,
        help="Registered environment name (qa_search, exam_qa, swe, mcp, ...). Required unless "
        "--training_config names one (environment_type).",
    )
    add_eval_args(p)
    p.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split (default: %(default)s).")
    p.add_argument("--prompt_field", default="prompt", help="Row field holding the prompt.")
    p.add_argument(
        "--answer_field",
        default=DEFAULT_ANSWER_FIELD,
        help="Row field holding the expected answer. A renamed field must name a column; the default may be absent.",
    )
    p.add_argument("--context_fields", nargs="*", default=[], help="Extra row fields to pass through as context.")
    p.add_argument("--group_by", default=None, help="Row field to bucket the report by.")
    p.add_argument(
        "--id_field",
        default=DEFAULT_ID_FIELD,
        help="Row field naming an example in the results and trajectories. A renamed field must name a column; "
        "the default may be absent.",
    )
    p.add_argument(
        "--env_kwargs", default="{}", help="JSON dict merged into the env config (e.g. search_backend, open_book)."
    )
    p.add_argument("--num_examples", type=int, default=100, help="Cap on examples (0 = all).")
    p.add_argument("--num_samples", type=int, default=1, help="Episodes per example (success@k).")
    p.add_argument(
        "--success_threshold",
        type=float,
        default=1.0,
        help="Reward at/above which a sample is a success, for an environment that reports no solve verdict "
        "of its own (one that does is scored on it).",
    )
    # No default: each env class carries its own, and passing one unconditionally would cap every env
    # at a number none of them chose.
    p.add_argument("--max_turns", type=int, default=None, help="Max env turns per episode (default: the env's own).")
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=f"Sampling temperature (default: the training config's under --training_config, else "
        f"{RolloutConfig.temperature}).",
    )
    p.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help=f"Max tokens per generation (default: the training config's under --training_config, else "
        f"{RolloutConfig.max_tokens}; reasoning models need a lot).",
    )
    p.add_argument("--max_workers", type=int, default=32, help="Concurrent episodes.")
    return p.parse_args()


def main() -> None:
    configure_cli_logging()
    run_generic_env_eval(parse_args())


if __name__ == "__main__":
    main()
