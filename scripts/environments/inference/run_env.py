#!/usr/bin/env python
"""
Evaluate a model on a registered environment (QA, exam, SWE, MCP, …) against an OpenAI-compatible
endpoint, reading prompts/answers from dataset columns.

This is the **generic** eval runner. It reads `--prompt_field` / `--answer_field` columns and runs the
env's `reset`/`step` rollout against `--base_url` + `--api_key` (vLLM *or* OpenRouter), reporting
mean reward and success@k. The rollout loop and reporting are shared via
:mod:`src.environments.eval_runner`.

For competitive programming (`code_contests` / `codeforces`) use
`scripts/environments/inference/run_code_contests.py` instead; it carries the dataset adapters,
solution language and rating-bucketed reporting, keeping that logic out of this generic runner.

Per-env settings go through `--env_kwargs` (a JSON dict merged into the env config), e.g.
`--env_kwargs '{"search_backend": "duckduckgo"}'` or `'{"open_book": true}'`. Tool-using envs need a
server with tool calling enabled.

Examples:
    # Factual QA over SimpleQA against a local vLLM server
    python scripts/environments/inference/run_env.py \
        --env_type qa_search --dataset basicv8vc/SimpleQA --split test \
        --prompt_field problem --answer_field answer \
        --base_url http://localhost:8000/v1 --model Qwen/Qwen3.6-35B-A3B --num_examples 100

    # Multiple-choice exam, bucketed by category, via OpenRouter
    python scripts/environments/inference/run_env.py \
        --env_type exam_qa --dataset <letter-answer-mc-dataset> --split test \
        --prompt_field question --answer_field answer --context_fields choices --group_by subject \
        --base_url https://openrouter.ai/api/v1 --api_key "$OPENROUTER_API_KEY" --model qwen/qwen3-235b-a22b

`exam_qa` grades multiple choice by letter: the row needs a `choices` column (the option strings,
passed through with `--context_fields choices`) and an `answer` that is already a letter A-J. Pointed
at a raw `cais/mmlu`, whose `answer` is an integer index, every episode scores zero without an error.
Convert the index to a letter during dataset preparation; nothing on the eval path can, since the
grader compares two strings and never sees the choice ordering. Schema and dataset notes:
`agent-docs/training-methods/grpo/environments/custom-environments.md`.
"""

import argparse
import asyncio
import json
import logging
from typing import Any

from scripts.environments._common import (
    add_endpoint_args,
    resolve_trajectory_path,
    rollout_config_from_args,
    write_eval_outputs,
)
from src.environments.eval_runner import (
    collect_results,
    load_hf_split,
    report,
)
from src.environments.registry import resolve_environment
from src.inference.openai_client import create_openai_client
from src.log import configure_cli_logging

configure_cli_logging()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a model on a registered environment via an OpenAI-compatible endpoint."
    )
    p.add_argument(
        "--env_type", required=True, help="Registered environment name (qa_search, exam_qa, swe, mcp, ...)."
    )
    add_endpoint_args(p)
    p.add_argument("--prompt_field", default="prompt", help="Row field holding the prompt.")
    p.add_argument("--answer_field", default="answer", help="Row field holding the expected answer.")
    p.add_argument("--context_fields", nargs="*", default=[], help="Extra row fields to pass through as context.")
    p.add_argument("--group_by", default=None, help="Row field to bucket the report by.")
    p.add_argument(
        "--env_kwargs", default="{}", help="JSON dict merged into the env config (e.g. search_backend, open_book)."
    )
    p.add_argument("--num_examples", type=int, default=100, help="Cap on examples (0 = all).")
    p.add_argument("--num_samples", type=int, default=1, help="Episodes per example (success@k).")
    p.add_argument("--success_threshold", type=float, default=1.0, help="Reward at/above which a sample is a success.")
    # No default: each env class carries its own, and passing one unconditionally would cap every env
    # at a number none of them chose.
    p.add_argument("--max_turns", type=int, default=None, help="Max env turns per episode (default: the env's own).")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    p.add_argument(
        "--max_tokens", type=int, default=32768, help="Max tokens per generation (reasoning models need a lot)."
    )
    p.add_argument("--max_workers", type=int, default=32, help="Concurrent episodes.")
    return p.parse_args()


def build_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Read eval examples from the dataset's prompt/answer columns."""
    split = load_hf_split(args.dataset, args.config, args.split)
    examples = []
    for row in split:
        prompt = row.get(args.prompt_field)
        if prompt is None:
            continue
        context = {}
        if args.answer_field in row:
            context["answer"] = row[args.answer_field]
        for field in args.context_fields:
            if field in row:
                context[field] = row[field]
        examples.append(
            {
                "prompt": prompt,
                "context": context,
                "group": row.get(args.group_by),
                "id": row.get("id") or row.get("problem_id"),
            }
        )
        if args.num_examples and len(examples) >= args.num_examples:
            break
    logger.info("Loaded %d examples from %s", len(examples), args.dataset)
    return examples


def main() -> None:
    args = parse_args()
    env_kwargs = json.loads(args.env_kwargs)
    turns_override = {"max_turns": args.max_turns} if args.max_turns is not None else {}
    env = resolve_environment(args.env_type, {**turns_override, **env_kwargs})
    examples = build_examples(args)
    client = create_openai_client(base_url=args.base_url, api_key_override=args.api_key)

    traj_path = resolve_trajectory_path(args, args.env_type, args.split)

    results = asyncio.run(
        collect_results(
            env,
            examples,
            client,
            rollout=rollout_config_from_args(args, temperature=args.temperature, max_tokens=args.max_tokens),
            num_samples=args.num_samples,
            success_threshold=args.success_threshold,
            max_workers=args.max_workers,
            collect_trajectories=bool(traj_path),
        )
    )
    report(
        results, num_samples=args.num_samples, title=f"{args.env_type} on {args.dataset}", group_label=args.group_by
    )
    write_eval_outputs(
        args,
        results,
        env=env,
        traj_path=traj_path,
        env_type=args.env_type,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_samples=args.num_samples,
        meta_extra={"env_kwargs": env_kwargs},
    )


if __name__ == "__main__":
    main()
