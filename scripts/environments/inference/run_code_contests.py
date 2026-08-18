#!/usr/bin/env python
"""
Evaluate a model on competitive-programming problems (`code_contests` / `codeforces` env) against an
OpenAI-compatible endpoint.

Specific to the coding-contest task: it applies a dataset adapter (`codeforces`, `deepcoder`, `hlce`,
`icpc` or `livecodebench`) to a raw contest dataset, prompts in a chosen solution `language`, and
reports `success@1` / `success@k`
bucketed by problem rating. The rollout loop and reward aggregation are shared with the other eval
scripts via
:mod:`src.environments.eval_runner`.

The server must serve the model with tool calling enabled (e.g. vLLM
`--tool-call-parser qwen3_xml --enable-auto-tool-choice`). A solution counts as solved when it passes
every graded test (`--success_threshold 1.0`). Reasoning models need a large `--max_tokens` — too low
truncates the chain of thought before any solution and scores 0.

Examples:
    # Codeforces verifiable test split, 50 problems, success@1, on a local vLLM server
    python scripts/environments/inference/run_code_contests.py \
        --dataset open-r1/codeforces --config verifiable --split test --adapter codeforces \
        --base_url http://localhost:8000/v1 --model Qwen/Qwen3.6-35B-A3B \
        --num_examples 50 --max_tokens 24576

    # DeepCoder (taco), success@4 against OpenRouter
    python scripts/environments/inference/run_code_contests.py \
        --dataset agentica-org/DeepCoder-Preview-Dataset --config taco --split train --adapter deepcoder \
        --base_url https://openrouter.ai/api/v1 --api_key "$OPENROUTER_API_KEY" \
        --model qwen/qwen3-235b-a22b --num_examples 100 --num_samples 4
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
from src.environments.envs.tasks.coding.code_contests import DEFAULT_REASONING_EFFORT, REASONING_EFFORT_PROFILES
from src.environments.envs.tasks.coding.datasets import CODE_DATASET_ADAPTERS, CodeDatasetAdapter
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

# thinking_tokens is the CoT budget, not a completion cap — headroom keeps the solution itself (and
# the tool call carrying it) from competing with the chain of thought for tokens.
SOLUTION_HEADROOM_TOKENS = 4096


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a model on competitive programming (code_contests/codeforces).")
    p.add_argument("--env_type", default="codeforces", choices=["codeforces", "code_contests"], help="Coding env.")
    add_endpoint_args(p)
    p.add_argument(
        "--adapter",
        default="codeforces",
        choices=sorted(CODE_DATASET_ADAPTERS),
        help="Adapter that composes raw contest rows into eval examples. This script always reads a raw "
        "dataset; one already prepared by scripts/environments/preparation/prepare_code_dataset.py is not its input.",
    )
    p.add_argument("--num_examples", type=int, default=50, help="Cap on problems (0 = all).")
    p.add_argument("--num_samples", type=int, default=1, help="Samples per problem (success@k).")
    p.add_argument("--language", default="python", help="Solution language to prompt for and grade (python/cpp/c).")
    p.add_argument(
        "--max_turns",
        type=int,
        default=15,
        help="Max env turns per episode. Agentic models iterate test→fix→submit, so a tight cap "
        "(e.g. 6) truncates them mid-loop.",
    )
    p.add_argument("--success_threshold", type=float, default=1.0, help="Reward at/above which a sample is solved.")
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    p.add_argument(
        "--reasoning_effort",
        default=DEFAULT_REASONING_EFFORT,
        choices=sorted(REASONING_EFFORT_PROFILES),
        help="Solver reasoning effort, passed to the model's chat template (low/medium/high). Sets the "
        "default --max_tokens unless --max_tokens is given.",
    )
    p.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Max tokens per generation. Default: the effort profile's thinking budget "
        + ", ".join(f"{level}={p['thinking_tokens']}" for level, p in REASONING_EFFORT_PROFILES.items())
        + f" plus {SOLUTION_HEADROOM_TOKENS} solution headroom.",
    )
    p.add_argument("--max_workers", type=int, default=16, help="Concurrent episodes.")
    p.add_argument(
        "--env_kwargs",
        default="{}",
        help="JSON dict merged into the env config, for the grading knobs this script does not expose "
        "(e.g. stop_on_first_failure, timeout_per_test, max_submissions, sandbox_backend).",
    )
    return p.parse_args()


def build_examples(args: argparse.Namespace, adapter: CodeDatasetAdapter) -> list[dict[str, Any]]:
    """Compose raw contest rows into eval examples via the chosen adapter, bucketed by its group field.

    Loading goes through the adapter's own ``load`` when it has one (LiveCodeBench / ICPC-Eval can't be
    read with a plain ``load_dataset``), otherwise the standard HF split loader.
    """
    rows = (
        adapter.load(args.dataset, args.config, args.split)
        if adapter.load
        else load_hf_split(args.dataset, args.config, args.split)
    )
    examples = []
    for row in rows:
        if not adapter.keep(row):
            continue
        examples.append(
            {
                "prompt": adapter.format_prompt(row),
                "context": {"answer": json.dumps(adapter.pack_verification(row))},
                "group": row.get(adapter.group_field),
                "id": row.get("id") or row.get("problem_id") or row.get("name"),
            }
        )
        if args.num_examples and len(examples) >= args.num_examples:
            break
    logger.info("Loaded %d problems from %s (%s)", len(examples), args.dataset, args.adapter)
    return examples


def main() -> None:
    args = parse_args()
    adapter = CODE_DATASET_ADAPTERS[args.adapter]
    env_kwargs = json.loads(args.env_kwargs)
    env = resolve_environment(
        args.env_type,
        {
            "max_turns": args.max_turns,
            "language": args.language,
            "reasoning_effort": args.reasoning_effort,
            **env_kwargs,
        },
    )
    examples = build_examples(args, adapter)
    client = create_openai_client(base_url=args.base_url, api_key_override=args.api_key)

    # The effort level sets the generation budget unless the caller overrode --max_tokens: too small a
    # budget truncates the chain of thought before any solution and scores the problem 0.
    max_tokens = (
        args.max_tokens
        or REASONING_EFFORT_PROFILES[args.reasoning_effort]["thinking_tokens"] + SOLUTION_HEADROOM_TOKENS
    )
    logger.info("reasoning_effort=%s, max_tokens=%d", args.reasoning_effort, max_tokens)

    traj_path = resolve_trajectory_path(args, args.adapter, args.split, args.language)

    results = asyncio.run(
        collect_results(
            env,
            examples,
            client,
            rollout=rollout_config_from_args(args, temperature=args.temperature, max_tokens=max_tokens),
            num_samples=args.num_samples,
            success_threshold=args.success_threshold,
            max_workers=args.max_workers,
            collect_trajectories=bool(traj_path),
        )
    )
    report(
        results,
        num_samples=args.num_samples,
        title=f"{args.env_type} on {args.dataset} ({args.adapter})",
        group_label=adapter.group_label,
    )
    write_eval_outputs(
        args,
        results,
        env=env,
        traj_path=traj_path,
        env_type=args.env_type,
        max_turns=args.max_turns,
        max_tokens=max_tokens,
        temperature=args.temperature,
        num_samples=args.num_samples,
        meta_extra={
            "adapter": args.adapter,
            "language": args.language,
            "reasoning_effort": args.reasoning_effort,
            "env_kwargs": env_kwargs,
            # The run's whole grading contract, so an offline re-grade reproduces the SAME verdicts
            # (per-problem checker/time_limit come from the dataset payload). Derived from the
            # dataclass, so a knob added to GradingSpec cannot be silently defaulted offline.
            "env_grading": env.grading_spec.to_meta(),
        },
    )


if __name__ == "__main__":
    main()
