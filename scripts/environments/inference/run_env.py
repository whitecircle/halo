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
solution language and per-benchmark report buckets, keeping that logic out of this generic runner.

Per-env settings go through `--env_kwargs` (a JSON dict merged into the env config), e.g.
`--env_kwargs '{"search_backend": "duckduckgo"}'` or `'{"open_book": true}'`. Tool-using envs need a
server with tool calling enabled. `--training_config <yaml>` evaluates a trained policy under its own
run's contract — the YAML's environment config and rollout settings (chat-template variables, stop
tokens, thinking budget, sampling) — with any flag passed explicitly laid over them.

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
    load_training_contract,
    resolve_setting,
    resolve_trajectory_path,
    rollout_config_from_args,
    write_eval_outputs,
)
from src.args.environmental_grpo_args import DEFAULT_ANSWER_FIELD
from src.configs.rollout_config import DEFAULT_ROLLOUT_MAX_TOKENS, DEFAULT_ROLLOUT_TEMPERATURE
from src.environments.eval_runner import (
    collect_results,
    load_hf_split,
    report,
    require_answers,
)
from src.environments.registry import resolve_environment
from src.inference.openai_client import create_openai_client
from src.log import configure_cli_logging

configure_cli_logging()
logger = logging.getLogger(__name__)

# The column naming an example when --id_field is not given; a dataset may carry none.
DEFAULT_ID_FIELD = "id"


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
    add_endpoint_args(p)
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
        f"{DEFAULT_ROLLOUT_TEMPERATURE}).",
    )
    p.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help=f"Max tokens per generation (default: the training config's under --training_config, else "
        f"{DEFAULT_ROLLOUT_MAX_TOKENS}; reasoning models need a lot).",
    )
    p.add_argument("--max_workers", type=int, default=32, help="Concurrent episodes.")
    return p.parse_args()


def require_field_columns(args: argparse.Namespace, columns: list[str]) -> None:
    """Refuse a field flag that names no column of the split, before a row is read: a mistyped
    ``--prompt_field`` would skip every row, and a mistyped answer, context field, bucket or id would
    vanish from every example. The default ``--answer_field`` and ``--id_field`` go unchecked: a dataset
    may carry neither, and whether an answer is needed is the environment's ``requires_answer``."""
    named = [("--prompt_field", args.prompt_field)]
    named += [("--answer_field", args.answer_field)] if args.answer_field != DEFAULT_ANSWER_FIELD else []
    named += [("--context_fields", field) for field in args.context_fields]
    named += [("--group_by", args.group_by)] if args.group_by is not None else []
    named += [("--id_field", args.id_field)] if args.id_field != DEFAULT_ID_FIELD else []
    missing = [f"{flag} {column!r}" for flag, column in named if column not in columns]
    if missing:
        raise SystemExit(
            f"{', '.join(missing)}: no such column in {args.dataset}; available columns: {sorted(columns)}"
        )


def build_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Read eval examples from the dataset's prompt/answer columns, every named field checked first."""
    split = load_hf_split(args.dataset, args.config, args.split)
    require_field_columns(args, split.column_names)
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
                "id": row.get(args.id_field),
            }
        )
        if args.num_examples and len(examples) >= args.num_examples:
            break
    logger.info("Loaded %d examples from %s", len(examples), args.dataset)
    return examples


def main() -> None:
    args = parse_args()
    contract = load_training_contract(args.training_config)
    trained_env = contract.env_config_dict() if contract is not None else {}
    env_type = resolve_setting(args.env_type, contract.env_config.environment_type if contract else None, None)
    if env_type is None:
        raise SystemExit("--env_type is required unless --training_config names an environment_type")
    env_kwargs = json.loads(args.env_kwargs)
    turns_override = {"max_turns": args.max_turns} if args.max_turns is not None else {}
    # The training run's env config first, the flags over it: an eval under a contract grades as the run did.
    env = resolve_environment(env_type, {**trained_env, **turns_override, **env_kwargs})
    # A judge or reward-model term is probed before any episode runs, as the trainer does at launch.
    env.verify_backend()
    examples = build_examples(args)
    require_answers(env, examples, f"the {args.answer_field!r} field of {args.dataset} (--answer_field)")
    client = create_openai_client(base_url=args.base_url, api_key_override=args.api_key)
    rollout = rollout_config_from_args(
        args, contract, default_temperature=DEFAULT_ROLLOUT_TEMPERATURE, default_max_tokens=DEFAULT_ROLLOUT_MAX_TOKENS
    )

    traj_path = resolve_trajectory_path(args, env_type, args.split)

    results = asyncio.run(
        collect_results(
            env,
            examples,
            client,
            rollout=rollout,
            num_samples=args.num_samples,
            success_threshold=args.success_threshold,
            max_workers=args.max_workers,
            collect_trajectories=bool(traj_path),
        )
    )
    env.close()
    report(results, num_samples=args.num_samples, title=f"{env_type} on {args.dataset}", group_label=args.group_by)
    write_eval_outputs(
        args,
        results,
        env=env,
        traj_path=traj_path,
        env_type=env_type,
        max_turns=args.max_turns,
        rollout=rollout,
        num_samples=args.num_samples,
        meta_extra={"env_kwargs": {**trained_env, **env_kwargs}},
    )


if __name__ == "__main__":
    main()
