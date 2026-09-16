#!/usr/bin/env python
"""
Evaluate a model on competitive-programming problems (`code_contests` / `codeforces` env) against an
OpenAI-compatible endpoint.

Specific to the coding-contest task: it applies a dataset adapter that scores raw rows (every
``CODE_DATASET_ADAPTERS`` entry without a ``normalize`` step) to a contest dataset, prompts in a chosen
solution `language`, and reports `success@1` / `success@k` bucketed by problem rating. The rollout loop
and reward aggregation are shared with the other eval scripts via :mod:`src.environments.eval_runner`.

The server must serve the model with tool calling enabled (e.g. vLLM
`--tool-call-parser qwen3_xml --enable-auto-tool-choice`). A solution counts as solved when it passes
every graded test (`--success_threshold 1.0`). Reasoning models need a large `--max_tokens`: too low
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
    load_training_contract,
    resolve_setting,
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

# thinking_tokens is the CoT budget rather than a completion cap; the headroom keeps the solution and
# the tool call carrying it from competing with the chain of thought for tokens.
SOLUTION_HEADROOM_TOKENS = 4096

# The coding envs this script's adapters, language prompt and rating buckets are written for.
CODING_ENV_TYPES = ("codeforces", "code_contests")
# Script defaults, applied where neither a flag nor ``--training_config`` sets the knob. No language
# or turn budget among them: both are the env class's own, and a script-side copy would silently grade
# under a different contract than the class ships once the class moves.
DEFAULT_ENV_TYPE = "codeforces"
DEFAULT_TEMPERATURE = 0.2


def parse_language_flag(value: str) -> str | list[str]:
    """``--language`` as the env's ``language``: one name, or the list a comma-separated value names."""
    languages = [name.strip() for name in value.split(",") if name.strip()]
    if not languages:
        raise SystemExit(f"--language names no language: {value!r}")
    return languages if len(languages) > 1 else languages[0]


def refuse_env_kwargs_language(env_kwargs: dict) -> None:
    """``--language`` names the trajectory path and the re-grader rebuilds the env from it; a language
    passed through ``--env_kwargs`` would run one set and record another."""
    if "language" in env_kwargs:
        raise SystemExit(
            "set the language with --language, not --env_kwargs: the trajectory metadata records the flag"
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
    add_endpoint_args(p)
    p.add_argument(
        "--adapter",
        default="codeforces",
        choices=sorted(name for name, a in CODE_DATASET_ADAPTERS.items() if a.scores_raw_rows),
        help="Adapter that composes raw contest rows into eval examples. This script always reads a raw "
        "dataset; one already prepared by scripts/environments/preparation/prepare_code_dataset.py is not its input.",
    )
    p.add_argument("--num_examples", type=int, default=50, help="Cap on problems (0 = all).")
    p.add_argument("--num_samples", type=int, default=1, help="Samples per problem (success@k).")
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
    p.add_argument("--success_threshold", type=float, default=1.0, help="Reward at/above which a sample is solved.")
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
        choices=sorted(REASONING_EFFORT_PROFILES),
        help=f"Solver reasoning effort, passed to the model's chat template (low/medium/high). Default: the "
        f"training config's under --training_config, else {DEFAULT_REASONING_EFFORT}. Sets the default "
        f"--max_tokens unless --max_tokens or --training_config is given.",
    )
    p.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Max tokens per generation. Default: the training config's rollout_max_tokens under "
        "--training_config, else the effort profile's thinking budget "
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

    Loading goes through the adapter's own ``load`` when it has one (LiveCodeBench and ICPC-Eval
    cannot be read with a plain ``load_dataset``), otherwise the standard HF split loader.
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
    if not examples:
        raise SystemExit(
            f"{args.dataset} ({args.adapter}) yielded no gradable problem; check the adapter, config and split"
        )
    logger.info("Loaded %d problems from %s (%s)", len(examples), args.dataset, args.adapter)
    return examples


def main() -> None:
    args = parse_args()
    contract = load_training_contract(args.training_config)
    trained_env = contract.env_config_dict() if contract is not None else {}
    adapter = CODE_DATASET_ADAPTERS[args.adapter]
    env_kwargs = json.loads(args.env_kwargs)
    refuse_env_kwargs_language(env_kwargs)
    env_type = resolve_setting(
        args.env_type, contract.env_config.environment_type if contract else None, DEFAULT_ENV_TYPE
    )
    if env_type not in CODING_ENV_TYPES:
        raise SystemExit(
            f"{args.training_config} trains environment_type={env_type!r}, not a coding env {CODING_ENV_TYPES}"
        )
    reasoning_effort = resolve_setting(
        args.reasoning_effort, trained_env.get("reasoning_effort"), DEFAULT_REASONING_EFFORT
    )
    max_turns = resolve_setting(args.max_turns, trained_env.get("max_turns"), None)
    # The training run's env config first, the resolved settings and flags over it: an eval under a
    # contract grades as the run did. An unset language or turn budget is left out entirely, so the
    # env class's own default applies.
    env = resolve_environment(
        env_type,
        {
            **trained_env,
            "max_turns": max_turns,
            **({"language": parse_language_flag(args.language)} if args.language else {}),
            "reasoning_effort": reasoning_effort,
            **env_kwargs,
        },
    )
    # Read the effective language set back off the env, which resolved the default.
    language = list(env.languages) if env.chooses_language else env.language
    language_label = ",".join(env.languages)
    # A judge or reward-model term is probed before any episode runs, as the trainer does at launch.
    env.verify_backend()
    examples = build_examples(args, adapter)
    client = create_openai_client(base_url=args.base_url, api_key_override=args.api_key)

    # Without a training config the flag's effort level sets the generation budget unless --max_tokens
    # overrides it: too small a budget truncates the chain of thought before any solution and scores
    # the problem 0. Under one the YAML's rollout_max_tokens is the default.
    flag_effort = args.reasoning_effort or DEFAULT_REASONING_EFFORT
    rollout = rollout_config_from_args(
        args,
        contract,
        default_temperature=DEFAULT_TEMPERATURE,
        default_max_tokens=REASONING_EFFORT_PROFILES[flag_effort]["thinking_tokens"] + SOLUTION_HEADROOM_TOKENS,
    )
    logger.info("reasoning_effort=%s, max_tokens=%d", reasoning_effort, rollout.max_tokens)

    traj_path = resolve_trajectory_path(args, args.adapter, args.split, language_label)

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
    report(
        results,
        num_samples=args.num_samples,
        title=f"{env_type} on {args.dataset} ({args.adapter})",
        group_label=adapter.group_label,
    )
    write_eval_outputs(
        args,
        results,
        env=env,
        traj_path=traj_path,
        env_type=env_type,
        max_turns=max_turns,
        rollout=rollout,
        num_samples=args.num_samples,
        meta_extra={
            "adapter": args.adapter,
            "language": language,
            "reasoning_effort": reasoning_effort,
            "env_kwargs": {**trained_env, **env_kwargs},
            # The run's whole grading contract, so an offline re-grade reproduces the same verdicts
            # (per-problem checker/time_limit come from the dataset payload). Derived from the
            # dataclass, so a knob added to GradingSpec cannot be defaulted offline.
            "env_grading": env.grading_spec.to_meta(),
        },
    )


if __name__ == "__main__":
    main()
