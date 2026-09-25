#!/usr/bin/env python
"""
Evaluate a model on competitive-programming problems (`code_contests` / `codeforces` env) against an
OpenAI-compatible endpoint.

Specific to the coding-contest task: it applies a dataset adapter that scores raw rows (every
``CODE_DATASET_ADAPTERS`` entry without a ``normalize`` step) to a contest dataset, prompts in a chosen
solution `language`, and reports `success@1` / `success@k` bucketed by the adapter's report field
(rating on Codeforces, difficulty on LiveCodeBench, contest on ICPC-Eval and HLCE). The rollout loop
and reward aggregation are shared with the other eval scripts via :mod:`src.environments.eval_runner`.
`--eval_protocol` names the evaluation contract (`harness`: the configured budgets; `leaderboard`: one
graded program, no scratchpad), and on a benchmark that stamps contest dates
`--start_date` / `--end_date` / `--platform` select the problems scored.

The server must serve the model with tool calling enabled (e.g. vLLM
`--tool-call-parser qwen3_xml --enable-auto-tool-choice`). A solution counts as solved when it passes
every test in the pool, the environment's own verdict whatever the shaped reward. Reasoning models
need a large `--max_tokens`: too low truncates the chain of thought before any solution and scores 0.

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

    # LiveCodeBench release_v6, AtCoder contests from 2025-01-04 on, leaderboard protocol, success@4
    python scripts/environments/inference/run_code_contests.py \
        --dataset livecodebench/code_generation_lite --config release_v6 --adapter livecodebench \
        --start_date 2025-01-04 --platform atcoder --eval_protocol leaderboard \
        --base_url http://localhost:8000/v1 --model <served-name> --num_examples 0 --num_samples 4
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
from src.configs.rollout_config import DEFAULT_ROLLOUT_MAX_TOKENS
from src.environments.envs.tasks.coding.code_contests import (
    DEFAULT_EVAL_PROTOCOL,
    DEFAULT_REASONING_EFFORT,
    EVAL_PROTOCOLS,
    REASONING_EFFORT_PROFILES,
    CodeContestsEnvironment,
    without_eval_protocol_pins,
)
from src.environments.envs.tasks.coding.datasets import CODE_DATASET_ADAPTERS, CodeDatasetAdapter, ContestSelection
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

# The coding envs this script's adapters, language prompt and report buckets are written for.
CODING_ENV_TYPES = ("codeforces", "code_contests")
# Script defaults, applied where neither a flag nor ``--training_config`` sets the knob. No language
# or turn budget among them: both are the env class's own, and a script-side copy would silently grade
# under a different contract than the class ships once the class moves.
DEFAULT_ENV_TYPE = "codeforces"
DEFAULT_TEMPERATURE = 0.2
# Env options with a flag of their own, the flag being their one spelling on this command line.
FLAG_OWNED_ENV_KWARGS = ("max_turns", "language", "eval_protocol", "reasoning_effort")
# ``--reasoning_effort``'s spelling of no level (``reasoning_effort=None``): no episode sends a level
# or a level's thinking budget, the setting a non-thinking model is evaluated under.
NO_REASONING_EFFORT = "none"


def parse_list_flag(flag: str, value: str) -> list[str]:
    """A comma-separated flag's entries, stripped; a value naming none exits."""
    entries = [entry.strip() for entry in value.split(",") if entry.strip()]
    if not entries:
        raise SystemExit(f"--{flag} names no {flag}: {value!r}")
    return entries


def parse_language_flag(value: str) -> str | list[str]:
    """``--language`` as the env's ``language``: one name, or the list a comma-separated value names."""
    languages = parse_list_flag("language", value)
    return languages if len(languages) > 1 else languages[0]


def refuse_flag_owned_env_kwargs(env_kwargs: dict) -> None:
    """A :data:`FLAG_OWNED_ENV_KWARGS` option also passed through ``--env_kwargs`` would silently
    override its flag, the JSON being laid over the flags."""
    for key in FLAG_OWNED_ENV_KWARGS:
        if key in env_kwargs:
            raise SystemExit(f"set the {key} with --{key}, not --env_kwargs, which would override the flag")


def resolve_selection(args: argparse.Namespace, adapter: CodeDatasetAdapter) -> ContestSelection:
    """The contest window and platforms the run scores, validated against the adapter before any row
    is read: an unparsable day, an empty window, an unknown platform, or a bound the dataset cannot
    apply exits."""
    platforms = parse_list_flag("platform", args.platform) if args.platform is not None else ()
    try:
        selection = ContestSelection.parse(args.start_date, args.end_date, platforms)
        adapter.require_selectable(selection)
    except ValueError as exc:
        raise SystemExit(f"--start_date/--end_date/--platform on --adapter {args.adapter}: {exc}") from exc
    return selection


def resolve_eval_protocol(flag: str | None, trained_env: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The run's protocol (the flag, else the training config's, else the default) and the training
    config's env options under it. A config written under another protocol gives way to this run's
    pins, as its effort profiles do; one that names this protocol itself keeps its values, so a
    contradiction there raises in the env, as one in ``--env_kwargs`` does."""
    eval_protocol = resolve_setting(flag, trained_env.get("eval_protocol"), DEFAULT_EVAL_PROTOCOL)
    if trained_env.get("eval_protocol") == eval_protocol:
        return eval_protocol, trained_env
    return eval_protocol, without_eval_protocol_pins(trained_env, eval_protocol, "the training config")


def resolve_effort_setting(flag: str | None, trained_env: dict[str, Any]) -> str | None:
    """The run's effort level: the flag (:data:`NO_REASONING_EFFORT` = no level), else the training
    config's, else :data:`DEFAULT_REASONING_EFFORT`. As in training, the default fills only a key the
    config omits: a config setting ``reasoning_effort: null`` trained at no level and is evaluated at none."""
    if flag is not None:
        return None if flag == NO_REASONING_EFFORT else flag
    return trained_env.get("reasoning_effort", DEFAULT_REASONING_EFFORT)


def default_max_tokens(flag: str | None) -> int:
    """``--max_tokens``' default without ``--training_config``: the flag's level's thinking budget plus
    :data:`SOLUTION_HEADROOM_TOKENS`. Without a level there is no thinking budget to size it from, so it
    takes the training rollout's own default. Read off the flag alone: under a training config the YAML's
    ``rollout_max_tokens`` is the default, and its level may be ``random``, which has no budget."""
    level = resolve_effort_setting(flag, {})
    if level is None:
        return DEFAULT_ROLLOUT_MAX_TOKENS
    return REASONING_EFFORT_PROFILES[level]["thinking_tokens"] + SOLUTION_HEADROOM_TOKENS


def resolve_env_config(args: argparse.Namespace, trained_env: dict[str, Any], env_kwargs: dict) -> dict[str, Any]:
    """The run's environment config, built once: the environment is made from it and the meta line
    records it, so the two cannot disagree. The training config's options come first (under the
    resolved protocol), the resolved settings and flags over them: an eval under a contract grades as
    the run did. An unset language is left out entirely, so the env class's own default applies."""
    eval_protocol, trained_env = resolve_eval_protocol(args.eval_protocol, trained_env)
    return {
        **trained_env,
        "max_turns": resolve_setting(args.max_turns, trained_env.get("max_turns"), None),
        **({"language": parse_language_flag(args.language)} if args.language else {}),
        "eval_protocol": eval_protocol,
        "reasoning_effort": resolve_effort_setting(args.reasoning_effort, trained_env),
        **env_kwargs,
    }


def run_trajectory_path(
    args: argparse.Namespace, env: CodeContestsEnvironment, selection: ContestSelection
) -> str | None:
    """The run's trajectory file: ``<model>__<adapter>__<split>__<language>``, plus the protocol and the
    selection only where they depart from the defaults, so a default run keeps its name. The meta line
    records both either way."""
    protocol = env.eval_protocol if env.eval_protocol != DEFAULT_EVAL_PROTOCOL else ""
    return resolve_trajectory_path(args, args.adapter, args.split, ",".join(env.languages), protocol, selection.label)


def contest_meta(
    adapter_name: str,
    selection: ContestSelection,
    env: CodeContestsEnvironment,
    reasoning_effort: str | None,
    env_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """The code-contest keys of the trajectory meta line: what the offline re-grader rebuilds the run's
    examples (by index, under the same selection) and environment from."""
    return {
        "adapter": adapter_name,
        "selection": selection.to_meta(),
        # The effective values, read back off the env that resolved the defaults.
        "language": list(env.languages) if env.chooses_language else env.language,
        "eval_protocol": env.eval_protocol,
        "reasoning_effort": reasoning_effort,
        "env_kwargs": env_kwargs,
        # The run's whole grading contract, so an offline re-grade reproduces the same verdicts
        # (per-problem checker/time_limit come from the dataset payload). Derived from the
        # dataclass, so a knob added to GradingSpec cannot be defaulted offline.
        "env_grading": env.grading_spec.to_meta(),
    }


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
        choices=[*sorted(REASONING_EFFORT_PROFILES), NO_REASONING_EFFORT],
        help=f"Solver reasoning effort, passed to the model's chat template (low/medium/high), or "
        f"{NO_REASONING_EFFORT}: no level and no thinking budget, for a non-thinking model. Default: the "
        f"training config's under --training_config (a null there is {NO_REASONING_EFFORT}), else "
        f"{DEFAULT_REASONING_EFFORT}. Sets the default --max_tokens unless --max_tokens or "
        f"--training_config is given.",
    )
    p.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Max tokens per generation. Default: the training config's rollout_max_tokens under "
        "--training_config, else the effort profile's thinking budget "
        + ", ".join(f"{level}={p['thinking_tokens']}" for level, p in REASONING_EFFORT_PROFILES.items())
        + f" plus {SOLUTION_HEADROOM_TOKENS} solution headroom; under --reasoning_effort {NO_REASONING_EFFORT}, "
        f"the training rollout's default {DEFAULT_ROLLOUT_MAX_TOKENS}.",
    )
    p.add_argument("--max_workers", type=int, default=16, help="Concurrent episodes.")
    p.add_argument(
        "--env_kwargs",
        default="{}",
        help="JSON dict merged into the env config, for the grading knobs this script does not expose "
        "(e.g. stop_on_first_failure, timeout_per_test, max_submissions, sandbox_backend).",
    )
    return p.parse_args()


def build_examples(
    args: argparse.Namespace, adapter: CodeDatasetAdapter, selection: ContestSelection
) -> list[dict[str, Any]]:
    """Compose the raw contest rows ``selection`` admits into eval examples via the chosen adapter,
    bucketed by its group field and named by its id field.

    Loading goes through the adapter's own ``load`` when it has one (LiveCodeBench and ICPC-Eval
    cannot be read with a plain ``load_dataset``), otherwise the standard HF split loader.
    """
    rows = (
        adapter.load(args.dataset, args.config, args.split)
        if adapter.load
        else load_hf_split(args.dataset, args.config, args.split)
    )
    examples = []
    for row in adapter.scored_rows(rows, selection):
        examples.append(
            {
                "prompt": adapter.format_prompt(row),
                "context": {"answer": json.dumps(adapter.pack_verification(row))},
                "group": row.get(adapter.group_field),
                "id": row.get(adapter.id_field),
            }
        )
        if args.num_examples and len(examples) >= args.num_examples:
            break
    if not examples:
        raise SystemExit(
            f"{args.dataset} ({args.adapter}) yielded no gradable problem; check the adapter, config, split "
            f"and the contest selection"
        )
    logger.info(
        "Loaded %d problems from %s (%s%s)",
        len(examples),
        args.dataset,
        args.adapter,
        f", {selection.label}" if selection.label else "",
    )
    return examples


def main() -> None:
    args = parse_args()
    adapter = CODE_DATASET_ADAPTERS[args.adapter]
    selection = resolve_selection(args, adapter)
    env_kwargs = json.loads(args.env_kwargs)
    refuse_flag_owned_env_kwargs(env_kwargs)
    contract = load_training_contract(args.training_config)
    trained_env = contract.env_config_dict() if contract is not None else {}
    env_type = resolve_setting(
        args.env_type, contract.env_config.environment_type if contract else None, DEFAULT_ENV_TYPE
    )
    if env_type not in CODING_ENV_TYPES:
        raise SystemExit(
            f"{args.training_config} trains environment_type={env_type!r}, not a coding env {CODING_ENV_TYPES}"
        )
    env_config = resolve_env_config(args, trained_env, env_kwargs)
    reasoning_effort, max_turns = env_config["reasoning_effort"], env_config["max_turns"]
    env = resolve_environment(env_type, env_config)
    # A judge or reward-model term is probed before any episode runs, as the trainer does at launch.
    env.verify_backend()
    examples = build_examples(args, adapter, selection)
    client = create_openai_client(base_url=args.base_url, api_key_override=args.api_key)

    # Without a training config the flag's effort level sets the generation budget unless --max_tokens
    # overrides it: too small a budget truncates the chain of thought before any solution and scores
    # the problem 0.
    rollout = rollout_config_from_args(
        args,
        contract,
        default_temperature=DEFAULT_TEMPERATURE,
        default_max_tokens=default_max_tokens(args.reasoning_effort),
    )
    logger.info("reasoning_effort=%s, max_tokens=%d", reasoning_effort, rollout.max_tokens)

    traj_path = run_trajectory_path(args, env, selection)

    results = asyncio.run(
        collect_results(
            env,
            examples,
            client,
            rollout=rollout,
            num_samples=args.num_samples,
            max_workers=args.max_workers,
            collect_trajectories=bool(traj_path),
        )
    )
    env.close()
    scope = ", ".join(part for part in (args.adapter, selection.label, f"{env.eval_protocol} protocol") if part)
    report(
        results,
        num_samples=args.num_samples,
        title=f"{env_type} on {args.dataset} ({scope})",
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
        meta_extra=contest_meta(args.adapter, selection, env, reasoning_effort, env_config),
    )


if __name__ == "__main__":
    main()
