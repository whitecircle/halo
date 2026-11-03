"""The code-contest eval behind ``scripts/environments/inference/eval_code_contests.py``: selection, protocol, effort
and environment config resolved from the flags over a training contract, and the scored run."""

import argparse
import asyncio
import json
import logging
from typing import Any

from halo.data.sources.loading import load_dataset_split
from halo.entrypoints.eval_contract import (
    DEFAULT_SPLIT,
    load_training_contract,
    resolve_setting,
    resolve_trajectory_path,
    rollout_config_from_args,
    write_eval_outputs,
)
from halo.environments.episode_keys import ANSWER_KEY
from halo.environments.eval_runner import collect_results, report
from halo.environments.registry import build_environment
from halo.environments.rollout_config import RolloutConfig
from halo.environments.tasks.coding.code_contests import (
    DEFAULT_EVAL_PROTOCOL,
    DEFAULT_REASONING_EFFORT,
    CodeContestsEnvironment,
    without_eval_protocol_pins,
)
from halo.environments.tasks.coding.contest_datasets import CODE_DATASET_ADAPTERS, CodeDatasetAdapter, ContestSelection
from halo.inference.openai_client import build_openai_client

logger = logging.getLogger(__name__)

# thinking_tokens is the CoT budget, so this keeps the solution and its tool call from competing with the CoT.
SOLUTION_HEADROOM_TOKENS = 4096
# The coding envs this eval's adapters, language prompt and report buckets are written for.
CODING_ENV_TYPES = ("codeforces", "code_contests")
# Defaults where neither a flag nor ``--training_config`` sets the knob. No language or turn budget: those are the env
# class's own, and a copy here would silently grade under a different contract once the class moves.
DEFAULT_ENV_TYPE = "codeforces"
DEFAULT_TEMPERATURE = 0.2
FLAG_OWNED_ENV_KWARGS = ("max_turns", "language", "eval_protocol", "reasoning_effort")
# ``--reasoning_effort``'s spelling of no level: no episode sends a level or its thinking budget (non-thinking models).
NO_REASONING_EFFORT = "none"


def parse_list_flag(flag: str, value: str) -> list[str]:
    """A comma-separated flag's entries, stripped; a value naming none exits."""
    entries = [entry.strip() for entry in value.split(",") if entry.strip()]
    if not entries:
        raise SystemExit(f"--{flag} names no {flag}: {value!r}")
    return entries


def parse_language_flag(value: str) -> str | list[str]:
    """One name, or the list a comma-separated value names."""
    languages = parse_list_flag("language", value)
    return languages if len(languages) > 1 else languages[0]


def reject_flag_owned_env_kwargs(env_kwargs: dict) -> None:
    """Such an option in ``--env_kwargs`` would silently override its flag, the JSON being laid over the flags."""
    for key in FLAG_OWNED_ENV_KWARGS:
        if key in env_kwargs:
            raise SystemExit(f"set the {key} with --{key}, not --env_kwargs, which would override the flag")


def resolve_selection(args: argparse.Namespace, adapter: CodeDatasetAdapter) -> ContestSelection:
    """Validated against the adapter before any row is read: an unparsable day, an empty window, an unknown platform, a
    bound the dataset cannot apply, or ``--include_examples_only`` on an adapter with no examples-only marker exits."""
    if args.include_examples_only and adapter.examples_only is None:
        raise SystemExit(f"--include_examples_only on --adapter {args.adapter}: it marks no problem examples-only")
    platforms = parse_list_flag("platform", args.platform) if args.platform is not None else ()
    try:
        selection = ContestSelection.parse(args.start_date, args.end_date, platforms, args.include_examples_only)
        adapter.reject_unselectable(selection)
    except ValueError as exc:
        raise SystemExit(f"--start_date/--end_date/--platform on --adapter {args.adapter}: {exc}") from exc
    return selection


def resolve_eval_protocol(flag: str | None, trained_env: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The run's protocol (flag, training config, default) and the training config's env options under it. A config
    written under another protocol gives way to this run's pins; one naming this protocol keeps its values, so a
    contradiction raises in the env."""
    eval_protocol = resolve_setting(flag, trained_env.get("eval_protocol"), DEFAULT_EVAL_PROTOCOL)
    if trained_env.get("eval_protocol") == eval_protocol:
        return eval_protocol, trained_env
    return eval_protocol, without_eval_protocol_pins(trained_env, eval_protocol, "the training config")


def resolve_effort_setting(flag: str | None, trained_env: dict[str, Any]) -> str | None:
    """The flag (:data:`NO_REASONING_EFFORT` = no level), else the training config's, else the default, which fills
    only a key the config omits: ``reasoning_effort: null`` trained at no level and is evaluated at none."""
    if flag is not None:
        return None if flag == NO_REASONING_EFFORT else flag
    return trained_env.get("reasoning_effort", DEFAULT_REASONING_EFFORT)


def default_max_tokens(env: CodeContestsEnvironment, flag: str | None) -> int:
    """``--max_tokens``' default without ``--training_config``: the flag's level's thinking budget as ``env`` binds it
    plus :data:`SOLUTION_HEADROOM_TOKENS`. Without a level or budget it takes the training rollout's default; a soft
    budget, which a turn may reason past, takes the larger of the two. The level is read off the flag alone, since a
    training config's may be ``random``, which has no budget."""
    level = resolve_effort_setting(flag, {})
    budget = env.thinking_budget_for_effort(level) if level is not None else None
    if budget is None:
        return RolloutConfig.max_tokens
    if not env.thinking_budget_enforced(level):
        return max(RolloutConfig.max_tokens, budget + SOLUTION_HEADROOM_TOKENS)
    return budget + SOLUTION_HEADROOM_TOKENS


def resolve_env_config(args: argparse.Namespace, trained_env: dict[str, Any], env_kwargs: dict) -> dict[str, Any]:
    """Built once, so the environment and the meta line cannot disagree. Training options come first, resolved settings
    and flags over them, so an eval under a contract grades as the run did. An unset language is left out, so the env
    class's default applies."""
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
    """``<model>__<adapter>__<split>__<language>``, plus the protocol and selection only where they depart from the
    defaults, so a default run keeps its name."""
    protocol = env.eval_protocol if env.eval_protocol != DEFAULT_EVAL_PROTOCOL else ""
    return resolve_trajectory_path(args, args.adapter, args.split, ",".join(env.languages), protocol, selection.label)


def contest_meta(
    adapter_name: str,
    selection: ContestSelection,
    env: CodeContestsEnvironment,
    reasoning_effort: str | None,
    env_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """What the offline re-grader rebuilds the run's examples (by index, same selection) and environment from."""
    return {
        "adapter": adapter_name,
        "selection": selection.to_meta(),
        # The effective values, read back off the env that resolved the defaults.
        "language": list(env.languages) if env.chooses_language else env.language,
        "eval_protocol": env.eval_protocol,
        "reasoning_effort": reasoning_effort,
        "env_kwargs": env_kwargs,
        # The run's whole grading contract, so an offline re-grade reproduces the same verdicts; derived from the
        # dataclass, so a knob added to GradingSpec cannot be defaulted offline.
        "env_grading": env.grading_spec.to_meta(),
    }


def build_examples(
    args: argparse.Namespace, adapter: CodeDatasetAdapter, selection: ContestSelection
) -> list[dict[str, Any]]:
    """Compose the rows ``selection`` admits into eval examples via the adapter, logging examples-only problems left
    out. The adapter's own ``load`` applies when it has one (LiveCodeBench and ICPC-Eval need it)."""
    rows = (
        adapter.load(args.dataset, args.config, args.split)
        if adapter.load
        else load_dataset_split(args.dataset, split=args.split, config=args.config)
    )
    examples = []
    left_out: list[Any] = []
    for row in adapter.scored_rows(rows, selection, left_out):
        examples.append(
            {
                "prompt": adapter.format_prompt(row),
                "context": {ANSWER_KEY: json.dumps(adapter.pack_verification(row))},
                "group": row.get(adapter.group_field),
                "id": row.get(adapter.id_field),
            }
        )
        if args.num_examples and len(examples) >= args.num_examples:
            break
    if left_out:
        logger.info(
            "Left out %d problems graded only on their statement's examples (--include_examples_only scores them)",
            len(left_out),
        )
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


def run_code_contests_eval(args: argparse.Namespace) -> None:
    """Score the selected problems, report success@k per the adapter's buckets and write results and trajectories."""
    adapter = CODE_DATASET_ADAPTERS[args.adapter]
    try:
        args.split = adapter.resolve_split(args.split, DEFAULT_SPLIT)
    except ValueError as exc:
        raise SystemExit(f"--split on --adapter {args.adapter}: {exc}") from exc
    selection = resolve_selection(args, adapter)
    env_kwargs = json.loads(args.env_kwargs)
    reject_flag_owned_env_kwargs(env_kwargs)
    contract = load_training_contract(args.training_config)
    trained_env = contract.env_config.to_env_config() if contract is not None else {}
    env_type = resolve_setting(
        args.env_type, contract.env_config.environment_type if contract else None, DEFAULT_ENV_TYPE
    )
    if env_type not in CODING_ENV_TYPES:
        raise SystemExit(
            f"{args.training_config} trains environment_type={env_type!r}, not a coding env {CODING_ENV_TYPES}"
        )
    env_config = resolve_env_config(args, trained_env, env_kwargs)
    reasoning_effort = env_config["reasoning_effort"]
    env = build_environment(env_type, env_config)
    try:
        # Without a training config the flag's effort level sets the budget: too small a budget truncates the chain of
        # thought before any solution and scores the problem 0.
        rollout = rollout_config_from_args(
            args,
            contract,
            env,
            default_temperature=DEFAULT_TEMPERATURE,
            default_max_tokens=default_max_tokens(env, args.reasoning_effort),
        )
        logger.info("reasoning_effort=%s, max_tokens=%d", reasoning_effort, rollout.max_tokens)
        # A judge or reward-model term is probed before any episode runs, as the trainer does at launch.
        env.verify_backend()
        examples = build_examples(args, adapter, selection)
        client = build_openai_client(base_url=args.base_url, api_key_override=args.api_key)

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
    finally:
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
        split=args.split,
        rollout=rollout,
        num_samples=args.num_samples,
        meta_extra=contest_meta(args.adapter, selection, env, reasoning_effort, env_config),
    )
