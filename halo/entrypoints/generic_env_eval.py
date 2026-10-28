"""The generic environment eval's run: the examples read from a dataset's prompt, answer and context columns, every
named column checked first, and the scored run of any registered environment over them. Behind
``scripts/environments/inference/run_env.py``."""

import argparse
import asyncio
import json
import logging
from typing import Any

from halo.configs.script_arguments import DEFAULT_ANSWER_FIELD
from halo.entrypoints.environment_eval import (
    load_training_contract,
    resolve_setting,
    resolve_trajectory_path,
    rollout_config_from_args,
    write_eval_outputs,
)
from halo.environments.episode_keys import ANSWER_KEY
from halo.environments.eval_runner import collect_results, load_hf_split, reject_answerless_examples, report
from halo.environments.registry import build_environment
from halo.environments.rollout_config import RolloutConfig
from halo.inference.openai_client import build_openai_client

logger = logging.getLogger(__name__)

# The column naming an example when --id_field is not given; a dataset may carry none.
DEFAULT_ID_FIELD = "id"


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


def build_column_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
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
            context[ANSWER_KEY] = row[args.answer_field]
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


def run_generic_env_eval(args: argparse.Namespace) -> None:
    """Evaluate the environment ``args`` names, or its training contract does, over the dataset's columns, report
    mean reward and success@k and write the results and trajectories."""
    contract = load_training_contract(args.training_config)
    trained_env = contract.env_config.to_env_config() if contract is not None else {}
    env_type = resolve_setting(args.env_type, contract.env_config.environment_type if contract else None, None)
    if env_type is None:
        raise SystemExit("--env_type is required unless --training_config names an environment_type")
    env_kwargs = json.loads(args.env_kwargs)
    turns_override = {"max_turns": args.max_turns} if args.max_turns is not None else {}
    # The training run's env config first, the flags over it: an eval under a contract grades as the run did.
    env = build_environment(env_type, {**trained_env, **turns_override, **env_kwargs})
    try:
        rollout = rollout_config_from_args(
            args,
            contract,
            env,
            default_temperature=RolloutConfig.temperature,
            default_max_tokens=RolloutConfig.max_tokens,
        )
        # A judge or reward-model term is probed before any episode runs, as the trainer does at launch.
        env.verify_backend()
        examples = build_column_examples(args)
        reject_answerless_examples(
            env, examples, f"the {args.answer_field!r} field of {args.dataset} (--answer_field)"
        )
        client = build_openai_client(base_url=args.base_url, api_key_override=args.api_key)

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
    finally:
        env.close()
    report(results, num_samples=args.num_samples, title=f"{env_type} on {args.dataset}", group_label=args.group_by)
    write_eval_outputs(
        args,
        results,
        env=env,
        traj_path=traj_path,
        env_type=env_type,
        split=args.split,
        rollout=rollout,
        num_samples=args.num_samples,
        meta_extra={"env_kwargs": {**trained_env, **env_kwargs}},
    )
