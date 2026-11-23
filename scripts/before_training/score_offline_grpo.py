#!/usr/bin/env python
"""Precompute an offline-GRPO dataset's ``rewards`` with the reward terms of a training YAML."""

import argparse
import asyncio

from halo.entrypoints.offline_reward_scoring import run_offline_reward_scoring
from halo.log import configure_cli_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", help="YAML whose 'rewards' list prices each completion (an online-GRPO recipe works)."
    )
    parser.add_argument("--input", required=True, help="JSONL with 'prompt' and 'completions' per row.")
    parser.add_argument("--output", required=True, help="JSONL written with each row's 'rewards'.")
    parser.add_argument(
        "--answer_field", default=None, help="Row field with the ground truth an accuracy term or a judge reads."
    )
    return parser.parse_args()


def main() -> None:
    configure_cli_logging()
    args = parse_args()
    asyncio.run(run_offline_reward_scoring(args.config, args.input, args.output, answer_field=args.answer_field))


if __name__ == "__main__":
    main()
