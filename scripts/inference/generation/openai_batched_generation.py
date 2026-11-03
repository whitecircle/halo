"""Batched generation for a dataset of prompts against an OpenAI-compatible API, with checkpointing and S3 dataset I/O.

``--dataset`` and ``--output_path`` are full ``s3://`` URIs or bare keys under ``--subfolder`` in
``HALO_S3_DEFAULT_BUCKET``. Output keeps the original columns minus prompt/system/follow-up/tools, plus
``conversation``, ``generated_message_indices``, ``finish_reason``, ``prompt_tokens`` and ``completion_tokens``."""

import argparse

from halo.entrypoints.resumable_generation import run_async_cli, run_batched_generation
from scripts.cli_arguments import add_openai_endpoint_args
from scripts.inference.inference_arguments import (
    add_checkpoint_interval_arg,
    add_generation_args,
    add_s3_dataset_args,
    parse_dataset_args,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Async batched generation via OpenAI-compatible API")

    add_openai_endpoint_args(parser, model_help="Model name for API requests")

    add_s3_dataset_args(parser)
    parser.add_argument(
        "--tools_field", type=str, default="tools", help="Field with tool definitions in OpenAI format"
    )

    add_generation_args(parser, temperature_default=0.0)
    add_checkpoint_interval_arg(parser)

    return parse_dataset_args(parser)


async def main() -> None:
    await run_batched_generation(parse_args())


if __name__ == "__main__":
    run_async_cli(main)
