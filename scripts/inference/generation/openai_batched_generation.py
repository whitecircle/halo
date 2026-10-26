"""Batched generation for a dataset of prompts against an OpenAI-compatible API.

Parallel requests with checkpointing (a resumed run reloads the partial dataset) and S3 dataset I/O.
Supports follow-up prompts, per-row tool definitions, and system prompts.

Paths are S3 keys under ``HALO_S3_DEFAULT_BUCKET`` rather than full URIs; they are resolved through
``build_s3_uri``, which prepends the bucket.

Usage:
    HALO_S3_DEFAULT_BUCKET=my-bucket \\
    python scripts/inference/generation/openai_batched_generation.py \\
        --model my-model \\
        --dataset prompts \\
        --output_path responses

Input dataset format (HuggingFace Dataset in S3):
    Required: id_field, prompt_field (list of message dicts)
    Optional: local_system_prompt_field, follow_up_prompt_field, tools_field

Output dataset format:
    Original columns (minus prompt/system/follow-up/tools fields) plus:
    - conversation: full dialogue with system prompt, user messages, model responses
    - generated_message_indices: indices of generated responses in conversation
    - finish_reason: list of finish reasons (stop, tool_calls, length, etc.)
    - prompt_tokens / completion_tokens: token usage
"""

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

    # Data configuration
    add_s3_dataset_args(parser)
    parser.add_argument(
        "--follow_up_prompt_field",
        type=str,
        default="follow_up_prompt",
        help="Row field with follow-up turns sent after the first answer (default: %(default)s).",
    )
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
