"""Batched generation for a dataset of prompts against an OpenAI-compatible API.

Parallel requests with checkpointing (a resumed run reloads the partial dataset) and S3 dataset I/O.
Supports follow-up prompts, per-row tool definitions, and system prompts.

Paths are S3 **keys** under ``HALO_S3_DEFAULT_BUCKET``, not full URIs — they are resolved through
``build_s3_uri``, which prepends the bucket itself.

Usage:
    HALO_S3_DEFAULT_BUCKET=my-bucket \
    python scripts/inference/generation/openai_batched_generation.py \
        --model_name my-model \
        --input_path prompts \
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

from loguru import logger

from scripts.inference._common import (
    add_checkpoint_interval_arg,
    add_generation_args,
    add_openai_endpoint_args,
    add_s3_dataset_args,
    assistant_message_from_response,
    load_prompts_with_resume,
    parse_dataset_args,
    reject_empty_results,
    run_async_cli,
    save_results_to_s3,
)
from src.data.pipeline.conversation import build_base_prompt, resolve_system_prompt
from src.inference.openai_client import create_openai_client, parallel_openai_requests


def parse_args():
    parser = argparse.ArgumentParser(description="Async batched generation via OpenAI-compatible API")

    add_openai_endpoint_args(parser, model_help="Model name for API requests")

    # Data configuration
    add_s3_dataset_args(parser)
    parser.add_argument("--follow_up_prompt_field", type=str, default="follow_up_prompt")
    parser.add_argument(
        "--tools_field", type=str, default="tools", help="Field with tool definitions in OpenAI format"
    )

    add_generation_args(parser, temperature_default=0.0)
    add_checkpoint_interval_arg(parser)

    return parse_dataset_args(parser)


def process_response(
    row: dict,
    response,
    args,
    follow_up_response=None,
    initial_messages=None,
) -> dict:
    """Process API response into output record.

    Args:
        row: Original dataset row
        response: First OpenAI response
        args: CLI arguments (for field names and model_name)
        follow_up_response: Optional follow-up response
        initial_messages: Messages sent to API (including system prompt)
    """
    conversation = list(initial_messages) if initial_messages else []

    conversation.append(assistant_message_from_response(response))

    generated_message_indices = [len(conversation) - 1]
    finish_reasons = [response.finish_reason]
    prompt_tokens = response.prompt_tokens
    completion_tokens = response.completion_tokens

    if follow_up_response:
        conversation.extend(row[args.follow_up_prompt_field])
        conversation.append(assistant_message_from_response(follow_up_response))
        generated_message_indices.append(len(conversation) - 1)
        finish_reasons.append(follow_up_response.finish_reason)
        prompt_tokens += follow_up_response.prompt_tokens
        completion_tokens += follow_up_response.completion_tokens

    # Build output record (drop prompt-related source fields)
    record = dict(row)
    for field in [
        args.prompt_field,
        args.local_system_prompt_field,
        args.follow_up_prompt_field,
        args.tools_field,
        "response_format",
    ]:
        record.pop(field, None)

    return {
        **record,
        "model": args.model_name,
        "conversation": conversation,
        "generated_message_indices": generated_message_indices,
        "finish_reason": finish_reasons,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


async def main() -> None:
    args = parse_args()

    client = create_openai_client(
        base_url=args.openai_base_url,
        api_key_override=args.openai_api_key,
    )

    rows, existing_results = load_prompts_with_resume(args)
    if not rows:
        logger.info("All prompts already processed.")
        return

    logger.info(f"Generating responses for {len(rows)} prompts...")

    # Build messages with system prompts and collect per-row tools
    messages = []
    tools_list = []
    for row in rows:
        sys_prompt = resolve_system_prompt(row, args.local_system_prompt_field, args.global_system_prompt)
        messages.append(build_base_prompt(row, args.prompt_field, sys_prompt))
        tools = row.get(args.tools_field)
        tools_list.append(tools)

    logger.info("Generating first responses...")
    first_responses = await parallel_openai_requests(
        model=args.model_name,
        user_messages=messages,
        response_format=None,
        use_native_json_schema=False,
        system_prompt=None,  # Already in messages
        temperature=args.temperature,
        max_tokens=args.max_gen_tokens,
        max_workers=args.n_parallel,
        checkpoint_interval=args.checkpoint_interval,
        disable_checkpoints=False,
        custom_client=client,
        tools=tools_list,
    )

    # Split: rows with follow-up vs. complete
    results = []
    rows_with_followup = []
    dropped_first_response = 0

    for idx, (row, msg, response) in enumerate(zip(rows, messages, first_responses, strict=True)):
        if response is None:
            dropped_first_response += 1
            logger.warning(f"Skipping row {idx}: error in first response")
            continue

        if args.follow_up_prompt_field in row and row[args.follow_up_prompt_field] is not None:
            rows_with_followup.append((idx, row, msg, response))
        else:
            results.append(process_response(row, response, args, initial_messages=msg))

    if rows_with_followup:
        logger.info(f"Generating {len(rows_with_followup)} follow-up responses...")

        followup_messages = []
        followup_tools = []
        for _idx, row, initial_msg, first_response in rows_with_followup:
            conversation = list(initial_msg)
            conversation.append(assistant_message_from_response(first_response))
            conversation.extend(row[args.follow_up_prompt_field])
            followup_messages.append(conversation)

            tools = row.get(args.tools_field)
            followup_tools.append(tools)

        followup_responses = await parallel_openai_requests(
            model=args.model_name,
            user_messages=followup_messages,
            response_format=None,
            use_native_json_schema=False,
            system_prompt=None,
            temperature=args.temperature,
            max_tokens=args.max_gen_tokens,
            max_workers=args.n_parallel,
            checkpoint_interval=args.checkpoint_interval,
            disable_checkpoints=False,
            custom_client=client,
            tools=followup_tools,
        )

        for (idx, row, initial_msg, first_response), followup_response in zip(
            rows_with_followup, followup_responses, strict=True
        ):
            if followup_response is None:
                logger.warning(f"Row {idx}: follow-up failed, saving first response only")
                results.append(process_response(row, first_response, args, initial_messages=initial_msg))
            else:
                results.append(
                    process_response(row, first_response, args, followup_response, initial_messages=initial_msg)
                )

    # A follow-up failure still records the first response, so an empty result set means every
    # pending row failed its FIRST request.
    if dropped_first_response:
        logger.warning(
            f"Dropped {dropped_first_response} of {len(rows)} pending row(s) whose first request "
            f"failed — they are NOT in the dataset about to be written, and a resumed run will "
            f"retry them. Check --openai_base_url / --model_name and the endpoint's logs."
        )
    reject_empty_results(
        len(results),
        len(rows),
        args.output_path,
        drops=f" (first-response failures={dropped_first_response})",
        check="--openai_base_url / --model_name and the endpoint's logs",
    )
    save_results_to_s3(existing_results, results, output_path=args.output_path, subfolder=args.subfolder)


if __name__ == "__main__":
    run_async_cli(main)
