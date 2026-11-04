"""The generation CLIs' run plumbing: the S3 prompt dataset with resume, the merged push, batched run, async runner."""

import asyncio
import logging
import signal
import sys
from collections.abc import Callable, Coroutine

from datasets import Dataset

from halo.data.conversation import build_base_prompt, resolve_system_prompt
from halo.data.sources.paths import parse_s3_uri
from halo.data.sources.s3_client import S3Client, build_s3_uri
from halo.data.sources.s3_dataset_store import load_dataset_from_s3_uri, push_dataset_to_s3_uri
from halo.inference.batch_requests import parallel_openai_requests
from halo.inference.openai_client import build_openai_client
from halo.log import configure_cli_logging

logger = logging.getLogger(__name__)


def load_prompts_with_resume(args, output_uri: str) -> tuple[list[dict], list[dict]]:
    """``(pending rows, rows already written to output_uri)``; the existing rows come back so the caller can rewrite
    them with the new ones. An empty first element means every prompt is done."""
    dataset_uri = build_s3_uri(args.dataset, args.subfolder, given_as="--dataset")
    logger.info(f"Loading dataset from S3: {dataset_uri}")
    dataset = load_dataset_from_s3_uri(dataset_uri)

    if args.prompt_field not in dataset.column_names or args.id_field not in dataset.column_names:
        raise ValueError(f"Dataset must contain '{args.prompt_field}' and '{args.id_field}' columns")

    processed_ids: set = set()
    existing_results: list[dict] = []
    output_bucket, output_key = parse_s3_uri(output_uri)
    if S3Client(bucket=output_bucket).exists(output_key):
        logger.info(f"Loading existing results from S3: {output_uri}")
        existing_dataset = load_dataset_from_s3_uri(output_uri)
        processed_ids = set(existing_dataset[args.id_field])
        existing_results = list(existing_dataset)
        logger.info(f"Skipping {len(processed_ids)} already completed prompts")

    return list(dataset.filter(lambda x: x[args.id_field] not in processed_ids)), existing_results


def reject_empty_results(produced: int, pending: int, destination, *, drops: str = "", check: str) -> None:
    """Abort a run that produced no usable row (a dead endpoint or misconfigured model): exiting 0 would report total
    failure as a completed run and, on S3, republish the resumed rows as the whole job. ``drops`` is the caller's
    per-reason tally, ``check`` the knobs to look at first."""
    if produced:
        return
    raise RuntimeError(
        f"No usable result for any of the {pending} pending prompt(s){drops} — nothing written to "
        f"{destination}. Check {check}."
    )


def save_results_to_s3(existing_results: list[dict], results: list[dict], output_uri: str) -> None:
    """Push with the record keys unioned: a resume mixes reloaded records with fresh ones whose columns may differ, and
    ``Dataset.from_list`` infers its schema from the leading records and drops keys absent there."""
    all_results = existing_results + results
    all_keys = {key for record in all_results for key in record}
    all_results = [{key: record.get(key) for key in all_keys} for record in all_results]
    logger.info(f"Saving {len(all_results)} results to S3: {output_uri}")
    push_dataset_to_s3_uri(Dataset.from_list(all_results), output_uri)
    logger.info("Done!")


def follow_up_messages(row, field: str) -> list[dict] | None:
    """The row's follow-up turns, or ``None``. An empty list is no follow-up: a second request on a conversation
    already ending on the answer would record two assistant turns in a row."""
    follow_up = row.get(field)
    return follow_up if isinstance(follow_up, list) and follow_up else None


def assistant_turn(content: str | None, tool_calls: list | None = None) -> dict:
    """``role`` and ``content``, plus ``tool_calls`` where the reply made any, never the reply's other SDK and engine
    fields. ``content`` stays ``None`` on a tool-call-only or empty reply: ``str(None)`` would send "None" onward."""
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [tool_call.model_dump() for tool_call in tool_calls]
    return message


def _signal_handler(signum, frame):
    """Exit 128 + signo: exiting 0 would report an interrupted job as successful, and a wrapper or `&&` chain would
    consume partial output as finished."""
    logger.info(f"\nReceived signal {signum}. Progress has been saved — resume by re-running.")
    sys.exit(128 + signum)


def run_async_cli(main: Callable[[], Coroutine]) -> None:
    """Run under SIGINT/SIGTERM handling, exiting non-zero on a fatal error."""
    configure_cli_logging()
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # A library restoring the default handler can bypass the handler above, so this reports the same way.
        logger.info("\nInterrupted. Progress has been saved — resume by re-running.")
        sys.exit(128 + signal.SIGINT)
    except Exception:
        # With the traceback: a raised result guard or an S3 or schema error in the merge is diagnosed from its origin.
        logger.exception("Fatal error")
        sys.exit(1)


def reject_per_row_response_format(rows: list[dict]) -> None:
    """Refuse a structured ``response_format``: every request here is plain text, so a per-row schema would be dropped
    unapplied. ``{"type": "text"}`` passes."""
    carrying = [row for row in rows if (row.get("response_format") or {}).get("type", "text") != "text"]
    if carrying:
        raise ValueError(
            f"{len(carrying)} of {len(rows)} rows carry a structured response_format, which this tool does not apply: "
            f"it would generate them as plain text and drop the column. Remove the column, or generate "
            f"structured rows with the reward-model scripts, which honour it."
        )


def _generation_record(
    row: dict,
    response,
    args,
    initial_messages: list[dict],
    follow_up_response=None,
) -> dict:
    conversation = list(initial_messages)
    conversation.append(assistant_turn(response.answer, response.tool_calls))

    generated_message_indices = [len(conversation) - 1]
    finish_reasons = [response.finish_reason]
    prompt_tokens = response.prompt_tokens
    completion_tokens = response.completion_tokens

    if follow_up_response:
        conversation.extend(row[args.follow_up_prompt_field])
        conversation.append(assistant_turn(follow_up_response.answer, follow_up_response.tool_calls))
        generated_message_indices.append(len(conversation) - 1)
        finish_reasons.append(follow_up_response.finish_reason)
        prompt_tokens += follow_up_response.prompt_tokens
        completion_tokens += follow_up_response.completion_tokens

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
        "model": args.model,
        "conversation": conversation,
        "generated_message_indices": generated_message_indices,
        "finish_reason": finish_reasons,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


async def run_batched_generation(args) -> None:
    """A row whose first request failed is dropped and retried by the next run."""
    client = build_openai_client(
        base_url=args.base_url,
        api_key_override=args.api_key,
    )

    # Resolved once, so the resume read, the empty-run refusal and the push name the same dataset.
    output_uri = build_s3_uri(args.output_path, args.subfolder, given_as="--output_path")
    rows, existing_results = load_prompts_with_resume(args, output_uri)
    if not rows:
        logger.info("All prompts already processed.")
        return
    reject_per_row_response_format(rows)

    logger.info(f"Generating responses for {len(rows)} prompts...")

    messages = []
    tools_list = []
    for row in rows:
        sys_prompt = resolve_system_prompt(row, args.local_system_prompt_field, args.global_system_prompt)
        messages.append(build_base_prompt(row, args.prompt_field, sys_prompt))
        tools = row.get(args.tools_field)
        tools_list.append(tools)

    logger.info("Generating first responses...")
    first_responses = await parallel_openai_requests(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
        flush_interval=args.checkpoint_interval,
        client=client,
        tools=tools_list,
    )

    results = []
    rows_with_followup = []
    dropped_first_response = 0

    for idx, (row, msg, response) in enumerate(zip(rows, messages, first_responses, strict=True)):
        if response is None:
            dropped_first_response += 1
            logger.warning(f"Skipping row {idx}: error in first response")
            continue

        if follow_up_messages(row, args.follow_up_prompt_field) is not None:
            rows_with_followup.append((idx, row, msg, response))
        else:
            results.append(_generation_record(row, response, args, msg))

    if rows_with_followup:
        logger.info(f"Generating {len(rows_with_followup)} follow-up responses...")

        followup_messages = []
        followup_tools = []
        for _idx, row, initial_msg, first_response in rows_with_followup:
            conversation = list(initial_msg)
            conversation.append(assistant_turn(first_response.answer, first_response.tool_calls))
            conversation.extend(row[args.follow_up_prompt_field])
            followup_messages.append(conversation)

            tools = row.get(args.tools_field)
            followup_tools.append(tools)

        followup_responses = await parallel_openai_requests(
            model=args.model,
            messages=followup_messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_workers=args.max_workers,
            flush_interval=args.checkpoint_interval,
            client=client,
            tools=followup_tools,
        )

        for (idx, row, initial_msg, first_response), followup_response in zip(
            rows_with_followup, followup_responses, strict=True
        ):
            if followup_response is None:
                logger.warning(f"Row {idx}: follow-up failed, saving first response only")
            results.append(_generation_record(row, first_response, args, initial_msg, followup_response))

    # A follow-up failure still records the first response, so an empty result means every first request failed.
    if dropped_first_response:
        logger.warning(
            f"Dropped {dropped_first_response} of {len(rows)} pending row(s) whose first request "
            f"failed — they are NOT in the dataset about to be written, and a resumed run will "
            f"retry them. Check --base_url / --model and the endpoint's logs."
        )
    reject_empty_results(
        len(results),
        len(rows),
        output_uri,
        drops=f" (first-response failures={dropped_first_response})",
        check="--base_url / --model and the endpoint's logs",
    )
    save_results_to_s3(existing_results, results, output_uri)
