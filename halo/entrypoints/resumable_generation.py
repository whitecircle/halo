"""The generation CLIs' run plumbing: the S3 prompt dataset with its resume, the merged push, the empty-run
refusal, the assistant-turn and follow-up spellings, the batched generation run and the async CLI runner."""

import asyncio
import logging
import signal
import sys
from collections.abc import Callable, Coroutine

from datasets import Dataset

from halo.data.pipeline.conversation import build_base_prompt, resolve_system_prompt
from halo.data.sources.s3_client import S3Client, build_s3_uri
from halo.data.sources.s3_dataset_store import load_dataset_from_s3_uri, push_dataset_to_s3_uri
from halo.inference.batch_requests import parallel_openai_requests
from halo.inference.openai_client import build_openai_client
from halo.log import configure_cli_logging

logger = logging.getLogger(__name__)


def load_prompts_with_resume(args) -> tuple[list[dict], list[dict]]:
    """``(pending rows, rows already written to --output_path)`` for the S3 prompt dataset.

    Ids present in the output dataset are dropped from the input, and the existing rows come back so
    the caller can write them out again alongside the new ones. An empty first element means every
    prompt is already done.
    """
    logger.info(f"Loading dataset from S3: {args.dataset} (subfolder: {args.subfolder})")
    dataset = load_dataset_from_s3_uri(build_s3_uri(args.dataset, args.subfolder))

    if args.prompt_field not in dataset.column_names or args.id_field not in dataset.column_names:
        raise ValueError(f"Dataset must contain '{args.prompt_field}' and '{args.id_field}' columns")

    processed_ids: set = set()
    existing_results: list[dict] = []
    if S3Client().exists(args.output_path, subfolder=args.subfolder):
        logger.info(f"Loading existing results from S3: {args.output_path}")
        existing_dataset = load_dataset_from_s3_uri(build_s3_uri(args.output_path, args.subfolder))
        processed_ids = set(existing_dataset[args.id_field])
        existing_results = list(existing_dataset)
        logger.info(f"Skipping {len(processed_ids)} already completed prompts")

    return list(dataset.filter(lambda x: x[args.id_field] not in processed_ids)), existing_results


def reject_empty_results(produced: int, pending: int, destination, *, drops: str = "", check: str) -> None:
    """Abort a run that produced no usable row at all, instead of writing an empty result.

    Every pending row failed, which means a dead endpoint or a misconfigured model. Exiting 0 there
    would report a total failure as a completed run and, on the S3 paths, republish the resumed rows
    as the whole job. Keyed on "nothing produced" rather than a single failure counter, since rows
    also drop for reasons no counter tracks.

    ``drops`` is the caller's own per-reason tally, ``check`` the knobs to look at first.
    """
    if produced:
        return
    raise RuntimeError(
        f"No usable result for any of the {pending} pending prompt(s){drops} — nothing written to "
        f"{destination}. Check {check}."
    )


def save_results_to_s3(existing_results: list[dict], results: list[dict], *, output_path: str, subfolder) -> None:
    """Push ``existing_results + results`` to the S3 output dataset, with the record keys unioned.

    A resume mixes records reloaded from a previously-saved dataset (carrying every column that run
    wrote) with freshly-built ones, whose optional source columns, or whole output format, may differ.
    ``Dataset.from_list`` infers its schema from the leading records and drops keys absent there, so
    the union (missing filled with ``None``) preserves the just-generated batch's own columns.
    """
    all_results = existing_results + results
    all_keys = {key for record in all_results for key in record}
    all_results = [{key: record.get(key) for key in all_keys} for record in all_results]
    logger.info(f"Saving {len(all_results)} results to S3: {output_path}")
    push_dataset_to_s3_uri(Dataset.from_list(all_results), build_s3_uri(output_path, subfolder))
    logger.info("Done!")


def follow_up_messages(row, field: str) -> list[dict] | None:
    """The row's follow-up turns (``--follow_up_prompt_field``), sent after its first answer: a non-empty
    message list, else ``None``. An empty list is no follow-up: a second request on a conversation that
    already ends on the assistant's answer would record two assistant turns in a row."""
    follow_up = row.get(field)
    return follow_up if isinstance(follow_up, list) and follow_up else None


def assistant_turn(content: str | None, tool_calls: list | None = None) -> dict:
    """A generated assistant turn as the wire and every saved record spell it: ``role`` and
    ``content``, plus ``tool_calls`` where the reply made any — never the reply object's other,
    mostly null, SDK and engine fields.

    ``content`` stays ``None`` on a tool-call-only or empty reply: ``str(None)`` sends the literal
    text "None" to the next turn, which the model reads as the assistant's answer.
    """
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [tool_call.model_dump() for tool_call in tool_calls]
    return message


def _signal_handler(signum, frame):
    """Exit on an interrupt with the shell's own convention for a signalled process: 128 + signo.

    Exiting 0 here would report an interrupted, incomplete job as a successful one, and a wrapper
    script, CI step or `&&` chain would then consume a partial output dataset as a finished run. The
    resume hint makes the non-zero exit actionable.
    """
    logger.info(f"\nReceived signal {signum}. Progress has been saved — resume by re-running.")
    sys.exit(128 + signum)


def run_async_cli(main: Callable[[], Coroutine]) -> None:
    """Run an async CLI ``main`` under SIGINT/SIGTERM handling, exiting non-zero on a fatal error."""
    configure_cli_logging()
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Same event as the handler above, which a library restoring the default handler can bypass,
        # so it reports the same way.
        logger.info("\nInterrupted. Progress has been saved — resume by re-running.")
        sys.exit(128 + signal.SIGINT)
    except Exception:
        # With the traceback: the failures that land here (a raised result guard, an S3 or schema
        # error inside the merge) are diagnosed from where they were raised, not from the message.
        logger.exception("Fatal error")
        sys.exit(1)


def reject_per_row_response_format(rows: list[dict]) -> None:
    """Refuse rows carrying a structured ``response_format``: every request here goes out as plain
    text, so a per-row schema (the reward-model scripts honour one) would be dropped from the output
    unapplied. A ``{"type": "text"}`` format asks for exactly that and passes."""
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
    """Process API response into output record.

    Args:
        row: Original dataset row
        response: First OpenAI response
        args: CLI arguments (for field names and the model)
        initial_messages: Messages sent to API (including system prompt)
        follow_up_response: Optional follow-up response
    """
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
        "model": args.model,
        "conversation": conversation,
        "generated_message_indices": generated_message_indices,
        "finish_reason": finish_reasons,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


async def run_batched_generation(args) -> None:
    """Generate every pending prompt of the S3 dataset (first turn, then any follow-up turn) and push the
    merged result; a row whose first request failed is dropped and retried by the next run."""

    client = build_openai_client(
        base_url=args.base_url,
        api_key_override=args.api_key,
    )

    rows, existing_results = load_prompts_with_resume(args)
    if not rows:
        logger.info("All prompts already processed.")
        return
    reject_per_row_response_format(rows)

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
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
        flush_interval=args.checkpoint_interval,
        client=client,
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

    # A follow-up failure still records the first response, so an empty result set means every
    # pending row failed its first request.
    if dropped_first_response:
        logger.warning(
            f"Dropped {dropped_first_response} of {len(rows)} pending row(s) whose first request "
            f"failed — they are NOT in the dataset about to be written, and a resumed run will "
            f"retry them. Check --base_url / --model and the endpoint's logs."
        )
    reject_empty_results(
        len(results),
        len(rows),
        args.output_path,
        drops=f" (first-response failures={dropped_first_response})",
        check="--base_url / --model and the endpoint's logs",
    )
    save_results_to_s3(existing_results, results, output_path=args.output_path, subfolder=args.subfolder)
