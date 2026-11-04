"""Reward-model rejection sampling: N hypotheses per prompt scored by the reward model, kept as a DPO/SMPO
preference pair (best and worst) or an offline-GRPO record (every completion with its reward)."""

import asyncio
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from halo.entrypoints.resumable_generation import reject_empty_results
from halo.entrypoints.reward_scoring import (
    OverlongConversationError,
    TruncatedGenerationError,
    append_jsonl_record,
    boot_scoring_run,
    build_output_path,
    generate_chat_message,
    load_local_jsonl_resume,
    load_prompts_dataframe,
    prepare_generation_prompt,
    resolve_correct_answer,
    score_conversations_offloaded,
)
from halo.inference.response import ENGINE_CUT_FINISH_REASONS

logger = logging.getLogger(__name__)


async def generate_hypotheses(
    client,
    row: pd.Series,
    args,
    semaphore: asyncio.Semaphore,
    scoring_queue: asyncio.Queue,
    stats: dict,
):
    async with semaphore:
        try:
            base_prompt, response_format = await prepare_generation_prompt(client, row, args)

            conversations = []
            truncated = 0
            for _ in range(args.n_hypos):
                answer, finish_reason = await generate_chat_message(client, base_prompt, args, response_format)
                if finish_reason in ENGINE_CUT_FINISH_REASONS:
                    # A fragment is not a hypothesis: the reward model would score it as a finished answer, landing in
                    # the preference file.
                    truncated += 1
                    continue
                conversations.append(base_prompt + [answer])
            if truncated:
                stats["truncated"] += truncated
            # argmax/argmin over a single hypothesis makes chosen == rejected.
            if len(conversations) < 2:
                stats["skipped_truncated"] += 1
                logger.warning(
                    f"Fewer than 2 usable hypotheses for {row.get(args.id_field, '?')} ({truncated} of "
                    f"{args.n_hypos} truncated at --max_tokens), row dropped"
                )
                return

            await scoring_queue.put((row, base_prompt, conversations))
        except TruncatedGenerationError as e:
            # The first turn of a follow-up row was cut, so every hypothesis would condition on a truncated generation.
            stats["truncated"] += 1
            stats["skipped_truncated"] += 1
            logger.warning(f"Truncated at --max_tokens for {row.get(args.id_field, '?')}: {e}")
        except Exception as e:
            # Counted: an uncounted dropped row makes a dead generation endpoint look like a short but clean run.
            stats["failed"] += 1
            logger.warning(f"Generation error for {row.get(args.id_field, '?')}: {e}")


def degenerate_hypotheses_reason(scores: Sequence[float], output_format: str) -> str | None:
    """Why a scored row cannot produce a useful record, or ``None``. Under 2 hypotheses or all-equal scores give chosen
    == rejected; ``offline_grpo`` keeps an all-equal vector, which the trainer's degenerate-group handling covers."""
    if len(scores) < 2:
        return f"insufficient hypotheses ({len(scores)} < 2)"
    if output_format != "offline_grpo" and len({float(score) for score in scores}) < 2:
        return f"all {len(scores)} hypotheses scored equally ({float(scores[0])})"
    return None


def best_worst_indices(scores: Sequence[float]) -> tuple[int, int]:
    """``(argmax, argmin)``, first index winning a tie, reading a list and a numpy vector alike."""
    ranked = range(len(scores))
    return max(ranked, key=scores.__getitem__), min(ranked, key=scores.__getitem__)


def build_preference_result(
    row,
    base_prompt,
    conversations,
    scores,
    args,
    correct_answer,
) -> dict:
    """The DPO/SMPO record (``agent-docs/data/dataset-formats.md``): best- and worst-scored hypothesis. Scores are
    plain floats, as ``numpy.float64`` does not survive ``json.dumps``. The whole scored set rides along so a
    re-pairing needs no re-generation; row id and ``target_answer`` only where the source has them."""
    completions = [conversation[-1] for conversation in conversations]
    best, worst = best_worst_indices(scores)
    record = {
        "prompt": base_prompt,
        "chosen": [completions[best]],
        "chosen_score": float(scores[best]),
        "rejected": [completions[worst]],
        "rejected_score": float(scores[worst]),
        "all_generations": completions,
        "all_scores": [float(score) for score in scores],
        "gen_model": args.model,
    }
    if args.id_field in row:
        record[args.id_field] = row[args.id_field]
    if correct_answer is not None:
        record["target_answer"] = correct_answer
    record["rm_model"] = args.rm_model_id
    return record


def build_offline_grpo_result(
    row,
    base_prompt,
    conversations,
    scores,
    args,
    correct_answer,
) -> dict:
    """The offline-GRPO record (``agent-docs/data/dataset-formats.md``): ``completions`` a list of message lists,
    ``rewards`` a parallel list of plain floats. The row id rides along only when the source carries it, so a dataset
    without the column gains no null one, and ``target_answer`` only where there is a ground truth."""
    record = {
        "prompt": base_prompt,
        "completions": [[conversation[-1]] for conversation in conversations],
        "rewards": [float(score) for score in scores],
    }
    if args.id_field in row:
        record[args.id_field] = row[args.id_field]
    if correct_answer is not None:
        record["target_answer"] = correct_answer
    return record


OUTPUT_FORMATS = {
    "preference": (build_preference_result, "rs"),
    "offline_grpo": (build_offline_grpo_result, "offline_grpo"),
}


async def score_and_select(
    scoring_queue: asyncio.Queue,
    result_queue: asyncio.Queue,
    executor: ThreadPoolExecutor,
    rm_tokenizer,
    rm_model,
    rm_device,
    args,
    stats: dict,
):
    build_fn, _ = OUTPUT_FORMATS[args.output_format]

    while True:
        # Outside the try: the cancellation arrives here, and a task_done() in the finally would count an untaken item.
        row, base_prompt, conversations = await scoring_queue.get()
        try:
            correct_answer = resolve_correct_answer(row, args)
            scores = await score_conversations_offloaded(
                executor, rm_tokenizer, rm_model, rm_device, conversations, correct_answer, args.rm_max_batch_size
            )

            reason = degenerate_hypotheses_reason(scores, args.output_format)
            if reason is not None:
                stats["skipped_degenerate"] += 1
                logger.warning(f"Skipping {row.get(args.id_field, '?')}: {reason}")
                continue

            result = build_fn(row, base_prompt, conversations, scores, args, correct_answer)
            await result_queue.put(result)
        except OverlongConversationError as e:
            stats["skipped_overlong"] += 1
            logger.warning(f"Over --rm_max_seq_len for {row.get(args.id_field, '?')}: row dropped ({e})")
        except ValueError as e:
            # The shape guard is a deterministic misconfiguration (wrong RM), so record it as fatal and let main abort.
            stats["fatal"] = str(e)
            stats["failed"] += 1
        except Exception as e:
            stats["failed"] += 1
            logger.warning(f"Scoring error: {e}")
        finally:
            scoring_queue.task_done()


async def save_results(
    result_queue: asyncio.Queue,
    output_path: Path,
    stats: dict,
):
    while True:
        result = await result_queue.get()
        try:
            append_jsonl_record(output_path, result)
            stats["written"] += 1
        except Exception as e:
            # Not row-local: ENOSPC or a read-only mount fail every following result, so catching this would drop the
            # run's output while reporting success.
            raise RuntimeError(
                f"Failed to append a result to {output_path} after {stats['written']} written "
                f"row(s): {e}. A write error is not row-local, so the run stops here instead of "
                f"discarding every remaining result and exiting 0."
            ) from e
        finally:
            result_queue.task_done()


async def join_worker_queue(queue: asyncio.Queue, worker: asyncio.Task) -> None:
    """Wait for ``queue`` to drain, surfacing ``worker``'s exception if it dies first: ``Queue.join`` would wait
    forever on items nobody will take."""
    join = asyncio.create_task(queue.join())
    done, _ = await asyncio.wait({join, worker}, return_when=asyncio.FIRST_COMPLETED)
    # Worker first: one dying as the queue drained lands in `done` beside the join, and the join would drop its error.
    if worker in done:
        join.cancel()
        await worker


async def run_rejection_sampling(args) -> None:
    client, rm_tokenizer, rm_model, rm_device = boot_scoring_run(args)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    _, suffix = OUTPUT_FORMATS[args.output_format]
    output_path = build_output_path(args.output_dir, args.dataset, args.model, suffix)

    df = load_prompts_dataframe(args)

    processed_ids, _existing = load_local_jsonl_resume(output_path, args.id_field)
    if processed_ids:
        logger.info(f"Skipping {len(processed_ids)} already completed prompts")

    pending = df[~df[args.id_field].isin(processed_ids)]
    if pending.empty:
        logger.info("All prompts already processed")
        return

    logger.info(f"Processing {len(pending)} prompts ({args.n_hypos} hypotheses each)...")

    scoring_queue = asyncio.Queue()
    result_queue = asyncio.Queue()
    semaphore = asyncio.Semaphore(args.max_workers)
    executor = ThreadPoolExecutor(max_workers=1)

    stats = {
        "written": 0,
        "failed": 0,
        "skipped_degenerate": 0,
        "truncated": 0,
        "skipped_truncated": 0,
        "skipped_overlong": 0,
        "fatal": None,
    }
    scoring_task = asyncio.create_task(
        score_and_select(
            scoring_queue,
            result_queue,
            executor,
            rm_tokenizer,
            rm_model,
            rm_device,
            args,
            stats,
        )
    )
    saving_task = asyncio.create_task(save_results(result_queue, output_path, stats))
    try:
        tasks = [
            generate_hypotheses(client, row, args, semaphore, scoring_queue, stats) for _, row in pending.iterrows()
        ]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await coro

        await join_worker_queue(scoring_queue, scoring_task)
        await join_worker_queue(result_queue, saving_task)
    finally:
        # So a worker that died mid-run does not leave its sibling pending when the exception unwinds.
        scoring_task.cancel()
        saving_task.cancel()
        executor.shutdown()

    if stats["fatal"]:
        raise RuntimeError(f"Reward-model scoring aborted: {stats['fatal']}")
    if stats["skipped_degenerate"]:
        logger.warning(
            f"Skipped {stats['skipped_degenerate']} prompt(s) with degenerate hypotheses "
            f"(<2 hypotheses or all-equal scores) — no preference signal to keep."
        )
    reject_empty_results(
        stats["written"],
        len(pending),
        output_path,
        drops=(
            f" (failed={stats['failed']}, degenerate={stats['skipped_degenerate']}, "
            f"truncated={stats['skipped_truncated']}, overlong={stats['skipped_overlong']})"
        ),
        check="--rm_model_id, --base_url, --rm_max_seq_len and the input columns",
    )
    logger.info(
        f"Results saved to {output_path} ({stats['written']} written, {stats['failed']} failed, "
        f"{stats['skipped_degenerate']} degenerate skipped, {stats['skipped_truncated']} truncated skipped "
        f"({stats['truncated']} truncated hypotheses), {stats['skipped_overlong']} over --rm_max_seq_len skipped, "
        f"{len(pending)} attempted)"
    )
