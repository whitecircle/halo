"""Score one generated response per prompt with a local reward model.

Generates against an OpenAI-compatible API, scores with the reward model, prints summary statistics.

Usage:
    python scripts/inference/reward_model/rm_scoring.py \
        --model_name my-model \
        --prompts_source data/prompts.jsonl \
        --rm_model_path path/to/reward-model

Input JSONL format:
    {"id": "...", "prompt": [{"role": "user", "content": "..."}], ...}

Output JSONL format:
    {"id": "...", "response": "...", "reward": 0.85, "full_conversation": [...], ...}
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from scripts.inference._common import reject_empty_results, run_async_cli
from scripts.inference.reward_model._common import (
    OverlongConversationError,
    boot_scoring_run,
    build_generation_parser,
    build_output_path,
    generate_chat_message,
    load_local_jsonl_resume,
    load_prompts_dataframe,
    prepare_generation_prompt,
    resolve_correct_answer,
    score_conversations_offloaded,
)
from src.inference.response import FINISH_REASON_LENGTH


def parse_args():
    parser = build_generation_parser("Async generation with RM scoring", temperature_default=0.0)
    return parser.parse_args()


async def generate_and_evaluate(
    client,
    row: pd.Series,
    args,
    semaphore: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    output_path: Path,
    executor: ThreadPoolExecutor,
    rm_tokenizer,
    rm_model,
    rm_device,
    responses: list,
    rewards: list,
    stats: dict,
):
    """Generate a response and evaluate with reward model."""
    async with semaphore:
        try:
            base_prompt, response_format = await prepare_generation_prompt(client, row, args)

            response, finish_reason = await generate_chat_message(client, base_prompt, args, response_format)
            if finish_reason == FINISH_REASON_LENGTH:
                # Scoring a fragment as a finished answer would write a reward for text the policy
                # never finished, so the row is dropped, counted and reported.
                stats["truncated"] += 1
                print(f"Truncated at --max_gen_tokens for {row.get(args.id_field, '?')}: dropped")
                return
            full_conversation = base_prompt + [response]

            correct_answer = resolve_correct_answer(row, args)
            scores = await score_conversations_offloaded(
                executor, rm_tokenizer, rm_model, rm_device, [full_conversation], correct_answer, 1
            )
            reward = float(scores[0])

            record = {
                args.id_field: row[args.id_field],
                "rm_model": args.rm_model_path,
                "temperature": args.temperature,
                "response": response["content"],
                "reward": reward,
                "full_conversation": full_conversation,
            }
            if correct_answer is not None:
                record["target_answer"] = correct_answer

            async with write_lock:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            responses.append(response["content"])
            rewards.append(reward)
        except OverlongConversationError as e:
            stats["overlong"] += 1
            print(f"Over --rm_max_seq_len for {row.get(args.id_field, '?')}: row dropped ({e})")
        except ValueError:
            # The single-output shape guard (score_conversations) is a deterministic
            # misconfiguration, so re-raise and let the run abort rather than scoring nothing.
            raise
        except Exception as e:
            # Counted rather than only printed: the statistics below are computed over the surviving
            # rows, so an uncounted drop would make a run that scored a tenth of its prompts read as
            # complete.
            stats["failed"] += 1
            print(f"Error for {row.get(args.id_field, '?')}: {e}")


def print_statistics(responses: list, rewards: list, model_name: str):
    """Print summary statistics using rich table."""
    console = Console()

    if not responses or not rewards:
        console.print("\n[bold red]No data to calculate statistics.[/bold red]")
        return

    response_lengths = [len(r) for r in responses]
    percentiles = [(3, "3%"), (10, "10%"), (25, "25%"), (50, "Median"), (75, "75%"), (90, "90%"), (97, "97%")]

    table = Table(title=f"Generation Statistics: {model_name}", show_lines=True)
    table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    table.add_column("Response Length", justify="right", style="green")
    table.add_column("Reward", justify="right", style="magenta")

    # Aggregate stats
    table.add_row("Mean", f"{np.mean(response_lengths):.4f}", f"{np.mean(rewards):.4f}")
    table.add_row("Std", f"{np.std(response_lengths):.4f}", f"{np.std(rewards):.4f}")
    table.add_row("Min", f"{np.min(response_lengths):.4f}", f"{np.min(rewards):.4f}")
    table.add_row("Max", f"{np.max(response_lengths):.4f}", f"{np.max(rewards):.4f}")

    # Percentiles
    for pct, label in percentiles:
        table.add_row(
            label,
            f"{np.percentile(response_lengths, pct):.4f}",
            f"{np.percentile(rewards, pct):.4f}",
        )

    console.print(table)


async def main():
    args = parse_args()

    client, rm_tokenizer, rm_model, rm_device = boot_scoring_run(args)

    # Output path
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    output_path = build_output_path(args.output_folder, args.prompts_source, args.model_name, "rm_scoring")

    # Load prompts
    df = load_prompts_dataframe(args)

    responses = []
    rewards = []

    # Resumed rows belong in the statistics below, so they are seeded into the running lists.
    processed_ids, existing = load_local_jsonl_resume(output_path, args.id_field)
    if processed_ids:
        responses.extend(existing["response"].tolist())
        rewards.extend(existing["reward"].tolist())
        print(f"Loaded {len(processed_ids)} existing results")

    # The resumed rows are already in `rewards` (they belong in the statistics), so this run's own
    # output is the growth from here, not the final length.
    resumed_count = len(rewards)
    pending = df[~df[args.id_field].isin(processed_ids)]
    stats = {"failed": 0, "truncated": 0, "overlong": 0}

    if not pending.empty:
        print(f"Processing {len(pending)} prompts...")

        semaphore = asyncio.Semaphore(args.n_parallel)
        write_lock = asyncio.Lock()
        executor = ThreadPoolExecutor(max_workers=1)

        try:
            tasks = [
                generate_and_evaluate(
                    client,
                    row,
                    args,
                    semaphore,
                    write_lock,
                    output_path,
                    executor,
                    rm_tokenizer,
                    rm_model,
                    rm_device,
                    responses,
                    rewards,
                    stats,
                )
                for _, row in pending.iterrows()
            ]
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                await coro
        finally:
            executor.shutdown()

        # Keyed on the growth: a length check passes as soon as one resumed row is loaded, so a
        # resumed run against a dead endpoint would report the previous run's rows as its own output
        # and exit 0. An empty `pending` never reaches here, where producing nothing is correct.
        reject_empty_results(
            len(rewards) - resumed_count,
            len(pending),
            output_path,
            drops=f" (failed={stats['failed']}, truncated={stats['truncated']}, overlong={stats['overlong']})",
            check="--model_name, the endpoint, and that the RM has a chat template",
        )

    print_statistics(responses, rewards, args.model_name)
    print(
        f"Results saved to {output_path} ({len(rewards) - resumed_count} scored / {stats['failed']} failed / "
        f"{stats['truncated']} truncated / {stats['overlong']} over --rm_max_seq_len / "
        f"{len(pending)} attempted this run; {len(rewards)} scored in total including {resumed_count} resumed)"
    )


if __name__ == "__main__":
    run_async_cli(main)
