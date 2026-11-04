"""Reward-model scoring for the generation CLIs: the model load, the batched scorer, answer-context helpers, the
local-JSONL output with resume, and the one-response run."""

import asyncio
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from halo.checkpoint.reading import reject_partial_checkpoint
from halo.data.chat_render import tokenize_rendered
from halo.data.conversation import build_base_prompt, reject_image_content, resolve_system_prompt
from halo.entrypoints.resumable_generation import assistant_turn, follow_up_messages, reject_empty_results
from halo.inference.openai_client import build_openai_client, chat_completion
from halo.inference.response import ENGINE_CUT_FINISH_REASONS, get_finish_reason
from halo.model_loading.checkpoint_coverage import from_pretrained_verified
from halo.model_loading.dtype import DTYPE_BY_NAME
from halo.model_loading.finalization import finalize_loaded_model
from halo.model_loading.model_preparation import apply_family_patches
from halo.model_loading.remote_code import apply_remote_code_compat_shims
from halo.models.attention.selection import verify_live_sinks_carried

logger = logging.getLogger(__name__)

# The role and content assistant_turn writes, and the tool-call surface these scripts never request.
_REPLY_FIELDS_DROPPED = {"role", "content", "function_call", "tool_calls", "refusal", "audio"}

# Bytes. Filesystems cap a single name at 255, and the failure is an uncaught ENAMETOOLONG after the reward model is on
# the GPU; vLLM's --served-model-name defaults to the model path, which --model then carries.
_MAX_OUTPUT_BASENAME = 255


class TruncatedGenerationError(RuntimeError):
    """A generation stopped at ``--max_tokens``: a per-row drop, where the scripts treat ``ValueError`` as fatal."""


class OverlongConversationError(RuntimeError):
    """A conversation past ``--rm_max_seq_len``: a per-row drop, since right-truncation cuts the assistant turn and the
    model would score the prompt fragment as the answer."""


def build_output_path(output_dir: str, dataset: str, model_name: str, suffix: str) -> Path:
    """``<prompts stem>_<model>_<suffix>.jsonl``. The model is in the name because both scripts resume from ids in this
    file, so a second model would read the first's rows as its own. A ``/`` in a hub id would open a missing directory.
    An over-long name is cut in bytes, keeping the suffix and a digest of the pair so two cut names cannot collide."""
    stem = Path(dataset).stem
    basename = f"{stem}_{model_name.replace('/', '__')}_{suffix}.jsonl"
    if len(basename.encode()) <= _MAX_OUTPUT_BASENAME:
        return Path(output_dir) / basename

    digest = hashlib.sha256(f"{stem}\0{model_name}".encode()).hexdigest()[:16]
    tail = f"_{digest}_{suffix}.jsonl"
    keep = max(_MAX_OUTPUT_BASENAME - len(tail.encode()), 0)
    head = basename.encode()[:keep].decode(errors="ignore")
    return Path(output_dir) / f"{head}{tail}"


def load_reward_model(
    model_path: str, attn_impl: str, max_seq_len: int, device_str: str, dtype: str, *, trust_remote_code: bool
):
    """``max_seq_len`` becomes ``model_max_length``, the cap :func:`encode_for_scoring` refuses conversations past."""
    # A per-rank EP-sharded dir loads non-expert weights while every expert is random, so scores would be noise.
    reject_partial_checkpoint(model_path)
    logger.info(f"Loading Reward Model from {model_path}...")
    # Before the config fetch, which can already import a remote modeling file.
    apply_remote_code_compat_shims()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    tokenizer.model_max_length = max_seq_len

    device = torch.device(device_str)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    # On CUDA, the training loads' family patches. A CPU scorer builds the backend as given, since the wide-head SDPA
    # pin turns off the backends CPU SDPA runs on.
    build_attn_impl = apply_family_patches(config, attn_impl) if device.type == "cuda" else attn_impl
    # excuse_task_head=False: on a plain causal-LM directory the head would be random and every score noise.
    model = from_pretrained_verified(
        AutoModelForSequenceClassification,
        model_path,
        config=config,
        excuse_task_head=False,
        attn_implementation=build_attn_impl,
        trust_remote_code=trust_remote_code,
    )
    model.to(dtype=DTYPE_BY_NAME[dtype], device=device).eval()
    # After the cast: transformers 5 hands non-persistent buffers back uninitialized, and ``Module.to(dtype=...)``
    # downcasts float buffers too, so repairing first would round the fp32 rotary table to bf16.
    finalize_loaded_model(model)
    # No run policy applies here: the checkpoint's own sinks decide whether --rm_attn_implementation may drop them.
    verify_live_sinks_carried(model)
    return tokenizer, model, device


def boot_scoring_run(args) -> tuple[Any, Any, Any, torch.device]:
    """``(api client, rm tokenizer, rm model, rm device)``; one call for both scripts, so ``--rm_*`` knobs match."""
    client = build_openai_client(base_url=args.base_url, api_key_override=args.api_key)
    rm_tokenizer, rm_model, rm_device = load_reward_model(
        args.rm_model_id,
        args.rm_attn_implementation,
        args.rm_max_seq_len,
        args.rm_device,
        args.rm_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    return client, rm_tokenizer, rm_model, rm_device


def resolve_correct_answer(row, args):
    if args.correct_answer_field in row and pd.notna(row[args.correct_answer_field]):
        return row[args.correct_answer_field]
    return None


async def generate_chat_message(client, messages: list[dict], args, response_format: dict) -> tuple[dict, str]:
    """``(assistant turn, finish_reason)``. The turn is :func:`assistant_turn`'s plus the reply's non-null extras, so
    an engine's ``reasoning_content`` reaches the reward model's template and the training record. The finish reason
    matters because a hypothesis cut at the token cap would be scored as a complete answer."""
    completion = await chat_completion(
        client,
        messages=messages,
        model=args.model,
        temperature=args.temperature,
        response_format=response_format,
        max_tokens=args.max_tokens,
    )
    choice = completion.choices[0]
    extras = choice.message.model_dump(exclude=_REPLY_FIELDS_DROPPED)
    turn = {**assistant_turn(choice.message.content), **{k: v for k, v in extras.items() if v is not None}}
    return turn, get_finish_reason(choice) or ""


async def prepare_generation_prompt(client, row: pd.Series, args) -> tuple[list[dict], dict]:
    """System prompt + base messages, applying the optional follow-up turn; ``(base_prompt, response_format)``."""
    system_prompt = resolve_system_prompt(row, args.local_system_prompt_field, args.global_system_prompt)
    base_prompt = build_base_prompt(row, args.prompt_field, system_prompt)
    has_format = "response_format" in row and pd.notna(row["response_format"])
    response_format = row["response_format"] if has_format else {"type": "text"}

    follow_up = follow_up_messages(row, args.follow_up_prompt_field)
    if follow_up is not None:
        answer, finish_reason = await generate_chat_message(client, base_prompt, args, response_format)
        if finish_reason in ENGINE_CUT_FINISH_REASONS:
            # The follow-up conditions on this answer, so a fragment here corrupts every hypothesis the row produces.
            raise TruncatedGenerationError(
                f"the first turn hit --max_tokens ({args.max_tokens}) and the follow-up turn "
                f"would condition on a fragment"
            )
        base_prompt = base_prompt + [answer] + follow_up

    return base_prompt, response_format


def prepend_answer_context(conversation: list[dict], correct_answer: str | None) -> list[dict]:
    """Pin the ground truth as a leading system turn, when known, so both scoring paths show the RM the same context.
    Every scored conversation passes here, so images are refused here: a text-only reward model would render
    placeholder tokens with no pixels and its reward would land in a training file."""
    reject_image_content(conversation, "reward-model scoring conversation")
    if correct_answer is None:
        return conversation
    return [{"role": "system", "content": f"The correct final answer must be: {correct_answer}"}] + conversation


def encode_for_scoring(tokenizer, conversations: list[list[dict]], correct_answer: str | None) -> dict:
    """Tokenize as the training rows were: render through the chat template, then :func:`tokenize_rendered`, so BOS
    lands once; re-tokenizing the rendered text would double it. A conversation past ``model_max_length`` raises
    :class:`OverlongConversationError`."""
    encoded = [
        tokenize_rendered(
            tokenizer,
            tokenizer.apply_chat_template(prepend_answer_context(conv, correct_answer), tokenize=False),
            truncation=False,
        )
        for conv in conversations
    ]
    longest = max(len(item["input_ids"]) for item in encoded)
    if longest > tokenizer.model_max_length:
        raise OverlongConversationError(
            f"a conversation renders to {longest} tokens, over --rm_max_seq_len={tokenizer.model_max_length}"
        )
    return tokenizer.pad(encoded, padding=True, return_tensors="pt")


def score_conversations(
    rm_tokenizer,
    rm_model,
    rm_device,
    conversations: list[list[dict]],
    correct_answer: str | None,
    max_batch_size: int,
) -> np.ndarray:
    scores: list[float] = []
    for i in range(0, len(conversations), max_batch_size):
        batch = encode_for_scoring(rm_tokenizer, conversations[i : i + max_batch_size], correct_answer)
        batch = {key: value.to(rm_device) for key, value in batch.items()}
        with torch.inference_mode():
            logits = rm_model(**batch).logits
        # A multi-class classifier would hand back an arbitrary, possibly misaligned, class logit as the reward.
        if logits.shape[-1] != 1:
            raise ValueError(
                f"Expected a single-output reward model, got logits shape {tuple(logits.shape)} "
                f"— this looks like a multi-class classifier, not an RM."
            )
        # .float() first: numpy has no bf16 dtype, and bf16 is the scoring default.
        scores.extend(logits.detach().cpu().float().view(-1).tolist())
    return np.array(scores, dtype=float)


async def score_conversations_offloaded(
    executor: ThreadPoolExecutor,
    rm_tokenizer,
    rm_model,
    rm_device,
    conversations: list[list[dict]],
    correct_answer: str | None,
    max_batch_size: int,
) -> np.ndarray:
    """Offloaded: the forward is a blocking GPU call that would stall every generation in flight on the loop. One
    offload for both scripts, so neither drifts off the other's argument order."""
    return await asyncio.get_running_loop().run_in_executor(
        executor,
        score_conversations,
        rm_tokenizer,
        rm_model,
        rm_device,
        conversations,
        correct_answer,
        max_batch_size,
    )


def read_jsonl_frame(path: str | Path) -> pd.DataFrame:
    """JSONL as an object-dtype DataFrame of ``json``-parsed values; a missing field reads as NaN. ``pd.read_json``
    rewrites values (``"007"`` to ``7``, ``*_at`` to timestamps, an int beside a missing field to float, ``0.3`` to
    ``0.30000000000000004``). One reader for prompts and resume output, so ids compare equal."""
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(records, dtype=object)


def load_prompts_dataframe(args) -> pd.DataFrame:
    logger.info("Loading prompts...")
    df = read_jsonl_frame(args.dataset)
    if args.prompt_field not in df or args.id_field not in df:
        raise ValueError(f"Input must contain '{args.prompt_field}' and '{args.id_field}' columns")
    return df


def load_local_jsonl_resume(output_path: Path, id_field: str) -> tuple[set, pd.DataFrame]:
    """``(ids already written, their rows)`` for a run resuming its own JSONL output; the local twin of
    ``load_prompts_with_resume``. The rows come back for summary statistics over the whole job."""
    if not output_path.exists():
        return set(), pd.DataFrame()
    existing = read_jsonl_frame(output_path)
    return set(existing[id_field]), existing


def append_jsonl_record(output_path: Path, record: dict) -> None:
    """Append one finished row. The write holds no ``await``, so coroutines sharing the loop never interleave lines."""
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def generate_and_evaluate(
    client,
    row: pd.Series,
    args,
    semaphore: asyncio.Semaphore,
    output_path: Path,
    executor: ThreadPoolExecutor,
    rm_tokenizer,
    rm_model,
    rm_device,
    responses: list,
    rewards: list,
    stats: dict,
):
    async with semaphore:
        try:
            base_prompt, response_format = await prepare_generation_prompt(client, row, args)

            response, finish_reason = await generate_chat_message(client, base_prompt, args, response_format)
            if finish_reason in ENGINE_CUT_FINISH_REASONS:
                # A fragment scored as finished writes a reward for text the policy never finished: drop and count it.
                stats["truncated"] += 1
                logger.warning(f"Truncated at --max_tokens for {row.get(args.id_field, '?')}: dropped")
                return
            full_conversation = base_prompt + [response]

            correct_answer = resolve_correct_answer(row, args)
            scores = await score_conversations_offloaded(
                executor, rm_tokenizer, rm_model, rm_device, [full_conversation], correct_answer, 1
            )
            reward = float(scores[0])

            record = {
                args.id_field: row[args.id_field],
                "rm_model": args.rm_model_id,
                "temperature": args.temperature,
                "response": response["content"],
                "reward": reward,
                "full_conversation": full_conversation,
            }
            if correct_answer is not None:
                record["target_answer"] = correct_answer

            append_jsonl_record(output_path, record)

            responses.append(response["content"])
            rewards.append(reward)
        except OverlongConversationError as e:
            stats["overlong"] += 1
            logger.warning(f"Over --rm_max_seq_len for {row.get(args.id_field, '?')}: row dropped ({e})")
        except TruncatedGenerationError as e:
            # The first turn of a follow-up row was cut, so the scored turn would condition on a fragment.
            stats["truncated"] += 1
            logger.warning(f"Truncated at --max_tokens for {row.get(args.id_field, '?')}: {e}")
        except ValueError:
            # The shape guard is a deterministic misconfiguration, so abort rather than score nothing.
            raise
        except Exception as e:
            # Counted: statistics cover surviving rows, so an uncounted drop would make a run that scored a tenth of
            # its prompts read as complete.
            stats["failed"] += 1
            logger.warning(f"Error for {row.get(args.id_field, '?')}: {e}")


def print_statistics(responses: list, rewards: list, model_name: str):
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

    table.add_row("Mean", f"{np.mean(response_lengths):.4f}", f"{np.mean(rewards):.4f}")
    table.add_row("Std", f"{np.std(response_lengths):.4f}", f"{np.std(rewards):.4f}")
    table.add_row("Min", f"{np.min(response_lengths):.4f}", f"{np.min(rewards):.4f}")
    table.add_row("Max", f"{np.max(response_lengths):.4f}", f"{np.max(rewards):.4f}")

    for pct, label in percentiles:
        table.add_row(
            label,
            f"{np.percentile(response_lengths, pct):.4f}",
            f"{np.percentile(rewards, pct):.4f}",
        )

    console.print(table)


async def run_reward_scoring(args) -> None:
    client, rm_tokenizer, rm_model, rm_device = boot_scoring_run(args)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = build_output_path(args.output_dir, args.dataset, args.model, "reward_scoring")

    df = load_prompts_dataframe(args)

    responses = []
    rewards = []

    # Resumed rows belong in the statistics below, so they are seeded into the running lists.
    processed_ids, existing = load_local_jsonl_resume(output_path, args.id_field)
    if processed_ids:
        responses.extend(existing["response"].tolist())
        rewards.extend(existing["reward"].tolist())
        logger.info(f"Loaded {len(processed_ids)} existing results")

    # The resumed rows are already in `rewards`, so this run's output is the growth from here.
    resumed_count = len(rewards)
    pending = df[~df[args.id_field].isin(processed_ids)]
    stats = {"failed": 0, "truncated": 0, "overlong": 0}

    if not pending.empty:
        logger.info(f"Processing {len(pending)} prompts...")

        semaphore = asyncio.Semaphore(args.max_workers)
        executor = ThreadPoolExecutor(max_workers=1)

        try:
            tasks = [
                generate_and_evaluate(
                    client,
                    row,
                    args,
                    semaphore,
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

        # Keyed on the growth: a length check would pass on one resumed row, so a resumed run against a dead endpoint
        # would report the previous run's rows and exit 0.
        reject_empty_results(
            len(rewards) - resumed_count,
            len(pending),
            output_path,
            drops=f" (failed={stats['failed']}, truncated={stats['truncated']}, overlong={stats['overlong']})",
            check="--model, the endpoint, and that the RM has a chat template",
        )

    print_statistics(responses, rewards, args.model)
    logger.info(
        f"Results saved to {output_path} ({len(rewards) - resumed_count} scored / {stats['failed']} failed / "
        f"{stats['truncated']} truncated / {stats['overlong']} over --rm_max_seq_len / "
        f"{len(pending)} attempted this run; {len(rewards)} scored in total including {resumed_count} resumed)"
    )
