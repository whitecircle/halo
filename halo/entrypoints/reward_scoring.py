"""Reward-model scoring for the generation CLIs: the reward model's load, the batched scorer and its
event-loop offload, the generation and answer-context helpers, the local-JSONL output and its resume, and the
one-response scoring run."""

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

# Reply fields a generated turn never carries: the role and content assistant_turn writes, and the
# tool-call surface these scripts never request.
_REPLY_FIELDS_DROPPED = {"role", "content", "function_call", "tool_calls", "refusal", "audio"}

# Longest basename these scripts will build, in bytes. Common filesystems cap a single name at 255
# bytes, and the failure surfaces as an uncaught OSError(ENAMETOOLONG) from the first
# `output_path.exists()`, after the reward model is already resident on the GPU. Reachable in
# practice: vLLM's --served-model-name defaults to the served model's path, which --model then
# carries.
_MAX_OUTPUT_BASENAME = 255


class TruncatedGenerationError(RuntimeError):
    """A generation stopped at ``--max_tokens`` instead of finishing.

    Its own class rather than ``ValueError``: the scoring scripts treat ``ValueError`` as a fatal
    misconfiguration, while a truncated hypothesis is a per-row drop.
    """


class OverlongConversationError(RuntimeError):
    """A conversation to score exceeds ``--rm_max_seq_len``.

    A per-row drop like :class:`TruncatedGenerationError` rather than a truncation: right-truncation
    cuts the assistant turn, and the reward model would score the prompt fragment as the answer.
    """


def build_output_path(output_dir: str, dataset: str, model_name: str, suffix: str) -> Path:
    """``<prompts stem>_<model>_<suffix>.jsonl`` under ``output_dir``.

    The generating model is part of the name because both scripts resume from the ids already present
    in this file: without it a second model's run would read the first model's rows as its own.

    A ``/`` in a hub id would otherwise open a directory that does not exist. An over-long name is cut
    down to fit, keeping the suffix and a digest of the identifying pair; the cap is measured in bytes
    and the digest keeps two cut names from colliding.
    """
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
    """Load reward model and tokenizer in ``dtype`` (a :data:`DTYPE_BY_NAME` name).

    ``max_seq_len`` becomes the tokenizer's ``model_max_length``, the cap :func:`encode_for_scoring`
    refuses conversations past.
    """
    # A per-rank EP-sharded dir loads its non-expert weights under their true names while every expert
    # is randomly initialized, so the scores would be plausible-looking noise.
    reject_partial_checkpoint(model_path)
    logger.info(f"Loading Reward Model from {model_path}...")
    # Before the config fetch, which can already import a remote modeling file.
    apply_remote_code_compat_shims()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    tokenizer.model_max_length = max_seq_len

    device = torch.device(device_str)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    # On CUDA, the family patches the training loads take, keyed on this model's config. A CPU scorer builds the
    # backend as given, since the wide-head SDPA pin turns off the backends CPU SDPA runs on.
    build_attn_impl = apply_family_patches(config, attn_impl) if device.type == "cuda" else attn_impl
    # excuse_task_head=False: scoring consumes the reward head, it never trains one. Pointed at a
    # plain causal-LM directory the head would initialize randomly and every score would be noise.
    model = from_pretrained_verified(
        AutoModelForSequenceClassification,
        model_path,
        config=config,
        excuse_task_head=False,
        attn_implementation=build_attn_impl,
        trust_remote_code=trust_remote_code,
    )
    model.to(dtype=DTYPE_BY_NAME[dtype], device=device).eval()
    # After the cast, not before: transformers 5 hands every non-persistent buffer back uninitialized
    # (a remote-code family overriding ``_init_weights`` scores off a dead RoPE), and
    # ``Module.to(dtype=...)`` downcasts float buffers too, so repairing first would write the fp32
    # rotary table and then round it to bf16.
    finalize_loaded_model(model)
    # No run policy applies here: the checkpoint's own sinks decide whether --rm_attn_implementation may drop them.
    verify_live_sinks_carried(model)
    return tokenizer, model, device


def boot_scoring_run(args) -> tuple[Any, Any, Any, torch.device]:
    """Open the two endpoints a scoring run needs: ``(api client, rm tokenizer, rm model, rm device)``.

    One call for both scripts, so neither drifts onto a different set of ``--rm_*`` knobs; every flag
    the reward model is loaded under is threaded here.
    """
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
    """The row's ground-truth answer (``args.correct_answer_field``), or None."""
    if args.correct_answer_field in row and pd.notna(row[args.correct_answer_field]):
        return row[args.correct_answer_field]
    return None


async def generate_chat_message(client, messages: list[dict], args, response_format: dict) -> tuple[dict, str]:
    """One chat completion under the script's generation args.

    Returns ``(assistant turn, finish_reason)``. The turn is :func:`assistant_turn`'s plus the reply's
    non-null extra fields, so an engine's reasoning (``reasoning_content``) reaches the chat template the
    reward model scores with and the record a preference run trains on; the null SDK fields stay out.
    The finish reason is part of the contract because a hypothesis cut at the token cap is a fragment:
    the reward model would score it as a complete answer and the number would land in a preference /
    offline-GRPO file.
    """
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
    """Assemble the row's generation prompt: system prompt + base messages, applying the optional
    follow-up turn (generate an initial response, then append it plus the follow-up messages).

    Returns ``(base_prompt, response_format)`` ready for the script's own generation calls.
    """
    system_prompt = resolve_system_prompt(row, args.local_system_prompt_field, args.global_system_prompt)
    base_prompt = build_base_prompt(row, args.prompt_field, system_prompt)
    has_format = "response_format" in row and pd.notna(row["response_format"])
    response_format = row["response_format"] if has_format else {"type": "text"}

    follow_up = follow_up_messages(row, args.follow_up_prompt_field)
    if follow_up is not None:
        answer, finish_reason = await generate_chat_message(client, base_prompt, args, response_format)
        if finish_reason in ENGINE_CUT_FINISH_REASONS:
            # The follow-up turn conditions on this answer, so a fragment (token cap or engine abort)
            # here corrupts every hypothesis the row goes on to produce.
            raise TruncatedGenerationError(
                f"the first turn hit --max_tokens ({args.max_tokens}) and the follow-up turn "
                f"would condition on a fragment"
            )
        base_prompt = base_prompt + [answer] + follow_up

    return base_prompt, response_format


def prepend_answer_context(conversation: list[dict], correct_answer: str | None) -> list[dict]:
    """``conversation`` with the ground-truth answer pinned as a leading system turn, when known.

    Both scoring paths have to show the RM the same context, or their scores are not comparable. The
    image refusal sits here because every scored conversation passes this seam on its way to the
    reward model's chat template: reward models are text-only sequence classifiers, so an image part
    would render as placeholder tokens with no pixels behind it and the returned reward would be
    computed over those, then written into a preference / offline-GRPO training file.
    """
    reject_image_content(conversation, "reward-model scoring conversation")
    if correct_answer is None:
        return conversation
    return [{"role": "system", "content": f"The correct final answer must be: {correct_answer}"}] + conversation


def encode_for_scoring(tokenizer, conversations: list[list[dict]], correct_answer: str | None) -> dict:
    """Tokenize ``conversations`` for the reward model the way its training rows were tokenized.

    Render through the chat template, then :func:`tokenize_rendered`, so BOS lands once on a
    tokenizer whose post-processor prepends it, as in TRL's reward tokenize map (``apply_chat_template``
    tokenizing its own render without special tokens): re-tokenizing the rendered text would double
    it and the scores would come from a token sequence the model never trained on. A conversation
    past ``tokenizer.model_max_length`` (``--rm_max_seq_len``) raises :class:`OverlongConversationError`.
    """
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
    """Reward-model scores for ``conversations``, scored ``max_batch_size`` at a time."""
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
    """:func:`score_conversations` on ``executor``'s thread, awaited from the event loop.

    The reward-model forward is a blocking GPU call; run inline it stalls every generation still in
    flight on the same loop. One offload for both scripts, so neither drifts off the other's argument
    order into scoring with the wrong batch size or a missing answer context.
    """
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
    """A JSONL file as an object-dtype DataFrame holding each field as the value ``json`` parses.

    ``pd.read_json`` rewrites values: by default ``"007"`` becomes ``7`` and a ``*_at`` column
    timestamps, and even with its inference off an int column beside a row lacking the field turns
    float (``5`` reaches the reward model's prompt as ``5.0``) and ``0.3`` parses as
    ``0.30000000000000004``. A field a row lacks reads as NaN. One reader for the prompts and the
    resume output, so an id compares equal across the two.
    """
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(records, dtype=object)


def load_prompts_dataframe(args) -> pd.DataFrame:
    """Load the JSONL prompts source and validate the required columns."""
    logger.info("Loading prompts...")
    df = read_jsonl_frame(args.dataset)
    if args.prompt_field not in df or args.id_field not in df:
        raise ValueError(f"Input must contain '{args.prompt_field}' and '{args.id_field}' columns")
    return df


def load_local_jsonl_resume(output_path: Path, id_field: str) -> tuple[set, pd.DataFrame]:
    """``(ids already written, the rows carrying them)`` for a run resuming its own JSONL output.

    The local-file twin of :func:`~halo.entrypoints.resumable_generation.load_prompts_with_resume` (S3 datasets):
    both scripts append one finished row per line, so the ids in the output file are the work a re-run
    skips. An output that does not exist yet resumes nothing. The rows themselves come back because a
    caller reporting summary statistics over the whole job needs them, not just their ids.
    """
    if not output_path.exists():
        return set(), pd.DataFrame()
    existing = read_jsonl_frame(output_path)
    return set(existing[id_field]), existing


def append_jsonl_record(output_path: Path, record: dict) -> None:
    """Append one finished row to the run's JSONL output, the file :func:`load_local_jsonl_resume` reads
    back. The write holds no ``await``, so coroutines sharing the loop never interleave lines."""
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
    """Generate a response and evaluate with reward model."""
    async with semaphore:
        try:
            base_prompt, response_format = await prepare_generation_prompt(client, row, args)

            response, finish_reason = await generate_chat_message(client, base_prompt, args, response_format)
            if finish_reason in ENGINE_CUT_FINISH_REASONS:
                # Scoring a fragment (token cap or engine abort) as a finished answer would write a
                # reward for text the policy never finished, so the row is dropped, counted and reported.
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
            # The first turn of a follow-up row was cut, so the scored turn would have conditioned on
            # a fragment: a truncation like the final-turn cut above, not a generation failure.
            stats["truncated"] += 1
            logger.warning(f"Truncated at --max_tokens for {row.get(args.id_field, '?')}: {e}")
        except ValueError:
            # The single-output shape guard (score_conversations) is a deterministic
            # misconfiguration, so re-raise and let the run abort rather than scoring nothing.
            raise
        except Exception as e:
            # Counted rather than only printed: the statistics below are computed over the surviving
            # rows, so an uncounted drop would make a run that scored a tenth of its prompts read as
            # complete.
            stats["failed"] += 1
            logger.warning(f"Error for {row.get(args.id_field, '?')}: {e}")


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


async def run_reward_scoring(args) -> None:
    """Generate one response per pending prompt, score it with the reward model, append it to the run's
    JSONL output and print summary statistics over every scored row, resumed ones included."""
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

    # The resumed rows are already in `rewards` (they belong in the statistics), so this run's own
    # output is the growth from here, not the final length.
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

        # Keyed on the growth: a length check passes as soon as one resumed row is loaded, so a
        # resumed run against a dead endpoint would report the previous run's rows as its own output
        # and exit 0. An empty `pending` never reaches here, where producing nothing is correct.
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
