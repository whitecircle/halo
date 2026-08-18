"""Shared helpers for the reward-model inference scripts (``rm_scoring`` / ``rm_rejection_sampling``).

Holds the run's boot (API client + reward model), the shared CLI parser, the generation helpers, the
batched scorer and its event-loop offload, and the local-JSONL resume both scripts share. Generic
prompt assembly (``resolve_system_prompt`` / ``build_base_prompt``) lives in
``src.data.pipeline.conversation`` — rows there may be a pandas Series or a plain dict.
"""

import argparse
import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts._common import add_trust_remote_code_arg
from scripts.inference._common import add_generation_args, add_openai_endpoint_args
from src.checkpoint.tool_io import reject_sharded_checkpoint
from src.data.pipeline.conversation import build_base_prompt, reject_image_content, resolve_system_prompt
from src.data.pipeline.rendered import tokenize_rendered
from src.inference.openai_client import create_openai_client
from src.inference.response import FINISH_REASON_LENGTH, get_finish_reason
from src.models.loading.checkpoint_coverage import from_pretrained_verified
from src.models.loading.dtype import DTYPE_BY_NAME
from src.models.patches.buffer_fixes import finalize_loaded_model

# Fields dropped from a serialized assistant message before it is re-appended to a conversation.
_MESSAGE_DUMP_EXCLUDE = {"function_call", "tool_calls", "refusal", "audio"}

# Longest basename these scripts will build, in bytes. Common filesystems cap a single name at 255
# bytes, and the failure surfaces as an uncaught OSError(ENAMETOOLONG) from the first
# `output_path.exists()`, after the reward model is already resident on the GPU. Reachable in
# practice: vLLM's --served-model-name defaults to the served model's path, which --model_name then
# carries.
_MAX_OUTPUT_BASENAME = 255


class TruncatedGenerationError(RuntimeError):
    """A generation stopped at ``--max_gen_tokens`` instead of finishing.

    Its own class rather than ``ValueError``: the scoring scripts treat ``ValueError`` as a fatal
    misconfiguration, while a truncated hypothesis is a per-row drop.
    """


class OverlongConversationError(RuntimeError):
    """A conversation to score exceeds ``--rm_max_seq_len``.

    A per-row drop like :class:`TruncatedGenerationError` rather than a truncation: right-truncation
    cuts the assistant turn, and the reward model would score the prompt fragment as the answer.
    """


def build_output_path(output_folder: str, prompts_source: str, model_name: str, suffix: str) -> Path:
    """``<prompts stem>_<model>_<suffix>.jsonl`` under ``output_folder``.

    The generating model is part of the name because both scripts resume from the ids already present
    in this file: without it a second model's run would read the first model's rows as its own.

    A ``/`` in a hub id would otherwise open a directory that does not exist. An over-long name is cut
    down to fit, keeping the suffix and a digest of the identifying pair; the cap is measured in bytes
    and the digest keeps two cut names from colliding.
    """
    stem = Path(prompts_source).stem
    basename = f"{stem}_{model_name.replace('/', '__')}_{suffix}.jsonl"
    if len(basename.encode()) <= _MAX_OUTPUT_BASENAME:
        return Path(output_folder) / basename

    digest = hashlib.sha256(f"{stem}\0{model_name}".encode()).hexdigest()[:16]
    tail = f"_{digest}_{suffix}.jsonl"
    keep = max(_MAX_OUTPUT_BASENAME - len(tail.encode()), 0)
    head = basename.encode()[:keep].decode(errors="ignore")
    return Path(output_folder) / f"{head}{tail}"


def load_reward_model(
    model_path: str, attn_impl: str, max_seq_len: int, device_str: str, dtype: str, *, trust_remote_code: bool
):
    """Load reward model and tokenizer in ``dtype`` (a :data:`DTYPE_BY_NAME` name).

    ``max_seq_len`` becomes the tokenizer's ``model_max_length``, the cap :func:`encode_for_scoring`
    refuses conversations past.
    """
    # A per-rank EP-sharded dir loads its non-expert weights under their true names while every expert
    # is randomly initialized, so the scores would be plausible-looking noise.
    reject_sharded_checkpoint(model_path)
    print(f"Loading Reward Model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    tokenizer.model_max_length = max_seq_len

    device = torch.device(device_str)
    # excuse_task_head=False: scoring consumes the reward head, it never trains one. Pointed at a
    # plain causal-LM directory the head would initialize randomly and every score would be noise.
    model = from_pretrained_verified(
        AutoModelForSequenceClassification,
        model_path,
        excuse_task_head=False,
        attn_implementation=attn_impl,
        trust_remote_code=trust_remote_code,
    )
    model.to(dtype=DTYPE_BY_NAME[dtype], device=device).eval()
    # After the cast, not before: transformers 5 hands every non-persistent buffer back uninitialized
    # (a remote-code family overriding ``_init_weights`` scores off a dead RoPE), and
    # ``Module.to(dtype=...)`` downcasts float buffers too, so repairing first would write the fp32
    # rotary table and then round it to bf16.
    finalize_loaded_model(model)
    return tokenizer, model, device


def boot_scoring_run(args) -> tuple[Any, Any, Any, torch.device]:
    """Open the two endpoints a scoring run needs: ``(api client, rm tokenizer, rm model, rm device)``.

    One call for both scripts, so neither drifts onto a different set of ``--rm_*`` knobs; every flag
    the reward model is loaded under is threaded here.
    """
    client = create_openai_client(base_url=args.openai_base_url, api_key_override=args.openai_api_key)
    rm_tokenizer, rm_model, rm_device = load_reward_model(
        args.rm_model_path,
        args.rm_model_atten_impl,
        args.rm_max_seq_len,
        args.rm_device,
        args.rm_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    return client, rm_tokenizer, rm_model, rm_device


def build_generation_parser(description: str, *, temperature_default: float) -> argparse.ArgumentParser:
    """The CLI parser shared by the RM inference scripts: API, data, generation, and RM arguments.

    Scripts add their own extras (``--n_hypos``, ``--output_format``, ``--rm_max_batch_size``, …)
    on the returned parser.
    """
    parser = argparse.ArgumentParser(description=description)
    add_openai_endpoint_args(parser, model_help="Model name for API requests")

    # Data configuration
    parser.add_argument("--prompts_source", type=str, required=True, help="Path to JSONL file with prompts")
    parser.add_argument("--output_folder", type=str, default="data")
    parser.add_argument("--id_field", type=str, default="id")
    parser.add_argument(
        "--prompt_field", type=str, default="prompt", help="Field containing the prompt as list of message dicts"
    )
    parser.add_argument("--follow_up_prompt_field", type=str, default="follow_up_prompt")
    parser.add_argument("--correct_answer_field", type=str, default="correct_answer")
    parser.add_argument("--local_system_prompt_field", type=str, default="system_prompt")
    parser.add_argument(
        "--global_system_prompt",
        type=str,
        default=None,
        help="System prompt applied to all rows (overridden by local)",
    )

    # Generation configuration
    add_generation_args(parser, temperature_default=temperature_default)

    # Reward model configuration
    parser.add_argument("--rm_model_path", type=str, required=True, help="Path to reward model")
    add_trust_remote_code_arg(parser, default=True)
    parser.add_argument("--rm_model_atten_impl", type=str, default="sdpa")
    parser.add_argument(
        "--rm_max_seq_len",
        type=int,
        default=16000,
        help="Longest conversation (tokens) the reward model scores; a longer one is dropped and counted, "
        "never truncated (default: %(default)s)",
    )
    parser.add_argument("--rm_device", type=str, default="cuda:0")
    parser.add_argument(
        "--rm_dtype",
        type=str,
        default="bfloat16",
        choices=list(DTYPE_BY_NAME),
        help="Reward-model compute dtype (default: bfloat16, the toolkit-wide default — float16's "
        "narrow range overflows on out-of-distribution reward logits)",
    )

    return parser


def _resolve_response_format(row) -> dict:
    """Per-row ``response_format`` override, defaulting to plain text."""
    if "response_format" in row and pd.notna(row["response_format"]):
        return row["response_format"]
    return {"type": "text"}


def resolve_correct_answer(row, args):
    """The row's ground-truth answer (``args.correct_answer_field``), or None."""
    if args.correct_answer_field in row and pd.notna(row[args.correct_answer_field]):
        return row[args.correct_answer_field]
    return None


async def generate_chat_message(client, messages: list[dict], args, response_format: dict) -> tuple[dict, str]:
    """One chat completion under the script's generation args.

    Returns ``(assistant message dict, finish_reason)``. The finish reason is part of the contract
    because a hypothesis cut at the token cap is a fragment: the reward model would score it as a
    complete answer and the number would land in a preference / offline-GRPO file.
    """
    completion = await client.chat.completions.create(
        messages=messages,
        model=args.model_name,
        temperature=args.temperature,
        response_format=response_format,
        max_tokens=args.max_gen_tokens,
    )
    choice = completion.choices[0]
    return choice.message.model_dump(exclude=_MESSAGE_DUMP_EXCLUDE), get_finish_reason(choice) or ""


async def prepare_generation_prompt(client, row: pd.Series, args) -> tuple[list[dict], dict]:
    """Assemble the row's generation prompt: system prompt + base messages, applying the optional
    follow-up turn (generate an initial response, then append it plus the follow-up messages).

    Returns ``(base_prompt, response_format)`` ready for the script's own generation calls.
    """
    system_prompt = resolve_system_prompt(row, args.local_system_prompt_field, args.global_system_prompt)
    base_prompt = build_base_prompt(row, args.prompt_field, system_prompt)
    response_format = _resolve_response_format(row)

    follow_up = row.get(args.follow_up_prompt_field)
    if isinstance(follow_up, list) and len(follow_up) > 0:
        answer, finish_reason = await generate_chat_message(client, base_prompt, args, response_format)
        if finish_reason == FINISH_REASON_LENGTH:
            # The follow-up turn conditions on this answer, so a fragment here corrupts every
            # hypothesis the row goes on to produce.
            raise TruncatedGenerationError(
                f"the first turn hit --max_gen_tokens ({args.max_gen_tokens}) and the follow-up turn "
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


def require_single_output(logits) -> None:
    """Reject a head that is not a single-output Bradley-Terry reward model.

    Pointed at a multi-class classifier, the score readers (``[0]`` / ``view(-1)``) would return an
    arbitrary, possibly misaligned, class logit as the reward.
    """
    if logits.shape[-1] != 1:
        raise ValueError(
            f"Expected a single-output reward model, got logits shape {tuple(logits.shape)} "
            f"— this looks like a multi-class classifier, not an RM."
        )


def encode_for_scoring(tokenizer, conversations: list[list[dict]], correct_answer: str | None) -> dict:
    """Tokenize ``conversations`` for the reward model the way its training rows were tokenized.

    Render through the chat template, then :func:`tokenize_rendered` (the Bradley-Terry reward map's
    own path, ``build_reward_preprocess_fn``), so BOS lands once on a tokenizer whose post-processor
    prepends it: re-tokenizing the rendered text would double it and the scores would come from a
    token sequence the model never trained on. A conversation past ``tokenizer.model_max_length``
    (``--rm_max_seq_len``) raises :class:`OverlongConversationError`.
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
        require_single_output(logits)
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


def load_prompts_dataframe(args) -> pd.DataFrame:
    """Load the JSONL prompts source and validate the required columns."""
    print("Loading prompts...")
    df = pd.read_json(args.prompts_source, lines=True)
    if args.prompt_field not in df or args.id_field not in df:
        raise ValueError(f"Input must contain '{args.prompt_field}' and '{args.id_field}' columns")
    return df


def load_local_jsonl_resume(output_path: Path, id_field: str) -> tuple[set, pd.DataFrame]:
    """``(ids already written, the rows carrying them)`` for a run resuming its own JSONL output.

    The local-file twin of :func:`scripts.inference._common.load_prompts_with_resume` (S3 datasets):
    both scripts append one finished row per line, so the ids in the output file are the work a re-run
    skips. An output that does not exist yet resumes nothing. The rows themselves come back because a
    caller reporting summary statistics over the whole job needs them, not just their ids.
    """
    if not output_path.exists():
        return set(), pd.DataFrame()
    existing = pd.read_json(output_path, lines=True)
    return set(existing[id_field]), existing
