"""Shared plumbing for the generation CLIs under ``scripts/inference``.

``generation/openai_batched_generation.py`` reads a prompt dataset from S3, resumes against a
partially written output dataset, merges the two and pushes the result back;
``reward_model/rm_rejection_sampling.py`` runs the same hypothesis-and-score shape
against local JSONL. The parts they share live here so the contracts cannot drift between them — a
mismatched resume silently re-generates or silently drops rows, a mismatched merge writes a batch's
own columns away, and a mismatched abort reports a total failure as a completed run.

The Gradio apps under ``playground/`` share the address block they bind and launch on for the same
reason: each drives a server-side client holding a live key, so a per-app copy is how one of them
ends up published on every interface.
"""

import argparse
import asyncio
import signal
import sys
from collections.abc import Callable, Coroutine, Sequence
from typing import TYPE_CHECKING

from datasets import Dataset
from loguru import logger

from src.data.sources.s3_client import build_s3_uri, exists, load_dataset_from_s3_uri, push_dataset_to_s3_uri
from src.inference.openai_client import DEFAULT_LOCAL_BASE_URL, resolve_local_api_key
from src.inference.response import OpenAIResponse

if TYPE_CHECKING:
    # Annotation only: the generation CLIs import this module too, and gradio is a seconds-long import.
    import gradio as gr

# The documented "write at the bucket root" spelling for --subfolder. argparse hands it over as the
# STRING "None", which would read and write s3://bucket/None/<key> — a resume that never finds its
# own output — so every parser here funnels through :func:`parse_dataset_args`.
NO_SUBFOLDER_SENTINEL = "None"

# Concurrency and resume cadence every checkpointing generation CLI shares. One spelling each: the
# scripts drive the same local rollout server through the same async client, so a per-script number
# is drift, not tuning — a lower one silently halves a sibling's throughput, and a rarer checkpoint
# silently widens what an interrupted run has to regenerate. Override per invocation with the flags.
DEFAULT_N_PARALLEL = 32
DEFAULT_CHECKPOINT_INTERVAL = 100
DEFAULT_MAX_GEN_TOKENS = 3072

# Loopback, for every Gradio app here: the UI drives a server-side client holding a live API key, so
# binding every interface hands that key's spend to anything that can route to the box, with no auth
# in front. `--host 0.0.0.0` still publishes, explicitly.
DEFAULT_GRADIO_HOST = "127.0.0.1"


def add_openai_endpoint_args(parser: argparse.ArgumentParser, *, model_help: str) -> argparse.ArgumentParser:
    """Add the OpenAI-compatible endpoint block every generation CLI here drives its rollout through.

    One spelling and one key-resolution policy: these scripts all point at the same local server, so
    a per-script copy is how ``--openai_base_url`` on one becomes ``--api_url`` on the next and an
    operator's pinned command line silently generates against the wrong endpoint. ``model_help``
    is the only genuinely per-script part.
    """
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=resolve_local_api_key(),
        help="API key for the generation endpoint (default: $VLLM_API_KEY, else $OPENAI_API_KEY, else the "
        "placeholder a keyless local server accepts)",
    )
    parser.add_argument("--openai_base_url", type=str, default=DEFAULT_LOCAL_BASE_URL)
    parser.add_argument("--model_name", type=str, required=True, help=model_help)
    return parser


def add_generation_args(parser: argparse.ArgumentParser, *, temperature_default: float) -> argparse.ArgumentParser:
    """Add the sampling/concurrency block shared by the generation CLIs.

    Only the temperature default is genuinely per-script (greedy for batched generation, sampled for
    the rejection samplers, which need distinct hypotheses); the concurrency bound and the token cap
    are one number each, for the reason :data:`DEFAULT_N_PARALLEL` documents.
    """
    parser.add_argument(
        "--n_parallel", type=int, default=DEFAULT_N_PARALLEL, help="Max parallel API requests (default: %(default)s)"
    )
    parser.add_argument("--temperature", type=float, default=temperature_default)
    parser.add_argument("--max_gen_tokens", type=int, default=DEFAULT_MAX_GEN_TOKENS)
    return parser


def add_checkpoint_interval_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the resume cadence every checkpointing generation CLI writes its partial output on."""
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help="Save a generation checkpoint every N results (default: %(default)s)",
    )
    return parser


def add_s3_dataset_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the S3 prompt-dataset block every checkpointing generation CLI reads its rows through.

    Paths are S3 **keys** under ``HALO_S3_DEFAULT_BUCKET``, resolved by ``build_s3_uri``; the field
    names are the ones :func:`load_prompts_with_resume` and the record builders consume, so a script
    that declared its own would resume against a differently-keyed output.
    """
    parser.add_argument("--input_path", type=str, required=True, help="S3 path to input dataset")
    parser.add_argument("--output_path", type=str, required=True, help="S3 path for output dataset")
    parser.add_argument(
        "--subfolder",
        type=str,
        default="datasets",
        help=f"S3 subfolder (default: datasets, use '{NO_SUBFOLDER_SENTINEL}' to skip)",
    )
    parser.add_argument("--id_field", type=str, default="id", help="Field holding the row's unique id")
    parser.add_argument(
        "--prompt_field", type=str, default="prompt", help="Field with the prompt as a list of message dicts"
    )
    parser.add_argument(
        "--local_system_prompt_field", type=str, default="system_prompt", help="Per-row system prompt field"
    )
    parser.add_argument(
        "--global_system_prompt",
        type=str,
        default=None,
        help="System prompt applied to all rows (overridden by the per-row one)",
    )
    return parser


def add_output_format_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the ``--output_format`` selector both rejection samplers dispatch their record builder on.

    Declared once: the two formats are trainer contracts (``agent-docs/data/dataset-formats.md``), so a
    per-script copy is one edit from a CLI that accepts a value its builders cannot emit.
    """
    parser.add_argument(
        "--output_format",
        type=str,
        default="preference",
        choices=["preference", "offline_grpo"],
        help="Output format: preference (DPO/SMPO) or offline_grpo (default: preference)",
    )
    return parser


def add_gradio_server_args(parser: argparse.ArgumentParser, *, port_default: int | None) -> argparse.ArgumentParser:
    """Add the ``--host``/``--port``/``--share`` block every Gradio app here serves itself on.

    One spelling and one bind policy, for the reason :data:`DEFAULT_GRADIO_HOST` documents. The port
    is the only per-app part: ``None`` leaves the choice to Gradio's own free-port search.
    """
    group = parser.add_argument_group("Server Configuration")
    group.add_argument(
        "--host",
        type=str,
        default=DEFAULT_GRADIO_HOST,
        help=(
            "Host the Gradio app binds to (default: %(default)s — loopback only). This app holds a "
            "live API key; pass --host 0.0.0.0 only on a network you trust, since that exposes the "
            "UI, and with it the key's spend, to everything that can reach this box."
        ),
    )
    port_help = port_default if port_default is not None else "the first free port from 7860, or $GRADIO_SERVER_PORT"
    group.add_argument("--port", type=int, default=port_default, help=f"Port to serve on (default: {port_help})")
    group.add_argument("--share", action="store_true", help="Create public Gradio link")
    return parser


def launch_gradio(demo: "gr.Blocks", args: argparse.Namespace) -> None:
    """Serve ``demo`` on the address :func:`add_gradio_server_args` parsed, requests queued."""
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


def parse_dataset_args(parser: argparse.ArgumentParser):
    """``parser.parse_args()`` with the ``--subfolder`` sentinel resolved to a real ``None``."""
    args = parser.parse_args()
    if args.subfolder == NO_SUBFOLDER_SENTINEL:
        args.subfolder = None
    return args


def load_prompts_with_resume(args) -> tuple[list[dict], list[dict]]:
    """``(pending rows, rows already written to --output_path)`` for the S3 prompt dataset.

    Ids present in the output dataset are dropped from the input, and the existing rows come back so
    the caller can write them out again alongside the new ones. An empty first element means every
    prompt is already done.
    """
    logger.info(f"Loading dataset from S3: {args.input_path} (subfolder: {args.subfolder})")
    dataset = load_dataset_from_s3_uri(build_s3_uri(args.input_path, args.subfolder))

    if args.prompt_field not in dataset.column_names or args.id_field not in dataset.column_names:
        raise ValueError(f"Dataset must contain '{args.prompt_field}' and '{args.id_field}' columns")

    processed_ids: set = set()
    existing_results: list[dict] = []
    if exists(args.output_path, subfolder=args.subfolder):
        logger.info(f"Loading existing results from S3: {args.output_path}")
        existing_dataset = load_dataset_from_s3_uri(build_s3_uri(args.output_path, args.subfolder))
        processed_ids = set(existing_dataset[args.id_field])
        existing_results = list(existing_dataset)
        logger.info(f"Skipping {len(processed_ids)} already completed prompts")

    return list(dataset.filter(lambda x: x[args.id_field] not in processed_ids)), existing_results


def reject_empty_results(produced: int, pending: int, destination, *, drops: str = "", check: str) -> None:
    """Abort a run that produced no usable row at all, instead of writing an empty result.

    Every pending row failed, so this is a dead endpoint or a misconfigured model — exiting 0 there
    reports a total failure as a completed run, and (on the S3 paths) republishes the resumed rows
    as if they were the whole job. Keyed on "nothing produced" rather than on any single failure
    counter, because rows also drop for reasons no counter owns.

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

    A resume mixes records reloaded from a previously-saved dataset (which carry every column that
    run wrote) with freshly-built ones, whose optional source columns — or whole output format —
    may differ. ``Dataset.from_list`` infers its schema from the leading records and silently DROPS
    keys absent there, so the union (missing filled with ``None``) is what keeps the just-generated
    batch's own columns from being thrown away.
    """
    all_results = existing_results + results
    all_keys = {key for record in all_results for key in record}
    all_results = [{key: record.get(key) for key in all_keys} for record in all_results]
    logger.info(f"Saving {len(all_results)} results to S3: {output_path}")
    push_dataset_to_s3_uri(Dataset.from_list(all_results), build_s3_uri(output_path, subfolder))
    logger.info("Done!")


def degenerate_hypotheses_reason(scores: Sequence[float], output_format: str) -> str | None:
    """Why a scored row cannot produce a useful record, or ``None`` if it can.

    One rule for every rejection sampler feeding the same trainers: fewer than 2 hypotheses can
    never separate a best from a worst, and all-equal scores
    make argmax == argmin — a chosen==rejected pair that teaches a preference trainer nothing.
    ``offline_grpo`` keeps an all-equal reward vector: the trainer's degenerate-group handling owns
    that case.
    """
    if len(scores) < 2:
        return f"insufficient hypotheses ({len(scores)} < 2)"
    if output_format != "offline_grpo" and len({float(score) for score in scores}) < 2:
        return f"all {len(scores)} hypotheses scored equally ({float(scores[0])})"
    return None


def best_worst_indices(scores: Sequence[float]) -> tuple[int, int]:
    """``(argmax, argmin)`` over ``scores``, first index winning a tie.

    One selection rule for both rejection samplers, reading a Python list and a numpy score vector
    the same way. :func:`degenerate_hypotheses_reason` has already refused the all-equal row where
    the two would name the same completion.
    """
    ranked = range(len(scores))
    return max(ranked, key=scores.__getitem__), min(ranked, key=scores.__getitem__)


def preference_record(
    prompt: list[dict],
    completions: Sequence[dict],
    scores: Sequence[float],
    *,
    row: dict,
    id_field: str,
    gen_model: str,
    correct_answer: str | None = None,
) -> dict:
    """The DPO/SMPO preference record: the best- and worst-scored completion of one prompt.

    One writer for both rejection samplers, because the preference trainers read one contract
    (``agent-docs/data/dataset-formats.md``): ``chosen``/``rejected`` are message LISTS and every score a
    plain float (a ``numpy.float64`` does not survive ``json.dumps``). The whole scored set rides
    along so a re-pairing needs no re-generation; the row id and ``correct_answer`` on the same terms
    as :func:`offline_grpo_record`. A sampler's own provenance columns are added by its caller.
    """
    best, worst = best_worst_indices(scores)
    record = {
        "prompt": prompt,
        "chosen": [completions[best]],
        "chosen_score": float(scores[best]),
        "rejected": [completions[worst]],
        "rejected_score": float(scores[worst]),
        "all_generations": list(completions),
        "all_scores": [float(score) for score in scores],
        "gen_model": gen_model,
    }
    if id_field in row:
        record[id_field] = row[id_field]
    if correct_answer is not None:
        record["target_answer"] = correct_answer
    return record


def offline_grpo_record(
    prompt: list[dict],
    completions: Sequence[Sequence[dict]],
    rewards: Sequence[float],
    *,
    row: dict,
    id_field: str,
    correct_answer: str | None = None,
) -> dict:
    """The offline-GRPO training record: ``{"prompt", "completions", "rewards"}`` (+ optional keys).

    One writer for both rejection samplers, because the trainer reads one contract
    (``agent-docs/data/dataset-formats.md``): ``completions`` is a list of message LISTS, ``rewards`` a
    parallel list of plain floats. The row id rides along only when the source row carries it — a
    dataset without the column must not gain a null one — and ``correct_answer`` only where the CLI
    has a ground-truth concept at all.
    """
    record = {
        "prompt": prompt,
        "completions": [list(completion) for completion in completions],
        "rewards": [float(reward) for reward in rewards],
    }
    if id_field in row:
        record[id_field] = row[id_field]
    if correct_answer is not None:
        record["target_answer"] = correct_answer
    return record


def assistant_message_from_response(response: OpenAIResponse) -> dict:
    """The assistant turn for a generated response, as the wire and the saved record both spell it.

    ``content`` stays ``None`` on a tool-call-only or empty reply: ``str(None)`` sends the literal
    text "None" to the next turn, which the model reads as the assistant's answer.
    """
    message: dict = {"role": "assistant", "content": response.answer if isinstance(response.answer, str) else None}
    if response.tool_calls:
        message["tool_calls"] = [tool_call.model_dump() for tool_call in response.tool_calls]
    return message


def _signal_handler(signum, frame):
    """Exit on an interrupt with the shell's own convention for a signalled process: 128 + signo.

    Exiting 0 here reports an interrupted, incomplete job as a successful one — a wrapper script,
    a CI step or a `&&` chain then proceeds to consume a partial output dataset as if the run had
    finished. The resume hint is what makes the non-zero exit actionable.
    """
    logger.info(f"\nReceived signal {signum}. Progress has been saved — resume by re-running.")
    sys.exit(128 + signum)


def run_async_cli(main: Callable[[], Coroutine]) -> None:
    """Run an async CLI ``main`` under SIGINT/SIGTERM handling, exiting non-zero on a fatal error."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Same event as the handler above (which a library that restores the default handler can
        # bypass), so it must report the same way.
        logger.info("\nInterrupted. Progress has been saved — resume by re-running.")
        sys.exit(128 + signal.SIGINT)
    except Exception:
        # With the traceback: the failures that land here (a raised result guard, an S3 or schema
        # error deep in the merge) are diagnosed from where they were raised, not from their message.
        logger.exception("Fatal error")
        sys.exit(1)
