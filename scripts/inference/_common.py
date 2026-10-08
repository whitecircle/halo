"""Shared plumbing for the generation CLIs under ``scripts/inference``.

``generation/openai_batched_generation.py`` reads a prompt dataset from S3, resumes against a
partially written output dataset, merges the two and pushes the result back;
``reward_model/rm_scoring.py`` and ``reward_model/rm_rejection_sampling.py`` generate against the
same endpoint from local JSONL and score with a reward model. The parts they share live here so the
sampling, resume, merge and abort contracts stay identical: a mismatched resume re-generates or drops
rows, a mismatched merge discards a batch's own columns, and a mismatched abort reports a total
failure as a completed run.

The Gradio apps under ``playground/`` share the address block they bind and launch on, since each
drives a server-side client holding a live key.
"""

import argparse
import asyncio
import logging
import signal
import sys
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING

from datasets import Dataset

from src.data.sources.s3_client import S3Client, build_s3_uri, load_dataset_from_s3_uri, push_dataset_to_s3_uri
from src.log import configure_cli_logging

if TYPE_CHECKING:
    # Annotation only: the generation CLIs import this module too, and gradio takes seconds to
    # import.
    import gradio as gr

logger = logging.getLogger(__name__)

# The documented "write at the bucket root" spelling for --subfolder. argparse hands it over as the
# string "None", which would read and write s3://bucket/None/<key> and so resume against an output
# it never finds, hence the :func:`parse_dataset_args` funnel.
NO_SUBFOLDER_SENTINEL = "None"

# Concurrency and resume cadence shared by the checkpointing generation CLIs. The scripts drive the
# same local rollout server through the same async client, so the numbers are shared rather than set
# per script: a lower concurrency halves a sibling's throughput and a rarer checkpoint widens what an
# interrupted run regenerates. Override per invocation with the flags.
DEFAULT_N_PARALLEL = 32
DEFAULT_CHECKPOINT_INTERVAL = 100
DEFAULT_MAX_GEN_TOKENS = 3072

# Loopback, for every Gradio app here: the UI drives a server-side client holding a live API key with
# no auth in front, so binding every interface would expose that key's spend to anything that can
# route to the host. `--host 0.0.0.0` still publishes, explicitly.
DEFAULT_GRADIO_HOST = "127.0.0.1"


def add_generation_args(parser: argparse.ArgumentParser, *, temperature_default: float) -> argparse.ArgumentParser:
    """Add the sampling/concurrency block shared by the generation CLIs.

    Only the temperature default is per-script (greedy for batched generation and reward-model
    scoring, sampled for the rejection sampler, which needs distinct hypotheses); the concurrency bound
    and the token cap are shared, for the reason :data:`DEFAULT_N_PARALLEL` gives.
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

    Paths are S3 keys under ``HALO_S3_DEFAULT_BUCKET``, resolved by ``build_s3_uri``; the field names
    are the ones :func:`load_prompts_with_resume` and the record builders consume, so a script
    declaring its own would resume against a differently-keyed output.
    """
    parser.add_argument("--input_path", type=str, required=True, help="S3 path to input dataset")
    parser.add_argument("--output_path", type=str, required=True, help="S3 path for output dataset")
    parser.add_argument(
        "--subfolder",
        type=str,
        default="datasets",
        help=f"S3 subfolder (default: datasets, use '{NO_SUBFOLDER_SENTINEL}' to skip)",
    )
    return add_prompt_field_args(parser)


def add_prompt_field_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the prompt-row fields every generation CLI reads: the row id, the prompt, and the per-row
    and global system prompts that ``resolve_system_prompt`` / ``build_base_prompt`` combine."""
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


def add_gradio_server_args(parser: argparse.ArgumentParser, *, port_default: int | None) -> argparse.ArgumentParser:
    """Add the ``--host``/``--port``/``--share`` block every Gradio app here serves itself on.

    One spelling and one bind policy, for the reason :data:`DEFAULT_GRADIO_HOST` gives. The port is
    the only per-app part: ``None`` leaves the choice to Gradio's own free-port search.
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


def launch_gradio(demo: "gr.Blocks", args: argparse.Namespace, *, theme: "gr.Theme | None" = None) -> None:
    """Serve ``demo`` on the address :func:`add_gradio_server_args` parsed, requests queued.

    The theme is a launch argument: Gradio 6 takes it here, not on the ``Blocks``.
    """
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share, theme=theme)


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
