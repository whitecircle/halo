"""Argparse flags shared by the generation CLIs under ``scripts/inference``, and the two calls that read them back (the
``--subfolder`` funnel and the Gradio launch)."""

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Annotation only: the generation CLIs import this module too, and gradio takes seconds to import.
    import gradio as gr

# The documented spelling for the bucket root. argparse hands it over as the string "None", which would read and
# write s3://bucket/None/<key>, hence the :func:`parse_dataset_args` funnel.
NO_SUBFOLDER_SENTINEL = "None"

# Shared because these CLIs drive one rollout server through one async client: a lower concurrency halves a sibling's
# throughput, a rarer flush widens what an interrupted run regenerates.
DEFAULT_MAX_WORKERS = 32
DEFAULT_CHECKPOINT_INTERVAL = 100
DEFAULT_MAX_TOKENS = 3072

# Loopback: the UI drives a server-side client holding a live API key with no auth in front. `--host 0.0.0.0` still
# publishes, explicitly.
DEFAULT_GRADIO_HOST = "127.0.0.1"


def add_generation_args(parser: argparse.ArgumentParser, *, temperature_default: float) -> argparse.ArgumentParser:
    """Only the temperature default is per-script (greedy for batched generation and scoring, sampled for the rejection
    sampler, which needs distinct hypotheses)."""
    parser.add_argument(
        "--max_workers", type=int, default=DEFAULT_MAX_WORKERS, help="Max parallel API requests (default: %(default)s)"
    )
    parser.add_argument(
        "--temperature", type=float, default=temperature_default, help="Sampling temperature (default: %(default)s)."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Generation cap per request. The reward-model scorers drop and count a reply cut at it; batched "
        "generation keeps it and records its finish_reason (default: %(default)s).",
    )
    return parser


def add_checkpoint_interval_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help="Flush finished results to the resume store every N results (default: %(default)s)",
    )
    return parser


def add_s3_dataset_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """A path is a full URI or a bare key under ``--subfolder`` in ``HALO_S3_DEFAULT_BUCKET``; the field names are the
    ones resume and the record builders consume, so a script's own would resume against another output."""
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Input prompt dataset: a full s3://<bucket>/<key> URI, or a bare key under --subfolder in "
        "$HALO_S3_DEFAULT_BUCKET.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output dataset, appended to and resumed from: a full s3:// URI or a bare key, read as --dataset is.",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="datasets",
        help=f"Prefix of a bare-key --dataset / --output_path, unused by a full s3:// URI (default: datasets, "
        f"use '{NO_SUBFOLDER_SENTINEL}' to skip)",
    )
    return add_prompt_field_args(parser)


def add_prompt_field_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The row id, the prompt, its follow-up turns, and the per-row and global system prompts."""
    parser.add_argument("--id_field", type=str, default="id", help="Field holding the row's unique id")
    parser.add_argument(
        "--prompt_field", type=str, default="prompt", help="Field with the prompt as a list of message dicts"
    )
    parser.add_argument(
        "--follow_up_prompt_field",
        type=str,
        default="follow_up_prompt",
        help="Row field with follow-up turns sent after the first answer (default: %(default)s).",
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
    """One bind policy for every Gradio app; a ``None`` port leaves the choice to Gradio's free-port search."""
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
    """The theme is a launch argument: Gradio 6 takes it here, not on the ``Blocks``."""
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share, theme=theme)


def parse_dataset_args(parser: argparse.ArgumentParser):
    """``parser.parse_args()`` with the ``--subfolder`` sentinel resolved to a real ``None``."""
    args = parser.parse_args()
    if args.subfolder == NO_SUBFOLDER_SENTINEL:
        args.subfolder = None
    return args
