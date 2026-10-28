"""Argparse flags shared by the generation CLIs under ``scripts/inference``: the sampling and concurrency block,
the prompt-row fields, the S3 prompt-dataset block with its ``--subfolder`` sentinel, and the Gradio server block
the ``playground/`` apps bind and launch on. Flags and the two calls that read them back (the ``--subfolder``
funnel and the Gradio launch); the runs they feed live in ``halo/entrypoints/``."""

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Annotation only: the generation CLIs import this module too, and gradio takes seconds to
    # import.
    import gradio as gr

# The documented "write at the bucket root" spelling for --subfolder. argparse hands it over as the
# string "None", which would read and write s3://bucket/None/<key> and so resume against an output
# it never finds, hence the :func:`parse_dataset_args` funnel.
NO_SUBFOLDER_SENTINEL = "None"

# Concurrency and resume cadence shared by the resumable generation CLIs. The scripts drive the
# same local rollout server through the same async client, so the numbers are shared rather than set
# per script: a lower concurrency halves a sibling's throughput and a rarer flush widens what an
# interrupted run regenerates. Override per invocation with the flags.
DEFAULT_MAX_WORKERS = 32
DEFAULT_CHECKPOINT_INTERVAL = 100
DEFAULT_MAX_TOKENS = 3072

# Loopback, for every Gradio app here: the UI drives a server-side client holding a live API key with
# no auth in front, so binding every interface would expose that key's spend to anything that can
# route to the host. `--host 0.0.0.0` still publishes, explicitly.
DEFAULT_GRADIO_HOST = "127.0.0.1"


def add_generation_args(parser: argparse.ArgumentParser, *, temperature_default: float) -> argparse.ArgumentParser:
    """Add the sampling/concurrency block shared by the generation CLIs.

    Only the temperature default is per-script (greedy for batched generation and reward-model
    scoring, sampled for the rejection sampler, which needs distinct hypotheses); the concurrency bound
    and the token cap are shared, for the reason :data:`DEFAULT_MAX_WORKERS` gives.
    """
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
        help="Generation cap per request; a reply cut at it is dropped, never scored (default: %(default)s).",
    )
    return parser


def add_checkpoint_interval_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the resume-store flush cadence of the batched generation CLI."""
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help="Flush finished results to the resume store every N results (default: %(default)s)",
    )
    return parser


def add_s3_dataset_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the S3 prompt-dataset block every checkpointing generation CLI reads its rows through.

    A path is a full ``s3://bucket/key`` URI, used as given, or a bare key under ``--subfolder`` in
    ``HALO_S3_DEFAULT_BUCKET`` (``build_s3_uri``); the field names are the ones
    :func:`load_prompts_with_resume` and the record builders consume, so a script declaring its own
    would resume against a differently-keyed output.
    """
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
