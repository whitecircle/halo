"""Argparse flags shared across the ``scripts/`` subtrees, so tools chained over one artifact and CLIs driving one
served model accept one spelling and one set of defaults."""

import argparse
from collections.abc import Sequence

from halo.checkpoint.format import DEFAULT_MAX_SHARD_SIZE
from halo.checkpoint.tools.offline_export import TASK_MODEL_CLASSES
from halo.inference.endpoints import DEFAULT_LOCAL_BASE_URL, resolve_local_api_key
from halo.model_loading.dtype import DTYPE_BY_NAME

# What ``--model_id`` accepts; kept in step with ``resolve_checkpoint_source``.
HUB_SOURCE_HELP = "Hub repo id or a local checkpoint directory"
# Each ``--trust_remote_code`` default's rationale and way out, by the source that earns it.
_TRUST_REMOTE_CODE_DEFAULTS = {
    True: (
        "True — the remote-code families, e.g. Bailing/Ling, do not load without it",
        "Pass --no-trust_remote_code for a source you do not trust",
    ),
    False: (
        "False — this source may be a Hub repo",
        "Pass --trust_remote_code for a remote-code family (Bailing/Ling, Laguna) you trust",
    ),
    None: (
        "on for a local checkpoint or an adapter on a local base, off for an adapter whose base is a Hub repo",
        "Pass --trust_remote_code for a remote-code family (Bailing/Ling, Laguna) you trust, "
        "or --no-trust_remote_code for a source you do not",
    ),
}


def add_max_shard_size_arg(parser: argparse.ArgumentParser, *, note: str = "") -> argparse.ArgumentParser:
    """``note`` appends a caveat for outputs ignoring the cap (adapter files, checkpoints rewritten in place)."""
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default=DEFAULT_MAX_SHARD_SIZE,
        help=f"Maximum size of one output safetensors shard, e.g. '2GB' (default: {DEFAULT_MAX_SHARD_SIZE})."
        + (f" {note}" if note else ""),
    )
    return parser


def add_hub_source_args(
    parser: argparse.ArgumentParser, *, source: str, default: str | None = None, revision: bool = True
) -> argparse.ArgumentParser:
    """``default`` names the release a converter targets and makes the flag optional; ``revision=False`` suits a tool
    threading no revision, which would otherwise advertise a pin it ignores."""
    parser.add_argument(
        "--model_id",
        type=str,
        default=default,
        required=default is None,
        help=f"{source} — {HUB_SOURCE_HELP}." + (f" Default: {default}." if default else ""),
    )
    if revision:
        parser.add_argument(
            "--revision",
            type=str,
            default=None,
            help="Hub revision to pin (ignored for a local source).",
        )
    return parser


def add_trust_remote_code_arg(
    parser: argparse.ArgumentParser, *, default: bool | None = True
) -> argparse.ArgumentParser:
    """The default follows the source: a local checkpoint defaults on, since remote-code families (Bailing/Ling,
    Laguna) ship their modeling files inside it; a possible Hub repo (``--model_id``) defaults off, so a downloaded
    third-party repo does not run its code. ``None`` is a tool loading the base an adapter names, which resolves the
    unset flag off that base."""
    note, remedy = _TRUST_REMOTE_CODE_DEFAULTS[default]
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=default,
        help=f"Execute the checkpoint's own modeling/config code when loading it (default: {note}). {remedy}.",
    )
    return parser


def add_dtype_arg(
    parser: argparse.ArgumentParser,
    *,
    flag: str = "--dtype",
    help: str = "Dtype of the output checkpoint (default: %(default)s).",
) -> argparse.ArgumentParser:
    """``--dtype`` for a tool writing a checkpoint, or ``flag``/``help`` for another model's compute dtype."""
    parser.add_argument(flag, type=str, default="bfloat16", choices=list(DTYPE_BY_NAME), help=help)
    return parser


def add_task_arg(
    parser: argparse.ArgumentParser, *, tasks: Sequence[str] = tuple(TASK_MODEL_CLASSES)
) -> argparse.ArgumentParser:
    """The head a checkpoint tool loads a checkpoint with, over ``tasks``."""
    parser.add_argument(
        "--task",
        choices=list(tasks),
        default="causal_lm",
        help="The head the checkpoint loads with: causal_lm (the widest class its config names, a VLM's "
        "included), classification (sequence classification) or base (the bare backbone), as offered "
        "(default: %(default)s).",
    )
    return parser


def add_device_map_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Device map for the model load, e.g. 'auto' or 'cpu' (default: none, the model loads on the CPU).",
    )
    return parser


def add_openai_endpoint_args(
    parser: argparse.ArgumentParser, *, model_help: str | None = None
) -> argparse.ArgumentParser:
    """``--base_url``, ``--api_key`` and, given ``model_help``, a required ``--model``; one key-resolution policy,
    since a command line has to carry from one CLI to the next."""
    parser.add_argument(
        "--base_url",
        type=str,
        default=DEFAULT_LOCAL_BASE_URL,
        help="OpenAI-compatible base URL (default: %(default)s).",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=resolve_local_api_key(),
        help="API key for the endpoint (default: $VLLM_API_KEY, else $OPENAI_API_KEY, else the placeholder a "
        "keyless local server accepts; a hosted endpoint such as OpenRouter needs a real key).",
    )
    if model_help is not None:
        parser.add_argument("--model", type=str, required=True, help=model_help)
    return parser
